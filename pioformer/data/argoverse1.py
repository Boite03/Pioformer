from itertools import permutations, product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from pioformer.data.cache import load_tensor_cache, save_tensor_cache


class Argoverse1Dataset(Dataset):
    """Argoverse 1 train/val dataset backed by tensor caches."""

    def __init__(self, root: str, split: str, local_radius: float = 50.0) -> None:
        if split not in ("train", "val"):
            raise ValueError("Only Argoverse 1 train and val splits are supported")
        self.root = Path(root)
        self.split = split
        self.local_radius = local_radius
        self.raw_dir = self.root / split / "data"
        self.processed_dir = self.root / split / "processed"
        if not self.raw_dir.is_dir():
            raise FileNotFoundError(
                "Expected Argoverse 1 layout <root>/{}/data/*.csv; missing {}".format(split, self.raw_dir)
            )
        self.raw_files = sorted(self.raw_dir.glob("*.csv"))
        if not self.raw_files:
            raise FileNotFoundError("No Argoverse 1 CSV files found in {}".format(self.raw_dir))
        self.cache_files = [self.processed_dir / (path.stem + ".pt") for path in self.raw_files]
        self.process_missing()

    def process_missing(self) -> None:
        missing = [(raw, cache) for raw, cache in zip(self.raw_files, self.cache_files) if not cache.exists()]
        if not missing:
            return
        try:
            from argoverse.map_representation.map_api import ArgoverseMap
        except ImportError as error:
            raise RuntimeError(
                "Argoverse API is required for preprocessing; install argoverse-api with --no-deps"
            ) from error
        argoverse_map = ArgoverseMap()
        for raw_path, cache_path in tqdm(missing, desc="Processing {} split".format(self.split)):
            save_tensor_cache(
                process_argoverse1(raw_path, argoverse_map, self.local_radius),
                str(cache_path),
            )

    def __len__(self) -> int:
        return len(self.cache_files)

    def __getitem__(self, index: int):
        path = self.cache_files[index]
        if not path.exists():
            raise FileNotFoundError("Missing processed cache: {}".format(path))
        return load_tensor_cache(str(path))


def process_argoverse1(raw_path: Path, argoverse_map, radius: float) -> Dict[str, torch.Tensor]:
    frame = pd.read_csv(raw_path)
    timestamps = list(np.sort(frame["TIMESTAMP"].unique()))
    if len(timestamps) < 20:
        raise ValueError("{} has fewer than 20 historical frames".format(raw_path))
    historical = frame[frame["TIMESTAMP"].isin(timestamps[:20])]
    actor_ids = list(historical["TRACK_ID"].unique())
    actor_index = {actor_id: index for index, actor_id in enumerate(actor_ids)}
    frame = frame[frame["TRACK_ID"].isin(actor_ids)]
    num_nodes = len(actor_ids)
    agent_frame = frame[frame["OBJECT_TYPE"] == "AGENT"].sort_values("TIMESTAMP")
    agent_history = agent_frame[agent_frame["TIMESTAMP"].isin(timestamps[:20])]
    if len(agent_history) < 20:
        raise ValueError("{} has an incomplete focal AGENT history".format(raw_path))
    focal_index = actor_index[agent_history.iloc[0]["TRACK_ID"]]
    city = frame["CITY_NAME"].values[0]
    origin = torch.tensor([agent_history.iloc[-1]["X"], agent_history.iloc[-1]["Y"]], dtype=torch.float32)
    heading = origin - torch.tensor(
        [agent_history.iloc[-2]["X"], agent_history.iloc[-2]["Y"]], dtype=torch.float32,
    )
    theta = torch.atan2(heading[1], heading[0])
    rotate = torch.stack((
        torch.stack((torch.cos(theta), -torch.sin(theta))),
        torch.stack((torch.sin(theta), torch.cos(theta))),
    ))

    total_steps = 50
    coordinates = torch.zeros(num_nodes, total_steps, 2)
    padding_mask = torch.ones(num_nodes, total_steps, dtype=torch.bool)
    bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes)
    edges = list(permutations(range(num_nodes), 2))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    timestamp_index = {timestamp: index for index, timestamp in enumerate(timestamps[:total_steps])}
    for actor_id, actor_frame in frame.groupby("TRACK_ID"):
        node = actor_index[actor_id]
        valid_rows = actor_frame[actor_frame["TIMESTAMP"].isin(timestamp_index)].sort_values("TIMESTAMP")
        steps = [timestamp_index[value] for value in valid_rows["TIMESTAMP"]]
        padding_mask[node, steps] = False
        if padding_mask[node, 19]:
            padding_mask[node, 20:] = True
        xy = torch.from_numpy(np.stack((valid_rows["X"].values, valid_rows["Y"].values), axis=-1)).float()
        coordinates[node, steps] = torch.matmul(xy - origin, rotate)
        history_steps = [step for step in steps if step < 20]
        if len(history_steps) > 1:
            vector = coordinates[node, history_steps[-1]] - coordinates[node, history_steps[-2]]
            rotate_angles[node] = torch.atan2(vector[1], vector[0])
        else:
            padding_mask[node, 20:] = True

    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1:] = padding_mask[:, :19] & ~padding_mask[:, 1:20]
    positions = coordinates.clone()
    displacements = coordinates.clone()
    displacements[:, 20:] = torch.where(
        (padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
        torch.zeros_like(displacements[:, 20:]),
        coordinates[:, 20:] - coordinates[:, 19].unsqueeze(1),
    )
    displacements[:, 1:20] = torch.where(
        (padding_mask[:, :19] | padding_mask[:, 1:20]).unsqueeze(-1),
        torch.zeros_like(displacements[:, 1:20]),
        coordinates[:, 1:20] - coordinates[:, :19],
    )
    displacements[:, 0] = 0

    present = frame[frame["TIMESTAMP"] == timestamps[19]]
    node_indices = [actor_index[actor_id] for actor_id in present["TRACK_ID"]]
    node_positions = torch.from_numpy(np.stack((present["X"].values, present["Y"].values), axis=-1)).float()
    lane = lane_features(argoverse_map, node_indices, node_positions, origin, rotate, city, radius)
    return {
        "x": displacements[:, :20],
        "positions": positions,
        "edge_index": edge_index,
        "y": displacements[:, 20:],
        "padding_mask": padding_mask,
        "bos_mask": bos_mask,
        "rotate_angles": rotate_angles,
        "lane_vectors": lane[0],
        "is_intersections": lane[1],
        "turn_directions": lane[2],
        "traffic_controls": lane[3],
        "lane_actor_index": lane[4],
        "lane_actor_vectors": lane[5],
        "agent_index": torch.tensor(focal_index, dtype=torch.long),
    }


def lane_features(argoverse_map, node_indices: List[int], node_positions: torch.Tensor,
                  origin: torch.Tensor, rotate: torch.Tensor, city: str, radius: float
                  ) -> Tuple[torch.Tensor, ...]:
    lane_ids = set()
    for position in node_positions:
        lane_ids.update(argoverse_map.get_lane_ids_in_xy_bbox(position[0], position[1], city, radius))
    transformed_nodes = torch.matmul(node_positions - origin, rotate).float()
    positions, vectors, intersections, turns, controls = [], [], [], [], []
    for lane_id in sorted(lane_ids):
        centerline = torch.from_numpy(
            argoverse_map.get_lane_segment_centerline(lane_id, city)[:, :2]
        ).float()
        centerline = torch.matmul(centerline - origin, rotate)
        count = centerline.size(0) - 1
        if count <= 0:
            continue
        positions.append(centerline[:-1])
        vectors.append(centerline[1:] - centerline[:-1])
        intersections.append(torch.full((count,), int(argoverse_map.lane_is_in_intersection(lane_id, city)),
                                        dtype=torch.uint8))
        direction = {"NONE": 0, "LEFT": 1, "RIGHT": 2}[argoverse_map.get_lane_turn_direction(lane_id, city)]
        turns.append(torch.full((count,), direction, dtype=torch.uint8))
        controls.append(torch.full(
            (count,), int(argoverse_map.lane_has_traffic_control_measure(lane_id, city)), dtype=torch.uint8
        ))
    if vectors:
        lane_positions = torch.cat(positions)
        lane_vectors = torch.cat(vectors)
        is_intersections = torch.cat(intersections)
        turn_directions = torch.cat(turns)
        traffic_controls = torch.cat(controls)
    else:
        lane_positions = torch.empty((0, 2))
        lane_vectors = torch.empty((0, 2))
        is_intersections = torch.empty((0,), dtype=torch.uint8)
        turn_directions = torch.empty((0,), dtype=torch.uint8)
        traffic_controls = torch.empty((0,), dtype=torch.uint8)
    pairs = list(product(range(lane_vectors.size(0)), node_indices))
    lane_actor_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    if not pairs:
        lane_actor_index = torch.empty((2, 0), dtype=torch.long)
        lane_actor_vectors = torch.empty((0, 2))
    else:
        lane_actor_vectors = (
            lane_positions.repeat_interleave(len(node_indices), dim=0)
            - transformed_nodes.repeat(lane_vectors.size(0), 1)
        )
        mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
        lane_actor_index = lane_actor_index[:, mask]
        lane_actor_vectors = lane_actor_vectors[mask]
    return (lane_vectors, is_intersections, turn_directions, traffic_controls,
            lane_actor_index, lane_actor_vectors)
