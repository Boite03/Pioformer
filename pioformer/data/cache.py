from pathlib import Path
from typing import Mapping

import torch

from pioformer.data.types import TemporalData


SCHEMA_VERSION = 2
TENSOR_FIELDS = (
    "x", "positions", "edge_index", "y", "padding_mask", "bos_mask", "rotate_angles",
    "lane_vectors", "is_intersections", "turn_directions", "traffic_controls",
    "lane_actor_index", "lane_actor_vectors", "agent_index",
)


def save_tensor_cache(values: Mapping[str, torch.Tensor], path: str) -> None:
    invalid = sorted(key for key, value in values.items() if not isinstance(value, torch.Tensor))
    if invalid:
        raise TypeError("Processed cache only accepts tensors; invalid fields: {}".format(invalid))
    payload = {"schema_version": torch.tensor(SCHEMA_VERSION), "data": dict(values)}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_tensor_cache(path: str) -> TemporalData:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("Processed cache requires PyTorch weights_only loading support") from error
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "data"}:
        raise ValueError("Invalid processed cache: {}".format(path))
    version = int(payload["schema_version"].item())
    if version != SCHEMA_VERSION:
        raise ValueError("Unsupported cache schema {} (expected {})".format(version, SCHEMA_VERSION))
    values = payload["data"]
    if not isinstance(values, dict) or any(not isinstance(value, torch.Tensor) for value in values.values()):
        raise ValueError("Processed cache data must be a pure tensor dictionary")
    missing = sorted(set(TENSOR_FIELDS) - set(values))
    if missing:
        raise ValueError("Processed cache is missing fields: {}".format(missing))
    kwargs = dict(values)
    kwargs["num_nodes"] = int(kwargs["x"].size(0))
    return TemporalData(**kwargs)
