from typing import Optional

import torch
from torch_geometric.data import Data


class TemporalData(Data):
    """PyG container with correct batching offsets for lane-to-actor edges."""

    def __init__(self, x: Optional[torch.Tensor] = None, positions: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 padding_mask: Optional[torch.Tensor] = None, bos_mask: Optional[torch.Tensor] = None,
                 rotate_angles: Optional[torch.Tensor] = None, lane_vectors: Optional[torch.Tensor] = None,
                 is_intersections: Optional[torch.Tensor] = None, turn_directions: Optional[torch.Tensor] = None,
                 traffic_controls: Optional[torch.Tensor] = None,
                 lane_actor_index: Optional[torch.Tensor] = None,
                 lane_actor_vectors: Optional[torch.Tensor] = None, **kwargs) -> None:
        if x is None:
            super().__init__(**kwargs)
            return
        super().__init__(x=x, positions=positions, edge_index=edge_index, y=y, num_nodes=num_nodes,
                         padding_mask=padding_mask, bos_mask=bos_mask, rotate_angles=rotate_angles,
                         lane_vectors=lane_vectors, is_intersections=is_intersections,
                          turn_directions=turn_directions, traffic_controls=traffic_controls,
                          lane_actor_index=lane_actor_index, lane_actor_vectors=lane_actor_vectors,
                          **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "lane_actor_index":
            return torch.tensor([[self.lane_vectors.size(0)], [self.num_nodes]], device=value.device)
        return super().__inc__(key, value, *args, **kwargs)
