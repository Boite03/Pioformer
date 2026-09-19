from typing import Optional, Tuple

import torch
from torch import nn


class DistanceDropEdge:
    def __init__(self, max_distance: Optional[float] = None) -> None:
        self.max_distance = max_distance

    def __call__(self, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_distance is None:
            return edge_index, edge_attr
        row, col = edge_index
        mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
        return torch.stack((row[mask], col[mask])), edge_attr[mask]


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = module.in_channels / module.groups
        fan_out = module.out_channels / module.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(module.weight, -bound, bound)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                for part in param.chunk(3, 0):
                    nn.init.xavier_uniform_(part)
            elif "weight_hh" in name:
                for part in param.chunk(3, 0):
                    nn.init.orthogonal_(part)
            elif "bias" in name:
                nn.init.zeros_(param)
