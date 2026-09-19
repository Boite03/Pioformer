from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from pioformer.model.hypergraph_layers import MS_HGNN_hyper


class HyperInteractor(nn.Module):
    """Scene-local high/low-order interaction used by TPN."""

    def __init__(self, hidden_dim: int, num_modes: int = 6, group_size: int = 4) -> None:
        super().__init__()
        self.model_dim = hidden_dim
        self.num_modes = num_modes
        self.group_size = group_size
        self.simple_embed = nn.Linear(2 * hidden_dim, hidden_dim)
        self.interaction_hyper = MS_HGNN_hyper(h_dim=hidden_dim)
        self.straight_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.weight = nn.Linear(2 * hidden_dim, 1)
        self.multihead_proj = nn.Linear(hidden_dim, hidden_dim * num_modes)

    @staticmethod
    def _scene_ranges(data, num_nodes: int) -> List[Tuple[int, int]]:
        if getattr(data, "ptr", None) is not None:
            ptr = data.ptr.detach().cpu().tolist()
            return list(zip(ptr[:-1], ptr[1:]))
        batch = getattr(data, "batch", None)
        if batch is not None and batch.numel():
            counts = torch.bincount(batch, minlength=int(batch.max()) + 1).detach().cpu().tolist()
            result, start = [], 0
            for count in counts:
                result.append((start, start + count))
                start += count
            return result
        return [(0, num_nodes)]

    def forward(self, data, local_embed: torch.Tensor, global_embed: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((global_embed, local_embed), dim=-1)
        outputs = []
        for start, end in self._scene_ranges(data, local_embed.size(0)):
            scene_input = combined[start:end].unsqueeze(0)
            scene = self.simple_embed(scene_input)
            query = F.normalize(scene, p=2, dim=-1)
            affinity = torch.matmul(query, query.transpose(1, 2))
            size = min(self.group_size, end - start)
            high_order, _ = self.interaction_hyper(scene, affinity, size)
            low_order = self.straight_layer(scene)
            gate = torch.sigmoid(self.weight(scene_input))
            outputs.append((gate * high_order + (1.0 - gate) * low_order).squeeze(0))
        output = torch.cat(outputs, dim=0)
        return self.multihead_proj(output).view(-1, self.num_modes, self.model_dim).transpose(0, 1)
