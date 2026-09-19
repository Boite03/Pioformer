import torch
from torch import nn
from torch.nn import functional as F

from pioformer.model.outputs import TrajectoryPrediction
from pioformer.model.utils import init_weights


class AuxiliaryDecoder(nn.Module):
    """CTN decoder. Member names intentionally match historical checkpoints."""

    def __init__(self, local_channels: int, global_channels: int, future_steps: int,
                 num_modes: int, min_scale: float = 1e-3) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.min_scale = min_scale
        self.aggr_embed = nn.Sequential(
            nn.Linear(local_channels + global_channels, local_channels),
            nn.LayerNorm(local_channels),
            nn.ReLU(inplace=True),
        )
        self.loc = nn.Sequential(
            nn.Linear(local_channels, local_channels), nn.LayerNorm(local_channels), nn.ReLU(inplace=True),
            nn.Linear(local_channels, future_steps * 2),
        )
        self.scale = nn.Sequential(
            nn.Linear(local_channels, local_channels), nn.LayerNorm(local_channels), nn.ReLU(inplace=True),
            nn.Linear(local_channels, future_steps * 2),
        )
        self.pi = nn.Sequential(
            nn.Linear(local_channels + global_channels, local_channels),
            nn.LayerNorm(local_channels), nn.ReLU(inplace=True),
            nn.Linear(local_channels, local_channels), nn.LayerNorm(local_channels), nn.ReLU(inplace=True),
            nn.Linear(local_channels, 1),
        )
        self.apply(init_weights)

    def forward(self, local_embed: torch.Tensor, global_embed: torch.Tensor) -> TrajectoryPrediction:
        local_modes = local_embed.unsqueeze(0).expand(self.num_modes, -1, -1)
        features = torch.cat((global_embed, local_modes), dim=-1)
        logits = self.pi(features).squeeze(-1).transpose(0, 1)
        hidden = self.aggr_embed(features)
        positions = self.loc(hidden).view(self.num_modes, -1, self.future_steps, 2)
        scale = F.elu(self.scale(hidden), alpha=1.0).view_as(positions) + 1.0 + self.min_scale
        return TrajectoryPrediction(positions=positions, scale=scale, logits=logits)


class MainDecoder(nn.Module):
    """TPN decoder fusing local, scene-global, and hypergraph features."""

    def __init__(self, local_channels: int, global_channels: int, future_steps: int,
                 num_modes: int, min_scale: float = 1e-3) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.min_scale = min_scale
        fused_channels = global_channels + 2 * local_channels
        self.hybrid_aggr_embed = nn.Sequential(
            nn.Linear(fused_channels, local_channels), nn.LayerNorm(local_channels), nn.ReLU(inplace=True),
        )
        self.loc = nn.Sequential(
            nn.Linear(local_channels, local_channels), nn.LayerNorm(local_channels), nn.ReLU(inplace=True),
            nn.Linear(local_channels, future_steps * 2),
        )
        self.scale = nn.Sequential(
            nn.Linear(local_channels, local_channels), nn.LayerNorm(local_channels), nn.ReLU(inplace=True),
            nn.Linear(local_channels, future_steps * 2),
        )
        self.hybrid_pi = nn.Sequential(
            nn.Linear(local_channels + 2 * global_channels, local_channels),
            nn.LayerNorm(local_channels), nn.ReLU(inplace=True),
            nn.Linear(local_channels, local_channels), nn.LayerNorm(local_channels), nn.ReLU(inplace=True),
            nn.Linear(local_channels, 1),
        )
        self.apply(init_weights)

    def forward(self, local_embed: torch.Tensor, global_embed: torch.Tensor,
                hyper_embed: torch.Tensor) -> TrajectoryPrediction:
        local_modes = local_embed.unsqueeze(0).expand(self.num_modes, -1, -1)
        global_modes = global_embed.unsqueeze(0).expand(self.num_modes, -1, -1)
        features = torch.cat((local_modes, global_modes, hyper_embed), dim=-1)
        logits = self.hybrid_pi(features).squeeze(-1).transpose(0, 1)
        hidden = self.hybrid_aggr_embed(features)
        positions = self.loc(hidden).view(self.num_modes, -1, self.future_steps, 2)
        scale = F.elu(self.scale(hidden), alpha=1.0).view_as(positions) + 1.0 + self.min_scale
        return TrajectoryPrediction(positions=positions, scale=scale, logits=logits)
