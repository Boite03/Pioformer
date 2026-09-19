import torch
from torch import nn

from pioformer.model.outputs import TrajectoryPrediction


class MLP(nn.Module):
    def __init__(self, hidden_size: int, out_features: int = None) -> None:
        super().__init__()
        out_features = hidden_size if out_features is None else out_features
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.layer_norm(self.linear(hidden_states)))


class RefinementNetwork(nn.Module):
    """PRN: bidirectional trajectory encoding plus position and confidence offsets."""

    def __init__(self, hidden_size: int, num_modes: int, future_steps: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.MLP = MLP(2, hidden_size)
        self.MLP_2 = MLP(hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.consist_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )
        self.refine_layer = self._head(hidden_size, future_steps * 2)
        self.refine_layer_pi = self._head(hidden_size, 1)

    @staticmethod
    def _head(hidden_size: int, output_size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_size * 5, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, full_traj: torch.Tensor, proposal: TrajectoryPrediction,
                global_embed: torch.Tensor, local_embed: torch.Tensor, hyper_embed: torch.Tensor,
                global_interactor, rel_embed: torch.Tensor, data) -> TrajectoryPrediction:
        embedded = self.MLP(full_traj)
        embedded = self.MLP_2(embedded) + embedded
        sequence = embedded.reshape(-1, full_traj.size(2), self.hidden_size).transpose(0, 1)
        _, hidden = self.gru(sequence)
        trajectory_embed = (hidden[0] + hidden[1]).view(self.num_modes, -1, self.hidden_size)
        consist_embed = self.consist_mlp(trajectory_embed)
        future_embed = global_interactor.interact_trajectories(data, trajectory_embed, rel_embed)
        fused = torch.cat((
            consist_embed,
            future_embed,
            hyper_embed,
            global_embed.unsqueeze(0).expand(self.num_modes, -1, -1),
            local_embed.unsqueeze(0).expand(self.num_modes, -1, -1),
        ), dim=-1)
        delta = self.refine_layer(fused).view(self.num_modes, -1, self.future_steps, 2)
        logit_delta = self.refine_layer_pi(fused).squeeze(-1).transpose(0, 1)
        return TrajectoryPrediction(
            positions=proposal.positions + delta,
            scale=None,
            logits=proposal.logits + logit_delta,
        )
