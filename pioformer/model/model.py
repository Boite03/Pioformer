from typing import Optional

import torch
from torch import nn

from pioformer.model.decoders import AuxiliaryDecoder, MainDecoder
from pioformer.model.global_interactor import GlobalInteractor
from pioformer.model.hyper_interactor import HyperInteractor
from pioformer.model.local_encoder import LocalEncoder
from pioformer.model.outputs import PioformerOutput
from pioformer.model.refinement import RefinementNetwork
from pioformer.model.stages import TrainingStage


class PioformerModel(nn.Module):
    """Paper-oriented CTN -> TPN -> PRN model without training-framework coupling."""

    def __init__(self, historical_steps: int = 20, future_steps: int = 30, num_modes: int = 6,
                 rotate: bool = True, node_dim: int = 2, edge_dim: int = 2, embed_dim: int = 64,
                 num_heads: int = 8, dropout: float = 0.1, num_temporal_layers: int = 4,
                 num_global_layers: int = 3, local_radius: float = 50.0, parallel: bool = False,
                 hyperedge_size: int = 4, stage: int = 3) -> None:
        super().__init__()
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.stage = TrainingStage.parse(stage)
        self.local_encoder = LocalEncoder(
            historical_steps, node_dim, edge_dim, embed_dim, num_heads, dropout,
            num_temporal_layers, local_radius, parallel,
        )
        self.global_interactor = GlobalInteractor(
            historical_steps, embed_dim, edge_dim, num_modes, num_heads,
            num_global_layers, dropout, rotate,
        )
        self.aux_decoder = AuxiliaryDecoder(embed_dim, embed_dim, future_steps, num_modes)
        self.hyper_interactor = HyperInteractor(embed_dim, num_modes, hyperedge_size)
        self.decoder = MainDecoder(embed_dim, embed_dim, future_steps, num_modes)
        self.refine = RefinementNetwork(embed_dim, num_modes, future_steps)
        self.set_stage(self.stage)

    def set_stage(self, stage: int) -> None:
        self.stage = TrainingStage.parse(stage)
        active = {
            TrainingStage.CTN: (self.local_encoder, self.global_interactor, self.aux_decoder),
            TrainingStage.TPN: (
                self.local_encoder, self.global_interactor, self.aux_decoder,
                self.hyper_interactor, self.decoder,
            ),
            TrainingStage.PRN: (
                self.local_encoder, self.global_interactor, self.aux_decoder,
                self.hyper_interactor, self.decoder, self.refine,
            ),
        }[self.stage]
        active_ids = {id(module) for module in active}
        for child in self.children():
            enabled = id(child) in active_ids
            child.train(enabled)
            for parameter in child.parameters():
                parameter.requires_grad = enabled

    def _set_rotation(self, data) -> Optional[torch.Tensor]:
        if not self.rotate:
            data.rotate_mat = None
            return None
        angles = data.rotate_angles
        sin_values, cos_values = torch.sin(angles), torch.cos(angles)
        rotate_mat = torch.stack((
            torch.stack((cos_values, -sin_values), dim=-1),
            torch.stack((sin_values, cos_values), dim=-1),
        ), dim=-2)
        data.rotate_mat = rotate_mat
        return rotate_mat

    def targets(self, data) -> Optional[torch.Tensor]:
        if getattr(data, "y", None) is None:
            return None
        if not self.rotate:
            return data.y
        rotate_mat = getattr(data, "rotate_mat", None)
        if rotate_mat is None:
            rotate_mat = self._set_rotation(data)
        return torch.bmm(data.y, rotate_mat)

    def forward(self, data, stage: Optional[int] = None) -> PioformerOutput:
        requested_stage = self.stage if stage is None else TrainingStage.parse(stage)
        self._set_rotation(data)
        local_embed = self.local_encoder(data)
        global_modes, global_embed, rel_embed = self.global_interactor.encode_scene(data, local_embed)
        auxiliary = self.aux_decoder(local_embed, global_modes)
        if requested_stage == TrainingStage.CTN:
            return PioformerOutput(auxiliary=auxiliary)

        hyper_embed = self.hyper_interactor(data, local_embed, global_embed)
        proposal = self.decoder(local_embed, global_embed, hyper_embed)
        if requested_stage == TrainingStage.TPN:
            return PioformerOutput(auxiliary=auxiliary, proposal=proposal)

        history = data.x[:, :self.historical_steps]
        if self.rotate:
            history = torch.bmm(history, data.rotate_mat)
        history = history.unsqueeze(0).expand(self.num_modes, -1, -1, -1)
        full_trajectory = torch.cat((history, proposal.positions), dim=2)
        refined = self.refine(
            full_trajectory, proposal, global_embed, local_embed, hyper_embed,
            self.global_interactor, rel_embed, data,
        )
        return PioformerOutput(auxiliary=auxiliary, proposal=proposal, refined=refined)
