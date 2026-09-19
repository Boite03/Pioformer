# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     httpwww.apache.orglicensesLICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

from pioformer.model.embedding import MultipleInputEmbedding
from pioformer.model.embedding import SingleInputEmbedding
from pioformer.data.types import TemporalData
from pioformer.model.utils import init_weights


class GlobalInteractor(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 edge_dim: int,
                 num_modes: int = 6,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 rotate: bool = True) -> None:
        super(GlobalInteractor, self).__init__()
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_modes = num_modes

        if rotate:
            self.rel_embed = MultipleInputEmbedding(in_channels=[edge_dim, edge_dim], out_channel=embed_dim)
        else:
            self.rel_embed = SingleInputEmbedding(in_channel=edge_dim, out_channel=embed_dim)
        self.global_interactor_layers = nn.ModuleList(
            [GlobalInteractorLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
             for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.multihead_proj = nn.Linear(embed_dim, num_modes * embed_dim)
        self.apply(init_weights)

    def encode_scene(self, data: TemporalData, local_embed: torch.Tensor):
        edge_index, _ = subgraph(subset=~data['padding_mask'][:, self.historical_steps - 1], edge_index=data.edge_index)
        rel_pos = data['positions'][edge_index[0], self.historical_steps - 1] - data['positions'][
            edge_index[1], self.historical_steps - 1]
        if data['rotate_mat'] is None:
            rel_embed = self.rel_embed(rel_pos)
        else:
            rel_pos = torch.bmm(rel_pos.unsqueeze(-2), data['rotate_mat'][edge_index[1]]).squeeze(-2)
            rel_theta = data['rotate_angles'][edge_index[0]] - data['rotate_angles'][edge_index[1]]
            rel_direction = torch.stack((torch.cos(rel_theta), torch.sin(rel_theta)), dim=-1)
            rel_embed = self.rel_embed([rel_pos, rel_direction])
        x = local_embed
        for layer in self.global_interactor_layers:
            x = layer(x, edge_index, rel_embed)
        scene_embed = self.norm(x)
        modes = self.multihead_proj(scene_embed).view(-1, self.num_modes, self.embed_dim).transpose(0, 1)
        return modes, scene_embed, rel_embed

    def interact_trajectories(self, data: TemporalData, trajectory_embed: torch.Tensor,
                              rel_embed: torch.Tensor) -> torch.Tensor:
        edge_index, _ = subgraph(subset=~data['padding_mask'][:, self.historical_steps - 1],
                                 edge_index=data.edge_index)
        num_modes, num_nodes, _ = trajectory_embed.shape
        offsets = torch.arange(num_modes, device=edge_index.device).view(-1, 1, 1) * num_nodes
        expanded_edges = (edge_index.unsqueeze(0) + offsets).transpose(0, 1).reshape(2, -1)
        expanded_rel = rel_embed.unsqueeze(0).expand(num_modes, -1, -1).reshape(-1, self.embed_dim)
        x = trajectory_embed.reshape(-1, self.embed_dim)
        for layer in self.global_interactor_layers:
            x = layer(x, expanded_edges, expanded_rel)
        return self.norm(x).view(num_modes, num_nodes, self.embed_dim)

    def forward(self, data: TemporalData, local_embed: torch.Tensor, rel_embed=None):
        if rel_embed is None:
            modes, scene_embed, rel_embed = self.encode_scene(data, local_embed)
            return modes, scene_embed, None, rel_embed
        future = self.interact_trajectories(data, local_embed, rel_embed)
        return None, None, future, rel_embed


class GlobalInteractorLayer(MessagePassing):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(GlobalInteractorLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lin_q_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_v_node = nn.Linear(embed_dim, embed_dim)
        self.lin_v_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))

    def forward(self,
                x: torch.Tensor,
                edge_index: Adj,
                edge_attr: torch.Tensor,
                size: Size = None) -> torch.Tensor:
        x = x + self._mha_block(self.norm1(x), edge_index, edge_attr, size)
        x = x + self._ff_block(self.norm2(x))
        return x

    def message(self,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        query = self.lin_q_node(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key_node = self.lin_k_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key_edge = self.lin_k_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_node = self.lin_v_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_edge = self.lin_v_edge(edge_attr).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * (key_node + key_edge)).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return (value_node + value_edge) * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x))
        return inputs + gate * (self.lin_self(x) - inputs)

    def _mha_block(self,
                   x: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   size: Size) -> torch.Tensor:
        x = self.out_proj(self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size))
        return self.proj_drop(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

