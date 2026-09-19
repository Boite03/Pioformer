import torch
import torch.nn as nn
import torch.nn.functional as F



class Mapping(nn.Module):
    def __init__(self, hidden_size):
        super(Mapping, self).__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size))

    def forward(self, x):
        return self.mlp(x)


class edge_aggregation(nn.Module):
    def __init__(self, input_dim):
        super(edge_aggregation, self).__init__()
        self.agg_mlp = Mapping(input_dim)

    def forward(self, edge_distribution, H, ori):
        batch = edge_distribution.shape[0]
        edges = edge_distribution.shape[1]
        edge_feature = torch.zeros(batch, edges, ori.shape[-1]).type_as(ori)
        edges = torch.matmul(H, ori)
        edge_feature += edge_distribution * self.agg_mlp(edges)

        node_feature = torch.cat((torch.matmul(H.permute(0, 2, 1), edge_feature), ori), dim=-1)
        return node_feature


class MS_HGNN_hyper(nn.Module):

    def __init__(
            self, h_dim=64, scale=2,
    ):
        super(MS_HGNN_hyper, self).__init__()

        self.h_dim = h_dim
        self.scale = scale

        hdim_extend = h_dim
        self.hdim_extend = hdim_extend
        self.edge_types = 1

        self.attention_mlp = nn.Sequential(
            nn.Linear(hdim_extend * 2, hdim_extend),
            nn.LayerNorm(hdim_extend),
            nn.ReLU(inplace=True),
            nn.Linear(hdim_extend, 1))

        self.edge_aggregation_block = edge_aggregation(input_dim=h_dim)
        self.listall = False

        self.node2edge_input_mapping = Mapping(self.h_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(hdim_extend * 2, hdim_extend),
            nn.LayerNorm(hdim_extend),
            nn.ReLU(inplace=True),
            nn.Linear(hdim_extend, hdim_extend))


    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def edge2node(self, x, ori, H, idx):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = self.edge_aggregation_block(x, H, ori)
        return incoming / incoming.size(1)

    def node2edge(self, x, H, idx):
        x = self.node2edge_input_mapping(x)
        edge_init = torch.matmul(H, x)
        node_num = x.shape[1]
        edge_num = edge_init.shape[1]
        x_rep = (x[:, :, None, :].transpose(2, 1)).repeat(1, edge_num, 1, 1)
        edge_rep = edge_init[:, :, None, :].repeat(1, 1, node_num, 1)
        node_edge_cat = torch.cat((x_rep, edge_rep), dim=-1)
        attention_weight = self.attention_mlp(node_edge_cat)[:, :, :, 0]
        H_weight = attention_weight * H
        H_weight = F.softmax(H_weight, dim=2)
        H_weight = H_weight * H
        edges = torch.matmul(H_weight, x)
        return edges

    def init_adj_attention(self, feat, feat_corr, scale_factor=2):
        batch = feat.shape[0]
        actor_number = feat.shape[1]
        if scale_factor == actor_number:
            H_matrix = torch.ones(batch, 1, actor_number).type_as(feat)
            return H_matrix
        group_size = scale_factor
        if group_size < 1:
            group_size = 1

        _, indice = torch.topk(feat_corr, dim=2, k=group_size, largest=True)
        H_matrix = torch.zeros(batch, actor_number, actor_number).type_as(feat)
        H_matrix = H_matrix.scatter(2, indice, 1)

        return H_matrix

    def forward(self, h_states, corr, scale):
        curr_hidden = h_states  # (num_pred, h_dim)
        H = self.init_adj_attention(curr_hidden, corr, scale_factor=scale)

        edge_feat = self.node2edge(curr_hidden, H, idx=0)
        node_feat = curr_hidden
        node_feat = self.edge2node(edge_feat, node_feat, H, 0)
        node_feat = self.output_proj(node_feat)
        return node_feat, node_feat



