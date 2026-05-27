import torch
import logging
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data,Batch 

import math
from typing import Optional

class TrajEncoder(nn.Module):
    """
    输入:
      tau: FloatTensor [B, M, L, F_t] 其中 tau[...,0] 是 road_id（pad=N）
    输出:
      E_t: FloatTensor [B, N, D]
    """
    def __init__(
        self,
        num_nodes: int,
        pad_value: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_traj_len: int = 20,
        feat_dim: int = 3,            # road_id + minutes + weeks
        use_base_node_emb: bool = True,
    ):
        super().__init__()
        assert pad_value == num_nodes

        self.num_nodes = num_nodes
        self.pad_value = pad_value
        self.d_model = d_model
        self.max_traj_len = max_traj_len
        self.use_base_node_emb = use_base_node_emb

        # road id emb（N+1含pad）
        self.node_emb = nn.Embedding(num_nodes + 1, d_model, padding_idx=pad_value)

        # 连续特征投影：minutes/weeks -> d_model
        self.feat_proj = nn.Linear(feat_dim - 1, d_model)  # 去掉 road_id

        # pos emb
        self.pos_emb = nn.Embedding(max_traj_len, d_model)
        self.register_buffer("pos_ids", torch.arange(max_traj_len), persistent=False)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        # tau: [B,M,L,F_t]
        B, M, L, F_t = tau.shape
        assert L <= self.max_traj_len

        road = tau[..., 0].long()                  # [B,M,L]
        cont = tau[..., 1:].float()                # [B,M,L,F_t-1]

        pad = self.pad_value
        kpm = (road == pad)                        # [B,M,L]
        token_mask = ~kpm

        x = self.node_emb(road) + self.feat_proj(cont)  # [B,M,L,D]把道路身份 + 时间信息融合成 token 表示
        pos = self.pos_emb(self.pos_ids[:L].to(tau.device))  # [L,D]
        x = x + pos.view(1, 1, L, self.d_model)

        x = x.view(B * M, L, self.d_model)
        h = self.encoder(x, src_key_padding_mask=kpm.view(B * M, L))
        h = self.out_norm(h).view(B, M, L, self.d_model)

        # token -> node 聚合（按 road_id）
        N, D = self.num_nodes, self.d_model
        ML = M * L
        h_flat = h.reshape(B * ML, D)
        ids_flat = road.reshape(B * ML)
        valid_flat = token_mask.reshape(B * ML)

        valid_idx = valid_flat.nonzero(as_tuple=False).squeeze(1)
        E_t = h.new_zeros((B, N, D))
        if valid_idx.numel() > 0:
            ids_v = ids_flat[valid_idx]
            src = h_flat[valid_idx]
            b_ids = valid_idx // ML

            good = (ids_v >= 0) & (ids_v < N)
            ids_v, src, b_ids = ids_v[good], src[good], b_ids[good]
            if ids_v.numel() > 0:
                flat_index = b_ids * N + ids_v
                node_sum = h.new_zeros((B * N, D))
                node_cnt = h.new_zeros((B * N, 1))
                node_sum.index_add_(0, flat_index, src)
                node_cnt.index_add_(0, flat_index, src.new_ones(src.size(0), 1))
                E_t = (node_sum / node_cnt.clamp_min(1.0)).view(B, N, D)

        if self.use_base_node_emb:
            base = self.node_emb.weight[:N]
            E_t = E_t + base.unsqueeze(0)

        return E_t



class SpatialEncoder(nn.Module):
    """
    高效版：输入 [B,N,F]，输出 [B,N,D]
    不使用 edge_weight
    """
    def __init__(self, in_feature, hidden_features, out_feature, num_layers=3,
                 conv_type="gcn", dropout=0.1, add_skip=True):
        super().__init__()
        hidden_dims = hidden_features * (num_layers - 1)  # 例如 [64,64]
        self.gnn = GNN(
            in_dim=in_feature,
            hidden_dims=hidden_dims,
            out_dim=out_feature,
            conv_type=conv_type,
            dropout=dropout,
            add_skip=add_skip
        )

    @staticmethod
    def batch_edge_index(edge_index, B, N, device):
        # edge_index: [2,E]
        E = edge_index.size(1)
        offsets = (torch.arange(B, device=device) * N).view(1, B, 1)   # [1,B,1]
        edge = edge_index.unsqueeze(1) + offsets                        # [2,B,E]
        return edge.reshape(2, B * E)                                   # [2,B*E]

    def forward(self, node_features, edge_index):
        B, N, F = node_features.shape
        x = node_features.reshape(B * N, F)
        edge_all = self.batch_edge_index(edge_index, B, N, node_features.device)
        out = self.gnn(x, edge_all)
        return out.view(B, N, -1)

    
    
class SparseGNNLayer(nn.Module):
    def __init__(
        self, in_dim, out_dim, conv_type="gcn",
        activation=None, dropout=0.1, add_skip=True, bias=True
    ):
        super().__init__()
        if conv_type == "gcn":
            self.conv = GCNConv(in_dim, out_dim, bias=bias, normalize=True)
        elif conv_type == "sage":
            self.conv = SAGEConv(in_dim, out_dim, bias=bias)
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")

        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.add_skip = add_skip and (in_dim == out_dim)

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)          # ✅ 不再传 edge_weight
        if self.activation is not None:
            out = self.activation(out)
        out = self.dropout(out)
        if self.add_skip:
            out = out + x
        return out


class GNN(nn.Module):
    def __init__(
        self, in_dim, hidden_dims, out_dim,
        conv_type="gcn", dropout=0.1, add_skip=True
    ):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            act = nn.ELU() if i < len(dims) - 2 else None
            self.layers.append(
                SparseGNNLayer(
                    in_dim=dims[i],
                    out_dim=dims[i+1],
                    conv_type=conv_type,
                    activation=act,
                    dropout=dropout,
                    add_skip=add_skip,
                )
            )

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x
