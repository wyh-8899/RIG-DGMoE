import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super().__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=heads, concat=True)
        self.conv2 = GATConv(out_channels * heads, out_channels, heads=heads, concat=False)

    def forward(self, x, edge_index, edge_weights):
        x = self.conv1(x, edge_index, edge_attr=edge_weights)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_weights)
        return x


class CrossModalAttention(nn.Module):
    """
    q:  [B, Lq, D]
    kv: [B, Lk, D]
    out:[B, Lq, D]
    """
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, q, kv):
        B, Lq, D = q.shape
        _, Lk, _ = kv.shape

        q0 = q
        Q = self.w_q(q).view(B, Lq, self.num_heads, self.d_head).transpose(1, 2)   # [B,H,Lq,Dh]
        K = self.w_k(kv).view(B, Lk, self.num_heads, self.d_head).transpose(1, 2)  # [B,H,Lk,Dh]
        V = self.w_v(kv).view(B, Lk, self.num_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)        # [B,H,Lq,Lk]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                                                  # [B,H,Lq,Dh]
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        out = self.w_o(out)

        return self.ln(q0 + out)


class BiCoAttentionBlock(nn.Module):
    """
    双向 co-attention:
      state <- traj
      traj  <- state
    """
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.state_from_traj = CrossModalAttention(d_model, num_heads, dropout)
        self.traj_from_state = CrossModalAttention(d_model, num_heads, dropout)

    def forward(self, hx, he):
        out_state = self.state_from_traj(hx, he)   # Q=hx, K/V=he
        out_traj = self.traj_from_state(he, hx)    # Q=he, K/V=hx
        return out_state, out_traj