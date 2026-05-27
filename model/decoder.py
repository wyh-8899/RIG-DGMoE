import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKMoEOut(nn.Module):
    """
    Top-k sparse MoE for large out_dim (e.g., num_nodes).
    - Only computes k experts per token.
    - Returns (y, aux) where aux has load-balance stats.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_experts: int = 4,
        k: int = 2,
        gate_dropout: float = 0.0,
        noisy_gate: bool = False,
        noise_std: float = 1.0,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.k = int(k)
        assert 1 <= self.k <= self.num_experts

        self.experts = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(self.num_experts)])
        self.gate = nn.Linear(in_dim, self.num_experts)

        self.gate_dropout = nn.Dropout(gate_dropout) if gate_dropout > 0 else nn.Identity()
        self.noisy_gate = bool(noisy_gate)
        self.noise_std = float(noise_std)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        x: [T, in_dim]  (T = B*M or B*N etc.)
        returns:
          y: [T, out_dim]
          aux (optional): dict with importance/load and balance_loss
        """
        T, _ = x.shape

        gate_logits = self.gate_dropout(self.gate(x))  # [T, E]

        # optional: noisy gating to reduce collapse
        if self.noisy_gate and self.training:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.noise_std

        # top-k selection
        topk_val, topk_idx = torch.topk(gate_logits, k=self.k, dim=-1)  # [T,k], [T,k]
        topk_w = F.softmax(topk_val, dim=-1)                            # [T,k]

        # compute balance stats (Switch-style)
        # importance: sum of softmax probs over ALL experts (dense softmax) is expensive;
        # use approximated importance from topk weights:
        importance = torch.zeros(T, self.num_experts, device=x.device, dtype=x.dtype)
        importance.scatter_add_(dim=1, index=topk_idx, src=topk_w)  # [T,E]

        # load: how many tokens routed to each expert (count top-1 route) or top-k count
        # here use top-1 (most common)
        top1 = topk_idx[:, 0]  # [T]
        load = torch.bincount(top1, minlength=self.num_experts).to(x.dtype) / float(T)  # [E]
        imp = importance.sum(dim=0) / float(T)  # [E]

        # balance loss: E * sum(imp * load)
        balance_loss = (self.num_experts * (imp * load).sum())

        # sparse expert compute
        y = torch.zeros(T, self.experts[0].out_features, device=x.device, dtype=x.dtype)

        # route tokens to experts (for each expert, gather routed tokens)
        # we compute contributions for all selected experts (k)
        for e_id, expert in enumerate(self.experts):
            # mask tokens where this expert is in topk
            # positions: [P,2] => (token_idx, which_of_k)
            pos = (topk_idx == e_id).nonzero(as_tuple=False)
            if pos.numel() == 0:
                continue

            tok_idx = pos[:, 0]          # [P]
            k_slot = pos[:, 1]           # [P]
            w = topk_w[tok_idx, k_slot]  # [P]

            y_e = expert(x[tok_idx])     # [P, out_dim]
            y[tok_idx] += y_e * w.unsqueeze(-1)

        if return_aux:
            aux = {
                "topk_idx": topk_idx,                 # [T,k]
                "topk_w": topk_w,                     # [T,k]
                "importance": imp.detach(),           # [E]
                "load": load.detach(),                # [E]
                "balance_loss": balance_loss,         # scalar tensor
            }
            return y, aux

        return y

    @torch.no_grad()
    def copy_from_linear(self, base_linear: nn.Linear):
        for e in self.experts:
            e.weight.copy_(base_linear.weight)
            e.bias.copy_(base_linear.bias)
class TemporalRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TemporalRNN, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim = hidden_dim
        
        # 使用 GRUCell
        self.cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, x_t, h_prev=None):
        """
        Args:
        x_t: 输入特征 [B, N, D_in]
        h_prev: 上一时刻的隐状态 [B, N, D_h]
        return: h_new: 更新后的隐状态 [B, N, D_h]
        """
        B, N, D_in = x_t.shape
        x_flat = x_t.reshape(B * N, D_in)   # [B, N, D_in] -> [B*N, D_in]
        
        if h_prev is None:   
            h_prev = torch.zeros(B * N, self.hidden_dim, device=x_t.device, dtype=x_t.dtype)
        else:
            h_prev = h_prev.reshape(B * N, self.hidden_dim)

        h_new_flat = self.cell(x_flat, h_prev)   # [B*N, D_h]
        h_new = h_new_flat.reshape(B, N, self.hidden_dim)  # [B, N, D_h]
        return h_new


import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajDecoder(nn.Module):
    """
    输入:
      Tau_latest: [B, M, L, 3]  (3: [road_id, minutes, weeks])
      P_t:        [B, N, N]
      I_t:        [B, N, D_i]
    输出:
      Tau_next:   [B, M, L, 3]下一步的轨迹 窗口左移 + append 新 road_id
    可选输出:
      logits:     [B, M, N] 每条轨迹下一个 road_id 在 N 个节点上的分类 logits
    """

    def __init__(
        self,
        num_nodes: int,
        d_i: int,
        id_emb: int = 64,
        hidden: int = 128,
        dropout: float = 0.1,
        prior_logit_weight: float = 1.0,
        eps: float = 1e-7,
        moe_traj_out: bool = False,
        moe_num_experts: int = 4,
        moe_gate_dropout: float = 0.0,
        moe_sync_from_base: bool = True,
        moe_top_k: int = 2,
        moe_noisy_gate: bool = False,
        moe_noise_std: float = 1.0,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.pad_value = int(num_nodes)  # pad == N

        self.prior_logit_weight = float(prior_logit_weight)
        self.eps = float(eps)

        # 1) road id emb (N+1 含 pad)
        self.id_emb = nn.Embedding(self.num_nodes + 1, id_emb, padding_idx=self.pad_value)

        # 2) encode L-token window
        #    input: [id_emb, minutes, weeks, impedance(last_road)]
        self.rnn = nn.GRU(id_emb + 2 + d_i, hidden, batch_first=True)

        # 3) predict next road logits (N classes: 0..N-1)
        self.out = nn.Linear(hidden, self.num_nodes)
        self.drop = nn.Dropout(dropout)

        self.moe_traj_out = bool(moe_traj_out)
        self.moe_sync_from_base = bool(moe_sync_from_base)

        if self.moe_traj_out:
            self.moe_out = TopKMoEOut(
                in_dim=hidden,
                out_dim=self.num_nodes,
                num_experts=int(moe_num_experts),
                k=int(moe_top_k),
                gate_dropout=float(moe_gate_dropout),
                noisy_gate=bool(moe_noisy_gate),
                noise_std=float(moe_noise_std),
            )
        else:
            self.moe_out = None
        self._moe_synced = False

    @torch.no_grad()
    def _sample_from_logits(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        logits: [B,M,N] -> sampled next road [B,M]
        """
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)

        probs = F.softmax(logits / temperature, dim=-1)  # [B,M,N]
        B, M, N = probs.shape
        probs2 = probs.view(B * M, N)
        idx = torch.multinomial(probs2, num_samples=1).view(B, M)  # [B,M]
        return idx

    def forward(
        self,
        Tau_latest: torch.Tensor,
        P_nb, nb_idx, nb_mask,
        I_t: torch.Tensor,
        *,
        return_logits: bool = False,
        # teacher forcing：提供 ground-truth 的 next road_id（形状 [B,M]）
        target_next_road: Optional[torch.Tensor] = None,
        teacher_forcing: bool = False,
        # 推理时可选：采样而非 argmax
        sample: bool = False,
        temperature: float = 1.0,
        # 可选：用外部时间特征覆盖最后 token 的 minutes/weeks（形状 [B,2] 或 [B,M,2]）
        time_next: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        """
        返回:
          - 默认: Tau_next
          - return_logits=True: (Tau_next, logits)
        """
        B, M, L, Ft = Tau_latest.shape
        assert Ft == 3, f"Tau_latest last dim must be 3, got {Ft}"
        N = self.num_nodes
        device = Tau_latest.device

        # ---- unpack ----
        road = Tau_latest[..., 0].long().clamp(0, self.pad_value)   # [B,M,L]
        minutes = Tau_latest[..., 1].float()                        # [B,M,L]
        weeks = Tau_latest[..., 2].float()                          # [B,M,L]

        # road: [B,M,L]  pad = self.pad_value
        pad = self.pad_value
        mask = (road != pad)                      # [B,M,L] bool
        valid_car = mask.any(dim=-1)              # [B,M] 只要出现过非pad就算有效

        # 计算最后一个非pad位置
        lengths = mask.long().sum(dim=-1).clamp_min(1)   # [B,M]
        last_idx = (lengths - 1).unsqueeze(-1)           # [B,M,1]

        road_last = road.gather(dim=-1, index=last_idx).squeeze(-1)  # [B,M]
        # 对无效 car（全pad）设成0
        road_last = torch.where(valid_car, road_last, torch.zeros_like(road_last))

        # ---- impedance of last road (pad -> zero) ----
        road_last_safe = road_last.clamp(0, N - 1)                  # [B,M]
        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, M)
        I_last = I_t[batch_idx, road_last_safe]                     # [B,M,D_i]
        I_last = I_last * valid_car.unsqueeze(-1)                   # pad car -> 0

        # ---- RNN encode L tokens ----
        id_e = self.id_emb(road)                                    # [B,M,L,id_emb]
        I_rep = I_last.unsqueeze(2).expand(B, M, L, -1)             # [B,M,L,D_i]
        rnn_in = torch.cat(
            [id_e, minutes.unsqueeze(-1), weeks.unsqueeze(-1), I_rep],
            dim=-1
        ).view(B * M, L, -1)                                        # [B*M,L,*]

        _, h_last = self.rnn(rnn_in)                                # h_last: [1,B*M,H]
        h_last = self.drop(h_last.squeeze(0))  # [B*M,H]

        moe_aux = None
        if self.moe_out is not None:
            if (not self._moe_synced) and self.moe_sync_from_base:
                self.moe_out.copy_from_linear(self.out)
                self._moe_synced = True

            if return_aux:
                logits_flat, moe_aux = self.moe_out(h_last, return_aux=True)   # [B*M,N], dict
            else:
                logits_flat = self.moe_out(h_last, return_aux=False)           # [B*M,N]
        else:
            logits_flat = self.out(h_last)
        logits = logits_flat.view(B, M, N)

        # invalid cars: keep logits finite
        logits = logits.masked_fill(~valid_car.unsqueeze(-1), 0.0)  # mask: [B,M,1]

        if self.prior_logit_weight != 0.0:
            P_row = P_nb[batch_idx, road_last_safe]          # [B,M,K]
            nb_ids = nb_idx[road_last_safe]                  # [B,M,K]
            nb_ok  = nb_mask[road_last_safe]                 # [B,M,K] bool

            eps = self.eps
            delta = (torch.log(P_row.clamp_min(eps)) - math.log(eps))    # [B,M,K]
            delta = delta * nb_ok.float() * valid_car.unsqueeze(-1).float()

            idx_safe = torch.where(nb_ok, nb_ids, torch.zeros_like(nb_ids))
            logits.scatter_add_(dim=-1, index=idx_safe, src=self.prior_logit_weight * delta)

        # ---- choose next road ----
        if teacher_forcing and (target_next_road is not None):
            next_road = target_next_road.long().clamp(0, self.pad_value)  # [B,M]
        else:
            if sample:
                next_road = self._sample_from_logits(logits, temperature=temperature)  # [B,M]
            else:
                next_road = torch.argmax(logits, dim=-1)  # [B,M]

        # invalid cars -> pad
        next_road = torch.where(
            valid_car,
            next_road,
            torch.full_like(next_road, self.pad_value)
        )

        # ---- build Tau_next (shift-left + append) ----
        Tau_next = Tau_latest.clone()
        Tau_next[:, :, :-1, :] = Tau_latest[:, :, 1:, :]
        Tau_next[:, :, -1, 0] = next_road.float()

        # minutes/weeks：默认沿用上一 token；如果你想用 Et[t+1]，可传 time_next 覆盖
        if time_next is None:
            Tau_next[:, :, -1, 1] = Tau_latest[:, :, -1, 1]
            Tau_next[:, :, -1, 2] = Tau_latest[:, :, -1, 2]
        else:
            # time_next: [B,2] or [B,M,2]
            if time_next.dim() == 2:
                tn = time_next.unsqueeze(1).expand(B, M, 2)  # [B,M,2]
            else:
                tn = time_next
            Tau_next[:, :, -1, 1] = tn[..., 0]
            Tau_next[:, :, -1, 2] = tn[..., 1]

        if return_logits and return_aux:
            return Tau_next, logits, moe_aux
        if return_logits:
            return Tau_next, logits
        if return_aux:
            return Tau_next, moe_aux
        return Tau_next


class FlowPropagation(MessagePassing):
    """
    Sparse Impedance-GAT Flow Propagation

    - Graph topology: edge_index [2, E] (static)
    - Q/K context: concat([X_t, I_t]) -> [B,N,D_s + D_i]
    - V: S_latest -> [B,N,F_s]
    - Physics prior: use log(p_ij) from P_t[b,i,j] as bias in attention score

    Inputs:
      S_latest:  [B, N, F_s]
      X_t:       [B, N, D_s]
      I_t:       [B, N, D_i]
      P_t:       [B, N, N]
      edge_index:[2, E]

    Output:
      S_next:    [B, N, F_s]
    """

    def __init__(self, F_s: int, D_s: int, D_i: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__(aggr="add", node_dim=0)

        self.F_s = F_s
        self.D_ctx = D_s + D_i
        self.num_heads = num_heads
        assert self.D_ctx % num_heads == 0, "D_s + D_i must be divisible by num_heads"
        self.head_dim = self.D_ctx // num_heads

        # Q/K/V projections
        self.lin_q = nn.Linear(self.D_ctx, self.D_ctx)
        self.lin_k = nn.Linear(self.D_ctx, self.D_ctx)
        self.lin_v = nn.Linear(F_s, F_s * num_heads)

        # output fusion
        self.lin_out = nn.Linear(F_s * num_heads, F_s)

        # gated residual
        self.update_gate = nn.Sequential(
            nn.Linear(F_s * 2, F_s),
            nn.Sigmoid()
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.Identity() if F_s == 1 else nn.LayerNorm(F_s)

        # one-slot cache (avoid unbounded growth)
        self._cache_key = None
        self._cache_edge = None

    @torch.no_grad()
    def _batched_edge_index(self, edge_index: torch.Tensor, B: int, N: int, device: torch.device) -> torch.Tensor:
        """
        Build a batched edge_index for B graphs sharing the same topology.
        Returns: [2, B*E], where graph b has node indices shifted by b*N.

        Cached per (B, N, device, edge_index ptr).
        """
        dev_key = (device.type, device.index)
        key = (B, N, dev_key, edge_index.data_ptr())
        if self._cache_key == key and self._cache_edge is not None:
            return self._cache_edge

        E = edge_index.size(1)
        # [2,E] -> [2,B,E]
        edge_rep = edge_index.unsqueeze(1).repeat(1, B, 1).contiguous()
        offsets = (torch.arange(B, device=device) * N).view(1, B, 1)  # [1,B,1]
        edge_all = (edge_rep + offsets).view(2, B * E)                # [2,B*E]

        self._cache_key = key
        self._cache_edge = edge_all
        return edge_all

    def forward(self, S_latest, X_t, I_t, P_nb, edge_index, edge2k):
        B, N, _ = S_latest.shape
        device = S_latest.device
        edge_index_all = self._batched_edge_index(edge_index, B, N, device)

        # flatten node tensors
        S_flat = S_latest.reshape(B * N, self.F_s)
        context_flat = torch.cat([X_t, I_t], dim=-1).reshape(B * N, self.D_ctx)

        src = edge_index[0]  # [E]
        dst = edge_index[1]  # [E]
        E = src.numel()

        # 从 P_nb 取每条边的概率
        P_src = P_nb[:, src, :]                                      # [B,E,K]
        k_idx = edge2k.view(1, E, 1).expand(B, E, 1)                 # [B,E,1]
        P_edge_prob = P_src.gather(-1, k_idx).squeeze(-1)            # [B,E]
        P_edge_prob_flat = P_edge_prob.contiguous().view(B * E, 1)   # [B*E,1]

        # message passing on the big batched graph
        S_update_flat = self.propagate(
            edge_index_all,
            S=S_flat,                  # Value uses S_j
            context=context_flat,      # Q/K use context_{i/j}
            P_edge_prob=P_edge_prob_flat
        )  # [B*N, F_s]

        S_update = S_update_flat.view(B, N, self.F_s)

        # gated residual + norm
        z = self.update_gate(torch.cat([S_latest, S_update], dim=-1))   # [B,N,F_s]
        S_next = self.layer_norm(S_latest + z * S_update)               # [B,N,F_s]
        return S_next

    def message(self, context_i, context_j, S_j, P_edge_prob, index):
        """
        context_i:   [B*E, D_ctx]
        context_j:   [B*E, D_ctx]
        S_j:         [B*E, F_s]
        P_edge_prob: [B*E, 1]
        index:       [B*E] target node indices for softmax grouping
        """
        Q = self.lin_q(context_i).view(-1, self.num_heads, self.head_dim)  # [B*E,H,hd]
        K = self.lin_k(context_j).view(-1, self.num_heads, self.head_dim)  # [B*E,H,hd]
        V = self.lin_v(S_j).view(-1, self.num_heads, self.F_s)            # [B*E,H,F_s]

        score = (Q * K).sum(dim=-1) / (self.head_dim ** 0.5)              # [B*E,H]
        score = score + torch.log(P_edge_prob.clamp_min(1e-7))            # physics bias

        alpha = softmax(score, index)                                      # [B*E,H]
        alpha = self.attn_dropout(alpha)

        return alpha.unsqueeze(-1) * V                                     # [B*E,H,F_s]

    def update(self, aggr_out):
        """
        aggr_out: [B*N, H, F_s] (PyG stacks head dim for us because message returns [*,H,F_s])
        """
        out = aggr_out.view(aggr_out.size(0), -1)                          # [B*N, H*F_s]
        return self.lin_out(out)                                           # [B*N, F_s]


class MultiLayerFlowPropagation(nn.Module):
    def __init__(self, F_s, D_s, D_i, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            FlowPropagation(F_s, D_s, D_i, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.activation = nn.ELU()

    def forward(self, S_latest, X_t, I_t, P_nb, edge_index, edge2k):
        curr = S_latest
        for i, layer in enumerate(self.layers):
            curr = layer(curr, X_t, I_t, P_nb, edge_index, edge2k)
            if i < self.num_layers - 1:
                curr = self.activation(curr)
        return curr
