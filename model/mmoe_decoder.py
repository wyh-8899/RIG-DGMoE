import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, BiCoAttentionBlock


class StateExpert(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, hx, he):
        out = self.ffn(hx)
        return out, torch.zeros_like(he)


class TrajExpert(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, hx, he):
        out = self.ffn(he)
        return torch.zeros_like(hx), out


class CoAttentionExpert(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.block = BiCoAttentionBlock(d_model, num_heads, dropout)

    def forward(self, hx, he):
        out_state, out_traj = self.block(hx, he)
        return out_state, out_traj


class DualGate(nn.Module):
    def __init__(self, d_model: int, num_experts: int, hidden_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        #in_dim = d_model * 5
        in_dim = d_model * 2
        hid = d_model * hidden_mult

        self.state_gate = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, num_experts),
        )
        self.traj_gate = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, num_experts),
        )

    def forward(self, h_state_ph, h_traj_ph, gap_state, gap_traj, tau: float = 1.0):
        # delta = gap_state - gap_traj
        # prod = gap_state * gap_traj

        # u_state = torch.cat([h_state_ph, gap_state, gap_traj, delta, prod], dim=-1)
        # u_traj = torch.cat([h_traj_ph, gap_state, gap_traj, delta, prod], dim=-1)
        # g_state = F.softmax(self.state_gate(u_state) / tau, dim=-1)  # [B, K]
        # g_traj = F.softmax(self.traj_gate(u_traj) / tau, dim=-1)     # [B, K]
        u = torch.cat([h_state_ph, h_traj_ph], dim=-1)
        g_state = F.softmax(self.state_gate(u) / tau, dim=-1)
        g_traj = F.softmax(self.traj_gate(u) / tau, dim=-1)
        return g_state, g_traj


class SingleGate(nn.Module):
    """单一共享门控：state 和 traj 使用同一组 gate 权重"""
    def __init__(self, d_model: int, num_experts: int, hidden_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        in_dim = d_model * 2
        hid = d_model * hidden_mult

        self.shared_gate = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, num_experts),
        )

    def forward(self, h_state_ph, h_traj_ph, gap_state, gap_traj, tau: float = 1.0):
        u = torch.cat([h_state_ph, h_traj_ph], dim=-1)
        g_shared = F.softmax(self.shared_gate(u) / tau, dim=-1)  # [B, K]
        return g_shared, g_shared


class SingleGateMMoEDecoder(nn.Module):
    """消融变体：用单一共享门控替代双门控"""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.experts = nn.ModuleList([
            StateExpert(d_model),
            TrajExpert(d_model),
            CoAttentionExpert(d_model, num_heads, dropout),
            CoAttentionExpert(d_model, num_heads, dropout),
        ])
        self.num_experts = len(self.experts)
        self.gates = SingleGate(d_model, self.num_experts, dropout=dropout)

        self.state_norm = nn.LayerNorm(d_model)
        self.traj_norm = nn.LayerNorm(d_model)

    def forward(self, Hx, He, h_state_ph, h_traj_ph):
        gap_state = Hx.mean(dim=1)
        gap_traj = He.mean(dim=1)

        g_state, g_traj = self.gates(h_state_ph, h_traj_ph, gap_state, gap_traj)

        state_out = Hx
        traj_out = He

        for k, expert in enumerate(self.experts):
            e_state, e_traj = expert(Hx, He)
            state_out = state_out + g_state[:, k].view(-1, 1, 1) * e_state
            traj_out = traj_out + g_traj[:, k].view(-1, 1, 1) * e_traj

        state_out = self.state_norm(state_out)
        traj_out = self.traj_norm(traj_out)

        return {
            "state_feat": state_out,
            "traj_feat": traj_out,
            "gate_state": g_state,
            "gate_traj": g_traj,
        }


class NoCoAttnMMoEDecoder(nn.Module):
    """消融变体：去掉 CoAttentionExpert，只保留 StateExpert + TrajExpert + DualGate"""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.experts = nn.ModuleList([
            StateExpert(d_model),
            TrajExpert(d_model),
        ])
        self.num_experts = len(self.experts)
        self.gates = DualGate(d_model, self.num_experts, dropout=dropout)

        self.state_norm = nn.LayerNorm(d_model)
        self.traj_norm = nn.LayerNorm(d_model)

    def forward(self, Hx, He, h_state_ph, h_traj_ph):
        gap_state = Hx.mean(dim=1)
        gap_traj = He.mean(dim=1)

        g_state, g_traj = self.gates(h_state_ph, h_traj_ph, gap_state, gap_traj)

        state_out = Hx
        traj_out = He

        for k, expert in enumerate(self.experts):
            e_state, e_traj = expert(Hx, He)
            state_out = state_out + g_state[:, k].view(-1, 1, 1) * e_state
            traj_out = traj_out + g_traj[:, k].view(-1, 1, 1) * e_traj

        state_out = self.state_norm(state_out)
        traj_out = self.traj_norm(traj_out)

        return {
            "state_feat": state_out,
            "traj_feat": traj_out,
            "gate_state": g_state,
            "gate_traj": g_traj,
        }


class DualGateMMoEDecoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.experts = nn.ModuleList([
            StateExpert(d_model),
            TrajExpert(d_model),
            CoAttentionExpert(d_model, num_heads, dropout),
            CoAttentionExpert(d_model, num_heads, dropout),
        ])
        self.num_experts = len(self.experts)
        self.gates = DualGate(d_model, self.num_experts, dropout=dropout)

        self.state_norm = nn.LayerNorm(d_model)
        self.traj_norm = nn.LayerNorm(d_model)

    def forward(self, Hx, He, h_state_ph, h_traj_ph):
        """
        Hx: [B, Kx, D]
        He: [B, Ke, D]
        h_state_ph: [B, D]
        h_traj_ph : [B, D]
        """
        gap_state = Hx.mean(dim=1)   # [B, D]
        gap_traj = He.mean(dim=1)    # [B, D]

        g_state, g_traj = self.gates(h_state_ph, h_traj_ph, gap_state, gap_traj)

        state_out = Hx
        traj_out = He

        for k, expert in enumerate(self.experts):
            e_state, e_traj = expert(Hx, He)
            state_out = state_out + g_state[:, k].view(-1, 1, 1) * e_state
            traj_out = traj_out + g_traj[:, k].view(-1, 1, 1) * e_traj

        state_out = self.state_norm(state_out)
        traj_out = self.traj_norm(traj_out)

        return {
            "state_feat": state_out,
            "traj_feat": traj_out,
            "gate_state": g_state,
            "gate_traj": g_traj,
        }