import torch
import torch.nn as nn
import torch.nn.functional as F

class ImpedanceNet(nn.Module):
    """
    Physics + (hx, he)-conditioned residual impedance encoder.

    Inputs:
      u_r_t:      [B, N]        measured speed
      hx, he:     [B, N, D_h]   temporal hidden states
      X_t:        [B, N, D_s]   (optional) spatial embedding
      E_t:        [B, N, D_t]   (optional) trajectory embedding
      node_feat:  [N, F] or [B,N,F] (optional) static node features

    Static (buffers):
      L_r:        [N] road length (raw)
      u_ff_r:     [N] free-flow speed (e.g., 95 percentile)

    Outputs:
      I_t:        [B, N, D_i]
      I_scalar:   [B, N, 1]  (optional)
    """
    def __init__(
        self,
        L_r: torch.Tensor,            # [N]
        u_ff_r: torch.Tensor,         # [N]
        d_h: int,                     # hx/he dim
        d_i: int = 16,                # impedance feature dim
        d_s: int = 0,                 # X_t dim (0 means not used)
        d_t: int = 0,                 # E_t dim (0 means not used)
        node_f_dim: int = 0,          # node feature dim
        hidden: int = 128,
        dropout: float = 0.1,
        use_dynamic_params: bool = True,  # whether alpha/beta depend on hx/he

    ):
        super().__init__()
        assert L_r.dim() == 1 and u_ff_r.dim() == 1
        assert L_r.shape == u_ff_r.shape

        self.N = L_r.numel()
        self.d_i = d_i
        self.use_dynamic_params = use_dynamic_params

        # register static as buffers (move with model.to(device))
        self.register_buffer("L_r", L_r.float(), persistent=True)       # [N]
        self.register_buffer("u_ff_r", u_ff_r.float(), persistent=True) # [N]

        # global learnable baseline (stable, interpretable)
        self.alpha_base = nn.Parameter(torch.tensor(0.15))
        self.beta_base  = nn.Parameter(torch.tensor(4.0))

        # ---- context encoder (for residual & gating) ----
        ctx_dim = 2 * d_h + (d_s if d_s > 0 else 0) + (d_t if d_t > 0 else 0) + (node_f_dim if node_f_dim > 0 else 0) + 2
        # +2 for [speed_ratio, congestion_level] (explicit physics signals)

        self.ctx_norm = nn.LayerNorm(ctx_dim)

        self.mlp = nn.Sequential(
            nn.Linear(ctx_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # residual head: produce a bounded correction in log-space
        self.res_head = nn.Linear(hidden, d_i)

        # gate between physics and residual intensity
        self.gate_head = nn.Linear(hidden, d_i)

        # optional: dynamic alpha/beta from hx/he (kept small for stability)
        if self.use_dynamic_params:
            self.ab_head = nn.Linear(hidden, 2)  # delta_alpha, delta_beta

        # final projection (feature shaping)
        self.out_proj = nn.Linear(d_i, d_i)


    def forward(
        self,
        u_r_t: torch.Tensor,                 # [B,N]
        hx: torch.Tensor,                    # [B,N,D_h]
        he: torch.Tensor,                    # [B,N,D_h]
        X_t: torch.Tensor = None,            # [B,N,D_s]
        E_t: torch.Tensor = None,            # [B,N,D_t]
        node_feat: torch.Tensor = None,      # [N,F] or [B,N,F]
        return_scalar: bool = False,
    ):
        B, N = u_r_t.shape
        assert N == self.N, f"u_r_t N={N} != static N={self.N}"

        device = u_r_t.device
        L_r = self.L_r.to(device).unsqueeze(0).expand(B, N)          # [B,N]
        u_ff = self.u_ff_r.to(device).unsqueeze(0).expand(B, N)      # [B,N]

        # ---- physics backbone (BPR-like) ----
        # speed ratio (>=1 means congested)
        u_ff = self.u_ff_r.to(device).unsqueeze(0).expand(B, N)
        u = torch.maximum(u_r_t, 0.05 * u_ff).clamp_min(1e-3)   # 至少是自由流的 5%
        ratio = (u_ff / u).clamp_min(1e-6).clamp_max(50.0)   # 比如最多 50 倍
        cong  = (ratio - 1.0).clamp_min(0.0).clamp_max(50.0)

        # baseline alpha/beta (positive, stable)
        alpha = F.softplus(self.alpha_base) + 1e-6
        beta  = F.softplus(self.beta_base)  + 1e-6

        # ---- build context for residual ----
        ctx_parts = [hx, he]

        if X_t is not None:
            ctx_parts.append(X_t)
        if E_t is not None:
            ctx_parts.append(E_t)

        if node_feat is not None:
            if node_feat.dim() == 2:  # [N,F] -> [B,N,F]
                node_feat = node_feat.unsqueeze(0).expand(B, N, -1)
            ctx_parts.append(node_feat)

        # explicit physics signals help residual learn faster
        phys_sig = torch.stack([ratio, cong], dim=-1)                # [B,N,2]
        ctx_parts.append(phys_sig)

        ctx = torch.cat(ctx_parts, dim=-1)                           # [B,N,ctx_dim]
        ctx = self.ctx_norm(ctx)
        h = self.mlp(ctx)                                            # [B,N,hidden]

        # ---- optional dynamic alpha/beta (small modulation) ----
        if self.use_dynamic_params:
            dab = self.ab_head(h)                                    # [B,N,2]
            # keep modulation mild to preserve interpretability/stability
            alpha_dyn = alpha * (1.0 + 0.2 * torch.tanh(dab[..., 0]))
            beta_dyn  = beta  * (1.0 + 0.2 * torch.tanh(dab[..., 1]))
        else:
            alpha_dyn, beta_dyn = alpha, beta

        # free-flow time
        t_ff = (L_r / u_ff.clamp_min(1e-3)).clamp_min(0.0)           # [B,N]

        # physical impedance scalar
        I_phys = t_ff * (1.0 + alpha_dyn * torch.pow(cong + 1e-6, beta_dyn))  # [B,N]
        I_phys = I_phys.clamp_min(1e-6)

        # ---- residual in log-space (multiplicative correction) ----
        # residual is bounded: exp(g * tanh(res)) so correction factor is in (exp(-g), exp(g))
        res = torch.tanh(self.res_head(h))                           # [B,N,d_i] in [-1,1]
        gate = torch.sigmoid(self.gate_head(h))                      # [B,N,d_i] in [0,1]

        # expand scalar physics to vector features, then apply correction
        I_vec = I_phys.unsqueeze(-1).expand(B, N, self.d_i)          # [B,N,d_i]
        corr = torch.exp(gate * res)                                 # [B,N,d_i] positive
        I_t = I_vec * corr                                           # [B,N,d_i]

        # final shaping + nonneg
        I_t = F.softplus(self.out_proj(I_t))

        if return_scalar:
            return I_t, I_phys.unsqueeze(-1)
        return I_t
