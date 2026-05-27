# -*- coding: utf-8 -*-
import math
from typing import Dict

import torch
import torch.nn as nn

from .ST_encoder import STRecurrentImpedanceEncoder
from .backbone import Backbone
from .token_adapter import STTokenAdapter, DualTokenCompressor
from .mmoe_decoder import DualGateMMoEDecoder


class STLLMMMoEModel(nn.Module):
    """
    v3: full-node residual traj refinement

    1) encoder -> base predictions + hx_enc / he_enc
    2) adapter -> full-resolution tokens
    3) compressor + LLM + DGMoE -> global reasoning
    4) state branch: residual refinement on top of pretrained state decoder output
    5) traj branch: full-node residual refinement
       y_traj = base_traj_logits + full-node residual logits
    """

    def __init__(
        self,
        encoder: STRecurrentImpedanceEncoder,
        d_hidden: int,
        llm_path: str = "./gpt2",
        num_latents_x: int = 32,
        num_latents_e: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_nodes: int = 512,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.backbone = Backbone(model_path=llm_path)
        d_llm = self.backbone.d_model
        self.d_llm = d_llm

        self.adapter = STTokenAdapter(d_in=d_hidden, d_llm=d_llm, dropout=dropout)
        self.compressor = DualTokenCompressor(
            d_model=d_llm,
            num_latents_x=num_latents_x,
            num_latents_e=num_latents_e,
            num_heads=num_heads,
        )

        # task placeholders
        self.state_ph = nn.Parameter(torch.randn(1, 1, d_llm) * 0.02)
        self.hop_ph = nn.Parameter(torch.randn(1, 1, d_llm) * 0.02)

        # global DGMoE
        self.decoder = DualGateMMoEDecoder(
            d_model=d_llm,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.freeze_encoder = freeze_encoder
        self.num_nodes = num_nodes
        self.output_window = encoder.output_window

        # ======================================================
        # 1) State branch
        # ======================================================
        self.state_node_emb = nn.Embedding(self.num_nodes, d_llm)
        self.state_time_emb = nn.Embedding(self.output_window, d_llm)

        nn.init.normal_(self.state_node_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.state_time_emb.weight, mean=0.0, std=0.02)

        self.state_global_proj = nn.Sequential(
            nn.Linear(d_llm * 2, d_llm),
            nn.GELU(),
            nn.LayerNorm(d_llm),
        )

        self.state_film = nn.Linear(d_llm, d_llm * 2)

        self.state_local_proj = nn.Sequential(
            nn.Linear(d_llm * 3, d_llm),
            nn.GELU(),
            nn.LayerNorm(d_llm),
        )

        self.state_res_mlp = nn.Sequential(
            nn.LayerNorm(d_llm),
            nn.Linear(d_llm, d_llm * 2),
            nn.GELU(),
            nn.Linear(d_llm * 2, d_llm),
        )

        self.state_et_proj = nn.Sequential(
            nn.Linear(2, d_llm),
            nn.GELU(),
            nn.LayerNorm(d_llm),
        )

        self.state_cross_fuse = nn.Sequential(
            nn.Linear(d_llm * 2, d_llm),
            nn.GELU(),
            nn.LayerNorm(d_llm),
        )

        self.state_traj_gate = nn.Sequential(
            nn.Linear(d_llm * 2, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm),
            nn.Sigmoid(),
        )

        self.state_base_norm = nn.LayerNorm(d_llm)

        self.state_time_fuse = nn.Sequential(
            nn.Linear(d_llm * 2, d_llm),
            nn.GELU(),
            nn.LayerNorm(d_llm),
        )

        self.state_out = nn.Linear(d_llm, 1)

        # 让 state 头一开始更像“残差修正”
        nn.init.zeros_(self.state_out.weight)
        nn.init.zeros_(self.state_out.bias)

        self.state_delta_scale = nn.Parameter(torch.tensor(0.05))

        # ======================================================
        # 2) Traj branch: full-node residual refinement
        # ======================================================
        self.traj_pad_token = nn.Parameter(torch.randn(1, 1, d_llm) * 0.02)

        self.traj_global_proj = nn.Sequential(
            nn.Linear(d_llm * 2, d_llm),
            nn.GELU(),
            nn.LayerNorm(d_llm),
        )

        self.traj_query_proj = nn.Sequential(
            nn.Linear(d_llm * 4, d_llm),
            nn.GELU(),
            nn.LayerNorm(d_llm),
        )

        self.traj_node_fuse = nn.Sequential(
            nn.Linear(d_llm * 2, d_llm),
            nn.GELU(),
            nn.LayerNorm(d_llm),
        )

        self.traj_node_id_emb = nn.Embedding(self.num_nodes, d_llm)
        nn.init.normal_(self.traj_node_id_emb.weight, mean=0.0, std=0.02)

        self.traj_base_scale = nn.Parameter(torch.tensor(1.00))
        self.traj_delta_scale = nn.Parameter(torch.tensor(0.05))

    def _split_hidden(self, H, Kx, Ke):
        """
        H = [Hx_tokens, He_tokens, <STATE_PH>, <HOP_PH>]
        """
        Hx = H[:, :Kx, :]
        He = H[:, Kx:Kx + Ke, :]
        h_state_ph = H[:, Kx + Ke, :]
        h_traj_ph = H[:, Kx + Ke + 1, :]
        return Hx, He, h_state_ph, h_traj_ph

    def _gather_nodes(self, node_feat: torch.Tensor, node_idx: torch.Tensor) -> torch.Tensor:
        """
        node_feat: [B, N, D]
        node_idx : [B, M]
        return   : [B, M, D]
        """
        B, N, D = node_feat.shape
        safe_idx = node_idx.clamp(min=0, max=N - 1)
        batch_idx = torch.arange(B, device=node_feat.device).view(B, 1).expand_as(safe_idx)
        return node_feat[batch_idx, safe_idx]

    def _extract_last_road(self, batch: Dict[str, torch.Tensor]):
        """
        从历史最后一个时间步轨迹里，取每个 agent 的最后有效 road_id
        """
        last_traj = batch["Trajectory"][:, -1]              # [B,M,L,3]
        last_mask = batch["Trajectory_token_mask"][:, -1]   # [B,M,L]

        road_seq = last_traj[..., 0].long()                 # [B,M,L]
        lengths = last_mask.long().sum(dim=-1)              # [B,M]
        has_valid = lengths > 0
        last_idx = lengths.clamp(min=1) - 1                 # [B,M]

        last_road = road_seq.gather(
            dim=-1,
            index=last_idx.unsqueeze(-1)
        ).squeeze(-1)                                       # [B,M]

        # invalid agent -> safe zero idx
        last_road = torch.where(has_valid, last_road, torch.zeros_like(last_road))
        last_road = last_road.clamp(min=0, max=self.num_nodes - 1)
        return last_road, has_valid

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # ======================================================
        # 1) encoder -> base predictions + hidden states
        # ======================================================
        batch_enc = dict(batch)

        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                base_state_pred, _, aux = self.encoder(
                    batch_enc,
                    encode_only=False,
                    return_aux=True,
                    tf_state=False,
                    tf_traj=False,
                )
        else:
            base_state_pred, _, aux = self.encoder(
                batch_enc,
                encode_only=False,
                return_aux=True,
                tf_state=False,
                tf_traj=False,
            )

        hx_enc = aux["hx_enc"]                      # [B,N,Dh]
        he_enc = aux["he_enc"]                      # [B,N,Dh]
        base_traj_logits_seq = aux["traj_logits_seq"]   # [B,Tf,M,N]

        # ======================================================
        # 2) full-resolution tokens
        # ======================================================
        hx_tok_full, he_tok_full = self.adapter(hx_enc, he_enc)   # [B,N,D], [B,N,D]

        # ======================================================
        # 3) compress -> LLM -> DGMoE
        # ======================================================
        z_x, z_e = self.compressor(hx_tok_full, he_tok_full)      # [B,Kx,D], [B,Ke,D]
        B = z_x.size(0)
        device = z_x.device

        state_ph = self.state_ph.expand(B, -1, -1)                # [B,1,D]
        hop_ph = self.hop_ph.expand(B, -1, -1)                    # [B,1,D]
        x = torch.cat([z_x, z_e, state_ph, hop_ph], dim=1)

        H = self.backbone(x)["hidden_states"]                     # [B,Kx+Ke+2,D]
        Hx, He, h_state_ph, h_traj_ph = self._split_hidden(H, z_x.size(1), z_e.size(1))

        dec_out = self.decoder(Hx, He, h_state_ph, h_traj_ph)
        state_feat = dec_out["state_feat"]                        # [B,Kx,D]
        traj_feat = dec_out["traj_feat"]                          # [B,Ke,D]

        # ======================================================
        # 4) State branch: residual refinement on base_state_pred
        # ======================================================
        g_state = self.state_global_proj(
            torch.cat([state_feat.mean(dim=1), h_state_ph], dim=-1)
        )  # [B,D]

        gamma, beta = self.state_film(g_state).chunk(2, dim=-1)
        gamma = 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)

        local_state = hx_tok_full * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)   # [B,N,D]
        traj_local = self.state_cross_fuse(torch.cat([hx_tok_full, he_tok_full], dim=-1))  # [B,N,D]
        fuse_gate = self.state_traj_gate(torch.cat([local_state, traj_local], dim=-1))      # [B,N,D]

        node_ids = torch.arange(self.num_nodes, device=device)
        node_emb = self.state_node_emb(node_ids).unsqueeze(0).expand(B, -1, -1)  # [B,N,D]

        base_state_feat = self.state_base_norm(
            local_state + 0.3 * fuse_gate * traj_local + node_emb
        )

        delta_scale = torch.clamp(self.state_delta_scale, min=0.0, max=0.5)

        future_states = []
        for t in range(self.output_window):
            t_ids = torch.full((B, self.num_nodes), t, device=device, dtype=torch.long)
            t_emb = self.state_time_emb(t_ids)   # [B,N,D]

            et_future = batch["Et"][:, self.encoder.input_window + t, :]   # [B,2]
            et_emb = self.state_et_proj(et_future).unsqueeze(1).expand(-1, self.num_nodes, -1)  # [B,N,D]

            g_expand = g_state.unsqueeze(1).expand(-1, self.num_nodes, -1)
            time_cond = self.state_time_fuse(torch.cat([t_emb, et_emb], dim=-1))   # [B,N,D]

            h_t = self.state_local_proj(
                torch.cat([base_state_feat, time_cond, g_expand], dim=-1)
            )   # [B,N,D]

            h_t = h_t + self.state_res_mlp(h_t)
            delta_t = torch.tanh(self.state_out(h_t)) * delta_scale
            s_t = base_state_pred[:, t] + delta_t
            future_states.append(s_t)

        y_state = torch.stack(future_states, dim=1)  # [B,Tf,N,1]

        # ======================================================
        # 5) Traj branch: full-node residual refinement
        # ======================================================
        g_traj = self.traj_global_proj(
            torch.cat([traj_feat.mean(dim=1), h_traj_ph], dim=-1)
        )  # [B,D]

        last_road, has_valid = self._extract_last_road(batch)     # [B,M], [B,M]
        last_hx = self._gather_nodes(hx_tok_full, last_road)      # [B,M,D]
        last_he = self._gather_nodes(he_tok_full, last_road)      # [B,M,D]

        pad_tok = self.traj_pad_token.expand(B, last_hx.size(1), -1)
        last_hx = torch.where(has_valid.unsqueeze(-1), last_hx, pad_tok)
        last_he = torch.where(has_valid.unsqueeze(-1), last_he, pad_tok)

        g_expand = g_traj.unsqueeze(1).expand(-1, last_hx.size(1), -1)   # [B,M,D]

        # agent query: 直接使用 LLM / MoE 的全局轨迹语义
        traj_q = self.traj_query_proj(
            torch.cat([last_hx, last_he, g_expand, last_hx * last_he], dim=-1)
        )  # [B,M,D]

        # full-node key: 对所有节点生成可打分表示
        node_feat = self.traj_node_fuse(torch.cat([hx_tok_full, he_tok_full], dim=-1))  # [B,N,D]

        node_ids = torch.arange(self.num_nodes, device=device)
        node_id_emb = self.traj_node_id_emb(node_ids).unsqueeze(0).expand(B, -1, -1)     # [B,N,D]
        node_feat = node_feat + node_id_emb

        # full-node residual logits
        traj_residual = torch.einsum("bmd,bnd->bmn", traj_q, node_feat) / math.sqrt(self.d_llm)  # [B,M,N]

        # base pretrained traj logits
        base_traj_logits = base_traj_logits_seq[:, 0]  # [B,M,N]

        traj_base_scale = torch.clamp(self.traj_base_scale, min=0.0, max=2.0)
        traj_delta_scale = torch.clamp(self.traj_delta_scale, min=0.0, max=1.0)

        y_traj = traj_base_scale * base_traj_logits + traj_delta_scale * traj_residual   # [B,M,N]

        invalid_agent = ~has_valid
        if invalid_agent.any():
            y_traj[invalid_agent] = 0.0

        return {
            "y_state": y_state,
            "y_traj": y_traj,
            "base_state_pred": base_state_pred,
            "base_traj_logits": base_traj_logits,
            "traj_residual": traj_residual,
            "gate_state": dec_out["gate_state"],
            "gate_traj": dec_out["gate_traj"],
            "hx_enc": hx_enc,
            "he_enc": he_enc,
        }