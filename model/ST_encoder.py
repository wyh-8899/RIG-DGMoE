import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .encoder import SpatialEncoder, TrajEncoder
from .impedance import ImpedanceNet
from .decoder import TemporalRNN, TrajDecoder, MultiLayerFlowPropagation
from .transitionprob import TransitionNet
from typing import Dict, Tuple, Optional, Any
import time
class STRecurrentImpedanceEncoder(nn.Module):
    """
    7模块串联：SpatialEncoder / TrajEncoder / ImpedanceNet /
             TemporalRNN(hx,he) / TransitionNet / FlowPropagation / TrajDecoder

    batch keys/shape 必须匹配 data_loader.py:
      TrafficState: [B, Th, N, 1]
      Trajectory:   [B, Th, M, L, 3]
      Et:           [B, Th+Tf, 2]
    输出:
      future_state: [B, Tf, N, 1]
      future_traj:  [B, Tf, M, L, 3]
    """

    def __init__(self, config: Dict, static: Dict):
        super().__init__()

        # ---------- static constants ----------
        self.num_nodes: int = int(static["num_nodes"])
        self.pad_value: int = int(static["pad_value"])
        self.node_fea_dim: int = int(static["node_fea_dim"])

        # register buffers (auto move with model.to(device))
        self.register_buffer("node_features", static["node_features"].float(), persistent=True)  # [N,F]
        self.register_buffer("adj_mx", static["adj_mx"].float(), persistent=True)                # [N,N]

        edge_index = static.get("edge_index", None)
        if edge_index is None or (isinstance(edge_index, torch.Tensor) and edge_index.numel() == 0):
            # 兜底：只有 self-loop，避免 message passing 空边导致异常
            idx = torch.arange(self.num_nodes, dtype=torch.long)
            edge_index = torch.stack([idx, idx], dim=0)  # [2,N]
        else:
            edge_index = edge_index.long()
        self.register_buffer("edge_index", edge_index, persistent=True)                          # [2,E]

        # impedance physical params
        L_r = static["L_r"].float()         # [N]
        u_ff_r = static["u_ff_r"].float()   # [N]

        # ---------- hyperparams ----------
        self.d_model: int = int(config.get("d_model", 128))
        self.d_hidden: int = int(config.get("d_hidden", 128))
        self.d_impedance: int = int(config.get("d_impedance", 16))
        self.use_impedance: bool = bool(config.get("use_impedance", True))

        self.num_heads: int = int(config.get("num_heads", 4))
        self.num_layers: int = int(config.get("num_layers", 2))
        self.dropout: float = float(config.get("dropout", 0.1))

        self.input_window: int = int(config.get("input_window", 6))    # Th
        self.output_window: int = int(config.get("output_window", 1))   # Tf
        self.max_traj_len: int = int(config.get("max_traj_len", 20))

        self.conv_type: str = str(config.get("conv_type", "gcn"))

        # ---------- En: node identity embedding (match pseudocode) ----------
        self.use_en: bool = bool(config.get("use_en", True))
        self.d_e: int = int(config.get("d_e", max(8, self.d_model // 4)))
        if not self.use_en:
            self.d_e = 0

        if self.use_en:
            self.En = nn.Embedding(self.num_nodes, self.d_e)
            self.register_buffer("_node_ids", torch.arange(self.num_nodes), persistent=True)

        # ---------- module 1: SpatialEncoder ----------
        # input = [traffic(1), En(d_e), node_features(F), Et(2)]
        spatial_in_dim = 1 + self.node_fea_dim + 2 + (self.d_e if self.use_en else 0)

        # ⚠️ 你的 SpatialEncoder 写法是：hidden_dims = hidden_features * (num_layers - 1)
        # 所以这里 hidden_features 应该给 [hidden]，不要给重复后的 list
        self.spatial_encoder = SpatialEncoder(
            in_feature=spatial_in_dim,
            hidden_features=[self.d_model // 2],
            out_feature=self.d_model,
            num_layers=self.num_layers,
            conv_type=self.conv_type,
            dropout=self.dropout,
            add_skip=True
        )

        # ---------- module 2: TrajEncoder ----------
        self.traj_encoder = TrajEncoder(
            num_nodes=self.num_nodes,
            pad_value=self.pad_value,
            d_model=self.d_model,
            nhead=self.num_heads,
            num_layers=self.num_layers,
            max_traj_len=self.max_traj_len,
            feat_dim=3,                  # [road_id, minutes, weeks]
            use_base_node_emb=True
        )

        # ---------- module 3: ImpedanceNet ----------
        self.impedance_net = ImpedanceNet(
            L_r=L_r,
            u_ff_r=u_ff_r,
            d_h=self.d_hidden,
            d_i=self.d_impedance,
            d_s=self.d_model,
            d_t=self.d_model,
            node_f_dim=self.node_fea_dim,
            hidden=self.d_model,
            dropout=self.dropout,
            use_dynamic_params=True,
        )

        # ---------- module 4: TemporalRNN (hx/he) ----------
        self.temporal_rnn_hx = TemporalRNN(input_dim=self.d_model + self.d_impedance, hidden_dim=self.d_hidden)
        self.temporal_rnn_he = TemporalRNN(input_dim=self.d_model + self.d_impedance, hidden_dim=self.d_hidden)

        # ---------- module 5: TransitionNet (dense adj) ----------
        adj_mx = static["adj_mx"]  # ✅ [N,N]，build_dataloaders 返回的 static 里就有
        self.transition_net = TransitionNet(
            in_dim=self.d_impedance,
            hidden_dim=self.d_model,
            dropout=self.dropout,
            adj_mx=adj_mx,          
            add_self_loop=True,
        )
        # nb_idx: [N,K], edge_index: [2,E]
        nb_idx = self.transition_net.nb_idx        # buffer
        src = self.edge_index[0]                   # [E]
        dst = self.edge_index[1]                   # [E]
        cand = nb_idx[src]                         # [E,K]
        match = (cand == dst.unsqueeze(1))         # [E,K]
        has = match.any(dim=1)
        edge2k = torch.where(has, match.long().argmax(dim=1), torch.zeros_like(src))
        assert bool(has.all().item()), "edge_index contains an edge not present in nb_idx (adj_mx mismatch)."
        self.register_buffer("edge2k", edge2k, persistent=True)
        # ---------- module 6: FlowPropagation (sparse edge_index) ----------
        self.flow_propagation = MultiLayerFlowPropagation(
            F_s=1,
            D_s=self.d_model,
            D_i=self.d_impedance,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

        # ---------- module 7: TrajDecoder ----------
        self.traj_decoder = TrajDecoder(
            num_nodes=self.num_nodes,
            d_i=self.d_impedance,
            id_emb=self.d_model // 4,
            hidden=self.d_model,
            dropout=self.dropout,
            prior_logit_weight=float(config.get("prior_logit_weight", 0.0)),
            eps=float(config.get("prior_eps", 1e-2)),
            moe_traj_out=bool(config.get("moe_traj_out", False)),
            moe_num_experts=int(config.get("moe_num_experts", 4)),
            moe_gate_dropout=float(config.get("moe_gate_dropout", 0.0)),
            moe_sync_from_base=bool(config.get("moe_sync_from_base", True)),
            moe_top_k=int(config.get("moe_top_k", 2)),
            moe_noisy_gate=bool(config.get("moe_noisy_gate", False)),
            moe_noise_std=float(config.get("moe_noise_std", 1.0)),
        )
        self.use_uniform_transition = bool(config.get("use_uniform_transition", False))
        self.use_state_moe = bool(config.get("moe_state_refiner", False))
        if self.use_state_moe:
            in_dim = 1 + self.d_model + self.d_impedance + 2  # S_next + X_t + I_t + et_future
            hidden = int(config.get("state_moe_hidden", 128))
            E = int(config.get("moe_num_experts", 4))
            self.state_moe_top_k = int(config.get("state_moe_top_k", 1))
            self.state_moe_noisy_gate = bool(config.get("state_moe_noisy_gate", False))
            self.state_moe_noise_std = float(config.get("state_moe_noise_std", 0.5))
            self.state_moe_gate = nn.Linear(in_dim, E)
            self.state_moe_experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                ) for _ in range(E)
            ])
            self.state_moe_scale = float(config.get("state_moe_scale", 0.0)) 

    def _build_spatial_input(self, traffic_x: torch.Tensor, et: torch.Tensor) -> torch.Tensor:
        """
        traffic_x: [B,N,1]
        et:        [B,2]
        return:    [B,N, 1 + d_e + F + 2] (if use_en) else [B,N, 1 + F + 2]
        """
        B, N, _ = traffic_x.shape
        nf = self.node_features.unsqueeze(0).expand(B, -1, -1)               # [B,N,F]
        etn = et.unsqueeze(1).expand(B, N, -1)                               # [B,N,2]

        if self.use_en:
            en = self.En(self._node_ids).unsqueeze(0).expand(B, -1, -1)      # [B,N,d_e]
            return torch.cat([traffic_x, en, nf, etn], dim=-1)
        else:
            return torch.cat([traffic_x, nf, etn], dim=-1)
    def _sanitize_traj(self, traj):
        road = traj[..., 0].long()
        pad = self.pad_value
        mask = (road != pad)
        road_clamped = road.clamp(0, self.num_nodes - 1)
        road_new = torch.where(mask, road_clamped, torch.full_like(road, pad))
        out = traj.clone()
        out[..., 0] = road_new
        return out
    def _compute_impedance(self, traffic_scalar, hx, he, X_t, E_t):
        """Return [B, N, d_impedance]. If use_impedance=False, bypass with zeros."""
        if self.use_impedance:
            return self.impedance_net(
                u_r_t=traffic_scalar,
                hx=hx, he=he,
                X_t=X_t, E_t=E_t,
                node_feat=self.node_features,
                return_scalar=False
            )
        B, N = traffic_scalar.shape
        return torch.zeros(B, N, self.d_impedance, device=traffic_scalar.device, dtype=X_t.dtype)

    def _encode(self, TrafficState, Trajectory, Et, no_grad: bool):
        B, Th, N, _ = TrafficState.shape
        device = TrafficState.device
        edge_index = self.edge_index

        hx = torch.zeros(B, self.num_nodes, self.d_hidden, device=device)
        he = torch.zeros(B, self.num_nodes, self.d_hidden, device=device)

        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            Th = TrafficState.size(1)
            assert Th >= self.input_window, f"Th={Th} < input_window={self.input_window}"
            for t in range(self.input_window):
                traffic_t = TrafficState[:, t]   # [B,N,1]
                traj_t = self._sanitize_traj(Trajectory[:, t])
                et_t      = Et[:, t]             # [B,2]

                spatial_in = self._build_spatial_input(traffic_t, et_t)
                X_t = self.spatial_encoder(spatial_in, edge_index)    # [B,N,d_model]
                E_t = self.traj_encoder(traj_t)                       # [B,N,d_model] (按你实现)
                I_t = self._compute_impedance(
                    traffic_scalar=traffic_t.squeeze(-1),
                    hx=hx, he=he,
                    X_t=X_t, E_t=E_t,
                )  # [B,N,d_impedance] ; zeros when use_impedance=False

                hx = self.temporal_rnn_hx(torch.cat([X_t, I_t], dim=-1), hx)
                he = self.temporal_rnn_he(torch.cat([E_t, I_t], dim=-1), he)

        return hx, he

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_aux: bool = False,
        tf_state: bool = False,   # teacher forcing for state (S_latest)
        tf_traj: bool = False,    # teacher forcing for traj (Tau_latest)
        encode_only: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:

        TrafficState = batch["TrafficState"]   # [B,Th,N,1]
        Trajectory   = batch["Trajectory"]     # [B,Th,M,L,3]
        Et           = batch["Et"]             # [B,Th+Tf,2]

        FutureState = batch.get("FutureState", None)  # [B,Tf,N,1]
        FutureTraj  = batch.get("FutureTraj", None)   # [B,Tf,M,L,3]
        aux = {}
        B, Th, N, Fs = TrafficState.shape
        device = TrafficState.device
        edge_index = self.edge_index
        adj_mx = self.adj_mx

        # ---------- Encoder ----------
        if encode_only:
            no_grad_enc = bool(batch.get("no_grad_enc", True)) if isinstance(batch, dict) else True
            hx_enc, he_enc = self._encode(TrafficState, Trajectory, Et, no_grad=no_grad_enc)
            if return_aux:
                return None, None, {"hx_enc": hx_enc, "he_enc": he_enc}
            return None, None, None

        # 训练/正常 forward：encoder 有梯度
        hx_enc, he_enc = self._encode(TrafficState, Trajectory, Et, no_grad=False)

        # decoder rollout 初始状态
        hx, he = hx_enc, he_enc
        # ---------- Decoder ----------
        S_latest  = TrafficState[:, -1]  # [B,N,1] (t=Th-1)
        Tau_latest = self._sanitize_traj(Trajectory[:, -1])

        future_state = []
        future_traj  = []
        P_nb_seq = []  
        traj_logits_seq = []  # [Tf] each is [B,M,N]

        for t in range(self.output_window):
            et_future = Et[:, self.input_window + t]  # time embedding for step t

            # ---------- teacher forcing input override ----------
            S_in = S_latest
            Tau_in = Tau_latest
            if t > 0:
                if tf_state and (FutureState is not None):
                    S_in = FutureState[:, t-1]
                if tf_traj and (FutureTraj is not None):
                    Tau_in = FutureTraj[:, t-1]
            Tau_in = self._sanitize_traj(Tau_in)
            # re-encode at this step
            spatial_in = self._build_spatial_input(S_in, et_future)
            X_t = self.spatial_encoder(spatial_in, edge_index)
            E_t = self.traj_encoder(Tau_in)

            I_t = self._compute_impedance(
                traffic_scalar=S_in.squeeze(-1),
                hx=hx, he=he,
                X_t=X_t, E_t=E_t,
            )
            if self.use_uniform_transition:
                nb_idx = self.transition_net.nb_idx
                nb_mask = self.transition_net.nb_mask
                P_nb = nb_mask.float().unsqueeze(0).expand(B, -1, -1)
                P_nb = P_nb / P_nb.sum(dim=-1, keepdim=True).clamp_min(1.0)
            else:
                P_nb, nb_idx, nb_mask = self.transition_net(I_t)
            P_nb_seq.append(P_nb)

            S_next = self.flow_propagation(S_in, X_t, I_t, P_nb, edge_index, self.edge2k)
            if self.use_state_moe:
                etn = et_future.unsqueeze(1).expand(B, self.num_nodes, -1)          # [B,N,2]
                feat = torch.cat([S_next, X_t, I_t, etn], dim=-1)                   # [B,N,in_dim]

                gate_logits = self.state_moe_gate(feat)                             # [B,N,E]
                if self.state_moe_noisy_gate and self.training:
                    gate_logits = gate_logits + torch.randn_like(gate_logits) * self.state_moe_noise_std

                k = min(self.state_moe_top_k, gate_logits.size(-1))
                topk_val, topk_idx = torch.topk(gate_logits, k=k, dim=-1)           # [B,N,k], [B,N,k]
                topk_w = torch.softmax(topk_val, dim=-1)                            # [B,N,k]

                # 稀疏计算：只算被选中的 expert
                delta = torch.zeros(B, self.num_nodes, 1, device=device, dtype=feat.dtype)  # [B,N,1]
                E = gate_logits.size(-1)
                feat_flat = feat.reshape(B * self.num_nodes, -1)                    # [BN,in_dim]
                idx_flat = topk_idx.reshape(B * self.num_nodes, k)                  # [BN,k]
                w_flat   = topk_w.reshape(B * self.num_nodes, k)                    # [BN,k]

                for e_id, expert in enumerate(self.state_moe_experts):
                    pos = (idx_flat == e_id).nonzero(as_tuple=False)
                    if pos.numel() == 0:
                        continue
                    tok = pos[:, 0]                                                 # [P]
                    slot= pos[:, 1]                                                 # [P]
                    w_e = w_flat[tok, slot]                                         # [P]
                    d_e = expert(feat_flat[tok])                                    # [P,1]
                    delta.view(-1, 1)[tok] += d_e * w_e.unsqueeze(-1)

                # 缩放/裁剪
                if self.state_moe_scale > 0:
                    delta = torch.tanh(delta) * self.state_moe_scale

                S_next = S_next + delta

                # （可选）记录 balance loss，方便诊断是否塌缩
                top1 = topk_idx[..., 0].reshape(-1)                                 # [BN]
                load = torch.bincount(top1, minlength=E).float() / float(top1.numel())
                # importance 用 topk_w 近似
                imp = torch.zeros(E, device=device)
                imp.scatter_add_(0, idx_flat[:, 0], torch.ones_like(idx_flat[:, 0], dtype=torch.float32, device=device))
                imp = imp / float(idx_flat.size(0))
                state_moe_balance = (E * (imp * load).sum())
                aux["state_moe_balance_loss"] = state_moe_balance
            # traj decoder: return logits for CE loss; optionally teacher-force next road when tf_traj=True
            target_next = None
            use_tf = False
            if tf_traj and (FutureTraj is not None):
                target_next = FutureTraj[:, t, :, -1, 0]  # [B,M] road_id at step t (pad=N)
                use_tf = True
            Tau_next, logits, moe_aux = self.traj_decoder(
                Tau_in,
                P_nb,
                self.transition_net.nb_idx,
                self.transition_net.nb_mask,
                I_t,
                return_logits=True,
                teacher_forcing=use_tf,
                target_next_road=target_next,
                time_next=et_future,
                return_aux=True,
            )

            aux["traj_moe"] = moe_aux
            aux["traj_moe_balance_loss"] = (moe_aux["balance_loss"] if moe_aux is not None else torch.tensor(0.0, device=device))
            traj_logits_seq.append(logits)

            future_state.append(S_next)
            future_traj.append(Tau_next)

            # update temporal states
            hx = self.temporal_rnn_hx(torch.cat([X_t, I_t], dim=-1), hx)
            he = self.temporal_rnn_he(torch.cat([E_t, I_t], dim=-1), he)

            # ---------- teacher forcing rollout state ----------
            if tf_state and (FutureState is not None):
                S_latest = FutureState[:, t]
            else:
                S_latest = S_next

            if tf_traj and (FutureTraj is not None):
                Tau_latest = FutureTraj[:, t]
            else:
                Tau_latest = Tau_next

        future_state = torch.stack(future_state, dim=1)  # [B,Tf,N,1]
        future_traj  = torch.stack(future_traj,  dim=1)  # [B,Tf,M,L,3]
        P_nb_seq = torch.stack(P_nb_seq, dim=1)           # [B,Tf,N,K]
        traj_logits_seq = torch.stack(traj_logits_seq, dim=1)  # [B,Tf,M,N]

        if return_aux:
            aux_out = {
                "P_nb_seq": P_nb_seq,     # [B,Tf,N,K]
                "nb_idx": self.transition_net.nb_idx,
                "nb_mask": self.transition_net.nb_mask,
                "traj_logits_seq": traj_logits_seq,
                "hx_enc": hx_enc,   
                "he_enc": he_enc,   
                "traj_moe": aux.get("traj_moe", None),
                "traj_moe_balance_loss": aux.get("traj_moe_balance_loss", torch.tensor(0.0, device=device)),
            }
            return future_state, future_traj, aux_out
        return future_state, future_traj, None

