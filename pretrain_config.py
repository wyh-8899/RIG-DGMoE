#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于配置文件的预训练脚本 - 读取YAML配置文件
"""
import os
import sys
import argparse
import yaml
from pathlib import Path
import torch

# 设置项目根目录
PROJ_ROOT = Path(__file__).resolve().parents[1]  # .../our_model
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from data_provider.data_loader import build_dataloaders
from model.ST_encoder import STRecurrentImpedanceEncoder

# -------------------------
# DDP helpers
# -------------------------
def ddp_is_on() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ

def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))

def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def is_main_process() -> bool:
    return get_rank() == 0

def ddp_setup(device_type: str = "cuda"):
    if not ddp_is_on():
        return None

    rank = get_rank()
    local_rank = get_local_rank()

    if device_type.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    torch.distributed.init_process_group(backend=backend)
    return backend

def ddp_cleanup():
    if ddp_is_on() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def ddp_barrier():
    if ddp_is_on() and torch.distributed.is_initialized():
        torch.distributed.barrier()

def ddp_all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if ddp_is_on() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    return x

# -------------------------
# 设置随机种子
# -------------------------
def set_seed(seed: int = 42):
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# 设备处理
# -------------------------
def to_device(batch, device):
    import torch
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out
def check_finite(name: str, x: torch.Tensor, extra: str = "", max_check_elems: int = 2_000_000):
    if not torch.is_tensor(x):
        return
    # 超大 tensor：抽样检查，避免 isfinite 生成同尺寸 bool 导致 OOM
    if x.numel() > max_check_elems:
        flat = x.reshape(-1)
        step = max(1, flat.numel() // max_check_elems)
        flat = flat[::step]  # 抽样
    else:
        flat = x

    if not torch.isfinite(flat).all():
        x_nanmin = torch.nanmin(x).item() if x.numel() > 0 else float("nan")
        x_nanmax = torch.nanmax(x).item() if x.numel() > 0 else float("nan")
        msg = (f"[NaN/Inf DETECTED] {name} {extra} | "
               f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
               f"nanmin={x_nanmin} nanmax={x_nanmax}")
        if is_main_process():
            print(msg, flush=True)
        valid_cnt = int(den_t.item())
        total_cnt = batch["FutureTraj"].shape[0] * batch["FutureTraj"].shape[2] * batch["FutureTraj"].shape[1]  # B*M*Tf
        if is_main_process():
            print(f"[Denom] next-hop valid={valid_cnt}/{total_cnt}", flush=True)
        if ddp_is_on() and torch.distributed.is_initialized():
            try:
                torch.distributed.abort()
            except Exception:
                pass
        raise RuntimeError(msg)

# -------------------------
# 轨迹NLL损失计算
# -------------------------
def traj_nll_sum_and_denom_sparse(P_nb, nb_idx, FutureTraj, FutureTraj_step_mask, eps=1e-7):
    """
    P_nb:   [B,Tf,N,K]
    nb_idx: [N,K]
    FutureTraj road: [B,Tf,M,L,3]
    mask: [B,Tf,M,L-1]
    """
    fut_road = FutureTraj[..., 0].long()      # [B,Tf,M,L]
    src = fut_road[..., :-1]                  # [B,Tf,M,L-1]
    dst = fut_road[...,  1:]                  # [B,Tf,M,L-1]
    mask = FutureTraj_step_mask.bool()        # [B,Tf,M,L-1]

    B, Tf, N, K = P_nb.shape
    KK = src.numel() // (B * Tf)              # M*(L-1)

    P_flat = P_nb.reshape(B * Tf, N, K)       # [BTf,N,K]
    src_f  = src.reshape(B * Tf, KK)
    dst_f  = dst.reshape(B * Tf, KK)
    m_bool = mask.reshape(B * Tf, KK)
    m_f    = m_bool.float()

    safe_src = torch.where(m_bool, src_f, torch.zeros_like(src_f)).clamp(0, N - 1)
    safe_dst = torch.where(m_bool, dst_f, torch.zeros_like(dst_f)).clamp(0, N - 1)

    bt = torch.arange(B * Tf, device=P_nb.device).unsqueeze(1).expand(-1, KK)

    # gather candidate neighbor ids for each src: [BTf,KK,K]
    cand = nb_idx[safe_src]                   # [BTf,KK,K]

    # gather probs for each src: [BTf,KK,K]
    prob_src = P_flat[bt, safe_src]           # [BTf,KK,K]

    # match dst in candidates
    hit = (cand == safe_dst.unsqueeze(-1))    # [BTf,KK,K]
    prob = (prob_src * hit.float()).sum(dim=-1).clamp_min(eps)  # [BTf,KK]

    nll = -torch.log(prob)
    num = (nll * m_f).sum()
    den = m_f.sum().clamp_min(1.0)
    return num, den

# -------------------------
# 轨迹解码器CE损失计算
# -------------------------
def traj_ce_sum_and_denom(traj_logits_seq, FutureTraj, pad_value: int, compute_topk: bool = False):
    import torch
    
    target = FutureTraj[..., -1, 0].long()  # [B,Tf,M]  <-- last token is next-hop
    B, Tf, M = target.shape
    N = traj_logits_seq.size(-1)

    mask = (target != int(pad_value))         # valid positions

    safe_target = torch.where(mask, target, torch.zeros_like(target)).clamp(0, N - 1)

    loss_flat = torch.nn.functional.cross_entropy(
        traj_logits_seq.reshape(B * Tf * M, N),
        safe_target.reshape(B * Tf * M),
        reduction="none",
    )
    mask_f = mask.reshape(B * Tf * M).float()
    num = (loss_flat * mask_f).sum()
    den = mask_f.sum().clamp_min(1.0)

    # acc1（永远算，便宜）
    pred = traj_logits_seq.argmax(dim=-1)
    correct1 = ((pred == safe_target) & mask).float().sum()

    # acc5/10（默认不算，只有日志时才算）
    correct5 = torch.tensor(0.0, device=traj_logits_seq.device)
    correct10 = torch.tensor(0.0, device=traj_logits_seq.device)

    if compute_topk:
        k5 = min(5, N)
        k10 = min(10, N)

        top5 = traj_logits_seq.topk(k5, dim=-1).indices
        top10 = traj_logits_seq.topk(k10, dim=-1).indices

        correct5 = ((top5 == safe_target.unsqueeze(-1)) & mask.unsqueeze(-1)).any(dim=-1).float().sum()
        correct10 = ((top10 == safe_target.unsqueeze(-1)) & mask.unsqueeze(-1)).any(dim=-1).float().sum()

    return num, den, correct1, correct5, correct10
def sanity_check_next_hop_labels(batch, num_nodes: int):
    import torch
    pad = int(batch["pad_value"].item())
    y = batch["FutureTraj"][..., -1, 0].long()  # next-hop labels
    m = (y != pad)

    if m.any():
        ymin = int(y[m].min().item())
        ymax = int(y[m].max().item())
        assert 0 <= ymin and ymax < num_nodes, f"[LabelRangeError] next-hop label range=({ymin},{ymax}), N={num_nodes}, pad={pad}"
def nexthop_metrics_sum_and_denom(traj_logits_seq, FutureTraj, pad_value: int, k: int = 5):
    """
    traj_logits_seq: [B,Tf,M,N]
    label: FutureTraj[...,0,0] -> [B,Tf,M]
    return: correct1_sum, rr_sum@k, ndcg_sum@k, denom
    """
    import torch
    y = FutureTraj[..., -1, 0].long()          # [B,Tf,M]
    B, Tf, M = y.shape
    N = traj_logits_seq.size(-1)
    k = min(k, N)

    mask = (y != int(pad_value))              # valid
    y_safe = torch.where(mask, y, torch.zeros_like(y)).clamp(0, N-1)

    # ACC
    pred1 = traj_logits_seq.argmax(dim=-1)
    correct1 = ((pred1 == y_safe) & mask).float().sum()

    # Top-k
    topk = traj_logits_seq.topk(k, dim=-1).indices               # [B,Tf,M,k]
    hit = (topk == y_safe.unsqueeze(-1))                         # bool
    hit_any = hit.any(dim=-1) & mask                              # [B,Tf,M]

    # rank in 1..k (meaningful only where hit_any==True)
    rank = hit.float().argmax(dim=-1) + 1                        # [B,Tf,M]

    rr = torch.zeros_like(rank, dtype=torch.float32)
    rr[hit_any] = 1.0 / rank[hit_any].float()                    # MRR uses 1/rank

    ndcg = torch.zeros_like(rank, dtype=torch.float32)
    ndcg[hit_any] = 1.0 / torch.log2(rank[hit_any].float() + 1.0)  # NDCG@k for single relevant item

    rr_sum = rr.sum()
    ndcg_sum = ndcg.sum()
    den = mask.float().sum().clamp_min(1.0)
    return correct1, rr_sum, ndcg_sum, den
# -------------------------
# 评估函数
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device, lambda_state,lambda_traj, lambda_trajdec, tf_state, tf_traj):    
    import math
    import torch
    model.eval()

    # state sums
    s_abs_sum = torch.tensor(0.0, device=device)
    s_sq_sum  = torch.tensor(0.0, device=device)
    s_s1_sum  = torch.tensor(0.0, device=device)
    s_den     = torch.tensor(0.0, device=device)
    s_ape_sum = torch.tensor(0.0, device=device)
    # traj sums
    t_nll_sum = torch.tensor(0.0, device=device)
    t_den_sum = torch.tensor(0.0, device=device)
    # traj decoder sums
    d_ce_sum = torch.tensor(0.0, device=device)
    d_ce_den_sum = torch.tensor(0.0, device=device)

    d_correct1_sum = torch.tensor(0.0, device=device)
    d_rr5_sum      = torch.tensor(0.0, device=device)
    d_ndcg5_sum    = torch.tensor(0.0, device=device)
    d_metric_den_sum = torch.tensor(0.0, device=device)

    for batch in loader:
        batch = to_device(batch, device)
        check_finite("batch[TrafficState]", batch["TrafficState"], extra="(eval)")
        check_finite("batch[FutureState]", batch["FutureState"], extra="(eval)")

        pred_state, _, aux = model(batch, return_aux=True, tf_state=tf_state, tf_traj=tf_traj)

        # autocast 外检查
        check_finite("pred_state", pred_state, extra="(eval)")
        check_finite("aux[P_nb_seq]", aux["P_nb_seq"], extra="(eval)")
        check_finite("aux[traj_logits_seq]", aux["traj_logits_seq"], extra="(eval)")
        gt_state = batch["FutureState"]

        err = (pred_state - gt_state)

        # ✅ mask 与 train 对齐（先用 g!=0）
        valid = (gt_state != 0).float()

        # MAE / RMSE：也只在 valid 上统计
        s_abs_sum += (err.abs() * valid).sum()
        s_sq_sum  += ((err * err) * valid).sum()
        eps = 1e-6
        ape = (err.abs() / gt_state.abs().clamp_min(eps)) * valid
        s_ape_sum += ape.sum()
        # SmoothL1：用 reduction="none" 再 masked sum
        per = torch.nn.functional.smooth_l1_loss(pred_state, gt_state, reduction="none")
        s_s1_sum += (per * valid).sum()

        # 分母：valid 的元素数
        s_den += valid.sum()

        num, den = traj_nll_sum_and_denom_sparse(
            aux["P_nb_seq"], aux["nb_idx"], batch["FutureTraj"], batch["FutureTraj_step_mask"]
        )        
        t_nll_sum += num
        t_den_sum += den
        pad_value = int(batch["pad_value"].item())
        numd, dend, _, _, _ = traj_ce_sum_and_denom(aux["traj_logits_seq"], batch["FutureTraj"], pad_value, compute_topk=False)        
        d_ce_sum += numd
        d_ce_den_sum += dend
        c1, rr5, ndcg5, den5 = nexthop_metrics_sum_and_denom(aux["traj_logits_seq"], batch["FutureTraj"], pad_value, k=5)
        d_correct1_sum += c1
        d_rr5_sum += rr5
        d_ndcg5_sum += ndcg5
        d_metric_den_sum += den5

    # DDP: reduce across GPUs
    s_abs_sum = ddp_all_reduce_sum(s_abs_sum)
    s_sq_sum  = ddp_all_reduce_sum(s_sq_sum)
    s_ape_sum = ddp_all_reduce_sum(s_ape_sum)
    s_s1_sum  = ddp_all_reduce_sum(s_s1_sum)
    s_den     = ddp_all_reduce_sum(s_den)
    t_nll_sum = ddp_all_reduce_sum(t_nll_sum)
    t_den_sum = ddp_all_reduce_sum(t_den_sum)
    d_ce_sum = ddp_all_reduce_sum(d_ce_sum)
    d_ce_den_sum = ddp_all_reduce_sum(d_ce_den_sum)

    d_correct1_sum = ddp_all_reduce_sum(d_correct1_sum)
    d_rr5_sum = ddp_all_reduce_sum(d_rr5_sum)
    d_ndcg5_sum = ddp_all_reduce_sum(d_ndcg5_sum)
    d_metric_den_sum = ddp_all_reduce_sum(d_metric_den_sum)

    if float(s_den.item()) < 1.0:
        return {}

    mae = (s_abs_sum / s_den).item()
    rmse = math.sqrt((s_sq_sum / s_den).item())
    mape = (100.0 * (s_ape_sum / s_den)).item()   # 百分数
    smoothl1 = (s_s1_sum / s_den).item()

    lt = (t_nll_sum / torch.clamp_min(t_den_sum, 1.0)).item()
    ppl = math.exp(min(lt, 50.0))

    ld = (d_ce_sum / torch.clamp_min(d_ce_den_sum, 1.0)).item()

    acc  = (d_correct1_sum / torch.clamp_min(d_metric_den_sum, 1.0)).item()
    mrr5 = (d_rr5_sum      / torch.clamp_min(d_metric_den_sum, 1.0)).item()
    ndcg5= (d_ndcg5_sum    / torch.clamp_min(d_metric_den_sum, 1.0)).item()

    val_loss = lambda_state*smoothl1 + lambda_traj * lt + lambda_trajdec * ld

    return {
        "val/loss": val_loss,
        "val/mae_state": mae,
        "val/rmse_state": rmse,
        "val/mape_state": mape,
        "val/smoothl1_state": smoothl1,
        "val/nll_traj": lt,
        "val/ppl_traj": ppl,
        "val/ce_trajdec": ld,
        "val/acc_trajdec": acc,
        "val/mrr5_trajdec": mrr5,
        "val/ndcg5_trajdec": ndcg5,
    }

# -------------------------
# 训练函数
# -------------------------
def train(config, device, use_wandb=False, resume_path=None):
    import torch
    import time
    import os
    
    # 设置随机种子
    set_seed(config.get("seed", 42))
    
    # DDP设置
    backend = ddp_setup(config.get("device", "cuda"))
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()

    # 设备设置
    device = torch.device(
        f"cuda:{local_rank}" if (config.get("device", "cuda").startswith("cuda") and torch.cuda.is_available() and ddp_is_on())
        else (config.get("device", "cuda") if torch.cuda.is_available() and config.get("device", "cuda").startswith("cuda") else "cpu")
    )

    # 创建保存目录
    save_dir = config.get("save_dir", "./checkpoints_pretrainA")
    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)

    # 数据加载
    if ddp_is_on():
        if is_main_process():
            train_loader, val_loader, test_loader, static = build_dataloaders(
                config, distributed=True, rank=rank, world_size=world_size
            )
        ddp_barrier()
        if not is_main_process():
            train_loader, val_loader, test_loader, static = build_dataloaders(
                config, distributed=True, rank=rank, world_size=world_size
            )
        ddp_barrier()
    else:
        train_loader, val_loader, test_loader, static = build_dataloaders(config, distributed=False, rank=0, world_size=1)

    # 模型初始化
    model = STRecurrentImpedanceEncoder(config=config, static=static).to(device)
    # ---- freeze transition_net when we don't use it (uniform P_nb) ----
    if bool(config.get("use_uniform_transition", False)):
        for p in model.transition_net.parameters():
            p.requires_grad_(False)
        if is_main_process():
            print("[Freeze] transition_net frozen because use_uniform_transition=True", flush=True)
    # DDP包装
    if ddp_is_on():
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=config.get("find_unused_parameters", False)
        )

    # 优化器
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.get("lr", 1e-3),
        weight_decay=float(config.get("weight_decay", 1e-4))
    )
    # 训练状态
    start_epoch = 1
    best = float("inf")
    # best_state = float("inf")   # 按 val/smoothl1_state
    # best_traj  = float("inf")   # 按 val/nll_traj
    # best_dec   = float("inf")   # 按 val/ce_trajdec

    # 恢复检查点
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        m = (model.module if hasattr(model, "module") else model)
        missing, unexpected = m.load_state_dict(ckpt["model"], strict=False)

        if is_main_process():
            print(f"[Resume] missing={len(missing)} unexpected={len(unexpected)}")
            if len(missing) > 0:
                print("  missing keys (first 20):", missing[:20])
            if len(unexpected) > 0:
                print("  unexpected keys (first 20):", unexpected[:20])

        # ✅ 只在参数组完全一致时才加载 optim
        load_optim = bool(config.get("resume_optim", False))
        if load_optim:
            try:
                optim.load_state_dict(ckpt["optim"])
            except Exception as e:
                if is_main_process():
                    print(f"[Resume] optim load failed -> skip. reason: {e}")

        start_epoch = ckpt.get("epoch", 0) + 1
        best = ckpt.get("best", best)
        # best_state = ckpt.get("best_state", best_state)
        # best_traj  = ckpt.get("best_traj", best_traj)
        # best_dec   = ckpt.get("best_dec", best_dec)

        if config.get("reset_best", False):
            best = float("inf")

        if is_main_process():
            print(f"[Resume] {resume_path} -> epoch {start_epoch}, best {best}")

    # 数据验证（仅rank0）
    if is_main_process():
        b0 = next(iter(train_loader))
        print("[Sanity] TrafficState", tuple(b0["TrafficState"].shape),
              "FutureState", tuple(b0["FutureState"].shape),
              "FutureTraj", tuple(b0["FutureTraj"].shape),
              "FutureTraj_step_mask", tuple(b0["FutureTraj_step_mask"].shape))
        print(f"[DDP] backend={backend} world_size={world_size}")
        sanity_check_next_hop_labels(b0, static["num_nodes"])
            # ---- ShiftCheck: verify FutureTraj is shifted window of Trajectory (t=Th-1) ----
        pad = int(b0["pad_value"].item())
        hist = b0["Trajectory"][:, -1, :, :, 0].long()      # [B,M,L]
        fut0 = b0["FutureTraj"][:, 0, :, :, 0].long()       # [B,M,L]

        ok = ((fut0[:, :, :-1] == hist[:, :, 1:]) |
            (fut0[:, :, :-1] == pad) |
            (hist[:, :, 1:] == pad))
        print("[ShiftCheck] ratio=", ok.float().mean().item(), flush=True)

        y = b0["FutureTraj"][:, 0, :, -1, 0].long()
        yv = y[y != pad]
        if yv.numel() > 0:
            print("[LabelCheck] y_minmax(valid)=", int(yv.min().item()), int(yv.max().item()), flush=True)

    # 训练参数
    lambda_traj = config.get("lambda_traj", 1.0)
    lambda_trajdec = config.get("lambda_trajdec", 1.0)
    tf_state = config.get("tf_state", False)
    tf_traj = config.get("tf_traj", False)
    grad_clip = config.get("grad_clip", 1.0)
    epochs = config.get("epochs", 20)
    log_every = config.get("log_every", 50)

    global_step = 0
    import copy

    overfit_one_batch = bool(config.get("overfit_one_batch", False))
    overfit_steps = int(config.get("overfit_steps", 1000))

    if overfit_one_batch:
        epochs = 1  # ✅避免 epochs 倍 overfit

    fixed_batch = None
    if overfit_one_batch:
        fixed_batch = copy.deepcopy(next(iter(train_loader)))
        if is_main_process():
            print(f"[Overfit] Enabled: repeating ONE batch for {overfit_steps} steps (single-epoch)")
            pad = int(fixed_batch["pad_value"].item())
            road = fixed_batch["Trajectory"][:, -1, :, :, 0].long()    # [B,M,L]
            y = fixed_batch["FutureTraj"][:, 0, :, -1, 0].long()        # [B,M]

            mask = (road != pad)
            hist_valid = mask.any(dim=-1)                               # [B,M] 只要轨迹里出现过非pad就算有历史
            label_valid = (y != pad)

            bad = (~hist_valid) & label_valid

            print("[SanityMask] label_valid_cnt=", int(label_valid.sum()), "/", label_valid.numel())
            print("[SanityMask] hist_valid_cnt =", int(hist_valid.sum()), "/", hist_valid.numel())
            print("[SanityMask] BAD(mismatch)  =", int(bad.sum()))
    if config.get("detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)
    debug_log = bool(config.get("debug_log", False))
    # 训练循环
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # DDP sampler设置epoch
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()

        run_loss = run_ls = run_lt = run_ld = 0.0
        nb = 0
        if overfit_one_batch:
            # 只跑一个“伪 epoch”：重复同一个 batch
            it_range = range(1, overfit_steps + 1)
            for it in it_range:
                batch = to_device(fixed_batch, device)
                check_finite("batch[TrafficState]", batch["TrafficState"], extra=f"(epoch={epoch}, it={it})")
                check_finite("batch[FutureState]", batch["FutureState"], extra=f"(epoch={epoch}, it={it})")
                
                pred_state, _, aux = model(batch, return_aux=True, tf_state=tf_state, tf_traj=tf_traj)

                gt_state = batch["FutureState"]

                # state loss (masked)
                g = gt_state
                p = pred_state
                valid = (g != 0).float()
                per = torch.nn.functional.smooth_l1_loss(p, g, reduction="none")
                ls = (per * valid).sum() / (valid.sum() + 1e-6)

                # traj nll
                num, den = traj_nll_sum_and_denom_sparse(
                    aux["P_nb_seq"], aux["nb_idx"], batch["FutureTraj"], batch["FutureTraj_step_mask"]
                )
                lt = (num / den)

                pad_value = int(batch["pad_value"].item())
                numd, dend, _, _, _ = traj_ce_sum_and_denom(aux["traj_logits_seq"], batch["FutureTraj"], pad_value, compute_topk=False)                
                ld = (numd / dend)

                lambda_state = float(config.get("lambda_state", 1.0))
                loss = lambda_state * ls + lambda_traj * lt + lambda_trajdec * ld

                check_finite("pred_state", pred_state, extra=f"(epoch={epoch}, it={it})")
                check_finite("aux[P_nb_seq]", aux["P_nb_seq"], extra=f"(epoch={epoch}, it={it})")
                check_finite("aux[traj_logits_seq]", aux["traj_logits_seq"], extra=f"(epoch={epoch}, it={it})")

                # 反向传播
                optim.zero_grad(set_to_none=True)
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))

                # （可选但推荐）NaN/Inf grad 检查
                m = model.module if hasattr(model, "module") else model
                for name, param in m.named_parameters():
                    if param.grad is not None and (not torch.isfinite(param.grad).all()):
                        raise RuntimeError(f"[NaN/Inf GRAD] {name} (epoch={epoch}, it={it})")

                optim.step()

                run_loss += float(loss.item())
                run_ls += float(ls.item())
                run_lt += float(lt.item())
                run_ld += float(ld.item())
                nb += 1
                global_step += 1

                # 日志记录
                if (it % log_every == 0):
                    with torch.no_grad():
                        pad_value = int(batch["pad_value"].item())
                        c1, rr5, ndcg5, den5 = nexthop_metrics_sum_and_denom(aux["traj_logits_seq"], batch["FutureTraj"], pad_value, k=5)

                    c1_t    = ddp_all_reduce_sum(c1.clone())
                    rr5_t   = ddp_all_reduce_sum(rr5.clone())
                    ndcg5_t = ddp_all_reduce_sum(ndcg5.clone())
                    den_t   = ddp_all_reduce_sum(den5.clone())

                    acc  = (c1_t / torch.clamp_min(den_t, 1.0)).item()
                    mrr5 = (rr5_t / torch.clamp_min(den_t, 1.0)).item()
                    ndcg5 = (ndcg5_t / torch.clamp_min(den_t, 1.0)).item()

                    if is_main_process():
                        msg = (f"Epoch {epoch:03d} | iter {it:04d} | "
                            f"loss={run_loss/nb:.6f} | ls={run_ls/nb:.6f} | lt={run_lt/nb:.6f} | ld={run_ld/nb:.6f} | "
                            f"ACC={acc:.4f} MRR5={mrr5:.4f} NDCG5={ndcg5:.4f} | "
                            f"gn={gn:.3f}")
                        print(msg,flush=True)

                    # --- reset window counters (for acc@k per log_every steps) ---
                    if debug_log:
                        with torch.no_grad():
                            x = batch["TrafficState"]
                            g = gt_state
                            p = pred_state
                            valid = (g != 0).float() 
                            print("valid_ratio=", valid.mean().item())

                            mae_all = (p - g).abs().mean()
                            mae_valid = ((p - g).abs() * valid).sum() / (valid.sum() + 1e-6)
                            print("mae_all=", mae_all.item(), "mae_valid=", mae_valid.item())
                            # 1) 最关键：shape/广播检查（必须打印）
                            print(f"    shapes: TrafficState={tuple(x.shape)}  gt={tuple(g.shape)}  pred={tuple(p.shape)}")

                            # 2) 强制防误训练：shape 不一致就直接报错（避免 silent broadcast）
                            assert p.shape == g.shape, f"Shape mismatch: pred {p.shape} vs gt {g.shape}"

                            # 3) 基本统计
                            print(
                                f"    TrafficState: mean={x.mean().item():.4f} std={x.std().item():.4f} "
                                f"min={x.min().item():.4f} max={x.max().item():.4f}"
                            )
                            print(
                                f"    gt_state:     mean={g.mean().item():.4f} std={g.std().item():.4f} "
                                f"min={g.min().item():.4f} max={g.max().item():.4f}"
                            )
                            print(
                                f"    pred_state:   mean={p.mean().item():.4f} std={p.std().item():.4f} "
                                f"min={p.min().item():.4f} max={p.max().item():.4f}"
                            )

                            e = p - g
                            print(
                                f"    err:          mean={e.mean().item():.4f} std={e.std().item():.4f} "
                                f"maxabs={e.abs().max().item():.4f}"
                            )

                            # 4) 全局 unique（你已经有了，保留）
                            u_all = torch.unique(p.detach().float().reshape(-1))
                            print(f"    pred_state unique(all)={u_all.numel()} first5={u_all[:5].tolist()}")

                            # 5) 进一步定位：到底是“整张图常数”还是“某一维塌缩”
                            # 下面是通用写法：随机抽一个样本，看各维度是否有变化
                            ps = p[0]  # 取 batch 的第0个样本
                            print(f"    pred_state[0] stats: mean={ps.mean().item():.4f} std={ps.std().item():.4f}")

                            # 如果 pred 至少有2维，我们看“行/列”是否有变化（尽量不依赖具体语义）
                            if ps.ndim >= 2:
                                # 对最后一维求 mean，看看倒数第二维（常见是 node 或 feature）有没有变化
                                v = ps.mean(dim=-1)  # shape: ps.shape[:-1]
                                u_v = torch.unique(v.detach().float().reshape(-1))
                                print(f"    unique(mean over last dim)={u_v.numel()} first5={u_v[:5].tolist()}")

                            if ps.ndim >= 3:
                                # 再对倒数第二维求 mean，看最后一维（常见 feature）有没有变化
                                w = ps.mean(dim=-2)  # shape: ps.shape[:-2] + (last,)
                                u_w = torch.unique(w.detach().float().reshape(-1))
                                print(f"    unique(mean over -2 dim)={u_w.numel()} first5={u_w[:5].tolist()}")

                    # wandb记录
                    if use_wandb:
                        import wandb
                        logd = {
                            "train/loss": run_loss/nb,
                            "train/loss_state": run_ls/nb,
                            "train/nll_traj": run_lt/nb,
                            "train/ce_trajdec": run_ld/nb,
                            "train/grad_norm": gn,
                            "train/lr": optim.param_groups[0]["lr"],
                            "epoch": epoch,
                            "step": global_step,
                        }

                        wandb.log(logd, step=global_step)
        else:
            for it, batch in enumerate(train_loader, start=1):
                batch = to_device(batch, device)

                check_finite("batch[TrafficState]", batch["TrafficState"], extra=f"(epoch={epoch}, it={it})")
                check_finite("batch[FutureState]", batch["FutureState"], extra=f"(epoch={epoch}, it={it})")
                
                pred_state, _, aux = model(batch, return_aux=True, tf_state=tf_state, tf_traj=tf_traj)
                gt_state = batch["FutureState"]

                # state loss
                g = gt_state
                p = pred_state
                valid = (g != 0).float()
                per = torch.nn.functional.smooth_l1_loss(p, g, reduction="none")
                ls = (per * valid).sum() / (valid.sum() + 1e-6)
                # traj nll
                num, den = traj_nll_sum_and_denom_sparse(
                    aux["P_nb_seq"], aux["nb_idx"], batch["FutureTraj"], batch["FutureTraj_step_mask"]
                )
                lt = (num / den)

                pad_value = int(batch["pad_value"].item())
                numd, dend, cor1, _, _ = traj_ce_sum_and_denom(aux["traj_logits_seq"], batch["FutureTraj"], pad_value, compute_topk=False)                
                ld = (numd / dend)
                
                lambda_state = float(config.get("lambda_state", 1.0))
                loss = lambda_state * ls + lambda_traj * lt + lambda_trajdec * ld

                check_finite("pred_state", pred_state, extra=f"(epoch={epoch}, it={it})")
                check_finite("aux[P_nb_seq]", aux["P_nb_seq"], extra=f"(epoch={epoch}, it={it})")
                check_finite("aux[traj_logits_seq]", aux["traj_logits_seq"], extra=f"(epoch={epoch}, it={it})")

                # 反向传播
                optim.zero_grad(set_to_none=True)
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))

                # （可选但推荐）NaN/Inf grad 检查
                m = model.module if hasattr(model, "module") else model
                for name, param in m.named_parameters():
                    if param.grad is not None and (not torch.isfinite(param.grad).all()):
                        raise RuntimeError(f"[NaN/Inf GRAD] {name} (epoch={epoch}, it={it})")

                optim.step()

                run_loss += float(loss.item())
                run_ls += float(ls.item())
                run_lt += float(lt.item())
                run_ld += float(ld.item())
                nb += 1
                global_step += 1

                # 日志记录
                if (it % log_every == 0):
                    with torch.no_grad():
                        pad_value = int(batch["pad_value"].item())
                        c1, rr5, ndcg5, den5 = nexthop_metrics_sum_and_denom(aux["traj_logits_seq"], batch["FutureTraj"], pad_value, k=5)
                        pad = int(batch["pad_value"].item())

                        road_hist = batch["Trajectory"][:, -1, :, :, 0].long()   # [B,M,L]
                        y = batch["FutureTraj"][:, 0, :, -1, 0].long()            # [B,M]

                        mask = (road_hist != pad)
                        hist_valid = mask.any(dim=-1)
                        label_valid = (y != pad)
                        valid = hist_valid & label_valid

                        lengths = mask.long().sum(dim=-1).clamp_min(1)
                        last_idx = (lengths - 1).unsqueeze(-1)
                        last_road = road_hist.gather(-1, last_idx).squeeze(-1)
                        last_road = torch.where(hist_valid, last_road, torch.zeros_like(last_road))

                        m = (model.module if hasattr(model, "module") else model)
                        N = m.num_nodes
                        last_road = last_road.clamp(0, N - 1)
                        y_safe = torch.where(label_valid, y, torch.zeros_like(y)).clamp(0, N - 1)

                        nb_idx = m.transition_net.nb_idx      # [N,K]
                        nb_mask = m.transition_net.nb_mask    # [N,K]  ✅关键

                        hit = (
                            ((nb_idx[last_road] == y_safe.unsqueeze(-1)) & nb_mask[last_road])
                            .any(dim=-1)
                        ) & valid
                        hit_ratio = hit.float().sum() / valid.float().sum().clamp_min(1.0)

                        copy_last = ((last_road == y_safe) & valid).float().sum() / valid.float().sum().clamp_min(1.0)

                        y0_ratio = ((y_safe == 0) & valid).float().sum() / valid.float().sum().clamp_min(1.0)

                        # if is_main_process():
                        #     print(f"[SanityNeighbor] valid={int(valid.sum())}/{valid.numel()}  hit_ratio={hit_ratio.item():.3f}  y0_ratio={y0_ratio.item():.3f}  copy_last={copy_last.item():.3f}", flush=True)
                    c1_t    = ddp_all_reduce_sum(c1.clone())
                    rr5_t   = ddp_all_reduce_sum(rr5.clone())
                    ndcg5_t = ddp_all_reduce_sum(ndcg5.clone())
                    den_t   = ddp_all_reduce_sum(den5.clone())

                    acc  = (c1_t / torch.clamp_min(den_t, 1.0)).item()
                    mrr5 = (rr5_t / torch.clamp_min(den_t, 1.0)).item()
                    ndcg5 = (ndcg5_t / torch.clamp_min(den_t, 1.0)).item()

                    if is_main_process():
                        msg = (f"Epoch {epoch:03d} | iter {it:04d} | "
                            f"loss={run_loss/nb:.6f} | ls={run_ls/nb:.6f} | lt={run_lt/nb:.6f} | ld={run_ld/nb:.6f} | "
                            f"ACC={acc:.4f} MRR5={mrr5:.4f} NDCG5={ndcg5:.4f} | "
                            f"gn={gn:.3f}")
                        print(msg,flush=True)

                    # --- reset window counters (for acc@k per log_every steps) ---
                    if debug_log:
                        with torch.no_grad():
                            g = gt_state
                            p = pred_state
                            e = p - g
                            valid = (g != 0).float()
                            print("valid_ratio=", valid.mean().item())
                            mae_all = (p - g).abs().mean()
                            mae_valid = ((p - g).abs() * valid).sum() / (valid.sum() + 1e-6)
                            print("mae_all=", mae_all.item(), "mae_valid=", mae_valid.item())
                            print(
                                f"    gt_state:   mean={g.mean().item():.4f} std={g.std().item():.4f} "
                                f"min={g.min().item():.4f} max={g.max().item():.4f}"
                            )
                            print(
                                f"    pred_state: mean={p.mean().item():.4f} std={p.std().item():.4f} "
                                f"min={p.min().item():.4f} max={p.max().item():.4f}"
                            )
                            print(
                                f"    err:        mean={e.mean().item():.4f} std={e.std().item():.4f} "
                                f"maxabs={e.abs().max().item():.4f}"
                            )

                    # wandb记录
                    if use_wandb:
                        import wandb
                        logd = {
                            "train/loss": run_loss/nb,
                            "train/loss_state": run_ls/nb,
                            "train/nll_traj": run_lt/nb,
                            "train/ce_trajdec": run_ld/nb,
                            "train/grad_norm": gn,
                            "train/lr": optim.param_groups[0]["lr"],
                            "epoch": epoch,
                            "step": global_step,
                        }

                        wandb.log(logd, step=global_step)

        # 验证（同一个 epoch 末尾做两种推理条件）
        val_tf = evaluate(model, val_loader, device, lambda_state, lambda_traj, lambda_trajdec, True,  tf_traj)
        val_no = evaluate(model, val_loader, device, lambda_state, lambda_traj, lambda_trajdec, False, tf_traj)        
        dt = time.time() - t0

        if is_main_process():
            v = val_no
            val_metrics = val_no
            print(
                f"[Val] Epoch {epoch:03d} | "
                f"loss={v['val/loss']:.4f} | "
                f"MAE={v['val/mae_state']:.4f} RMSE={v['val/rmse_state']:.4f} MAPE={v['val/mape_state']:.2f}% | "
                f"ACC={v['val/acc_trajdec']:.4f} MRR5={v['val/mrr5_trajdec']:.4f} NDCG5={v['val/ndcg5_trajdec']:.4f} | "
                f"NLL={v['val/nll_traj']:.4f} CE={v['val/ce_trajdec']:.4f} | "
                f"time={dt:.1f}s"
            )
            
            # wandb记录
            if use_wandb:
                import wandb
                wandb.log({**val_metrics, "epoch": epoch, "step": global_step}, step=global_step)

            # 保存检查点
            os.makedirs(save_dir, exist_ok=True)
            last_path = os.path.join(save_dir, f"{config.get('run_name', 'pretrainA')}_last.pt")
            torch.save({
                "epoch": epoch,
                "best": best,
                # "best_state": best_state,
                # "best_traj": best_traj,
                # "best_dec": best_dec,
                "model": (model.module if hasattr(model, "module") else model).state_dict(),
                "optim": optim.state_dict(),
                "args": config,
            }, last_path)

            # 保存最佳模型
            # ---- multi-best ckpt (推荐：用 NO 推理条件) ----
            if val_metrics:
                # # 1) best by state
                # if val_metrics["val/smoothl1_state"] < best_state:
                #     best_state = val_metrics["val/smoothl1_state"]
                #     best_path = os.path.join(save_dir, f"{config.get('run_name','pretrainA')}_best_state.pt")
                #     torch.save({
                #         "epoch": epoch,
                #         "best_state": best_state,
                #         "model": (model.module if hasattr(model, "module") else model).state_dict(),
                #         "optim": optim.state_dict(),
                #         "args": config,
                #         "val_metrics": val_metrics,
                #     }, best_path)
                #     print(f"[CKPT] best_state saved -> {best_path}")

                # # 2) best by traj nll
                # if val_metrics["val/nll_traj"] < best_traj:
                #     best_traj = val_metrics["val/nll_traj"]
                #     best_path = os.path.join(save_dir, f"{config.get('run_name','pretrainA')}_best_traj_nll.pt")
                #     torch.save({
                #         "epoch": epoch,
                #         "best_traj": best_traj,
                #         "model": (model.module if hasattr(model, "module") else model).state_dict(),
                #         "optim": optim.state_dict(),
                #         "args": config,
                #         "val_metrics": val_metrics,
                #     }, best_path)
                #     print(f"[CKPT] best_traj_nll saved -> {best_path}")

                # # 3) best by trajdec ce
                # if val_metrics["val/ce_trajdec"] < best_dec:
                #     best_dec = val_metrics["val/ce_trajdec"]
                #     best_path = os.path.join(save_dir, f"{config.get('run_name','pretrainA')}_best_trajdec_ce.pt")
                #     torch.save({
                #         "epoch": epoch,
                #         "best_dec": best_dec,
                #         "model": (model.module if hasattr(model, "module") else model).state_dict(),
                #         "optim": optim.state_dict(),
                #         "args": config,
                #         "val_metrics": val_metrics,
                #     }, best_path)
                #     print(f"[CKPT] best_trajdec_ce saved -> {best_path}")

                # # （可选）4) 仍然保留一个 best_total（仅同一组lambda内部比较）
                if val_metrics["val/loss"] < best:
                    best = val_metrics["val/loss"]
                    best_path = os.path.join(save_dir, f"{config.get('run_name','pretrainA')}_best_total.pt")
                    torch.save({
                        "epoch": epoch,
                        "best": best,
                        "model": (model.module if hasattr(model, "module") else model).state_dict(),
                        "optim": optim.state_dict(),
                        "args": config,
                        "val_metrics": val_metrics,
                    }, best_path)
                    print(f"[CKPT] best_total saved -> {best_path}")

    # 清理
    if ddp_is_on():
        ddp_cleanup()

# -------------------------
# 主函数
# -------------------------
def main():
    parser = argparse.ArgumentParser(description='基于配置文件的预训练')
    parser.add_argument('--config', type=str, default='./config/pretrain_schemeA.yaml',
                        help='配置文件路径')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='要使用的GPU ID (例如: "0", "0,1")，如果为空则使用配置文件中的值')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径，覆盖配置文件中的值')
    parser.add_argument('--use_wandb', action='store_true',
                        help='使用Weights & Biases记录实验，覆盖配置文件中的值')
    
    args = parser.parse_args()
    
    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置GPU
    # gpu_ids = args.gpu_ids if args.gpu_ids else config.get('gpu_ids', '0')
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    # print(f"Using GPU(s): {gpu_ids}")
    
    # 1) 只在非 torchrun / 单卡场景才允许设置 CUDA_VISIBLE_DEVICES
    is_torchrun = ("LOCAL_RANK" in os.environ) or ("RANK" in os.environ) or ("WORLD_SIZE" in os.environ)

    if not is_torchrun:
        gpu_ids = args.gpu_ids if args.gpu_ids else config.get('gpu_ids', None)
        if gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            print(f"Using GPU(s) (single-process): {gpu_ids}")
    else:
        # 2) torchrun 场景：让 torchrun 控制可见卡 + 用 LOCAL_RANK 选卡
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[DDP] torchrun detected, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}, "
                f"world_size={os.environ.get('WORLD_SIZE','?')}")
    # 覆盖配置文件中的参数
    if args.resume:
        config['resume'] = args.resume
    if args.use_wandb:
        config['use_wandb'] = True
    
    # wandb初始化
    if config.get('use_wandb', False) and config.get('wandb_mode', 'online') != 'disabled' and is_main_process():
        import wandb
        wandb.init(project=config.get('wandb_project', 'st-encoder'), 
                   name=config.get('run_name', 'pretrainA'), 
                   config=config)

    # 开始训练
    train(config, device=None, use_wandb=config.get('use_wandb', False), resume_path=config.get('resume', None))

    # 完成wandb
    if config.get('use_wandb', False) and is_main_process():
        import wandb
        wandb.finish()

if __name__ == "__main__":
    main()