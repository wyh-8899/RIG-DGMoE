# -*- coding: utf-8 -*-
import os
import sys
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

# =========================================================
# 0. Path setup
# =========================================================
THIS_FILE = Path(__file__).resolve()
OUR_MODEL_ROOT = THIS_FILE.parent

if str(OUR_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(OUR_MODEL_ROOT))

from data_provider.data_loader import build_dataloaders
from model.ST_encoder import STRecurrentImpedanceEncoder
from model.st_llm_mmoe_model_v2 import STLLMMMoEModel


# =========================================================
# 1. DDP helpers
# =========================================================
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


def setup_ddp():
    if ddp_is_on():
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")


def cleanup_ddp():
    if ddp_is_on() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


# =========================================================
# 2. Utils
# =========================================================
def set_seed(seed: int = 42):
    rank = get_rank()
    seed = int(seed) + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    raw = unwrap_model(model)
    total = sum(p.numel() for p in raw.parameters())
    trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    return trainable, total


def maybe_get_scalar_pad_value(batch: Dict[str, Any]) -> int:
    pad_value = batch["pad_value"]
    if torch.is_tensor(pad_value):
        return int(pad_value.reshape(-1)[0].item())
    return int(pad_value)


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    if not all(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


def load_encoder_checkpoint(encoder: nn.Module, ckpt_path: str, strict: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    sd = strip_prefix_if_present(sd)

    if any(k.startswith("encoder.") for k in sd.keys()):
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = encoder.load_state_dict(enc_sd, strict=strict)
    else:
        missing, unexpected = encoder.load_state_dict(sd, strict=strict)

    if is_main_process():
        print(f"[Checkpoint] Loaded encoder ckpt from: {ckpt_path}")
        print(f"[Checkpoint] missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("  missing (first 20):", missing[:20])
        if len(unexpected) > 0:
            print("  unexpected (first 20):", unexpected[:20])


def load_model_only_checkpoint(model: nn.Module, ckpt_path: str, strict: bool = False):
    raw_model = unwrap_model(model)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    sd = strip_prefix_if_present(sd)
    missing, unexpected = raw_model.load_state_dict(sd, strict=strict)

    if is_main_process():
        print(f"[Init] Loaded model-only ckpt from: {ckpt_path}")
        print(f"[Init] missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("  missing (first 20):", missing[:20])
        if len(unexpected) > 0:
            print("  unexpected (first 20):", unexpected[:20])


def load_full_checkpoint(model: nn.Module, optimizer: Optional[optim.Optimizer], ckpt_path: str):
    raw_model = unwrap_model(model)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_score = ckpt.get("best_score", None)
    return start_epoch, best_score, ckpt


# =========================================================
# 3. Metrics / losses
# =========================================================
def masked_state_loss_and_metrics(
    pred_state: torch.Tensor,
    gt_state: torch.Tensor,
    alpha_rel: float = 0.05,
    eps: float = 1e-3,
):
    """
    pred_state, gt_state: [B, Tf, N, 1]
    valid mask: gt_state != 0

    返回:
      loss    : 训练用损失 = huber_loss + alpha_rel * rel_loss
      abs_sum : 计算 MAE 用
      sq_sum  : 计算 RMSE 用
      ape_sum : 计算 MAPE 用
      den     : 有效元素个数
    """
    valid = (gt_state != 0).float()
    err = pred_state - gt_state
    den = valid.sum() + 1e-6

    per_huber = F.smooth_l1_loss(pred_state, gt_state, reduction="none")
    huber_loss = (per_huber * valid).sum() / den

    per_rel = err.abs() / gt_state.abs().clamp_min(eps)
    rel_loss = (per_rel * valid).sum() / den

    loss = huber_loss + alpha_rel * rel_loss

    abs_sum = (err.abs() * valid).sum()
    sq_sum = ((err * err) * valid).sum()
    ape_sum = (per_rel * valid).sum()

    return loss, abs_sum, sq_sum, ape_sum, den

def extract_next_hop_targets(batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    从 FutureTraj 的第一个 future step 中，提取每个 agent 的最后有效 road_id 作为下一跳标签
    """
    fut = batch["FutureTraj"][:, 0]                  # [B,M,L,3]
    fut_mask = batch["FutureTraj_token_mask"][:, 0]  # [B,M,L]

    road_seq = fut[..., 0].long()                    # [B,M,L]
    lengths = fut_mask.long().sum(dim=-1)            # [B,M]
    has_valid = lengths > 0
    last_idx = lengths.clamp(min=1) - 1              # [B,M]

    y = road_seq.gather(
        dim=-1,
        index=last_idx.unsqueeze(-1)
    ).squeeze(-1)                                    # [B,M]

    pad_value = maybe_get_scalar_pad_value(batch)

    y = torch.where(
        has_valid,
        y,
        torch.full_like(y, pad_value)
    )

    mask = (y != pad_value)
    return y, mask, pad_value


def masked_next_hop_ce_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    full-node CE
    logits: [B, M, N] or [B, N]
    y:      [B, M] or [B]
    """
    if logits.dim() == 2 and y.dim() == 1:
        ce = F.cross_entropy(logits, y, reduction="none")
        return (ce * mask.float()).sum() / (mask.float().sum() + 1e-6)

    if logits.dim() != 3 or y.dim() != 2:
        raise ValueError(f"Unexpected traj shapes: logits={tuple(logits.shape)}, y={tuple(y.shape)}")

    B, M, N = logits.shape
    y_safe = torch.where(mask, y, torch.zeros_like(y)).clamp(0, N - 1)
    ce = F.cross_entropy(
        logits.reshape(B * M, N),
        y_safe.reshape(B * M),
        reduction="none",
    ).reshape(B, M)

    return (ce * mask.float()).sum() / (mask.float().sum() + 1e-6)


def candidate_level_ce_loss(cand_score: torch.Tensor, cand_idx: torch.Tensor,
                            y: torch.Tensor, mask: torch.Tensor):
    """
    candidate-level CE
    cand_score: [B,M,K]
    cand_idx  : [B,M,K]
    y         : [B,M]
    mask      : [B,M]
    """
    hit = (cand_idx == y.unsqueeze(-1))          # [B,M,K]
    covered = hit.any(dim=-1) & mask             # [B,M]

    if covered.sum() == 0:
        return cand_score.sum() * 0.0, torch.tensor(0.0, device=cand_score.device)

    target_k = hit.float().argmax(dim=-1)        # [B,M]
    ce = F.cross_entropy(
        cand_score[covered],                     # [P,K]
        target_k[covered],                       # [P]
        reduction="mean",
    )
    cov = covered.float().mean()
    return ce, cov


def next_hop_metrics(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, k: int = 5):
    """
    logits: [B, M, N] or [B, N]
    y:      [B, M] or [B]
    """
    if logits.dim() == 2 and y.dim() == 1:
        logits = logits.unsqueeze(1)
        y = y.unsqueeze(1)
        mask = mask.unsqueeze(1)

    B, M, N = logits.shape
    y_safe = torch.where(mask, y, torch.zeros_like(y)).clamp(0, N - 1)

    pred1 = logits.argmax(dim=-1)  # [B,M]
    correct1 = ((pred1 == y_safe) & mask).float().sum()

    topk = logits.topk(k, dim=-1).indices  # [B,M,k]
    hit = (topk == y_safe.unsqueeze(-1))
    hit_any = hit.any(dim=-1) & mask

    rank = hit.float().argmax(dim=-1) + 1

    rr = torch.zeros_like(rank, dtype=torch.float32)
    rr[hit_any] = 1.0 / rank[hit_any].float()

    ndcg = torch.zeros_like(rank, dtype=torch.float32)
    ndcg[hit_any] = 1.0 / torch.log2(rank[hit_any].float() + 1.0)

    den = mask.float().sum().clamp_min(1.0)
    return correct1, rr.sum(), ndcg.sum(), den


def gate_balance_loss(gate: torch.Tensor) -> torch.Tensor:
    mean_usage = gate.mean(dim=0)
    target = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
    return torch.mean((mean_usage - target) ** 2)


# =========================================================
# 4. Evaluate
# =========================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    lambda_state: float = 1.0,
    lambda_traj: float = 1.0,
    lambda_gate: float = 0.0,
    lambda_cand: float = 0.3,
    amp_enabled: bool = True,
):
    model.eval()

    s_abs_sum = torch.tensor(0.0, device=device)
    s_sq_sum = torch.tensor(0.0, device=device)
    s_ape_sum = torch.tensor(0.0, device=device)
    s_den = torch.tensor(0.0, device=device)

    t_correct1_sum = torch.tensor(0.0, device=device)
    t_rr5_sum = torch.tensor(0.0, device=device)
    t_ndcg5_sum = torch.tensor(0.0, device=device)
    t_den_sum = torch.tensor(0.0, device=device)

    total_loss_sum = torch.tensor(0.0, device=device)
    total_cov_sum = torch.tensor(0.0, device=device)
    total_steps = 0

    for batch in loader:
        batch = to_device(batch, device)

        with autocast(device_type=device.type, enabled=amp_enabled):
            out = model(batch)

            pred_state = out["y_state"]
            pred_traj = out["y_traj"]

            gt_state = batch["FutureState"]
            y_hop, hop_mask, _ = extract_next_hop_targets(batch)

            ls, abs_sum, sq_sum, ape_sum, sden = masked_state_loss_and_metrics(pred_state, gt_state)

            lt_full = masked_next_hop_ce_loss(pred_traj, y_hop, hop_mask)

            lt_cand = torch.tensor(0.0, device=device)
            cov_traj = torch.tensor(1.0, device=device)
            if "cand_score" in out and "cand_idx" in out:
                lt_cand, cov_traj = candidate_level_ce_loss(
                    out["cand_score"], out["cand_idx"], y_hop, hop_mask
                )

            lt = lt_full + lambda_cand * lt_cand

            lg = torch.tensor(0.0, device=device)
            if lambda_gate > 0.0:
                if "gate_state" in out:
                    lg = lg + gate_balance_loss(out["gate_state"])
                if "gate_traj" in out:
                    lg = lg + gate_balance_loss(out["gate_traj"])

            loss = lambda_state * ls + lambda_traj * lt + lambda_gate * lg

        c1, rr5, ndcg5, tden = next_hop_metrics(pred_traj, y_hop, hop_mask, k=5)

        total_loss_sum += loss.detach()
        total_cov_sum += cov_traj.detach()
        total_steps += 1

        s_abs_sum += abs_sum
        s_sq_sum += sq_sum
        s_ape_sum += ape_sum
        s_den += sden

        t_correct1_sum += c1
        t_rr5_sum += rr5
        t_ndcg5_sum += ndcg5
        t_den_sum += tden

    total_steps_tensor = torch.tensor(float(total_steps), device=device)
    if ddp_is_on():
        for t in [
            s_abs_sum, s_sq_sum, s_ape_sum, s_den,
            t_correct1_sum, t_rr5_sum, t_ndcg5_sum, t_den_sum,
            total_loss_sum, total_cov_sum, total_steps_tensor,
        ]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    total_steps = int(total_steps_tensor.item())

    if float(s_den.item()) < 1.0 or float(t_den_sum.item()) < 1.0:
        return {}

    mae = (s_abs_sum / s_den).item()
    rmse = math.sqrt((s_sq_sum / s_den).item())
    mape = (100.0 * (s_ape_sum / s_den)).item()

    acc = (t_correct1_sum / t_den_sum).item()
    mrr5 = (t_rr5_sum / t_den_sum).item()
    ndcg5 = (t_ndcg5_sum / t_den_sum).item()
    cov = (total_cov_sum / max(total_steps, 1)).item()

    return {
        "val/loss": (total_loss_sum / max(total_steps, 1)).item(),
        "val/mae_state": mae,
        "val/rmse_state": rmse,
        "val/mape_state": mape,
        "val/acc_traj": acc,
        "val/mrr5_traj": mrr5,
        "val/ndcg5_traj": ndcg5,
        "val/cov_traj": cov,
    }


# =========================================================
# 5. Training
# =========================================================
def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    lambda_state: float,
    lambda_traj: float,
    lambda_gate: float,
    lambda_cand: float,
    grad_clip: float,
    amp_enabled: bool,
    log_every: int = 50,
    epoch: int = 1
):
    model.train()

    run_loss = 0.0
    run_ls = 0.0
    run_lt = 0.0
    run_lg = 0.0
    run_cov = 0.0
    n_steps = 0

    for it, batch in enumerate(loader, start=1):
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=amp_enabled):
            out = model(batch)

            pred_state = out["y_state"]
            pred_traj = out["y_traj"]

            gt_state = batch["FutureState"]
            y_hop, hop_mask, _ = extract_next_hop_targets(batch)

            ls, _, _, _, _ = masked_state_loss_and_metrics(pred_state, gt_state)

            lt_full = masked_next_hop_ce_loss(pred_traj, y_hop, hop_mask)

            lt_cand = torch.tensor(0.0, device=device)
            cov_traj = torch.tensor(1.0, device=device)
            if "cand_score" in out and "cand_idx" in out:
                lt_cand, cov_traj = candidate_level_ce_loss(
                    out["cand_score"], out["cand_idx"], y_hop, hop_mask
                )

            lt = lt_full + lambda_cand * lt_cand

            lg = torch.tensor(0.0, device=device)
            if lambda_gate > 0.0:
                if "gate_state" in out:
                    lg = lg + gate_balance_loss(out["gate_state"])
                if "gate_traj" in out:
                    lg = lg + gate_balance_loss(out["gate_traj"])

            loss = lambda_state * ls + lambda_traj * lt + lambda_gate * lg

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        run_loss += float(loss.item())
        run_ls += float(ls.item())
        run_lt += float(lt.item())
        run_lg += float(lg.item())
        run_cov += float(cov_traj.item())
        n_steps += 1

        if is_main_process() and (it % log_every == 0):
            print(
                f"[Train][Epoch {epoch:03d}] it={it:04d} "
                f"loss={run_loss / n_steps:.6f} "
                f"ls={run_ls / n_steps:.6f} "
                f"lt={run_lt / n_steps:.6f} "
                f"lg={run_lg / n_steps:.6f} "
                f"cov={run_cov / n_steps:.6f}"
            )

    sums = torch.tensor([run_loss, run_ls, run_lt, run_lg, run_cov, float(n_steps)], device=device)
    if ddp_is_on():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)

    run_loss, run_ls, run_lt, run_lg, run_cov, n_steps = sums.tolist()
    n_steps = max(int(n_steps), 1)

    return {
        "train/loss": run_loss / n_steps,
        "train/loss_state": run_ls / n_steps,
        "train/loss_traj": run_lt / n_steps,
        "train/loss_gate": run_lg / n_steps,
        "train/cov_traj": run_cov / n_steps,
    }


# =========================================================
# 6. Build model
# =========================================================
def build_model(args, static, device: torch.device):
    enc_config = {
        "d_model": args.d_model,
        "d_hidden": args.d_hidden,
        "d_impedance": args.d_impedance,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "input_window": args.input_window,
        "output_window": args.output_window,
        "max_traj_len": args.max_traj_len,
        "conv_type": args.conv_type,
        "use_en": bool(getattr(args, "use_en", True)),
        "use_uniform_transition": args.use_uniform_transition,
        "moe_traj_out": args.moe_traj_out,
        "moe_num_experts": args.moe_num_experts,
        "moe_gate_dropout": args.moe_gate_dropout,
        "moe_sync_from_base": args.moe_sync_from_base,
        "moe_top_k": args.moe_top_k,
        "moe_noisy_gate": args.moe_noisy_gate,
        "moe_noise_std": args.moe_noise_std,
        "moe_state_refiner": args.moe_state_refiner,
        "state_moe_hidden": args.state_moe_hidden,
        "state_moe_top_k": args.state_moe_top_k,
        "state_moe_noisy_gate": args.state_moe_noisy_gate,
        "state_moe_noise_std": args.state_moe_noise_std,
        "state_moe_scale": args.state_moe_scale,
    }

    encoder = STRecurrentImpedanceEncoder(enc_config, static)

    model = STLLMMMoEModel(
        encoder=encoder,
        d_hidden=args.d_hidden,
        llm_path=args.llm_path,
        num_latents_x=args.num_latents_x,
        num_latents_e=args.num_latents_e,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_nodes=int(static["num_nodes"]),
        freeze_encoder=args.freeze_encoder,
    )

    if getattr(args, "encoder_ckpt", ""):
        load_encoder_checkpoint(model.encoder, args.encoder_ckpt, strict=False)

    model.to(device)
    return model


# =========================================================
# 7. Config / args
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--encoder_ckpt", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--llm_path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init_ckpt", type=str, default=None)

    parser.add_argument("--freeze_encoder", type=str, default=None)
    parser.add_argument("--amp", type=str, default=None)
    parser.add_argument("--find_unused_parameters", type=str, default=None)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cli = vars(args)
    for k, v in cli.items():
        if k == "config":
            continue
        if v is not None:
            if k in ["freeze_encoder", "amp", "find_unused_parameters"] and isinstance(v, str):
                v = v.lower() in ["1", "true", "yes", "y"]
            cfg[k] = v

    cfg["config"] = args.config
    return argparse.Namespace(**cfg)


# =========================================================
# 8. Main
# =========================================================
def main():
    args = parse_args()
    setup_ddp()

    if ddp_is_on():
        device = torch.device(f"cuda:{get_local_rank()}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)

    if ddp_is_on():
        dist.barrier()

    set_seed(args.seed)

    if is_main_process():
        print(f"[Device] {device}")
        print(f"[DDP] enabled={ddp_is_on()} rank={get_rank()} world_size={get_world_size()} local_rank={get_local_rank()}")

    config = vars(args)
    train_loader, val_loader, test_loader, static = build_dataloaders(
        config,
        distributed=ddp_is_on(),
        rank=get_rank(),
        world_size=get_world_size(),
    )

    if is_main_process():
        with open(os.path.join(args.save_dir, "config_merged.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(vars(args), f, allow_unicode=True, sort_keys=False)

    model = build_model(args, static, device)

    if getattr(args, "init_ckpt", None):
        if os.path.exists(args.init_ckpt):
            load_model_only_checkpoint(model, args.init_ckpt, strict=False)
        else:
            if is_main_process():
                print(f"[Init] file not found: {args.init_ckpt}")

    if ddp_is_on():
        model = DDP(
            model,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
            find_unused_parameters=bool(getattr(args, "find_unused_parameters", False)),
        )

    trainable, total = count_trainable_params(model)
    if is_main_process():
        print(f"[Params] trainable={trainable:,} / total={total:,}")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    amp_enabled = bool(getattr(args, "amp", True)) and (device.type == "cuda")
    scaler = GradScaler(device.type, enabled=amp_enabled)

    start_epoch = 1
    best_score = None
    best_path = os.path.join(args.save_dir, "best.pt")
    last_path = os.path.join(args.save_dir, "last.pt")

    if getattr(args, "resume", None):
        if os.path.exists(args.resume):
            start_epoch, best_score, _ = load_full_checkpoint(model, optimizer, args.resume)
            if is_main_process():
                print(f"[Resume] loaded from {args.resume}, start_epoch={start_epoch}, best_score={best_score}")
        else:
            if is_main_process():
                print(f"[Resume] file not found: {args.resume}")

    for epoch in range(start_epoch, args.epochs + 1):
        if ddp_is_on() and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        t0 = time.time()

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            lambda_state=args.lambda_state,
            lambda_traj=args.lambda_traj,
            lambda_gate=args.lambda_gate,
            lambda_cand=getattr(args, "lambda_cand", 0.3),
            grad_clip=args.grad_clip,
            amp_enabled=amp_enabled,
            log_every=args.log_every,
            epoch=epoch
        )

        val_stats = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            lambda_state=args.lambda_state,
            lambda_traj=args.lambda_traj,
            lambda_gate=args.lambda_gate,
            lambda_cand=getattr(args, "lambda_cand", 0.3),
            amp_enabled=amp_enabled,
        )

        elapsed = time.time() - t0
        log = {"epoch": epoch, "time_sec": round(elapsed, 2), **train_stats, **val_stats}

        if is_main_process():
            print(json.dumps(log, ensure_ascii=False, indent=2))

        if is_main_process():
            raw_model = unwrap_model(model)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "val_stats": val_stats,
                    "best_score": best_score,
                },
                last_path,
            )

        if val_stats:
            score = (
                -2.0 * val_stats["val/mae_state"]
                -0.4 * val_stats["val/rmse_state"]
                -0.1 * val_stats["val/mape_state"]
                +1.8 * val_stats["val/acc_traj"]
                +0.9 * val_stats["val/mrr5_traj"]
                +0.9 * val_stats["val/ndcg5_traj"]
            )

            if (best_score is None) or (score > best_score):
                best_score = score
                if is_main_process():
                    raw_model = unwrap_model(model)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": raw_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "args": vars(args),
                            "val_stats": val_stats,
                            "best_score": best_score,
                        },
                        best_path,
                    )
                    print(f"[Save] best checkpoint saved to {best_path}")

    if ddp_is_on():
        dist.barrier()

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        raw_model = unwrap_model(model)
        raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if is_main_process():
            print(f"[Load] best checkpoint from {best_path}")

    test_stats = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        lambda_state=args.lambda_state,
        lambda_traj=args.lambda_traj,
        lambda_gate=args.lambda_gate,
        lambda_cand=getattr(args, "lambda_cand", 0.3),
        amp_enabled=amp_enabled,
    )

    if is_main_process():
        print("[Test]")
        print(json.dumps(test_stats, ensure_ascii=False, indent=2))

    cleanup_ddp()


if __name__ == "__main__":
    main()