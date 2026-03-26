#!/usr/bin/env python3
"""Train a Conditional Normalizing Flow emulator on CAMELS profile outputs.

Replaces the ANP architecture from train_anp_emulator.py with a conditional
Neural Spline Flow (NSF) from the zuko library.  Reuses the same data loading,
normalization, mean-prior, and evaluation infrastructure.

Key difference: the flow is a *pointwise* density model — each (x, y) point is
an independent draw, so there is no context/target split, no KL term, and no
latent variable.  The conditioning vector per point is the full x feature
(log_M, log_r/R500, theta, redshift).  The flow models the conditional
distribution p(y | x).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import zuko.flows

# Reuse data loading / normalization / mean-prior infrastructure wholesale.
from train_anp_emulator import (
    ALL_PROFILE_LOG_TARGETS,
    ALL_PROFILE_TARGETS,
    MeanModel,
    RunFamilyTask,
    RunTask,
    add_mean_back,
    apply_residual_prior,
    build_tasks,
    cleanup_distributed,
    compute_norm_stats,
    denorm_y,
    dist_barrier,
    dist_rank,
    dist_world_size,
    evaluate_mean_model_metrics,
    flatten_family_tasks,
    filter_snapshots_in_families,
    get_mean_model_config,
    is_dist_ready,
    is_main_process,
    load_mean_checkpoint,
    normalize_tasks,
    parse_snapshot_redshifts,
    predict_mean_from_raw_x,
    remap_flat_tasks_to_families,
    reduce_mean_scalar,
    reduce_mean_stats,
    restore_all_profiles_physical_units,
    save_mean_checkpoint,
    set_seed,
    setup_distributed,
    split_tasks,
    train_mean_model,
    unwrap_model,
    SINGLE_TARGET_CHOICES,
    TARGET_CHOICES,
)


# ──────────────────────────────────────────────────────────────────────
# Flatten RunFamilyTask data into flat (x, y) tensors for the NF
# ──────────────────────────────────────────────────────────────────────

def flatten_tasks_to_tensors(
    families: List[RunFamilyTask],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten all family/snapshot/halo/radial data into (N, x_dim) and (N, y_dim)."""
    xs, ys = [], []
    for fam in families:
        for t in fam.snapshots:
            xs.append(t.x.reshape(-1, t.x.shape[-1]))
            ys.append(t.y.reshape(-1, t.y.shape[-1]))
    x_np = np.concatenate(xs, axis=0).astype(np.float32)
    y_np = np.concatenate(ys, axis=0).astype(np.float32)
    return torch.from_numpy(x_np), torch.from_numpy(y_np)


# ──────────────────────────────────────────────────────────────────────
# Conditional NF model wrapper
# ──────────────────────────────────────────────────────────────────────

class ConditionalNF(nn.Module):
    """Thin wrapper around a zuko NSF conditioned on x to model p(y | x)."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_features: List[int],
        n_transforms: int = 8,
        n_bins: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.flow = zuko.flows.NSF(
            features=y_dim,
            context=x_dim,
            bins=n_bins,
            transforms=n_transforms,
            hidden_features=hidden_features,
            randperm=True,
        )

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute log p(y | x) for each point. Returns shape (N,)."""
        dist = self.flow(x)
        return dist.log_prob(y)

    def sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Sample from p(y | x). Returns shape (n_samples, N, y_dim)."""
        dist = self.flow(x)
        return dist.sample((n_samples,))


# ──────────────────────────────────────────────────────────────────────
# Training and evaluation
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_clip: float = 1.0,
    accum_steps: int = 1,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.train()
    total_nll = 0.0
    total_count = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (xb, yb) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            log_p = model.module.log_prob(xb, yb) if hasattr(model, "module") else model.log_prob(xb, yb)
            loss = -log_p.mean() / accum_steps

        scaler.scale(loss).backward()

        if ((step + 1) % accum_steps == 0) or (step + 1 == len(loader)):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bsz = int(xb.shape[0])
        total_nll += float(-log_p.sum().detach().cpu())
        total_count += bsz

    mean_nll = total_nll / max(1, total_count)
    return {"nll": mean_nll}


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            log_p = model.module.log_prob(xb, yb) if hasattr(model, "module") else model.log_prob(xb, yb)

        bsz = int(xb.shape[0])
        total_nll += float(-log_p.sum().detach().cpu())
        total_count += bsz

    mean_nll = total_nll / max(1, total_count)
    return {"nll": mean_nll}


@torch.no_grad()
def evaluate_flow_metrics(
    model: ConditionalNF,
    x: torch.Tensor,
    y: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    mean_model: Optional[MeanModel],
    target_names: List[str],
    n_samples: int = 50,
    batch_size: int = 4096,
    device: torch.device = torch.device("cpu"),
    core_radius_frac: float = 0.2,
    core_radius_min_bins: int = 3,
) -> Dict[str, Any]:
    """Compute RMSE and NLL metrics for the flow in original (physical) units."""
    model.eval()
    N = x.shape[0]
    y_dim = y.shape[-1]

    all_mu = []
    all_std = []
    all_nll = []

    for i in range(0, N, batch_size):
        xb = x[i : i + batch_size].to(device)
        yb = y[i : i + batch_size].to(device)

        # NLL in normalised space
        log_p = model.log_prob(xb, yb)
        all_nll.append(-log_p.cpu())

        # Sample to get mean/std predictions
        samples = model.sample(xb, n_samples=n_samples)  # (n_samples, B, y_dim)
        mu = samples.mean(dim=0)  # (B, y_dim)
        std = samples.std(dim=0)  # (B, y_dim)
        all_mu.append(mu.cpu())
        all_std.append(std.cpu())

    mu_norm = torch.cat(all_mu, dim=0)   # (N, y_dim)
    std_norm = torch.cat(all_std, dim=0)
    nll_vals = torch.cat(all_nll, dim=0)

    # De-normalise predictions
    mu_orig = denorm_y(mu_norm, y_mean.cpu(), y_std.cpu())
    y_orig = denorm_y(y, y_mean.cpu(), y_std.cpu())

    # Add mean model back if present
    if mean_model is not None:
        x_raw = x * x_std.cpu().unsqueeze(0) + x_mean.cpu().unsqueeze(0)
        # We need 3D tensors for add_mean_back
        mu_orig_3d = mu_orig.unsqueeze(0)
        x_norm_3d = x.unsqueeze(0)
        mean_contrib = predict_mean_from_raw_x(x_raw.unsqueeze(0).to(mean_model.net[0].weight.device), mean_model)
        mean_contrib = mean_contrib.cpu().squeeze(0)
        mu_orig = mu_orig + mean_contrib
        y_orig = y_orig + mean_contrib

    # Restore physical units for log-space channels
    if len(target_names) > 1:
        y_orig_3d = y_orig.unsqueeze(0)
        mu_orig_3d = mu_orig.unsqueeze(0)
        y_orig_3d, mu_orig_3d, _ = restore_all_profiles_physical_units(
            y_orig_3d, mu_orig_3d, None, target_names
        )
        y_orig = y_orig_3d.squeeze(0)
        mu_orig = mu_orig_3d.squeeze(0)

    resid_sq = (mu_orig - y_orig).pow(2)
    rmse_all = float(resid_sq.mean().sqrt().item())
    mean_nll = float(nll_vals.mean().item())

    # Per-target RMSE
    per_target: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(target_names):
        per_target[name] = {
            "rmse_original_units": float(resid_sq[:, i].mean().sqrt().item()),
        }

    return {
        "rmse_original_units": rmse_all,
        "nll_normalized": mean_nll,
        "per_target": per_target,
    }


# ──────────────────────────────────────────────────────────────────────
# CLI argument parser
# ──────────────────────────────────────────────────────────────────────

def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a Conditional Normalizing Flow emulator on CAMELS profiles"
    )

    # ── Data arguments (mirroring train_anp_emulator.py) ──
    p.add_argument("--profiles-base", type=str, required=True)
    p.add_argument("--param-csv", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="nf_training_runs")
    p.add_argument("--suite", type=str, default="IllustrisTNG")
    p.add_argument("--sim-set", type=str, default="SB35")
    p.add_argument("--snapnum", type=int, default=90)
    p.add_argument("--snapnums", type=int, nargs="+", default=None)
    p.add_argument(
        "--snapshot-redshifts", type=str,
        default="90:0.0,74:0.5,60:1.0,44:2.0",
    )
    p.add_argument("--min-snapshots-per-run", type=int, default=1)
    p.add_argument("--target-name", type=str, default="all_profiles", choices=TARGET_CHOICES)
    p.add_argument(
        "--all-profiles-subset", type=str, nargs="+", default=None,
        choices=ALL_PROFILE_TARGETS,
    )

    p.add_argument("--theta-dim", type=int, default=35)
    p.add_argument("--theta-start-idx", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eps", type=float, default=1e-30)

    p.add_argument("--max-runs", type=int, default=0)
    p.add_argument("--min-halos", type=int, default=4)
    p.add_argument("--max-halos-per-run", type=int, default=0)
    p.add_argument("--radial-stride", type=int, default=1)
    p.add_argument("--r500-physical-factor", type=float, default=1.0)

    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)

    # ── Training arguments ──
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--accum-steps", type=int, default=1)

    # ── NF architecture ──
    p.add_argument(
        "--nf-hidden", type=int, nargs="+", default=[256, 256],
        help="Hidden layer sizes for each NSF coupling transform.",
    )
    p.add_argument(
        "--nf-transforms", type=int, default=8,
        help="Number of NSF coupling transform layers.",
    )
    p.add_argument(
        "--nf-bins", type=int, default=8,
        help="Number of rational-quadratic spline bins per transform.",
    )
    p.add_argument("--dropout", type=float, default=0.0)

    # ── Mean prior (reused from ANP pipeline) ──
    p.add_argument("--disable-mean-prior", action="store_true")
    p.add_argument("--mean-use-theta", action="store_true")
    p.add_argument("--mean-hidden-dim", type=int, default=128)
    p.add_argument("--mean-epochs", type=int, default=80)
    p.add_argument("--mean-lr", type=float, default=1e-3)
    p.add_argument("--mean-weight-decay", type=float, default=1e-3)
    p.add_argument("--mean-batch-size", type=int, default=131072)
    p.add_argument("--mean-log-every", type=int, default=10)
    p.add_argument("--mean-predict-batch-size", type=int, default=262144)
    p.add_argument(
        "--training-stage", type=str, default="full",
        choices=["full", "mean_only", "nf_only"],
    )
    p.add_argument("--mean-checkpoint-path", type=str, default="")
    p.add_argument("--mean-output-path", type=str, default="")

    # ── Evaluation ──
    p.add_argument("--eval-samples", type=int, default=50)
    p.add_argument("--save-every-epochs", type=int, default=20)
    p.add_argument("--val-detailed-every", type=int, default=5)

    # ── Misc ──
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--enable-ddp", action="store_true")
    p.add_argument("--ddp-timeout-sec", type=int, default=3600)
    p.add_argument("--ddp-num-workers", type=int, default=0)
    p.add_argument(
        "--disable-continuous-redshift-feature", action="store_true",
    )

    # build_tasks compatibility stubs
    p.add_argument("--cc-indicator", action="store_true", default=False)
    p.add_argument("--cc-indicator-core-bins", type=int, default=6)

    return p


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────

def main():
    args = make_arg_parser().parse_args()

    if args.theta_start_idx < 2:
        raise ValueError("theta_start_idx must be >= 2")

    # Resolve target names
    if args.target_name == "all_profiles":
        if args.all_profiles_subset:
            selected = set(args.all_profiles_subset)
            target_names = [n for n in ALL_PROFILE_TARGETS if n in selected]
        else:
            target_names = list(ALL_PROFILE_TARGETS)
    else:
        target_names = [args.target_name]

    args.resolved_all_profile_targets = target_names
    args.resolved_snapnums = list(args.snapnums) if args.snapnums else [int(args.snapnum)]
    args.redshift_by_snap = parse_snapshot_redshifts(args.snapshot_redshifts)
    args.use_continuous_redshift_feature = not args.disable_continuous_redshift_feature
    args.redshift_feature_idx = int(args.theta_start_idx + args.theta_dim)

    cc_offset = args.theta_start_idx + args.theta_dim
    if args.use_continuous_redshift_feature:
        cc_offset += 1
    args.cc_indicator_feature_idx = int(cc_offset)

    missing_snap_map = [s for s in args.resolved_snapnums if int(s) not in args.redshift_by_snap]
    if missing_snap_map:
        raise ValueError(f"Missing redshift mapping for snapshots: {missing_snap_map}")
    if args.min_snapshots_per_run < 1:
        raise ValueError("min_snapshots_per_run must be >= 1")
    if len(args.resolved_snapnums) > 1 and args.min_snapshots_per_run < 2:
        args.min_snapshots_per_run = 2

    if args.training_stage == "nf_only":
        if args.disable_mean_prior:
            raise ValueError("--training-stage=nf_only requires mean prior (remove --disable-mean-prior)")
        if not args.mean_checkpoint_path:
            raise ValueError("--training-stage=nf_only requires --mean-checkpoint-path")
    if args.training_stage == "mean_only" and args.disable_mean_prior:
        raise ValueError("--training-stage=mean_only requires mean prior (remove --disable-mean-prior)")

    # ── DDP setup ──
    if args.enable_ddp:
        ddp_enabled, rank, world_size, local_rank = setup_distributed(
            timeout_sec=int(args.ddp_timeout_sec)
        )
    else:
        ddp_enabled, rank, world_size, local_rank = False, 0, 1, 0
    main_proc = rank == 0

    set_seed(args.seed)

    if ddp_enabled and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    if main_proc:
        print(f"Device: {device}, AMP: {use_amp}, DDP: {ddp_enabled} (world_size={world_size})")

    # ── Output directory ──
    out_root = Path(args.output_dir)
    if main_proc:
        out_root.mkdir(parents=True, exist_ok=True)
    if ddp_enabled:
        dist_barrier(device)

    run_tag = time.strftime("%Y%m%d_%H%M%S") if main_proc else ""
    if ddp_enabled:
        obj = [run_tag]
        dist.broadcast_object_list(obj, src=0)
        run_tag = str(obj[0])
    run_prefix = "mean" if args.training_stage == "mean_only" else "nf"
    out_dir = out_root / f"{run_prefix}_{args.target_name}_{run_tag}"
    if main_proc:
        out_dir.mkdir(parents=True, exist_ok=True)
    if ddp_enabled:
        dist_barrier(device)

    # ── Build tasks ──
    tasks = build_tasks(args)
    if len(tasks) < 20:
        raise RuntimeError(f"Too few run families discovered ({len(tasks)})")

    train_raw, val_raw, test_raw = split_tasks(
        tasks, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed
    )

    if main_proc:
        n_tr = sum(len(f.snapshots) for f in train_raw)
        n_va = sum(len(f.snapshots) for f in val_raw)
        n_te = sum(len(f.snapshots) for f in test_raw)
        print(
            f"Split sizes (families): train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)} | "
            f"snapshots: train={n_tr}, val={n_va}, test={n_te}"
        )

    y_dim = train_raw[0].snapshots[0].y.shape[-1]

    # ── Mean prior ──
    mean_theta_dim = int(args.theta_dim) if getattr(args, "mean_use_theta", False) else 0
    args.mean_theta_dim = mean_theta_dim
    args.mean_n_hidden = 3 if mean_theta_dim > 0 else 2
    mean_model: Optional[MeanModel] = None
    mean_prior_enabled = not args.disable_mean_prior
    mean_prior_source = "disabled"
    mean_checkpoint_used: Optional[str] = None
    mean_metrics_summary: Optional[Dict[str, Any]] = None

    if mean_prior_enabled:
        train_raw_flat_original = flatten_family_tasks(train_raw)
        val_raw_flat_original = flatten_family_tasks(val_raw)
        test_raw_flat_original = flatten_family_tasks(test_raw)

        if args.training_stage == "nf_only":
            mean_ckpt = Path(args.mean_checkpoint_path)
            if main_proc:
                print(f"Loading frozen mean model from: {mean_ckpt}")
            mean_model, _ = load_mean_checkpoint(mean_ckpt, device=device)
            mean_prior_source = "loaded"
            mean_checkpoint_used = str(mean_ckpt)
        else:
            if main_proc:
                print("Pre-training mean profile model...")
            mean_model = train_mean_model(
                train_raw_flat_original,
                y_dim=y_dim,
                args=args,
                device=device,
                verbose=main_proc,
            )
            mean_prior_source = "trained"

        if main_proc:
            mean_metrics_summary = {
                "train": evaluate_mean_model_metrics(
                    train_raw_flat_original, mean_model, device=device,
                    batch_size=int(args.mean_predict_batch_size),
                    target_names=target_names,
                ),
                "val": evaluate_mean_model_metrics(
                    val_raw_flat_original, mean_model, device=device,
                    batch_size=int(args.mean_predict_batch_size),
                    target_names=target_names,
                ),
                "test": evaluate_mean_model_metrics(
                    test_raw_flat_original, mean_model, device=device,
                    batch_size=int(args.mean_predict_batch_size),
                    target_names=target_names,
                ),
            }
            print(
                f"Mean model RMSE (orig units): "
                f"train={mean_metrics_summary['train']['rmse_original_units']:.4g}, "
                f"val={mean_metrics_summary['val']['rmse_original_units']:.4g}, "
                f"test={mean_metrics_summary['test']['rmse_original_units']:.4g}"
            )

        if main_proc and args.training_stage in ("full", "mean_only"):
            mean_ckpt_path = (
                Path(args.mean_output_path) if args.mean_output_path
                else (out_dir / "mean_model.pt")
            )
            save_mean_checkpoint(
                mean_ckpt_path, mean_model, args=args,
                target_names=target_names, mean_metrics=mean_metrics_summary,
            )
            mean_checkpoint_used = str(mean_ckpt_path)
            print(f"Saved mean model checkpoint: {mean_ckpt_path}")

        if args.training_stage == "mean_only":
            if main_proc:
                metrics = {
                    "target_name": args.target_name,
                    "target_names": target_names,
                    "training_stage": args.training_stage,
                    "mean_metrics": mean_metrics_summary,
                }
                with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                with (out_dir / "args.json").open("w", encoding="utf-8") as f:
                    json.dump(vars(args), f, indent=2)
            if ddp_enabled:
                dist_barrier(device)
                cleanup_distributed()
            return

        # Apply residual prior: y <- y - mean(logM, logr, theta, z)
        train_raw_flat = apply_residual_prior(
            train_raw_flat_original, mean_model,
            device=device, batch_size=args.mean_predict_batch_size,
        )
        val_raw_flat = apply_residual_prior(
            val_raw_flat_original, mean_model,
            device=device, batch_size=args.mean_predict_batch_size,
        )
        test_raw_flat = apply_residual_prior(
            test_raw_flat_original, mean_model,
            device=device, batch_size=args.mean_predict_batch_size,
        )
        train_raw = remap_flat_tasks_to_families(train_raw, train_raw_flat)
        val_raw = remap_flat_tasks_to_families(val_raw, val_raw_flat)
        test_raw = remap_flat_tasks_to_families(test_raw, test_raw_flat)
        if main_proc:
            print("Applied residual targets: y <- y - mean_model(x)")
    else:
        if main_proc:
            print("Mean profile prior disabled; training NF on direct targets.")

    # ── Normalization ──
    norm_np = compute_norm_stats(train_raw)
    train_tasks = normalize_tasks(train_raw, norm_np)
    val_tasks = normalize_tasks(val_raw, norm_np)
    test_tasks = normalize_tasks(test_raw, norm_np)

    y_mean = torch.tensor(norm_np["y_mean"], dtype=torch.float32, device=device)
    y_std = torch.tensor(norm_np["y_std"], dtype=torch.float32, device=device)
    x_mean_t = torch.tensor(norm_np["x_mean"], dtype=torch.float32, device=device)
    x_std_t = torch.tensor(norm_np["x_std"], dtype=torch.float32, device=device)

    # ── Flatten to point-level tensors for flow training ──
    train_x, train_y = flatten_tasks_to_tensors(train_tasks)
    val_x, val_y = flatten_tasks_to_tensors(val_tasks)
    test_x, test_y = flatten_tasks_to_tensors(test_tasks)

    if main_proc:
        print(
            f"Dataset sizes (points): train={train_x.shape[0]}, "
            f"val={val_x.shape[0]}, test={test_x.shape[0]}"
        )
        print(f"x_dim={train_x.shape[1]}, y_dim={train_y.shape[1]}")

    # ── Data loaders ──
    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)
    test_ds = TensorDataset(test_x, test_y)

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if ddp_enabled else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if ddp_enabled else None
    )
    loader_workers = int(args.ddp_num_workers) if ddp_enabled else int(args.num_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ── Build model ──
    x_dim = int(train_x.shape[1])
    model = ConditionalNF(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_features=list(args.nf_hidden),
        n_transforms=args.nf_transforms,
        n_bins=args.nf_bins,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if main_proc:
        print(f"ConditionalNF: {n_params:,} trainable parameters")
        print(f"  transforms={args.nf_transforms}, bins={args.nf_bins}, hidden={args.nf_hidden}")

    if ddp_enabled:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Training loop ──
    best = {"score": float("inf"), "epoch": -1, "state": None}
    history: List[Dict[str, Any]] = []
    stale = 0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        tr_stats = train_one_epoch(
            model, train_loader, opt, scaler, device,
            grad_clip=args.grad_clip,
            accum_steps=args.accum_steps,
            use_amp=use_amp,
        )
        va_stats = validate_epoch(model, val_loader, device, use_amp=use_amp)

        if ddp_enabled:
            tr_stats = reduce_mean_stats(tr_stats, device)
            va_stats = reduce_mean_stats(va_stats, device)

        sch.step()

        row: Dict[str, Any] = {
            "epoch": epoch + 1,
            "lr": float(opt.param_groups[0]["lr"]),
            "train_nll": tr_stats["nll"],
            "val_nll": va_stats["nll"],
        }

        # Detailed evaluation periodically
        need_detailed = (
            (int(args.val_detailed_every) > 0 and ((epoch + 1) % int(args.val_detailed_every) == 0))
            or (epoch == 0)
            or ((epoch + 1) == args.epochs)
        )

        stop_now = False
        if main_proc:
            core_model = unwrap_model(model)

            if need_detailed:
                det_metrics = evaluate_flow_metrics(
                    core_model,
                    val_x, val_y,
                    y_mean=y_mean, y_std=y_std,
                    x_mean=x_mean_t, x_std=x_std_t,
                    mean_model=mean_model,
                    target_names=target_names,
                    n_samples=min(args.eval_samples, 20),
                    device=device,
                )
                row["val_rmse_original_units"] = det_metrics["rmse_original_units"]
                per_target = det_metrics.get("per_target", {})
                for tname, tvals in per_target.items():
                    row[f"val_{tname}_rmse_orig"] = tvals["rmse_original_units"]

            print(
                f"Epoch {epoch+1:03d}/{args.epochs} "
                f"lr={row['lr']:.2e} "
                f"train_nll={tr_stats['nll']:.4f} "
                f"val_nll={va_stats['nll']:.4f}"
                + (f"  val_rmse_orig={row['val_rmse_original_units']:.4g}" if "val_rmse_original_units" in row else "")
            )
            if need_detailed and "per_target" in det_metrics:
                parts = [
                    f"{k}={v['rmse_original_units']:.4g}"
                    for k, v in det_metrics["per_target"].items()
                ]
                print(f"  per-target RMSE: {', '.join(parts)}")

            score = va_stats["nll"]
            if score < (best["score"] - float(args.early_stop_min_delta)):
                best = {
                    "score": float(score),
                    "epoch": epoch + 1,
                    "state": {
                        k: v.detach().cpu().clone()
                        for k, v in core_model.state_dict().items()
                    },
                }
                stale = 0
            else:
                stale += 1

            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                stop_now = True

            if args.save_every_epochs > 0 and ((epoch + 1) % args.save_every_epochs == 0):
                periodic_path = out_dir / f"epoch_{epoch+1:04d}.pt"
                torch.save(
                    {
                        "model_state_dict": core_model.state_dict(),
                        "model_config": {
                            "x_dim": x_dim,
                            "y_dim": y_dim,
                            "hidden_features": list(args.nf_hidden),
                            "n_transforms": args.nf_transforms,
                            "n_bins": args.nf_bins,
                            "dropout": args.dropout,
                        },
                        "args": vars(args),
                        "norm": norm_np,
                        "epoch": epoch + 1,
                        "metrics": row,
                        "target_names": target_names,
                        "mean_prior_enabled": mean_prior_enabled,
                        "mean_prior_source": mean_prior_source,
                        "mean_checkpoint": mean_checkpoint_used,
                        "mean_model_state_dict": (
                            None if mean_model is None else mean_model.state_dict()
                        ),
                        "mean_model_config": (
                            None if mean_model is None else get_mean_model_config(mean_model)
                        ),
                    },
                    periodic_path,
                )
                print(f"Saved periodic checkpoint: {periodic_path}")

        history.append(row)

        if ddp_enabled:
            stop_tensor = torch.tensor(
                1 if stop_now else 0, device=device, dtype=torch.int32
            )
            dist.broadcast(stop_tensor, src=0)
            stop_now = bool(stop_tensor.item())
        if stop_now:
            break

    # ── Restore best model and final evaluation ──
    if main_proc and best["state"] is not None:
        core_model = unwrap_model(model)
        core_model.load_state_dict(best["state"])
        print(f"Restored best checkpoint from epoch {best['epoch']} (val_nll={best['score']:.4f})")

        eval_model = core_model

        test_metrics = evaluate_flow_metrics(
            eval_model,
            test_x, test_y,
            y_mean=y_mean, y_std=y_std,
            x_mean=x_mean_t, x_std=x_std_t,
            mean_model=mean_model,
            target_names=target_names,
            n_samples=args.eval_samples,
            device=device,
        )
        print(f"Test metrics: {test_metrics}")

        # Save best checkpoint
        ckpt_path = out_dir / "best_model.pt"
        torch.save(
            {
                "model_state_dict": eval_model.state_dict(),
                "model_config": {
                    "x_dim": x_dim,
                    "y_dim": y_dim,
                    "hidden_features": list(args.nf_hidden),
                    "n_transforms": args.nf_transforms,
                    "n_bins": args.nf_bins,
                    "dropout": args.dropout,
                },
                "args": vars(args),
                "norm": norm_np,
                "best": best,
                "target_name": args.target_name,
                "target_names": target_names,
                "mean_prior_enabled": mean_prior_enabled,
                "mean_prior_source": mean_prior_source,
                "mean_checkpoint": mean_checkpoint_used,
                "mean_model_state_dict": (
                    None if mean_model is None else mean_model.state_dict()
                ),
                "mean_model_config": (
                    None if mean_model is None else get_mean_model_config(mean_model)
                ),
            },
            ckpt_path,
        )

        metrics_out = {
            "target_name": args.target_name,
            "target_names": target_names,
            "training_stage": args.training_stage,
            "model_type": "ConditionalNF",
            "nf_config": {
                "n_transforms": args.nf_transforms,
                "n_bins": args.nf_bins,
                "hidden_features": list(args.nf_hidden),
            },
            "n_params": n_params,
            "n_train": int(train_x.shape[0]),
            "n_val": int(val_x.shape[0]),
            "n_test": int(test_x.shape[0]),
            "best_epoch": best["epoch"],
            "best_val_nll": best["score"],
            "test": test_metrics,
            "mean_prior": {
                "enabled": mean_prior_enabled,
                "source": mean_prior_source,
                "checkpoint": mean_checkpoint_used,
                "metrics": mean_metrics_summary,
            },
            "normalization": {
                "y_mean": norm_np["y_mean"].tolist(),
                "y_std": norm_np["y_std"].tolist(),
            },
            "checkpoint": str(ckpt_path),
        }
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics_out, f, indent=2)
        with (out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        with (out_dir / "args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

        print(f"Artifacts written to: {out_dir}")

    if ddp_enabled:
        dist_barrier(device)
        cleanup_distributed()


if __name__ == "__main__":
    main()
