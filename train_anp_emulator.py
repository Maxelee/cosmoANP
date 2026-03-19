#!/usr/bin/env python3
"""Train an Attentive Neural Process emulator on CAMELS profile outputs.

This script is designed to run outside notebooks for full-scale GPU training.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler


SINGLE_TARGET_CHOICES = [
    "log_pressure",
    "log_gas_density",
    "log_temperature",
    "log_metallicity",
    "radial_gas_velocity",
    "gas_velocity_dispersion",
    "radial_gas_velocity_dispersion",
    "rotational_gas_velocity",
    "xsb_proxy",
]

ALL_PROFILE_TARGETS = [
    "potential",
    "DM_density",
    "stellar_density",
    "gas_density",
    "temperature",
    "pressure",
    "metallicity",
    "radial_gas_velocity",
    "radial_gas_velocity_dispersion",
    "rotational_gas_velocity",
    "gas_velocity_dispersion",
    "hot_gas_density",
    "hot_temperature",
    "hot_pressure",
    "hot_metallicity",
    "hot_radial_gas_velocity",
    "hot_radial_gas_velocity_dispersion",
    "hot_rotational_gas_velocity",
    "hot_gas_velocity_dispersion",
]

# Positive-definite channels that should be modeled in log10 space for
# numerical conditioning and physical positivity in all_profiles mode.
ALL_PROFILE_LOG_TARGETS = {
    "potential",
    "DM_density",
    "stellar_density",
    "gas_density",
    "temperature",
    "pressure",
    "metallicity",
    "hot_gas_density",
    "hot_temperature",
    "hot_pressure",
    "hot_metallicity",
}

TARGET_CHOICES = SINGLE_TARGET_CHOICES + ["all_profiles"]


def unwrap_model(model: Any) -> nn.Module:
    if hasattr(model, "module"):
        return model.module
    return model


def is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def dist_rank() -> int:
    return dist.get_rank() if is_dist_ready() else 0


def dist_world_size() -> int:
    return dist.get_world_size() if is_dist_ready() else 1


def is_main_process() -> bool:
    return dist_rank() == 0


def setup_distributed(timeout_sec: int = 3600) -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=int(timeout_sec)),
    )
    return True, rank, world_size, local_rank


def reduce_mean_scalar(value: float, device: torch.device) -> float:
    if not is_dist_ready():
        return float(value)
    t = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / float(dist_world_size())
    return float(t.item())


def reduce_mean_stats(stats: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not is_dist_ready():
        return stats
    return {k: reduce_mean_scalar(v, device) for k, v in stats.items()}


def cleanup_distributed() -> None:
    if is_dist_ready():
        dist.destroy_process_group()


def dist_barrier(device: Optional[torch.device] = None) -> None:
    if not is_dist_ready():
        return
    if dist.get_backend() == "nccl" and device is not None and device.type == "cuda":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


def broadcast_module_state(module: nn.Module, src: int = 0) -> None:
    if not is_dist_ready():
        return
    for _, tensor in module.state_dict().items():
        if torch.is_tensor(tensor):
            dist.broadcast(tensor, src=src)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class RunTask:
    run_id: int
    x: np.ndarray
    y: np.ndarray
    n_halo: int
    n_r: int


class CAMELSRunTaskDataset(Dataset):
    def __init__(self, tasks: List[RunTask]):
        self.tasks = tasks

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        t = self.tasks[idx]
        return t.x, t.y, t.run_id


class MeanModel(nn.Module):
    def __init__(self, y_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, y_dim),
        )

    def forward(self, log_m: torch.Tensor, log_r: torch.Tensor) -> torch.Tensor:
        return self.net(torch.stack([log_m, log_r], dim=-1))


def _flatten_mean_training_data(tasks: List[RunTask]) -> Tuple[np.ndarray, np.ndarray]:
    x_mr = np.concatenate([t.x[:, :, :2].reshape(-1, 2) for t in tasks], axis=0).astype(np.float32)
    y = np.concatenate([t.y.reshape(-1, t.y.shape[-1]) for t in tasks], axis=0).astype(np.float32)
    return x_mr, y


def train_mean_model(
    tasks: List[RunTask],
    y_dim: int,
    args,
    device: torch.device,
    verbose: bool = True,
    num_workers_override: Optional[int] = None,
) -> MeanModel:
    x_np, y_np = _flatten_mean_training_data(tasks)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    # Balance channels with very different scales (critical for all_profiles).
    # Without this, large-amplitude targets dominate the mean-prior fit and
    # small-amplitude channels (e.g., pressure, gas_density) can degrade.
    y_scale_np = np.std(y_np, axis=0).astype(np.float32)
    y_scale_np = np.where(y_scale_np < 1e-6, 1.0, y_scale_np).astype(np.float32)
    y_scale = torch.from_numpy(y_scale_np).to(device)

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=args.mean_batch_size,
        shuffle=True,
        num_workers=args.num_workers if num_workers_override is None else int(num_workers_override),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = MeanModel(y_dim=y_dim, hidden_dim=args.mean_hidden_dim).to(device)
    opt = torch.optim.AdamW(list(model.parameters()), lr=args.mean_lr, weight_decay=args.mean_weight_decay)  # type: ignore[arg-type]
    loss_fn = nn.MSELoss()

    for epoch in range(args.mean_epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(xb[:, 0], xb[:, 1])
            # Channel-balanced MSE in normalized target space.
            loss = loss_fn((pred - yb) / y_scale.view(1, -1), torch.zeros_like(yb))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))

        if verbose and ((epoch + 1) % max(1, args.mean_log_every) == 0 or epoch == 0 or (epoch + 1) == args.mean_epochs):
            rmse = math.sqrt(max(0.0, float(np.mean(losses)))) if losses else float("nan")
            print(f"Mean model epoch {epoch+1:03d}/{args.mean_epochs:03d}: RMSE={rmse:.6f}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def predict_mean_from_raw_x(raw_x: torch.Tensor, mean_model: MeanModel) -> torch.Tensor:
    return mean_model(raw_x[..., 0], raw_x[..., 1])


@torch.no_grad()
def apply_residual_prior(tasks: List[RunTask], mean_model: MeanModel, device: torch.device, batch_size: int) -> List[RunTask]:
    out: List[RunTask] = []
    for t in tasks:
        flat_x = torch.from_numpy(t.x[:, :, :2].reshape(-1, 2)).to(device)
        preds = []
        for i in range(0, flat_x.shape[0], batch_size):
            xb = flat_x[i : i + batch_size]
            preds.append(mean_model(xb[:, 0], xb[:, 1]).detach().cpu())
        mean_y = torch.cat(preds, dim=0).numpy().reshape(t.y.shape).astype(np.float32)
        y_resid = (t.y - mean_y).astype(np.float32)
        out.append(RunTask(run_id=t.run_id, x=t.x, y=y_resid, n_halo=t.n_halo, n_r=t.n_r))
    return out


@torch.no_grad()
def add_mean_back(
    y_resid: torch.Tensor,
    tgt_x_norm: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    mean_model: Optional[MeanModel],
) -> torch.Tensor:
    if mean_model is None:
        return y_resid
    x_raw = tgt_x_norm * x_std.view(1, 1, -1) + x_mean.view(1, 1, -1)
    mean_y = predict_mean_from_raw_x(x_raw, mean_model)
    return y_resid + mean_y


def get_profile_filename(run: int, suite: str, sim_set: str, snapnum: int) -> str:
    return f"{suite}_{sim_set}_{run}_snap{snapnum:03d}.npz"


def resolve_profile_file(run: int, base_path: Path, suite: str, sim_set: str, snapnum: int) -> Path:
    name = get_profile_filename(run, suite=suite, sim_set=sim_set, snapnum=snapnum)
    candidates = [
        base_path / name,
        base_path / sim_set / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Run {run} profile not found. Tried: {candidates}")


def discover_runs(base_path: Path, suite: str, sim_set: str, snapnum: int) -> List[int]:
    patterns = [
        f"{suite}_{sim_set}_*_snap{snapnum:03d}.npz",
        f"{sim_set}/{suite}_{sim_set}_*_snap{snapnum:03d}.npz",
    ]
    out = set()
    for pat in patterns:
        for p in base_path.glob(pat):
            token = p.stem.split("_")[-2]
            if token.isdigit():
                out.add(int(token))
    return sorted(out)


def load_theta_table(param_csv: Path, target_theta_dim: int = 35) -> Dict[int, np.ndarray]:
    if not param_csv.exists():
        raise FileNotFoundError(f"Parameter CSV not found: {param_csv}")

    arr = np.loadtxt(param_csv, delimiter=",", skiprows=1, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != target_theta_dim:
        raise ValueError(
            f"Expected {target_theta_dim} columns in parameter CSV, got {arr.shape[1]} from {param_csv}"
        )

    return {int(i): arr[i] for i in range(arr.shape[0])}


def select_target(
    npz_data,
    target_name: str,
    mu_e: float,
    mp: float,
    eps: float,
    all_profile_targets: Optional[List[str]] = None,
) -> np.ndarray:
    rho = npz_data["gas_density_array"].astype(np.float32)
    temp = npz_data["temperature_array"].astype(np.float32)

    target_map = {
        "potential": npz_data["potential_array"].astype(np.float32),
        "DM_density": npz_data["DM_density_array"].astype(np.float32),
        "stellar_density": npz_data["stellar_density_array"].astype(np.float32),
        "gas_density": npz_data["gas_density_array"].astype(np.float32),
        "temperature": npz_data["temperature_array"].astype(np.float32),
        "pressure": npz_data["pressure_array"].astype(np.float32),
        "metallicity": npz_data["metallicity_array"].astype(np.float32),
        "radial_gas_velocity": npz_data["radial_gas_velocity_array"].astype(np.float32),
        "radial_gas_velocity_dispersion": npz_data["radial_gas_velocity_dispersion_array"].astype(np.float32),
        "rotational_gas_velocity": npz_data["rotational_gas_velocity_array"].astype(np.float32),
        "gas_velocity_dispersion": npz_data["gas_velocity_dispersion_array"].astype(np.float32),
        "hot_gas_density": npz_data["hot_gas_density_array"].astype(np.float32),
        "hot_temperature": npz_data["hot_temperature_array"].astype(np.float32),
        "hot_pressure": npz_data["hot_pressure_array"].astype(np.float32),
        "hot_metallicity": npz_data["hot_metallicity_array"].astype(np.float32),
        "hot_radial_gas_velocity": npz_data["hot_radial_gas_velocity_array"].astype(np.float32),
        "hot_radial_gas_velocity_dispersion": npz_data["hot_radial_gas_velocity_dispersion_array"].astype(np.float32),
        "hot_rotational_gas_velocity": npz_data["hot_rotational_gas_velocity_array"].astype(np.float32),
        "hot_gas_velocity_dispersion": npz_data["hot_gas_velocity_dispersion_array"].astype(np.float32),
        "log_pressure": np.log10(np.clip(npz_data["pressure_array"].astype(np.float32), eps, None)),
        "log_gas_density": np.log10(np.clip(rho, eps, None)),
        "log_temperature": np.log10(np.clip(temp, eps, None)),
        "log_metallicity": np.log10(np.clip(npz_data["metallicity_array"].astype(np.float32), eps, None)),
    }

    if target_name == "xsb_proxy":
        n_e = rho / (mu_e * mp)
        sx_proxy = np.clip(n_e**2 * np.sqrt(np.clip(temp, 1e-6, None)), eps, None)
        return np.log10(sx_proxy).astype(np.float32)

    if target_name == "all_profiles":
        selected_targets = all_profile_targets if all_profile_targets else ALL_PROFILE_TARGETS
        channels = []
        for name in selected_targets:
            arr = target_map[name]
            if name in ALL_PROFILE_LOG_TARGETS:
                arr = np.log10(np.clip(arr, eps, None))
            channels.append(arr)
        stacked = np.stack(channels, axis=-1)
        return stacked.astype(np.float32)

    if target_name not in target_map:
        raise ValueError(f"Unsupported target {target_name}; choose from {TARGET_CHOICES}")

    return target_map[target_name]


def build_tasks(args) -> List[RunTask]:
    mu_e = 2.0 / (1.0 + 0.76)
    mp = 1.67e-24

    base_path = Path(args.profiles_base)
    theta_by_run = load_theta_table(Path(args.param_csv), target_theta_dim=args.theta_dim)
    runs = discover_runs(base_path, suite=args.suite, sim_set=args.sim_set, snapnum=args.snapnum)
    if args.max_runs > 0:
        runs = runs[: args.max_runs]

    tasks: List[RunTask] = []
    skipped = 0
    skip_reasons = 0

    for run in runs:
        if run not in theta_by_run:
            skipped += 1
            continue

        try:
            fpath = resolve_profile_file(run, base_path=base_path, suite=args.suite, sim_set=args.sim_set, snapnum=args.snapnum)
            with np.load(fpath) as data:
                m500c = data["M500c"].astype(np.float32)
                r500c = data["R500c"].astype(np.float32)
                r = data["radial_bins"].astype(np.float32)
                y = select_target(
                    data,
                    target_name=args.target_name,
                    mu_e=mu_e,
                    mp=mp,
                    eps=args.eps,
                    all_profile_targets=getattr(args, "resolved_all_profile_targets", None),
                )

            if y.ndim == 2:
                y = y[..., None]
            if y.ndim != 3:
                raise ValueError(f"Expected target with ndim 2 or 3; got shape {y.shape}")

            if args.radial_stride > 1:
                r = r[:: args.radial_stride]
                y = y[:, :: args.radial_stride, :]

            if args.max_halos_per_run > 0 and m500c.shape[0] > args.max_halos_per_run:
                rng = np.random.default_rng(args.seed + run)
                pick = np.sort(rng.choice(m500c.shape[0], size=args.max_halos_per_run, replace=False))
                m500c = m500c[pick]
                r500c = r500c[pick]
                y = y[pick]

            n_halo, n_r, _ = y.shape
            if n_halo < args.min_halos:
                skipped += 1
                continue

            log_m = np.log10(np.clip(m500c, 1e10, None))[:, None]
            r500_for_ratio = r500c * float(args.r500_physical_factor)
            log_r_scaled = np.log10(np.clip(r[None, :] / r500_for_ratio[:, None], 1e-4, None))

            x = np.zeros((n_halo, n_r, args.theta_start_idx + args.theta_dim), dtype=np.float32)
            x[..., 0] = log_m
            x[..., 1] = log_r_scaled
            x[..., args.theta_start_idx : args.theta_start_idx + args.theta_dim] = theta_by_run[run][None, None, :]

            tasks.append(RunTask(run_id=run, x=x, y=y.astype(np.float32), n_halo=n_halo, n_r=n_r))
        except Exception as e:
            skipped += 1
            # Show a few concrete failure reasons to avoid silent all-run skips.
            if skip_reasons < 5:
                print(f"[build_tasks] Skipping run {run}: {type(e).__name__}: {e}")
                skip_reasons += 1

    print(f"Built {len(tasks)} tasks from {len(runs)} discovered runs (skipped {skipped}).")
    if tasks:
        print(f"Example task shape x={tasks[0].x.shape}, y={tasks[0].y.shape}")
    return tasks


def split_tasks(tasks: List[RunTask], train_frac: float, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(tasks))
    n_train = int(len(tasks) * train_frac)
    n_val = int(len(tasks) * val_frac)
    tr = [tasks[i] for i in idx[:n_train]]
    va = [tasks[i] for i in idx[n_train : n_train + n_val]]
    te = [tasks[i] for i in idx[n_train + n_val :]]
    return tr, va, te


def compute_norm_stats(train_tasks: List[RunTask], eps: float = 1e-6):
    x_stack = np.concatenate([t.x.reshape(-1, t.x.shape[-1]) for t in train_tasks], axis=0)
    y_stack = np.concatenate([t.y.reshape(-1, t.y.shape[-1]) for t in train_tasks], axis=0)

    x_mean = x_stack.mean(axis=0).astype(np.float32)
    x_std = x_stack.std(axis=0).astype(np.float32)
    x_std = np.where(x_std < eps, 1.0, x_std).astype(np.float32)

    y_mean = y_stack.mean(axis=0).astype(np.float32)
    y_std = y_stack.std(axis=0).astype(np.float32)
    y_std = np.where(y_std < eps, 1.0, y_std).astype(np.float32)

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def normalize_tasks(tasks: List[RunTask], stats) -> List[RunTask]:
    out: List[RunTask] = []
    for t in tasks:
        x = ((t.x - stats["x_mean"][None, None, :]) / stats["x_std"][None, None, :]).astype(np.float32)
        y = ((t.y - stats["y_mean"][None, None, :]) / stats["y_std"][None, None, :]).astype(np.float32)
        out.append(RunTask(run_id=t.run_id, x=x, y=y, n_halo=t.n_halo, n_r=t.n_r))
    return out


def anp_collate(batch):
    ctx_x_list, ctx_y_list, tgt_x_list, tgt_y_list = [], [], [], []
    ctx_mask_list, tgt_mask_list = [], []
    meta = []

    for x_np, y_np, run_id in batch:
        x = torch.tensor(x_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32)

        n_halo, n_r, xdim = x.shape
        ydim = y.shape[-1]
        n_c = int(np.exp(np.random.uniform(np.log(1), np.log(n_halo))))
        n_c = max(1, min(n_c, n_halo - 1))

        perm = torch.randperm(n_halo)
        ctx_h = perm[:n_c]
        tgt_h = perm

        ctx_x = x[ctx_h].reshape(-1, xdim)
        ctx_y = y[ctx_h].reshape(-1, ydim)
        tgt_x = x[tgt_h].reshape(-1, xdim)
        tgt_y = y[tgt_h].reshape(-1, ydim)

        ctx_x_list.append(ctx_x)
        ctx_y_list.append(ctx_y)
        tgt_x_list.append(tgt_x)
        tgt_y_list.append(tgt_y)

        ctx_mask_list.append(torch.ones(ctx_x.shape[0], dtype=torch.bool))
        tgt_mask_list.append(torch.ones(tgt_x.shape[0], dtype=torch.bool))

        meta.append({"run_id": int(run_id), "n_halo": int(n_halo), "n_r": int(n_r), "n_c": int(n_c)})

    def pad_2d(seq, pad_val=0.0):
        b = len(seq)
        max_len = max(t.shape[0] for t in seq)
        feat = seq[0].shape[1]
        out = torch.full((b, max_len, feat), pad_val, dtype=seq[0].dtype)
        for i, t in enumerate(seq):
            out[i, : t.shape[0]] = t
        return out

    def pad_1d(seq, pad_val=0.0):
        b = len(seq)
        max_len = max(t.shape[0] for t in seq)
        out = torch.full((b, max_len), pad_val, dtype=seq[0].dtype)
        for i, t in enumerate(seq):
            out[i, : t.shape[0]] = t
        return out

    def pad_mask(seq):
        b = len(seq)
        max_len = max(t.shape[0] for t in seq)
        out = torch.zeros((b, max_len), dtype=torch.bool)
        for i, t in enumerate(seq):
            out[i, : t.shape[0]] = t
        return out

    return {
        "ctx_x": pad_2d(ctx_x_list),
        "ctx_y": pad_2d(ctx_y_list),
        "tgt_x": pad_2d(tgt_x_list),
        "tgt_y": pad_2d(tgt_y_list),
        "ctx_mask": pad_mask(ctx_mask_list),
        "tgt_mask": pad_mask(tgt_mask_list),
        "meta": meta,
    }


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8):
    w = mask.float().unsqueeze(-1)
    s = (x * w).sum(dim=dim)
    c = w.sum(dim=dim).clamp_min(eps)
    return s / c


class PreNormResidual(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return x + self.fn(self.norm(x), *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(dim, mult=4, dropout=dropout)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + a
        x = x + self.ff(self.n2(x))
        return x


class DeepMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FourierEmbedding(nn.Module):
    def __init__(self, n_freq: int = 16, scale: float = 1.0):
        super().__init__()
        if n_freq <= 0:
            raise ValueError("n_freq must be > 0 for FourierEmbedding")
        self.n_freq = int(n_freq)
        self.register_buffer("B", torch.randn(1, self.n_freq) * float(scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x * self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class LatentEncoder(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, d_model: int, d_latent: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.point = DeepMLP(x_dim + y_dim, hidden_dim=d_model, out_dim=d_model, n_layers=2, dropout=dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])
        self.mu = nn.Linear(d_model, d_latent)
        self.log_sigma = nn.Linear(d_model, d_latent)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> Normal:
        h = self.point(torch.cat([x, y], dim=-1))
        key_padding_mask = ~mask
        for blk in self.blocks:
            h = blk(h, key_padding_mask=key_padding_mask)
        pooled = masked_mean(h, mask, dim=1)
        mu = self.mu(pooled)
        sigma = 0.1 + 0.9 * F.softplus(self.log_sigma(pooled))
        return Normal(mu, sigma)


class DeterministicPath(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, d_model: int, n_heads: int, n_ctx_layers: int, dropout: float):
        super().__init__()
        self.ctx_point = DeepMLP(x_dim + y_dim, hidden_dim=d_model, out_dim=d_model, n_layers=2, dropout=dropout)
        self.q_proj = DeepMLP(x_dim, hidden_dim=d_model, out_dim=d_model, n_layers=2, dropout=dropout)
        self.ctx_blocks = nn.ModuleList([TransformerBlock(d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_ctx_layers)])
        self.cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.post = DeepMLP(d_model, hidden_dim=d_model, out_dim=d_model, n_layers=2, dropout=dropout)

    def forward(self, ctx_x, ctx_y, ctx_mask, tgt_x, tgt_mask):
        v = self.ctx_point(torch.cat([ctx_x, ctx_y], dim=-1))
        k = self.q_proj(ctx_x)
        q = self.q_proj(tgt_x)
        key_padding_mask = ~ctx_mask
        for blk in self.ctx_blocks:
            v = blk(v, key_padding_mask=key_padding_mask)
        r, _ = self.cross(q, k, v, key_padding_mask=key_padding_mask)
        r = self.post(r)
        return r * tgt_mask.unsqueeze(-1).float()


class Decoder(nn.Module):
    def __init__(
        self,
        x_dim: int,
        d_model: int,
        d_latent: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        y_dim: int,
        theta_dim: int,
        theta_start_idx: int,
        theta_film_scale: float,
    ):
        super().__init__()
        self.y_dim = y_dim
        self.theta_dim = theta_dim
        self.theta_start_idx = theta_start_idx
        self.theta_film_scale = theta_film_scale
        self.trunk = DeepMLP(x_dim + d_model + d_latent, hidden_dim=hidden_dim, out_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
        self.theta_film = nn.Sequential(
            nn.Linear(theta_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.mu = nn.Linear(hidden_dim, y_dim)
        self.log_sigma = nn.Linear(hidden_dim, y_dim)

    def forward(self, tgt_x, r, z):
        nt = tgt_x.shape[1]
        z_exp = z.unsqueeze(1).expand(-1, nt, -1)
        h = self.trunk(torch.cat([tgt_x, r, z_exp], dim=-1))

        # Condition decoder features directly on run-level theta via FiLM.
        theta = tgt_x[:, 0, self.theta_start_idx : self.theta_start_idx + self.theta_dim]
        film = self.theta_film(theta)
        gamma, beta = film.chunk(2, dim=-1)
        h = h * (1.0 + self.theta_film_scale * torch.tanh(gamma).unsqueeze(1))
        h = h + self.theta_film_scale * beta.unsqueeze(1)

        mu = self.mu(h)
        sigma = 0.1 + 0.9 * F.softplus(self.log_sigma(h))
        return mu, sigma


class StrongANP(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        d_model: int,
        d_latent: int,
        n_heads: int,
        n_latent_layers: int,
        n_ctx_layers: int,
        dec_hidden: int,
        dec_layers: int,
        dropout: float,
        max_latent_points: int = 4096,
        raw_x_dim: Optional[int] = None,
        radial_feature_idx: int = 1,
        radius_fourier_n_freq: int = 16,
        radius_fourier_scale: float = 1.0,
        radius_fourier_include_raw_radius: bool = True,
        theta_start_idx: int = 2,
        theta_film_scale: float = 0.1,
        smoothness_weight: float = 0.0,
        var_cal_weight: float = 0.0,
        sigma_shrink_weight: float = 0.0,
        use_task_uncertainty_weighting: bool = False,
        task_uncertainty_l2_weight: float = 0.0,
        task_uncertainty_clip: float = 0.0,
        channel_balance_loss: bool = False,
        channel_balance_alpha: float = 1.0,
        channel_balance_eps: float = 1e-6,
        core_radius_weight: float = 1.0,
        core_radius_frac: float = 0.2,
        core_radius_min_bins: int = 3,
    ):
        super().__init__()

        self.radius_fourier_n_freq = int(radius_fourier_n_freq)
        self.radial_feature_idx = int(radial_feature_idx)
        self.radius_fourier_include_raw_radius = bool(radius_fourier_include_raw_radius)
        self.radius_fourier_extra_dim = 0
        self.radius_fourier: Optional[FourierEmbedding] = None
        if self.radius_fourier_n_freq > 0:
            self.radius_fourier = FourierEmbedding(n_freq=self.radius_fourier_n_freq, scale=radius_fourier_scale)
            # New layout keeps raw radius and appends Fourier features.
            # Legacy layout replaces raw radius with Fourier features.
            self.radius_fourier_extra_dim = 2 * self.radius_fourier_n_freq
            if not self.radius_fourier_include_raw_radius:
                self.radius_fourier_extra_dim -= 1

        if raw_x_dim is None:
            raw_x_dim = x_dim - self.radius_fourier_extra_dim
        self.raw_x_dim = int(raw_x_dim)
        if self.raw_x_dim + self.radius_fourier_extra_dim != x_dim:
            raise ValueError(
                f"Inconsistent x dimensions: raw_x_dim={self.raw_x_dim}, "
                f"radius_fourier_extra_dim={self.radius_fourier_extra_dim}, x_dim={x_dim}"
            )
        if not (0 <= self.radial_feature_idx < self.raw_x_dim):
            raise ValueError(f"radial_feature_idx={self.radial_feature_idx} out of bounds for raw_x_dim={self.raw_x_dim}")

        theta_start_idx_model = int(theta_start_idx)
        if self.radius_fourier is not None and self.radial_feature_idx < theta_start_idx_model:
            theta_start_idx_model += self.radius_fourier_extra_dim

        theta_dim = max(1, x_dim - theta_start_idx_model)
        self.latent = LatentEncoder(x_dim, y_dim, d_model, d_latent, n_heads, n_latent_layers, dropout)
        self.det = DeterministicPath(x_dim, y_dim, d_model, n_heads, n_ctx_layers, dropout)
        self.dec = Decoder(
            x_dim,
            d_model,
            d_latent,
            hidden_dim=dec_hidden,
            n_layers=dec_layers,
            dropout=dropout,
            y_dim=y_dim,
            theta_dim=theta_dim,
            theta_start_idx=theta_start_idx_model,
            theta_film_scale=theta_film_scale,
        )
        self.max_latent_points = max_latent_points
        self.smoothness_weight = smoothness_weight
        self.var_cal_weight = var_cal_weight
        # Kept for backward compatibility with older checkpoints/configs.
        self.sigma_shrink_weight = sigma_shrink_weight
        self.use_task_uncertainty_weighting = bool(use_task_uncertainty_weighting)
        self.task_uncertainty_l2_weight = float(task_uncertainty_l2_weight)
        self.task_uncertainty_clip = float(task_uncertainty_clip)
        self.channel_balance_loss = bool(channel_balance_loss)
        self.channel_balance_alpha = float(channel_balance_alpha)
        self.channel_balance_eps = float(channel_balance_eps)
        self.core_radius_weight = float(core_radius_weight)
        self.core_radius_frac = float(core_radius_frac)
        self.core_radius_min_bins = int(core_radius_min_bins)
        if self.use_task_uncertainty_weighting:
            # Kendall et al. (2018): one homoscedastic log-uncertainty per output channel.
            self.log_sigma_task = nn.Parameter(torch.zeros(y_dim))
        else:
            self.log_sigma_task = None

    def _build_radius_weights(self, tgt_mask: torch.Tensor, meta: List[Dict]) -> torch.Tensor:
        # Emphasize inner radial bins where profile interiors are systematically harder.
        bsz, max_pts = tgt_mask.shape
        w = torch.ones((bsz, max_pts), dtype=torch.float32, device=tgt_mask.device)

        if self.core_radius_weight <= 1.0 or self.core_radius_frac <= 0.0:
            return w

        for b in range(bsz):
            n_halo = int(meta[b]["n_halo"])
            n_r = int(meta[b]["n_r"])
            n_pts = int(n_halo * n_r)
            if n_halo <= 0 or n_r <= 0 or n_pts <= 0:
                continue

            n_core = max(self.core_radius_min_bins, int(math.ceil(self.core_radius_frac * n_r)))
            n_core = min(max(0, n_core), n_r)
            if n_core <= 0:
                continue

            radial_idx = torch.arange(n_r, device=tgt_mask.device)
            core_mask_1h = radial_idx < n_core
            core_mask = core_mask_1h.repeat(n_halo)
            w[b, :n_pts] = torch.where(core_mask, torch.tensor(self.core_radius_weight, device=tgt_mask.device), w[b, :n_pts])

        return w

    def _embed_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.radius_fourier is None:
            return x
        if x.shape[-1] != self.raw_x_dim:
            raise ValueError(f"Expected raw x with dim={self.raw_x_dim}, got {x.shape[-1]}")

        ridx = self.radial_feature_idx
        x_left = x[..., :ridx]
        x_rad = x[..., ridx : ridx + 1]
        x_right = x[..., ridx + 1 :]
        x_rad_ff = self.radius_fourier(x_rad)
        if self.radius_fourier_include_raw_radius:
            return torch.cat([x_left, x_rad, x_rad_ff, x_right], dim=-1)
        return torch.cat([x_left, x_rad_ff, x_right], dim=-1)

    @staticmethod
    def _subsample_masked_points(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, max_points: int):
        if max_points <= 0 or x.shape[1] <= max_points:
            return x, y, mask

        bsz, _, xdim = x.shape
        x_out = torch.zeros((bsz, max_points, xdim), dtype=x.dtype, device=x.device)
        y_out = torch.zeros((bsz, max_points, y.shape[-1]), dtype=y.dtype, device=y.device)
        m_out = torch.zeros((bsz, max_points), dtype=mask.dtype, device=mask.device)

        for b in range(bsz):
            valid = torch.where(mask[b])[0]
            if valid.numel() <= max_points:
                sel = valid
            else:
                perm = torch.randperm(valid.numel(), device=valid.device)
                sel = valid[perm[:max_points]]
            nsel = sel.numel()
            if nsel > 0:
                x_out[b, :nsel] = x[b, sel]
                y_out[b, :nsel] = y[b, sel]
                m_out[b, :nsel] = True

        return x_out, y_out, m_out

    @staticmethod
    def _profile_smoothness_penalty(mu: torch.Tensor, mask: torch.Tensor, meta: List[Dict]) -> torch.Tensor:
        penalties = []
        for b in range(mu.shape[0]):
            n_halo = int(meta[b]["n_halo"])
            n_r = int(meta[b]["n_r"])
            n_pts = n_halo * n_r
            if n_r < 3:
                continue
            valid = mask[b, :n_pts]
            yb = mu[b, :n_pts]
            if valid.sum() < 3:
                continue
            y2d = yb.reshape(n_halo, n_r, -1)
            d2 = y2d[:, 2:] - 2.0 * y2d[:, 1:-1] + y2d[:, :-2]
            penalties.append((d2**2).mean())

        if not penalties:
            return torch.zeros((), device=mu.device, dtype=mu.dtype)
        return torch.stack(penalties).mean()

    def forward(self, batch, device, beta: float = 1.0):
        ctx_x_raw = batch["ctx_x"].to(device)
        ctx_y = batch["ctx_y"].to(device)
        tgt_x_raw = batch["tgt_x"].to(device)
        tgt_y = batch["tgt_y"].to(device)
        ctx_mask = batch["ctx_mask"].to(device)
        tgt_mask = batch["tgt_mask"].to(device)

        ctx_x = self._embed_x(ctx_x_raw)
        tgt_x = self._embed_x(tgt_x_raw)

        q_ctx = self.latent(ctx_x, ctx_y, ctx_mask)
        lat_x, lat_y, lat_mask = self._subsample_masked_points(
            tgt_x,
            tgt_y,
            tgt_mask,
            max_points=self.max_latent_points,
        )
        q_all = self.latent(lat_x, lat_y, lat_mask)
        z = q_all.rsample()

        r = self.det(ctx_x, ctx_y, ctx_mask, tgt_x, tgt_mask)
        mu, sigma = self.dec(tgt_x, r, z)

        mask_f = tgt_mask.float()
        mask_3d = mask_f.unsqueeze(-1)
        radius_w = self._build_radius_weights(tgt_mask, batch["meta"]).unsqueeze(-1).to(mask_3d.dtype)
        weighted_mask_3d = mask_3d * radius_w
        denom = weighted_mask_3d.sum().clamp_min(1.0)

        # Optional dynamic channel balancing to avoid multi-output collapse where
        # one channel dominates optimization and context usage degrades.
        channel_w = None
        if self.channel_balance_loss and mu.shape[-1] > 1:
            ch_denom = mask_3d.sum(dim=(0, 1)).clamp_min(1.0)
            resid_sq_det = ((mu - tgt_y).detach() ** 2)
            ch_rmse = torch.sqrt((resid_sq_det * mask_3d).sum(dim=(0, 1)) / ch_denom).clamp_min(self.channel_balance_eps)
            inv = ch_rmse.pow(-self.channel_balance_alpha)
            channel_w = (inv / inv.mean().clamp_min(1e-12)).view(1, 1, -1)
        else:
            channel_w = torch.ones((1, 1, mu.shape[-1]), dtype=mu.dtype, device=mu.device)

        dist = Normal(mu, sigma)
        log_prob = dist.log_prob(tgt_y)
        recon = (log_prob * channel_w * weighted_mask_3d).sum() / denom

        resid_sq = (mu - tgt_y) ** 2

        if self.use_task_uncertainty_weighting and self.log_sigma_task is not None:
            log_sigma_task = self.log_sigma_task
            if self.task_uncertainty_clip > 0.0:
                clipv = float(self.task_uncertainty_clip)
                log_sigma_task = torch.clamp(log_sigma_task, min=-clipv, max=clipv)
            sigma_t_sq = log_sigma_task.exp().pow(2)
            # Keep ELBO reconstruction unmodified; apply optional task weighting to MSE only.
            task_mse = 0.5 * ((resid_sq / (sigma_t_sq.view(1, 1, -1) + 1e-6)) * channel_w * weighted_mask_3d).sum() / denom
            task_reg = log_sigma_task.mean()
            task_l2 = (log_sigma_task**2).mean()
            task_uncertainty_loss = task_mse + task_reg + self.task_uncertainty_l2_weight * task_l2
        else:
            task_uncertainty_loss = torch.zeros((), device=mu.device, dtype=mu.dtype)

        kl = kl_divergence(q_all, q_ctx).sum(dim=1).mean()
        mse = (resid_sq * channel_w * weighted_mask_3d).sum() / denom

        # Calibrate predicted variance against residual variance at each valid point.
        var_cal_pt = (torch.log(sigma**2 + 1e-8) - torch.log(resid_sq.detach() + 1e-8)) ** 2
        var_cal = (var_cal_pt * weighted_mask_3d).sum() / denom
        sigma_mean = (sigma * mask_3d).sum() / mask_3d.sum().clamp_min(1.0)
        smooth = self._profile_smoothness_penalty(mu, tgt_mask, batch["meta"])

        loss = -(recon - beta * kl) + task_uncertainty_loss
        loss = loss + self.var_cal_weight * var_cal
        loss = loss + self.smoothness_weight * smooth

        rmse_norm = torch.sqrt(mse)
        return {
            "loss": loss,
            "recon": recon,
            "kl": kl,
            "rmse_norm": rmse_norm,
            "var_cal": var_cal,
            "sigma_mean": sigma_mean,
            "smooth": smooth,
        }

    @torch.no_grad()
    def predict(self, batch, device, n_samples: int = 30):
        self.eval()
        ctx_x_raw = batch["ctx_x"].to(device)
        ctx_y = batch["ctx_y"].to(device)
        tgt_x_raw = batch["tgt_x"].to(device)
        ctx_mask = batch["ctx_mask"].to(device)
        tgt_mask = batch["tgt_mask"].to(device)

        ctx_x = self._embed_x(ctx_x_raw)
        tgt_x = self._embed_x(tgt_x_raw)

        q_ctx = self.latent(ctx_x, ctx_y, ctx_mask)
        r = self.det(ctx_x, ctx_y, ctx_mask, tgt_x, tgt_mask)

        mus, sigs = [], []
        for _ in range(n_samples):
            z = q_ctx.rsample()
            mu, sig = self.dec(tgt_x, r, z)
            mus.append(mu)
            sigs.append(sig)

        mus = torch.stack(mus, dim=0)
        sigs = torch.stack(sigs, dim=0)

        pred_mean = mus.mean(0)
        aleatoric_var = (sigs**2).mean(0)
        epistemic_var = mus.var(0, unbiased=False)
        total_std = (aleatoric_var + epistemic_var).sqrt()

        return pred_mean, total_std, aleatoric_var.sqrt(), epistemic_var.sqrt()


def denorm_y(y, y_mean: torch.Tensor, y_std: torch.Tensor):
    return y * y_std + y_mean


def _all_profile_log_indices(target_names: List[str]) -> List[int]:
    return [i for i, name in enumerate(target_names) if name in ALL_PROFILE_LOG_TARGETS]


def restore_all_profiles_physical_units(
    y: torch.Tensor,
    mu: torch.Tensor,
    std: Optional[torch.Tensor],
    target_names: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    log_idx = _all_profile_log_indices(target_names)
    if not log_idx:
        return y, mu, std

    idx = torch.tensor(log_idx, device=mu.device, dtype=torch.long)

    y_phys = y.clone()
    mu_phys = mu.clone()
    y_phys[..., idx] = torch.pow(10.0, y[..., idx])
    mu_phys[..., idx] = torch.pow(10.0, mu[..., idx])

    if std is None:
        return y_phys, mu_phys, None

    # First-order delta method for transforming std from log10 space.
    # Use a dtype-aware tiny floor to avoid forcing an unphysical absolute
    # uncertainty scale for very small-magnitude channels.
    ln10 = math.log(10.0)
    std_phys = std.clone()
    tiny = torch.finfo(std.dtype).tiny
    std_phys[..., idx] = (ln10 * mu_phys[..., idx] * std[..., idx]).clamp_min(tiny)
    return y_phys, mu_phys, std_phys


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, kl_warmup, grad_clip, accum_steps, use_amp):
    model.train()
    beta = min(1.0, float(epoch + 1) / float(max(1, kl_warmup)))

    meter = {
        "loss": [],
        "recon": [],
        "kl": [],
        "rmse_norm": [],
        "var_cal": [],
        "sigma_mean": [],
        "smooth": [],
    }
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(batch, device=device, beta=beta)
            loss = out["loss"] / accum_steps

        scaler.scale(loss).backward()

        if ((step + 1) % accum_steps == 0) or (step + 1 == len(loader)):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        for k in meter:
            meter[k].append(float(out[k].detach().cpu()))

    return {k: float(np.mean(v)) for k, v in meter.items()}, beta


@torch.no_grad()
def validate_one_epoch(model, loader, device, epoch, kl_warmup):
    model.eval()
    beta = min(1.0, float(epoch + 1) / float(max(1, kl_warmup)))
    meter = {
        "loss": [],
        "recon": [],
        "kl": [],
        "rmse_norm": [],
        "var_cal": [],
        "sigma_mean": [],
        "smooth": [],
    }

    for batch in loader:
        out = model(batch, device=device, beta=beta)
        for k in meter:
            meter[k].append(float(out[k].detach().cpu()))

    return {k: float(np.mean(v)) for k, v in meter.items()}


@torch.no_grad()
def evaluate_test_metrics(
    model,
    loader,
    device,
    y_mean,
    y_std,
    x_mean,
    x_std,
    mean_model: Optional[MeanModel],
    n_samples: int,
    target_names: List[str],
    core_radius_frac: float = 0.2,
    core_radius_min_bins: int = 3,
):
    model.eval()
    rmses = []
    nlls = []
    sum_sq = None
    sum_nll = None
    sum_w = None
    sum_sq_core = None
    sum_w_core = None
    sum_sq_outer = None
    sum_w_outer = None

    for batch in loader:
        y = batch["tgt_y"].to(device)
        mask = batch["tgt_mask"].to(device)
        mu, std, _, _ = model.predict(batch, device=device, n_samples=n_samples)

        y_o = denorm_y(y, y_mean, y_std)
        mu_o = denorm_y(mu, y_mean, y_std)
        y_o = add_mean_back(y_o, batch["tgt_x"].to(device), x_mean, x_std, mean_model)
        mu_o = add_mean_back(mu_o, batch["tgt_x"].to(device), x_mean, x_std, mean_model)
        std_o = (std * y_std).clamp_min(1e-6)
        if len(target_names) > 1:
            y_o, mu_o, std_o = restore_all_profiles_physical_units(y_o, mu_o, std_o, target_names)

        mask_3d = mask.float().unsqueeze(-1)
        core_mask = torch.zeros_like(mask, dtype=torch.bool)
        for b in range(mask.shape[0]):
            n_halo = int(batch["meta"][b]["n_halo"])
            n_r = int(batch["meta"][b]["n_r"])
            n_pts = int(min(mask.shape[1], n_halo * n_r))
            if n_halo <= 0 or n_r <= 0 or n_pts <= 0:
                continue
            n_core = max(int(core_radius_min_bins), int(math.ceil(float(core_radius_frac) * n_r)))
            n_core = min(max(0, n_core), n_r)
            if n_core <= 0:
                continue
            core_mask_1h = (torch.arange(n_r, device=device) < n_core).repeat(n_halo)
            core_mask[b, :n_pts] = core_mask_1h[:n_pts]
        core_mask = core_mask & mask
        outer_mask = mask & (~core_mask)
        core_mask_3d = core_mask.float().unsqueeze(-1)
        outer_mask_3d = outer_mask.float().unsqueeze(-1)

        rmse = torch.sqrt((((mu_o - y_o) ** 2) * mask_3d).sum() / mask_3d.sum())
        nll = -(Normal(mu_o, std_o).log_prob(y_o) * mask_3d).sum() / mask_3d.sum()

        ss = (((mu_o - y_o) ** 2) * mask_3d).sum(dim=(0, 1))
        sn = (-(Normal(mu_o, std_o).log_prob(y_o) * mask_3d)).sum(dim=(0, 1))
        ww = mask_3d.sum(dim=(0, 1))
        ss_core = (((mu_o - y_o) ** 2) * core_mask_3d).sum(dim=(0, 1))
        ww_core = core_mask_3d.sum(dim=(0, 1))
        ss_outer = (((mu_o - y_o) ** 2) * outer_mask_3d).sum(dim=(0, 1))
        ww_outer = outer_mask_3d.sum(dim=(0, 1))
        if sum_sq is None:
            sum_sq = ss
            sum_nll = sn
            sum_w = ww
            sum_sq_core = ss_core
            sum_w_core = ww_core
            sum_sq_outer = ss_outer
            sum_w_outer = ww_outer
        else:
            sum_sq = sum_sq + ss
            sum_nll = sum_nll + sn
            sum_w = sum_w + ww
            sum_sq_core = sum_sq_core + ss_core
            sum_w_core = sum_w_core + ww_core
            sum_sq_outer = sum_sq_outer + ss_outer
            sum_w_outer = sum_w_outer + ww_outer

        rmses.append(float(rmse.cpu()))
        nlls.append(float(nll.cpu()))

    if (
        sum_sq is None
        or sum_nll is None
        or sum_w is None
        or sum_sq_core is None
        or sum_w_core is None
        or sum_sq_outer is None
        or sum_w_outer is None
    ):
        y_dim = len(target_names)
        rmse_by_target = [float("nan")] * y_dim
        nll_by_target = [float("nan")] * y_dim
        rmse_core_by_target = [float("nan")] * y_dim
        rmse_outer_by_target = [float("nan")] * y_dim
        rmse_core = float("nan")
        rmse_outer = float("nan")
    else:
        rmse_by_target = torch.sqrt(sum_sq / sum_w.clamp_min(1.0)).detach().cpu().numpy().tolist()
        nll_by_target = (sum_nll / sum_w.clamp_min(1.0)).detach().cpu().numpy().tolist()
        rmse_core_by_target = torch.sqrt(sum_sq_core / sum_w_core.clamp_min(1.0)).detach().cpu().numpy().tolist()
        rmse_outer_by_target = torch.sqrt(sum_sq_outer / sum_w_outer.clamp_min(1.0)).detach().cpu().numpy().tolist()

        tot_core_sq = float(sum_sq_core.sum().detach().cpu())
        tot_core_w = float(sum_w_core.sum().detach().cpu())
        rmse_core = float(math.sqrt(tot_core_sq / max(1.0, tot_core_w)))

        tot_outer_sq = float(sum_sq_outer.sum().detach().cpu())
        tot_outer_w = float(sum_w_outer.sum().detach().cpu())
        rmse_outer = float(math.sqrt(tot_outer_sq / max(1.0, tot_outer_w)))

    per_target = {
        name: {
            "rmse_original_units": float(rmse_by_target[i]),
            "nll_original_units": float(nll_by_target[i]),
            "rmse_core_original_units": float(rmse_core_by_target[i]),
            "rmse_outer_original_units": float(rmse_outer_by_target[i]),
        }
        for i, name in enumerate(target_names)
    }

    return {
        "rmse_original_units": float(np.mean(rmses)),
        "nll_original_units": float(np.mean(nlls)),
        "rmse_core_original_units": rmse_core,
        "rmse_outer_original_units": rmse_outer,
        "per_target": per_target,
    }


def compute_weighted_selection_score(
    detailed_metrics: Dict[str, Any],
    target_names: List[str],
    pressure_w: float,
    temperature_w: float,
    pressure_core_w: float,
    temperature_core_w: float,
) -> float:
    base = float(detailed_metrics.get("rmse_original_units", float("inf")))
    per = detailed_metrics.get("per_target", {})

    score = base
    if "pressure" in target_names and "pressure" in per:
        score += float(pressure_w) * float(per["pressure"]["rmse_original_units"])
        score += float(pressure_core_w) * float(per["pressure"].get("rmse_core_original_units", float("inf")))
    if "temperature" in target_names and "temperature" in per:
        score += float(temperature_w) * float(per["temperature"]["rmse_original_units"])
        score += float(temperature_core_w) * float(per["temperature"].get("rmse_core_original_units", float("inf")))
    return float(score)


@torch.no_grad()
def empirical_coverage(
    model,
    loader,
    device,
    y_mean,
    y_std,
    x_mean,
    x_std,
    mean_model: Optional[MeanModel],
    sigma_level: float,
    n_samples: int,
    target_names: List[str],
):
    model.eval()
    covered = 0
    total = 0
    for batch in loader:
        y = batch["tgt_y"].to(device)
        mask = batch["tgt_mask"].to(device)
        mu, std, _, _ = model.predict(batch, device=device, n_samples=n_samples)

        y_o = denorm_y(y, y_mean, y_std)
        mu_o = denorm_y(mu, y_mean, y_std)
        y_o = add_mean_back(y_o, batch["tgt_x"].to(device), x_mean, x_std, mean_model)
        mu_o = add_mean_back(mu_o, batch["tgt_x"].to(device), x_mean, x_std, mean_model)
        std_o = std * y_std

        if len(target_names) > 1:
            y_o, mu_o, std_o = restore_all_profiles_physical_units(y_o, mu_o, std_o, target_names)
        if std_o is None:
            raise RuntimeError("std tensor unexpectedly None in empirical_coverage")

        hit = (torch.abs(y_o - mu_o) <= sigma_level * std_o) & mask.unsqueeze(-1)
        covered += int(hit.sum().item())
        total += int(mask.sum().item() * y_o.shape[-1])

    return covered / max(1, total)


@torch.no_grad()
def estimate_context_sensitivity(
    model,
    loader,
    device,
    n_samples: int,
    n_batches: int,
) -> float:
    model.eval()
    deltas = []
    rels = []

    for i, batch in enumerate(loader):
        if i >= max(1, int(n_batches)):
            break

        mu_ctx, _, _, _ = model.predict(batch, device=device, n_samples=n_samples)

        b0 = {
            "ctx_x": batch["ctx_x"],
            "ctx_y": torch.zeros_like(batch["ctx_y"]),
            "tgt_x": batch["tgt_x"],
            "tgt_y": batch["tgt_y"],
            "ctx_mask": batch["ctx_mask"],
            "tgt_mask": batch["tgt_mask"],
            "meta": batch["meta"],
        }
        mu_nctx, _, _, _ = model.predict(b0, device=device, n_samples=n_samples)

        mask = batch["tgt_mask"].to(device).float().unsqueeze(-1)
        delta = ((mu_ctx - mu_nctx).abs() * mask).sum() / mask.sum().clamp_min(1.0)
        tgt_scale = (batch["tgt_y"].to(device).abs() * mask).sum() / mask.sum().clamp_min(1.0)
        deltas.append(float(delta.detach().cpu()))
        rels.append(float((delta / tgt_scale.clamp_min(1e-8)).detach().cpu()))

    if not rels:
        return float("nan")
    return float(np.mean(rels))


def set_context_halos_for_batch(batch, n_c: int):
    tgt_x = batch["tgt_x"]
    tgt_y = batch["tgt_y"]
    tgt_mask = batch["tgt_mask"]
    meta = batch["meta"]

    bsz = tgt_x.shape[0]
    ctx_x = torch.zeros_like(tgt_x)
    ctx_y = torch.zeros_like(tgt_y)
    ctx_mask = torch.zeros_like(tgt_mask)

    for b in range(bsz):
        n_halo = meta[b]["n_halo"]
        n_r = meta[b]["n_r"]
        n_c_eff = max(1, min(n_c, n_halo - 1))
        max_pts = n_halo * n_r

        perm = torch.randperm(n_halo)
        ctx_h = perm[:n_c_eff]
        idx = (ctx_h.unsqueeze(1) * n_r + torch.arange(n_r).unsqueeze(0)).reshape(-1)
        idx = idx[idx < max_pts]

        ctx_x[b, : idx.numel()] = tgt_x[b, idx]
        ctx_y[b, : idx.numel()] = tgt_y[b, idx]
        ctx_mask[b, : idx.numel()] = True

    return {
        "ctx_x": ctx_x,
        "ctx_y": ctx_y,
        "tgt_x": tgt_x,
        "tgt_y": tgt_y,
        "ctx_mask": ctx_mask,
        "tgt_mask": tgt_mask,
        "meta": meta,
    }


@torch.no_grad()
def few_shot_curve(
    model,
    loader,
    device,
    y_mean,
    y_std,
    x_mean,
    x_std,
    mean_model: Optional[MeanModel],
    n_context_list,
    n_repeats,
    n_samples,
    target_names: List[str],
):
    model.eval()
    out = {int(nc): [] for nc in n_context_list}

    for _ in range(n_repeats):
        for batch in loader:
            y = batch["tgt_y"].to(device)
            mask = batch["tgt_mask"].to(device)
            mask_3d = mask.float().unsqueeze(-1)
            y_o = denorm_y(y, y_mean, y_std)
            y_o = add_mean_back(y_o, batch["tgt_x"].to(device), x_mean, x_std, mean_model)
            if len(target_names) > 1:
                y_o, _, _ = restore_all_profiles_physical_units(y_o, y_o, None, target_names)

            for nc in n_context_list:
                b2 = set_context_halos_for_batch(batch, n_c=int(nc))
                mu, _, _, _ = model.predict(b2, device=device, n_samples=n_samples)
                mu_o = denorm_y(mu, y_mean, y_std)
                mu_o = add_mean_back(mu_o, b2["tgt_x"].to(device), x_mean, x_std, mean_model)
                if len(target_names) > 1:
                    _, mu_o, _ = restore_all_profiles_physical_units(mu_o, mu_o, None, target_names)
                rmse = torch.sqrt((((mu_o - y_o) ** 2) * mask_3d).sum() / mask_3d.sum())
                out[int(nc)].append(float(rmse.cpu()))

    return {k: float(np.mean(v)) for k, v in out.items()}


def make_arg_parser():
    p = argparse.ArgumentParser(description="Train strong ANP emulator on CAMELS profiles")

    p.add_argument("--profiles-base", type=str, default="/mnt/home/mlee1/ceph/Profiles_cy")
    p.add_argument("--param-csv", type=str, default="/mnt/home/mlee1/50Mpc_boxes/data/param_df.csv")
    p.add_argument("--output-dir", type=str, default="./anp_training_runs")

    p.add_argument("--suite", type=str, default="IllustrisTNG")
    p.add_argument("--sim-set", type=str, default="SB35")
    p.add_argument("--snapnum", type=int, default=90)
    p.add_argument("--target-name", type=str, default="log_pressure", choices=TARGET_CHOICES)
    p.add_argument(
        "--all-profiles-subset",
        type=str,
        nargs="+",
        default=None,
        choices=ALL_PROFILE_TARGETS,
        help="Optional subset of all_profiles channels to train jointly. Only used when --target-name=all_profiles.",
    )

    p.add_argument("--theta-dim", type=int, default=35)
    p.add_argument("--theta-start-idx", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eps", type=float, default=1e-30)

    p.add_argument("--max-runs", type=int, default=0, help="0 means all discovered runs")
    p.add_argument("--min-halos", type=int, default=4)
    p.add_argument("--max-halos-per-run", type=int, default=0, help="0 means keep all halos")
    p.add_argument("--radial-stride", type=int, default=1)
    p.add_argument(
        "--r500-physical-factor",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to R500c before computing r/R500. "
            "Use a/h when radial_bins are in physical kpc but R500c is comoving kpc/h."
        ),
    )

    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--kl-warmup-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation improvement required to reset patience.",
    )
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument(
        "--select-metric",
        type=str,
        default="rmse_norm",
        choices=["loss", "rmse", "rmse_norm", "weighted_orig"],
        help="Validation metric used for early stopping/checkpoint selection",
    )
    p.add_argument(
        "--val-detailed-every",
        type=int,
        default=5,
        help="Run detailed validation (original-units per-target metrics) every N epochs.",
    )
    p.add_argument(
        "--val-detailed-samples",
        type=int,
        default=10,
        help="Posterior samples used for detailed validation metrics.",
    )
    p.add_argument(
        "--selection-pressure-weight",
        type=float,
        default=0.15,
        help="Additional weight on pressure RMSE in weighted_orig selection metric.",
    )
    p.add_argument(
        "--selection-temperature-weight",
        type=float,
        default=0.15,
        help="Additional weight on temperature RMSE in weighted_orig selection metric.",
    )
    p.add_argument(
        "--selection-pressure-core-weight",
        type=float,
        default=0.0,
        help="Additional weight on core-only pressure RMSE in weighted_orig selection metric.",
    )
    p.add_argument(
        "--selection-temperature-core-weight",
        type=float,
        default=0.0,
        help="Additional weight on core-only temperature RMSE in weighted_orig selection metric.",
    )

    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--d-latent", type=int, default=128)
    p.add_argument("--radius-fourier-n-freq", type=int, default=16)
    p.add_argument("--radius-fourier-scale", type=float, default=1.0)
    p.add_argument("--disable-radius-fourier", action="store_true")
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-latent-layers", type=int, default=2)
    p.add_argument("--n-ctx-layers", type=int, default=2)
    p.add_argument("--max-latent-points", type=int, default=4096)
    p.add_argument("--dec-hidden", type=int, default=512)
    p.add_argument("--dec-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--theta-film-scale", type=float, default=0.1)
    p.add_argument("--smoothness-weight", type=float, default=0.005)
    p.add_argument("--var-cal-weight", type=float, default=0.0)
    p.add_argument(
        "--task-uncertainty-l2-weight",
        type=float,
        default=0.0,
        help="L2 regularization strength for all_profiles log_sigma_task parameters.",
    )
    p.add_argument(
        "--task-uncertainty-clip",
        type=float,
        default=0.0,
        help="If >0, clamp log_sigma_task to [-clip, clip] during forward pass.",
    )
    p.add_argument(
        "--disable-task-uncertainty-weighting",
        action="store_true",
        help="Disable homoscedastic per-channel task-uncertainty weighting for all_profiles runs.",
    )
    p.add_argument(
        "--channel-balance-loss",
        action="store_true",
        help="Dynamically rebalance multi-output reconstruction/loss terms by inverse per-channel RMSE.",
    )
    p.add_argument(
        "--channel-balance-alpha",
        type=float,
        default=1.0,
        help="Power for inverse-RMSE channel balancing weights.",
    )
    p.add_argument(
        "--channel-balance-eps",
        type=float,
        default=1e-6,
        help="Numerical floor for channel balancing RMSE.",
    )
    p.add_argument(
        "--core-radius-weight",
        type=float,
        default=1.0,
        help="Multiplicative weight for inner radial bins in reconstruction/MSE terms (1.0 disables).",
    )
    p.add_argument(
        "--core-radius-frac",
        type=float,
        default=0.2,
        help="Inner-bin fraction per halo treated as core for weighted loss.",
    )
    p.add_argument(
        "--core-radius-min-bins",
        type=int,
        default=3,
        help="Minimum number of inner bins per halo treated as core.",
    )
    p.add_argument(
        "--context-sensitivity-every",
        type=int,
        default=5,
        help="Compute context-ablation sensitivity on validation every N epochs (0 disables).",
    )
    p.add_argument(
        "--context-sensitivity-batches",
        type=int,
        default=2,
        help="Number of validation batches to estimate context sensitivity.",
    )
    p.add_argument(
        "--context-sensitivity-samples",
        type=int,
        default=4,
        help="Posterior samples used by context sensitivity estimate.",
    )
    p.add_argument("--save-every-epochs", type=int, default=20)

    p.add_argument("--disable-mean-prior", action="store_true")
    p.add_argument("--mean-hidden-dim", type=int, default=128)
    p.add_argument("--mean-epochs", type=int, default=80)
    p.add_argument("--mean-lr", type=float, default=1e-3)
    p.add_argument("--mean-weight-decay", type=float, default=1e-5)
    p.add_argument("--mean-batch-size", type=int, default=131072)
    p.add_argument("--mean-log-every", type=int, default=10)
    p.add_argument("--mean-predict-batch-size", type=int, default=262144)

    p.add_argument("--eval-samples", type=int, default=50)
    p.add_argument("--fewshot-contexts", type=int, nargs="+", default=[1, 2, 5, 10])
    p.add_argument("--fewshot-repeats", type=int, default=6)

    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--enable-data-parallel", action="store_true", help="Enable nn.DataParallel across visible GPUs (disabled by default)")
    p.add_argument("--enable-ddp", action="store_true", help="Enable DistributedDataParallel (disabled by default)")
    p.add_argument("--ddp-timeout-sec", type=int, default=3600, help="DDP process-group timeout in seconds")
    p.add_argument(
        "--ddp-num-workers",
        type=int,
        default=0,
        help="Per-rank DataLoader workers when running DDP (defaults to 0 for stability)",
    )

    return p


def main():
    args = make_arg_parser().parse_args()
    if args.theta_start_idx < 2:
        raise ValueError("theta_start_idx must be >= 2 because x[...,0] and x[...,1] are reserved for mass/radius features")
    if args.target_name == "all_profiles":
        if args.all_profiles_subset:
            # Keep canonical ALL_PROFILE_TARGETS ordering for stable channel mapping.
            selected = set(args.all_profiles_subset)
            target_names = [name for name in ALL_PROFILE_TARGETS if name in selected]
        else:
            target_names = ALL_PROFILE_TARGETS
    else:
        if args.all_profiles_subset:
            raise ValueError("--all-profiles-subset is only valid when --target-name=all_profiles")
        target_names = [args.target_name]
    args.resolved_all_profile_targets = target_names

    if args.enable_ddp:
        ddp_enabled, rank, world_size, local_rank = setup_distributed(timeout_sec=int(args.ddp_timeout_sec))
    else:
        ddp_enabled, rank, world_size, local_rank = False, 0, 1, 0
    main_proc = (rank == 0)

    set_seed(args.seed)

    if ddp_enabled and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    if main_proc:
        print(f"Device: {device}, AMP: {use_amp}, DDP: {ddp_enabled} (world_size={world_size})")
        if float(args.r500_physical_factor) != 1.0:
            print(f"[INFO] Using R500 scaling factor for radius ratio: {args.r500_physical_factor:.6g}")

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
    out_dir = out_root / f"anp_{args.target_name}_{run_tag}"
    if main_proc:
        out_dir.mkdir(parents=True, exist_ok=True)
    if ddp_enabled:
        dist_barrier(device)

    tasks = build_tasks(args)
    if len(tasks) < 20:
        raise RuntimeError(f"Too few tasks discovered ({len(tasks)}). Check file paths and filters.")

    train_raw, val_raw, test_raw = split_tasks(tasks, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed)
    if main_proc:
        print(f"Split sizes train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}")

    mean_model: Optional[MeanModel] = MeanModel(y_dim=tasks[0].y.shape[-1], hidden_dim=args.mean_hidden_dim).to(device)
    mean_prior_enabled = not args.disable_mean_prior
    if mean_prior_enabled:
        if main_proc:
            print("Pre-training frozen mean profile model for residual targets...")
        if ddp_enabled:
            if main_proc:
                mean_model = train_mean_model(
                    train_raw,
                    y_dim=tasks[0].y.shape[-1],
                    args=args,
                    device=device,
                    verbose=True,
                    num_workers_override=0,
                )
            dist_barrier(device)
            broadcast_module_state(mean_model, src=0)
            dist_barrier(device)
            mean_model.eval()
            for p in mean_model.parameters():
                p.requires_grad_(False)
        else:
            mean_model = train_mean_model(
                train_raw,
                y_dim=tasks[0].y.shape[-1],
                args=args,
                device=device,
                verbose=main_proc,
                num_workers_override=None,
            )
        train_raw = apply_residual_prior(train_raw, mean_model, device=device, batch_size=args.mean_predict_batch_size)
        val_raw = apply_residual_prior(val_raw, mean_model, device=device, batch_size=args.mean_predict_batch_size)
        test_raw = apply_residual_prior(test_raw, mean_model, device=device, batch_size=args.mean_predict_batch_size)
        if main_proc:
            print("Applied residual targets per split: y <- y - y_mean(logM, logr)")
    else:
        if main_proc:
            print("Mean profile prior disabled; training ANP on direct targets.")
        mean_model = None

    norm_np = compute_norm_stats(train_raw)
    train_tasks = normalize_tasks(train_raw, norm_np)
    val_tasks = normalize_tasks(val_raw, norm_np)
    test_tasks = normalize_tasks(test_raw, norm_np)

    y_mean = torch.tensor(norm_np["y_mean"], dtype=torch.float32, device=device)
    y_std = torch.tensor(norm_np["y_std"], dtype=torch.float32, device=device)
    x_mean = torch.tensor(norm_np["x_mean"], dtype=torch.float32, device=device)
    x_std = torch.tensor(norm_np["x_std"], dtype=torch.float32, device=device)

    train_ds = CAMELSRunTaskDataset(train_tasks)
    val_ds = CAMELSRunTaskDataset(val_tasks)
    test_ds = CAMELSRunTaskDataset(test_tasks)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp_enabled else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if ddp_enabled else None
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False) if ddp_enabled else None

    loader_workers = int(args.ddp_num_workers) if ddp_enabled else int(args.num_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=anp_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=anp_collate,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=anp_collate,
        drop_last=False,
    )

    raw_x_dim = train_tasks[0].x.shape[-1]
    radius_fourier_n_freq = 0 if args.disable_radius_fourier else int(args.radius_fourier_n_freq)
    radius_fourier_extra_dim = (2 * radius_fourier_n_freq) if radius_fourier_n_freq > 0 else 0
    x_dim = raw_x_dim + radius_fourier_extra_dim
    if main_proc:
        print(
            f"Input dimensions: raw_x_dim={raw_x_dim}, model_x_dim={x_dim}, "
            f"radius_fourier_n_freq={radius_fourier_n_freq}"
        )

    model = StrongANP(
        x_dim=x_dim,
        raw_x_dim=raw_x_dim,
        y_dim=train_tasks[0].y.shape[-1],
        d_model=args.d_model,
        d_latent=args.d_latent,
        radial_feature_idx=1,
        radius_fourier_n_freq=radius_fourier_n_freq,
        radius_fourier_scale=args.radius_fourier_scale,
        n_heads=args.n_heads,
        n_latent_layers=args.n_latent_layers,
        n_ctx_layers=args.n_ctx_layers,
        max_latent_points=args.max_latent_points,
        theta_start_idx=args.theta_start_idx,
        dec_hidden=args.dec_hidden,
        dec_layers=args.dec_layers,
        dropout=args.dropout,
        theta_film_scale=args.theta_film_scale,
        smoothness_weight=args.smoothness_weight,
        var_cal_weight=args.var_cal_weight,
        use_task_uncertainty_weighting=(
            (args.target_name == "all_profiles") and (not args.disable_task_uncertainty_weighting)
        ),
        task_uncertainty_l2_weight=args.task_uncertainty_l2_weight,
        task_uncertainty_clip=args.task_uncertainty_clip,
        channel_balance_loss=args.channel_balance_loss,
        channel_balance_alpha=args.channel_balance_alpha,
        channel_balance_eps=args.channel_balance_eps,
        core_radius_weight=args.core_radius_weight,
        core_radius_frac=args.core_radius_frac,
        core_radius_min_bins=args.core_radius_min_bins,
    ).to(device)

    use_data_parallel = bool(args.enable_data_parallel) and (not ddp_enabled) and (device.type == "cuda") and (torch.cuda.device_count() > 1)
    if use_data_parallel:
        n_gpu = torch.cuda.device_count()
        if main_proc:
            print(f"Using DataParallel across {n_gpu} GPUs")
        # Type checker can be stricter than runtime for torch module subtypes.
        # Runtime behavior is valid because StrongANP subclasses nn.Module.
        model = nn.DataParallel(model)  # type: ignore[arg-type]
    if ddp_enabled:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    opt = torch.optim.AdamW(list(model.parameters()), lr=args.lr, weight_decay=args.weight_decay)  # type: ignore[arg-type]
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    select_metric = "rmse_norm" if args.select_metric == "rmse" else args.select_metric
    sel_key = f"val_{select_metric}"
    best = {"score": float("inf"), "epoch": -1, "state": None, "val_loss": float("inf"), "val_rmse_norm": float("inf")}
    history = []
    stale = 0

    val_loader_eval = None
    if main_proc:
        val_loader_eval = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=anp_collate,
            drop_last=False,
        )

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        tr_stats, beta = train_one_epoch(
            model,
            train_loader,
            opt,
            scaler,
            device,
            epoch=epoch,
            kl_warmup=args.kl_warmup_epochs,
            grad_clip=args.grad_clip,
            accum_steps=args.accum_steps,
            use_amp=use_amp,
        )
        va_stats = validate_one_epoch(model, val_loader, device, epoch=epoch, kl_warmup=args.kl_warmup_epochs)
        tr_stats = reduce_mean_stats(tr_stats, device)
        va_stats = reduce_mean_stats(va_stats, device)
        sch.step()

        row = {
            "epoch": epoch + 1,
            "beta": beta,
            "lr": float(opt.param_groups[0]["lr"]),
            **{f"train_{k}": v for k, v in tr_stats.items()},
            **{f"val_{k}": v for k, v in va_stats.items()},
        }
        core_model = unwrap_model(model)
        if getattr(core_model, "log_sigma_task", None) is not None:
            sigma_task = core_model.log_sigma_task.detach().exp().float().cpu().numpy()
            row["task_sigma_min"] = float(np.min(sigma_task))
            row["task_sigma_mean"] = float(np.mean(sigma_task))
            row["task_sigma_max"] = float(np.max(sigma_task))
        history.append(row)

        stop_now = False
        if main_proc:
            need_detailed = (
                (select_metric == "weighted_orig")
                or (int(args.val_detailed_every) > 0 and ((epoch + 1) % int(args.val_detailed_every) == 0))
                or (epoch == 0)
            )
            detailed_metrics = None
            if need_detailed and val_loader_eval is not None:
                detailed_metrics = evaluate_test_metrics(
                    unwrap_model(model),
                    val_loader_eval,
                    device,
                    y_mean=y_mean,
                    y_std=y_std,
                    x_mean=x_mean,
                    x_std=x_std,
                    mean_model=mean_model,
                    n_samples=int(args.val_detailed_samples),
                    target_names=target_names,
                    core_radius_frac=float(args.core_radius_frac),
                    core_radius_min_bins=int(args.core_radius_min_bins),
                )
                row["val_rmse_original_units"] = float(detailed_metrics["rmse_original_units"])
                row["val_nll_original_units"] = float(detailed_metrics["nll_original_units"])
                row["val_rmse_core_original_units"] = float(detailed_metrics["rmse_core_original_units"])
                row["val_rmse_outer_original_units"] = float(detailed_metrics["rmse_outer_original_units"])

                per_target = detailed_metrics.get("per_target", {})
                if "pressure" in per_target:
                    row["val_pressure_rmse_original_units"] = float(per_target["pressure"]["rmse_original_units"])
                    row["val_pressure_nll_original_units"] = float(per_target["pressure"]["nll_original_units"])
                    row["val_pressure_rmse_core_original_units"] = float(per_target["pressure"].get("rmse_core_original_units", float("nan")))
                if "temperature" in per_target:
                    row["val_temperature_rmse_original_units"] = float(per_target["temperature"]["rmse_original_units"])
                    row["val_temperature_nll_original_units"] = float(per_target["temperature"]["nll_original_units"])
                    row["val_temperature_rmse_core_original_units"] = float(per_target["temperature"].get("rmse_core_original_units", float("nan")))

                row["val_weighted_orig"] = compute_weighted_selection_score(
                    detailed_metrics,
                    target_names=target_names,
                    pressure_w=float(args.selection_pressure_weight),
                    temperature_w=float(args.selection_temperature_weight),
                    pressure_core_w=float(args.selection_pressure_core_weight),
                    temperature_core_w=float(args.selection_temperature_core_weight),
                )

            if int(args.context_sensitivity_every) > 0 and val_loader_eval is not None:
                if ((epoch + 1) % int(args.context_sensitivity_every) == 0) or (epoch == 0):
                    ctx_sens = estimate_context_sensitivity(
                        unwrap_model(model),
                        val_loader_eval,
                        device,
                        n_samples=int(args.context_sensitivity_samples),
                        n_batches=int(args.context_sensitivity_batches),
                    )
                    row["val_context_sensitivity"] = float(ctx_sens)

            print(
                f"Epoch {epoch+1:03d} "
                f"beta={beta:.3f} "
                f"train(loss={tr_stats['loss']:.3f}, rmse={tr_stats['rmse_norm']:.3f}, sig={tr_stats['sigma_mean']:.3f}) "
                f"val(loss={va_stats['loss']:.3f}, rmse={va_stats['rmse_norm']:.3f}, sig={va_stats['sigma_mean']:.3f}, varcal={va_stats['var_cal']:.4f})"
            )
            if "val_weighted_orig" in row:
                p_rmse = row.get("val_pressure_rmse_original_units", float("nan"))
                t_rmse = row.get("val_temperature_rmse_original_units", float("nan"))
                p_core = row.get("val_pressure_rmse_core_original_units", float("nan"))
                t_core = row.get("val_temperature_rmse_core_original_units", float("nan"))
                print(
                    f"  val(orig) rmse={row['val_rmse_original_units']:.4g} nll={row['val_nll_original_units']:.4g} "
                    f"pressure_rmse={p_rmse:.4g} temperature_rmse={t_rmse:.4g} "
                    f"pressure_core_rmse={p_core:.4g} temperature_core_rmse={t_core:.4g} "
                    f"weighted={row['val_weighted_orig']:.4g}"
                )
            if "val_context_sensitivity" in row:
                print(f"  val(context_sensitivity)={row['val_context_sensitivity']:.4g}")
            if getattr(core_model, "log_sigma_task", None) is not None:
                sigma_task = core_model.log_sigma_task.detach().exp().float().cpu().numpy()
                idx_lo = np.argsort(sigma_task)[: min(3, sigma_task.shape[0])]
                idx_hi = np.argsort(sigma_task)[-min(3, sigma_task.shape[0]) :][::-1]
                lo_str = ", ".join([f"{target_names[i]}={sigma_task[i]:.3f}" for i in idx_lo])
                hi_str = ", ".join([f"{target_names[i]}={sigma_task[i]:.3f}" for i in idx_hi])
                print(
                    f"  task_sigma(min/mean/max)="
                    f"{float(np.min(sigma_task)):.3f}/{float(np.mean(sigma_task)):.3f}/{float(np.max(sigma_task)):.3f} "
                    f"low:[{lo_str}] high:[{hi_str}]"
                )

            if select_metric == "weighted_orig":
                if "val_weighted_orig" not in row:
                    raise RuntimeError(
                        "weighted_orig selection requested but detailed validation metrics were not computed"
                    )
                score = float(row["val_weighted_orig"])
            else:
                score = row.get(sel_key, va_stats["rmse_norm"])
            if score < (best["score"] - float(args.early_stop_min_delta)):
                best = {
                    "score": float(score),
                    "val_loss": va_stats["loss"],
                    "val_rmse_norm": va_stats["rmse_norm"],
                    "epoch": epoch + 1,
                    "state": {k: v.detach().cpu().clone() for k, v in unwrap_model(model).state_dict().items()},
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
                        "model_state_dict": unwrap_model(model).state_dict(),
                        "args": vars(args),
                        "norm": norm_np,
                        "epoch": epoch + 1,
                        "metrics": row,
                        "target_names": target_names,
                        "mean_prior_enabled": mean_prior_enabled,
                        "mean_model_state_dict": None if mean_model is None else mean_model.state_dict(),
                    },
                    periodic_path,
                )
                print(f"Saved periodic checkpoint: {periodic_path}")

        if ddp_enabled:
            stop_tensor = torch.tensor(1 if stop_now else 0, device=device, dtype=torch.int32)
            dist.broadcast(stop_tensor, src=0)
            stop_now = bool(stop_tensor.item())
        if stop_now:
            break

    if main_proc and best["state"] is not None:
        unwrap_model(model).load_state_dict(best["state"])
        print(
            f"Restored best checkpoint from epoch {best['epoch']} "
            f"({select_metric}={best['score']:.4f}, val_loss={best['val_loss']:.4f}, val_rmse_norm={best['val_rmse_norm']:.4f})"
        )

    eval_model = unwrap_model(model)
    if main_proc:
        # Rebuild non-distributed test loader so metrics are computed once over full test set.
        test_loader_eval = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=anp_collate,
            drop_last=False,
        )

        test_metrics = evaluate_test_metrics(
            eval_model,
            test_loader_eval,
            device,
            y_mean=y_mean,
            y_std=y_std,
            x_mean=x_mean,
            x_std=x_std,
            mean_model=mean_model,
            n_samples=args.eval_samples,
            target_names=target_names,
            core_radius_frac=float(args.core_radius_frac),
            core_radius_min_bins=int(args.core_radius_min_bins),
        )
        cov1 = empirical_coverage(
            eval_model,
            test_loader_eval,
            device,
            y_mean,
            y_std,
            x_mean,
            x_std,
            mean_model,
            sigma_level=1.0,
            n_samples=args.eval_samples,
            target_names=target_names,
        )
        cov2 = empirical_coverage(
            eval_model,
            test_loader_eval,
            device,
            y_mean,
            y_std,
            x_mean,
            x_std,
            mean_model,
            sigma_level=2.0,
            n_samples=args.eval_samples,
            target_names=target_names,
        )
        fshot = few_shot_curve(
            eval_model,
            test_loader_eval,
            device,
            y_mean,
            y_std,
            x_mean,
            x_std,
            mean_model,
            n_context_list=args.fewshot_contexts,
            n_repeats=args.fewshot_repeats,
            n_samples=args.eval_samples,
            target_names=target_names,
        )

        base = fshot[min(fshot.keys())]
        rel = {int(k): 100.0 * (base - v) / base for k, v in fshot.items()}

        print("Test metrics:", test_metrics)
        print(f"Coverage@1sigma={cov1:.3f}, Coverage@2sigma={cov2:.3f}")
        print("Few-shot RMSE:", fshot)
        print("Few-shot relative improvement (%):", rel)

        ckpt_path = out_dir / "best_model.pt"
        torch.save(
            {
                "model_state_dict": eval_model.state_dict(),
                "args": vars(args),
                "norm": norm_np,
                "best": best,
                "target_name": args.target_name,
                "target_names": target_names,
                "mean_prior_enabled": mean_prior_enabled,
                "mean_model_state_dict": None if mean_model is None else mean_model.state_dict(),
            },
            ckpt_path,
        )

        metrics = {
            "target_name": args.target_name,
            "n_tasks_total": len(tasks),
            "n_train": len(train_tasks),
            "n_val": len(val_tasks),
            "n_test": len(test_tasks),
            "select_metric": select_metric,
            "best_epoch": best["epoch"],
            "best_score": best["score"],
            "best_val_loss": best["val_loss"],
            "best_val_rmse_norm": best["val_rmse_norm"],
            "test": test_metrics,
            "coverage": {"1sigma": cov1, "2sigma": cov2},
            "fewshot_rmse": fshot,
            "fewshot_rel_improvement_pct": rel,
            "mean_prior": {
                "enabled": mean_prior_enabled,
                "hidden_dim": args.mean_hidden_dim,
                "epochs": args.mean_epochs,
                "lr": args.mean_lr,
                "weight_decay": args.mean_weight_decay,
            },
            "normalization": {
                "y_mean": norm_np["y_mean"].tolist(),
                "y_std": norm_np["y_std"].tolist(),
            },
            "target_names": target_names,
            "checkpoint": str(ckpt_path),
        }

        with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
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
