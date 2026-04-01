#!/usr/bin/env python3
"""Train an Attentive Neural Process emulator on CAMELS profile outputs.

This script is designed to run outside notebooks for full-scale GPU training.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, StudentT, kl_divergence
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
    snapnum: int
    snap_idx: int
    redshift: float
    x: np.ndarray
    y: np.ndarray
    n_halo: int
    n_r: int
    valid_mask: Optional[np.ndarray] = None  # (n_halo, n_r, y_dim) bool


@dataclass
class RunFamilyTask:
    run_id: int
    snapshots: List[RunTask]


class CAMELSRunFamilyDataset(Dataset):
    def __init__(self, families: List[RunFamilyTask]):
        self.families = families

    def __len__(self) -> int:
        return len(self.families)

    def __getitem__(self, idx: int) -> RunFamilyTask:
        return self.families[idx]


def flatten_family_tasks(families: List[RunFamilyTask]) -> List[RunTask]:
    out: List[RunTask] = []
    for fam in families:
        out.extend(fam.snapshots)
    return out


def remap_flat_tasks_to_families(families: List[RunFamilyTask], flat_tasks: List[RunTask]) -> List[RunFamilyTask]:
    out: List[RunFamilyTask] = []
    k = 0
    for fam in families:
        n = len(fam.snapshots)
        out.append(RunFamilyTask(run_id=fam.run_id, snapshots=flat_tasks[k : k + n]))
        k += n
    if k != len(flat_tasks):
        raise ValueError(f"Flat/family remap size mismatch: consumed={k}, total_flat={len(flat_tasks)}")
    return out


def parse_snapshot_redshifts(mapping_text: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    chunks = [c.strip() for c in str(mapping_text).split(",") if c.strip()]
    for chunk in chunks:
        snap_str, sep, z_str = chunk.partition(":")
        if sep == "":
            raise ValueError(
                "Invalid --snapshot-redshifts entry "
                f"'{chunk}'. Expected format snap:z, e.g. '90:0,74:0.5'."
            )
        out[int(snap_str.strip())] = float(z_str.strip())
    return out


class MeanModel(nn.Module):
    def __init__(self, y_dim: int, hidden_dim: int = 128, use_redshift: bool = False,
                 theta_dim: int = 0, theta_start_idx: int = 2, n_hidden_layers: int = 2):
        super().__init__()
        in_dim = 2 + theta_dim + (1 if use_redshift else 0)
        self.use_redshift = use_redshift
        self.theta_dim = theta_dim
        self.theta_start_idx = theta_start_idx
        layers: List[nn.Module] = []
        prev = in_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(prev, hidden_dim), nn.SiLU()]
            prev = hidden_dim
        layers.append(nn.Linear(prev, y_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, log_m: torch.Tensor, log_r: torch.Tensor,
                theta: Optional[torch.Tensor] = None,
                redshift: Optional[torch.Tensor] = None) -> torch.Tensor:
        parts = [log_m.unsqueeze(-1), log_r.unsqueeze(-1)]
        if self.theta_dim > 0 and theta is not None:
            if theta.dim() < 2:
                theta = theta.unsqueeze(-1)
            parts.append(theta)
        if self.use_redshift and redshift is not None:
            parts.append(redshift.unsqueeze(-1))
        return self.net(torch.cat(parts, dim=-1))


class PerSnapshotMeanModel(nn.Module):
    """Wrapper holding a separate MeanModel per snapshot (redshift).

    Interface-compatible with MeanModel so that ``apply_residual_prior``,
    ``predict_mean_from_raw_x``, ``add_mean_back``, and the inference API
    work unchanged.

    Keys stored as ``"z0p00"`` etc. inside ``nn.ModuleDict`` (dots are not
    allowed).  The public interface uses clean float-string keys like ``"0.00"``.
    """

    @staticmethod
    def _sanitize_key(z_key: str) -> str:
        return "z" + z_key.replace(".", "p")

    def __init__(self, models: Dict[str, "MeanModel"], z_tolerance: float = 0.05):
        super().__init__()
        sanitized = {self._sanitize_key(k): v for k, v in models.items()}
        self.snapshot_models = nn.ModuleDict(sanitized)
        # Map clean key <-> sanitized key
        self.z_keys = sorted(models.keys())
        self._key_map = {k: self._sanitize_key(k) for k in self.z_keys}
        self.z_values = [float(k) for k in self.z_keys]
        self.z_tolerance = z_tolerance
        first = next(iter(models.values()))
        self.theta_dim = first.theta_dim
        self.theta_start_idx = first.theta_start_idx
        # Expose use_redshift=True so callers pass redshift to us for routing.
        self.use_redshift = True
        # Build a fake `net` attribute so get_mean_model_config can read y_dim.
        self.net = first.net

    def get_model(self, z_key: str) -> "MeanModel":
        """Get sub-model by clean key like '0.00'."""
        return self.snapshot_models[self._key_map[z_key]]

    def _find_model(self, z_val: float) -> Optional["MeanModel"]:
        for key, z in zip(self.z_keys, self.z_values):
            if abs(z - z_val) < self.z_tolerance:
                return self.snapshot_models[self._key_map[key]]
        return None  # unmatched z (e.g. padding positions)

    def forward(
        self,
        log_m: torch.Tensor,
        log_r: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
        redshift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if redshift is None:
            raise ValueError("PerSnapshotMeanModel requires redshift input")
        y_dim = self.net[-1].out_features
        out = torch.zeros(*log_m.shape, y_dim, device=log_m.device, dtype=log_m.dtype)
        unique_z = torch.unique(redshift)
        for z_val in unique_z:
            model = self._find_model(z_val.item())
            if model is None:
                continue  # padding or unseen redshift → leave zeros
            mask = (redshift - z_val).abs() < self.z_tolerance
            out[mask] = model(
                log_m[mask],
                log_r[mask],
                theta=theta[mask] if theta is not None else None,
                redshift=None,
            )
        return out


def _flatten_mean_training_data(
    tasks: List[RunTask],
    include_redshift: bool = False,
    theta_dim: int = 0,
    theta_start_idx: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    parts = []
    y_parts = []
    for t in tasks:
        mr = t.x[:, :, :2].reshape(-1, 2)
        y_flat = t.y.reshape(-1, t.y.shape[-1])
        # Exclude rows where any channel is invalid (zero-masked).
        if t.valid_mask is not None:
            row_valid = t.valid_mask.reshape(-1, t.valid_mask.shape[-1]).all(axis=1)
            keep = row_valid
        else:
            keep = np.ones(mr.shape[0], dtype=np.bool_)
        mr = mr[keep]
        y_flat = y_flat[keep]
        cols = [mr]
        if theta_dim > 0:
            theta_cols = t.x[:, :, theta_start_idx:theta_start_idx + theta_dim].reshape(-1, theta_dim)
            theta_cols = theta_cols[keep]
            cols.append(theta_cols)
        if include_redshift:
            z_col = np.full((mr.shape[0], 1), t.redshift, dtype=np.float32)
            cols.append(z_col)
        parts.append(np.concatenate(cols, axis=1))
        y_parts.append(y_flat)
    x_out = np.concatenate(parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.float32)
    return x_out, y


def train_mean_model(
    tasks: List[RunTask],
    y_dim: int,
    args,
    device: torch.device,
    verbose: bool = True,
    num_workers_override: Optional[int] = None,
) -> MeanModel:
    # Detect multi-snapshot: condition mean model on redshift when > 1 unique z.
    unique_z = set(float(t.redshift) for t in tasks)
    multi_snap = len(unique_z) > 1
    mean_theta_dim = int(getattr(args, "mean_theta_dim", 0))
    theta_start_idx = int(getattr(args, "theta_start_idx", 2))
    x_np, y_np = _flatten_mean_training_data(
        tasks, include_redshift=multi_snap,
        theta_dim=mean_theta_dim, theta_start_idx=theta_start_idx,
    )
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    # Build per-sample snapshot-balancing weights so each redshift contributes
    # equally to the mean-model loss regardless of data volume.
    snapshot_balanced = bool(getattr(args, "snapshot_balanced_loss", False)) and multi_snap
    if snapshot_balanced:
        z_col_idx = x_np.shape[1] - 1  # redshift is the last column
        z_vals = x_np[:, z_col_idx]
        unique_z_arr, inv_idx, z_counts = np.unique(
            np.round(z_vals, 4), return_inverse=True, return_counts=True
        )
        n_snaps = len(unique_z_arr)
        # weight_i = n_snaps / count(z_i)  →  each snapshot sums to n_snaps
        w_np = (float(n_snaps) / z_counts[inv_idx].astype(np.float32))
        w_np *= float(len(z_vals)) / float(w_np.sum())  # renormalize to mean=1
        sample_w = torch.from_numpy(w_np)
        if verbose:
            # Show the effective per-sample weight after renormalization.
            renorm = float(len(z_vals)) / float(n_snaps)
            for zi, cnt in zip(unique_z_arr, z_counts):
                eff_w = renorm / float(cnt)
                print(f"  Mean model snapshot balance: z={zi:.2f}  n={cnt}  weight={eff_w:.4f}")
    else:
        sample_w = torch.ones(x.shape[0], dtype=torch.float32)

    # Balance channels with very different scales (critical for all_profiles).
    # Without this, large-amplitude targets dominate the mean-prior fit and
    # small-amplitude channels (e.g., pressure, gas_density) can degrade.
    y_scale_np = np.std(y_np, axis=0).astype(np.float32)
    y_scale_np = np.where(y_scale_np < 1e-6, 1.0, y_scale_np).astype(np.float32)
    y_scale = torch.from_numpy(y_scale_np).to(device)

    mean_ds = TensorDataset(x, y, sample_w)

    mean_sampler = (
        DistributedSampler(
            mean_ds,
            num_replicas=dist_world_size(),
            rank=dist_rank(),
            shuffle=True,
        )
        if is_dist_ready()
        else None
    )
    if num_workers_override is None:
        workers = int(args.ddp_num_workers) if is_dist_ready() else int(args.num_workers)
    else:
        workers = int(num_workers_override)

    loader = DataLoader(
        mean_ds,
        batch_size=args.mean_batch_size,
        shuffle=(mean_sampler is None),
        sampler=mean_sampler,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    mean_n_hidden = int(getattr(args, "mean_n_hidden", 2))
    model = MeanModel(
        y_dim=y_dim, hidden_dim=args.mean_hidden_dim, use_redshift=multi_snap,
        theta_dim=mean_theta_dim, theta_start_idx=theta_start_idx,
        n_hidden_layers=mean_n_hidden,
    ).to(device)
    if verbose and (multi_snap or mean_theta_dim > 0):
        extras = []
        if mean_theta_dim > 0:
            extras.append(f"theta_dim={mean_theta_dim}")
        if multi_snap:
            extras.append(f"z={sorted(unique_z)}")
        print(f"Mean model conditioned on: {', '.join(extras)}")
    if is_dist_ready():
        if device.type == "cuda":
            model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)
        else:
            model = nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.AdamW(list(model.parameters()), lr=args.mean_lr, weight_decay=args.mean_weight_decay)  # type: ignore[arg-type]
    mean_loss_kind = str(getattr(args, "mean_loss", "mse")).lower()
    if mean_loss_kind == "huber":
        loss_fn = nn.HuberLoss(delta=0.5, reduction="none")
    elif mean_loss_kind == "mae":
        loss_fn = nn.L1Loss(reduction="none")
    else:
        loss_fn = nn.MSELoss(reduction="none")

    for epoch in range(args.mean_epochs):
        if mean_sampler is not None:
            mean_sampler.set_epoch(epoch)
        model.train()
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        for xb, yb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)

            theta_in = xb[:, 2:2 + mean_theta_dim] if mean_theta_dim > 0 else None
            z_idx = 2 + mean_theta_dim
            z_in = xb[:, z_idx] if multi_snap else None
            pred = model(xb[:, 0], xb[:, 1], theta=theta_in, redshift=z_in)
            # Channel-balanced loss in normalized target space, with optional
            # per-sample snapshot balancing weights.
            per_sample = loss_fn(
                (pred - yb) / y_scale.view(1, -1),
                torch.zeros_like(yb),
            ).mean(dim=-1)  # (B,)
            loss = (per_sample * wb).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            bsz = int(xb.shape[0])
            epoch_loss_sum += float(loss.detach().cpu()) * float(bsz)
            epoch_sample_count += bsz

        if is_dist_ready():
            loss_cnt = torch.tensor([epoch_loss_sum, float(epoch_sample_count)], dtype=torch.float64, device=device)
            dist.all_reduce(loss_cnt, op=dist.ReduceOp.SUM)
            epoch_loss_sum = float(loss_cnt[0].item())
            epoch_sample_count = int(loss_cnt[1].item())

        if verbose and ((epoch + 1) % max(1, args.mean_log_every) == 0 or epoch == 0 or (epoch + 1) == args.mean_epochs):
            epoch_mse = (epoch_loss_sum / float(max(1, epoch_sample_count))) if epoch_sample_count > 0 else float("nan")
            rmse = math.sqrt(max(0.0, epoch_mse)) if math.isfinite(epoch_mse) else float("nan")
            print(f"Mean model epoch {epoch+1:03d}/{args.mean_epochs:03d}: RMSE={rmse:.6f}")

    core_model = unwrap_model(model)
    if not isinstance(core_model, MeanModel):
        raise TypeError(f"Expected MeanModel, got {type(core_model)}")
    core_model = cast(MeanModel, core_model)
    core_model.eval()
    for p in core_model.parameters():
        p.requires_grad_(False)
    return core_model


def train_per_snapshot_mean_models(
    tasks: List[RunTask],
    y_dim: int,
    args,
    device: torch.device,
    verbose: bool = True,
    num_workers_override: Optional[int] = None,
) -> PerSnapshotMeanModel:
    """Train a separate MeanModel per unique redshift and wrap them."""
    from collections import defaultdict

    snap_tasks: Dict[str, List[RunTask]] = defaultdict(list)
    for t in tasks:
        z_key = f"{float(t.redshift):.2f}"
        snap_tasks[z_key].append(t)

    if verbose:
        for z_key in sorted(snap_tasks):
            n_pts = sum(t.n_halo * t.n_r for t in snap_tasks[z_key])
            print(f"Per-snapshot mean model: z={z_key}  tasks={len(snap_tasks[z_key])}  points={n_pts}")

    models: Dict[str, MeanModel] = {}
    for z_key in sorted(snap_tasks):
        if verbose:
            print(f"\n--- Training mean model for z={z_key} ---")
        model = train_mean_model(
            snap_tasks[z_key],
            y_dim=y_dim,
            args=args,
            device=device,
            verbose=verbose,
            num_workers_override=num_workers_override,
        )
        models[z_key] = model

    wrapper = PerSnapshotMeanModel(models)
    wrapper.eval()
    return wrapper


def get_mean_model_config(mean_model: MeanModel) -> Dict[str, Any]:
    hidden_dim = 0
    if len(mean_model.net) > 0 and isinstance(mean_model.net[0], nn.Linear):
        hidden_dim = int(mean_model.net[0].out_features)
    n_hidden_layers = int(sum(1 for m in mean_model.net if isinstance(m, nn.SiLU)))
    y_dim = 0
    if len(mean_model.net) > 0 and isinstance(mean_model.net[-1], nn.Linear):
        y_dim = int(mean_model.net[-1].out_features)
    return {
        "y_dim": y_dim,
        "hidden_dim": hidden_dim,
        "use_redshift": bool(mean_model.use_redshift),
        "theta_dim": int(mean_model.theta_dim),
        "theta_start_idx": int(mean_model.theta_start_idx),
        "n_hidden_layers": n_hidden_layers,
    }


def save_mean_checkpoint(
    checkpoint_path: Path,
    mean_model: Union[MeanModel, PerSnapshotMeanModel],
    args,
    target_names: List[str],
    mean_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    if isinstance(mean_model, PerSnapshotMeanModel):
        payload = {
            "per_snapshot_mean_models": True,
            "mean_model_configs": {
                z_key: get_mean_model_config(mean_model.get_model(z_key))
                for z_key in mean_model.z_keys
            },
            "mean_model_state_dicts": {
                z_key: mean_model.get_model(z_key).state_dict()
                for z_key in mean_model.z_keys
            },
            "target_name": args.target_name,
            "target_names": target_names,
            "args": vars(args),
            "mean_metrics": mean_metrics,
        }
    else:
        payload = {
            "mean_model_state_dict": mean_model.state_dict(),
            "mean_model_config": get_mean_model_config(mean_model),
            "target_name": args.target_name,
            "target_names": target_names,
            "args": vars(args),
            "mean_metrics": mean_metrics,
        }
    torch.save(payload, checkpoint_path)


def _mean_model_checkpoint_entries(
    mean_model: Optional[Union[MeanModel, PerSnapshotMeanModel]],
) -> Dict[str, Any]:
    """Return checkpoint dict entries that describe the mean model."""
    if mean_model is None:
        return {
            "mean_model_state_dict": None,
            "mean_model_config": None,
            "per_snapshot_mean_models": False,
        }
    if isinstance(mean_model, PerSnapshotMeanModel):
        return {
            "per_snapshot_mean_models": True,
            "mean_model_state_dict": mean_model.state_dict(),
            "mean_model_config": None,
            "mean_model_configs": {
                z_key: get_mean_model_config(mean_model.get_model(z_key))
                for z_key in mean_model.z_keys
            },
            "mean_model_state_dicts": {
                z_key: mean_model.get_model(z_key).state_dict()
                for z_key in mean_model.z_keys
            },
        }
    return {
        "mean_model_state_dict": mean_model.state_dict(),
        "mean_model_config": get_mean_model_config(mean_model),
        "per_snapshot_mean_models": False,
    }


def load_mean_checkpoint(
    checkpoint_path: Path, device: torch.device,
) -> Tuple[Union[MeanModel, PerSnapshotMeanModel], Dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Mean checkpoint not found: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location=device)

    # ---- per-snapshot mean models ----
    if raw.get("per_snapshot_mean_models", False):
        configs = raw["mean_model_configs"]
        states = raw["mean_model_state_dicts"]
        models: Dict[str, MeanModel] = {}
        for z_key in sorted(configs):
            cfg = configs[z_key]
            m = MeanModel(
                y_dim=int(cfg["y_dim"]),
                hidden_dim=int(cfg["hidden_dim"]),
                use_redshift=False,
                theta_dim=int(cfg["theta_dim"]),
                theta_start_idx=int(cfg["theta_start_idx"]),
                n_hidden_layers=int(cfg.get("n_hidden_layers", 2)),
            ).to(device)
            m.load_state_dict(states[z_key])
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            models[z_key] = m
        wrapper = PerSnapshotMeanModel(models).to(device)
        wrapper.eval()
        return wrapper, raw

    # ---- single shared mean model (legacy) ----
    cfg = raw.get("mean_model_config")
    state = raw.get("mean_model_state_dict")
    if cfg is None or state is None:
        raise ValueError(
            "Mean checkpoint is missing required keys: 'mean_model_config' and/or 'mean_model_state_dict'"
        )

    model = MeanModel(
        y_dim=int(cfg["y_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        use_redshift=bool(cfg["use_redshift"]),
        theta_dim=int(cfg["theta_dim"]),
        theta_start_idx=int(cfg["theta_start_idx"]),
        n_hidden_layers=int(cfg.get("n_hidden_layers", 2)),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, raw


@torch.no_grad()
def evaluate_mean_model_metrics(
    tasks: List[RunTask],
    mean_model: MeanModel,
    device: torch.device,
    batch_size: int,
    target_names: List[str],
    core_radius_frac: float = 0.2,
    core_radius_min_bins: int = 3,
) -> Dict[str, Any]:
    if len(tasks) == 0:
        return {
            "rmse_original_units": float("nan"),
            "rmse_core_original_units": float("nan"),
            "rmse_outer_original_units": float("nan"),
            "per_target": {},
            "per_snapshot": {},
            "n_tasks": 0,
        }

    y_dim = int(tasks[0].y.shape[-1])
    sum_sq = torch.zeros((y_dim,), dtype=torch.float64, device=device)
    sum_w = torch.zeros((y_dim,), dtype=torch.float64, device=device)
    sum_sq_core = torch.zeros((y_dim,), dtype=torch.float64, device=device)
    sum_w_core = torch.zeros((y_dim,), dtype=torch.float64, device=device)
    sum_sq_outer = torch.zeros((y_dim,), dtype=torch.float64, device=device)
    sum_w_outer = torch.zeros((y_dim,), dtype=torch.float64, device=device)

    per_snap_sq: Dict[int, float] = {}
    per_snap_w: Dict[int, float] = {}

    ts = int(mean_model.theta_start_idx)
    td = int(mean_model.theta_dim)

    for t in tasks:
        flat_x = torch.from_numpy(t.x[:, :, :2].reshape(-1, 2)).to(device)
        theta_flat = None
        if td > 0:
            theta_flat = torch.from_numpy(t.x[:, :, ts : ts + td].reshape(-1, td)).to(device)
        z_col = (
            torch.full((flat_x.shape[0],), t.redshift, dtype=torch.float32, device=device)
            if mean_model.use_redshift
            else None
        )

        preds = []
        for i in range(0, flat_x.shape[0], max(1, int(batch_size))):
            xb = flat_x[i : i + batch_size]
            tb = theta_flat[i : i + batch_size] if theta_flat is not None else None
            zb = z_col[i : i + batch_size] if z_col is not None else None
            preds.append(mean_model(xb[:, 0], xb[:, 1], theta=tb, redshift=zb))

        mu = torch.cat(preds, dim=0).reshape(t.y.shape)
        y = torch.from_numpy(t.y).to(device)

        if len(target_names) > 1:
            y, mu, _ = restore_all_profiles_physical_units(y, mu, None, target_names)

        resid_sq = (mu - y).pow(2).to(torch.float64)
        valid_mask_f = None
        if t.valid_mask is not None:
            valid_mask_f = torch.from_numpy(t.valid_mask).to(device=device, dtype=torch.float64)
            resid_sq = resid_sq * valid_mask_f
        n_halo = int(t.n_halo)
        n_r = int(t.n_r)

        sum_sq += resid_sq.sum(dim=(0, 1))
        if valid_mask_f is not None:
            sum_w += valid_mask_f.sum(dim=(0, 1))
        else:
            w_total = float(max(1, n_halo * n_r))
            sum_w += torch.full((y_dim,), w_total, dtype=torch.float64, device=device)

        n_core = max(int(core_radius_min_bins), int(math.ceil(float(core_radius_frac) * n_r)))
        n_core = min(max(0, n_core), n_r)
        if n_core > 0 and n_halo > 0 and n_r > 0:
            core_mask = (torch.arange(n_r, device=device) < n_core).view(1, n_r, 1).expand(n_halo, n_r, 1)
            outer_mask = ~core_mask

            core_mask_f = core_mask.to(resid_sq.dtype)
            outer_mask_f = outer_mask.to(resid_sq.dtype)
            if valid_mask_f is not None:
                core_mask_f = core_mask_f * valid_mask_f
                outer_mask_f = outer_mask_f * valid_mask_f

            sum_sq_core += (resid_sq * core_mask_f).sum(dim=(0, 1))
            sum_sq_outer += (resid_sq * outer_mask_f).sum(dim=(0, 1))

            if valid_mask_f is not None:
                sum_w_core += core_mask_f.sum(dim=(0, 1))
                sum_w_outer += outer_mask_f.sum(dim=(0, 1))
            else:
                w_core = float(core_mask.sum().item())
                w_outer = float(outer_mask.sum().item())
                sum_w_core += torch.full((y_dim,), w_core, dtype=torch.float64, device=device)
                sum_w_outer += torch.full((y_dim,), w_outer, dtype=torch.float64, device=device)

        snap_sq = float(resid_sq.sum().detach().cpu().item())
        if valid_mask_f is not None:
            snap_w = float(max(1.0, valid_mask_f.sum().detach().cpu().item()))
        else:
            snap_w = float(max(1, n_halo * n_r))
        per_snap_sq[t.snapnum] = per_snap_sq.get(t.snapnum, 0.0) + snap_sq
        per_snap_w[t.snapnum] = per_snap_w.get(t.snapnum, 0.0) + snap_w

    rmse_by_target = torch.sqrt(sum_sq / sum_w.clamp_min(1.0)).detach().cpu().numpy().tolist()
    rmse_core_by_target = torch.sqrt(sum_sq_core / sum_w_core.clamp_min(1.0)).detach().cpu().numpy().tolist()
    rmse_outer_by_target = torch.sqrt(sum_sq_outer / sum_w_outer.clamp_min(1.0)).detach().cpu().numpy().tolist()

    total_sq = float(sum_sq.sum().detach().cpu().item())
    total_w = float(sum_w.sum().detach().cpu().item())
    total_sq_core = float(sum_sq_core.sum().detach().cpu().item())
    total_w_core = float(sum_w_core.sum().detach().cpu().item())
    total_sq_outer = float(sum_sq_outer.sum().detach().cpu().item())
    total_w_outer = float(sum_w_outer.sum().detach().cpu().item())

    per_target = {
        name: {
            "rmse_original_units": float(rmse_by_target[i]),
            "rmse_core_original_units": float(rmse_core_by_target[i]),
            "rmse_outer_original_units": float(rmse_outer_by_target[i]),
        }
        for i, name in enumerate(target_names)
    }

    per_snapshot = {
        int(snap): {
            "rmse_original_units": float(math.sqrt(sq / max(1.0, per_snap_w[snap]))),
            "n_points": int(per_snap_w[snap]),
        }
        for snap, sq in sorted(per_snap_sq.items())
    }

    return {
        "rmse_original_units": float(math.sqrt(total_sq / max(1.0, total_w))),
        "rmse_core_original_units": float(math.sqrt(total_sq_core / max(1.0, total_w_core))),
        "rmse_outer_original_units": float(math.sqrt(total_sq_outer / max(1.0, total_w_outer))),
        "per_target": per_target,
        "per_snapshot": per_snapshot,
        "n_tasks": int(len(tasks)),
    }


@torch.no_grad()
def predict_mean_from_raw_x(raw_x: torch.Tensor, mean_model: MeanModel) -> torch.Tensor:
    theta_in = None
    if mean_model.theta_dim > 0:
        ts = mean_model.theta_start_idx
        theta_in = raw_x[..., ts:ts + mean_model.theta_dim]
    z_in = None
    if mean_model.use_redshift:
        z_in = raw_x[..., -1]
    return mean_model(raw_x[..., 0], raw_x[..., 1], theta=theta_in, redshift=z_in)


@torch.no_grad()
def apply_residual_prior(tasks: List[RunTask], mean_model: MeanModel, device: torch.device, batch_size: int) -> List[RunTask]:
    out: List[RunTask] = []
    ts = mean_model.theta_start_idx
    td = mean_model.theta_dim
    for t in tasks:
        flat_x = torch.from_numpy(t.x[:, :, :2].reshape(-1, 2)).to(device)
        theta_flat = None
        if td > 0:
            theta_flat = torch.from_numpy(
                t.x[:, :, ts:ts + td].reshape(-1, td)
            ).to(device)
        z_col = torch.full((flat_x.shape[0],), t.redshift, dtype=torch.float32, device=device) if mean_model.use_redshift else None
        preds = []
        for i in range(0, flat_x.shape[0], batch_size):
            xb = flat_x[i : i + batch_size]
            tb = theta_flat[i : i + batch_size] if theta_flat is not None else None
            zb = z_col[i : i + batch_size] if z_col is not None else None
            preds.append(mean_model(xb[:, 0], xb[:, 1], theta=tb, redshift=zb).detach().cpu())
        mean_y = torch.cat(preds, dim=0).numpy().reshape(t.y.shape).astype(np.float32)
        y_resid = (t.y - mean_y).astype(np.float32)
        out.append(
            RunTask(
                run_id=t.run_id,
                snapnum=t.snapnum,
                snap_idx=t.snap_idx,
                redshift=t.redshift,
                x=t.x,
                y=y_resid,
                n_halo=t.n_halo,
                n_r=t.n_r,
                valid_mask=t.valid_mask,
            )
        )
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


# ── Cool-core indicator ──────────────────────────────────────────────
# Physical constants for analytical T500 (CGS).
_CC_G = 6.674e-8        # cm^3 g^-1 s^-2
_CC_KB = 1.381e-16       # erg K^-1
_CC_MP = 1.673e-24       # g
_CC_MU = 0.59            # mean molecular weight (ionised primordial gas)
_CC_MSUN = 1.989e33      # g
_CC_KPC = 3.086e21       # cm
_CC_KEV_TO_K = 1.16e7    # K per keV


def compute_cc_indicator(
    m500c: np.ndarray,
    r500c: np.ndarray,
    temperature_array: np.ndarray,
    core_bins: int = 6,
    eps: float = 1e-30,
) -> np.ndarray:
    """Compute log10(T_core / T500_analytic) per halo.

    Parameters
    ----------
    m500c : (n_halo,) M500c in solar masses.
    r500c : (n_halo,) R500c in physical kpc.
    temperature_array : (n_halo, n_r) temperature profile in keV.
    core_bins : number of innermost radial bins to average for T_core.

    Returns
    -------
    cc : (n_halo,) log10(T_core / T500) — negative ≈ cool-core, positive ≈ NCC.
    """
    m = np.asarray(m500c, dtype=np.float64)
    R = np.asarray(r500c, dtype=np.float64)
    T = np.asarray(temperature_array, dtype=np.float64)

    T500_K = _CC_MU * _CC_MP * _CC_G * (m * _CC_MSUN) / (2.0 * _CC_KB * (R * _CC_KPC))
    T500_keV = T500_K / _CC_KEV_TO_K

    T_core = T[:, :core_bins].mean(axis=1)
    ratio = np.clip(T_core, eps, None) / np.clip(T500_keV, eps, None)
    return np.log10(ratio).astype(np.float32)


def fit_cc_prior(
    families: List["RunFamilyTask"],
    cc_feature_idx: int,
    n_mass_bins: int = 10,
) -> Dict[str, Any]:
    """Fit an empirical CC-indicator prior p(cc | log_M) from training data.

    Returns a dict suitable for storing in a checkpoint and sampling at inference.
    The prior is a per-mass-bin Gaussian: for each bin we store (mean_cc, std_cc).
    """
    all_logM = []
    all_cc = []
    for fam in families:
        for task in fam.snapshots:
            # x shape: (n_halo, n_r, x_dim) — CC is constant across r,
            # so take first radial bin.
            logM = task.x[:, 0, 0].astype(np.float64)  # un-normalized at this point
            cc = task.x[:, 0, cc_feature_idx].astype(np.float64)
            all_logM.append(logM)
            all_cc.append(cc)

    all_logM = np.concatenate(all_logM)
    all_cc = np.concatenate(all_cc)

    # Global fallback.
    global_mean = float(np.mean(all_cc))
    global_std = float(np.std(all_cc))

    # Bin edges by percentile to get roughly equal counts.
    bin_edges = np.percentile(all_logM, np.linspace(0, 100, n_mass_bins + 1))
    bin_edges[0] -= 1e-3
    bin_edges[-1] += 1e-3

    bin_mean = np.full(n_mass_bins, global_mean)
    bin_std = np.full(n_mass_bins, global_std)
    for i in range(n_mass_bins):
        mask = (all_logM >= bin_edges[i]) & (all_logM < bin_edges[i + 1])
        if mask.sum() > 2:
            bin_mean[i] = float(np.mean(all_cc[mask]))
            bin_std[i] = float(max(np.std(all_cc[mask]), 0.01))

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_mean": bin_mean.tolist(),
        "bin_std": bin_std.tolist(),
        "global_mean": global_mean,
        "global_std": max(global_std, 0.01),
        "n_halos": int(len(all_cc)),
    }


class CCPredictor(nn.Module):
    """Small MLP that predicts CC indicator distribution from (logM, theta).

    Outputs (mu_cc, log_sigma_cc) — a Gaussian p(cc | logM, theta).
    Trained on training-set halos before ANP training begins, analogous
    to the mean model.
    """

    def __init__(self, theta_dim: int = 35, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        in_dim = 1 + theta_dim  # logM + theta
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden_dim), nn.GELU()]
            prev = hidden_dim
        layers.append(nn.Linear(prev, 2))  # (mu_cc, log_sigma_cc)
        self.net = nn.Sequential(*layers)
        self.theta_dim = theta_dim

    def forward(self, log_mass: torch.Tensor, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu_cc, sigma_cc) per halo.

        Parameters
        ----------
        log_mass : (N,) log10(M500c)
        theta : (N, theta_dim) feedback / cosmological parameters
        """
        x = torch.cat([log_mass.unsqueeze(-1), theta], dim=-1)
        out = self.net(x)
        mu = out[:, 0]
        sigma = out[:, 1].exp().clamp(min=0.01, max=2.0)
        return mu, sigma


def train_cc_predictor(
    families: List["RunFamilyTask"],
    cc_feature_idx: int,
    theta_start_idx: int = 2,
    theta_dim: int = 35,
    hidden_dim: int = 128,
    n_layers: int = 3,
    lr: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 4096,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> CCPredictor:
    """Pre-train a CCPredictor on training data CC indicators.

    Collects (logM, theta, cc) from all training halos and fits
    p(cc | logM, theta) as a heteroscedastic Gaussian.
    """
    all_logM = []
    all_theta = []
    all_cc = []
    for fam in families:
        for task in fam.snapshots:
            n_h = task.x.shape[0]
            logM = task.x[:, 0, 0].astype(np.float32)           # (n_halo,)
            theta = task.x[:, 0, theta_start_idx:theta_start_idx + theta_dim].astype(np.float32)  # (n_halo, theta_dim)
            cc = task.x[:, 0, cc_feature_idx].astype(np.float32)  # (n_halo,)
            all_logM.append(logM)
            all_theta.append(theta)
            all_cc.append(cc)

    logM_t = torch.from_numpy(np.concatenate(all_logM))
    theta_t = torch.from_numpy(np.concatenate(all_theta))
    cc_t = torch.from_numpy(np.concatenate(all_cc))

    n_total = logM_t.shape[0]
    if verbose:
        print(f"Training CCPredictor on {n_total} halos, theta_dim={theta_dim}, "
              f"hidden={hidden_dim}, layers={n_layers}")

    model = CCPredictor(theta_dim=theta_dim, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    params = cast(List[torch.Tensor], list(model.parameters()))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    dataset = torch.utils.data.TensorDataset(logM_t, theta_t, cc_t)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False,
    )

    for epoch in range(epochs):
        model.train()
        epoch_nll = 0.0
        n_samples = 0
        for logM_b, theta_b, cc_b in loader:
            logM_b = logM_b.to(device)
            theta_b = theta_b.to(device)
            cc_b = cc_b.to(device)

            mu, sigma = model(logM_b, theta_b)
            # Negative log-likelihood of Gaussian.
            nll = 0.5 * (((cc_b - mu) / sigma) ** 2 + 2.0 * sigma.log() + math.log(2.0 * math.pi))
            loss = nll.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            epoch_nll += float(loss.detach()) * logM_b.shape[0]
            n_samples += logM_b.shape[0]

        scheduler.step()

        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0 or (epoch + 1) == epochs):
            avg_nll = epoch_nll / max(1, n_samples)
            print(f"  CCPredictor epoch {epoch+1:03d}/{epochs:03d}: NLL={avg_nll:.4f}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
# ─────────────────────────────────────────────────────────────────────


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
    log_channel_floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y, valid_mask).  valid_mask is True where the raw value > 0
    for positive-definite (log-transformed) channels, True everywhere else.
    If log_channel_floor > 0, values below it are additionally marked invalid."""
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
        y = np.log10(sx_proxy).astype(np.float32)
        return y, np.ones_like(y, dtype=np.bool_)

    if target_name == "all_profiles":
        selected_targets = all_profile_targets if all_profile_targets else ALL_PROFILE_TARGETS
        channels = []
        valid_channels = []
        log_floor = float(log_channel_floor) if log_channel_floor > 0 else 0.0
        for name in selected_targets:
            arr = target_map[name]
            if name in ALL_PROFILE_LOG_TARGETS:
                threshold = log_floor if log_floor > 0 else 0.0
                valid_channels.append((arr > threshold))
                arr = np.log10(np.clip(arr, eps, None))
            else:
                valid_channels.append(np.ones(arr.shape, dtype=np.bool_))
            channels.append(arr)
        stacked = np.stack(channels, axis=-1)
        valid_stacked = np.stack(valid_channels, axis=-1)
        return stacked.astype(np.float32), valid_stacked

    if target_name not in target_map:
        raise ValueError(f"Unsupported target {target_name}; choose from {TARGET_CHOICES}")

    y = target_map[target_name]
    return y, np.ones_like(y, dtype=np.bool_)


def build_tasks(args) -> List[RunFamilyTask]:
    mu_e = 2.0 / (1.0 + 0.76)
    mp = 1.67e-24

    base_path = Path(args.profiles_base)
    theta_by_run = load_theta_table(Path(args.param_csv), target_theta_dim=args.theta_dim)
    snapnums = list(args.resolved_snapnums)
    snapnum_to_idx = {int(s): i for i, s in enumerate(snapnums)}
    redshift_by_snap = dict(args.redshift_by_snap)
    use_cc_indicator = bool(getattr(args, "cc_indicator", False))

    discovered_by_snap = {
        int(s): discover_runs(base_path, suite=args.suite, sim_set=args.sim_set, snapnum=int(s))
        for s in snapnums
    }
    candidate_runs = sorted(set().union(*[set(v) for v in discovered_by_snap.values()]))
    if args.max_runs > 0:
        candidate_runs = candidate_runs[: args.max_runs]

    families: List[RunFamilyTask] = []
    skipped = 0
    skip_reasons = 0

    for run in candidate_runs:
        if run not in theta_by_run:
            skipped += 1
            continue

        try:
            snapshots: List[RunTask] = []
            for snap in snapnums:
                try:
                    fpath = resolve_profile_file(
                        run,
                        base_path=base_path,
                        suite=args.suite,
                        sim_set=args.sim_set,
                        snapnum=int(snap),
                    )
                except FileNotFoundError:
                    continue

                with np.load(fpath) as data:
                    m500c = data["M500c"].astype(np.float32)
                    r500c = data["R500c"].astype(np.float32)
                    r = data["radial_bins"].astype(np.float32)
                    y, valid_mask = select_target(
                        data,
                        target_name=args.target_name,
                        mu_e=mu_e,
                        mp=mp,
                        eps=args.eps,
                        all_profile_targets=getattr(args, "resolved_all_profile_targets", None),
                        log_channel_floor=float(getattr(args, "log_channel_floor", 0.0)),
                    )
                    # Load temperature for CC indicator before NPZ context closes.
                    if use_cc_indicator:
                        temperature_for_cc = data["temperature_array"].astype(np.float32)

                # Skip snapshots where the simulation produced no halos
                # (e.g., low-Omega0 runs at high redshift).  Without this
                # guard the reshape below raises ValueError and the entire
                # run (all snapshots) is discarded, biasing parameter priors.
                if m500c.shape[0] == 0:
                    continue

                if y.ndim == 2:
                    y = y[..., None]
                    valid_mask = valid_mask[..., None]
                if y.ndim != 3:
                    raise ValueError(f"Expected target with ndim 2 or 3; got shape {y.shape}")

                if args.radial_stride > 1:
                    r = r[:: args.radial_stride]
                    y = y[:, :: args.radial_stride, :]
                    valid_mask = valid_mask[:, :: args.radial_stride, :]

                # ---- Mass floor: drop halos below threshold ----
                mass_floor = float(getattr(args, "mass_floor", 0.0))
                if mass_floor > 0.0:
                    log_m_raw = np.log10(np.clip(m500c, 1e10, None))
                    mass_keep = log_m_raw >= mass_floor
                    if mass_keep.sum() == 0:
                        continue
                    if mass_keep.sum() < m500c.shape[0]:
                        m500c = m500c[mass_keep]
                        r500c = r500c[mass_keep]
                        y = y[mass_keep]
                        valid_mask = valid_mask[mass_keep]
                        if use_cc_indicator:
                            temperature_for_cc = temperature_for_cc[mass_keep]

                # ---- Drop halos with too many invalid (zero) points ----
                min_valid_frac = float(getattr(args, "min_valid_frac", 0.5))
                if min_valid_frac > 0.0 and valid_mask is not None:
                    # Fraction of valid points per halo across all channels and radii.
                    per_halo_valid_frac = valid_mask.reshape(valid_mask.shape[0], -1).mean(axis=1)
                    valid_keep = per_halo_valid_frac >= min_valid_frac
                    if valid_keep.sum() == 0:
                        continue
                    if valid_keep.sum() < m500c.shape[0]:
                        m500c = m500c[valid_keep]
                        r500c = r500c[valid_keep]
                        y = y[valid_keep]
                        valid_mask = valid_mask[valid_keep]
                        if use_cc_indicator:
                            temperature_for_cc = temperature_for_cc[valid_keep]

                if args.max_halos_per_run > 0 and m500c.shape[0] > args.max_halos_per_run:
                    rng = np.random.default_rng(args.seed + run + int(snap))
                    pick = np.sort(rng.choice(m500c.shape[0], size=args.max_halos_per_run, replace=False))
                    m500c = m500c[pick]
                    r500c = r500c[pick]
                    y = y[pick]
                    valid_mask = valid_mask[pick]
                    if use_cc_indicator:
                        temperature_for_cc = temperature_for_cc[pick]

                n_halo, n_r, _ = y.shape
                if n_halo < args.min_halos:
                    continue

                log_m = np.log10(np.clip(m500c, 1e10, None))[:, None]
                r500_for_ratio = r500c * float(args.r500_physical_factor)
                log_r_scaled = np.log10(np.clip(r[None, :] / r500_for_ratio[:, None], 1e-4, None))

                x_dim = int(args.theta_start_idx + args.theta_dim)
                use_continuous_redshift = bool(getattr(args, "use_continuous_redshift_feature", False))
                if use_continuous_redshift:
                    x_dim += 1
                if use_cc_indicator:
                    x_dim += 1
                x = np.zeros((n_halo, n_r, x_dim), dtype=np.float32)
                x[..., 0] = log_m
                x[..., 1] = log_r_scaled
                x[..., args.theta_start_idx : args.theta_start_idx + args.theta_dim] = theta_by_run[run][None, None, :]
                if use_continuous_redshift:
                    z_idx = int(getattr(args, "redshift_feature_idx", args.theta_start_idx + args.theta_dim))
                    if z_idx != (args.theta_start_idx + args.theta_dim):
                        raise ValueError(
                            "redshift_feature_idx must equal theta_start_idx + theta_dim "
                            "for the current appended-redshift layout"
                        )
                    x[..., z_idx] = float(redshift_by_snap[int(snap)])
                if use_cc_indicator:
                    cc_idx = int(args.cc_indicator_feature_idx)
                    cc_vals = compute_cc_indicator(
                        m500c, r500c, temperature_for_cc,
                        core_bins=int(args.cc_indicator_core_bins),
                        eps=args.eps,
                    )
                    x[..., cc_idx] = cc_vals[:, None]  # broadcast to all r

                snapshots.append(
                    RunTask(
                        run_id=run,
                        snapnum=int(snap),
                        snap_idx=int(snapnum_to_idx[int(snap)]),
                        redshift=float(redshift_by_snap[int(snap)]),
                        x=x,
                        y=y.astype(np.float32),
                        n_halo=n_halo,
                        n_r=n_r,
                        valid_mask=valid_mask,
                    )
                )

            if len(snapshots) < int(args.min_snapshots_per_run):
                skipped += 1
                continue
            families.append(RunFamilyTask(run_id=run, snapshots=snapshots))
        except Exception as e:
            skipped += 1
            # Show a few concrete failure reasons to avoid silent all-run skips.
            if skip_reasons < 5:
                print(f"[build_tasks] Skipping run {run}: {type(e).__name__}: {e}")
                skip_reasons += 1

    n_snapshots_total = sum(len(f.snapshots) for f in families)
    print(
        f"Built {len(families)} run families ({n_snapshots_total} snapshots) "
        f"from {len(candidate_runs)} discovered runs (skipped {skipped})."
    )
    if families:
        print(
            f"Example family run_id={families[0].run_id}, snapshots={len(families[0].snapshots)}, "
            f"x={families[0].snapshots[0].x.shape}, y={families[0].snapshots[0].y.shape}"
        )
    return families


def split_tasks(tasks: List[RunFamilyTask], train_frac: float, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(tasks))
    n_train = int(len(tasks) * train_frac)
    n_val = int(len(tasks) * val_frac)
    tr = [tasks[i] for i in idx[:n_train]]
    va = [tasks[i] for i in idx[n_train : n_train + n_val]]
    te = [tasks[i] for i in idx[n_train + n_val :]]
    return tr, va, te


def filter_snapshots_in_families(
    families: List[RunFamilyTask],
    drop_snapnums: Optional[set[int]] = None,
    keep_snapnums: Optional[set[int]] = None,
    min_snapshots_per_family: int = 1,
) -> List[RunFamilyTask]:
    out: List[RunFamilyTask] = []
    for fam in families:
        snaps = fam.snapshots
        if drop_snapnums:
            snaps = [s for s in snaps if int(s.snapnum) not in drop_snapnums]
        if keep_snapnums is not None:
            snaps = [s for s in snaps if int(s.snapnum) in keep_snapnums]
        if len(snaps) >= int(min_snapshots_per_family):
            out.append(RunFamilyTask(run_id=fam.run_id, snapshots=snaps))
    return out


def count_snapshot_presence(families: List[RunFamilyTask], snapnum: int) -> int:
    return int(sum(1 for fam in families for s in fam.snapshots if int(s.snapnum) == int(snapnum)))


def compute_norm_stats(
    train_tasks: List[RunFamilyTask],
    eps: float = 1e-6,
    robust: bool = False,
    mass_redshift_aware: bool = False,
    mass_bin_edges: Optional[np.ndarray] = None,
):
    """Compute normalization statistics from training data.

    When *robust* is True, use median / MAD (scaled to equivalent std)
    instead of mean / std for the y-channels.  This is more resistant to
    heavy-tailed outliers from extreme-mass or poorly-resolved halos.

    When *mass_redshift_aware* is True, additionally compute per-bin
    y-statistics keyed by (mass_bin, redshift).  At normalization time
    each halo is normalized with the stats from its own bin, producing
    a tighter, more homogeneous distribution for the network.
    """
    flat = flatten_family_tasks(train_tasks)
    x_stack = np.concatenate([t.x.reshape(-1, t.x.shape[-1]) for t in flat], axis=0)
    y_stack = np.concatenate([t.y.reshape(-1, t.y.shape[-1]) for t in flat], axis=0)

    # Build per-channel validity mask to exclude zero-valued points.
    has_valid = any(t.valid_mask is not None for t in flat)
    if has_valid:
        vm_stack = np.concatenate(
            [
                (t.valid_mask if t.valid_mask is not None
                 else np.ones_like(t.y, dtype=np.bool_)).reshape(-1, t.y.shape[-1])
                for t in flat
            ],
            axis=0,
        )
    else:
        vm_stack = np.ones_like(y_stack, dtype=np.bool_)

    x_mean = x_stack.mean(axis=0).astype(np.float32)
    x_std = x_stack.std(axis=0).astype(np.float32)
    x_std = np.where(x_std < eps, 1.0, x_std).astype(np.float32)

    y_dim = y_stack.shape[1]
    y_mean = np.zeros(y_dim, dtype=np.float32)
    y_std = np.ones(y_dim, dtype=np.float32)
    n_masked_total = 0
    for ch in range(y_dim):
        valid_vals = y_stack[:, ch][vm_stack[:, ch]]
        n_masked_total += int((~vm_stack[:, ch]).sum())
        if len(valid_vals) > 0:
            if robust:
                med = float(np.median(valid_vals))
                mad = float(np.median(np.abs(valid_vals - med)))
                y_mean[ch] = np.float32(med)
                y_std[ch] = np.float32(max(mad * 1.4826, eps))  # MAD → std scale
            else:
                y_mean[ch] = valid_vals.mean().astype(np.float32)
                y_std[ch] = valid_vals.std().astype(np.float32)
                if y_std[ch] < eps:
                    y_std[ch] = 1.0
    if n_masked_total > 0:
        print(f"[compute_norm_stats] Excluded {n_masked_total} zero-valued "
              f"point-channels from y normalization stats.")
    if robust:
        print("[compute_norm_stats] Used robust (median/MAD) normalization for y-channels.")

    result: Dict[str, Any] = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

    # ---- Mass / redshift conditioned stats ----
    if mass_redshift_aware:
        if mass_bin_edges is None:
            mass_bin_edges = np.array([11.0, 12.0, 12.5, 13.0, 13.5, 14.0, 16.0], dtype=np.float32)
        mass_bin_edges = np.asarray(mass_bin_edges, dtype=np.float32)
        # Collect per-halo mass and redshift alongside the y-values.
        logm_list, z_list, y_halo_list, vm_halo_list = [], [], [], []
        for t in flat:
            # x[..., 0] is log_m (broadcast across radii); take first radial bin.
            logm_per_halo = t.x[:, 0, 0]  # (n_halo,)
            z_per_halo = np.full(t.n_halo, t.redshift, dtype=np.float32)
            logm_list.append(logm_per_halo)
            z_list.append(z_per_halo)
            y_halo_list.append(t.y)  # (n_halo, n_r, y_dim)
            if t.valid_mask is not None:
                vm_halo_list.append(t.valid_mask)
            else:
                vm_halo_list.append(np.ones_like(t.y, dtype=np.bool_))

        logm_all = np.concatenate(logm_list, axis=0)     # (N,)
        z_all = np.concatenate(z_list, axis=0)            # (N,)
        y_all = np.concatenate(y_halo_list, axis=0)       # (N, n_r, y_dim)
        vm_all = np.concatenate(vm_halo_list, axis=0)     # (N, n_r, y_dim)

        unique_z = np.sort(np.unique(np.round(z_all, decimals=4)))
        mass_bin_idx = np.digitize(logm_all, mass_bin_edges) - 1
        mass_bin_idx = np.clip(mass_bin_idx, 0, len(mass_bin_edges) - 2)
        n_mass_bins = len(mass_bin_edges) - 1

        # Per-bin stats: dict keyed by (mass_bin_idx, z_rounded) -> {y_mean, y_std}
        bin_stats: Dict[str, Dict[str, np.ndarray]] = {}
        for mi in range(n_mass_bins):
            for z_val in unique_z:
                sel = (mass_bin_idx == mi) & (np.abs(z_all - z_val) < 0.01)
                key = f"{mi}_{z_val:.4f}"
                if sel.sum() < 20:
                    # Fall back to global stats when too few halos in this bin.
                    bin_stats[key] = {"y_mean": y_mean.copy(), "y_std": y_std.copy()}
                    continue
                y_sel = y_all[sel].reshape(-1, y_dim)     # (n_sel * n_r, y_dim)
                vm_sel = vm_all[sel].reshape(-1, y_dim)
                bm = np.zeros(y_dim, dtype=np.float32)
                bs = np.ones(y_dim, dtype=np.float32)
                for ch in range(y_dim):
                    vals = y_sel[:, ch][vm_sel[:, ch]]
                    if len(vals) < 10:
                        bm[ch] = y_mean[ch]
                        bs[ch] = y_std[ch]
                    elif robust:
                        med = float(np.median(vals))
                        mad = float(np.median(np.abs(vals - med)))
                        bm[ch] = np.float32(med)
                        bs[ch] = np.float32(max(mad * 1.4826, eps))
                    else:
                        bm[ch] = vals.mean().astype(np.float32)
                        s = vals.std().astype(np.float32)
                        bs[ch] = s if s >= eps else 1.0
                bin_stats[key] = {"y_mean": bm, "y_std": bs}

        result["mass_redshift_aware"] = True
        result["mass_bin_edges"] = mass_bin_edges
        result["unique_z"] = unique_z
        result["bin_stats"] = bin_stats
        n_bins_filled = sum(1 for v in bin_stats.values()
                           if not np.array_equal(v["y_mean"], y_mean))
        print(f"[compute_norm_stats] Mass-redshift aware: {n_mass_bins} mass bins × "
              f"{len(unique_z)} redshifts = {len(bin_stats)} bins "
              f"({n_bins_filled} with enough data).")

    return result


def _lookup_bin_stats(
    stats: Dict[str, Any],
    log_m_per_halo: np.ndarray,
    redshift: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Look up per-halo y_mean, y_std from mass-redshift bins.

    Returns arrays of shape (n_halo, 1, y_dim) for broadcasting with
    y of shape (n_halo, n_r, y_dim).
    """
    mass_bin_edges = stats["mass_bin_edges"]
    bin_stats = stats["bin_stats"]
    unique_z = stats["unique_z"]

    # Snap redshift to nearest available z value.
    z_idx = int(np.argmin(np.abs(unique_z - redshift)))
    z_val = unique_z[z_idx]

    mass_bin_idx = np.digitize(log_m_per_halo, mass_bin_edges) - 1
    mass_bin_idx = np.clip(mass_bin_idx, 0, len(mass_bin_edges) - 2)

    n_halo = len(log_m_per_halo)
    y_dim = stats["y_mean"].shape[0]
    y_mean_out = np.zeros((n_halo, 1, y_dim), dtype=np.float32)
    y_std_out = np.ones((n_halo, 1, y_dim), dtype=np.float32)

    for mi in np.unique(mass_bin_idx):
        key = f"{mi}_{z_val:.4f}"
        bs = bin_stats.get(key, {"y_mean": stats["y_mean"], "y_std": stats["y_std"]})
        mask = mass_bin_idx == mi
        y_mean_out[mask, 0, :] = bs["y_mean"]
        y_std_out[mask, 0, :] = bs["y_std"]

    return y_mean_out, y_std_out


def normalize_tasks(tasks: List[RunFamilyTask], stats) -> List[RunFamilyTask]:
    mass_redshift_aware = bool(stats.get("mass_redshift_aware", False))

    out: List[RunFamilyTask] = []
    for fam in tasks:
        snaps: List[RunTask] = []
        for t in fam.snapshots:
            x = ((t.x - stats["x_mean"][None, None, :]) / stats["x_std"][None, None, :]).astype(np.float32)

            if mass_redshift_aware:
                # Per-halo normalization based on mass and redshift bin.
                log_m_per_halo = t.x[:, 0, 0]  # (n_halo,) log10(M500c)
                y_m, y_s = _lookup_bin_stats(stats, log_m_per_halo, t.redshift)
                y = ((t.y - y_m) / y_s).astype(np.float32)
            else:
                y = ((t.y - stats["y_mean"][None, None, :]) / stats["y_std"][None, None, :]).astype(np.float32)

            # Replace invalid points with 0 (= channel mean in normalized space)
            # so encoder/decoder never see extreme outlier values.
            if t.valid_mask is not None:
                y = np.where(t.valid_mask, y, 0.0).astype(np.float32)
            snaps.append(
                RunTask(
                    run_id=t.run_id,
                    snapnum=t.snapnum,
                    snap_idx=t.snap_idx,
                    redshift=t.redshift,
                    x=x,
                    y=y,
                    n_halo=t.n_halo,
                    n_r=t.n_r,
                    valid_mask=t.valid_mask,
                )
            )
        out.append(RunFamilyTask(run_id=fam.run_id, snapshots=snaps))
    return out


def anp_collate(
    batch,
    max_aux_snapshots: int = 2,
    aux_halo_frac: float = 0.5,
    target_snapnum: Optional[int] = None,
):
    ctx_x_list, ctx_y_list, tgt_x_list, tgt_y_list = [], [], [], []
    ctx_snap_list, tgt_snap_list = [], []
    ctx_mask_list, tgt_mask_list = [], []
    tgt_vm_list = []
    meta = []

    for fam in batch:
        snapshots = fam.snapshots
        if len(snapshots) == 0:
            continue

        if target_snapnum is None:
            tgt_task = snapshots[np.random.randint(0, len(snapshots))]
        else:
            candidates = [s for s in snapshots if int(s.snapnum) == int(target_snapnum)]
            if len(candidates) == 0:
                continue
            tgt_task = candidates[np.random.randint(0, len(candidates))]
        aux_candidates = [s for s in snapshots if s.snap_idx != tgt_task.snap_idx]

        x = torch.tensor(tgt_task.x, dtype=torch.float32)
        y = torch.tensor(tgt_task.y, dtype=torch.float32)
        vm = tgt_task.valid_mask  # (n_halo, n_r, C) or None

        n_halo, n_r, xdim = x.shape
        ydim = y.shape[-1]
        n_c = int(np.exp(np.random.uniform(np.log(1), np.log(n_halo))))
        n_c = max(1, min(n_c, n_halo - 1))

        perm = torch.randperm(n_halo)
        ctx_h = perm[:n_c]
        tgt_h = perm

        ctx_x = x[ctx_h].reshape(-1, xdim)
        ctx_y = y[ctx_h].reshape(-1, ydim)
        ctx_snap_chunks = [torch.full((ctx_x.shape[0],), int(tgt_task.snap_idx), dtype=torch.long)]
        tgt_x = x[tgt_h].reshape(-1, xdim)
        tgt_y = y[tgt_h].reshape(-1, ydim)

        # Build per-channel validity mask for target points.
        if vm is not None:
            vm_t = torch.tensor(vm, dtype=torch.bool)
            tgt_vm = vm_t[tgt_h].reshape(-1, ydim)
        else:
            tgt_vm = torch.ones(tgt_y.shape[0], ydim, dtype=torch.bool)

        # Pull context from neighboring cosmic times for the same simulation.
        if aux_candidates and max_aux_snapshots > 0:
            n_aux = min(int(max_aux_snapshots), len(aux_candidates))
            sel = np.random.choice(len(aux_candidates), size=n_aux, replace=False)
            for j in sel:
                aux_task = aux_candidates[int(j)]
                x_aux = torch.tensor(aux_task.x, dtype=torch.float32)
                y_aux = torch.tensor(aux_task.y, dtype=torch.float32)
                n_halo_aux = int(x_aux.shape[0])
                n_aux_halo = max(1, int(round(float(aux_halo_frac) * n_c)))
                n_aux_halo = min(n_aux_halo, n_halo_aux)
                perm_aux = torch.randperm(n_halo_aux)
                pick_aux = perm_aux[:n_aux_halo]
                x_aux_flat = x_aux[pick_aux].reshape(-1, xdim)
                y_aux_flat = y_aux[pick_aux].reshape(-1, ydim)
                ctx_x = torch.cat([ctx_x, x_aux_flat], dim=0)
                ctx_y = torch.cat([ctx_y, y_aux_flat], dim=0)
                ctx_snap_chunks.append(torch.full((x_aux_flat.shape[0],), int(aux_task.snap_idx), dtype=torch.long))

        ctx_x_list.append(ctx_x)
        ctx_y_list.append(ctx_y)
        tgt_x_list.append(tgt_x)
        tgt_y_list.append(tgt_y)

        ctx_snap = torch.cat(ctx_snap_chunks, dim=0)
        tgt_snap = torch.full((tgt_x.shape[0],), int(tgt_task.snap_idx), dtype=torch.long)
        ctx_snap_list.append(ctx_snap)
        tgt_snap_list.append(tgt_snap)

        ctx_mask_list.append(torch.ones(ctx_x.shape[0], dtype=torch.bool))
        tgt_mask_list.append(torch.ones(tgt_x.shape[0], dtype=torch.bool))
        tgt_vm_list.append(tgt_vm)

        meta.append({
            "run_id": int(fam.run_id),
            "snapnum": int(tgt_task.snapnum),
            "snap_idx": int(tgt_task.snap_idx),
            "redshift": float(tgt_task.redshift),
            "n_halo": int(n_halo),
            "n_r": int(n_r),
            "n_c": int(n_c),
        })

    if len(ctx_x_list) == 0:
        raise RuntimeError(
            "anp_collate produced an empty batch. "
            "This can happen if target_snapnum is set but no families contain that snapshot."
        )

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

    def pad_mask_2d(seq):
        b = len(seq)
        max_len = max(t.shape[0] for t in seq)
        feat = seq[0].shape[1]
        out = torch.zeros((b, max_len, feat), dtype=torch.bool)
        for i, t in enumerate(seq):
            out[i, : t.shape[0]] = t
        return out

    def pad_long(seq):
        b = len(seq)
        max_len = max(t.shape[0] for t in seq)
        out = torch.zeros((b, max_len), dtype=torch.long)
        for i, t in enumerate(seq):
            out[i, : t.shape[0]] = t
        return out

    return {
        "ctx_x": pad_2d(ctx_x_list),
        "ctx_y": pad_2d(ctx_y_list),
        "tgt_x": pad_2d(tgt_x_list),
        "tgt_y": pad_2d(tgt_y_list),
        "ctx_snap": pad_long(ctx_snap_list),
        "tgt_snap": pad_long(tgt_snap_list),
        "ctx_mask": pad_mask(ctx_mask_list),
        "tgt_mask": pad_mask(tgt_mask_list),
        "tgt_valid_mask": pad_mask_2d(tgt_vm_list),
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
        cc_dual_head: bool = False,
    ):
        super().__init__()
        self.y_dim = y_dim
        self.theta_dim = theta_dim
        self.theta_start_idx = theta_start_idx
        self.theta_film_scale = theta_film_scale
        self.cc_dual_head = cc_dual_head
        self.trunk = DeepMLP(x_dim + d_model + d_latent, hidden_dim=hidden_dim, out_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
        self.theta_film = nn.Sequential(
            nn.Linear(theta_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        # Default (NCC) head — also used when cc_dual_head is False.
        self.mu = nn.Linear(hidden_dim, y_dim)
        self.log_sigma = nn.Linear(hidden_dim, y_dim)
        # CC-specific head (only when dual-head mode is enabled).
        if cc_dual_head:
            self.mu_cc = nn.Linear(hidden_dim, y_dim)
            self.log_sigma_cc = nn.Linear(hidden_dim, y_dim)

    def _compute_h(self, tgt_x, r, z):
        """Shared trunk + FiLM conditioning → hidden features h."""
        nt = tgt_x.shape[1]
        z_exp = z.unsqueeze(1).expand(-1, nt, -1)
        h = self.trunk(torch.cat([tgt_x, r, z_exp], dim=-1))

        # Condition decoder features directly on run-level theta via FiLM.
        theta = tgt_x[:, 0, self.theta_start_idx : self.theta_start_idx + self.theta_dim]
        film = self.theta_film(theta)
        gamma, beta = film.chunk(2, dim=-1)
        h = h * (1.0 + self.theta_film_scale * torch.tanh(gamma).unsqueeze(1))
        h = h + self.theta_film_scale * beta.unsqueeze(1)
        return h

    @staticmethod
    def _head_output(mu_layer, log_sigma_layer, h):
        mu = mu_layer(h)
        sigma = 0.1 + 0.9 * F.softplus(log_sigma_layer(h))
        return mu, sigma

    def forward(self, tgt_x, r, z):
        h = self._compute_h(tgt_x, r, z)
        return self._head_output(self.mu, self.log_sigma, h)

    def forward_dual(self, tgt_x, r, z):
        """Return ((mu_ncc, sigma_ncc), (mu_cc, sigma_cc))."""
        h = self._compute_h(tgt_x, r, z)
        ncc = self._head_output(self.mu, self.log_sigma, h)
        cc = self._head_output(self.mu_cc, self.log_sigma_cc, h)
        return ncc, cc


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
        core_bias_weight: float = 0.0,
        num_snapshots: int = 1,
        time_feature_scale: float = 0.1,
        ideal_gas_weight: float = 0.0,
        ideal_gas_channel_indices: Optional[Tuple[int, int, int]] = None,
        cc_dual_head: bool = False,
        cc_indicator_feature_idx: int = -1,
        decoder_likelihood: str = "gaussian",
        student_t_df: float = 5.0,
        context_dropout_rate: float = 0.0,
        input_noise_std: float = 0.0,
        beta_nll_weight: float = 0.0,
        free_bits: float = 0.0,
        snapshot_balanced_loss: bool = False,
        stratified_var_cal_weight: float = 0.0,
        stratified_mass_bins: int = 4,
        stratified_radius_bins: int = 4,
        stratified_min_points: int = 16,
    ):
        super().__init__()

        self.snapshot_balanced_loss = bool(snapshot_balanced_loss)
        self.context_dropout_rate = float(context_dropout_rate)
        self.input_noise_std = float(input_noise_std)
        self.beta_nll_weight = float(beta_nll_weight)
        self.free_bits = float(free_bits)
        self.stratified_var_cal_weight = float(stratified_var_cal_weight)
        self.stratified_mass_bins = max(1, int(stratified_mass_bins))
        self.stratified_radius_bins = max(1, int(stratified_radius_bins))
        self.stratified_min_points = max(1, int(stratified_min_points))

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
            cc_dual_head=cc_dual_head,
        )
        self.cc_dual_head = bool(cc_dual_head)
        self.cc_indicator_feature_idx = int(cc_indicator_feature_idx)
        self.decoder_likelihood = str(decoder_likelihood).lower()
        if self.decoder_likelihood not in {"gaussian", "student_t"}:
            raise ValueError(f"Unsupported decoder_likelihood={decoder_likelihood}")
        self.student_t_df = float(student_t_df)
        if self.decoder_likelihood == "student_t" and self.student_t_df <= 2.0:
            raise ValueError("student_t_df must be > 2.0 when decoder_likelihood=student_t")
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
        self.core_bias_weight = float(core_bias_weight)
        if self.use_task_uncertainty_weighting:
            # Kendall et al. (2018): one homoscedastic log-uncertainty per output channel.
            self.log_sigma_task = nn.Parameter(torch.zeros(y_dim))
        else:
            self.log_sigma_task = None

        self.ideal_gas_weight = float(ideal_gas_weight)
        self.ideal_gas_channel_indices = ideal_gas_channel_indices  # (rho_idx, T_idx, P_idx)
        # Buffer for y_std needed by ideal-gas penalty to convert normalized
        # errors back to log10 space.  Populated by set_y_std() after model
        # construction.  Defaults to ones (no-op scaling).
        self.register_buffer("y_std_buf", torch.ones(y_dim))

        self.num_snapshots = max(1, int(num_snapshots))
        self.time_feature_scale = float(time_feature_scale)
        self.time_embedding: Optional[nn.Embedding] = None
        self.time_mlp: Optional[nn.Module] = None
        if self.num_snapshots > 1:
            self.time_embedding = nn.Embedding(self.num_snapshots, x_dim)
            self.time_mlp = nn.Sequential(
                nn.Linear(x_dim, x_dim),
                nn.SiLU(),
                nn.Linear(x_dim, x_dim),
            )

    def set_y_std(self, y_std: torch.Tensor) -> None:
        """Store per-channel y_std for the ideal gas penalty."""
        self.y_std_buf = y_std.detach().clone().to(self.y_std_buf.device)

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

    def _fuse_time(self, x: torch.Tensor, snap_idx: torch.Tensor) -> torch.Tensor:
        if self.time_embedding is None or self.time_mlp is None:
            return x
        snap_idx = torch.clamp(snap_idx.long(), min=0, max=self.num_snapshots - 1)
        t = self.time_embedding(snap_idx)
        dt = self.time_mlp(t)
        return x + self.time_feature_scale * dt

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

    def _output_log_prob(self, y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if self.decoder_likelihood == "student_t":
            z = (y - mu) / sigma
            return StudentT(df=self.student_t_df).log_prob(z) - torch.log(sigma)
        return Normal(mu, sigma).log_prob(y)

    def _aleatoric_var_from_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        if self.decoder_likelihood == "student_t":
            df = max(self.student_t_df, 2.0001)
            return sigma**2 * (df / (df - 2.0))
        return sigma**2

    def forward(self, batch, device, beta: float = 1.0):
        ctx_x_raw = batch["ctx_x"].to(device)
        ctx_y = batch["ctx_y"].to(device)
        tgt_x_raw = batch["tgt_x"].to(device)
        tgt_y = batch["tgt_y"].to(device)
        ctx_snap = batch["ctx_snap"].to(device)
        tgt_snap = batch["tgt_snap"].to(device)
        ctx_mask = batch["ctx_mask"].to(device)
        tgt_mask = batch["tgt_mask"].to(device)

        ctx_x = self._fuse_time(self._embed_x(ctx_x_raw), ctx_snap)
        tgt_x = self._fuse_time(self._embed_x(tgt_x_raw), tgt_snap)

        # Training-time input noise regularization.
        if self.training and self.input_noise_std > 0:
            ctx_x = ctx_x + torch.randn_like(ctx_x) * self.input_noise_std
            tgt_x = tgt_x + torch.randn_like(tgt_x) * self.input_noise_std

        # Training-time context dropout: randomly mask out context points
        # so the model cannot rely on having rich context and must also
        # learn good priors and robust latent representations.
        if self.training and self.context_dropout_rate > 0:
            keep = torch.rand(ctx_mask.shape, device=ctx_mask.device) > self.context_dropout_rate
            # Always keep at least one context point per batch element.
            first_valid = ctx_mask.float().argmax(dim=1)
            for b in range(ctx_mask.shape[0]):
                keep[b, first_valid[b]] = True
            ctx_mask = ctx_mask & keep

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

        # ---- Dual-head CC/NCC routing ----
        if self.cc_dual_head:
            (mu_ncc, sigma_ncc), (mu_cc, sigma_cc) = self.dec.forward_dual(tgt_x, r, z)
            # Build per-point CC mask from raw (pre-Fourier) target features.
            # CC indicator < 0 → cool-core halo.
            cc_feat_idx = self.cc_indicator_feature_idx
            # tgt_x_raw is pre-embedding: (B, max_pts, raw_x_dim)
            cc_val = tgt_x_raw[:, :, cc_feat_idx]          # (B, max_pts)
            is_cc = (cc_val < 0.0).unsqueeze(-1).float()   # (B, max_pts, 1)
            is_ncc = 1.0 - is_cc
            mu    = mu_ncc    * is_ncc + mu_cc    * is_cc
            sigma = sigma_ncc * is_ncc + sigma_cc * is_cc
        else:
            mu, sigma = self.dec(tgt_x, r, z)

        mask_f = tgt_mask.float()
        pad_mask_3d = mask_f.unsqueeze(-1)                     # (B, N, 1) padding mask

        # Per-channel validity: exclude zero-valued profile points from loss.
        tgt_valid = batch.get("tgt_valid_mask")
        if tgt_valid is not None:
            tgt_valid = tgt_valid.to(device).float()           # (B, N, C)
            mask_3d = pad_mask_3d * tgt_valid                  # (B, N, C)
        else:
            mask_3d = pad_mask_3d                              # (B, N, 1)

        radius_w = self._build_radius_weights(tgt_mask, batch["meta"]).unsqueeze(-1).to(mask_3d.dtype)
        weighted_mask_3d = mask_3d * radius_w

        # Snapshot-balanced loss: reweight each batch element so all redshifts
        # contribute equally, preventing high-data-volume snapshots from
        # dominating the loss signal.
        if self.snapshot_balanced_loss and self.training:
            # Compute weights in float32 to avoid AMP float16 rounding issues,
            # then cast to match weighted_mask_3d.
            z_per_elem = torch.tensor(
                [float(m["redshift"]) for m in batch["meta"]],
                device=mu.device, dtype=torch.float32,
            )
            z_rounded = torch.round(z_per_elem * 1e4) / 1e4
            unique_z_batch = z_rounded.unique()
            n_snaps_batch = len(unique_z_batch)
            snap_w = torch.ones_like(z_rounded)
            for uz in unique_z_batch:
                mask_uz = (z_rounded == uz)
                count_uz = mask_uz.sum().float().clamp_min(1.0)
                snap_w[mask_uz] = float(n_snaps_batch) / count_uz
            # Normalize to mean=1 so total loss magnitude is unchanged.
            snap_w = snap_w / snap_w.mean().clamp_min(1e-12)
            weighted_mask_3d = weighted_mask_3d * snap_w.to(weighted_mask_3d.dtype).view(-1, 1, 1)

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

        log_prob = self._output_log_prob(tgt_y, mu, sigma)
        # Beta-NLL (Seitzer et al. 2022): weight log-prob by detached sigma^(2*beta)
        # to prevent overconfident sigma collapse during training.
        if self.beta_nll_weight > 0:
            beta_w = sigma.detach().pow(2 * self.beta_nll_weight)
            recon = (log_prob * beta_w * channel_w * weighted_mask_3d).sum() / (beta_w * channel_w * weighted_mask_3d).sum().clamp_min(1.0)
        else:
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

        # Free bits: clamp per-dimension KL to a minimum to prevent
        # posterior collapse (Kingma et al. 2016).
        kl_per_dim = kl_divergence(q_all, q_ctx)  # (batch, d_latent)
        if self.free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        kl = kl_per_dim.sum(dim=1).mean()
        mse = (resid_sq * channel_w * weighted_mask_3d).sum() / denom

        # Calibrate predicted variance against residual variance at each valid point.
        pred_var = self._aleatoric_var_from_scale(sigma)
        var_cal_pt = (torch.log(pred_var + 1e-8) - torch.log(resid_sq.detach() + 1e-8)) ** 2
        var_cal = (var_cal_pt * weighted_mask_3d).sum() / denom
        stratified_var_cal = torch.zeros((), device=mu.device, dtype=mu.dtype)
        if self.stratified_var_cal_weight > 0.0:
            cell_losses: List[torch.Tensor] = []
            n_total_pts = int(tgt_mask.shape[1])
            for b in range(tgt_mask.shape[0]):
                n_halo = int(batch["meta"][b]["n_halo"])
                n_r = int(batch["meta"][b]["n_r"])
                n_pts = int(min(n_total_pts, n_halo * n_r))
                if n_halo <= 0 or n_r <= 0 or n_pts <= 0:
                    continue

                n_mass_bins = min(self.stratified_mass_bins, n_halo)
                n_rad_bins = min(self.stratified_radius_bins, n_r)
                if n_mass_bins <= 0 or n_rad_bins <= 0:
                    continue

                # Mass is constant across radius for each halo; use first radius value.
                mass_vals = tgt_x_raw[b, :n_pts, 0].reshape(n_halo, n_r)[:, 0].detach()
                if not torch.isfinite(mass_vals).all():
                    continue
                if n_mass_bins > 1:
                    q = torch.linspace(0.0, 1.0, n_mass_bins + 1, device=mu.device, dtype=mass_vals.dtype)
                    mass_edges = torch.quantile(mass_vals, q)
                    mass_bin_idx = torch.bucketize(mass_vals, mass_edges[1:-1], right=False)
                else:
                    mass_bin_idx = torch.zeros(n_halo, device=mu.device, dtype=torch.long)

                r_idx = torch.arange(n_r, device=mu.device, dtype=torch.long)
                rad_bin_idx = torch.div(r_idx * n_rad_bins, n_r, rounding_mode="floor").clamp(max=n_rad_bins - 1)

                halo_bin_flat = mass_bin_idx.repeat_interleave(n_r)
                rad_bin_flat = rad_bin_idx.repeat(n_halo)
                for mb in range(n_mass_bins):
                    for rb in range(n_rad_bins):
                        point_sel = (halo_bin_flat == mb) & (rad_bin_flat == rb)
                        if int(point_sel.sum().item()) < self.stratified_min_points:
                            continue
                        cell_mask = torch.zeros(n_total_pts, device=mu.device, dtype=mu.dtype)
                        cell_mask[:n_pts] = point_sel.to(mu.dtype)
                        cell_mask_3d = cell_mask.view(1, -1, 1) * weighted_mask_3d[b : b + 1]
                        cell_denom = cell_mask_3d.sum().clamp_min(1.0)
                        cell_losses.append((var_cal_pt[b : b + 1] * cell_mask_3d).sum() / cell_denom)

            if cell_losses:
                stratified_var_cal = torch.stack(cell_losses).mean()

        sigma_mean = (sigma * mask_3d).sum() / mask_3d.sum().clamp_min(1.0)
        smooth = self._profile_smoothness_penalty(mu, tgt_mask, batch["meta"])

        core_bias = torch.zeros((), device=mu.device, dtype=mu.dtype)
        if self.core_bias_weight > 0.0:
            core_mask = torch.zeros_like(tgt_mask, dtype=torch.bool)
            for b in range(tgt_mask.shape[0]):
                n_halo = int(batch["meta"][b]["n_halo"])
                n_r = int(batch["meta"][b]["n_r"])
                n_pts = int(min(tgt_mask.shape[1], n_halo * n_r))
                if n_halo <= 0 or n_r <= 0 or n_pts <= 0:
                    continue
                n_core = max(int(self.core_radius_min_bins), int(math.ceil(float(self.core_radius_frac) * n_r)))
                n_core = min(max(0, n_core), n_r)
                if n_core <= 0:
                    continue
                core_mask_1h = (torch.arange(n_r, device=mu.device) < n_core).repeat(n_halo)
                core_mask[b, :n_pts] = core_mask_1h[:n_pts]
            core_mask = core_mask & tgt_mask
            core_mask_3d = core_mask.float().unsqueeze(-1)
            if tgt_valid is not None:
                core_mask_3d = core_mask_3d * tgt_valid
            core_denom = core_mask_3d.sum(dim=(0, 1)).clamp_min(1.0)
            core_mean_resid = ((mu - tgt_y) * core_mask_3d).sum(dim=(0, 1)) / core_denom
            core_bias = (core_mean_resid ** 2).mean()

        # Ideal-gas penalty: log10(P) = log10(rho) + log10(T) + const.
        # In normalized space the constant and mean-model offsets cancel when
        # expressed on prediction *errors*:
        #   err_P * sigma_P  ==  err_rho * sigma_rho  +  err_T * sigma_T
        # violation^2 penalises thermodynamic inconsistency.
        ideal_gas = torch.zeros((), device=mu.device, dtype=mu.dtype)
        if self.ideal_gas_weight > 0.0 and self.ideal_gas_channel_indices is not None:
            rho_i, T_i, P_i = self.ideal_gas_channel_indices
            err = mu - tgt_y                    # (B, N, C) normalised
            ys = self.y_std_buf                  # (C,)
            violation = (err[..., P_i] * ys[P_i]
                         - err[..., rho_i] * ys[rho_i]
                         - err[..., T_i] * ys[T_i])   # (B, N)
            # Only penalize where all three involved channels are valid.
            ig_mask = mask_f
            if tgt_valid is not None:
                ig_valid = (tgt_valid[:, :, rho_i] * tgt_valid[:, :, T_i] * tgt_valid[:, :, P_i])
                ig_mask = ig_mask * ig_valid
            ideal_gas = (violation ** 2 * ig_mask).sum() / ig_mask.sum().clamp_min(1.0)

        loss = -(recon - beta * kl) + task_uncertainty_loss
        loss = loss + self.var_cal_weight * var_cal
        loss = loss + self.stratified_var_cal_weight * stratified_var_cal
        loss = loss + self.smoothness_weight * smooth
        loss = loss + self.core_bias_weight * core_bias
        loss = loss + self.ideal_gas_weight * ideal_gas

        rmse_norm = torch.sqrt(mse)
        return {
            "loss": loss,
            "recon": recon,
            "kl": kl,
            "rmse_norm": rmse_norm,
            "var_cal": var_cal,
            "stratified_var_cal": stratified_var_cal,
            "sigma_mean": sigma_mean,
            "smooth": smooth,
            "core_bias": core_bias,
            "ideal_gas": ideal_gas,
        }

    @torch.no_grad()
    def predict(self, batch, device, n_samples: int = 30, cc_weights: Optional[torch.Tensor] = None):
        """Run MC prediction.

        Parameters
        ----------
        cc_weights : optional (B, 1, 1) tensor of p(CC) per halo-batch.
            When provided and ``cc_dual_head`` is True, blend both heads:
            ``output = (1-w)*ncc + w*cc``.  If None and dual-head is on,
            falls back to the NCC head (equivalent to w=0).
        """
        self.eval()
        ctx_x_raw = batch["ctx_x"].to(device)
        ctx_y = batch["ctx_y"].to(device)
        tgt_x_raw = batch["tgt_x"].to(device)
        ctx_snap = batch["ctx_snap"].to(device)
        tgt_snap = batch["tgt_snap"].to(device)
        ctx_mask = batch["ctx_mask"].to(device)
        tgt_mask = batch["tgt_mask"].to(device)

        ctx_x = self._fuse_time(self._embed_x(ctx_x_raw), ctx_snap)
        tgt_x = self._fuse_time(self._embed_x(tgt_x_raw), tgt_snap)

        q_ctx = self.latent(ctx_x, ctx_y, ctx_mask)
        r = self.det(ctx_x, ctx_y, ctx_mask, tgt_x, tgt_mask)

        mus, sigs = [], []
        for _ in range(n_samples):
            z = q_ctx.rsample()
            if self.cc_dual_head:
                (mu_ncc, sig_ncc), (mu_cc, sig_cc) = self.dec.forward_dual(tgt_x, r, z)
                if cc_weights is not None:
                    w = cc_weights  # (B, 1, 1) or broadcastable
                    mu = (1 - w) * mu_ncc + w * mu_cc
                    sig = (1 - w) * sig_ncc + w * sig_cc
                else:
                    # Fallback: use NCC head when no weights provided.
                    mu, sig = mu_ncc, sig_ncc
            else:
                mu, sig = self.dec(tgt_x, r, z)
            mus.append(mu)
            sigs.append(sig)

        mus = torch.stack(mus, dim=0)
        sigs = torch.stack(sigs, dim=0)

        pred_mean = mus.mean(0)
        aleatoric_var = self._aleatoric_var_from_scale(sigs).mean(0)
        epistemic_var = mus.var(0, unbiased=False)
        total_std = (aleatoric_var + epistemic_var).sqrt()

        return pred_mean, total_std, aleatoric_var.sqrt(), epistemic_var.sqrt()


def denorm_y(y, y_mean: torch.Tensor, y_std: torch.Tensor):
    return y * y_std + y_mean


def _output_log_prob(
    y: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    decoder_likelihood: str,
    student_t_df: float,
):
    if decoder_likelihood == "student_t":
        z = (y - mu) / std
        return StudentT(df=float(student_t_df)).log_prob(z) - torch.log(std)
    return Normal(mu, std).log_prob(y)


def _coverage_halfwidth_multiplier(
    sigma_level: float,
    decoder_likelihood: str,
    student_t_df: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Match Gaussian ±kσ central coverage and map to Student-t quantile width.
    p_two_sided = math.erf(float(sigma_level) / math.sqrt(2.0))
    if decoder_likelihood != "student_t":
        return torch.tensor(float(sigma_level), device=device, dtype=dtype)

    q = 0.5 * (1.0 + p_two_sided)
    # StudentT.icdf is not implemented in PyTorch; use scipy instead.
    from scipy.stats import t as scipy_t
    val = float(scipy_t.ppf(q, df=float(student_t_df)))
    return torch.tensor(val, device=device, dtype=dtype)


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
        "stratified_var_cal": [],
        "sigma_mean": [],
        "smooth": [],
        "core_bias": [],
        "ideal_gas": [],
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
        "stratified_var_cal": [],
        "sigma_mean": [],
        "smooth": [],
        "core_bias": [],
        "ideal_gas": [],
    }

    for batch in loader:
        out = model(batch, device=device, beta=beta)
        for k in meter:
            meter[k].append(float(out[k].detach().cpu()))

    return {k: float(np.mean(v)) for k, v in meter.items()}


def _oracle_cc_weights(model, batch, device) -> Optional[torch.Tensor]:
    """Extract oracle CC weights from batch data for dual-head models.

    Returns (B, max_pts, 1) float tensor of p(CC) per point, or None.
    """
    core = unwrap_model(model)
    if not core.cc_dual_head:
        return None
    cc_idx = core.cc_indicator_feature_idx
    tgt_x_raw = batch["tgt_x"].to(device)
    cc_val = tgt_x_raw[:, :, cc_idx]  # (B, max_pts)
    return (cc_val < 0.0).float().unsqueeze(-1)  # (B, max_pts, 1)


def make_zeroshot_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Build a neutral-context batch for strict zero-shot evaluation.

    Keeps targets unchanged and replaces context with a single unmasked zero
    token, mirroring the inference API behavior.
    """
    tgt_x = batch["tgt_x"]
    tgt_y = batch["tgt_y"]
    tgt_snap = batch["tgt_snap"]
    tgt_mask = batch["tgt_mask"]
    meta = batch["meta"]

    bsz = int(tgt_x.shape[0])
    raw_x_dim = int(tgt_x.shape[-1])
    y_dim = int(tgt_y.shape[-1])

    ctx_x = torch.zeros((bsz, 1, raw_x_dim), dtype=tgt_x.dtype)
    ctx_y = torch.zeros((bsz, 1, y_dim), dtype=tgt_y.dtype)
    ctx_mask = torch.ones((bsz, 1), dtype=torch.bool)
    if tgt_snap.shape[1] > 0:
        ctx_snap = tgt_snap[:, :1].clone()
    else:
        ctx_snap = torch.zeros((bsz, 1), dtype=torch.long)

    return {
        "ctx_x": ctx_x,
        "ctx_y": ctx_y,
        "ctx_snap": ctx_snap,
        "tgt_x": tgt_x,
        "tgt_y": tgt_y,
        "tgt_snap": tgt_snap,
        "ctx_mask": ctx_mask,
        "tgt_mask": tgt_mask,
        "tgt_valid_mask": batch.get("tgt_valid_mask"),
        "meta": meta,
    }


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
    force_zeroshot_context: bool = False,
):
    model.eval()
    model_core = cast(StrongANP, unwrap_model(model))
    decoder_likelihood = str(getattr(model_core, "decoder_likelihood", "gaussian"))
    student_t_df = float(getattr(model_core, "student_t_df", 5.0))
    rmses = []
    nlls = []
    sum_sq = None
    sum_nll = None
    sum_w = None
    sum_sq_core = None
    sum_w_core = None
    sum_sq_outer = None
    sum_w_outer = None

    # Per-snapshot accumulators for stratified metrics.
    per_snap_sq: Dict[int, torch.Tensor] = {}
    per_snap_w: Dict[int, torch.Tensor] = {}

    for batch in loader:
        batch_eval = make_zeroshot_batch(batch) if force_zeroshot_context else batch

        y = batch_eval["tgt_y"].to(device)
        mask = batch_eval["tgt_mask"].to(device)
        tgt_valid = batch_eval.get("tgt_valid_mask")
        if tgt_valid is not None:
            tgt_valid = tgt_valid.to(device).float()
        cc_w = _oracle_cc_weights(model, batch_eval, device)
        mu, std, _, _ = model.predict(batch_eval, device=device, n_samples=n_samples, cc_weights=cc_w)

        y_o = denorm_y(y, y_mean, y_std)
        mu_o = denorm_y(mu, y_mean, y_std)
        y_o = add_mean_back(y_o, batch_eval["tgt_x"].to(device), x_mean, x_std, mean_model)
        mu_o = add_mean_back(mu_o, batch_eval["tgt_x"].to(device), x_mean, x_std, mean_model)
        std_o = (std * y_std).clamp_min(1e-6)
        if len(target_names) > 1:
            y_o, mu_o, std_o = restore_all_profiles_physical_units(y_o, mu_o, std_o, target_names)
        if std_o is None:
            raise RuntimeError("std tensor unexpectedly None in evaluate_test_metrics")
        std_o_t = cast(torch.Tensor, std_o)

        mask_3d = mask.float().unsqueeze(-1)
        if tgt_valid is not None:
            mask_3d = mask_3d * tgt_valid
        core_mask = torch.zeros_like(mask, dtype=torch.bool)
        for b in range(mask.shape[0]):
            n_halo = int(batch_eval["meta"][b]["n_halo"])
            n_r = int(batch_eval["meta"][b]["n_r"])
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
        if tgt_valid is not None:
            core_mask_3d = core_mask_3d * tgt_valid
            outer_mask_3d = outer_mask_3d * tgt_valid

        rmse = torch.sqrt((((mu_o - y_o) ** 2) * mask_3d).sum() / mask_3d.sum())
        log_prob = _output_log_prob(y_o, mu_o, std_o_t, decoder_likelihood=decoder_likelihood, student_t_df=student_t_df)
        nll = -(log_prob * mask_3d).sum() / mask_3d.sum()

        ss = (((mu_o - y_o) ** 2) * mask_3d).sum(dim=(0, 1))
        sn = (-(log_prob * mask_3d)).sum(dim=(0, 1))
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

        # Per-snapshot RMSE accumulation.
        for b in range(mask.shape[0]):
            snap_b = int(batch_eval["meta"][b]["snapnum"])
            n_halo_b = int(batch_eval["meta"][b]["n_halo"])
            n_r_b = int(batch_eval["meta"][b]["n_r"])
            n_pts_b = min(mask.shape[1], n_halo_b * n_r_b)
            s_b = (((mu_o - y_o) ** 2) * mask_3d)[b, :n_pts_b].sum()
            w_b = mask_3d[b, :n_pts_b].sum()
            if snap_b not in per_snap_sq:
                per_snap_sq[snap_b] = s_b
                per_snap_w[snap_b] = w_b
            else:
                per_snap_sq[snap_b] = per_snap_sq[snap_b] + s_b
                per_snap_w[snap_b] = per_snap_w[snap_b] + w_b

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

    # Per-snapshot RMSE summary.
    per_snapshot = {}
    for snap_k in sorted(per_snap_sq.keys()):
        sq_val = float(per_snap_sq[snap_k].detach().cpu())
        w_val = float(per_snap_w[snap_k].detach().cpu())
        per_snapshot[snap_k] = {
            "rmse_original_units": float(math.sqrt(sq_val / max(1.0, w_val))),
            "n_points": int(w_val),
        }

    return {
        "rmse_original_units": float(np.mean(rmses)),
        "nll_original_units": float(np.mean(nlls)),
        "rmse_core_original_units": rmse_core,
        "rmse_outer_original_units": rmse_outer,
        "per_target": per_target,
        "per_snapshot": per_snapshot,
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

    # Penalize large per-snapshot disparity: add worst-snapshot RMSE to prevent
    # early stopping from ignoring poorly-performing redshifts.
    per_snap = detailed_metrics.get("per_snapshot", {})
    if len(per_snap) > 1:
        snap_rmses = [float(v["rmse_original_units"]) for v in per_snap.values() if math.isfinite(v["rmse_original_units"])]
        if snap_rmses:
            worst_snap_rmse = max(snap_rmses)
            score += 0.5 * worst_snap_rmse

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
    model_core = cast(StrongANP, unwrap_model(model))
    decoder_likelihood = str(getattr(model_core, "decoder_likelihood", "gaussian"))
    student_t_df = float(getattr(model_core, "student_t_df", 5.0))
    covered = 0
    total = 0
    for batch in loader:
        y = batch["tgt_y"].to(device)
        mask = batch["tgt_mask"].to(device)
        tgt_valid = batch.get("tgt_valid_mask")
        if tgt_valid is not None:
            tgt_valid = tgt_valid.to(device)
        cc_w = _oracle_cc_weights(model, batch, device)
        mu, std, _, _ = model.predict(batch, device=device, n_samples=n_samples, cc_weights=cc_w)

        y_o = denorm_y(y, y_mean, y_std)
        mu_o = denorm_y(mu, y_mean, y_std)
        y_o = add_mean_back(y_o, batch["tgt_x"].to(device), x_mean, x_std, mean_model)
        mu_o = add_mean_back(mu_o, batch["tgt_x"].to(device), x_mean, x_std, mean_model)
        std_o = std * y_std

        if len(target_names) > 1:
            y_o, mu_o, std_o = restore_all_profiles_physical_units(y_o, mu_o, std_o, target_names)
        if std_o is None:
            raise RuntimeError("std tensor unexpectedly None in empirical_coverage")

        halfwidth = _coverage_halfwidth_multiplier(
            sigma_level=sigma_level,
            decoder_likelihood=decoder_likelihood,
            student_t_df=student_t_df,
            device=std_o.device,
            dtype=std_o.dtype,
        )
        metric_mask = mask.unsqueeze(-1)
        if tgt_valid is not None:
            metric_mask = metric_mask & tgt_valid
        hit = (torch.abs(y_o - mu_o) <= halfwidth * std_o) & metric_mask
        covered += int(hit.sum().item())
        total += int(metric_mask.sum().item())

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

        cc_w = _oracle_cc_weights(model, batch, device)
        mu_ctx, _, _, _ = model.predict(batch, device=device, n_samples=n_samples, cc_weights=cc_w)

        b0 = {
            "ctx_x": batch["ctx_x"],
            "ctx_y": torch.zeros_like(batch["ctx_y"]),
            "ctx_snap": batch["ctx_snap"],
            "tgt_x": batch["tgt_x"],
            "tgt_y": batch["tgt_y"],
            "tgt_snap": batch["tgt_snap"],
            "ctx_mask": batch["ctx_mask"],
            "tgt_mask": batch["tgt_mask"],
            "tgt_valid_mask": batch.get("tgt_valid_mask"),
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
    tgt_snap = batch["tgt_snap"]
    tgt_mask = batch["tgt_mask"]
    meta = batch["meta"]

    bsz = tgt_x.shape[0]
    ctx_x = torch.zeros_like(tgt_x)
    ctx_y = torch.zeros_like(tgt_y)
    ctx_snap = torch.zeros_like(tgt_snap)
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
        ctx_snap[b, : idx.numel()] = tgt_snap[b, idx]
        ctx_mask[b, : idx.numel()] = True

    return {
        "ctx_x": ctx_x,
        "ctx_y": ctx_y,
        "ctx_snap": ctx_snap,
        "tgt_x": tgt_x,
        "tgt_y": tgt_y,
        "tgt_snap": tgt_snap,
        "ctx_mask": ctx_mask,
        "tgt_mask": tgt_mask,
        "tgt_valid_mask": batch.get("tgt_valid_mask"),
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
            tgt_valid = batch.get("tgt_valid_mask")
            mask_3d = mask.float().unsqueeze(-1)
            if tgt_valid is not None:
                mask_3d = mask_3d * tgt_valid.to(device).float()
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
    p.add_argument(
        "--snapnums",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of snapshots to use jointly. Defaults to [--snapnum].",
    )
    p.add_argument(
        "--snapshot-redshifts",
        type=str,
        default="90:0.0,74:0.5,60:1.0,44:2.0",
        help="Comma-separated map 'snap:z' used for metadata and temporal grouping.",
    )
    p.add_argument(
        "--min-snapshots-per-run",
        type=int,
        default=1,
        help="Minimum snapshots required for a run-family to be kept.",
    )
    p.add_argument("--target-name", type=str, default="all_profiles", choices=TARGET_CHOICES)
    p.add_argument(
        "--all-profiles-subset",
        type=str,
        nargs="+",
        default=["temperature", "pressure", "gas_density", "metallicity"],
        choices=ALL_PROFILE_TARGETS,
        help="Optional subset of all_profiles channels to train jointly. Only used when --target-name=all_profiles.",
    )

    p.add_argument("--theta-dim", type=int, default=35)
    p.add_argument("--theta-start-idx", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eps", type=float, default=1e-30)
    p.add_argument("--log-channel-floor", type=float, default=0.0,
                   help="Physical-unit floor for log-space channels. Values below this are treated as invalid. "
                        "0 (default) keeps the original behaviour (valid if > 0). "
                        "A value like 1e-29 removes extreme resolution-floor artefacts.")

    p.add_argument("--max-runs", type=int, default=1024, help="0 means all discovered runs")
    p.add_argument("--min-halos", type=int, default=2)
    p.add_argument("--max-halos-per-run", type=int, default=128, help="0 means keep all halos")
    p.add_argument("--radial-stride", type=int, default=1)
    p.add_argument(
        "--mass-floor",
        type=float,
        default=0.0,
        help=(
            "Minimum log10(M500c) to include in training. "
            "Halos with log10(M500c) < this value are dropped. "
            "Set to 12.5 to exclude poorly-resolved low-mass groups. 0 disables."
        ),
    )
    p.add_argument(
        "--min-valid-frac",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of valid (non-zero) points per halo across all "
            "channels.  Halos with fewer valid points are dropped to avoid "
            "feeding heavily-masked profiles to the encoder.  0 disables."
        ),
    )
    p.add_argument(
        "--physical-floor-quantile",
        type=float,
        default=0.0,
        help=(
            "If > 0, replace the global --eps floor with the per-channel "
            "quantile of positive values from training data.  E.g., 0.001 uses "
            "the 0.1th-percentile positive value as the floor for log10 clipping. "
            "Applied after build_tasks; 0 disables (uses --eps as-is)."
        ),
    )
    p.add_argument(
        "--robust-norm",
        action="store_true",
        default=False,
        help=(
            "Use median / MAD instead of mean / std for y normalization. "
            "More resistant to heavy-tailed outliers from extreme halos."
        ),
    )
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

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--kl-warmup-epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0001,
        help="Minimum validation improvement required to reset patience.",
    )
    p.add_argument("--accum-steps", type=int, default=16)
    p.add_argument(
        "--select-metric",
        type=str,
        default="weighted_orig",
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
        default=0.2,
        help="Additional weight on pressure RMSE in weighted_orig selection metric.",
    )
    p.add_argument(
        "--selection-temperature-weight",
        type=float,
        default=0.2,
        help="Additional weight on temperature RMSE in weighted_orig selection metric.",
    )
    p.add_argument(
        "--selection-pressure-core-weight",
        type=float,
        default=0.25,
        help="Additional weight on core-only pressure RMSE in weighted_orig selection metric.",
    )
    p.add_argument(
        "--selection-temperature-core-weight",
        type=float,
        default=0.35,
        help="Additional weight on core-only temperature RMSE in weighted_orig selection metric.",
    )

    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--d-latent", type=int, default=96)
    p.add_argument("--radius-fourier-n-freq", type=int, default=16)
    p.add_argument("--radius-fourier-scale", type=float, default=2.0)
    p.add_argument("--disable-radius-fourier", action="store_true")
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-latent-layers", type=int, default=3)
    p.add_argument("--n-ctx-layers", type=int, default=3)
    p.add_argument("--max-latent-points", type=int, default=1024)
    p.add_argument("--dec-hidden", type=int, default=384)
    p.add_argument("--dec-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--theta-film-scale", type=float, default=0.1)
    p.add_argument(
        "--decoder-likelihood",
        type=str,
        default="student_t",
        choices=["gaussian", "student_t"],
        help="Likelihood family used for decoder reconstruction/NLL.",
    )
    p.add_argument(
        "--student-t-df",
        type=float,
        default=4.0,
        help="Degrees of freedom for Student-t decoder likelihood (must be > 2).",
    )
    p.add_argument("--smoothness-weight", type=float, default=0.005)
    p.add_argument(
        "--ideal-gas-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the ideal-gas physics penalty enforcing "
            "log10(P) = log10(rho) + log10(T) + const across channels. "
            "Only active when gas_density, temperature, and pressure are all present."
        ),
    )
    p.add_argument("--var-cal-weight", type=float, default=0.05)
    p.add_argument(
        "--stratified-var-cal-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for mass-radius stratified variance calibration. "
            "Computes variance-calibration loss per mass/radius cell and averages across cells."
        ),
    )
    p.add_argument(
        "--stratified-mass-bins",
        type=int,
        default=4,
        help="Number of mass bins per training batch element for stratified variance calibration.",
    )
    p.add_argument(
        "--stratified-radius-bins",
        type=int,
        default=4,
        help="Number of radius bins per halo for stratified variance calibration.",
    )
    p.add_argument(
        "--stratified-min-points",
        type=int,
        default=16,
        help="Minimum points required in a mass-radius cell before contributing to stratified calibration.",
    )
    p.add_argument(
        "--context-dropout-rate",
        type=float,
        default=0.3,
        help=(
            "Fraction of context points randomly masked during training. "
            "Forces the model to learn robust priors and prevents over-reliance "
            "on context memorization. 0 disables. Recommended: 0.2-0.4."
        ),
    )
    p.add_argument(
        "--input-noise-std",
        type=float,
        default=0.02,
        help=(
            "Standard deviation of Gaussian noise added to input features "
            "during training for regularization. 0 disables. Recommended: 0.01-0.03."
        ),
    )
    p.add_argument(
        "--beta-nll-weight",
        type=float,
        default=0.5,
        help=(
            "Beta-NLL (Seitzer et al. 2022): weight reconstruction log-prob by "
            "sigma^(2*beta) (detached) to prevent overconfident sigma collapse. "
            "0 disables. Recommended: 0.5."
        ),
    )
    p.add_argument(
        "--free-bits",
        type=float,
        default=0.5,
        help=(
            "Free bits: minimum KL divergence per latent dimension (nats). "
            "Prevents posterior collapse while maintaining useful latent structure. "
            "0 disables. Recommended: 0.25-1.0."
        ),
    )
    p.add_argument(
        "--task-uncertainty-l2-weight",
        type=float,
        default=0.0005,
        help="L2 regularization strength for all_profiles log_sigma_task parameters.",
    )
    p.add_argument(
        "--task-uncertainty-clip",
        type=float,
        default=5.0,
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
        default=True,
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
        "--snapshot-balanced-loss",
        action="store_true",
        help=(
            "Reweight loss so each redshift contributes equally, regardless of "
            "data volume per snapshot.  Applied to both the mean model and the "
            "ANP reconstruction terms."
        ),
    )
    p.add_argument(
        "--per-snapshot-mean",
        action="store_true",
        help=(
            "Train a separate mean model for each redshift instead of one "
            "shared model.  Each per-snapshot model specialises to its own "
            "redshift, eliminating cross-z bias (e.g. cool-core bimodality "
            "at z=0)."
        ),
    )
    p.add_argument(
        "--core-radius-weight",
        type=float,
        default=2.0,
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
        default=6,
        help="Minimum number of inner bins per halo treated as core.",
    )
    p.add_argument(
        "--core-bias-weight",
        type=float,
        default=0.0,
        help=(
            "Penalty weight on squared mean signed residual in the core region. "
            "Use to suppress systematic inner-region over/under-prediction. 0 disables."
        ),
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

    p.add_argument(
        "--max-aux-snapshots",
        type=int,
        default=2,
        help="Max number of auxiliary snapshots used as context for each training episode.",
    )
    p.add_argument(
        "--aux-halo-frac",
        type=float,
        default=0.5,
        help="Auxiliary-context halo count as fraction of target-context halo count.",
    )
    p.add_argument(
        "--time-feature-scale",
        type=float,
        default=0.1,
        help="Strength of learned snapshot-time modulation in model feature space.",
    )
    p.add_argument(
        "--disable-continuous-redshift-feature",
        action="store_true",
        help=(
            "Disable continuous redshift as an explicit input feature. "
            "By default, redshift is appended to x so inference can condition on arbitrary z."
        ),
    )
    p.add_argument(
        "--cc-indicator",
        action="store_true",
        help=(
            "Append a cool-core indicator feature log10(T_core / T500_analytic) "
            "to the input vector. T_core is the mean temperature of the first N "
            "radial bins and T500 is derived analytically from M500 and R500."
        ),
    )
    p.add_argument(
        "--cc-indicator-core-bins",
        type=int,
        default=6,
        help="Number of innermost radial bins averaged for T_core in the CC indicator.",
    )
    p.add_argument(
        "--cc-dual-head",
        action="store_true",
        help=(
            "Fork the decoder Gaussian head into CC and NCC heads. "
            "Requires --cc-indicator. At training time loss is routed to "
            "the appropriate head per halo; at inference both heads are "
            "blended using CCPredictor probabilities."
        ),
    )

    p.add_argument(
        "--temporal-holdout-snapnum",
        type=int,
        default=-1,
        help="If >=0, exclude this snapshot from training/validation and report a dedicated holdout evaluation.",
    )
    p.add_argument(
        "--temporal-holdout-require-context",
        action="store_true",
        help="Require at least one non-holdout snapshot in a family for temporal holdout evaluation.",
    )

    p.add_argument("--disable-mean-prior", action="store_true")
    p.add_argument("--mean-use-theta", action="store_true",
                   help="Condition the mean model on theta (feedback/cosmo params) in addition to mass and radius.")
    p.add_argument("--mean-loss", type=str, default="mse", choices=["mse", "huber", "mae"],
                   help="Loss function for mean model. 'huber' (delta=0.5) or 'mae' reduces influence of core outliers; produces median-like predictions.")
    p.add_argument("--mean-hidden-dim", type=int, default=128)
    p.add_argument("--mean-n-hidden", type=int, default=-1,
                   help="Number of hidden layers in mean model. Default: auto (2 base, 3 with theta).")
    p.add_argument("--mean-epochs", type=int, default=80)
    p.add_argument("--mean-lr", type=float, default=1e-3)
    p.add_argument("--mean-weight-decay", type=float, default=1e-3)
    p.add_argument("--mean-batch-size", type=int, default=131072)
    p.add_argument("--mean-log-every", type=int, default=10)
    p.add_argument("--mean-predict-batch-size", type=int, default=262144)
    p.add_argument(
        "--training-stage",
        type=str,
        default="full",
        choices=["full", "mean_only", "anp_only"],
        help=(
            "Training stage mode: 'full' trains mean model then ANP residuals; "
            "'mean_only' trains/evaluates/saves only the mean model; "
            "'anp_only' loads a pre-trained mean model and trains ANP residuals."
        ),
    )
    p.add_argument(
        "--mean-checkpoint-path",
        type=str,
        default="",
        help="Path to a saved mean-model checkpoint (required for --training-stage=anp_only).",
    )
    p.add_argument(
        "--mean-output-path",
        type=str,
        default="",
        help="Optional explicit output path for saved mean model checkpoint.",
    )

    p.add_argument("--eval-samples", type=int, default=30)
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
    if getattr(args, "cc_dual_head", False) and not getattr(args, "cc_indicator", False):
        raise ValueError("--cc-dual-head requires --cc-indicator")
    if str(args.decoder_likelihood).lower() == "student_t" and float(args.student_t_df) <= 2.0:
        raise ValueError("--student-t-df must be > 2.0 when --decoder-likelihood=student_t")
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
    args.resolved_snapnums = list(args.snapnums) if args.snapnums else [int(args.snapnum)]
    args.redshift_by_snap = parse_snapshot_redshifts(args.snapshot_redshifts)
    args.use_continuous_redshift_feature = bool(not args.disable_continuous_redshift_feature)
    args.redshift_feature_idx = int(args.theta_start_idx + args.theta_dim)
    # CC indicator is appended after all prior features (theta, optional redshift).
    cc_offset = args.theta_start_idx + args.theta_dim
    if args.use_continuous_redshift_feature:
        cc_offset += 1
    args.cc_indicator_feature_idx = int(cc_offset)
    missing_snap_map = [s for s in args.resolved_snapnums if int(s) not in args.redshift_by_snap]
    if missing_snap_map:
        raise ValueError(
            "Missing redshift mapping for snapshots: "
            f"{missing_snap_map}. Update --snapshot-redshifts."
        )
    if args.min_snapshots_per_run < 1:
        raise ValueError("min_snapshots_per_run must be >= 1")
    if len(args.resolved_snapnums) > 1 and args.min_snapshots_per_run < 2:
        args.min_snapshots_per_run = 2

    if args.training_stage == "mean_only" and args.disable_mean_prior:
        raise ValueError("--training-stage=mean_only requires mean prior to be enabled (remove --disable-mean-prior)")
    if args.training_stage == "anp_only":
        if args.disable_mean_prior:
            raise ValueError("--training-stage=anp_only requires mean prior to be enabled (remove --disable-mean-prior)")
        if not args.mean_checkpoint_path:
            raise ValueError("--training-stage=anp_only requires --mean-checkpoint-path")

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
        if float(getattr(args, "mass_floor", 0.0)) > 0.0:
            print(f"[INFO] Mass floor: log10(M500c) >= {args.mass_floor:.2f}")
        if float(getattr(args, "min_valid_frac", 0.0)) > 0.0:
            print(f"[INFO] Min valid fraction per halo: {args.min_valid_frac:.2f}")
        if bool(getattr(args, "robust_norm", False)):
            print("[INFO] Robust normalization enabled (median / MAD)")
        print("[INFO] Mass-redshift-aware normalization: enabled")

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
    run_prefix = "mean" if args.training_stage == "mean_only" else "anp"
    out_dir = out_root / f"{run_prefix}_{args.target_name}_{run_tag}"
    if main_proc:
        out_dir.mkdir(parents=True, exist_ok=True)
    if ddp_enabled:
        dist_barrier(device)

    tasks = build_tasks(args)
    if len(tasks) < 20:
        raise RuntimeError(f"Too few run families discovered ({len(tasks)}). Check file paths and filters.")

    train_raw, val_raw, test_raw = split_tasks(tasks, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed)
    holdout_snap = int(args.temporal_holdout_snapnum)
    temporal_holdout_enabled = holdout_snap >= 0
    holdout_eval_raw: Optional[List[RunFamilyTask]] = None

    if temporal_holdout_enabled:
        if holdout_snap not in set(int(s) for s in args.resolved_snapnums):
            raise ValueError(
                f"temporal-holdout-snapnum={holdout_snap} not in configured --snapnums={args.resolved_snapnums}"
            )

        holdout_eval_raw = filter_snapshots_in_families(
            test_raw,
            keep_snapnums={holdout_snap},
            min_snapshots_per_family=1,
        )
        if args.temporal_holdout_require_context:
            holdout_eval_raw = [
                fam for fam in test_raw
                if any(int(s.snapnum) == holdout_snap for s in fam.snapshots)
                and any(int(s.snapnum) != holdout_snap for s in fam.snapshots)
            ]

        train_raw = filter_snapshots_in_families(
            train_raw,
            drop_snapnums={holdout_snap},
            min_snapshots_per_family=1,
        )
        val_raw = filter_snapshots_in_families(
            val_raw,
            drop_snapnums={holdout_snap},
            min_snapshots_per_family=1,
        )
        test_raw = filter_snapshots_in_families(
            test_raw,
            drop_snapnums={holdout_snap},
            min_snapshots_per_family=1,
        )

        if len(train_raw) == 0:
            raise RuntimeError("Temporal holdout removed all training families; reduce holdout strictness or increase data.")

    if main_proc:
        n_tr_snap = sum(len(f.snapshots) for f in train_raw)
        n_va_snap = sum(len(f.snapshots) for f in val_raw)
        n_te_snap = sum(len(f.snapshots) for f in test_raw)
        print(
            f"Split sizes (families) train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)} | "
            f"snapshots train={n_tr_snap}, val={n_va_snap}, test={n_te_snap}"
        )
        if temporal_holdout_enabled:
            n_hold = 0 if holdout_eval_raw is None else sum(
                1 for fam in holdout_eval_raw for s in fam.snapshots if int(s.snapnum) == holdout_snap
            )
            print(
                f"Temporal holdout enabled for snap {holdout_snap}: "
                f"removed from train/val/test, holdout eval targets={n_hold} snapshots"
            )

    y_dim = train_raw[0].snapshots[0].y.shape[-1]
    mean_theta_dim = int(args.theta_dim) if getattr(args, "mean_use_theta", False) else 0
    args.mean_theta_dim = mean_theta_dim
    if int(getattr(args, "mean_n_hidden", -1)) > 0:
        mean_n_hidden = int(args.mean_n_hidden)
    else:
        mean_n_hidden = 3 if mean_theta_dim > 0 else 2
    args.mean_n_hidden = mean_n_hidden
    mean_model: Optional[MeanModel] = None
    mean_prior_enabled = not args.disable_mean_prior
    mean_prior_source = "disabled"
    mean_checkpoint_used: Optional[str] = None
    mean_metrics_summary: Optional[Dict[str, Any]] = None
    if mean_prior_enabled:
        train_raw_flat_original = flatten_family_tasks(train_raw)
        val_raw_flat_original = flatten_family_tasks(val_raw)
        test_raw_flat_original = flatten_family_tasks(test_raw)

        if args.training_stage == "anp_only":
            mean_ckpt = Path(args.mean_checkpoint_path)
            if main_proc:
                print(f"Loading frozen mean profile model from checkpoint: {mean_ckpt}")
            mean_model, mean_payload = load_mean_checkpoint(mean_ckpt, device=device)
            if int(get_mean_model_config(mean_model)["y_dim"]) != int(y_dim):
                raise ValueError(
                    f"Loaded mean model y_dim={get_mean_model_config(mean_model)['y_dim']} does not match data y_dim={y_dim}"
                )
            loaded_target_name = mean_payload.get("target_name")
            if loaded_target_name is not None and str(loaded_target_name) != str(args.target_name):
                raise ValueError(
                    "Loaded mean model target_name does not match current run: "
                    f"ckpt={loaded_target_name}, run={args.target_name}"
                )
            mean_prior_source = "loaded"
            mean_checkpoint_used = str(mean_ckpt)
        else:
            use_per_snap = bool(getattr(args, "per_snapshot_mean", False))
            if main_proc:
                label = "per-snapshot" if use_per_snap else "shared"
                print(f"Pre-training frozen {label} mean profile model for residual targets...")
            if use_per_snap:
                mean_model = train_per_snapshot_mean_models(
                    train_raw_flat_original,
                    y_dim=y_dim,
                    args=args,
                    device=device,
                    verbose=main_proc,
                    num_workers_override=None,
                )
            else:
                mean_model = train_mean_model(
                    train_raw_flat_original,
                    y_dim=y_dim,
                    args=args,
                    device=device,
                    verbose=main_proc,
                    num_workers_override=None,
                )
            mean_prior_source = "trained"

        if mean_model is None:
            raise RuntimeError("Internal error: mean prior enabled but mean model was not initialized")

        if main_proc:
            mean_metrics_summary = {
                "train": evaluate_mean_model_metrics(
                    train_raw_flat_original,
                    mean_model,
                    device=device,
                    batch_size=int(args.mean_predict_batch_size),
                    target_names=target_names,
                    core_radius_frac=float(args.core_radius_frac),
                    core_radius_min_bins=int(args.core_radius_min_bins),
                ),
                "val": evaluate_mean_model_metrics(
                    val_raw_flat_original,
                    mean_model,
                    device=device,
                    batch_size=int(args.mean_predict_batch_size),
                    target_names=target_names,
                    core_radius_frac=float(args.core_radius_frac),
                    core_radius_min_bins=int(args.core_radius_min_bins),
                ),
                "test": evaluate_mean_model_metrics(
                    test_raw_flat_original,
                    mean_model,
                    device=device,
                    batch_size=int(args.mean_predict_batch_size),
                    target_names=target_names,
                    core_radius_frac=float(args.core_radius_frac),
                    core_radius_min_bins=int(args.core_radius_min_bins),
                ),
            }
            print(
                "Mean model RMSE (orig units): "
                f"train={mean_metrics_summary['train']['rmse_original_units']:.4g}, "
                f"val={mean_metrics_summary['val']['rmse_original_units']:.4g}, "
                f"test={mean_metrics_summary['test']['rmse_original_units']:.4g}"
            )

        if main_proc and args.training_stage in ("full", "mean_only"):
            mean_ckpt_path = Path(args.mean_output_path) if args.mean_output_path else (out_dir / "mean_model.pt")
            save_mean_checkpoint(
                mean_ckpt_path,
                mean_model,
                args=args,
                target_names=target_names,
                mean_metrics=mean_metrics_summary,
            )
            mean_checkpoint_used = str(mean_ckpt_path)
            print(f"Saved mean model checkpoint: {mean_ckpt_path}")

        if args.training_stage == "mean_only":
            if main_proc:
                metrics = {
                    "target_name": args.target_name,
                    "target_names": target_names,
                    "training_stage": args.training_stage,
                    "mean_prior": {
                        "enabled": True,
                        "source": mean_prior_source,
                        "checkpoint": mean_checkpoint_used,
                        "hidden_dim": args.mean_hidden_dim,
                        "epochs": args.mean_epochs,
                        "lr": args.mean_lr,
                        "weight_decay": args.mean_weight_decay,
                        "use_theta": bool(args.mean_use_theta),
                    },
                    "mean_metrics": mean_metrics_summary,
                    "n_tasks_total": len(tasks),
                    "n_train": len(train_raw),
                    "n_val": len(val_raw),
                    "n_test": len(test_raw),
                }
                with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                with (out_dir / "args.json").open("w", encoding="utf-8") as f:
                    json.dump(vars(args), f, indent=2)
                print(f"Mean-only artifacts written to: {out_dir}")
            if ddp_enabled:
                dist_barrier(device)
                cleanup_distributed()
            return

        train_raw_flat = apply_residual_prior(
            train_raw_flat_original,
            mean_model,
            device=device,
            batch_size=args.mean_predict_batch_size,
        )
        val_raw_flat = apply_residual_prior(
            val_raw_flat_original,
            mean_model,
            device=device,
            batch_size=args.mean_predict_batch_size,
        )
        test_raw_flat = apply_residual_prior(
            test_raw_flat_original,
            mean_model,
            device=device,
            batch_size=args.mean_predict_batch_size,
        )
        train_raw = remap_flat_tasks_to_families(train_raw, train_raw_flat)
        val_raw = remap_flat_tasks_to_families(val_raw, val_raw_flat)
        test_raw = remap_flat_tasks_to_families(test_raw, test_raw_flat)
        if main_proc:
            print("Applied residual targets per split: y <- y - y_mean(logM, logr)")
    else:
        if main_proc:
            print("Mean profile prior disabled; training ANP on direct targets.")
        mean_model = None

    norm_np = compute_norm_stats(
        train_raw,
        robust=bool(getattr(args, "robust_norm", False)),
        mass_redshift_aware=True,
    )

    # ---- Physical floor: raise eps to a per-channel data-driven floor ----
    phys_floor_q = float(getattr(args, "physical_floor_quantile", 0.0))
    if phys_floor_q > 0.0 and main_proc:
        flat_train = flatten_family_tasks(train_raw)
        y_all = np.concatenate([t.y.reshape(-1, t.y.shape[-1]) for t in flat_train], axis=0)
        vm_all = np.concatenate(
            [(t.valid_mask if t.valid_mask is not None else np.ones_like(t.y, dtype=np.bool_)).reshape(-1, t.y.shape[-1])
             for t in flat_train], axis=0,
        )
        y_dim = y_all.shape[1]
        phys_floors = np.full(y_dim, float(args.eps), dtype=np.float32)
        for ch in range(y_dim):
            vals = y_all[:, ch][vm_all[:, ch]]
            if len(vals) > 0:
                phys_floors[ch] = float(np.quantile(vals, phys_floor_q))
        print(f"[INFO] Physical floor quantile {phys_floor_q}: "
              f"floors={[f'{v:.4g}' for v in phys_floors]}")
        # Re-clip y in all splits to the physical floor.
        for split in (train_raw, val_raw, test_raw):
            for fam in split:
                for t in fam.snapshots:
                    for ch in range(y_dim):
                        t.y[:, :, ch] = np.clip(t.y[:, :, ch], phys_floors[ch], None)
        # Recompute norm stats after floor adjustment.
        norm_np = compute_norm_stats(
            train_raw,
            robust=bool(getattr(args, "robust_norm", False)),
            mass_redshift_aware=True,
        )

    # Fit CC-indicator prior from raw (un-normalized) training data.
    cc_prior_stats: Optional[Dict[str, Any]] = None
    cc_predictor_model: Optional[CCPredictor] = None
    if args.cc_indicator:
        cc_prior_stats = fit_cc_prior(train_raw, cc_feature_idx=int(args.cc_indicator_feature_idx))
        if main_proc:
            print(
                f"CC indicator prior fitted: global mean={cc_prior_stats['global_mean']:.3f}, "
                f"std={cc_prior_stats['global_std']:.3f} from {cc_prior_stats['n_halos']} halos"
            )
        # Train parameter-aware CC predictor f(logM, theta) -> (mu_cc, sigma_cc).
        if main_proc:
            print("Training parameter-aware CC predictor...")
        cc_predictor_model = train_cc_predictor(
            train_raw,
            cc_feature_idx=int(args.cc_indicator_feature_idx),
            theta_start_idx=int(args.theta_start_idx),
            theta_dim=int(args.theta_dim),
            hidden_dim=128,
            n_layers=3,
            lr=1e-3,
            epochs=200,
            batch_size=4096,
            device=device,
            verbose=main_proc,
        )

    train_tasks = normalize_tasks(train_raw, norm_np)
    val_tasks = normalize_tasks(val_raw, norm_np)
    test_tasks = normalize_tasks(test_raw, norm_np)

    y_mean = torch.tensor(norm_np["y_mean"], dtype=torch.float32, device=device)
    y_std = torch.tensor(norm_np["y_std"], dtype=torch.float32, device=device)
    x_mean = torch.tensor(norm_np["x_mean"], dtype=torch.float32, device=device)
    x_std = torch.tensor(norm_np["x_std"], dtype=torch.float32, device=device)

    train_ds = CAMELSRunFamilyDataset(train_tasks)
    val_ds = CAMELSRunFamilyDataset(val_tasks)
    test_ds = CAMELSRunFamilyDataset(test_tasks)
    holdout_test_tasks = normalize_tasks(holdout_eval_raw, norm_np) if holdout_eval_raw is not None else None
    holdout_test_ds = CAMELSRunFamilyDataset(holdout_test_tasks) if holdout_test_tasks is not None else None

    collate_fn = functools.partial(
        anp_collate,
        max_aux_snapshots=int(args.max_aux_snapshots),
        aux_halo_frac=float(args.aux_halo_frac),
    )
    holdout_collate_fn = functools.partial(
        anp_collate,
        max_aux_snapshots=int(args.max_aux_snapshots),
        aux_halo_frac=float(args.aux_halo_frac),
        target_snapnum=(None if not temporal_holdout_enabled else holdout_snap),
    )

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
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=loader_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    raw_x_dim = train_tasks[0].snapshots[0].x.shape[-1]
    radius_fourier_n_freq = 0 if args.disable_radius_fourier else int(args.radius_fourier_n_freq)
    radius_fourier_extra_dim = (2 * radius_fourier_n_freq) if radius_fourier_n_freq > 0 else 0
    x_dim = raw_x_dim + radius_fourier_extra_dim
    if main_proc:
        print(
            f"Input dimensions: raw_x_dim={raw_x_dim}, model_x_dim={x_dim}, "
            f"radius_fourier_n_freq={radius_fourier_n_freq}"
        )

    # Resolve ideal-gas channel indices from target_names when all three
    # thermodynamic channels are present and weight > 0.
    ideal_gas_channel_indices: Optional[Tuple[int, int, int]] = None
    if args.ideal_gas_weight > 0.0:
        _ig_names = ("gas_density", "temperature", "pressure")
        if all(n in target_names for n in _ig_names):
            ideal_gas_channel_indices = (
                target_names.index("gas_density"),
                target_names.index("temperature"),
                target_names.index("pressure"),
            )
            if main_proc:
                print(
                    f"Ideal-gas penalty enabled: weight={args.ideal_gas_weight}, "
                    f"channel indices rho={ideal_gas_channel_indices[0]} "
                    f"T={ideal_gas_channel_indices[1]} P={ideal_gas_channel_indices[2]}"
                )
        else:
            if main_proc:
                print(
                    f"WARNING: --ideal-gas-weight={args.ideal_gas_weight} but not all "
                    f"channels (gas_density, temperature, pressure) are present in "
                    f"target_names={target_names}. Disabling ideal-gas penalty."
                )
            args.ideal_gas_weight = 0.0

    model = StrongANP(
        x_dim=x_dim,
        raw_x_dim=raw_x_dim,
        y_dim=train_tasks[0].snapshots[0].y.shape[-1],
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
        core_bias_weight=args.core_bias_weight,
        num_snapshots=len(args.resolved_snapnums),
        time_feature_scale=args.time_feature_scale,
        ideal_gas_weight=args.ideal_gas_weight,
        ideal_gas_channel_indices=ideal_gas_channel_indices,
        cc_dual_head=bool(getattr(args, "cc_dual_head", False)),
        cc_indicator_feature_idx=int(getattr(args, "cc_indicator_feature_idx", -1)),
        decoder_likelihood=str(args.decoder_likelihood),
        student_t_df=float(args.student_t_df),
        context_dropout_rate=float(getattr(args, "context_dropout_rate", 0.0)),
        input_noise_std=float(getattr(args, "input_noise_std", 0.0)),
        beta_nll_weight=float(getattr(args, "beta_nll_weight", 0.0)),
        free_bits=float(getattr(args, "free_bits", 0.0)),
        snapshot_balanced_loss=bool(getattr(args, "snapshot_balanced_loss", False)),
        stratified_var_cal_weight=float(getattr(args, "stratified_var_cal_weight", 0.0)),
        stratified_mass_bins=int(getattr(args, "stratified_mass_bins", 4)),
        stratified_radius_bins=int(getattr(args, "stratified_radius_bins", 4)),
        stratified_min_points=int(getattr(args, "stratified_min_points", 16)),
    ).to(device)

    # Populate the y_std buffer so the ideal-gas penalty can convert normalised
    # errors back to log10 space.
    unwrap_model(model).set_y_std(y_std)

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
            collate_fn=collate_fn,
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
                detailed_metrics_zeroshot = evaluate_test_metrics(
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
                    force_zeroshot_context=True,
                )
                row["val_rmse_original_units"] = float(detailed_metrics["rmse_original_units"])
                row["val_nll_original_units"] = float(detailed_metrics["nll_original_units"])
                row["val_rmse_core_original_units"] = float(detailed_metrics["rmse_core_original_units"])
                row["val_rmse_outer_original_units"] = float(detailed_metrics["rmse_outer_original_units"])
                row["val_zeroshot_rmse_original_units"] = float(detailed_metrics_zeroshot["rmse_original_units"])
                row["val_zeroshot_nll_original_units"] = float(detailed_metrics_zeroshot["nll_original_units"])

                per_target = detailed_metrics.get("per_target", {})
                if "pressure" in per_target:
                    row["val_pressure_rmse_original_units"] = float(per_target["pressure"]["rmse_original_units"])
                    row["val_pressure_nll_original_units"] = float(per_target["pressure"]["nll_original_units"])
                    row["val_pressure_rmse_core_original_units"] = float(per_target["pressure"].get("rmse_core_original_units", float("nan")))
                if "temperature" in per_target:
                    row["val_temperature_rmse_original_units"] = float(per_target["temperature"]["rmse_original_units"])
                    row["val_temperature_nll_original_units"] = float(per_target["temperature"]["nll_original_units"])
                    row["val_temperature_rmse_core_original_units"] = float(per_target["temperature"].get("rmse_core_original_units", float("nan")))

                per_snap = detailed_metrics.get("per_snapshot", {})
                for snap_k, snap_v in per_snap.items():
                    row[f"val_snap{snap_k}_rmse"] = float(snap_v["rmse_original_units"])

                row["val_weighted_orig"] = compute_weighted_selection_score(
                    detailed_metrics,
                    target_names=target_names,
                    pressure_w=float(args.selection_pressure_weight),
                    temperature_w=float(args.selection_temperature_weight),
                    pressure_core_w=float(args.selection_pressure_core_weight),
                    temperature_core_w=float(args.selection_temperature_core_weight),
                )
                row["val_zeroshot_weighted_orig"] = compute_weighted_selection_score(
                    detailed_metrics_zeroshot,
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
                snap_parts = []
                if detailed_metrics is not None:
                    for ks, vs in detailed_metrics.get("per_snapshot", {}).items():
                        col = f"val_snap{ks}_rmse"
                        if col in row:
                            snap_parts.append(f"snap{ks}={row[col]:.4g}")
                if snap_parts:
                    print(f"  val(per-snap) {' '.join(snap_parts)}")
            if "val_zeroshot_rmse_original_units" in row:
                print(
                    f"  val(zero-shot) rmse={row['val_zeroshot_rmse_original_units']:.4g} "
                    f"nll={row['val_zeroshot_nll_original_units']:.4g} "
                    f"weighted={row.get('val_zeroshot_weighted_orig', float('nan')):.4g}"
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
                        "mean_prior_source": mean_prior_source,
                        "mean_checkpoint": mean_checkpoint_used,
                        **_mean_model_checkpoint_entries(mean_model),
                        "cc_prior": cc_prior_stats,
                        "cc_predictor_state_dict": None if cc_predictor_model is None else cc_predictor_model.state_dict(),
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
            collate_fn=collate_fn,
            drop_last=False,
        )

        holdout_loader_eval = None
        if temporal_holdout_enabled and holdout_test_ds is not None and len(holdout_test_ds) > 0:
            holdout_loader_eval = DataLoader(
                holdout_test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=holdout_collate_fn,
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
        test_metrics_zeroshot = evaluate_test_metrics(
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
            force_zeroshot_context=True,
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
        print("Test zero-shot metrics:", test_metrics_zeroshot)
        print(f"Coverage@1sigma={cov1:.3f}, Coverage@2sigma={cov2:.3f}")
        print("Few-shot RMSE:", fshot)
        print("Few-shot relative improvement (%):", rel)

        temporal_holdout_metrics = None
        temporal_holdout_metrics_zeroshot = None
        if holdout_loader_eval is not None:
            temporal_holdout_metrics = evaluate_test_metrics(
                eval_model,
                holdout_loader_eval,
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
            temporal_holdout_metrics_zeroshot = evaluate_test_metrics(
                eval_model,
                holdout_loader_eval,
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
                force_zeroshot_context=True,
            )
            print(f"Temporal holdout (snap={holdout_snap}) metrics:", temporal_holdout_metrics)
            print(f"Temporal holdout (snap={holdout_snap}) zero-shot metrics:", temporal_holdout_metrics_zeroshot)

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
                "mean_prior_source": mean_prior_source,
                "mean_checkpoint": mean_checkpoint_used,
                **_mean_model_checkpoint_entries(mean_model),
                "cc_prior": cc_prior_stats,
                "cc_predictor_state_dict": None if cc_predictor_model is None else cc_predictor_model.state_dict(),
            },
            ckpt_path,
        )

        metrics = {
            "target_name": args.target_name,
            "training_stage": args.training_stage,
            "n_tasks_total": len(tasks),
            "n_train": len(train_tasks),
            "n_val": len(val_tasks),
            "n_test": len(test_tasks),
            "n_train_snapshots": int(sum(len(f.snapshots) for f in train_tasks)),
            "n_val_snapshots": int(sum(len(f.snapshots) for f in val_tasks)),
            "n_test_snapshots": int(sum(len(f.snapshots) for f in test_tasks)),
            "temporal_holdout": {
                "enabled": bool(temporal_holdout_enabled),
                "snapnum": int(holdout_snap) if temporal_holdout_enabled else None,
                "require_context": bool(args.temporal_holdout_require_context),
                "n_holdout_target_snapshots": (
                    0
                    if holdout_eval_raw is None
                    else int(sum(1 for fam in holdout_eval_raw for s in fam.snapshots if int(s.snapnum) == holdout_snap))
                ),
                "metrics": temporal_holdout_metrics,
                "metrics_zeroshot": temporal_holdout_metrics_zeroshot,
            },
            "select_metric": select_metric,
            "best_epoch": best["epoch"],
            "best_score": best["score"],
            "best_val_loss": best["val_loss"],
            "best_val_rmse_norm": best["val_rmse_norm"],
            "test": test_metrics,
            "test_zeroshot": test_metrics_zeroshot,
            "coverage": {"1sigma": cov1, "2sigma": cov2},
            "fewshot_rmse": fshot,
            "fewshot_rel_improvement_pct": rel,
            "mean_prior": {
                "enabled": mean_prior_enabled,
                "source": mean_prior_source,
                "checkpoint": mean_checkpoint_used,
                "hidden_dim": args.mean_hidden_dim,
                "epochs": args.mean_epochs,
                "lr": args.mean_lr,
                "weight_decay": args.mean_weight_decay,
                "metrics": mean_metrics_summary,
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
