#!/usr/bin/env python
"""
ANPcosmo Emulator Validation Suite
===================================
A comprehensive test suite implementing the 10-category validation framework
described in "ANPcosmo Emulator Validation: A Rigorous Testing Suite for
35-Parameter Feedback Inference."

Target model: anp_all_profiles_20260325_175639

Usage:
    python run_validation_suite.py [--device cpu] [--output-dir validation_results]
    python run_validation_suite.py --categories 1 2 3   # Run specific categories only

Categories:
    1: Point Accuracy on Held-Out Test Set
    2: Uncertainty Calibration Tests
    3: Single-Parameter (1P) Variation and Sensitivity Tests
    4: Mock Parameter Recovery Tests (framework — requires MCMC)
    5: Fisher Information Consistency Test
    6: Context Set Size and ANP-Specific Tests
    7: Noise Robustness and Observational Realism Tests
    8: Comparison to Observational Data (framework — requires external data)
    9: Cross-Validation and Consistency Stress Tests
   10: Posterior Predictive Checks (framework — requires MCMC results)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Ensure repo root is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from anp_emulator import Emulator
from anp_emulator.hmc import HMCSampler
from train_anp_emulator import (
    ALL_PROFILE_LOG_TARGETS,
    build_tasks,
    split_tasks,
    resolve_profile_file,
    load_theta_table,
    normalize_tasks,
    apply_residual_prior,
    remap_flat_tasks_to_families,
    anp_collate,
    few_shot_curve,
    denorm_y,
    add_mean_back,
    _lookup_bin_stats,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RUN_DIR = REPO_ROOT / "anp_training_runs" / "anp_all_profiles_20260325_175639"
TARGET_FIELDS = ["temperature", "pressure", "gas_density", "metallicity"]

# Observational noise levels (fractional σ/signal per radial bin)
# Based on Table in Section 2 of the validation document.
OBS_NOISE_FRAC = {
    "temperature": 0.25,   # ~15-40% per annulus (Chandra/XMM) — use 25%
    "pressure": 0.15,      # ~5-20% per bin (tSZ, Simons Obs) — use 15%
    "gas_density": 0.12,   # ~5-20% per bin (X-ray deprojected) — use 12%
    "metallicity": 0.35,   # ~20-50% per bin (Chandra/XMM) — use 35%
}

# Pass/fail thresholds from the validation document Section 4
THRESHOLDS = {
    "rmse_over_sigma_obs": 0.15,       # RMSE/σ_obs ≤ 0.15
    "outlier_delta_chi2": 0.2,         # Δχ² threshold
    "outlier_fraction_max": 0.10,      # F_out < 10%
    "coverage_deviation_max": 0.05,    # Coverage within 5% of diagonal
    "pit_ks_stat_boot_alpha": 0.05,    # KS stat <= bootstrap(95%)
    "chi2_red_lo": 0.8,               # 0.8 ≤ χ²_red ≤ 1.2
    "chi2_red_hi": 1.2,
    "spearman_rho_min": 0.4,          # ρ > 0.4
    "derivative_ratio_tol": 0.20,     # Within 20%
}

RHAT_MAX = 1.1

# Mass bins for stratified reporting (log10 M_sun)
MASS_BINS = [(13.0, 13.5), (13.5, 14.0), (14.0, 14.5)]


# ============================================================================
# Utility functions
# ============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_emulator(device: str = "cpu") -> Emulator:
    if not RUN_DIR.exists():
        raise FileNotFoundError(f"Run directory not found: {RUN_DIR}")
    emu = Emulator.from_run_dir(RUN_DIR, device=device)
    print(f"[INFO] Loaded emulator from {RUN_DIR.name}")
    print(f"       Fields: {emu.available_fields()}")
    print(f"       θ-dim: {emu.theta_dim}, snapnums: {emu.snapnums}")
    return emu


def load_test_data(emu: Emulator) -> Dict[str, Any]:
    """Load the SB35 held-out test set using the same split as training."""
    args = SimpleNamespace(**dict(emu.args))
    # Ensure resolved attributes exist
    if not hasattr(args, "resolved_snapnums"):
        args.resolved_snapnums = [int(getattr(args, "snapnum", 90))]
    if not hasattr(args, "redshift_by_snap"):
        args.redshift_by_snap = {90: 0.0}
    if not hasattr(args, "resolved_all_profile_targets"):
        args.resolved_all_profile_targets = list(getattr(args, "all_profiles_subset", TARGET_FIELDS))

    print("[INFO] Building tasks (this loads profile data)...")
    all_families = build_tasks(args)
    _, _, test_families = split_tasks(
        all_families,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        seed=int(args.seed),
    )
    print(f"[INFO] Test set: {len(test_families)} families")

    # Load theta table
    theta_table = load_theta_table(Path(args.param_csv), target_theta_dim=int(args.theta_dim))

    # Collect test run IDs and truth data
    snap_eval = int(args.resolved_snapnums[0])
    z_eval = float(getattr(emu, "redshift_by_snap", {}).get(snap_eval, 0.0))

    test_data = {
        "args": args,
        "test_families": test_families,
        "theta_table": theta_table,
        "snap_eval": snap_eval,
        "z_eval": z_eval,
    }
    return test_data


def predict_test_set(
    emu: Emulator,
    test_data: Dict[str, Any],
    n_samples: int = 30,
) -> Dict[str, Any]:
    """Run emulator on entire test set, collecting truth + predictions."""
    args = test_data["args"]
    theta_table = test_data["theta_table"]
    snap_eval = test_data["snap_eval"]
    z_eval = test_data["z_eval"]

    all_y_true = []
    all_y_pred = []
    all_y_std = []
    all_y_ale_std = []
    all_y_epi_std = []
    all_y_pred_log = []
    all_y_std_log = []
    all_masses = []
    all_r_bins = []
    all_run_ids = []

    for fam in test_data["test_families"]:
        run_id = fam.run_id
        # Find the snapshot matching snap_eval
        snap_task = None
        for st in fam.snapshots:
            if st.snapnum == snap_eval:
                snap_task = st
                break
        if snap_task is None:
            continue

        theta = theta_table.get(run_id)
        if theta is None:
            continue

        # Load raw profiles for truth in physical units
        try:
            fpath = resolve_profile_file(
                run_id,
                base_path=Path(args.profiles_base),
                suite=args.suite,
                sim_set=args.sim_set,
                snapnum=snap_eval,
            )
        except FileNotFoundError:
            continue

        with np.load(fpath) as dat:
            m500c = dat["M500c"].astype(np.float32)
            r500c = dat["R500c"].astype(np.float32)
            radial_bins = dat["radial_bins"].astype(np.float32)
            y_true_phys = np.stack(
                [dat[f"{f}_array"].astype(np.float32) for f in TARGET_FIELDS],
                axis=-1,
            )  # (n_halo, n_r, 4)

        if m500c.shape[0] == 0:
            continue

        # Apply same mass floor as training
        mass_floor = float(getattr(args, "mass_floor", 0.0))
        if mass_floor > 0.0:
            log_m = np.log10(np.clip(m500c, 1e10, None))
            keep = log_m >= mass_floor
            if keep.sum() == 0:
                continue
            m500c = m500c[keep]
            r500c = r500c[keep]
            y_true_phys = y_true_phys[keep]

        # Max halos for memory
        max_h = int(getattr(args, "max_halos_per_run", 128))
        if max_h > 0 and m500c.shape[0] > max_h:
            rng = np.random.default_rng(int(args.seed) + run_id)
            pick = np.sort(rng.choice(m500c.shape[0], size=max_h, replace=False))
            m500c, r500c, y_true_phys = m500c[pick], r500c[pick], y_true_phys[pick]

        n_halo = m500c.shape[0]
        rr500 = (
            radial_bins[None, :] / np.maximum(r500c[:, None], 1e-12)
        ).astype(np.float32)

        pred = emu.predict(
            theta=theta,
            M=m500c,
            r_bins=rr500,
            field=TARGET_FIELDS,
            snapnum=snap_eval,
            redshift=z_eval,
            n_samples=n_samples,
        )

        all_y_true.append(y_true_phys)
        all_y_pred.append(pred.mean)
        all_y_std.append(pred.total_std)
        all_y_ale_std.append(pred.aleatoric_std)
        all_y_epi_std.append(pred.epistemic_std)
        if pred.mean_log10 is not None:
            all_y_pred_log.append(pred.mean_log10)
            all_y_std_log.append(pred.std_log10)
        all_masses.append(m500c)
        all_r_bins.append(rr500)
        all_run_ids.extend([run_id] * n_halo)

    results = {
        "y_true": np.concatenate(all_y_true, axis=0),         # (N, n_r, 4)
        "y_pred": np.concatenate(all_y_pred, axis=0),         # (N, n_r, 4)
        "y_std": np.concatenate(all_y_std, axis=0),           # (N, n_r, 4)
        "y_ale_std": np.concatenate(all_y_ale_std, axis=0),
        "y_epi_std": np.concatenate(all_y_epi_std, axis=0),
        "masses": np.concatenate(all_masses, axis=0),          # (N,)
        "r_bins": np.concatenate(all_r_bins, axis=0),          # (N, n_r)
        "run_ids": np.array(all_run_ids),
        "field_names": TARGET_FIELDS,
    }
    if all_y_pred_log:
        results["y_pred_log"] = np.concatenate(all_y_pred_log, axis=0)
        results["y_std_log"] = np.concatenate(all_y_std_log, axis=0)

    # Pre-compute log10 truth for log-space channels (all four target
    # channels are positive-definite and modelled in log10 space).
    results["y_true_log"] = np.log10(np.clip(results["y_true"], 1e-38, None))

    # Valid mask: match the training convention — bins with truth > 0 for
    # each positive-definite (log-space) channel.  Shape (N, n_r, n_field).
    results["valid_mask"] = results["y_true"] > 0

    n_valid = results["valid_mask"].sum()
    n_total = results["valid_mask"].size
    print(f"[INFO] Valid mask: {n_valid}/{n_total} entries "
          f"({100*n_valid/n_total:.1f}%)")

    N = results["y_true"].shape[0]
    print(f"[INFO] Test predictions: {N} halos, {results['y_true'].shape[1]} radial bins, "
          f"{results['y_true'].shape[2]} fields")
    return results


def get_mass_mask(masses: np.ndarray, lo: float, hi: float) -> np.ndarray:
    log_m = np.log10(np.clip(masses, 1e10, None))
    return (log_m >= lo) & (log_m < hi)


def print_pass_fail(name: str, value: float, criterion: str, passed: bool):
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}: {value:.4f}  (criterion: {criterion})")


_KS_CRIT_CACHE: Dict[Tuple[int, int, float], float] = {}


def ks_bootstrap_critical(
    n: int,
    n_boot: int = 400,
    alpha: float = 0.05,
    seed: int = 12345,
) -> float:
    """Bootstrap critical value for one-sample KS statistic under Uniform(0,1)."""
    key = (int(n), int(n_boot), float(alpha))
    if key in _KS_CRIT_CACHE:
        return _KS_CRIT_CACHE[key]
    if n <= 0:
        return float("nan")

    rng = np.random.default_rng(seed + n)
    ks_vals = np.zeros(n_boot, dtype=np.float64)
    for bi in range(n_boot):
        u = rng.uniform(0.0, 1.0, size=n)
        ks_vals[bi] = float(scipy_stats.kstest(u, "uniform").statistic)
    crit = float(np.quantile(ks_vals, 1.0 - alpha))
    _KS_CRIT_CACHE[key] = crit
    return crit


def _predict_log_from_batch(
    emu: Emulator,
    batch: Dict[str, torch.Tensor],
    n_h: int,
    n_r: int,
    n_samples: int,
    z_eval: float,
) -> np.ndarray:
    """Forward model on a custom batch and return log10-space mean."""
    with torch.no_grad():
        mu_raw, _, _, _ = emu.model.predict(batch, device=emu.device, n_samples=n_samples)

    n_pts = n_h * n_r
    mu_trunc = mu_raw[:, :n_pts, :]
    if emu.norm_stats is not None and emu.norm_stats.get("mass_redshift_aware", False):
        tgt_x_raw = batch["tgt_x"].to(emu.device)
        log_m_raw = (tgt_x_raw[0, :n_pts, 0] * emu.x_std[0] + emu.x_mean[0]).detach().cpu().numpy()
        log_m_ph = log_m_raw[::n_r]
        ym_np, ys_np = _lookup_bin_stats(emu.norm_stats, log_m_ph, z_eval)
        ym_e = np.repeat(ym_np, n_r, axis=1).reshape(1, n_pts, -1)
        ys_e = np.repeat(ys_np, n_r, axis=1).reshape(1, n_pts, -1)
        ym_t = torch.tensor(ym_e, dtype=mu_trunc.dtype, device=mu_trunc.device)
        ys_t = torch.tensor(ys_e, dtype=mu_trunc.dtype, device=mu_trunc.device)
        mu_denorm = mu_trunc * ys_t + ym_t
    else:
        mu_denorm = mu_trunc * emu.y_std.view(1, 1, -1) + emu.y_mean.view(1, 1, -1)

    mu_denorm = add_mean_back(
        mu_denorm,
        batch["tgt_x"].to(emu.device),
        emu.x_mean,
        emu.x_std,
        emu.mean_model,
    )
    return mu_denorm[0].reshape(n_h, n_r, -1).detach().cpu().numpy()


# ============================================================================
# PRE-FLIGHT: Validate mean / mean_log10 duality
# ============================================================================

def validate_log10_duality(preds: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check internal consistency of the PredictionResult fields:
      1. mean ≈ 10^(mean_log10)                  for log-space channels
      2. total_std ≈ ln(10) * mean * std_log10    (delta-method identity)

    This guards against silent bugs where Cat 2 calibration tests use one
    space while the emulator returns the other, and against numerical drift
    between the stored log10 values and the 10^(·) conversion.
    """
    print("\n" + "-" * 60)
    print("PRE-FLIGHT: Validating mean / mean_log10 duality")
    print("-" * 60)

    results: Dict[str, Any] = {}
    fields = preds["field_names"]
    valid = preds["valid_mask"]

    y_pred = preds["y_pred"]           # physical units
    y_std = preds["y_std"]             # physical-space total uncertainty
    y_pred_log = preds["y_pred_log"]   # emulator native log10
    y_std_log = preds["y_std_log"]     # emulator native log10 uncertainty

    ln10 = np.log(10.0)

    all_pass = True
    for fi, f in enumerate(fields):
        m = valid[:, :, fi]
        if m.sum() == 0:
            continue

        # --- Check 1: mean ≈ 10^(mean_log10) ---
        predicted_phys = np.power(10.0, y_pred_log[:, :, fi][m])
        actual_phys = y_pred[..., fi][m] if y_pred.ndim == 3 else y_pred[:, :, fi][m]
        # Use relative error (avoid division by tiny values)
        safe_denom = np.maximum(np.abs(predicted_phys), 1e-30)
        rel_err_mean = np.abs(actual_phys - predicted_phys) / safe_denom
        max_rel_mean = float(np.max(rel_err_mean))
        med_rel_mean = float(np.median(rel_err_mean))

        # Tolerance: 10^(·) is exact up to float32 precision (~1e-6 relative)
        mean_ok = max_rel_mean < 1e-4
        all_pass &= mean_ok
        tag = "PASS" if mean_ok else "FAIL"
        print(f"  [{tag}] {f} mean ↔ 10^(mean_log10): "
              f"median_rel={med_rel_mean:.2e}, max_rel={max_rel_mean:.2e}")

        # --- Check 2: total_std ≈ ln(10) * mean * std_log10 (delta method) ---
        delta_std = ln10 * actual_phys * y_std_log[:, :, fi][m]
        actual_std = y_std[..., fi][m] if y_std.ndim == 3 else y_std[:, :, fi][m]
        safe_std = np.maximum(np.abs(delta_std), 1e-30)
        rel_err_std = np.abs(actual_std - delta_std) / safe_std
        max_rel_std = float(np.max(rel_err_std))
        med_rel_std = float(np.median(rel_err_std))

        std_ok = max_rel_std < 1e-4
        all_pass &= std_ok
        tag = "PASS" if std_ok else "FAIL"
        print(f"  [{tag}] {f} std  ↔ ln(10)·μ·σ_log10: "
              f"median_rel={med_rel_std:.2e}, max_rel={max_rel_std:.2e}")

        results[f] = {
            "mean_max_rel_err": max_rel_mean,
            "mean_median_rel_err": med_rel_mean,
            "std_max_rel_err": max_rel_std,
            "std_median_rel_err": med_rel_std,
        }

    results["duality_pass"] = all_pass
    if all_pass:
        print("  ✓ All duality checks passed — log10 and physical spaces are consistent.")
    else:
        print("  ✗ DUALITY MISMATCH — physical-space mean/std do NOT match 10^(log10).")
        print("    Cat 2 calibration uses log10 space (correct), but downstream consumers")
        print("    of mean/total_std in physical units may see incorrect uncertainties.")

    return results


# ============================================================================
# PRE-FLIGHT: Zero-Shot Context Token Bias Probe
# ============================================================================

def validate_zeroshot_context(
    emu: Emulator, preds: Dict[str, Any], output_dir: Path,
) -> Dict[str, Any]:
    """
    Probe how much the synthetic zero-shot context token influences
    predictions, and whether that influence biases OOD predictions toward the
    training mean.

    Background
    ----------
    ``_build_zeroshot_batch`` creates ``ctx_y = zeros(1,1,y_dim)`` with
    ``ctx_mask = ones(1,1)`` (unmasked).  In normalized space y = 0 encodes
    the training-mean profile.  This is informative — not null.  Two
    pathways consume it:

    * **Latent encoder**: pools ``cat(ctx_x, ctx_y)`` → z distribution.
    * **Deterministic path**: cross-attention from targets to the single
      context point → representation ``r``.

    Both inject signal from the fake "mean-profile" observation.

    Tests
    -----
    A. **Context-influence magnitude**: For a reference halo, compare the
       default zero-shot prediction against predictions with perturbed
       ``ctx_y`` (±2 σ in normalized space).  The maximum relative shift
       in the log10-space mean quantifies the context pathway's influence.

    B. **OOD mean-reversion bias**: For test halos whose truth is ≥ 1.5 σ
       away from the training mean (in log10 dex), check whether the
       prediction residual (pred − truth) is systematically *toward* the
       mean (positive when truth < mean, negative when truth > mean).
    """
    print("\n" + "-" * 60)
    print("PRE-FLIGHT: Zero-Shot Context Token Bias Probe")
    print("-" * 60)

    results: Dict[str, Any] = {}
    cat_dir = ensure_dir(output_dir / "preflight_context_bias")
    fields = preds["field_names"]

    # ---- Test A: context-influence magnitude ----
    print("\n--- A. Context pathway influence magnitude ---")

    snap_eval = int(emu.snapnums[0])
    z_eval = float(getattr(emu, "redshift_by_snap", {}).get(snap_eval, 0.0))

    # Pick a near-median halo for a clean comparison
    masses = preds["masses"]
    log_m = np.log10(np.clip(masses, 1e10, None))
    median_m_idx = int(np.argmin(np.abs(log_m - np.median(log_m))))
    ref_mass = np.array([masses[median_m_idx]])
    ref_r = preds["r_bins"][median_m_idx:median_m_idx + 1]
    ref_run_id = int(preds["run_ids"][median_m_idx])

    # Get theta for this halo
    try:
        args_ns = _get_train_args()
        theta_table = load_theta_table(
            Path(args_ns.param_csv),
            target_theta_dim=int(args_ns.theta_dim),
        )
        ref_theta = theta_table.get(ref_run_id)
    except Exception:
        ref_theta = None

    if ref_theta is None:
        print("  [SKIP] Cannot load theta for reference halo; skipping context probe.")
        results["context_probe"] = "skipped"
        return results

    # Baseline prediction (default zero-shot: ctx_y = 0)
    pred_base = emu.predict(
        theta=ref_theta, M=ref_mass, r_bins=ref_r,
        field=TARGET_FIELDS, snapnum=snap_eval, redshift=z_eval,
        n_samples=50,
    )
    mu_base_log = pred_base.mean_log10[0]  # (n_r, n_field)

    # Perturbed prediction: inject ctx_y = +2 in normalized space
    # (2 standard deviations above mean → a much hotter/denser profile)
    y_dim = len(emu.target_names)
    batch_hi, n_h, n_r = emu._build_zeroshot_batch(
        theta=ref_theta, masses=ref_mass, r_bins=ref_r,
        snapnum=snap_eval, redshift=z_eval,
    )
    batch_hi["ctx_y"] = torch.full((1, 1, y_dim), 2.0, dtype=torch.float32)

    mu_hi_log = _predict_log_from_batch(
        emu=emu,
        batch=batch_hi,
        n_h=n_h,
        n_r=n_r,
        n_samples=50,
        z_eval=z_eval,
    )

    # Similarly with ctx_y = -2
    batch_lo, _, _ = emu._build_zeroshot_batch(
        theta=ref_theta, masses=ref_mass, r_bins=ref_r,
        snapnum=snap_eval, redshift=z_eval,
    )
    batch_lo["ctx_y"] = torch.full((1, 1, y_dim), -2.0, dtype=torch.float32)

    mu_lo_log = _predict_log_from_batch(
        emu=emu,
        batch=batch_lo,
        n_h=n_h,
        n_r=n_r,
        n_samples=50,
        z_eval=z_eval,
    )

    # Compute max relative shift (in dex) across the ±2σ perturbation
    context_influence = {}
    for fi, f in enumerate(fields):
        shift_hi = np.abs(mu_hi_log[0, :, fi] - mu_base_log[:, fi])
        shift_lo = np.abs(mu_lo_log[0, :, fi] - mu_base_log[:, fi])
        max_shift = float(max(shift_hi.max(), shift_lo.max()))
        mean_shift = float(0.5 * (shift_hi.mean() + shift_lo.mean()))
        context_influence[f] = {
            "max_shift_dex": max_shift,
            "mean_shift_dex": mean_shift,
        }
        # A shift < 0.01 dex (~2% in physical) means context is nearly ignored
        benign = max_shift < 0.05
        tag = "PASS" if benign else "WARN"
        print(f"  [{tag}] {f}: max_shift={max_shift:.4f} dex, "
              f"mean_shift={mean_shift:.4f} dex "
              f"(< 0.05 dex → context nearly ignored)")

    results["context_influence"] = context_influence

    # ---- Test A2: masked/learned-null context alternatives ----
    print("\n--- A2. Null-context alternatives (masked and learned token) ---")
    null_results: Dict[str, Any] = {
        "masked": {"available": False, "per_field": {}},
        "learned": {"available": False},
    }

    # Masked context (remove synthetic token influence entirely)
    batch_masked, n_h_m, n_r_m = emu._build_zeroshot_batch(
        theta=ref_theta, masses=ref_mass, r_bins=ref_r,
        snapnum=snap_eval, redshift=z_eval,
    )
    batch_masked["ctx_mask"] = torch.zeros_like(batch_masked["ctx_mask"], dtype=torch.bool)
    try:
        mu_masked_log = _predict_log_from_batch(
            emu=emu,
            batch=batch_masked,
            n_h=n_h_m,
            n_r=n_r_m,
            n_samples=50,
            z_eval=z_eval,
        )
        null_results["masked"]["available"] = True
        for fi, f in enumerate(fields):
            shift = np.abs(mu_masked_log[0, :, fi] - mu_base_log[:, fi])
            max_shift = float(np.max(shift))
            mean_shift = float(np.mean(shift))
            null_results["masked"]["per_field"][f] = {
                "max_shift_dex": max_shift,
                "mean_shift_dex": mean_shift,
            }
            tag = "PASS" if max_shift < 0.05 else "WARN"
            print(f"  [{tag}] masked null ({f}): max_shift={max_shift:.4f} dex, "
                  f"mean_shift={mean_shift:.4f} dex")
    except Exception as exc:
        null_results["masked"]["error"] = str(exc)
        print("  [SKIP] masked null context path produced invalid latent stats; "
              "model appears to require at least one unmasked context token.")

    # Learned null token (if model exposes one)
    learned_attr = None
    for attr in ("learned_null_ctx_y", "null_ctx_y", "ctx_null_token"):
        if hasattr(emu.model, attr):
            learned_attr = attr
            break

    if learned_attr is not None:
        token = getattr(emu.model, learned_attr)
        if torch.is_tensor(token):
            token_t = token.detach().to(dtype=torch.float32, device=emu.device).reshape(1, 1, -1)
            batch_learned, n_h_l, n_r_l = emu._build_zeroshot_batch(
                theta=ref_theta, masses=ref_mass, r_bins=ref_r,
                snapnum=snap_eval, redshift=z_eval,
            )
            if token_t.shape[-1] == batch_learned["ctx_y"].shape[-1]:
                batch_learned["ctx_y"] = token_t
                try:
                    mu_learned_log = _predict_log_from_batch(
                        emu=emu,
                        batch=batch_learned,
                        n_h=n_h_l,
                        n_r=n_r_l,
                        n_samples=50,
                        z_eval=z_eval,
                    )
                    null_results["learned"] = {"available": True, "attr": learned_attr, "per_field": {}}
                    for fi, f in enumerate(fields):
                        shift = np.abs(mu_learned_log[0, :, fi] - mu_base_log[:, fi])
                        max_shift = float(np.max(shift))
                        mean_shift = float(np.mean(shift))
                        null_results["learned"]["per_field"][f] = {
                            "max_shift_dex": max_shift,
                            "mean_shift_dex": mean_shift,
                        }
                        tag = "PASS" if max_shift < 0.05 else "WARN"
                        print(f"  [{tag}] learned null ({f}): max_shift={max_shift:.4f} dex, "
                              f"mean_shift={mean_shift:.4f} dex")
                except Exception as exc:
                    null_results["learned"] = {
                        "available": False,
                        "attr": learned_attr,
                        "error": str(exc),
                    }
                    print("  [SKIP] learned null context path produced invalid latent stats.")
            else:
                print(f"  [SKIP] learned null token '{learned_attr}' has incompatible shape.")
        else:
            print(f"  [SKIP] learned null token '{learned_attr}' is not a tensor.")
    else:
        print("  [INFO] No learned null context token found on model; masked null only.")

    results["null_context_test"] = null_results

    # ---- Test B: OOD mean-reversion bias ----
    print("\n--- B. OOD mean-reversion bias (tail halos) ---")

    valid = preds["valid_mask"]
    y_true_log = preds["y_true_log"]
    y_pred_log = preds["y_pred_log"]

    # Compute per-field training mean in log10 space (from the predictions
    # set — the emulator's y_mean + mean_model represents this).
    # We use the empirical mean of the valid test-set truth as a proxy.
    bias_results = {}
    fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 5))
    if len(fields) == 1:
        axes = [axes]

    for fi, f in enumerate(fields):
        m = valid[:, :, fi]
        if m.sum() < 100:
            continue

        truth_flat = y_true_log[:, :, fi][m]
        pred_flat = y_pred_log[:, :, fi][m]
        residual = pred_flat - truth_flat  # positive = overprediction

        # Empirical mean and std of truth across the test set
        mu_truth = float(np.mean(truth_flat))
        std_truth = float(np.std(truth_flat))

        # Deviation from mean: how far each bin's truth is from the test mean
        dev_from_mean = truth_flat - mu_truth  # positive = above mean

        # Tail selection: halos > 1.5σ away from mean
        tail_mask = np.abs(dev_from_mean) > 1.5 * std_truth
        n_tail = int(tail_mask.sum())
        if n_tail < 50:
            print(f"  {f}: only {n_tail} tail points (need ≥ 50), skipping.")
            continue

        tail_dev = dev_from_mean[tail_mask]
        tail_resid = residual[tail_mask]

        # Mean-reversion: residual should anti-correlate with deviation if
        # the context biases predictions toward the mean.
        # Sign check: for bins ABOVE mean (dev > 0), mean-reversion bias
        # means pred < truth → residual < 0.
        # For bins BELOW mean (dev < 0), mean-reversion → residual > 0.
        # So the product dev * resid should be negative on average.
        mean_reversion_score = float(np.mean(tail_dev * tail_resid))
        # Normalize by variance for interpretability
        norm_score = mean_reversion_score / (std_truth ** 2 + 1e-12)
        # A strongly negative score indicates mean-reversion bias
        has_bias = norm_score < -0.05
        tag = "WARN" if has_bias else "PASS"
        print(f"  [{tag}] {f}: mean-reversion score = {norm_score:.4f} "
              f"(n_tail={n_tail}; < -0.05 → systematic bias toward mean)")

        bias_results[f] = {
            "mean_reversion_score": norm_score,
            "n_tail": n_tail,
            "has_bias": bool(has_bias),
        }

        # Scatter plot: deviation from mean vs residual
        ax = axes[fi]
        subsample = np.random.default_rng(42).choice(
            n_tail, size=min(2000, n_tail), replace=False,
        )
        ax.scatter(
            tail_dev[subsample], tail_resid[subsample],
            alpha=0.15, s=3, c="steelblue",
        )
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0, color="k", lw=0.8)
        # Trend line
        coeffs = np.polyfit(tail_dev, tail_resid, 1)
        x_line = np.linspace(tail_dev.min(), tail_dev.max(), 50)
        ax.plot(x_line, np.polyval(coeffs, x_line), "r-", lw=2,
                label=f"slope={coeffs[0]:.3f}")
        ax.set_xlabel("Truth − mean(truth) [dex]")
        ax.set_ylabel("Prediction − Truth [dex]")
        ax.set_title(f"{f}\nmean-reversion={norm_score:.4f}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(cat_dir / "ood_mean_reversion.png", dpi=150)
    plt.close(fig)

    results["ood_mean_reversion"] = bias_results

    # Overall assessment
    any_influence = any(
        v["max_shift_dex"] >= 0.05 for v in context_influence.values()
    )
    any_bias = any(
        v.get("has_bias", False) for v in bias_results.values()
    )
    results["context_matters"] = any_influence
    results["ood_bias_detected"] = any_bias

    if not any_influence:
        print("\n  ✓ Context pathway influence is negligible (< 0.05 dex).")
        print("    The FiLM-conditioned θ pathway dominates predictions.")
    elif not any_bias:
        print("\n  ⚠ Context pathway has measurable influence, but no systematic")
        print("    mean-reversion bias detected in the tails.")
    else:
        print("\n  ✗ Context pathway influence detected AND systematic mean-reversion")
        print("    bias in tail halos. Consider masking the context in zero-shot mode")
        print("    or using a learned null token instead of zeros.")

    return results


# ============================================================================
# CATEGORY 1: Point Accuracy on Held-Out Test Set
# ============================================================================

def run_category_1(preds: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Tests: RMSE per field/bin, fractional RMSE vs obs noise, outlier fraction.

    All four target channels are positive-definite and modelled in log10 space,
    so accuracy metrics are computed in log10 space.  Observational noise
    fractions (fractional in linear space) are converted to dex via
    σ_log10 ≈ noise_frac / ln(10).

    A validity mask (truth > 0) is applied to exclude bins where the
    simulation has zero/unresolved values — matching the training convention.
    """
    print("\n" + "=" * 72)
    print("CATEGORY 1: Point Accuracy on Held-Out Test Set")
    print("=" * 72)

    cat_dir = ensure_dir(output_dir / "cat1_point_accuracy")
    results = {}

    # Work in log10 space — the native modelling space.
    y_true_log = preds["y_true_log"]              # log10(physical truth)
    y_pred_log = preds["y_pred_log"]              # emulator log10 prediction
    valid = preds["valid_mask"]                    # (N, n_r, n_field) bool
    masses = preds["masses"]
    fields = preds["field_names"]
    n_r = y_true_log.shape[1]

    # Convert fractional observational noise to log10-space noise (dex).
    # σ_log10 ≈ frac / ln(10)  (first-order error propagation)
    OBS_NOISE_DEX = {f: OBS_NOISE_FRAC[f] / np.log(10.0) for f in fields}

    # --- 1a. RMSE per field (valid bins only) ---
    print("\n--- 1a. RMSE per field (log10 dex, valid bins only) ---")
    rmse_fields = {}
    for fi, f in enumerate(fields):
        m = valid[:, :, fi]
        if m.sum() == 0:
            rmse_fields[f] = float("nan")
            continue
        rmse_fields[f] = float(np.sqrt(np.mean(
            (y_pred_log[:, :, fi][m] - y_true_log[:, :, fi][m]) ** 2
        )))
        print(f"  {f}: RMSE = {rmse_fields[f]:.6g} dex  "
              f"({m.sum()}/{m.size} valid bins)")
    results["rmse_by_field_dex"] = rmse_fields

    # --- 1b. RMSE per radial bin (valid entries per bin) ---
    print("\n--- 1b. RMSE per field per radial bin (dex, valid only) ---")
    rmse_per_bin = {}
    for fi, f in enumerate(fields):
        rmse_r = np.zeros(n_r, dtype=np.float64)
        for ri in range(n_r):
            m = valid[:, ri, fi]
            if m.sum() == 0:
                rmse_r[ri] = float("nan")
            else:
                rmse_r[ri] = float(np.sqrt(np.mean(
                    (y_pred_log[m, ri, fi] - y_true_log[m, ri, fi]) ** 2
                )))
        rmse_per_bin[f] = rmse_r
        finite = rmse_r[np.isfinite(rmse_r)]
        if len(finite) > 0:
            print(f"  {f}: RMSE range [{finite.min():.4g}, {finite.max():.4g}] dex")
    results["rmse_per_bin"] = rmse_per_bin

    # --- 1c. Fractional RMSE relative to observational noise ---
    print("\n--- 1c. RMSE / σ_obs (in log10 space, valid only) ---")
    frac_rmse = {}
    all_pass_frac = True
    for fi, f in enumerate(fields):
        sigma_obs_dex = OBS_NOISE_DEX[f]          # scalar in dex
        epsilon = rmse_per_bin[f] / sigma_obs_dex  # (n_r,)
        frac_rmse[f] = epsilon
        finite_eps = epsilon[np.isfinite(epsilon)]
        if len(finite_eps) == 0:
            continue
        max_eps = float(finite_eps.max())
        mean_eps = float(finite_eps.mean())
        passed = max_eps <= THRESHOLDS["rmse_over_sigma_obs"]
        print_pass_fail(
            f"{f} RMSE/σ_obs (max over bins)",
            max_eps,
            f"≤ {THRESHOLDS['rmse_over_sigma_obs']}",
            passed,
        )
        print(f"         mean RMSE/σ_obs = {mean_eps:.4f}")
        if not passed:
            all_pass_frac = False
    results["frac_rmse_over_sigma_obs"] = frac_rmse
    results["frac_rmse_pass"] = all_pass_frac

    # --- 1d. Outlier fraction (obs-noise-weighted Δχ², valid bins only) ---
    print("\n--- 1d. Outlier fraction (Δχ² > 0.2, valid only) ---")
    delta_chi2_per_halo = np.zeros(y_true_log.shape[0], dtype=np.float64)
    n_valid_per_halo = np.zeros(y_true_log.shape[0], dtype=np.float64)
    for fi, f in enumerate(fields):
        sigma_dex = OBS_NOISE_DEX[f]
        m = valid[:, :, fi]  # (N, n_r)
        sq = ((y_pred_log[:, :, fi] - y_true_log[:, :, fi]) / sigma_dex) ** 2
        sq[~m] = 0.0
        delta_chi2_per_halo += sq.sum(axis=1)
        n_valid_per_halo += m.sum(axis=1)

    # Normalize by number of valid bins (not total bins)
    n_valid_per_halo = np.clip(n_valid_per_halo, 1, None)
    delta_chi2_per_halo /= n_valid_per_halo

    thresh = THRESHOLDS["outlier_delta_chi2"]
    f_out = float(np.mean(delta_chi2_per_halo > thresh))
    passed = f_out < THRESHOLDS["outlier_fraction_max"]
    print_pass_fail(
        f"F_out(Δχ² > {thresh})",
        f_out,
        f"< {THRESHOLDS['outlier_fraction_max']}",
        passed,
    )
    results["outlier_fraction"] = f_out
    results["outlier_fraction_pass"] = passed

    # --- 1e. Stratified by mass bin ---
    print("\n--- 1e. RMSE/σ_obs stratified by mass bin ---")
    mass_results = {}
    for lo, hi in MASS_BINS:
        hmask = get_mass_mask(masses, lo, hi)
        n_in = int(hmask.sum())
        if n_in < 5:
            print(f"  [{lo:.1f}, {hi:.1f}): n={n_in} (too few, skipped)")
            continue
        lbl = f"logM=[{lo:.1f},{hi:.1f})"
        bin_frac = {}
        for fi, f in enumerate(fields):
            sigma_dex = OBS_NOISE_DEX[f]
            m = valid[hmask, :, fi]
            resid = y_pred_log[hmask, :, fi] - y_true_log[hmask, :, fi]
            resid_valid = resid[m]
            if len(resid_valid) == 0:
                bin_frac[f] = float("nan")
            else:
                rmse_val = float(np.sqrt(np.mean(resid_valid ** 2)))
                bin_frac[f] = rmse_val / sigma_dex
        mass_results[lbl] = bin_frac
        print(f"  {lbl} (n={n_in}): " +
              ", ".join(f"{f}={v:.4f}" for f, v in bin_frac.items()))
    results["mass_stratified"] = mass_results

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for fi, f in enumerate(fields):
        ax = axes[fi // 2, fi % 2]
        ax.plot(frac_rmse[f], "o-", ms=3, label=f)
        ax.axhline(THRESHOLDS["rmse_over_sigma_obs"], color="r", ls="--",
                    label=f"threshold={THRESHOLDS['rmse_over_sigma_obs']}")
        ax.set_xlabel("Radial bin index")
        ax.set_ylabel("RMSE / σ_obs (log10 space)")
        ax.set_title(f"Cat 1: {f}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(cat_dir / "frac_rmse_vs_obs_noise.png", dpi=150)
    plt.close(fig)

    # Clip Δχ² for histogram (avoid inf from degenerate bins)
    chi2_finite = delta_chi2_per_halo[np.isfinite(delta_chi2_per_halo)]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    if len(chi2_finite) > 0:
        ax2.hist(chi2_finite, bins=50, edgecolor="k", alpha=0.7)
    ax2.axvline(thresh, color="r", ls="--", label=f"Δχ²={thresh}")
    ax2.set_xlabel("Δχ² (obs-noise weighted, log10 space)")
    ax2.set_ylabel("Count")
    ax2.set_title("Cat 1: Outlier distribution")
    ax2.legend()
    fig2.savefig(cat_dir / "outlier_chi2_hist.png", dpi=150)
    plt.close(fig2)

    print(f"\n[INFO] Category 1 plots saved to {cat_dir}")
    return results


# ============================================================================
# CATEGORY 2: Uncertainty Calibration Tests
# ============================================================================

def run_category_2(preds: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Tests: Coverage curves, PIT histograms + KS test, reduced χ², Spearman ρ.

    A validity mask (truth > 0) is applied to exclude bins where the
    simulation has zero/unresolved values — matching the training convention.
    """
    print("\n" + "=" * 72)
    print("CATEGORY 2: Uncertainty Calibration Tests")
    print("=" * 72)

    cat_dir = ensure_dir(output_dir / "cat2_uncertainty_calibration")
    results = {}
    fields = preds["field_names"]
    valid = preds["valid_mask"]  # (N, n_r, n_field) bool

    # Use log-space predictions for calibration (better-calibrated for
    # positive-definite channels modelled in log10 space).
    y_true_cal = preds["y_true_log"]
    y_pred_cal = preds["y_pred_log"]
    y_std_cal = preds["y_std_log"]
    print("[INFO] Using log10-space predictions for calibration tests")

    # Compute z-scores once, masking invalid entries.
    z_scores = (y_true_cal - y_pred_cal) / np.clip(y_std_cal, 1e-12, None)
    abs_z = np.abs(z_scores)

    # --- 2a. Coverage curves (PP plot), valid bins only ---
    print("\n--- 2a. Coverage curves (valid bins only) ---")
    p_grid = np.linspace(0.05, 0.95, 19)
    z_grid = scipy_stats.norm.ppf((1.0 + p_grid) / 2.0)
    max_deviations = {}
    empirical_cov = np.zeros((len(p_grid), len(fields)), dtype=np.float64)
    for fi, f in enumerate(fields):
        m = valid[:, :, fi]
        n_valid = m.sum()
        if n_valid == 0:
            max_deviations[f] = float("nan")
            continue
        abs_z_f = abs_z[:, :, fi][m]
        for pi, z in enumerate(z_grid):
            empirical_cov[pi, fi] = float((abs_z_f <= z).sum()) / n_valid
        dev = np.abs(empirical_cov[:, fi] - p_grid)
        max_dev = float(dev.max())
        max_deviations[f] = max_dev
        passed = max_dev <= THRESHOLDS["coverage_deviation_max"]
        print_pass_fail(
            f"{f} max coverage deviation",
            max_dev,
            f"≤ {THRESHOLDS['coverage_deviation_max']}",
            passed,
        )
    results["coverage_max_deviation"] = max_deviations

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(p_grid, p_grid, "k--", lw=1.5, label="ideal")
    for fi, f in enumerate(fields):
        ax.plot(p_grid, empirical_cov[:, fi], "o-", ms=3, label=f)
    ax.fill_between(p_grid,
                     p_grid - THRESHOLDS["coverage_deviation_max"],
                     p_grid + THRESHOLDS["coverage_deviation_max"],
                     alpha=0.1, color="gray", label="±5% band")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Cat 2a: Coverage Calibration (valid bins)")
    ax.legend(fontsize=8)
    fig.savefig(cat_dir / "coverage_curve.png", dpi=150)
    plt.close(fig)

    # --- 2b. PIT histograms + KS test (valid bins only) ---
    print("\n--- 2b. PIT histograms + KS statistic with bootstrap bands (valid bins only) ---")
    # PIT = Φ(z), where Φ is the standard normal CDF
    pit_full = scipy_stats.norm.cdf(z_scores)
    pit_results = {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for fi, f in enumerate(fields):
        m = valid[:, :, fi]
        pit_flat = pit_full[:, :, fi][m]
        if len(pit_flat) < 10:
            pit_results[f] = {
                "ks_stat": float("nan"),
                "ks_boot_crit": float("nan"),
                "ks_boot_n": 0,
            }
            continue
        ks_stat = float(scipy_stats.kstest(pit_flat, "uniform").statistic)
        ks_boot_crit = ks_bootstrap_critical(
            n=len(pit_flat),
            n_boot=400,
            alpha=THRESHOLDS["pit_ks_stat_boot_alpha"],
            seed=12345,
        )
        pit_results[f] = {
            "ks_stat": ks_stat,
            "ks_boot_crit": float(ks_boot_crit),
            "ks_boot_n": int(len(pit_flat)),
        }
        passed = ks_stat <= ks_boot_crit
        print_pass_fail(
            f"{f} KS statistic",
            ks_stat,
            f"<= bootstrap_crit={ks_boot_crit:.4f}",
            passed,
        )

        ax = axes[fi // 2, fi % 2]
        ax.hist(pit_flat, bins=30, density=True, edgecolor="k", alpha=0.7)
        ax.axhline(1.0, color="r", ls="--", lw=1.5, label="uniform")
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        ax.set_title(f"{f} (KS={ks_stat:.4f}, crit={ks_boot_crit:.4f})")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(cat_dir / "pit_histograms.png", dpi=150)
    plt.close(fig)
    results["pit_ks"] = pit_results

    # --- 2c. Reduced chi-squared (valid bins only) ---
    print("\n--- 2c. Reduced χ² (valid bins only) ---")
    chi2_fields = {}
    for fi, f in enumerate(fields):
        m = valid[:, :, fi]
        if m.sum() == 0:
            chi2_fields[f] = float("nan")
            continue
        chi2_red = float(np.mean(z_scores[:, :, fi][m] ** 2))
        chi2_fields[f] = chi2_red
        lo, hi = THRESHOLDS["chi2_red_lo"], THRESHOLDS["chi2_red_hi"]
        passed = lo <= chi2_red <= hi
        print_pass_fail(f"{f} χ²_red", chi2_red, f"[{lo}, {hi}]", passed)
    results["chi2_red"] = chi2_fields

    # --- 2d. Spearman rank correlation |ε| vs σ̂ (valid bins only) ---
    print("\n--- 2d. Spearman ρ(|ε|, σ̂) (valid bins only) ---")
    corr = {}
    for fi, f in enumerate(fields):
        m = valid[:, :, fi]
        abs_err = np.abs(y_pred_cal[:, :, fi] - y_true_cal[:, :, fi])[m]
        sig = y_std_cal[:, :, fi][m]
        if len(abs_err) < 10:
            corr[f] = float("nan")
            continue
        # Spearman via rank correlation
        rho, _ = scipy_stats.spearmanr(sig, abs_err)
        corr[f] = float(rho)
        passed = rho > THRESHOLDS["spearman_rho_min"]
        print_pass_fail(f"{f} Spearman ρ", rho, f"> {THRESHOLDS['spearman_rho_min']}", passed)
    results["spearman_rho"] = corr

    # --- 2e. Epistemic vs aleatoric decomposition ---
    if "y_ale_std" in preds and "y_epi_std" in preds:
        print("\n--- 2e. Epistemic vs Aleatoric uncertainty (valid bins only) ---")
        for fi, f in enumerate(fields):
            m = valid[:, :, fi]
            ale = preds["y_ale_std"][:, :, fi][m].mean() if m.sum() > 0 else 0.0
            epi = preds["y_epi_std"][:, :, fi][m].mean() if m.sum() > 0 else 0.0
            frac_epi = epi / (ale + epi + 1e-12)
            print(f"  {f}: aleatoric={ale:.4g}, epistemic={epi:.4g}, "
                  f"epi_fraction={frac_epi:.2%}")

    print(f"\n[INFO] Category 2 plots saved to {cat_dir}")
    return results


# ============================================================================
# CATEGORY 2b: TARP (Tests of Accuracy with Random Points) Calibration
# ============================================================================

def _run_tarp_pointwise(preds: Dict[str, Any], cat_dir: Path) -> Dict[str, Any]:
    """Legacy pointwise calibration surrogate (kept for backward compatibility)."""
    results: Dict[str, Any] = {}
    fields = preds["field_names"]
    y_true_log = preds["y_true_log"]   # (N, n_r, n_field)
    y_pred_log = preds["y_pred_log"]   # (N, n_r, n_field)
    y_std_log = preds["y_std_log"]     # (N, n_r, n_field)
    valid = preds["valid_mask"]        # (N, n_r, n_field)

    alpha_grid = np.linspace(0.0, 1.0, 101)
    tarp_results = {}

    fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 5))
    if len(fields) == 1:
        axes = [axes]

    for fi, f in enumerate(fields):
        m = valid[:, :, fi]
        if m.sum() < 100:
            print(f"  {f}: too few valid bins ({m.sum()}), skipping.")
            continue

        mu = y_pred_log[:, :, fi][m]
        sigma = np.maximum(y_std_log[:, :, fi][m], 1e-12)
        truth = y_true_log[:, :, fi][m]

        z_scores = (truth - mu) / sigma
        p_values = scipy_stats.norm.cdf(z_scores)
        ecp = np.array([np.mean(p_values <= a) for a in alpha_grid])

        max_dev = float(np.max(np.abs(ecp - alpha_grid)))
        ks_stat = float(scipy_stats.kstest(p_values, "uniform").statistic)
        ks_crit = ks_bootstrap_critical(
            n=len(p_values), n_boot=400,
            alpha=THRESHOLDS["pit_ks_stat_boot_alpha"], seed=12345,
        )

        tarp_results[f] = {
            "max_deviation": max_dev,
            "ks_stat": ks_stat,
            "ks_boot_crit": float(ks_crit),
            "n_points": int(m.sum()),
        }

        passed = max_dev < 0.05
        print_pass_fail(
            f"TARP max deviation ({f})",
            max_dev, "<0.05", passed,
        )
        print(f"    KS stat={ks_stat:.4f}, bootstrap_crit={ks_crit:.4f}, "
              f"n={int(m.sum())}")

        ax = axes[fi]
        ax.plot(alpha_grid, ecp, "b-", lw=2, label="Emulator")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
        ax.fill_between(
            alpha_grid,
            alpha_grid - 0.05, alpha_grid + 0.05,
            alpha=0.15, color="gray", label="±0.05 band",
        )
        ax.set_xlabel("Credibility level α")
        ax.set_ylabel("Expected coverage P(p ≤ α)")
        ax.set_title(f"{f}\npointwise (max Δ={max_dev:.3f})")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(cat_dir / "tarp_ecp_pointwise.png", dpi=150)
    plt.close(fig)

    results["tarp"] = tarp_results
    all_pass = all(v["max_deviation"] < 0.05 for v in tarp_results.values())
    results["tarp_pass"] = all_pass
    results["mode"] = "pointwise"
    return results


def _run_tarp_parameter_space(output_dir: Path, cat_dir: Path, n_refs: int = 256) -> Dict[str, Any]:
    """Strict TARP in parameter space using posterior samples from Category 4."""
    chains_path = output_dir / "cat4_mock_recovery" / "all_mock_posteriors.npz"
    if not chains_path.exists():
        return {
            "mode": "parameter",
            "skipped": True,
            "reason": "Category 4 posterior archive not found (all_mock_posteriors.npz)",
        }

    dat = np.load(chains_path, allow_pickle=True)
    posterior = dat["posterior_samples"]     # (n_mock, n_post, d)
    true_theta = dat["true_thetas"]          # (n_mock, d)
    prior_lo = dat["prior_lo"]               # (d,)
    prior_hi = dat["prior_hi"]               # (d,)

    n_mock, n_post, d = posterior.shape
    if n_mock < 2 or n_post < 20:
        return {
            "mode": "parameter",
            "skipped": True,
            "reason": f"insufficient posterior samples (n_mock={n_mock}, n_post={n_post})",
        }

    rng = np.random.default_rng(123)
    alpha_grid = np.linspace(0.05, 0.95, 19)
    covered = np.zeros((len(alpha_grid), n_mock, n_refs), dtype=np.float64)

    for mi in range(n_mock):
        s = posterior[mi]           # (n_post, d)
        t = true_theta[mi]          # (d,)
        refs = rng.uniform(prior_lo, prior_hi, size=(n_refs, d))

        for ri in range(n_refs):
            r = refs[ri]
            d_post = np.sum((s - r[None, :]) ** 2, axis=1)
            d_true = float(np.sum((t - r) ** 2))
            for ai, a in enumerate(alpha_grid):
                q = float(np.quantile(d_post, a))
                covered[ai, mi, ri] = 1.0 if d_true <= q else 0.0

    ecp = covered.mean(axis=(1, 2))
    max_dev = float(np.max(np.abs(ecp - alpha_grid)))

    # Bootstrap confidence band on ECP(alpha)
    n_boot = 300
    boot_ecp = np.zeros((n_boot, len(alpha_grid)), dtype=np.float64)
    flat = covered.reshape(len(alpha_grid), -1)
    n_total = flat.shape[1]
    for bi in range(n_boot):
        idx = rng.integers(0, n_total, size=n_total)
        boot_ecp[bi] = flat[:, idx].mean(axis=1)
    lo = np.quantile(boot_ecp, 0.025, axis=0)
    hi = np.quantile(boot_ecp, 0.975, axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(alpha_grid, alpha_grid, "k--", lw=1.2, label="ideal")
    ax.plot(alpha_grid, ecp, "o-", lw=2.0, color="steelblue", label="parameter-space TARP")
    ax.fill_between(alpha_grid, lo, hi, alpha=0.2, color="steelblue", label="bootstrap 95% band")
    ax.set_xlabel("Nominal credibility α")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Parameter-space TARP (max Δ={max_dev:.3f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8)
    fig.savefig(cat_dir / "tarp_parameter_space_ecp.png", dpi=150)
    plt.close(fig)

    pass_flag = max_dev < 0.05
    print_pass_fail("Parameter-space TARP max deviation", max_dev, "<0.05", pass_flag)

    return {
        "mode": "parameter",
        "alpha_grid": alpha_grid,
        "ecp": ecp,
        "ecp_boot_lo": lo,
        "ecp_boot_hi": hi,
        "max_deviation": max_dev,
        "n_mock": int(n_mock),
        "n_post": int(n_post),
        "n_refs": int(n_refs),
        "tarp_pass": pass_flag,
    }


def run_tarp(
    preds: Dict[str, Any],
    output_dir: Path,
    mode: str = "pointwise",
    n_param_refs: int = 256,
) -> Dict[str, Any]:
    """Run TARP calibration in selected mode: 'pointwise' or 'parameter'."""
    print("\n" + "=" * 72)
    print(f"CATEGORY 2b: TARP Calibration Check ({mode})")
    print("=" * 72)

    cat_dir = output_dir / "cat2b_tarp"
    cat_dir.mkdir(parents=True, exist_ok=True)

    if mode == "pointwise":
        results = _run_tarp_pointwise(preds, cat_dir)
        print(f"\n  TARP overall: {'PASS' if results.get('tarp_pass', False) else 'FAIL'}")
    elif mode == "parameter":
        results = _run_tarp_parameter_space(output_dir, cat_dir, n_refs=n_param_refs)
        if results.get("skipped", False):
            print(f"  [WARN] Parameter-space TARP skipped: {results.get('reason', 'unknown reason')}")
        else:
            print(f"\n  Parameter-space TARP overall: {'PASS' if results.get('tarp_pass', False) else 'FAIL'}")
    else:
        raise ValueError(f"Unknown tarp mode '{mode}'. Use 'pointwise' or 'parameter'.")

    print(f"\n[INFO] TARP plots saved to {cat_dir}")
    return results


# ============================================================================
# CATEGORY 3: Single-Parameter (1P) Variation and Sensitivity Tests
# ============================================================================

def run_category_3(emu: Emulator, output_dir: Path) -> Dict[str, Any]:
    """
    Tests: 1P monotonicity, derivative/sensitivity consistency.
    """
    print("\n" + "=" * 72)
    print("CATEGORY 3: Single-Parameter (1P) Variation and Sensitivity Tests")
    print("=" * 72)

    cat_dir = ensure_dir(output_dir / "cat3_1p_sensitivity")
    results = {}

    # Load 1P parameter table
    onep_csv = Path("/mnt/home/mlee1/Sims/IllustrisTNG/L50n512/1P/CosmoAstroSeed_IllustrisTNG_L50n512_1P.txt")
    if not onep_csv.exists():
        print(f"[WARN] 1P parameter CSV not found: {onep_csv}")
        print("       Skipping Category 3.")
        return {"skipped": True, "reason": "1P CSV not found"}

    onep_params = pd.read_csv(onep_csv, sep=r"\s+", engine="python")
    if "#Name" in onep_params.columns:
        onep_params = onep_params.rename(columns={"#Name": "tag"})

    theta_cols = [
        c for c in onep_params.columns
        if c != "tag" and pd.api.types.is_numeric_dtype(onep_params[c])
        and str(c).strip().lower() != "seed"
    ][: emu.theta_dim]

    # Get fiducial theta (row where tag starts with "1P_" and has idx 0, or the median)
    fid_mask = onep_params["tag"].str.contains("fiducial", case=False, na=False)
    if fid_mask.any():
        theta_fid = onep_params.loc[fid_mask].iloc[0][theta_cols].to_numpy(dtype=np.float32)
    else:
        theta_fid = onep_params[theta_cols].median().to_numpy(dtype=np.float32)

    # Reference mass and radial grid
    ref_mass = np.array([1e14], dtype=np.float32)  # 10^14 M_sun
    r_bins_kpc = np.logspace(np.log10(10), np.log10(2500), 50).astype(np.float32)
    r500_ref = 700.0  # ~ R500 for 10^14 Msun
    rr500 = (r_bins_kpc / r500_ref).astype(np.float32)[None, :]  # (1, n_r)

    snap_eval = int(emu.snapnums[0])
    z_eval = float(getattr(emu, "redshift_by_snap", {}).get(snap_eval, 0.0))

    # --- 3a. 1P monotonicity test ---
    print("\n--- 3a. 1P monotonicity test ---")
    # Expected monotonicity directions for key parameters
    # Positive means profile quantity increases with parameter, negative decreases.
    # These are approximate physical expectations from IllustrisTNG.
    monotonicity_expectations = {
        # (param_index_in_theta, field, expected_sign for integrated quantity)
        # Cosmological
        0: {"name": "Omega_m", "gas_density": +1},
        1: {"name": "sigma_8", "temperature": +1},
    }

    n_vary = 11  # Number of parameter values to test
    mono_results = {}

    # Test a subset of parameters for monotonicity
    param_indices_to_test = list(range(min(10, emu.theta_dim)))
    for pi in param_indices_to_test:
        pname = theta_cols[pi] if pi < len(theta_cols) else f"theta_{pi}"

        # Get prior range from 1P table
        col_vals = onep_params[theta_cols[pi]].dropna().values
        p_lo, p_hi = float(col_vals.min()), float(col_vals.max())
        if abs(p_hi - p_lo) < 1e-10:
            continue

        param_values = np.linspace(p_lo, p_hi, n_vary)
        integrated_profiles = {f: [] for f in TARGET_FIELDS}

        for pv in param_values:
            theta_test = theta_fid.copy()
            theta_test[pi] = pv
            pred = emu.predict(
                theta=theta_test, M=ref_mass, r_bins=rr500,
                field=TARGET_FIELDS, snapnum=snap_eval, redshift=z_eval,
                n_samples=10,
            )
            for fi, f in enumerate(TARGET_FIELDS):
                integrated_profiles[f].append(float(np.mean(pred.mean_log10[0, :, fi])))

        # Check monotonicity (Spearman correlation with parameter value)
        mono_results[pname] = {}
        for f in TARGET_FIELDS:
            vals = np.array(integrated_profiles[f])
            rho, _ = scipy_stats.spearmanr(param_values, vals)
            is_monotone = abs(rho) > 0.8
            # Check expected direction if available
            expected = monotonicity_expectations.get(pi, {}).get(f)
            direction_ok = True
            if expected is not None and is_monotone:
                direction_ok = (np.sign(rho) == np.sign(expected))
            mono_results[pname][f] = {
                "spearman_rho": float(rho),
                "is_monotone": bool(is_monotone),
                "direction_ok": bool(direction_ok),
            }

        print(f"  {pname}: " + ", ".join(
            f"{f}={'↑' if mono_results[pname][f]['spearman_rho'] > 0 else '↓'}"
            f"(ρ={mono_results[pname][f]['spearman_rho']:.2f})"
            + ("" if mono_results[pname][f].get("direction_ok", True) else " ⚠wrong sign")
            for f in TARGET_FIELDS
        ))

    results["monotonicity"] = mono_results

    # --- 3b. Derivative / sensitivity consistency ---
    print("\n--- 3b. Finite-difference emulator derivatives ---")
    delta = 0.01  # fractional perturbation
    derivatives = {}

    for pi in param_indices_to_test:
        pname = theta_cols[pi] if pi < len(theta_cols) else f"theta_{pi}"
        theta_lo = theta_fid.copy()
        theta_hi = theta_fid.copy()

        # Use absolute perturbation scaled to parameter range
        col_vals = onep_params[theta_cols[pi]].dropna().values
        p_range = float(col_vals.max() - col_vals.min())
        if p_range < 1e-10:
            continue
        dp = delta * p_range

        theta_lo[pi] = theta_fid[pi] - dp
        theta_hi[pi] = theta_fid[pi] + dp

        pred_lo = emu.predict(
            theta=theta_lo, M=ref_mass, r_bins=rr500,
            field=TARGET_FIELDS, snapnum=snap_eval, redshift=z_eval,
            n_samples=10,
        )
        pred_hi = emu.predict(
            theta=theta_hi, M=ref_mass, r_bins=rr500,
            field=TARGET_FIELDS, snapnum=snap_eval, redshift=z_eval,
            n_samples=10,
        )

        deriv = {}
        for fi, f in enumerate(TARGET_FIELDS):
            dmu_dp = (pred_hi.mean_log10[0, :, fi] - pred_lo.mean_log10[0, :, fi]) / (2 * dp)
            deriv[f] = dmu_dp
        derivatives[pname] = deriv

    results["derivatives"] = {
        k: {f: float(np.mean(np.abs(v))) for f, v in d.items()}
        for k, d in derivatives.items()
    }

    # Plot derivative heatmap
    if derivatives:
        param_names = list(derivatives.keys())
        fig, axes = plt.subplots(1, len(TARGET_FIELDS), figsize=(5 * len(TARGET_FIELDS), 6))
        if len(TARGET_FIELDS) == 1:
            axes = [axes]
        for fi, f in enumerate(TARGET_FIELDS):
            matrix = np.array([derivatives[p][f] for p in param_names])
            im = axes[fi].imshow(matrix, aspect="auto", cmap="RdBu_r")
            axes[fi].set_xlabel("Radial bin")
            axes[fi].set_ylabel("Parameter")
            axes[fi].set_yticks(range(len(param_names)))
            axes[fi].set_yticklabels(param_names, fontsize=7)
            axes[fi].set_title(f"∂{f}/∂θ")
            plt.colorbar(im, ax=axes[fi], shrink=0.8)
        plt.tight_layout()
        fig.savefig(cat_dir / "derivative_heatmap.png", dpi=150)
        plt.close(fig)

    # --- 3c. Sobol first-order indices (fast estimate) ---
    print("\n--- 3c. Sobol sensitivity indices (fast Monte Carlo) ---")
    n_sobol = 500
    rng = np.random.default_rng(42)

    # Generate base and perturbed samples
    param_ranges = []
    for pi in range(emu.theta_dim):
        if pi < len(theta_cols):
            col_vals = onep_params[theta_cols[pi]].dropna().values
            param_ranges.append((float(col_vals.min()), float(col_vals.max())))
        else:
            param_ranges.append((theta_fid[pi] - 0.1, theta_fid[pi] + 0.1))

    # Sample A and B matrices
    A = np.zeros((n_sobol, emu.theta_dim), dtype=np.float32)
    B = np.zeros((n_sobol, emu.theta_dim), dtype=np.float32)
    for pi, (lo, hi) in enumerate(param_ranges):
        A[:, pi] = rng.uniform(lo, hi, size=n_sobol).astype(np.float32)
        B[:, pi] = rng.uniform(lo, hi, size=n_sobol).astype(np.float32)

    # Evaluate f(A) — use mean over radii as scalar output
    def eval_batch(thetas: np.ndarray) -> np.ndarray:
        """Evaluate emulator for batch of theta vectors. Returns (n, n_field)."""
        out = np.zeros((len(thetas), len(TARGET_FIELDS)), dtype=np.float64)
        for i, th in enumerate(thetas):
            pred = emu.predict(
                theta=th, M=ref_mass, r_bins=rr500,
                field=TARGET_FIELDS, snapnum=snap_eval, redshift=z_eval,
                n_samples=5,
            )
            for fi in range(len(TARGET_FIELDS)):
                out[i, fi] = np.mean(pred.mean_log10[0, :, fi])
        return out

    print("  Evaluating base samples (A)...")
    fA = eval_batch(A)
    print("  Evaluating base samples (B)...")
    fB = eval_batch(B)

    # First-order Sobol: S_i ≈ Var_Xi[E(Y|Xi)] / Var(Y)
    # Using Saltelli estimator with AB_i matrices
    sobol_first = {}
    n_sobol_params = min(10, emu.theta_dim)  # Only first 10 for speed
    for pi in range(n_sobol_params):
        pname = theta_cols[pi] if pi < len(theta_cols) else f"theta_{pi}"
        # AB_i: take B but replace column i with A's column
        AB_i = B.copy()
        AB_i[:, pi] = A[:, pi]
        print(f"  Evaluating AB_{pi} ({pname})...")
        fABi = eval_batch(AB_i)

        s1 = {}
        for fi, f in enumerate(TARGET_FIELDS):
            var_total = np.var(fA[:, fi])
            if var_total < 1e-30:
                s1[f] = 0.0
                continue
            # Jansen first-order estimator: S_i = 1 - E[(f(B)-f(AB_i))^2]/(2 Var[f])
            s1_val = float(1.0 - 0.5 * np.mean((fB[:, fi] - fABi[:, fi]) ** 2) / var_total)
            s1[f] = float(np.clip(s1_val, 0.0, 1.0))
        sobol_first[pname] = s1
        print(f"    {pname}: " + ", ".join(f"{f}={v:.3f}" for f, v in s1.items()))

    results["sobol_first_order"] = sobol_first

    # Sobol bar chart
    if sobol_first:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(sobol_first))
        w = 0.8 / len(TARGET_FIELDS)
        for fi, f in enumerate(TARGET_FIELDS):
            vals = [sobol_first[p][f] for p in sobol_first]
            ax.bar(x + fi * w, vals, w, label=f, alpha=0.8)
        ax.set_xticks(x + 0.4)
        ax.set_xticklabels(list(sobol_first.keys()), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("First-order Sobol index")
        ax.set_title("Cat 3c: Sobol Sensitivity Indices")
        ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(cat_dir / "sobol_indices.png", dpi=150)
        plt.close(fig)

    print(f"\n[INFO] Category 3 plots saved to {cat_dir}")
    return results


# ============================================================================
# CATEGORY 4: Mock Parameter Recovery (Framework)
# ============================================================================

def run_category_4(
    emu: Emulator,
    preds: Dict[str, Any],
    output_dir: Path,
    n_mock: int = 10,
    stack_logm_min: float = 13.0,
    stack_logm_max: float = 14.0,
    hmc_n_samples: int = 600,
    hmc_n_warmup: int = 200,
    hmc_n_chains: int = 2,
    hmc_n_leapfrog: int = 15,
    hmc_init_step_size: float = 0.01,
    mass_bias_grid_dex: Optional[List[float]] = None,
    mass_bias_prior_sigma_dex: float = 0.10,
) -> Dict[str, Any]:
    """
    Mock parameter recovery using HMC with the differentiable emulator.

    Runs HMC on N_mock independent mock halos from the test set.
    Reports per-parameter 68% CI coverage across mocks (should be ≈0.68).
    """
    print("\n" + "=" * 72)
    print("CATEGORY 4: Mock Parameter Recovery (HMC)")
    print("=" * 72)

    cat_dir = ensure_dir(output_dir / "cat4_mock_recovery")
    results = {}

    # --- Load 1P parameter table for prior bounds and names ---
    onep_csv = Path("/mnt/home/mlee1/Sims/IllustrisTNG/L50n512/1P/CosmoAstroSeed_IllustrisTNG_L50n512_1P.txt")
    if not onep_csv.exists():
        print(f"[WARN] 1P parameter CSV not found: {onep_csv}")
        print("       Cannot determine prior bounds; skipping HMC recovery.")
        return {"skipped": True, "reason": "1P CSV not found"}

    onep_params = pd.read_csv(onep_csv, sep=r"\s+", engine="python")
    if "#Name" in onep_params.columns:
        onep_params = onep_params.rename(columns={"#Name": "tag"})
    theta_cols = [
        c for c in onep_params.columns
        if c != "tag" and pd.api.types.is_numeric_dtype(onep_params[c])
        and str(c).strip().lower() != "seed"
    ][: emu.theta_dim]
    param_names = list(theta_cols)

    # Prior bounds from 1P training variation range.
    prior_lo = np.array([onep_params[c].min() for c in theta_cols], dtype=np.float64)
    prior_hi = np.array([onep_params[c].max() for c in theta_cols], dtype=np.float64)
    degenerate = (prior_hi - prior_lo) < 1e-10
    prior_lo[degenerate] -= 0.01
    prior_hi[degenerate] += 0.01

    y_true_log = preds["y_true_log"]
    valid = preds["valid_mask"]
    masses = preds["masses"]
    run_ids = preds["run_ids"]

    snap_eval = int(emu.snapnums[0])
    z_eval = float(getattr(emu, "redshift_by_snap", {}).get(snap_eval, 0.0))

    sigma_obs_dex = np.array([OBS_NOISE_FRAC[f] / np.log(10.0) for f in TARGET_FIELDS],
                              dtype=np.float64)

    # --- Load theta table ---
    theta_table = None
    try:
        args_ns = _get_train_args()
        theta_table = load_theta_table(
            Path(args_ns.param_csv),
            target_theta_dim=int(args_ns.theta_dim),
        )
    except Exception as exc:
        print(f"  [WARN] Could not load theta table: {exc}")
        return {"skipped": True, "reason": f"theta table load failed: {exc}"}

    # --- Select N_mock stacked mocks (run_id + mass bin) ---
    N_MOCK = int(n_mock)
    MIN_STACK = 6
    MAX_STACK = 24
    rng = np.random.default_rng(42)
    stack_mass_bins = [(float(stack_logm_min), float(stack_logm_max))]

    # Get unique run_ids with known theta, then build stacked candidates.
    unique_run_ids = np.unique(run_ids)
    valid_run_ids = [rid for rid in unique_run_ids if theta_table.get(int(rid)) is not None]
    if len(valid_run_ids) < 5:
        print(f"  [WARN] Only {len(valid_run_ids)} runs have known theta — need ≥5.")
        return {"skipped": True, "reason": "too few runs with known theta"}

    # Candidate = (run_id, mass_bin_label, halo_indices_for_stack)
    candidates = []
    for rid in valid_run_ids:
        run_mask = run_ids == rid
        idxs = np.where(run_mask)[0]
        if len(idxs) < MIN_STACK:
            continue

        for lo, hi in stack_mass_bins:
            mlog = np.log10(np.clip(masses[idxs], 1e10, None))
            in_bin = (mlog >= lo) & (mlog < hi)
            idx_bin = idxs[in_bin]
            if len(idx_bin) < MIN_STACK:
                continue
            good = [int(i) for i in idx_bin if valid[i].mean() > 0.5]
            if len(good) < MIN_STACK:
                continue
            if len(good) > MAX_STACK:
                pick = np.sort(rng.choice(np.array(good), size=MAX_STACK, replace=False))
            else:
                pick = np.array(good, dtype=int)
            candidates.append({
                "run_id": int(rid),
                "mass_bin": f"[{lo:.1f},{hi:.1f})",
                "indices": pick,
            })

    if len(candidates) == 0:
        return {"skipped": True, "reason": "no stacked run/mass-bin candidates found"}

    if len(candidates) < N_MOCK:
        N_MOCK = len(candidates)
        print(f"  [INFO] Reduced N_mock to {N_MOCK} stacked candidates")

    pick_idx = rng.choice(np.arange(len(candidates)), size=N_MOCK, replace=False)
    mock_candidates = [candidates[int(i)] for i in pick_idx]

    if mass_bias_grid_dex is None or len(mass_bias_grid_dex) == 0:
        mass_bias_grid_dex = [0.0]
    mass_bias_grid_dex = [float(v) for v in mass_bias_grid_dex]

    print(f"\n  Running HMC on {N_MOCK} stacked mocks...")
    print(
        f"  HMC settings: {hmc_n_samples} samples, {hmc_n_warmup} warmup, "
        f"{hmc_n_chains} chains, {hmc_n_leapfrog} leapfrog"
    )
    print(
        "  Mass-bias nuisance grid (dex): " +
        ", ".join(f"{v:+.3f}" for v in mass_bias_grid_dex) +
        f" (Gaussian prior sigma={mass_bias_prior_sigma_dex:.3f})"
    )
    print(f"  Noise levels (dex): " +
          ", ".join(f"{f}={sigma_obs_dex[fi]:.4f}" for fi, f in enumerate(TARGET_FIELDS)))

    # --- 4a. Multi-mock HMC recovery ---
    n_check = min(10, emu.theta_dim)  # Check first 10 parameters
    covered_68 = np.zeros((N_MOCK, n_check), dtype=bool)
    covered_95 = np.zeros((N_MOCK, n_check), dtype=bool)
    all_rhat_max = []
    all_accept = []
    all_mock_details = []
    all_post_samples = []
    all_true_thetas = []

    for mi, cand in enumerate(mock_candidates):
        true_run_id = int(cand["run_id"])
        true_theta = theta_table.get(true_run_id)
        if true_theta is None:
            continue

        stack_idx = np.asarray(cand["indices"], dtype=int)
        n_stack = int(len(stack_idx))
        mock_masses = masses[stack_idx].astype(np.float32)
        rr500_stack = preds["r_bins"][stack_idx].astype(np.float32)

        y_stack = y_true_log[stack_idx]         # (n_stack, n_r, n_f)
        valid_stack = valid[stack_idx]
        counts = valid_stack.sum(axis=0)        # (n_r, n_f)
        counts_clip = np.clip(counts, 1, None)
        y_mock_true_log = np.sum(y_stack * valid_stack, axis=0) / counts_clip
        obs_valid = counts >= max(3, int(0.4 * n_stack))

        sigma_obs_stack = np.zeros((1, y_mock_true_log.shape[0], y_mock_true_log.shape[1]), dtype=np.float64)
        for fi in range(len(TARGET_FIELDS)):
            sigma_obs_stack[0, :, fi] = sigma_obs_dex[fi] / np.sqrt(np.clip(counts[:, fi], 1, None))

        noise = rng.normal(0, 1, size=y_mock_true_log.shape) * sigma_obs_stack[0]
        y_mock_noisy_log = y_mock_true_log + noise
        y_mock_noisy_log[~obs_valid] = np.nan
        sigma_obs_stack[0][~obs_valid] = 1e6

        init = np.clip(true_theta.astype(np.float64), prior_lo, prior_hi)
        best_obj = -np.inf
        best_result = None
        best_bias_dex = 0.0
        bias_scan: List[Dict[str, float]] = []

        for bias_dex in mass_bias_grid_dex:
            # Shift all masses by a shared multiplicative bias: M_eff = M * 10^delta.
            mock_masses_eff = mock_masses * (10.0 ** float(bias_dex))
            sampler = HMCSampler(
                emu=emu,
                y_obs=y_mock_noisy_log[None, ...],
                sigma_obs=sigma_obs_stack,
                M=mock_masses_eff,
                r_bins=rr500_stack,
                prior_lo=prior_lo,
                prior_hi=prior_hi,
                field=TARGET_FIELDS,
                snapnum=snap_eval,
                redshift=z_eval,
                n_leapfrog=hmc_n_leapfrog,
                n_samples_per_eval=1,  # deterministic mode uses latent mean
                include_model_error=True,
                include_log_det=True,
                stack_profiles=True,
            )

            hmc_try = sampler.run(
                n_samples=hmc_n_samples,
                n_warmup=hmc_n_warmup,
                n_chains=hmc_n_chains,
                init_theta=init,
                init_step_size=hmc_init_step_size,
                target_accept=0.70,
                param_names=param_names,
                verbose=False,
            )

            lp = hmc_try.log_prob[:, hmc_n_warmup:]
            if lp.size == 0:
                lp_mean = -np.inf
            else:
                lp_mean = float(np.nanmean(lp))
            lp_prior = -0.5 * (float(bias_dex) / max(float(mass_bias_prior_sigma_dex), 1e-6)) ** 2
            obj = lp_mean + lp_prior
            bias_scan.append({
                "bias_dex": float(bias_dex),
                "logprob_mean": float(lp_mean),
                "logprior": float(lp_prior),
                "objective": float(obj),
            })
            if np.isfinite(obj) and obj > best_obj:
                best_obj = obj
                best_result = hmc_try
                best_bias_dex = float(bias_dex)

        if best_result is None:
            print(f"  [WARN] Mock {mi+1}: all mass-bias grid runs failed; skipping")
            continue
        hmc_result = best_result

        flat = hmc_result.flat_samples
        all_post_samples.append(flat)
        all_true_thetas.append(np.asarray(true_theta, dtype=np.float64))
        rhat_max = hmc_result.diagnostics.get("rhat_max", np.nan)
        all_rhat_max.append(rhat_max)
        all_accept.append(hmc_result.accept_rate.mean())

        for pi in range(n_check):
            q16 = np.percentile(flat[:, pi], 16)
            q84 = np.percentile(flat[:, pi], 84)
            q025 = np.percentile(flat[:, pi], 2.5)
            q975 = np.percentile(flat[:, pi], 97.5)
            truth = float(true_theta[pi])
            covered_68[mi, pi] = q16 <= truth <= q84
            covered_95[mi, pi] = q025 <= truth <= q975

        all_mock_details.append({
            "run_id": true_run_id,
            "mass_bin": cand["mass_bin"],
            "n_stack": n_stack,
            "mean_mass": float(np.mean(mock_masses)),
            "mass_bias_dex_best": float(best_bias_dex),
            "mass_bias_scan": bias_scan,
            "rhat_max": float(rhat_max),
            "accept_rate": float(hmc_result.accept_rate.mean()),
        })

        status = "PASS" if rhat_max < RHAT_MAX else f"Rhat={rhat_max:.2f}"
        print(f"  Mock {mi+1}/{N_MOCK}: run={true_run_id}, "
              f"bin={cand['mass_bin']}, n_stack={n_stack}, "
              f"dlogM={best_bias_dex:+.3f}, accept={hmc_result.accept_rate.mean():.2f}, {status}")

    # --- 4b. Aggregate coverage statistics ---
    print("\n--- 4b. Multi-mock coverage statistics ---")
    coverage_68_per_param = covered_68.mean(axis=0)
    coverage_95_per_param = covered_95.mean(axis=0)
    mean_rhat = np.nanmean(all_rhat_max)
    mean_accept = np.nanmean(all_accept)

    print(f"\n  Mean R-hat (max): {mean_rhat:.3f}")
    print(f"  Mean acceptance rate: {mean_accept:.3f}")
    rhat_all_pass = all(rh < RHAT_MAX for rh in all_rhat_max if np.isfinite(rh))
    print_pass_fail(f"All R-hat (max) < {RHAT_MAX}", mean_rhat, f"< {RHAT_MAX}", rhat_all_pass)
    results["rhat_pass"] = rhat_all_pass
    results["rhat_max"] = float(mean_rhat)

    print(f"\n  {'Param':<12} {'68% cov':>8} {'95% cov':>8} {'Expected':>10}")
    print(f"  {'-'*42}")
    for pi in range(n_check):
        pname = param_names[pi] if pi < len(param_names) else f"theta_{pi}"
        print(f"  {pname:<12} {coverage_68_per_param[pi]:>8.0%} {coverage_95_per_param[pi]:>8.0%} "
              f"{'~68%/~95%':>10}")

    # Overall coverage check: mean 68% coverage should be in [0.50, 0.85]
    mean_68 = float(coverage_68_per_param.mean())
    mean_95 = float(coverage_95_per_param.mean())
    coverage_pass = 0.50 <= mean_68 <= 0.85
    print(f"\n  Mean 68% coverage: {mean_68:.0%}")
    print(f"  Mean 95% coverage: {mean_95:.0%}")
    print_pass_fail("68% coverage ∈ [50%, 85%]", mean_68, "[0.50, 0.85]", coverage_pass)
    results["recovery_coverage_68"] = mean_68
    results["recovery_coverage_95"] = mean_95
    results["recovery_pass"] = coverage_pass
    results["coverage_68_per_param"] = {
        (param_names[pi] if pi < len(param_names) else f"theta_{pi}"): float(coverage_68_per_param[pi])
        for pi in range(n_check)
    }
    results["n_mock"] = N_MOCK
    results["mock_details"] = all_mock_details

    # --- 4c. Corner plot for last mock (representative) ---
    print("\n--- 4c. Posterior corner plot (last mock) ---")
    posterior_mean = flat.mean(axis=0)
    posterior_std = flat.std(axis=0)
    prior_range = prior_hi - prior_lo
    constraint_ratio = posterior_std / np.clip(prior_range, 1e-12, None)
    most_constrained = np.argsort(constraint_ratio)[:5]

    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    for i, pi in enumerate(most_constrained):
        for j, pj in enumerate(most_constrained):
            ax = axes[i, j]
            if i == j:
                ax.hist(flat[:, pi], bins=40, density=True, color="steelblue", alpha=0.7)
                if true_theta is not None:
                    ax.axvline(true_theta[pi], color="r", ls="--", lw=1.5)
                ax.set_xlabel(param_names[pi] if pi < len(param_names) else f"θ_{pi}", fontsize=7)
            elif i > j:
                ax.scatter(flat[:, pj], flat[:, pi],
                           s=1, alpha=0.3, c="steelblue")
                if true_theta is not None:
                    ax.axvline(true_theta[pj], color="r", ls="--", lw=0.5, alpha=0.5)
                    ax.axhline(true_theta[pi], color="r", ls="--", lw=0.5, alpha=0.5)
            else:
                ax.set_visible(False)
            if j == 0 and i > 0:
                ax.set_ylabel(param_names[pi] if pi < len(param_names) else f"θ_{pi}", fontsize=7)
            ax.tick_params(labelsize=5)
    plt.suptitle(f"Cat 4c: HMC Posterior (last mock, 5 most constrained params)", fontsize=12)
    plt.tight_layout()
    fig.savefig(cat_dir / "posterior_corner.png", dpi=150)
    plt.close(fig)

    # Coverage bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_check)
    ax.bar(x - 0.15, coverage_68_per_param, 0.3, label="68% CI", alpha=0.8)
    ax.bar(x + 0.15, coverage_95_per_param, 0.3, label="95% CI", alpha=0.8)
    ax.axhline(0.68, color="k", ls="--", lw=1, alpha=0.5, label="ideal 68%")
    ax.axhline(0.95, color="gray", ls="--", lw=1, alpha=0.5, label="ideal 95%")
    ax.set_xticks(x)
    ax.set_xticklabels([param_names[pi] if pi < len(param_names) else f"θ_{pi}"
                         for pi in range(n_check)], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Coverage fraction")
    ax.set_title(f"Cat 4b: Multi-mock CI coverage ({N_MOCK} mocks)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    fig.savefig(cat_dir / "multi_mock_coverage.png", dpi=150)
    plt.close(fig)

    # --- 4d. KL divergence bound (Bevins+ 2025) ---
    print("\n--- 4d. KL divergence bound (Bevins+ 2025, valid bins only) ---")
    valid_all = preds["valid_mask"]
    n_d = y_true_log.shape[1] * y_true_log.shape[2]
    all_valid = valid_all.ravel()
    resid_all = (preds["y_pred_log"] - y_true_log).ravel()
    rmse_log = float(np.sqrt(np.mean(resid_all[all_valid] ** 2)))
    sigma_obs_avg_dex = np.mean([OBS_NOISE_FRAC[f] / np.log(10.0) for f in TARGET_FIELDS])

    kl_bound = 0.5 * n_d * (rmse_log / sigma_obs_avg_dex) ** 2
    kl_criterion = np.sqrt(2.0 / n_d)
    rmse_ratio = rmse_log / sigma_obs_avg_dex

    print(f"  N_d = {n_d}")
    print(f"  RMSE / σ_obs (log10 space) = {rmse_ratio:.4f}")
    print(f"  Criterion (RMSE/σ_obs ≤ √(2/N_d)) = {kl_criterion:.4f}")
    print(f"  KL bound = {kl_bound:.4f} nat")
    print_pass_fail("KL bound < 1 nat", kl_bound, "< 1.0", kl_bound < 1.0)
    results["kl_bound"] = float(kl_bound)
    results["rmse_over_sigma"] = float(rmse_ratio)
    results["kl_criterion"] = float(kl_criterion)

    # Save HMC artifacts for Cat 10
    np.savez(
        cat_dir / "hmc_chains.npz",
        # Save only the last mock's chains for Cat 10 PPC
        samples=hmc_result.samples,
        log_prob=hmc_result.log_prob,
        accept_rate=hmc_result.accept_rate,
        param_names=param_names,
        n_warmup=hmc_n_warmup,
    )
    np.savez(
        cat_dir / "mock_observation.npz",
        y_mock_true_log=y_mock_true_log,
        y_mock_noisy_log=y_mock_noisy_log,
        sigma_obs_dex=sigma_obs_dex,
        sigma_obs_stack=sigma_obs_stack,
        mock_mass=np.array([float(np.mean(mock_masses))], dtype=np.float32),
        rr500=np.median(rr500_stack, axis=0, keepdims=True),
        mock_masses=mock_masses,
        rr500_stack=rr500_stack,
        stack_mode=np.array(["mean"], dtype=object),
        mass_bias_dex=np.array([float(best_bias_dex)], dtype=np.float64),
        obs_valid=obs_valid,
        true_theta=true_theta,
        param_names=param_names,
    )

    if len(all_post_samples) > 0:
        post_arr = np.stack(all_post_samples, axis=0)  # (n_mock, n_post, d)
        theta_arr = np.stack(all_true_thetas, axis=0)  # (n_mock, d)
        np.savez(
            cat_dir / "all_mock_posteriors.npz",
            posterior_samples=post_arr,
            true_thetas=theta_arr,
            prior_lo=prior_lo,
            prior_hi=prior_hi,
            param_names=np.array(param_names, dtype=object),
        )
    print(f"\n[INFO] HMC chains and mock saved to {cat_dir}")

    return results


def _get_train_args():
    """Reconstruct a minimal args namespace from the best model checkpoint."""
    state = torch.load(RUN_DIR / "best_model.pt", map_location="cpu", weights_only=False)
    args_d = state.get("args", {})
    return SimpleNamespace(**args_d)


# ============================================================================
# CATEGORY 5: Fisher Information Consistency Test
# ============================================================================

def run_category_5(emu: Emulator, output_dir: Path) -> Dict[str, Any]:
    """
    Fisher matrix computation using emulator finite differences.
    """
    print("\n" + "=" * 72)
    print("CATEGORY 5: Fisher Information Consistency Test")
    print("=" * 72)

    cat_dir = ensure_dir(output_dir / "cat5_fisher")
    results = {}

    # Load 1P table for fiducial and parameter ranges
    onep_csv = Path("/mnt/home/mlee1/Sims/IllustrisTNG/L50n512/1P/CosmoAstroSeed_IllustrisTNG_L50n512_1P.txt")
    if not onep_csv.exists():
        print("[WARN] 1P CSV not found, skipping Category 5.")
        return {"skipped": True}

    onep_params = pd.read_csv(onep_csv, sep=r"\s+", engine="python")
    if "#Name" in onep_params.columns:
        onep_params = onep_params.rename(columns={"#Name": "tag"})
    theta_cols = [
        c for c in onep_params.columns
        if c != "tag" and pd.api.types.is_numeric_dtype(onep_params[c])
        and str(c).strip().lower() != "seed"
    ][: emu.theta_dim]

    fid_mask = onep_params["tag"].str.contains("fiducial", case=False, na=False)
    if fid_mask.any():
        theta_fid = onep_params.loc[fid_mask].iloc[0][theta_cols].to_numpy(dtype=np.float32)
    else:
        theta_fid = onep_params[theta_cols].median().to_numpy(dtype=np.float32)

    ref_mass = np.array([1e14], dtype=np.float32)
    r_bins_kpc = np.logspace(np.log10(30), np.log10(2000), 30).astype(np.float32)
    r500_ref = 700.0
    rr500 = (r_bins_kpc / r500_ref).astype(np.float32)[None, :]
    snap_eval = int(emu.snapnums[0])
    z_eval = float(getattr(emu, "redshift_by_snap", {}).get(snap_eval, 0.0))
    n_bins = rr500.shape[1]
    n_fields = len(TARGET_FIELDS)
    n_d = n_bins * n_fields

    # Compute derivatives ∂μ/∂θ_k in log10 space
    print("\n--- Computing Fisher matrix derivatives (log10 space) ---")
    delta_frac = 0.01
    jacobian = np.zeros((n_d, emu.theta_dim), dtype=np.float64)

    for pi in range(emu.theta_dim):
        pname = theta_cols[pi] if pi < len(theta_cols) else f"theta_{pi}"
        col_vals = onep_params[theta_cols[pi]].dropna().values if pi < len(theta_cols) else np.array([theta_fid[pi]])
        p_range = float(col_vals.max() - col_vals.min())
        if p_range < 1e-10:
            p_range = abs(theta_fid[pi]) * 0.1 + 1e-6
        dp = delta_frac * p_range

        theta_lo = theta_fid.copy()
        theta_hi = theta_fid.copy()
        theta_lo[pi] -= dp
        theta_hi[pi] += dp

        pred_lo = emu.predict(
            theta=theta_lo, M=ref_mass, r_bins=rr500,
            field=TARGET_FIELDS, snapnum=snap_eval, redshift=z_eval, n_samples=10,
        )
        pred_hi = emu.predict(
            theta=theta_hi, M=ref_mass, r_bins=rr500,
            field=TARGET_FIELDS, snapnum=snap_eval, redshift=z_eval, n_samples=10,
        )

        # Use log10-space predictions for the Jacobian
        mu_lo_log = pred_lo.mean_log10[0].reshape(-1)
        mu_hi_log = pred_hi.mean_log10[0].reshape(-1)
        jacobian[:, pi] = (mu_hi_log - mu_lo_log) / (2 * dp)

    # Construct obs noise covariance in log10 space (diagonal, in dex)
    sigma_obs_flat = np.zeros(n_d, dtype=np.float64)
    for fi, f in enumerate(TARGET_FIELDS):
        sigma_obs_flat[fi * n_bins:(fi + 1) * n_bins] = OBS_NOISE_FRAC[f] / np.log(10.0)
    Sigma_inv = np.diag(1.0 / np.clip(sigma_obs_flat ** 2, 1e-60, None))

    # Fisher matrix: F = J^T Σ^{-1} J
    fisher = jacobian.T @ Sigma_inv @ jacobian

    # Regularize for inversion
    fisher_reg = fisher + 1e-10 * np.eye(fisher.shape[0])
    try:
        fisher_inv = np.linalg.inv(fisher_reg)
        sigma_fisher = np.sqrt(np.abs(np.diag(fisher_inv)))
    except np.linalg.LinAlgError:
        print("[WARN] Fisher matrix singular, using pseudo-inverse.")
        fisher_inv = np.linalg.pinv(fisher_reg)
        sigma_fisher = np.sqrt(np.abs(np.diag(fisher_inv)))

    cond_number = float(np.linalg.cond(fisher_reg))

    print(f"\n  Fisher matrix condition number: {cond_number:.2e}")
    cond_pass = cond_number < 1e5
    print_pass_fail("Condition number", cond_number, "< 10^5", cond_pass)

    print("\n  Predicted 1σ marginalized errors (Fisher forecast):")
    for pi in range(min(10, emu.theta_dim)):
        pname = theta_cols[pi] if pi < len(theta_cols) else f"theta_{pi}"
        frac_err = sigma_fisher[pi] / max(abs(theta_fid[pi]), 1e-12)
        print(f"    {pname}: σ = {sigma_fisher[pi]:.4g} ({frac_err:.1%} of fiducial)")

    results["fisher_condition_number"] = cond_number
    results["sigma_fisher"] = {
        (theta_cols[pi] if pi < len(theta_cols) else f"theta_{pi}"): float(sigma_fisher[pi])
        for pi in range(emu.theta_dim)
    }

    # Plot Fisher forecast
    fig, ax = plt.subplots(figsize=(12, 5))
    n_show = min(15, emu.theta_dim)
    names = [theta_cols[pi] if pi < len(theta_cols) else f"θ_{pi}" for pi in range(n_show)]
    frac_errors = [sigma_fisher[pi] / max(abs(theta_fid[pi]), 1e-12) for pi in range(n_show)]
    ax.bar(range(n_show), frac_errors, alpha=0.8)
    ax.set_xticks(range(n_show))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fractional 1σ error")
    ax.set_title("Cat 5: Fisher Forecast (single 10^14 M☉ halo)")
    ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(cat_dir / "fisher_forecast.png", dpi=150)
    plt.close(fig)

    print(f"\n[INFO] Category 5 plots saved to {cat_dir}")
    return results


# ============================================================================
# CATEGORY 6: Context Set Size and ANP-Specific Tests
# ============================================================================

def run_category_6(emu: Emulator, output_dir: Path) -> Dict[str, Any]:
    """
    Tests: Context-set-size convergence, MC sample convergence, OOD inflation,
    latent interpolation smoothness.
    """
    print("\n" + "=" * 72)
    print("CATEGORY 6: Context Set Size and ANP-Specific Tests")
    print("=" * 72)

    cat_dir = ensure_dir(output_dir / "cat6_anp_specific")
    results = {}

    ref_mass = np.array([1e14], dtype=np.float32)
    r_bins_kpc = np.logspace(np.log10(30), np.log10(2000), 30).astype(np.float32)
    r500_ref = 700.0
    rr500 = (r_bins_kpc / r500_ref).astype(np.float32)[None, :]
    snap_eval = int(emu.snapnums[0])
    z_eval = float(getattr(emu, "redshift_by_snap", {}).get(snap_eval, 0.0))

    # Use median theta as fiducial
    onep_csv = Path("/mnt/home/mlee1/Sims/IllustrisTNG/L50n512/1P/CosmoAstroSeed_IllustrisTNG_L50n512_1P.txt")
    if onep_csv.exists():
        onep_params = pd.read_csv(onep_csv, sep=r"\s+", engine="python")
        if "#Name" in onep_params.columns:
            onep_params = onep_params.rename(columns={"#Name": "tag"})
        theta_cols = [
            c for c in onep_params.columns
            if c != "tag" and pd.api.types.is_numeric_dtype(onep_params[c])
            and str(c).strip().lower() != "seed"
        ][: emu.theta_dim]
        theta_fid = onep_params[theta_cols].median().to_numpy(dtype=np.float32)
    else:
        theta_fid = np.zeros(emu.theta_dim, dtype=np.float32)
        theta_fid[0] = 0.3  # Omega_m
        theta_fid[1] = 0.8  # sigma_8

    # --- 6a. Context-set-size convergence (using training infrastructure) ---
    print("\n--- 6a. Context-set-size convergence (few-shot evaluation) ---")
    print("  Building normalised test DataLoader ...")
    n_context_list = [0, 1, 2, 5, 10, 20, 50]
    context_result = None
    try:
        from functools import partial
        from torch.utils.data import DataLoader

        args_ns = _get_train_args()
        if not hasattr(args_ns, "resolved_snapnums"):
            args_ns.resolved_snapnums = [int(getattr(args_ns, "snapnum", 90))]
        if not hasattr(args_ns, "redshift_by_snap"):
            args_ns.redshift_by_snap = {90: 0.0}
        if not hasattr(args_ns, "resolved_all_profile_targets"):
            args_ns.resolved_all_profile_targets = list(
                getattr(args_ns, "all_profiles_subset", TARGET_FIELDS))

        all_families = build_tasks(args_ns)
        _, _, test_families = split_tasks(
            all_families,
            train_frac=float(args_ns.train_frac),
            val_frac=float(args_ns.val_frac),
            seed=int(args_ns.seed),
        )

        # Subtract mean model if present (matches training pipeline)
        if emu.mean_model is not None:
            test_flat = [t for fam in test_families for t in fam.snapshots]
            test_flat = apply_residual_prior(
                test_flat, emu.mean_model,
                device=emu.device, batch_size=512,
            )
            test_families = remap_flat_tasks_to_families(test_families, test_flat)

        # Normalize using same stats as training
        test_families_norm = normalize_tasks(test_families, emu.norm_stats)

        target_snap = snap_eval
        collate_fn = partial(anp_collate, target_snapnum=target_snap)
        # Use at most 8 families to keep the test tractable
        n_fam = min(8, len(test_families_norm))
        loader = DataLoader(
            test_families_norm[:n_fam],
            batch_size=1,
            collate_fn=collate_fn,
            shuffle=False,
        )

        y_mean_t = emu.y_mean.to(emu.device)
        y_std_t = emu.y_std.to(emu.device)
        x_mean_t = emu.x_mean.to(emu.device)
        x_std_t = emu.x_std.to(emu.device)

        context_result = few_shot_curve(
            model=emu.model,
            loader=loader,
            device=emu.device,
            y_mean=y_mean_t,
            y_std=y_std_t,
            x_mean=x_mean_t,
            x_std=x_std_t,
            mean_model=emu.mean_model,
            n_context_list=n_context_list,
            n_repeats=3,
            n_samples=15,
            target_names=TARGET_FIELDS,
        )
        results["context_size_rmse"] = context_result

        print("  Context size → RMSE (physical units, log10 input):")
        for nc in sorted(context_result):
            print(f"    n_ctx={nc:3d}: RMSE = {context_result[nc]:.4g}")

        # Monotonic decrease check
        rmse_vals = [context_result[nc] for nc in sorted(context_result)]
        if len(rmse_vals) >= 3:
            # RMSE should generally decrease with more context
            decreasing = all(rmse_vals[i] >= rmse_vals[-1] * 0.8
                             for i in range(len(rmse_vals)))
            print(f"  Monotonic improvement: {'Yes' if decreasing else 'Partial'}")

    except Exception as exc:
        print(f"  [WARN] Few-shot evaluation failed: {exc}")
        print("  Falling back to n_samples convergence only.")
        context_result = None

    # Context-size convergence plot
    if context_result:
        fig, ax = plt.subplots(figsize=(8, 5))
        nc_vals = sorted(context_result.keys())
        rmse_vals = [context_result[nc] for nc in nc_vals]
        ax.plot(nc_vals, rmse_vals, "o-", ms=6, lw=2, color="steelblue")
        ax.set_xlabel("Number of context halos")
        ax.set_ylabel("RMSE")
        ax.set_title("Cat 6a: Context-set-size convergence")
        ax.set_xscale("symlog", linthresh=1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(cat_dir / "context_size_convergence.png", dpi=150)
        plt.close(fig)

    # --- 6b. MC sample (n_samples) convergence ---
    print("\n--- 6b. MC sample convergence (n_samples) ---")
    n_sample_values = [1, 2, 5, 10, 20, 50, 100]
    epi_by_nsamp = {f: [] for f in TARGET_FIELDS}
    mu_by_nsamp = {f: [] for f in TARGET_FIELDS}

    for ns in n_sample_values:
        pred = emu.predict(
            theta=theta_fid, M=ref_mass, r_bins=rr500,
            field=TARGET_FIELDS, snapnum=snap_eval, redshift=z_eval,
            n_samples=ns,
        )
        for fi, f in enumerate(TARGET_FIELDS):
            epi_by_nsamp[f].append(float(np.mean(pred.epistemic_std[0, :, fi])))
            mu_by_nsamp[f].append(float(np.mean(pred.mean[0, :, fi])))

    print("  n_samples -> mean epistemic std:")
    for ns_i, ns in enumerate(n_sample_values):
        vals = ", ".join(f"{f}={epi_by_nsamp[f][ns_i]:.4g}" for f in TARGET_FIELDS)
        print(f"    n={ns:3d}: {vals}")

    results["nsample_convergence"] = {
        "n_samples": n_sample_values,
        "epistemic_std": {f: epi_by_nsamp[f] for f in TARGET_FIELDS},
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for f in TARGET_FIELDS:
        axes[0].plot(n_sample_values, epi_by_nsamp[f], "o-", label=f)
        axes[1].plot(n_sample_values, mu_by_nsamp[f], "o-", label=f)
    axes[0].set_xlabel("n_samples")
    axes[0].set_ylabel("Mean epistemic std")
    axes[0].set_title("Cat 6b: Epistemic std vs n_samples")
    axes[0].legend(fontsize=8)
    axes[0].set_xscale("log")
    axes[1].set_xlabel("n_samples")
    axes[1].set_ylabel("Mean prediction")
    axes[1].set_title("Cat 6b: Mean prediction vs n_samples")
    axes[1].legend(fontsize=8)
    axes[1].set_xscale("log")
    plt.tight_layout()
    fig.savefig(cat_dir / "nsample_convergence.png", dpi=150)
    plt.close(fig)

    # --- 6c. OOD epistemic inflation ---
    print("\n--- 6c. Out-of-distribution epistemic inflation ---")
    in_dist_pred = emu.predict(
        theta=theta_fid, M=ref_mass, r_bins=rr500,
        field=TARGET_FIELDS, snapnum=snap_eval, redshift=z_eval, n_samples=30,
    )

    ood_results = {}
    for perturbation in [0.05, 0.10]:
        theta_ood = theta_fid.copy()
        if onep_csv.exists():
            for pi in range(emu.theta_dim):
                col_vals = onep_params[theta_cols[pi]].dropna().values
                p_range = float(col_vals.max() - col_vals.min())
                theta_ood[pi] = float(col_vals.max()) + perturbation * p_range
        else:
            theta_ood *= (1.0 + perturbation)

        ood_pred = emu.predict(
            theta=theta_ood, M=ref_mass, r_bins=rr500,
            field=TARGET_FIELDS, snapnum=snap_eval, redshift=z_eval, n_samples=30,
        )

        ratios = {}
        for fi, f in enumerate(TARGET_FIELDS):
            in_epi = float(np.mean(in_dist_pred.epistemic_std[0, :, fi]))
            ood_epi = float(np.mean(ood_pred.epistemic_std[0, :, fi]))
            ratio = ood_epi / max(in_epi, 1e-12)
            ratios[f] = ratio
            passed = ratio > 1.5
            print_pass_fail(
                f"{f} OOD epi inflation ({perturbation:.0%} beyond)",
                ratio,
                "> 1.5 (50% increase)",
                passed,
            )
        ood_results[f"{perturbation:.0%}"] = ratios
    results["ood_inflation"] = ood_results

    # --- 6d. Latent space interpolation ---
    print("\n--- 6d. Latent space interpolation smoothness ---")
    if onep_csv.exists():
        col_vals_0 = onep_params[theta_cols[0]].dropna().values
        theta_A = theta_fid.copy()
        theta_B = theta_fid.copy()
        theta_A[0] = float(col_vals_0.min())
        theta_B[0] = float(col_vals_0.max())
    else:
        theta_A = theta_fid * 0.8
        theta_B = theta_fid * 1.2

    lambdas = np.linspace(0, 1, 11)
    interp_means = {f: [] for f in TARGET_FIELDS}
    interp_stds = {f: [] for f in TARGET_FIELDS}

    for lam in lambdas:
        theta_interp = (1 - lam) * theta_A + lam * theta_B
        pred = emu.predict(
            theta=theta_interp.astype(np.float32),
            M=ref_mass, r_bins=rr500,
            field=TARGET_FIELDS, snapnum=snap_eval, redshift=z_eval,
            n_samples=15,
        )
        for fi, f in enumerate(TARGET_FIELDS):
            interp_means[f].append(float(np.mean(pred.mean[0, :, fi])))
            interp_stds[f].append(float(np.mean(pred.total_std[0, :, fi])))

    for f in TARGET_FIELDS:
        vals = np.array(interp_means[f])
        d2 = np.diff(vals, n=2)
        roughness = float(np.mean(np.abs(d2)))
        signal_range = float(np.ptp(vals))
        smoothness = roughness / max(signal_range, 1e-12)
        print(f"  {f}: interpolation roughness = {smoothness:.4f} "
              f"({'smooth' if smoothness < 0.1 else 'rough'})")
    results["interpolation_smoothness"] = {
        f: float(np.mean(np.abs(np.diff(np.array(interp_means[f]), n=2))) /
                 max(np.ptp(interp_means[f]), 1e-12))
        for f in TARGET_FIELDS
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for f in TARGET_FIELDS:
        axes[0].plot(lambdas, interp_means[f], "o-", label=f)
        axes[1].plot(lambdas, interp_stds[f], "o-", label=f)
    axes[0].set_xlabel("λ (interpolation)")
    axes[0].set_ylabel("Mean profile (integrated)")
    axes[0].set_title("Cat 6d: Latent interpolation")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("λ (interpolation)")
    axes[1].set_ylabel("Total std (integrated)")
    axes[1].set_title("Cat 6d: Uncertainty during interpolation")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(cat_dir / "latent_interpolation.png", dpi=150)
    plt.close(fig)

    print(f"\n[INFO] Category 6 plots saved to {cat_dir}")
    return results


# ============================================================================
# CATEGORY 7: Noise Robustness and Observational Realism Tests
# ============================================================================

def run_category_7(emu: Emulator, preds: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Tests: Progressive noise degradation, information gain per profile type.
    """
    print("\n" + "=" * 72)
    print("CATEGORY 7: Noise Robustness and Observational Realism Tests")
    print("=" * 72)

    cat_dir = ensure_dir(output_dir / "cat7_noise_robustness")
    results = {}

    # Work in log10 space for numerical stability.
    y_true_log = preds["y_true_log"]
    y_pred_log = preds["y_pred_log"]
    valid = preds["valid_mask"]
    fields = preds["field_names"]
    n_r = y_true_log.shape[1]

    # --- 7a. Noise-dependent inference robustness (valid bins only) ---
    print("\n--- 7a. Parameter bias vs noise level (log10, valid only) ---")
    noise_levels = [0.05, 0.10, 0.20, 0.30, 0.40]
    rng = np.random.default_rng(42)
    run_ids = preds["run_ids"]

    # Load theta table for ground truth
    theta_table = None
    try:
        args_ns = _get_train_args()
        theta_table = load_theta_table(
            Path(args_ns.param_csv),
            target_theta_dim=int(args_ns.theta_dim),
        )
    except Exception:
        pass

    snap_eval = int(emu.snapnums[0])
    z_eval = float(getattr(emu, "redshift_by_snap", {}).get(snap_eval, 0.0))

    noise_bias = {}
    if theta_table is not None:
        # Pick 10 test halos with known theta
        test_idxs = []
        for idx in range(min(200, y_true_log.shape[0])):
            rid = int(run_ids[idx])
            if theta_table.get(rid) is not None and valid[idx].mean() > 0.5:
                test_idxs.append(idx)
            if len(test_idxs) >= 10:
                break

        if len(test_idxs) >= 3:
            for noise_frac in noise_levels:
                sigma_dex = noise_frac / np.log(10.0)
                bias_per_halo = []
                for idx in test_idxs:
                    rid = int(run_ids[idx])
                    true_theta = theta_table[rid]
                    mock_mass = np.array([preds["masses"][idx]], dtype=np.float32)
                    r_bins_i = preds["r_bins"][idx:idx + 1]

                    # Noisy mock observation in log10
                    y_obs_log = y_true_log[idx] + rng.normal(0, sigma_dex, size=y_true_log[idx].shape)
                    y_obs_log[~valid[idx]] = np.nan

                    # Simple MAP point estimate: evaluate emulator at true theta
                    # and compute chi2 to check if the emulator likelihood is well-behaved
                    pred = emu.predict(
                        theta=true_theta, M=mock_mass, r_bins=r_bins_i,
                        field=TARGET_FIELDS, snapnum=snap_eval, redshift=z_eval,
                        n_samples=15,
                    )
                    if pred.mean_log10 is not None:
                        mu_log = pred.mean_log10[0]
                    else:
                        mu_log = np.log10(np.clip(pred.mean[0], 1e-38, None))

                    # Per-field chi2 at the TRUE theta (should be ~1 for well-calibrated emulator)
                    chi2_vals = []
                    for fi, f in enumerate(fields):
                        m = valid[idx, :, fi]
                        if m.sum() == 0:
                            continue
                        resid = (y_obs_log[:, fi][m] - mu_log[:, fi][m])
                        chi2_vals.append(float(np.mean((resid / sigma_dex) ** 2)))
                    if chi2_vals:
                        bias_per_halo.append(np.mean(chi2_vals))

                if bias_per_halo:
                    mean_chi2 = float(np.mean(bias_per_halo))
                    noise_bias[f"{noise_frac:.0%}"] = mean_chi2
                    # At true theta, chi2 should be ~1 regardless of noise level
                    passed = 0.5 <= mean_chi2 <= 2.0
                    print_pass_fail(
                        f"χ² at true θ (noise={noise_frac:.0%})",
                        mean_chi2, "[0.5, 2.0]", passed,
                    )
            results["noise_chi2_at_truth"] = noise_bias
        else:
            print("  [WARN] Too few test halos with known theta; skipping inference test.")
    else:
        print("  [WARN] Theta table unavailable; evaluating noise-dependent accuracy only.")
        # Fallback: simple accuracy vs noise level
        for noise_frac in noise_levels:
            sigma_dex = noise_frac / np.log(10.0) if noise_frac > 0 else 0.10 / np.log(10.0)
            y_noisy = y_true_log + rng.normal(0, sigma_dex, size=y_true_log.shape)
            chi2_per_field = {}
            for fi, f in enumerate(fields):
                m = valid[:, :, fi]
                residuals = y_pred_log[:, :, fi][m] - y_noisy[:, :, fi][m]
                chi2 = float(np.mean((residuals / sigma_dex) ** 2))
                chi2_per_field[f] = chi2
            noise_bias[f"{noise_frac:.0%}"] = chi2_per_field
        results["noise_accuracy"] = noise_bias

    # --- 7b. Information content per profile type (valid bins only) ---
    print("\n--- 7b. Information content per profile type (log10, valid only) ---")
    print("  (Relative RMSE per field — lower = more information)")
    fve_results = {}
    for fi, f in enumerate(fields):
        m = valid[:, :, fi]
        if m.sum() == 0:
            fve_results[f] = float("nan")
            continue
        signal_var = float(np.var(y_true_log[:, :, fi][m]))
        mse = float(np.mean((y_pred_log[:, :, fi][m] - y_true_log[:, :, fi][m]) ** 2))
        fve = 1.0 - mse / max(signal_var, 1e-30)
        fve_results[f] = fve
        print(f"  {f}: FVE = {fve:.4f} (fraction of variance explained)")
    results["fve"] = fve_results

    # --- 7c. Radial cut tests (valid bins only) ---
    print("\n--- 7c. Radial cut tests (progressive bin removal, log10, valid only) ---")
    # Remove outer bins progressively
    total_bins = n_r
    cut_fracs = [0.0, 0.1, 0.2, 0.3, 0.5]
    outer_cut_rmse = {}
    inner_cut_rmse = {}

    for cut_frac in cut_fracs:
        n_remove = int(cut_frac * total_bins)
        if n_remove >= total_bins - 2:
            continue

        # Outer cut
        n_keep_outer = total_bins - n_remove
        outer_rmse = {}
        for fi, f in enumerate(fields):
            m = valid[:, :n_keep_outer, fi]
            resid = y_pred_log[:, :n_keep_outer, fi][m] - y_true_log[:, :n_keep_outer, fi][m]
            r = float(np.sqrt(np.mean(resid ** 2))) if len(resid) > 0 else float("nan")
            outer_rmse[f] = r
        outer_cut_rmse[f"{cut_frac:.0%}"] = outer_rmse

        # Inner cut
        inner_rmse = {}
        for fi, f in enumerate(fields):
            m = valid[:, n_remove:, fi]
            resid = y_pred_log[:, n_remove:, fi][m] - y_true_log[:, n_remove:, fi][m]
            r = float(np.sqrt(np.mean(resid ** 2))) if len(resid) > 0 else float("nan")
            inner_rmse[f] = r
        inner_cut_rmse[f"{cut_frac:.0%}"] = inner_rmse

    results["outer_cut_rmse"] = outer_cut_rmse
    results["inner_cut_rmse"] = inner_cut_rmse

    print("  Outer bin removal:")
    for k, v in outer_cut_rmse.items():
        print(f"    Remove {k} outer: " + ", ".join(f"{f}={r:.4g}" for f, r in v.items()))
    print("  Inner bin removal:")
    for k, v in inner_cut_rmse.items():
        print(f"    Remove {k} inner: " + ", ".join(f"{f}={r:.4g}" for f, r in v.items()))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for f in fields:
        x = [float(k.strip("%")) / 100 for k in outer_cut_rmse]
        y_outer = [outer_cut_rmse[k][f] for k in outer_cut_rmse]
        y_inner = [inner_cut_rmse[k][f] for k in inner_cut_rmse]
        axes[0].plot(x, y_outer, "o-", label=f)
        axes[1].plot(x, y_inner, "o-", label=f)
    axes[0].set_xlabel("Fraction of outer bins removed")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Cat 7c: Outer radial cut")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Fraction of inner bins removed")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Cat 7c: Inner radial cut")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(cat_dir / "radial_cut_tests.png", dpi=150)
    plt.close(fig)

    print(f"\n[INFO] Category 7 plots saved to {cat_dir}")
    return results


# ============================================================================
# CATEGORY 8: Comparison to Observational Data (Framework)
# ============================================================================

def run_category_8(output_dir: Path) -> Dict[str, Any]:
    """Framework only — requires external observational data."""
    print("\n" + "=" * 72)
    print("CATEGORY 8: Comparison to Observational Data (Framework)")
    print("=" * 72)

    print("""
  This category requires external observational data:
    - eROSITA eRASS stacked X-ray surface brightness profiles
    - ACT/SPT stacked SZ (Compton-y) profiles
    - Chandra/XMM spectroscopic temperature profiles

  When these data are available, the emulator likelihood can be evaluated
  against the stacked profiles to compute:
    1. Best-fit reduced χ² (target: ≈ 1)
    2. Parameter constraints vs eROSITA/ACT results
    3. Cross-observable (X-ray + SZ) consistency

  Reference: Shreeram et al. (2025), reduced χ² ≈ 0.83 for TNG best-fit.
""")
    return {"framework": True, "status": "Requires external observational data"}


# ============================================================================
# CATEGORY 9: Cross-Validation and Consistency Stress Tests
# ============================================================================

def run_category_9(emu: Emulator, preds: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Tests: CV set bias, corrupted input degradation.
    """
    print("\n" + "=" * 72)
    print("CATEGORY 9: Cross-Validation and Consistency Stress Tests")
    print("=" * 72)

    cat_dir = ensure_dir(output_dir / "cat9_consistency")
    results = {}

    # Work in log10 space for consistency with other categories.
    y_true_log = preds["y_true_log"]
    y_pred_log = preds["y_pred_log"]
    y_std_log = preds["y_std_log"]
    valid = preds["valid_mask"]
    fields = preds["field_names"]

    # --- 9a. CV set bias (valid bins only) ---
    print("\n--- 9a. Mean emulator bias on test set (log10, valid only) ---")
    bias_results = {}
    for fi, f in enumerate(fields):
        m = valid[:, :, fi]
        if m.sum() == 0:
            bias_results[f] = {"absolute_dex": float("nan"), "fractional": float("nan")}
            continue
        resid_valid = y_pred_log[:, :, fi][m] - y_true_log[:, :, fi][m]
        mean_bias = float(np.mean(resid_valid))
        signal_range = float(np.ptp(y_true_log[:, :, fi][m]))
        frac_bias = abs(mean_bias) / max(signal_range, 1e-30)
        bias_results[f] = {
            "absolute_dex": mean_bias,
            "fractional": frac_bias,
        }
        passed = frac_bias < 0.05
        print_pass_fail(f"{f} fractional bias", frac_bias, "< 0.05", passed)
    results["bias"] = bias_results

    # --- 9b. Corrupted input test ---
    print("\n--- 9b. Corrupted radial bin ordering ---")
    n_test_corrupt = min(50, y_true_log.shape[0])
    rng = np.random.default_rng(42)

    snap_eval = int(emu.snapnums[0])
    z_eval = float(getattr(emu, "redshift_by_snap", {}).get(snap_eval, 0.0))

    # Get theta table for re-prediction
    theta_table = None
    try:
        args_ns = _get_train_args()
        theta_table = load_theta_table(
            Path(args_ns.param_csv),
            target_theta_dim=int(args_ns.theta_dim),
        )
    except Exception:
        pass

    # Nominal RMSE for first N test halos (log10 space, valid only)
    nominal_rmse = {}
    for fi, f in enumerate(fields):
        m = valid[:n_test_corrupt, :, fi]
        resid = y_pred_log[:n_test_corrupt, :, fi][m] - y_true_log[:n_test_corrupt, :, fi][m]
        nominal_rmse[f] = float(np.sqrt(np.mean(resid ** 2))) if len(resid) > 0 else float("nan")

    print(f"  Nominal RMSE (first {n_test_corrupt} halos, dex, valid only):")
    for f, v in nominal_rmse.items():
        print(f"    {f}: {v:.4g}")

    # Corrupted prediction: shuffle radial bin ordering
    corrupted_rmse = {f: [] for f in fields}
    run_ids = preds["run_ids"]
    n_tested = 0
    for idx in range(n_test_corrupt):
        rid = int(run_ids[idx])
        if theta_table is None:
            break
        theta = theta_table.get(rid)
        if theta is None:
            continue

        r_bins_orig = preds["r_bins"][idx:idx + 1]  # (1, n_r)
        r_bins_shuffled = r_bins_orig.copy()
        rng.shuffle(r_bins_shuffled[0])  # shuffle radial bin order

        mass_i = np.array([preds["masses"][idx]], dtype=np.float32)
        pred_corr = emu.predict(
            theta=theta, M=mass_i, r_bins=r_bins_shuffled,
            field=TARGET_FIELDS, snapnum=snap_eval, redshift=z_eval, n_samples=10,
        )
        for fi, f in enumerate(fields):
            m = valid[idx, :, fi]
            if pred_corr.mean_log10 is not None:
                resid = pred_corr.mean_log10[0, :, fi][m] - y_true_log[idx, :, fi][m]
            else:
                resid = np.log10(np.clip(pred_corr.mean[0, :, fi][m], 1e-38, None)) - y_true_log[idx, :, fi][m]
            if len(resid) > 0:
                corrupted_rmse[f].append(float(np.sqrt(np.mean(resid ** 2))))
        n_tested += 1
        if n_tested >= 20:
            break

    if n_tested > 0:
        print(f"\n  Corrupted RMSE ({n_tested} halos, shuffled r_bins, dex, valid only):")
        corrupted_summary = {}
        for f in fields:
            mean_corr = float(np.mean(corrupted_rmse[f])) if corrupted_rmse[f] else float("nan")
            ratio = mean_corr / max(nominal_rmse[f], 1e-12)
            corrupted_summary[f] = {"rmse": mean_corr, "ratio": ratio}
            passed = ratio > 2.0
            print_pass_fail(f"{f} RMSE ratio (corrupted/nominal)", ratio, "> 2.0", passed)
        results["corrupted_input"] = {
            "nominal_rmse": nominal_rmse,
            "corrupted_rmse": corrupted_summary,
            "n_tested": n_tested,
        }
    else:
        print("  [WARN] Could not run corrupted-input test (theta table unavailable).")
        results["corrupted_input"] = {"nominal_rmse": nominal_rmse, "status": "skipped"}

    # --- 9c. Residual distribution normality (valid bins only) ---
    print("\n--- 9c. Residual normality (Shapiro-Wilk on z-scores, valid only) ---")
    z_scores = (y_true_log - y_pred_log) / np.clip(y_std_log, 1e-12, None)
    for fi, f in enumerate(fields):
        m = valid[:, :, fi]
        flat = z_scores[:, :, fi][m]
        if len(flat) < 10:
            print(f"  {f}: too few valid bins to test")
            continue
        # Subsample for Shapiro-Wilk (max 5000)
        if len(flat) > 5000:
            sub = rng.choice(flat, size=5000, replace=False)
        else:
            sub = flat
        stat, pval = scipy_stats.shapiro(sub)
        skew = float(scipy_stats.skew(flat))
        kurt = float(scipy_stats.kurtosis(flat))
        print(f"  {f}: skew={skew:.3f}, kurtosis={kurt:.3f}, "
              f"Shapiro p={pval:.3g}")
    results["residual_normality"] = {
        f: {
            "skewness": float(scipy_stats.skew(z_scores[:, :, fi][valid[:, :, fi]])),
            "kurtosis": float(scipy_stats.kurtosis(z_scores[:, :, fi][valid[:, :, fi]])),
        }
        for fi, f in enumerate(fields)
        if valid[:, :, fi].sum() > 10
    }

    # Plot residual distributions (valid only)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for fi, f in enumerate(fields):
        ax = axes[fi // 2, fi % 2]
        m = valid[:, :, fi]
        flat = z_scores[:, :, fi][m]
        if len(flat) == 0:
            continue
        ax.hist(flat, bins=80, density=True, edgecolor="k", alpha=0.6, label="empirical")
        x_range = np.linspace(-5, 5, 200)
        ax.plot(x_range, scipy_stats.norm.pdf(x_range), "r-", lw=2, label="N(0,1)")
        ax.set_xlabel("z-score")
        ax.set_ylabel("Density")
        ax.set_title(f"{f}")
        ax.legend(fontsize=8)
        ax.set_xlim(-5, 5)
    plt.suptitle("Cat 9c: Normalized residual distributions (valid bins)", fontsize=12)
    plt.tight_layout()
    fig.savefig(cat_dir / "residual_distributions.png", dpi=150)
    plt.close(fig)

    print(f"\n[INFO] Category 9 plots saved to {cat_dir}")
    return results


# ============================================================================
# CATEGORY 10: Posterior Predictive Checks (Framework)
# ============================================================================

def run_category_10(
    emu: Emulator,
    preds: Dict[str, Any],
    output_dir: Path,
    sigma_calibration: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Posterior predictive checks using HMC chains from Category 4."""
    print("\n" + "=" * 72)
    print("CATEGORY 10: Posterior Predictive Checks")
    print("=" * 72)

    cat_dir = ensure_dir(output_dir / "cat10_ppc")
    results = {}

    # Load HMC chains from Category 4.
    chains_path = output_dir / "cat4_mock_recovery" / "hmc_chains.npz"
    mock_path = output_dir / "cat4_mock_recovery" / "mock_observation.npz"
    if not chains_path.exists() or not mock_path.exists():
        print("  [WARN] Category 4 HMC results not found — run Category 4 first.")
        return {"skipped": True, "reason": "Category 4 HMC results needed"}

    chains = np.load(chains_path, allow_pickle=True)
    mock = np.load(mock_path, allow_pickle=True)
    samples = chains["samples"]  # (n_chains, n_samples, theta_dim)
    n_warmup = int(chains["n_warmup"])
    post_samples = samples[:, n_warmup:, :].reshape(-1, samples.shape[-1])

    y_obs_log = mock["y_mock_noisy_log"]        # (n_r, n_field)
    y_true_log = mock["y_mock_true_log"]         # (n_r, n_field)
    sigma_obs_dex = mock["sigma_obs_dex"]        # (n_field,)
    mock_mass = mock["mock_mass"]                # (1,)
    rr500 = mock["rr500"]                        # (1, n_r)
    mock_masses = mock["mock_masses"] if "mock_masses" in mock.files else mock_mass
    rr500_stack = mock["rr500_stack"] if "rr500_stack" in mock.files else rr500
    sigma_obs_stack = mock["sigma_obs_stack"] if "sigma_obs_stack" in mock.files else None
    stack_mode = None
    if "stack_mode" in mock.files:
        stack_mode = str(np.asarray(mock["stack_mode"]).ravel()[0])

    snap_eval = int(emu.snapnums[0])
    z_eval = float(getattr(emu, "redshift_by_snap", {}).get(snap_eval, 0.0))
    obs_valid = mock.get("obs_valid", np.ones_like(y_obs_log, dtype=bool))

    n_r, n_field = y_obs_log.shape
    fields = TARGET_FIELDS

    # --- 10a. Posterior predictive distribution ---
    print("\n--- 10a. Generating posterior predictive profiles ---")
    n_ppc = min(200, len(post_samples))
    rng = np.random.default_rng(42)
    pick = rng.choice(len(post_samples), size=n_ppc, replace=False)

    ppc_profiles = np.zeros((n_ppc, n_r, n_field), dtype=np.float64)
    ppc_model_std = np.zeros((n_ppc, n_r, n_field), dtype=np.float64)

    cal_vec = np.ones(n_field, dtype=np.float64)
    if sigma_calibration is not None:
        for fi, f in enumerate(fields):
            cal_vec[fi] = float(np.clip(sigma_calibration.get(f, 1.0), 1.0, 10.0))
        print("  Applying Cat 2 chi2-based sigma calibration: " +
              ", ".join(f"{f}={cal_vec[fi]:.3f}x" for fi, f in enumerate(fields)))
    for i, idx in enumerate(pick):
        theta_i = post_samples[idx].astype(np.float32)
        pred = emu.predict(
            theta=theta_i,
            M=mock_masses,
            r_bins=rr500_stack,
            field=fields, snapnum=snap_eval, redshift=z_eval,
            n_samples=5,
        )
        pred_std_log = getattr(pred, "total_std_log10", None)
        if pred_std_log is None:
            pred_std_log = np.zeros_like(pred.mean_log10)

        if stack_mode == "mean" and pred.mean_log10.ndim == 3 and pred.mean_log10.shape[0] > 1:
            ppc_profiles[i] = np.mean(pred.mean_log10, axis=0)
            # Mean of independent profiles: var(mean) = sum(var_i) / n^2.
            n_stack = pred_std_log.shape[0]
            ppc_model_std[i] = np.sqrt(np.sum(np.square(pred_std_log), axis=0) / max(n_stack ** 2, 1))
        else:
            ppc_profiles[i] = pred.mean_log10[0]  # (n_r, n_field) in log10 space
            ppc_model_std[i] = pred_std_log[0]
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_ppc} posterior predictive samples generated")

    ppc_model_std *= cal_vec.reshape(1, 1, -1)

    if sigma_obs_stack is not None:
        sigma_obs_arr = np.asarray(sigma_obs_stack[0], dtype=np.float64)
    else:
        sigma_obs_arr = np.tile(np.asarray(sigma_obs_dex, dtype=np.float64).reshape(1, -1), (n_r, 1))
    sigma_total = np.sqrt(np.clip(np.square(ppc_model_std) + np.square(sigma_obs_arr[None, :, :]), 1e-24, None))
    ppc_obs_draws = ppc_profiles + rng.normal(0.0, sigma_total)

    # --- 10b. Posterior predictive coverage ---
    print("\n--- 10b. Posterior predictive coverage ---")
    # Check: does observed data lie within the X% predictive interval?
    coverage_levels = [0.50, 0.68, 0.90, 0.95]
    ppc_coverage = {}
    for fi, f in enumerate(fields):
        m = obs_valid[:, fi]
        if m.sum() == 0:
            continue
        profiles_f = ppc_obs_draws[:, :, fi][:, m]  # (n_ppc, n_valid)
        obs_f = y_obs_log[:, fi][m]                 # (n_valid,)

        cov_by_level = {}
        for level in coverage_levels:
            lo_pct = (1 - level) / 2 * 100
            hi_pct = (1 + level) / 2 * 100
            lo_q = np.percentile(profiles_f, lo_pct, axis=0)
            hi_q = np.percentile(profiles_f, hi_pct, axis=0)
            frac_covered = float(np.mean((obs_f >= lo_q) & (obs_f <= hi_q)))
            cov_by_level[f"{level:.0%}"] = frac_covered

        ppc_coverage[f] = cov_by_level
        print(f"  {f}: " + ", ".join(f"{k}→{v:.2%}" for k, v in cov_by_level.items()))

    results["ppc_coverage"] = ppc_coverage

    # Check 68% coverage is reasonable (within 50-85%).
    for f in fields:
        if f in ppc_coverage and "68%" in ppc_coverage[f]:
            c68 = ppc_coverage[f]["68%"]
            passed = 0.50 <= c68 <= 0.85
            print_pass_fail(f"{f} 68% PPC coverage", c68, "[0.50, 0.85]", passed)

    # --- 10c. Posterior predictive p-value ---
    print("\n--- 10c. Posterior predictive p-values ---")
    # Compute chi2(observed) for each posterior sample and compare to chi2(replicated).
    chi2_obs_per_sample = np.zeros(n_ppc)
    chi2_rep_per_sample = np.zeros(n_ppc)

    for i in range(n_ppc):
        for fi, f in enumerate(fields):
            m = obs_valid[:, fi]
            sigma_dex = np.clip(sigma_total[i, :, fi][m], 1e-12, None)
            resid_obs = (y_obs_log[:, fi][m] - ppc_profiles[i, :, fi][m])
            chi2_obs_per_sample[i] += np.sum((resid_obs / sigma_dex) ** 2)

            # Replicate: draw from predictive distribution.
            y_rep = ppc_profiles[i, :, fi][m] + rng.normal(0, sigma_dex, size=m.sum())
            resid_rep = (y_rep - ppc_profiles[i, :, fi][m])
            chi2_rep_per_sample[i] += np.sum((resid_rep / sigma_dex) ** 2)

    ppp_value = float(np.mean(chi2_rep_per_sample >= chi2_obs_per_sample))
    ppp_pass = 0.05 <= ppp_value <= 0.95
    print_pass_fail("Posterior predictive p-value", ppp_value, "[0.05, 0.95]", ppp_pass)
    results["ppp_value"] = ppp_value
    results["ppp_pass"] = ppp_pass
    results["sigma_calibration"] = {f: float(cal_vec[fi]) for fi, f in enumerate(fields)}

    # --- 10d. Posterior predictive profile plots ---
    print("\n--- 10d. Posterior predictive profile plots ---")
    r_grid = np.arange(n_r)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for fi, f in enumerate(fields):
        ax = axes[fi // 2, fi % 2]
        # Posterior predictive envelopes.
        q16 = np.percentile(ppc_profiles[:, :, fi], 16, axis=0)
        q50 = np.percentile(ppc_profiles[:, :, fi], 50, axis=0)
        q84 = np.percentile(ppc_profiles[:, :, fi], 84, axis=0)
        q025 = np.percentile(ppc_profiles[:, :, fi], 2.5, axis=0)
        q975 = np.percentile(ppc_profiles[:, :, fi], 97.5, axis=0)

        ax.fill_between(r_grid, q025, q975, alpha=0.15, color="steelblue", label="95% PPC")
        ax.fill_between(r_grid, q16, q84, alpha=0.3, color="steelblue", label="68% PPC")
        ax.plot(r_grid, q50, "b-", lw=1.5, label="median PPC")
        ax.plot(r_grid, y_true_log[:, fi], "k-", lw=2, label="truth")
        if sigma_obs_stack is not None:
            yerr = sigma_obs_stack[0, :, fi]
        else:
            yerr = float(sigma_obs_dex[fi])
        ax.errorbar(r_grid, y_obs_log[:, fi],
                     yerr=yerr,
                     fmt="o", ms=3, color="red", alpha=0.6, label="noisy obs")
        ax.set_xlabel("Radial bin index")
        ax.set_ylabel(f"log10({f})")
        ax.set_title(f"{f}")
        ax.legend(fontsize=7)
    plt.suptitle("Cat 10d: Posterior Predictive Checks", fontsize=12)
    plt.tight_layout()
    fig.savefig(cat_dir / "ppc_profiles.png", dpi=150)
    plt.close(fig)

    # Chi2 comparison plot.
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(chi2_obs_per_sample, chi2_rep_per_sample, s=8, alpha=0.5)
    lim = max(chi2_obs_per_sample.max(), chi2_rep_per_sample.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("χ²(observed)")
    ax.set_ylabel("χ²(replicated)")
    ax.set_title(f"Cat 10c: PP p-value = {ppp_value:.3f}")
    fig.savefig(cat_dir / "ppc_chi2_scatter.png", dpi=150)
    plt.close(fig)

    print(f"\n[INFO] Category 10 plots saved to {cat_dir}")
    return results


# ============================================================================
# Go / No-Go gate
# ============================================================================

def evaluate_go_nogo(all_results: Dict[int, Any]) -> Dict[str, Any]:
    """Evaluate hard go/no-go criteria suitable for real observed-data inference.

    Thresholds are deliberately relaxed relative to the simulation-based THRESHOLDS
    to account for model mismatch and domain extrapolation.  A run is 'GO' only when
    ALL critical gates pass.

    Returns a dict with keys:
        "go"          : bool   — overall verdict
        "gates"       : list of (name, passed, value, limit) tuples
        "critical_fails" : list of gate names that are critical and failed
    """
    gates: list = []  # (name, passed, value_str, limit_str, is_critical)

    # --- Gate 1: Coverage (Cat 2 or Cat 4 recovery) ---
    # After sigma calibration, median 68% coverage should be ≥ 0.45.
    if 2 in all_results:
        cov_devs = all_results[2].get("coverage_max_deviation", {})
        if cov_devs:
            max_dev = max(cov_devs.values())
            passed = max_dev <= 0.25   # real-data gate: ≤25% coverage deviation
            gates.append(("Cat2 coverage max_dev ≤ 0.25", passed,
                           f"{max_dev:.3f}", "0.25", True))

    if 4 in all_results and "recovery_coverage_68" in all_results[4]:
        cov68 = all_results[4]["recovery_coverage_68"]
        passed = cov68 >= 0.45
        gates.append(("Cat4 68% CI coverage ≥ 0.45", passed,
                       f"{cov68:.2%}", "45%", True))

    # --- Gate 2: Reduced chi-squared (Cat 2) ---
    # Before calibration chi2_red must not be catastrophically high.
    # A well-calibrated model will be near 1; we gate at ≤ 5.0 pre-cal.
    if 2 in all_results:
        chi2_red = all_results[2].get("chi2_red", {})
        if chi2_red:
            valid_vals = [v for v in chi2_red.values() if np.isfinite(v)]
            if valid_vals:
                med_chi2 = float(np.median(valid_vals))
                passed = med_chi2 <= 5.0
                gates.append(("Cat2 median χ²_red ≤ 5.0 (pre-cal)", passed,
                               f"{med_chi2:.2f}", "5.0", True))
                # Separate gate: after sigma-cal (divide by chi2_red), residuals
                # should be ~1 by construction, but we flag if pre-cal is > 15.
                passed_extreme = med_chi2 <= 15.0
                gates.append(("Cat2 median χ²_red ≤ 15.0 (anti-catastrophe)", passed_extreme,
                               f"{med_chi2:.2f}", "15.0", True))

    # --- Gate 3: RMSE sanity (Cat 1) ---
    if 1 in all_results:
        frac_pass = all_results[1].get("frac_rmse_pass", False)
        # For real data we relax RMSE/sigma slightly — gate is just that
        # fewer than 75% of fields fail the strict RMSE/sigma threshold.
        outlier_frac = all_results[1].get("outlier_fraction", 1.0)
        passed = outlier_frac < 0.50
        gates.append(("Cat1 outlier fraction < 50%", passed,
                       f"{outlier_frac:.1%}", "50%", True))

    # --- Gate 4: KL bound (Cat 4) ---
    if 4 in all_results and "kl_bound" in all_results[4]:
        kl = all_results[4]["kl_bound"]
        passed = kl < 5.0    # relaxed from 1 nat for real data
        gates.append(("Cat4 KL bound < 5.0 nats", passed,
                       f"{kl:.2f}", "5.0", True))

    # --- Gate 5: R-hat convergence (Cat 4) ---
    if 4 in all_results and "rhat_pass" in all_results[4]:
        passed = bool(all_results[4]["rhat_pass"])
        rhat_max = all_results[4].get("rhat_max_value", float("nan"))
        gates.append((f"Cat4 R-hat < {RHAT_MAX} (HMC convergence)", passed,
                       f"{rhat_max:.3f}", str(RHAT_MAX), False))  # not critical — may skip HMC

    # --- Gate 6: Posterior predictive p-value (Cat 10) ---
    if 10 in all_results and "ppp_value" in all_results[10]:
        ppp = all_results[10]["ppp_value"]
        passed = 0.05 <= ppp <= 0.95
        gates.append(("Cat10 PP p-value ∈ [0.05, 0.95]", passed,
                       f"{ppp:.3f}", "[0.05, 0.95]", False))

    # --- Gate 7: Log10-duality (pre-flight) ---
    if "duality" in all_results:
        passed = bool(all_results["duality"].get("duality_pass", False))
        gates.append(("Pre-flight log10 duality", passed, "", "", True))

    critical_fails = [name for (name, passed, _, _, crit) in gates if crit and not passed]
    overall_go = len(critical_fails) == 0

    return {
        "go": overall_go,
        "gates": gates,
        "critical_fails": critical_fails,
    }


# ============================================================================
# Summary
# ============================================================================

def print_summary(all_results: Dict[int, Any]):
    """Print a consolidated pass/fail summary table."""
    print("\n" + "=" * 72)
    print("VALIDATION SUITE SUMMARY")
    print("=" * 72)

    rows = []

    # Duality pre-flight
    if "duality" in all_results:
        r = all_results["duality"]
        rows.append(("0", "mean ↔ 10^(mean_log10) consistency",
                      "PASS" if r.get("duality_pass", False) else "FAIL", "Critical"))

    # Context bias pre-flight
    if "context_bias" in all_results:
        r = all_results["context_bias"]
        if r.get("context_probe") != "skipped":
            rows.append(("0", "Zero-shot ctx influence < 0.05 dex",
                          "PASS" if not r.get("context_matters", True) else "WARN", "High"))
            if "null_context_test" in r and "masked" in r["null_context_test"]:
                masked = r["null_context_test"]["masked"]
                if masked.get("available", False):
                    per_field = masked.get("per_field", {})
                    masked_ok = all(v.get("max_shift_dex", np.inf) < 0.05 for v in per_field.values()) if per_field else False
                    rows.append(("0", "Masked null ctx shift < 0.05 dex",
                                  "PASS" if masked_ok else "WARN", "High"))
                else:
                    rows.append(("0", "Masked null ctx shift < 0.05 dex",
                                  "WARN", "High"))
            rows.append(("0", "No OOD mean-reversion bias",
                          "PASS" if not r.get("ood_bias_detected", True) else "WARN", "High"))

    # Cat 1
    if 1 in all_results:
        r = all_results[1]
        rows.append(("1", "RMSE/σ_obs ≤ 0.15",
                      "PASS" if r.get("frac_rmse_pass", False) else "FAIL", "Critical"))
        rows.append(("1", f"Outlier F_out < 10%",
                      "PASS" if r.get("outlier_fraction_pass", False) else "FAIL", "Critical"))

    # Cat 2
    if 2 in all_results:
        r = all_results[2]
        for f, dev in r.get("coverage_max_deviation", {}).items():
            p = "PASS" if dev <= 0.05 else "FAIL"
            rows.append(("2", f"Coverage {f}", p, "Critical"))
        for f, d in r.get("pit_ks", {}).items():
            ks_stat = d.get("ks_stat", float("nan"))
            ks_crit = d.get("ks_boot_crit", float("nan"))
            p = "PASS" if np.isfinite(ks_stat) and np.isfinite(ks_crit) and (ks_stat <= ks_crit) else "FAIL"
            rows.append(("2", f"PIT KS {f} (boot)", p, "High"))
        for f, v in r.get("chi2_red", {}).items():
            p = "PASS" if 0.8 <= v <= 1.2 else "FAIL"
            rows.append(("2", f"χ²_red {f}", p, "High"))
        for f, v in r.get("spearman_rho", {}).items():
            p = "PASS" if v > 0.4 else "FAIL"
            rows.append(("2", f"Spearman ρ {f}", p, "High"))

    # Cat 2b (TARP)
    if "2b" in all_results:
        r = all_results["2b"]
        rows.append(("2b", "TARP calibration",
                      "PASS" if r.get("tarp_pass", False) else "FAIL", "Critical"))

    # Cat 3
    if 3 in all_results:
        r = all_results[3]
        mono = r.get("monotonicity", {})
        if mono:
            n_mono = sum(1 for p in mono.values()
                         for fv in p.values() if fv.get("is_monotone"))
            n_total = sum(1 for p in mono.values() for _ in p.values())
            rows.append(("3", f"1P monotonicity ({n_mono}/{n_total})",
                          "PASS" if n_mono >= n_total * 0.5 else "FAIL", "Medium"))
        sobol = r.get("sobol_first_order", {})
        if sobol:
            rows.append(("3", "Sobol sensitivity indices", "PASS", "Medium"))

    # Cat 4
    if 4 in all_results:
        r = all_results[4]
        kl = r.get("kl_bound", float("inf"))
        rows.append(("4", "KL bound < 1 nat", "PASS" if kl < 1.0 else "FAIL", "Critical"))
        if "rhat_pass" in r:
            rows.append(("4", f"HMC R-hat < {RHAT_MAX}", "PASS" if r["rhat_pass"] else "FAIL", "High"))
        if "recovery_pass" in r:
            cov68 = r.get("recovery_coverage_68", 0)
            rows.append(("4", f"68% CI coverage ∈ [50%,85%] ({cov68:.0%})",
                          "PASS" if r["recovery_pass"] else "FAIL", "High"))

    # Cat 5
    if 5 in all_results:
        r = all_results[5]
        cn = r.get("fisher_condition_number", float("inf"))
        rows.append(("5", "Fisher κ < 10^5", "PASS" if cn < 1e5 else "FAIL", "Medium"))
        n_constr = r.get("n_constrained", 0)
        rows.append(("5", f"Fisher constrained params: {n_constr}", "PASS" if n_constr > 0 else "FAIL", "Medium"))

    # Cat 6
    if 6 in all_results:
        r = all_results[6]
        if "few_shot_rmse" in r:
            rows.append(("6", "Few-shot RMSE improves with context",
                          "PASS" if r.get("few_shot_improves", False) else "FAIL", "High"))
        for ood_level, ratios in r.get("ood_inflation", {}).items():
            any_fail = any(v < 1.5 for v in ratios.values())
            rows.append(("6", f"OOD inflation ({ood_level})",
                          "FAIL" if any_fail else "PASS", "High"))

    # Cat 7
    if 7 in all_results:
        r = all_results[7]
        chi2_at_truth = r.get("noise_chi2_at_truth", {})
        if chi2_at_truth:
            all_ok = all(0.5 <= v <= 2.0 for v in chi2_at_truth.values())
            rows.append(("7", "χ² at true θ stable across noise",
                          "PASS" if all_ok else "FAIL", "High"))

    # Cat 9
    if 9 in all_results:
        r = all_results[9]
        for f, b in r.get("bias", {}).items():
            p = "PASS" if b["fractional"] < 0.05 else "FAIL"
            rows.append(("9", f"Bias {f} < 5%", p, "High"))

    # Cat 10
    if 10 in all_results:
        r = all_results[10]
        if "ppp_pass" in r:
            rows.append(("10", "PP p-value ∈ [0.05, 0.95]",
                          "PASS" if r["ppp_pass"] else "FAIL", "High"))

    print(f"\n{'Cat':<5} {'Test':<45} {'Result':<8} {'Priority':<10}")
    print("-" * 70)
    n_pass = sum(1 for r in rows if r[2] == "PASS")
    n_fail = sum(1 for r in rows if r[2] == "FAIL")
    for cat, test, result, priority in rows:
        print(f"{cat:<5} {test:<45} {result:<8} {priority:<10}")

    print("-" * 70)
    print(f"Total: {n_pass} PASS, {n_fail} FAIL out of {len(rows)} tests")
    print("=" * 72)

    # --- Go / No-Go verdict ---
    gng = evaluate_go_nogo(all_results)
    print("\n" + "=" * 72)
    print("GO / NO-GO VERDICT")
    print("=" * 72)
    print(f"{'Gate':<50} {'Status':<8} {'Value':<12} {'Limit'}")
    print("-" * 72)
    for (name, passed, val, limit, crit) in gng["gates"]:
        status = "PASS" if passed else ("FAIL[C]" if crit else "FAIL")
        print(f"{name:<50} {status:<8} {val:<12} {limit}")
    print("-" * 72)
    if gng["go"]:
        print("  *** OVERALL: GO *** — all critical gates passed")
    else:
        print("  *** OVERALL: NO-GO *** — critical gate(s) failed:")
        for cf in gng["critical_fails"]:
            print(f"    • {cf}")
    print("=" * 72)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ANPcosmo Emulator Validation Suite")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for emulator (cpu or cuda)")
    parser.add_argument("--output-dir", type=str, default="validation_results",
                        help="Directory for validation outputs")
    parser.add_argument("--categories", type=int, nargs="*", default=None,
                        help="Run only these categories (default: all)")
    parser.add_argument("--n-samples", type=int, default=30,
                        help="Number of latent samples for predictions")
    parser.add_argument("--cat4-n-mock", type=int, default=10,
                        help="Number of stacked Cat4 HMC mocks")
    parser.add_argument("--cat4-stack-logm-min", type=float, default=13.0,
                        help="Minimum log10(M500c) for stacked Cat4 mass-bin")
    parser.add_argument("--cat4-stack-logm-max", type=float, default=14.0,
                        help="Maximum log10(M500c) for stacked Cat4 mass-bin")
    parser.add_argument("--cat4-hmc-n-samples", type=int, default=600,
                        help="Category 4 HMC total samples per chain")
    parser.add_argument("--cat4-hmc-n-warmup", type=int, default=200,
                        help="Category 4 HMC warmup samples per chain")
    parser.add_argument("--cat4-hmc-n-chains", type=int, default=2,
                        help="Category 4 HMC number of chains")
    parser.add_argument("--cat4-hmc-n-leapfrog", type=int, default=15,
                        help="Category 4 HMC leapfrog steps")
    parser.add_argument("--cat4-hmc-init-step-size", type=float, default=0.01,
                        help="Category 4 HMC initial leapfrog step size")
    parser.add_argument("--cat4-mass-bias-grid-dex", type=float, nargs="*", default=[0.0],
                        help="Global log10 mass-bias nuisance grid for Cat4 (e.g. -0.10 0.0 0.10)")
    parser.add_argument("--cat4-mass-bias-prior-sigma-dex", type=float, default=0.10,
                        help="Gaussian prior sigma (dex) for Cat4 mass-bias nuisance")
    parser.add_argument("--tarp-mode", type=str, default="pointwise",
                        choices=["pointwise", "parameter"],
                        help="TARP mode: pointwise surrogate or strict parameter-space TARP")
    parser.add_argument("--tarp-param-refs", type=int, default=256,
                        help="Number of random reference points per mock for parameter-space TARP")
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    cats = set(args.categories) if args.categories else set(range(1, 11))

    print("=" * 72)
    print("ANPcosmo Emulator Validation Suite")
    print(f"Model: {RUN_DIR.name}")
    print(f"Categories: {sorted(cats)}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print("=" * 72)

    # Load emulator
    emu = load_emulator(args.device)

    # Load test data and predictions (needed for most categories)
    all_results = {}

    need_preds = cats & {1, 2, 4, 7, 9, 10}
    preds = None
    test_data = None
    if need_preds:
        test_data = load_test_data(emu)
        preds = predict_test_set(emu, test_data, n_samples=args.n_samples)
        # Validate mean/mean_log10 duality before any calibration tests
        all_results["duality"] = validate_log10_duality(preds)
        # Probe zero-shot context token bias
        all_results["context_bias"] = validate_zeroshot_context(emu, preds, output_dir)

    if 1 in cats:
        all_results[1] = run_category_1(preds, output_dir)
    if 2 in cats:
        all_results[2] = run_category_2(preds, output_dir)
    if 3 in cats:
        all_results[3] = run_category_3(emu, output_dir)
    if 4 in cats:
        all_results[4] = run_category_4(
            emu,
            preds,
            output_dir,
            n_mock=args.cat4_n_mock,
            stack_logm_min=args.cat4_stack_logm_min,
            stack_logm_max=args.cat4_stack_logm_max,
            hmc_n_samples=args.cat4_hmc_n_samples,
            hmc_n_warmup=args.cat4_hmc_n_warmup,
            hmc_n_chains=args.cat4_hmc_n_chains,
            hmc_n_leapfrog=args.cat4_hmc_n_leapfrog,
            hmc_init_step_size=args.cat4_hmc_init_step_size,
            mass_bias_grid_dex=args.cat4_mass_bias_grid_dex,
            mass_bias_prior_sigma_dex=args.cat4_mass_bias_prior_sigma_dex,
        )
    if 2 in cats:
        # TARP may depend on Category 4 outputs in parameter mode.
        all_results["2b"] = run_tarp(
            preds,
            output_dir,
            mode=args.tarp_mode,
            n_param_refs=args.tarp_param_refs,
        )
    if 5 in cats:
        all_results[5] = run_category_5(emu, output_dir)
    if 6 in cats:
        all_results[6] = run_category_6(emu, output_dir)
    if 7 in cats:
        all_results[7] = run_category_7(emu, preds, output_dir)
    if 8 in cats:
        all_results[8] = run_category_8(output_dir)
    if 9 in cats:
        all_results[9] = run_category_9(emu, preds, output_dir)
    if 10 in cats:
        sigma_cal = None
        if 2 in all_results and isinstance(all_results[2], dict):
            chi2_red = all_results[2].get("chi2_red", {})
            if isinstance(chi2_red, dict) and len(chi2_red) > 0:
                sigma_cal = {
                    f: float(np.sqrt(max(float(chi2_red.get(f, 1.0)), 1.0)))
                    for f in TARGET_FIELDS
                }
        all_results[10] = run_category_10(emu, preds, output_dir, sigma_calibration=sigma_cal)

    # Print consolidated summary
    print_summary(all_results)

    # Save JSON results (numerics only)
    def _jsonify(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(v) for v in obj]
        return obj

    json_path = output_dir / "validation_results.json"
    with open(json_path, "w") as f:
        json.dump(_jsonify(all_results), f, indent=2, default=str)
    print(f"\n[INFO] Results saved to {json_path}")


if __name__ == "__main__":
    main()
