from __future__ import annotations

from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from .api import PredictionResult


def _as_3d(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 2:
        return x[..., None]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected 2D or 3D array, got shape {x.shape}")


def rmse_by_field(y_true: np.ndarray, y_pred: np.ndarray, field_names: Sequence[str]) -> Dict[str, float]:
    yt = _as_3d(y_true)
    yp = _as_3d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch y_true={yt.shape}, y_pred={yp.shape}")
    if yt.shape[-1] != len(field_names):
        raise ValueError("field_names length must match final array dimension")

    out: Dict[str, float] = {}
    for i, name in enumerate(field_names):
        out[str(name)] = float(np.sqrt(np.mean((yp[..., i] - yt[..., i]) ** 2)))
    return out


def residual_radius_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    r_bins: np.ndarray,
    field_names: Sequence[str],
    quantiles: Sequence[float] = (0.16, 0.5, 0.84),
) -> Dict[str, Dict[str, np.ndarray]]:
    yt = _as_3d(y_true)
    yp = _as_3d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch y_true={yt.shape}, y_pred={yp.shape}")

    r_bins = np.asarray(r_bins)
    if r_bins.ndim != 1:
        raise ValueError("r_bins must be a 1D array")
    if yt.shape[1] != r_bins.shape[0]:
        raise ValueError(f"Second dimension of y arrays ({yt.shape[1]}) must match len(r_bins) ({r_bins.shape[0]})")

    resid = yp - yt
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for i, name in enumerate(field_names):
        q = np.quantile(resid[..., i], q=np.asarray(quantiles), axis=0)
        out[str(name)] = {
            "quantiles": np.asarray(quantiles, dtype=np.float64),
            "residual_quantiles_by_radius": q,
        }
    return out


def coverage_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    field_names: Sequence[str],
    p_grid: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    yt = _as_3d(y_true)
    yp = _as_3d(y_pred)
    ys = _as_3d(y_std)

    if yt.shape != yp.shape or yt.shape != ys.shape:
        raise ValueError(f"Shape mismatch: y_true={yt.shape}, y_pred={yp.shape}, y_std={ys.shape}")

    if p_grid is None:
        p_grid = np.linspace(0.05, 0.95, 19)

    p_grid = np.asarray(p_grid, dtype=np.float64)
    if np.any((p_grid <= 0.0) | (p_grid >= 1.0)):
        raise ValueError("p_grid values must be in (0, 1)")

    abs_z = np.abs((yt - yp) / np.clip(ys, 1e-12, None))
    z_grid = norm.ppf((1.0 + p_grid) / 2.0)

    empirical = np.zeros((p_grid.shape[0], yt.shape[-1]), dtype=np.float64)
    for i, z in enumerate(z_grid):
        empirical[i] = (abs_z <= z).mean(axis=(0, 1))

    return {
        "p_nominal": p_grid,
        "p_empirical": empirical,
        "field_names": np.asarray([str(x) for x in field_names]),
    }


def pit_values(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    yt = _as_3d(y_true)
    yp = _as_3d(y_pred)
    ys = _as_3d(y_std)
    if yt.shape != yp.shape or yt.shape != ys.shape:
        raise ValueError(f"Shape mismatch: y_true={yt.shape}, y_pred={yp.shape}, y_std={ys.shape}")

    z = (yt - yp) / np.clip(ys, 1e-12, None)
    pit = 0.5 * (1.0 + np.vectorize(np.math.erf)(z / np.sqrt(2.0)))
    return np.clip(pit, 0.0, 1.0)


def _spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y lengths must match")

    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))

    xr = xr.astype(np.float64)
    yr = yr.astype(np.float64)
    xr = (xr - xr.mean()) / (xr.std() + 1e-12)
    yr = (yr - yr.mean()) / (yr.std() + 1e-12)
    return float(np.mean(xr * yr))


def uncertainty_error_rank_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    field_names: Sequence[str],
) -> Dict[str, float]:
    yt = _as_3d(y_true)
    yp = _as_3d(y_pred)
    ys = _as_3d(y_std)

    out: Dict[str, float] = {}
    for i, name in enumerate(field_names):
        abs_err = np.abs(yp[..., i] - yt[..., i]).reshape(-1)
        sig = ys[..., i].reshape(-1)
        out[str(name)] = _spearman_rank_corr(sig, abs_err)
    return out


def build_diagnostics_report(
    y_true: np.ndarray,
    prediction: PredictionResult,
    r_bins: np.ndarray,
) -> Dict[str, object]:
    rmse = rmse_by_field(y_true, prediction.mean, prediction.field_names)
    resid = residual_radius_summary(y_true, prediction.mean, r_bins, prediction.field_names)
    coverage = coverage_curve(y_true, prediction.mean, prediction.total_std, prediction.field_names)
    corr = uncertainty_error_rank_correlation(y_true, prediction.mean, prediction.total_std, prediction.field_names)

    return {
        "rmse_by_field": rmse,
        "residual_radius_summary": resid,
        "coverage": coverage,
        "uncertainty_error_rank_corr": corr,
    }


def plot_coverage_curve(report: Dict[str, object], field_name: str | None = None, ax=None):
    cov = report["coverage"]
    p_nominal = np.asarray(cov["p_nominal"])
    p_empirical = np.asarray(cov["p_empirical"])
    names = [str(x) for x in np.asarray(cov["field_names"]).tolist()]

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.2))

    ax.plot(p_nominal, p_nominal, "k--", lw=1.2, label="ideal")

    if field_name is None:
        for i, name in enumerate(names):
            ax.plot(p_nominal, p_empirical[:, i], lw=1.1, alpha=0.9, label=name)
    else:
        if field_name not in names:
            raise ValueError(f"Unknown field {field_name}; valid fields: {names}")
        idx = names.index(field_name)
        ax.plot(p_nominal, p_empirical[:, idx], lw=2.0, label=field_name)

    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage calibration")
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_residual_vs_radius(report: Dict[str, object], r_bins: np.ndarray, field_name: str, ax=None):
    resid = report["residual_radius_summary"]
    if field_name not in resid:
        raise ValueError(f"Unknown field {field_name}; valid fields: {list(resid.keys())}")

    summary = resid[field_name]
    quantiles = np.asarray(summary["quantiles"])
    rq = np.asarray(summary["residual_quantiles_by_radius"])

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.2))

    q_mid = int(np.argmin(np.abs(quantiles - 0.5)))
    q_lo = int(np.argmin(np.abs(quantiles - 0.16)))
    q_hi = int(np.argmin(np.abs(quantiles - 0.84)))

    ax.plot(r_bins, rq[q_mid], lw=2.0, label="median residual")
    ax.fill_between(r_bins, rq[q_lo], rq[q_hi], alpha=0.25, label="16-84%")
    ax.axhline(0.0, color="k", ls="--", lw=1.0)
    ax.set_xlabel("r / r500")
    ax.set_ylabel("prediction - truth")
    ax.set_title(f"Residual vs radius: {field_name}")
    ax.legend(loc="best", fontsize=8)
    return ax
