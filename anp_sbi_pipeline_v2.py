"""
Multi-redshift, multi-channel SBI pipeline for ANP emulator inference.

Upgrades over v1 (anp_sbi_pipeline.ipynb):
  - Supports multi-z emulator predictions (z=0, 0.5, 1.0)
  - Mass x redshift binning in summary statistics
  - Uses anp_emulator.observations for standardized data loading
  - Supports expanded observational datasets beyond ACCEPT+X-COP
  - Configurable observable channels: kT, ne, P, Z, K (entropy),
    y (Compton-y proxy), plus cross-channel features
  - Survey selection function modeling
  - Simulation-Based Calibration (SBC) validation
  - Multi-dataset ablation analysis

Usage from a notebook or script:

    from anp_sbi_pipeline_v2 import SBIPipeline, SBIConfig
    cfg = SBIConfig(
        emu_checkpoint="path/to/best_model.pt",
        summary_channels=("kT", "ne", "P", "Z", "K"),
        cross_features=("T_core_ratio", "K_slope"),
    )
    pipe = SBIPipeline(cfg)
    pipe.load_observations("/path/to/obs_expanded.npz")
    pipe.run(n_sims=200_000, n_posterior=50_000)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MU_E = 1.176
M_PROTON = 1.6726e-24
LOG10_MU_E_MP = np.log10(MU_E * M_PROTON)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SBIConfig:
    """Configuration for the SBI pipeline."""

    # Emulator
    emu_checkpoint: str = ""
    device: str = "cuda"

    # Observations
    obs_npz: str = ""
    logM_min: float = 13.2
    logM_max: float = 15.0

    # Mass binning
    mass_bin_edges: Tuple[float, ...] = (13.2, 13.6, 14.0, 14.5, 15.0)

    # Redshift binning for multi-z (single bin for z=0-only emulator)
    # Use (0.0, 1.0) for single-z to include all clusters like v1.
    z_bin_edges: Tuple[float, ...] = (0.0, 1.0)

    # Channels to use in forward model (passed to emulator.predict)
    emu_fields: Tuple[str, ...] = ("temperature", "gas_density")

    # Observable channels in summary statistics.
    # Each can be:
    #   'kT'  - temperature (from emulator 'temperature')
    #   'ne'  - electron density (from emulator 'gas_density')
    #   'P'   - pressure = kT + ne in log10 space (derived)
    #   'Z'   - metallicity (from emulator 'metallicity'; needs emu_fields)
    #   'K'   - entropy K = kT - (2/3)*ne in log10 space (derived)
    #   'y'   - Compton-y proxy = ne + kT in log10 space (same as P, for tSZ)
    summary_channels: Tuple[str, ...] = ("kT", "ne", "P")

    # Cross-channel features added as extra scalars per bin:
    #   'T_core_ratio' - T(r_min) / T(r_max) ratio (cool-core indicator)
    #   'K_slope'      - entropy slope d(log K)/d(log r)
    #   'ne_slope'     - density slope d(log ne)/d(log r)
    cross_features: Tuple[str, ...] = ()

    # Summary statistics
    n_radii_per_bin: int = 5
    max_r_by_mass: Dict[int, float] = field(default_factory=lambda: {
        0: 60.0, 1: 100.0, 2: 300.0, 3: 600.0
    })

    # Selection function: per-survey detection probability p(observed | M, z)
    # Maps survey name -> callable(logM, z) -> probability
    # If empty, no selection function is applied.
    selection_functions: Dict[str, object] = field(default_factory=dict)

    # Simulation settings
    n_sims: int = 200_000
    n_anp_samples: int = 10
    max_gpu_rows: int = 16384

    # NPE training
    max_epochs: int = 200
    batch_size_npe: int = 256
    patience_npe: int = 30

    # Posterior
    n_posterior: int = 50_000
    sample_with: str = "direct"  # 'direct', 'mcmc', or 'vi'
    mcmc_method: str = "slice_np_vectorized"

    # Prior
    fix_cosmo: bool = False

    # Reproducibility
    seed: int = 12345

    @property
    def n_mass_bins(self) -> int:
        return len(self.mass_bin_edges) - 1

    @property
    def n_z_bins(self) -> int:
        return len(self.z_bin_edges) - 1

    @property
    def is_multiz(self) -> bool:
        return self.n_z_bins > 1


# ---------------------------------------------------------------------------
# R500 estimation from M500 (CAMELS calibrated)
# ---------------------------------------------------------------------------

def fit_R500_relation(logM500: np.ndarray, R500: np.ndarray) -> np.poly1d:
    """Fit the log10(R500) = a*log10(M500) + b relation from training data."""
    valid = (R500 > 0) & np.isfinite(logM500)
    coeffs = np.polyfit(logM500[valid], np.log10(R500[valid]), 1)
    return np.poly1d(coeffs)


# ---------------------------------------------------------------------------
# Channel derivation helpers
# ---------------------------------------------------------------------------

# Map from summary channel name -> how to compute from emulator log10 profiles
# 'direct' channels map to an emulator field index; 'derived' are computed.
CHANNEL_SOURCES = {
    "kT": "direct:temperature",
    "ne": "direct:gas_density",   # needs mu_e*mp correction
    "P":  "derived:kT+ne",       # log10(P) = log10(kT) + log10(ne)
    "Z":  "direct:metallicity",
    "K":  "derived:kT-2/3*ne",   # log10(K) = log10(kT) - (2/3)*log10(ne)
    "y":  "derived:kT+ne",       # Compton-y proxy (same as P in log space)
}


def _compute_channel_profiles(
    result_i,
    emu_fields: Sequence[str],
    summary_channels: Sequence[str],
    bs: int,
    n_halos: int,
    n_radial: int,
    rng: np.random.RandomState,
) -> Dict[str, np.ndarray]:
    """Extract and derive channel profiles from emulator output.

    Returns dict mapping channel name -> (bs, n_halos, n_radial) log10 profiles.
    """
    ch_profiles = {}

    # Extract direct channels first
    field_list = list(emu_fields)
    direct_cache = {}  # cache for log10 profiles of direct fields

    for ch in summary_channels:
        source = CHANNEL_SOURCES.get(ch)
        if source is None:
            raise ValueError(f"Unknown summary channel '{ch}'")

        if source.startswith("direct:"):
            field_name = source.split(":")[1]
            if field_name not in field_list:
                raise ValueError(
                    f"Channel '{ch}' requires emulator field '{field_name}' "
                    f"but emu_fields={emu_fields}"
                )
            if field_name not in direct_cache:
                fi = field_list.index(field_name)
                log_prof = result_i.mean_log10[:, :, fi].reshape(bs, n_halos, n_radial)
                std_prof = result_i.std_log10[:, :, fi].reshape(bs, n_halos, n_radial)
                noise = rng.randn(bs, n_halos, n_radial) * std_prof
                direct_cache[field_name] = log_prof + noise
            raw = direct_cache[field_name]

            if ch == "ne":
                # gas_density -> electron number density
                ch_profiles[ch] = raw - LOG10_MU_E_MP
            else:
                ch_profiles[ch] = raw

    # Ensure kT and ne are available for derived channels
    need_kT_ne = any(CHANNEL_SOURCES.get(ch, "").startswith("derived:")
                     for ch in summary_channels)
    if need_kT_ne:
        if "temperature" not in direct_cache:
            if "temperature" in field_list:
                fi = field_list.index("temperature")
                log_prof = result_i.mean_log10[:, :, fi].reshape(bs, n_halos, n_radial)
                std_prof = result_i.std_log10[:, :, fi].reshape(bs, n_halos, n_radial)
                direct_cache["temperature"] = log_prof + rng.randn(bs, n_halos, n_radial) * std_prof
        if "gas_density" not in direct_cache:
            if "gas_density" in field_list:
                fi = field_list.index("gas_density")
                log_prof = result_i.mean_log10[:, :, fi].reshape(bs, n_halos, n_radial)
                std_prof = result_i.std_log10[:, :, fi].reshape(bs, n_halos, n_radial)
                direct_cache["gas_density"] = log_prof + rng.randn(bs, n_halos, n_radial) * std_prof

    # Compute derived channels
    for ch in summary_channels:
        if ch in ch_profiles:
            continue
        source = CHANNEL_SOURCES[ch]
        if source == "derived:kT+ne":
            log_kT = direct_cache.get("temperature")
            log_ne_raw = direct_cache.get("gas_density")
            if log_kT is None or log_ne_raw is None:
                raise ValueError(f"Derived channel '{ch}' requires temperature and gas_density")
            log_ne = log_ne_raw - LOG10_MU_E_MP
            ch_profiles[ch] = log_kT + log_ne
        elif source == "derived:kT-2/3*ne":
            log_kT = direct_cache.get("temperature")
            log_ne_raw = direct_cache.get("gas_density")
            if log_kT is None or log_ne_raw is None:
                raise ValueError(f"Derived channel '{ch}' requires temperature and gas_density")
            log_ne = log_ne_raw - LOG10_MU_E_MP
            ch_profiles[ch] = log_kT - (2.0 / 3.0) * log_ne

    return ch_profiles


def _compute_cross_features(
    ch_profiles: Dict[str, np.ndarray],
    r_idx: np.ndarray,
    radial_bins: np.ndarray,
    cross_features: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Compute cross-channel features for a single bin.

    Parameters
    ----------
    ch_profiles : dict channel -> (N_sims, n_halos, n_radial)
    r_idx : selected radial indices
    radial_bins : full radial grid (kpc)
    cross_features : list of feature names to compute

    Returns
    -------
    dict mapping feature name -> (N_sims,) scalar per simulation
    """
    features = {}
    if not cross_features:
        return features

    for feat in cross_features:
        if feat == "T_core_ratio":
            # log10(T_core / T_outer) = log10(T) at r_min - log10(T) at r_max
            if "kT" in ch_profiles and len(r_idx) >= 2:
                kT = ch_profiles["kT"]  # (N, n_halos, n_radial)
                inner = np.median(kT[:, :, r_idx[0]], axis=1)
                outer = np.median(kT[:, :, r_idx[-1]], axis=1)
                features[feat] = inner - outer  # log ratio
            else:
                features[feat] = None

        elif feat == "K_slope":
            # Entropy slope: d(log K)/d(log r) between inner and outer radii
            if "kT" in ch_profiles and len(r_idx) >= 2:
                kT = ch_profiles["kT"]
                ne = ch_profiles.get("ne")
                if ne is not None:
                    K = kT - (2.0 / 3.0) * ne  # log10 entropy
                    K_inner = np.median(K[:, :, r_idx[0]], axis=1)
                    K_outer = np.median(K[:, :, r_idx[-1]], axis=1)
                    log_r_inner = np.log10(radial_bins[r_idx[0]])
                    log_r_outer = np.log10(radial_bins[r_idx[-1]])
                    dr = log_r_outer - log_r_inner
                    features[feat] = (K_outer - K_inner) / max(dr, 0.01)
                else:
                    features[feat] = None
            else:
                features[feat] = None

        elif feat == "ne_slope":
            # Density slope: d(log ne)/d(log r)
            if "ne" in ch_profiles and len(r_idx) >= 2:
                ne = ch_profiles["ne"]
                ne_inner = np.median(ne[:, :, r_idx[0]], axis=1)
                ne_outer = np.median(ne[:, :, r_idx[-1]], axis=1)
                log_r_inner = np.log10(radial_bins[r_idx[0]])
                log_r_outer = np.log10(radial_bins[r_idx[-1]])
                dr = log_r_outer - log_r_inner
                features[feat] = (ne_outer - ne_inner) / max(dr, 0.01)
            else:
                features[feat] = None
        else:
            logger.warning("Unknown cross-feature '%s', skipping", feat)
            features[feat] = None

    return features


# ---------------------------------------------------------------------------
# Forward model: single mass-z bin
# ---------------------------------------------------------------------------

def simulate_bin(
    emu,
    theta_full: np.ndarray,
    mass_range: Tuple[float, float],
    n_halos: int,
    radial_bins: np.ndarray,
    r_idx: np.ndarray,
    estimate_R500,
    emu_fields: Sequence[str],
    summary_channels: Sequence[str],
    cross_features: Sequence[str],
    n_anp_samples: int,
    max_gpu_rows: int,
    rng: np.random.RandomState,
    redshift: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Run the ANP forward model for a single (mass, z) bin.

    Returns dict with keys for each summary channel (shape N_sims, n_r_selected),
    cross-feature scalars, and 'median_mass'.
    """
    N_sims = theta_full.shape[0]
    n_radial = len(radial_bins)
    nr = len(r_idx)

    # Draw random masses within this bin for each simulation
    masses_sim = rng.uniform(mass_range[0], mass_range[1],
                             size=(N_sims, n_halos))

    # Storage: per-channel profiles
    all_ch_profiles = {ch: np.zeros((N_sims, n_halos, n_radial))
                       for ch in summary_channels}
    # Cross-feature storage
    all_cross = {feat: np.zeros(N_sims) for feat in cross_features}

    sims_per_call = max(1, max_gpu_rows // n_halos)

    for si in range(0, N_sims, sims_per_call):
        se = min(si + sims_per_call, N_sims)
        bs = se - si

        flat_logM = masses_sim[si:se].ravel()
        flat_M = 10.0 ** flat_logM
        flat_R500 = estimate_R500(flat_logM)
        flat_r_over_R500 = (radial_bins[None, :] / flat_R500[:, None]).astype(np.float32)

        flat_theta = np.repeat(theta_full[si:se], n_halos, axis=0)

        predict_kwargs = dict(
            M=flat_M,
            r_bins=flat_r_over_R500,
            field=list(emu_fields),
            n_samples=n_anp_samples,
        )
        if redshift is not None:
            predict_kwargs["redshift"] = redshift

        result_i = emu.predict(flat_theta, **predict_kwargs)

        # Compute all channel profiles using the unified helper
        ch_profiles = _compute_channel_profiles(
            result_i, emu_fields, summary_channels,
            bs, n_halos, n_radial, rng,
        )

        for ch in summary_channels:
            all_ch_profiles[ch][si:se] = ch_profiles[ch]

        # Cross-channel features
        if cross_features:
            feats = _compute_cross_features(
                ch_profiles, r_idx, radial_bins, cross_features,
            )
            for feat, val in feats.items():
                if val is not None:
                    all_cross[feat][si:se] = val

    # Aggregate: median over halos at selected radii
    result = {}
    for ch in summary_channels:
        result[ch] = np.median(all_ch_profiles[ch][:, :, r_idx], axis=1)
    result["median_mass"] = np.median(masses_sim, axis=1)
    for feat in cross_features:
        result[f"cross_{feat}"] = all_cross[feat]

    return result


# ---------------------------------------------------------------------------
# Summary construction (mass × redshift bins)
# ---------------------------------------------------------------------------

def build_summary_layout(
    mass_bin_edges: np.ndarray,
    z_bin_edges: np.ndarray,
    obs_logM: np.ndarray,
    obs_z: np.ndarray,
    radial_bins: np.ndarray,
    max_r_by_mass: Dict[int, float],
    n_radii_per_bin: int = 5,
    summary_channels: Sequence[str] = ("kT", "ne", "P"),
    cross_features: Sequence[str] = (),
    n_scalars: int = 2,
) -> Dict:
    """Compute summary layout: dimensions, radial indices, halo counts per (mass, z) bin."""
    n_mass = len(mass_bin_edges) - 1
    n_z = len(z_bin_edges) - 1
    n_obs_channels = len(summary_channels)
    n_cross = len(cross_features)

    layout = {
        "mass_bin_edges": mass_bin_edges,
        "z_bin_edges": z_bin_edges,
        "summary_channels": list(summary_channels),
        "cross_features": list(cross_features),
        "bins": [],
        "n_summary": 0,
    }

    for mb in range(n_mass):
        for zb in range(n_z):
            in_bin = ((obs_logM >= mass_bin_edges[mb]) &
                      (obs_logM < mass_bin_edges[mb + 1]) &
                      (obs_z >= z_bin_edges[zb]) &
                      (obs_z < z_bin_edges[zb + 1]))
            n_in = int(in_bin.sum())

            max_r = max_r_by_mass.get(mb, 300.0)
            valid_r = np.where(radial_bins <= max_r)[0]
            if len(valid_r) > n_radii_per_bin:
                sel = np.linspace(0, len(valid_r) - 1,
                                  n_radii_per_bin, dtype=int)
                r_idx = valid_r[sel]
            else:
                r_idx = valid_r
            nr = len(r_idx)

            dims = nr * n_obs_channels + n_cross + n_scalars
            z_center = 0.5 * (z_bin_edges[zb] + z_bin_edges[zb + 1])

            layout["bins"].append({
                "mb": mb, "zb": zb,
                "mass_lo": mass_bin_edges[mb],
                "mass_hi": mass_bin_edges[mb + 1],
                "z_lo": z_bin_edges[zb],
                "z_hi": z_bin_edges[zb + 1],
                "z_center": z_center,
                "r_idx": r_idx,
                "nr": nr,
                "n_halos": n_in,
                "dims": dims,
                "offset": layout["n_summary"],
            })
            layout["n_summary"] += dims

    return layout


def compute_obs_summary(
    layout: Dict,
    obs_profiles: Dict[str, np.ndarray],
    obs_logM: np.ndarray,
    obs_z: np.ndarray,
    radial_bins: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the observed summary vector from interpolated profiles.

    Parameters
    ----------
    layout : output of build_summary_layout()
    obs_profiles : dict mapping channel name -> (n_clusters, n_r) log10 profiles
        Expected keys: one entry per channel in layout['summary_channels'].
        Derived channels (P, K, y) are computed from kT and ne if not provided.
    obs_logM : (n_clusters,)
    obs_z : (n_clusters,)
    radial_bins : radial grid (kpc), needed for cross-features
    """
    x_obs = np.zeros(layout["n_summary"], dtype=np.float32)
    summary_channels = layout["summary_channels"]
    cross_features = layout.get("cross_features", [])
    n_channels = len(summary_channels)
    n_cross = len(cross_features)

    # Pre-compute derived profiles if needed
    derived = dict(obs_profiles)
    if "P" in summary_channels and "P" not in derived:
        if "kT" in derived and "ne" in derived:
            derived["P"] = derived["kT"] + derived["ne"]
    if "K" in summary_channels and "K" not in derived:
        if "kT" in derived and "ne" in derived:
            derived["K"] = derived["kT"] - (2.0 / 3.0) * derived["ne"]
    if "y" in summary_channels and "y" not in derived:
        if "kT" in derived and "ne" in derived:
            derived["y"] = derived["kT"] + derived["ne"]

    for b in layout["bins"]:
        in_bin = ((obs_logM >= b["mass_lo"]) & (obs_logM < b["mass_hi"]) &
                  (obs_z >= b["z_lo"]) & (obs_z < b["z_hi"]))
        n_in = int(in_bin.sum())
        off = b["offset"]
        nr = b["nr"]
        r_idx = b["r_idx"]

        # Profile channels
        for ci, ch in enumerate(summary_channels):
            ch_off = off + ci * nr
            if n_in > 0 and nr > 0 and ch in derived:
                x_obs[ch_off:ch_off + nr] = np.nanmedian(
                    derived[ch][in_bin][:, r_idx], axis=0)

        # Cross-channel features
        cross_off = off + n_channels * nr
        for fi, feat in enumerate(cross_features):
            if n_in > 0 and nr > 0:
                if feat == "T_core_ratio" and "kT" in derived:
                    inner = np.nanmedian(derived["kT"][in_bin][:, r_idx[0]])
                    outer = np.nanmedian(derived["kT"][in_bin][:, r_idx[-1]])
                    x_obs[cross_off + fi] = inner - outer
                elif feat == "K_slope" and "kT" in derived and "ne" in derived:
                    K = derived["kT"] - (2.0 / 3.0) * derived["ne"]
                    K_inner = np.nanmedian(K[in_bin][:, r_idx[0]])
                    K_outer = np.nanmedian(K[in_bin][:, r_idx[-1]])
                    if radial_bins is not None:
                        dr = np.log10(radial_bins[r_idx[-1]]) - np.log10(radial_bins[r_idx[0]])
                        x_obs[cross_off + fi] = (K_outer - K_inner) / max(dr, 0.01)
                elif feat == "ne_slope" and "ne" in derived:
                    ne_inner = np.nanmedian(derived["ne"][in_bin][:, r_idx[0]])
                    ne_outer = np.nanmedian(derived["ne"][in_bin][:, r_idx[-1]])
                    if radial_bins is not None:
                        dr = np.log10(radial_bins[r_idx[-1]]) - np.log10(radial_bins[r_idx[0]])
                        x_obs[cross_off + fi] = (ne_outer - ne_inner) / max(dr, 0.01)

        # Scalars: log1p(count) + median_mass (match simulate_all_bins)
        scalar_off = off + n_channels * nr + n_cross
        x_obs[scalar_off] = np.log1p(n_in)
        x_obs[scalar_off + 1] = np.median(obs_logM[in_bin]) if n_in > 0 else 0.0

    return np.nan_to_num(x_obs, nan=0.0)


def simulate_all_bins(
    emu,
    theta_full: np.ndarray,
    layout: Dict,
    radial_bins: np.ndarray,
    estimate_R500,
    emu_fields: Sequence[str],
    summary_channels: Sequence[str],
    cross_features: Sequence[str],
    n_anp_samples: int,
    max_gpu_rows: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Run the full forward model across all (mass, z) bins.

    Returns x_all: (N_sims, n_summary)
    """
    N_sims = theta_full.shape[0]
    x_all = np.zeros((N_sims, layout["n_summary"]), dtype=np.float32)
    n_channels = len(summary_channels)
    n_cross = len(cross_features)

    for bi, b in enumerate(layout["bins"]):
        n_halos = b["n_halos"]
        r_idx = b["r_idx"]
        nr = b["nr"]
        off = b["offset"]
        mass_range = (b["mass_lo"], b["mass_hi"])

        if n_halos == 0:
            scalar_off = off + n_channels * nr + n_cross
            x_all[:, scalar_off + 1] = 0.5 * (b["mass_lo"] + b["mass_hi"])
            logger.info("  Bin %d (M[%.1f-%.1f] z[%.2f-%.2f]): 0 halos, skip",
                        bi, b["mass_lo"], b["mass_hi"], b["z_lo"], b["z_hi"])
            continue

        # Use bin center redshift for multi-z emulator
        z_val = b["z_center"] if b["z_center"] > 0.01 else None

        result = simulate_bin(
            emu=emu,
            theta_full=theta_full,
            mass_range=mass_range,
            n_halos=n_halos,
            radial_bins=radial_bins,
            r_idx=r_idx,
            estimate_R500=estimate_R500,
            emu_fields=emu_fields,
            summary_channels=summary_channels,
            cross_features=cross_features,
            n_anp_samples=n_anp_samples,
            max_gpu_rows=max_gpu_rows,
            rng=rng,
            redshift=z_val,
        )

        # Fill profile channels
        for ci, ch in enumerate(summary_channels):
            ch_off = off + ci * nr
            x_all[:, ch_off:ch_off + nr] = result[ch]

        # Fill cross-features
        cross_off = off + n_channels * nr
        for fi, feat in enumerate(cross_features):
            key = f"cross_{feat}"
            if key in result:
                x_all[:, cross_off + fi] = result[key]

        # Scalars: log1p(count) + median_mass  (match compute_obs_summary)
        scalar_off = off + n_channels * nr + n_cross
        x_all[:, scalar_off] = np.log1p(n_halos)
        x_all[:, scalar_off + 1] = result["median_mass"]

        logger.info("  Bin %d (M[%.1f-%.1f] z[%.2f-%.2f]): %d halos",
                    bi, b["mass_lo"], b["mass_hi"], b["z_lo"], b["z_hi"], n_halos)

    return x_all


# ---------------------------------------------------------------------------
# High-level pipeline class
# ---------------------------------------------------------------------------

class SBIPipeline:
    """End-to-end SBI pipeline with multi-z support."""

    def __init__(self, config: Optional[SBIConfig] = None, **kwargs):
        if config is None:
            config = SBIConfig(**kwargs)
        self.cfg = config
        self.emu = None
        self.obs_catalog = None
        self.layout = None
        self.x_obs = None
        self.theta_internal = None
        self.x_all = None
        self.posterior = None
        self.samples = None

    def load_emulator(self, checkpoint: Optional[str] = None):
        """Load the ANP emulator from a checkpoint."""
        from anp_emulator import Emulator
        ckpt = checkpoint or self.cfg.emu_checkpoint
        self.emu = Emulator.from_checkpoint(ckpt, device=self.cfg.device)
        logger.info("Loaded emulator: %s (channels: %s)",
                    Path(ckpt).name, self.emu.target_names)
        return self

    def load_observations(
        self,
        obs_npz: Optional[str] = None,
        additional_catalogs: Optional[Dict] = None,
    ):
        """Load and process observational data.

        Parameters
        ----------
        obs_npz : path to ACCEPT+X-COP obs_expanded.npz
        additional_catalogs : dict mapping catalog_name -> ObsCatalog
            for additional survey data
        """
        from anp_emulator.observations import (
            load_accept_xcop, interpolate_to_grid, ObsCatalog,
        )

        npz = obs_npz or self.cfg.obs_npz

        # Load primary catalog
        if npz and Path(npz).exists():
            self.obs_catalog = load_accept_xcop(
                npz,
                logM_min=self.cfg.logM_min,
                logM_max=self.cfg.logM_max,
            )
        else:
            self.obs_catalog = ObsCatalog([], "empty")

        # Merge additional catalogs
        if additional_catalogs:
            for name, cat in additional_catalogs.items():
                filtered = cat.filter_mass(self.cfg.logM_min, self.cfg.logM_max)
                self.obs_catalog.clusters.extend(filtered.clusters)
                logger.info("Added %d clusters from %s", len(filtered.clusters), name)

        logger.info("Total observations: %d clusters, z=[%.3f, %.3f], logM=[%.1f, %.1f]",
                    self.obs_catalog.n_clusters,
                    self.obs_catalog.z_range[0], self.obs_catalog.z_range[1],
                    self.obs_catalog.logM_range[0], self.obs_catalog.logM_range[1])
        return self

    def load_training_data(self, train_npz: str):
        """Load CAMELS training data for prior bounds and R500 relation."""
        data = np.load(train_npz, allow_pickle=True)
        self.all_params = data["params"]
        self.param_names = list(data["param_names"])
        self.fiducial_params = data["fiducial_params"] if "fiducial_params" in data else None
        # Handle both logM500c and M500c keys
        if "logM500c" in data:
            self.all_logM = data["logM500c"]
        else:
            self.all_logM = np.log10(data["M500c"])
        self.all_R500 = data["R500c"]
        self.radial_bins_train = data["radial_bins"] if "radial_bins" in data else None

        # Fit R500(M) relation
        valid = (self.all_R500 > 0) & np.isfinite(self.all_logM)
        mass_mask = (self.all_logM >= self.cfg.logM_min) & (self.all_logM <= self.cfg.logM_max)
        mask = valid & mass_mask
        self._R500_poly = np.poly1d(np.polyfit(
            self.all_logM[mask], np.log10(self.all_R500[mask]), 1,
        ))

        logger.info("Training data: %d runs, %d params, R500 relation fit from %d halos",
                    self.all_params.shape[0], self.all_params.shape[1], mask.sum())
        return self

    def estimate_R500(self, logM: np.ndarray) -> np.ndarray:
        """Estimate R500 from M500 using CAMELS-calibrated relation."""
        return 10.0 ** self._R500_poly(logM)

    def setup_prior(self):
        """Set up SBI prior from training data bounds."""
        from sbi.utils import BoxUniform

        LOG_UNIFORM_NAMES = {
            "WindEnergyIn1e51erg", "RadioFeedbackFactor", "VariableWindVelFactor",
            "RadioFeedbackReiorientationFactor", "MaxSfrTimescale", "FactorForSofterEQS",
            "ThermalWindFraction", "WindFreeTravelDensFac", "WindEnergyReductionFactor",
            "WindEnergyReductionMetallicity", "SeedBlackHoleMass", "BlackHoleAccretionFactor",
            "BlackHoleEddingtonFactor", "BlackHoleFeedbackFactor", "BlackHoleRadiativeEfficiency",
            "QuasarThreshold", "UVBH0beta", "UVBHepbeta", "SofteningComovingType01",
        }

        COSMO_PARAMS = {"Omega0", "sigma8", "OmegaBaryon", "HubbleParam", "n_s"}
        FIDUCIAL_COSMO = {"Omega0": 0.3, "sigma8": 0.8, "OmegaBaryon": 0.049,
                          "HubbleParam": 0.6711, "n_s": 0.9624}

        all_param_names = list(self.param_names)

        if self.cfg.fix_cosmo:
            self.free_idx = np.array([i for i, pn in enumerate(all_param_names)
                                      if pn not in COSMO_PARAMS])
            self.fixed_idx = np.array([i for i, pn in enumerate(all_param_names)
                                       if pn in COSMO_PARAMS])
            self.fiducial_cosmo = np.array([FIDUCIAL_COSMO[all_param_names[i]]
                                            for i in self.fixed_idx])
        else:
            self.free_idx = np.arange(len(all_param_names))
            self.fixed_idx = np.array([], dtype=int)
            self.fiducial_cosmo = np.array([])

        self.selected_param_names = [all_param_names[i] for i in self.free_idx]
        self.is_log_param = np.array([pn in LOG_UNIFORM_NAMES
                                      for pn in self.selected_param_names])
        self.log_idx = np.where(self.is_log_param)[0]
        self.lin_idx = np.where(~self.is_log_param)[0]

        bounds_lo = np.array([self.all_params[:, i].min() for i in self.free_idx])
        bounds_hi = np.array([self.all_params[:, i].max() for i in self.free_idx])

        self.prior_lo = bounds_lo.copy()
        self.prior_hi = bounds_hi.copy()
        self.prior_lo[self.log_idx] = np.log10(bounds_lo[self.log_idx])
        self.prior_hi[self.log_idx] = np.log10(bounds_hi[self.log_idx])

        self.sbi_prior = BoxUniform(
            low=torch.tensor(self.prior_lo, dtype=torch.float32),
            high=torch.tensor(self.prior_hi, dtype=torch.float32),
        )

        logger.info("Prior: %d free params (%d log-uniform, %d linear)",
                    len(self.free_idx), len(self.log_idx), len(self.lin_idx))
        return self

    def theta_to_physical(self, theta_internal: np.ndarray) -> np.ndarray:
        t = theta_internal.copy()
        t[..., self.log_idx] = 10.0 ** t[..., self.log_idx]
        return t

    def expand_to_full(self, theta_free_phys: np.ndarray) -> np.ndarray:
        if len(self.fixed_idx) == 0:
            return theta_free_phys
        squeeze = theta_free_phys.ndim == 1
        if squeeze:
            theta_free_phys = theta_free_phys[None, :]
        N = theta_free_phys.shape[0]
        full = np.zeros((N, len(self.param_names)), dtype=theta_free_phys.dtype)
        full[:, self.free_idx] = theta_free_phys
        full[:, self.fixed_idx] = self.fiducial_cosmo
        return full[0] if squeeze else full

    def build_layout(self, radial_bins: np.ndarray | None = None, inner_skip: int = 4):
        """Set up the summary statistic layout.

        Parameters
        ----------
        radial_bins : array, optional
            Physical kpc grid for obs interpolation and emulator calls.
            Defaults to the CAMELS training grid with *inner_skip* inner
            bins removed (matching v1 behaviour).
        inner_skip : int
            Number of inner radial bins to skip (default 4, matching v1).
        """
        from anp_emulator.observations import interpolate_to_grid

        if radial_bins is None:
            if self.radial_bins_train is None:
                raise RuntimeError("Call load_training_data() before build_layout()")
            radial_bins = self.radial_bins_train[inner_skip:]

        # Interpolate observations to grid
        # Determine which raw channels to interpolate based on summary_channels
        interp_channels = set()
        for ch in self.cfg.summary_channels:
            if ch in ("kT", "ne"):
                interp_channels.add(ch)
            elif ch in ("P", "K", "y"):
                interp_channels.update(("kT", "ne"))
            elif ch == "Z":
                interp_channels.add("Z")
        # Always need kT and ne for cross-features
        if self.cfg.cross_features:
            interp_channels.update(("kT", "ne"))

        obs_interp = interpolate_to_grid(
            self.obs_catalog, radial_bins, channels=tuple(sorted(interp_channels)),
        )
        self._obs_interp = obs_interp
        self._radial_bins = radial_bins

        # Determine z_bin_edges based on emulator capabilities
        z_edges = np.array(self.cfg.z_bin_edges)

        self.layout = build_summary_layout(
            mass_bin_edges=np.array(self.cfg.mass_bin_edges),
            z_bin_edges=z_edges,
            obs_logM=obs_interp["logM"],
            obs_z=obs_interp["z"],
            radial_bins=radial_bins,
            max_r_by_mass=self.cfg.max_r_by_mass,
            n_radii_per_bin=self.cfg.n_radii_per_bin,
            summary_channels=self.cfg.summary_channels,
            cross_features=self.cfg.cross_features,
        )

        # Build obs profile dict for compute_obs_summary
        obs_profiles = {}
        for ch in ("kT", "ne", "Z", "compton_y"):
            key = f"{ch}_profiles"
            if key in obs_interp:
                obs_profiles[ch] = obs_interp[key]

        # Observed summary
        self.x_obs = compute_obs_summary(
            self.layout,
            obs_profiles,
            obs_interp["logM"],
            obs_interp["z"],
            radial_bins=radial_bins,
        )

        logger.info("Summary: %d dims across %d (mass×z) bins",
                    self.layout["n_summary"], len(self.layout["bins"]))
        for b in self.layout["bins"]:
            logger.info("  M[%.1f-%.1f] z[%.2f-%.2f]: %d halos, %d dims",
                        b["mass_lo"], b["mass_hi"], b["z_lo"], b["z_hi"],
                        b["n_halos"], b["dims"])
        return self

    def generate_simulations(self):
        """Generate N_sims simulated summary vectors."""
        rng = np.random.RandomState(self.cfg.seed)

        # Sample from prior
        theta_int = self.sbi_prior.sample((self.cfg.n_sims,))
        theta_phys = theta_int.clone()
        theta_phys[:, self.log_idx] = 10.0 ** theta_phys[:, self.log_idx]
        theta_full = self.expand_to_full(theta_phys.cpu().numpy())

        self.theta_internal = theta_int.cpu().numpy()

        t0 = time.time()
        self.x_all = simulate_all_bins(
            emu=self.emu,
            theta_full=theta_full,
            layout=self.layout,
            radial_bins=self._radial_bins,
            estimate_R500=self.estimate_R500,
            emu_fields=self.cfg.emu_fields,
            summary_channels=self.cfg.summary_channels,
            cross_features=self.cfg.cross_features,
            n_anp_samples=self.cfg.n_anp_samples,
            max_gpu_rows=self.cfg.max_gpu_rows,
            rng=rng,
        )
        elapsed = time.time() - t0
        logger.info("Generated %d simulations in %.1fs", self.cfg.n_sims, elapsed)
        return self

    def check_coverage(self) -> np.ndarray:
        """Check if x_obs is within the simulated range for each dimension."""
        x_min = self.x_all.min(axis=0)
        x_max = self.x_all.max(axis=0)
        in_range = (self.x_obs >= x_min) & (self.x_obs <= x_max)
        nonzero = self.x_obs != 0
        n_out = int((~in_range & nonzero).sum())
        n_total = int(nonzero.sum())
        logger.info("Coverage: %d/%d non-zero dims in range (%.1f%%)",
                    n_total - n_out, n_total,
                    100 * (n_total - n_out) / max(n_total, 1))
        return in_range

    def train_npe(self):
        """Train the Neural Posterior Estimator."""
        from sbi.inference import SNPE

        theta_tensor = torch.tensor(self.theta_internal, dtype=torch.float32)
        x_tensor = torch.tensor(self.x_all, dtype=torch.float32)

        # Drop constant-variance columns that would produce NaN after
        # standardization and carry no parameter information.
        x_std = x_tensor.std(dim=0)
        self._varying_mask = x_std > 1e-8
        n_dropped = int((~self._varying_mask).sum())
        if n_dropped > 0:
            logger.info("Dropping %d constant summary dims (of %d)",
                        n_dropped, x_tensor.shape[1])
        x_tensor = x_tensor[:, self._varying_mask]

        inference = SNPE(prior=self.sbi_prior)
        inference.append_simulations(theta_tensor, x_tensor)

        self.density_estimator = inference.train(
            training_batch_size=self.cfg.batch_size_npe,
            max_num_epochs=self.cfg.max_epochs,
            stop_after_epochs=self.cfg.patience_npe,
            show_train_summary=True,
        )
        self.posterior = inference.build_posterior(
            self.density_estimator,
            sample_with=self.cfg.sample_with,
            mcmc_method=self.cfg.mcmc_method,
        )
        logger.info("NPE trained successfully (sample_with=%s)", self.cfg.sample_with)
        return self

    def sample_posterior(self) -> np.ndarray:
        """Draw posterior samples conditioned on x_obs."""
        # Apply same column mask as training
        x_obs_filtered = torch.tensor(
            self.x_obs, dtype=torch.float32
        )[self._varying_mask].unsqueeze(0)

        sample_kwargs = dict(x=x_obs_filtered)
        # Only add MCMC kwargs if the actual posterior is MCMC-based
        from sbi.inference.posteriors.mcmc_posterior import MCMCPosterior
        if isinstance(self.posterior, MCMCPosterior):
            n_params = len(self.selected_param_names)
            sample_kwargs.update(
                thin=max(1, n_params // 5),
                warmup_steps=200,
                num_chains=20,
            )
            logger.info("MCMC sampling: %d chains, thin=%d, warmup=200",
                        sample_kwargs["num_chains"], sample_kwargs["thin"])

        raw_samples = self.posterior.sample(
            (self.cfg.n_posterior,), **sample_kwargs
        ).cpu().numpy()

        # Back-transform to physical
        phys_samples = self.theta_to_physical(raw_samples)

        # Expand to full 35-param if cosmo was fixed
        full_samples = self.expand_to_full(phys_samples)

        self.samples = full_samples
        self.samples_internal = raw_samples
        logger.info("Posterior: %d samples drawn", self.cfg.n_posterior)
        return full_samples

    def save_results(self, results_dir: str | Path, tag: str = "v2"):
        """Save posterior samples, metadata, and constraints."""
        results_dir = Path(results_dir)
        results_dir.mkdir(exist_ok=True, parents=True)

        # Save samples
        np.savez_compressed(
            results_dir / f"anp_npe_{tag}_results.npz",
            samples=self.samples,
            samples_internal=self.samples_internal,
            param_names=self.param_names,
            selected_param_names=self.selected_param_names,
            prior_lo=self.prior_lo,
            prior_hi=self.prior_hi,
            x_obs=self.x_obs,
            mass_bin_edges=np.array(self.cfg.mass_bin_edges),
            z_bin_edges=np.array(self.cfg.z_bin_edges),
            fix_cosmo=self.cfg.fix_cosmo,
        )

        # Compute and save constraints (use internal/log space for log-uniform)
        import json
        constraints = {}
        for i, pn in enumerate(self.param_names):
            s_phys = self.samples[:, i]
            # For log-uniform params, compute ratio in log-space
            if pn in self.selected_param_names:
                si = list(self.selected_param_names).index(pn)
                if self.is_log_param[si]:
                    s_int = self.samples_internal[:, si]
                    prior_width = self.prior_hi[si] - self.prior_lo[si]
                    ratio = float((np.percentile(s_int, 84) - np.percentile(s_int, 16))
                                  / (prior_width + 1e-30))
                else:
                    ratio = float(
                        (np.percentile(s_phys, 84) - np.percentile(s_phys, 16)) /
                        (self.all_params[:, i].max() - self.all_params[:, i].min() + 1e-30)
                    )
            else:
                ratio = float(
                    (np.percentile(s_phys, 84) - np.percentile(s_phys, 16)) /
                    (self.all_params[:, i].max() - self.all_params[:, i].min() + 1e-30)
                )

            constraints[pn] = {
                "median": float(np.median(s_phys)),
                "lo16": float(np.percentile(s_phys, 16)),
                "hi84": float(np.percentile(s_phys, 84)),
                "width_prior_ratio": ratio,
            }

        with open(results_dir / f"anp_npe_{tag}_constraints.json", "w") as f:
            json.dump(constraints, f, indent=2)

        logger.info("Results saved to %s", results_dir)
        return self

    # -------------------------------------------------------------------
    # Phase 4: Simulation-Based Calibration (SBC)
    # -------------------------------------------------------------------

    def run_sbc(
        self,
        n_sbc_trials: int = 200,
        n_posterior_per_trial: int = 1000,
        credible_levels: Sequence[float] = (0.5, 0.75, 0.9, 0.95),
    ) -> Dict:
        """Run Simulation-Based Calibration to validate posterior coverage.

        Draws test parameters from the prior, generates simulated data,
        then checks whether the truth falls within the expected credible
        intervals at the specified levels.

        Returns dict with:
          'coverage': {level: fraction_covered} per param
          'ranks': (n_trials, n_params) fractional rank of truth in posterior
          'expected': ideal coverage at each level
        """
        rng = np.random.RandomState(self.cfg.seed + 9999)

        n_params = len(self.selected_param_names)
        ranks = np.zeros((n_sbc_trials, n_params))
        coverage = {lvl: np.zeros(n_params) for lvl in credible_levels}

        logger.info("SBC: %d trials, %d posterior samples each",
                    n_sbc_trials, n_posterior_per_trial)

        for trial in range(n_sbc_trials):
            # Draw a test parameter from the prior
            theta_test = self.sbi_prior.sample((1,)).cpu().numpy()[0]

            # Convert to physical space and expand
            theta_phys = theta_test.copy()
            theta_phys[self.log_idx] = 10.0 ** theta_phys[self.log_idx]
            theta_full = self.expand_to_full(theta_phys)

            # Generate simulated summary for this test theta
            x_test = simulate_all_bins(
                emu=self.emu,
                theta_full=theta_full[None, :],
                layout=self.layout,
                radial_bins=self._radial_bins,
                estimate_R500=self.estimate_R500,
                emu_fields=self.cfg.emu_fields,
                summary_channels=self.cfg.summary_channels,
                cross_features=self.cfg.cross_features,
                n_anp_samples=self.cfg.n_anp_samples,
                max_gpu_rows=self.cfg.max_gpu_rows,
                rng=rng,
            )[0]

            # Draw posterior samples conditioned on x_test
            x_test_filtered = torch.tensor(
                x_test, dtype=torch.float32
            )[self._varying_mask].unsqueeze(0)

            try:
                post_samples = self.posterior.sample(
                    (n_posterior_per_trial,), x=x_test_filtered
                ).cpu().numpy()
            except Exception as e:
                logger.warning("SBC trial %d failed: %s", trial, e)
                ranks[trial] = np.nan
                continue

            # Compute rank: fraction of posterior below truth
            for pi in range(n_params):
                ranks[trial, pi] = np.mean(post_samples[:, pi] < theta_test[pi])

            # Coverage: does truth fall in the p-credible interval?
            for lvl in credible_levels:
                lo_q = (1 - lvl) / 2
                hi_q = 1 - lo_q
                lo = np.percentile(post_samples, lo_q * 100, axis=0)
                hi = np.percentile(post_samples, hi_q * 100, axis=0)
                inside = (theta_test >= lo) & (theta_test <= hi)
                coverage[lvl] += inside.astype(float)

            if (trial + 1) % 50 == 0:
                logger.info("  SBC trial %d/%d", trial + 1, n_sbc_trials)

        # Normalize coverage
        valid_trials = np.sum(np.isfinite(ranks[:, 0]))
        for lvl in credible_levels:
            coverage[lvl] /= max(valid_trials, 1)

        result = {
            "ranks": ranks,
            "coverage": coverage,
            "expected": {lvl: lvl for lvl in credible_levels},
            "param_names": self.selected_param_names,
            "n_trials": int(valid_trials),
        }

        # Log summary
        for lvl in credible_levels:
            mean_cov = coverage[lvl].mean()
            logger.info("  SBC %d%% CI: mean coverage = %.1f%% (expected %d%%)",
                        int(lvl * 100), mean_cov * 100, int(lvl * 100))

        self.sbc_results = result
        return result

    # -------------------------------------------------------------------
    # Phase 4: Multi-dataset ablation
    # -------------------------------------------------------------------

    def run_ablation(
        self,
        survey_groups: Optional[Dict[str, list]] = None,
        n_sims: int = 50_000,
        n_posterior: int = 5_000,
    ) -> Dict[str, Dict]:
        """Run ablation analysis: remove each survey group and re-run inference.

        Parameters
        ----------
        survey_groups : dict mapping group_name -> list of survey source names
            E.g. {"ACCEPT": ["ACCEPT"], "XCOP": ["X-COP"], "eROSITA": ["eROSITA_eFEDS"]}
        n_sims, n_posterior : reduced counts for speed

        Returns
        -------
        dict mapping group_name -> constraint dict (same format as save_results)
        """
        if survey_groups is None:
            # Autodetect from cluster sources
            sources = set(c.source for c in self.obs_catalog.clusters)
            survey_groups = {s: [s] for s in sorted(sources)}

        ablation_results = {}

        # Baseline: full dataset (already computed)
        if self.samples is not None:
            ablation_results["full"] = self._compute_constraint_dict()

        for group_name, source_list in survey_groups.items():
            logger.info("Ablation: removing %s (%s)", group_name, source_list)

            # Create catalog with this group removed
            kept = [c for c in self.obs_catalog.clusters
                    if c.source not in source_list]
            if len(kept) == 0:
                logger.warning("  No clusters remain — skipping")
                continue
            if len(kept) == len(self.obs_catalog.clusters):
                logger.info("  No clusters removed — skipping")
                continue

            from anp_emulator.observations import ObsCatalog, interpolate_to_grid
            ablated_catalog = ObsCatalog(kept, f"ablated_{group_name}")

            # Rebuild layout with ablated catalog
            interp_channels = set()
            for ch in self.cfg.summary_channels:
                if ch in ("kT", "ne"):
                    interp_channels.add(ch)
                elif ch in ("P", "K", "y"):
                    interp_channels.update(("kT", "ne"))
                elif ch == "Z":
                    interp_channels.add("Z")
            if self.cfg.cross_features:
                interp_channels.update(("kT", "ne"))

            obs_interp = interpolate_to_grid(
                ablated_catalog, self._radial_bins,
                channels=tuple(sorted(interp_channels)),
            )

            layout = build_summary_layout(
                mass_bin_edges=np.array(self.cfg.mass_bin_edges),
                z_bin_edges=np.array(self.cfg.z_bin_edges),
                obs_logM=obs_interp["logM"],
                obs_z=obs_interp["z"],
                radial_bins=self._radial_bins,
                max_r_by_mass=self.cfg.max_r_by_mass,
                n_radii_per_bin=self.cfg.n_radii_per_bin,
                summary_channels=self.cfg.summary_channels,
                cross_features=self.cfg.cross_features,
            )

            obs_profiles = {}
            for ch in ("kT", "ne", "Z", "compton_y"):
                key = f"{ch}_profiles"
                if key in obs_interp:
                    obs_profiles[ch] = obs_interp[key]

            x_obs_abl = compute_obs_summary(
                layout, obs_profiles,
                obs_interp["logM"], obs_interp["z"],
                radial_bins=self._radial_bins,
            )

            # Generate simulations with reduced n_sims
            rng = np.random.RandomState(self.cfg.seed + hash(group_name) % 10000)
            theta_int = self.sbi_prior.sample((n_sims,))
            theta_phys = theta_int.clone()
            theta_phys[:, self.log_idx] = 10.0 ** theta_phys[:, self.log_idx]
            theta_full = self.expand_to_full(theta_phys.cpu().numpy())

            x_all = simulate_all_bins(
                emu=self.emu,
                theta_full=theta_full,
                layout=layout,
                radial_bins=self._radial_bins,
                estimate_R500=self.estimate_R500,
                emu_fields=self.cfg.emu_fields,
                summary_channels=self.cfg.summary_channels,
                cross_features=self.cfg.cross_features,
                n_anp_samples=self.cfg.n_anp_samples,
                max_gpu_rows=self.cfg.max_gpu_rows,
                rng=rng,
            )

            # Train NPE
            from sbi.inference import SNPE
            theta_tensor = torch.tensor(theta_int.cpu().numpy(), dtype=torch.float32)
            x_tensor = torch.tensor(x_all, dtype=torch.float32)
            x_std = x_tensor.std(dim=0)
            mask = x_std > 1e-8
            x_tensor = x_tensor[:, mask]

            inference = SNPE(prior=self.sbi_prior)
            inference.append_simulations(theta_tensor, x_tensor)
            de = inference.train(
                training_batch_size=self.cfg.batch_size_npe,
                max_num_epochs=self.cfg.max_epochs,
                stop_after_epochs=self.cfg.patience_npe,
            )
            posterior = inference.build_posterior(de)

            x_obs_filtered = torch.tensor(
                x_obs_abl, dtype=torch.float32
            )[mask].unsqueeze(0)

            samples = posterior.sample(
                (n_posterior,), x=x_obs_filtered
            ).cpu().numpy()

            # Compute constraints
            constraints = {}
            for i, pn in enumerate(self.selected_param_names):
                s = samples[:, i]
                pw = self.prior_hi[i] - self.prior_lo[i]
                ratio = float((np.percentile(s, 84) - np.percentile(s, 16)) / (pw + 1e-30))
                constraints[pn] = {"width_prior_ratio": ratio}

            ablation_results[f"minus_{group_name}"] = {
                "constraints": constraints,
                "n_clusters": len(kept),
                "n_sims": n_sims,
            }
            logger.info("  Ablation -%s: %d clusters, done", group_name, len(kept))

        self.ablation_results = ablation_results
        return ablation_results

    def _compute_constraint_dict(self) -> Dict:
        """Compute constraint ratios from current samples."""
        constraints = {}
        for i, pn in enumerate(self.selected_param_names):
            s = self.samples_internal[:, i]
            pw = self.prior_hi[i] - self.prior_lo[i]
            ratio = float((np.percentile(s, 84) - np.percentile(s, 16)) / (pw + 1e-30))
            constraints[pn] = {"width_prior_ratio": ratio}
        return {"constraints": constraints,
                "n_clusters": self.obs_catalog.n_clusters}

    def run(
        self,
        train_npz: str,
        obs_npz: str,
        radial_bins: Optional[np.ndarray] = None,
        results_dir: str = "sbi_results",
        tag: str = "v2",
    ) -> np.ndarray:
        """Full pipeline end-to-end."""
        if self.emu is None:
            self.load_emulator()
        self.load_observations(obs_npz)
        self.load_training_data(train_npz)
        self.setup_prior()

        if radial_bins is None:
            radial_bins = np.logspace(np.log10(15), np.log10(2000), 21).astype(np.float32)

        self.build_layout(radial_bins)
        self.check_coverage()  # will log warning if out-of-range
        self.generate_simulations()
        self.check_coverage()
        self.train_npe()
        samples = self.sample_posterior()
        self.save_results(results_dir, tag)
        return samples
