#!/usr/bin/env python3
"""Run temperature-only Pyro NUTS on observed cluster profiles and save full chains.

This script mirrors the notebook workflow but is cluster-friendly and reproducible.
It supports three free-parameter setups:
- 4 parameters: indices [2, 3, 4, 5]
- 6 parameters: indices [0, 1, 2, 3, 4, 5]
- 35 parameters: all emulator parameters

By default, it runs all three setups sequentially and writes one .npz file per setup,
including full per-chain theta samples.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS

from anp_emulator import Emulator


def normalize_cluster_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def parse_cluster_catalog(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 7:
                continue
            rows.append(
                {
                    "name": parts[0],
                    "z": float(parts[3]),
                    "r500_arcmin": float(parts[6]),
                }
            )
    if not rows:
        raise RuntimeError(f"No usable rows parsed from {path}")
    return rows


def e_z(z: float, omega_m: float = 0.3, omega_l: float = 0.7) -> float:
    return float(np.sqrt(omega_m * (1.0 + z) ** 3 + omega_l))


def angular_diameter_distance_mpc(
    z: float,
    h0: float = 70.0,
    omega_m: float = 0.3,
    omega_l: float = 0.7,
) -> float:
    c_km_s = 299792.458
    z_grid = np.linspace(0.0, z, 1024)
    ez = np.sqrt(omega_m * (1.0 + z_grid) ** 3 + omega_l)
    d_c_mpc = (c_km_s / h0) * np.trapz(1.0 / ez, z_grid)
    return float(d_c_mpc / (1.0 + z))


def r500_mpc_from_arcmin(r500_arcmin: float, z: float) -> float:
    theta_rad = np.deg2rad(r500_arcmin / 60.0)
    d_a = angular_diameter_distance_mpc(z)
    return float(theta_rad * d_a)


def m500_from_r500(r500_mpc: float, z: float, h0: float = 70.0) -> float:
    h = h0 / 100.0
    rho_c0 = 2.775e11 * h * h
    rho_c_z = rho_c0 * e_z(z) ** 2
    return float((4.0 / 3.0) * np.pi * 500.0 * rho_c_z * (r500_mpc**3))


@dataclass
class ObservedProfiles:
    names: np.ndarray
    y_obs_log: np.ndarray
    sigma_obs_log: np.ndarray
    m500: np.ndarray
    r_bins: np.ndarray
    valid_counts: np.ndarray


def load_temperature_profiles(
    kt_dir: Path,
    catalog_path: Path,
    rr500_nbin: int,
    sigma_floor_frac: float,
) -> ObservedProfiles:
    catalog_rows = parse_cluster_catalog(catalog_path)
    catalog_by_norm = {normalize_cluster_name(r["name"]): r for r in catalog_rows}

    kt_files = sorted(kt_dir.glob("*_kt.dat"))
    if not kt_files:
        raise FileNotFoundError(f"No *_kt.dat files found in {kt_dir}")

    rr500_grid = np.linspace(0.05, 1.0, rr500_nbin, dtype=np.float32)
    names: List[str] = []
    m500_list: List[float] = []
    temp_grid_list: List[np.ndarray] = []
    err_grid_list: List[np.ndarray] = []

    missing_catalog = 0
    for fp in kt_files:
        cluster_name = fp.stem.replace("_kt", "")
        row = catalog_by_norm.get(normalize_cluster_name(cluster_name))
        if row is None:
            missing_catalog += 1
            continue

        arr = np.genfromtxt(
            fp,
            dtype=float,
            comments="#",
            usecols=(0, 4, 5, 6),
            invalid_raise=False,
            ndmin=2,
        )
        if arr.size == 0:
            continue

        r_arcmin = arr[:, 0]
        t_keV = arr[:, 1]
        e_lo = np.abs(arr[:, 2])
        e_hi = np.abs(arr[:, 3])
        e_keV = 0.5 * (e_lo + e_hi)

        m = np.isfinite(r_arcmin) & np.isfinite(t_keV) & np.isfinite(e_keV) & (t_keV > 0)
        if int(np.sum(m)) < 5:
            continue

        r500_arcmin = float(row["r500_arcmin"])
        z_cl = float(row["z"])
        r_over_r500 = r_arcmin[m] / r500_arcmin

        t_i = np.interp(rr500_grid, r_over_r500, t_keV[m], left=np.nan, right=np.nan)
        e_i = np.interp(rr500_grid, r_over_r500, e_keV[m], left=np.nan, right=np.nan)

        r500_mpc = r500_mpc_from_arcmin(r500_arcmin, z_cl)
        m500 = m500_from_r500(r500_mpc, z_cl)

        names.append(cluster_name)
        m500_list.append(m500)
        temp_grid_list.append(t_i)
        err_grid_list.append(e_i)

    if not names:
        raise RuntimeError("No valid profiles after catalog matching and filtering")

    temp_keV = np.asarray(temp_grid_list, dtype=np.float64)
    err_keV = np.asarray(err_grid_list, dtype=np.float64)

    sigma_floor_dex = float(sigma_floor_frac) / np.log(10.0)
    sigma_data_dex = err_keV / np.clip(temp_keV * np.log(10.0), 1e-12, None)
    sigma_obs_log = np.maximum(sigma_data_dex, sigma_floor_dex)

    valid = np.isfinite(temp_keV) & (temp_keV > 0)
    sigma_obs_log[~valid] = 1e6
    y_obs_log = np.log10(np.clip(temp_keV, 1e-12, None))

    valid_counts = np.sum(np.isfinite(y_obs_log) & (sigma_obs_log < 1e5), axis=1).astype(int)

    print(f"Loaded {len(names)} matched halos from {kt_dir}")
    print(f"Missing catalog matches: {missing_catalog}")
    print(f"Median valid bins per halo: {float(np.median(valid_counts)):.1f}")

    return ObservedProfiles(
        names=np.asarray(names),
        y_obs_log=y_obs_log[..., None],
        sigma_obs_log=sigma_obs_log[..., None],
        m500=np.asarray(m500_list, dtype=np.float32),
        r_bins=np.repeat(rr500_grid[None, :], len(names), axis=0).astype(np.float32),
        valid_counts=valid_counts,
    )


def get_free_indices(kind: str, theta_dim: int) -> np.ndarray:
    if kind == "4":
        return np.array([2, 3, 4, 5], dtype=int)
    if kind == "6":
        return np.arange(6, dtype=int)
    if kind == "35":
        return np.arange(theta_dim, dtype=int)
    raise ValueError(f"Unsupported free-set: {kind}")


def compute_split_rhat(chains: np.ndarray, fit_idxs: Sequence[int]) -> Dict[str, float]:
    # chains shape: (n_chain, n_draw, theta_dim)
    out: Dict[str, float] = {}
    post = chains[:, :, fit_idxs]
    for ii, j in enumerate(fit_idxs):
        x = post[:, :, ii]
        n_chain, n_draw = x.shape
        if n_chain < 2 or n_draw < 2:
            out[str(j)] = float("nan")
            continue
        chain_means = x.mean(axis=1)
        chain_vars = x.var(axis=1, ddof=1)
        w = float(chain_vars.mean())
        b = float(n_draw * chain_means.var(ddof=1))
        var_hat = ((n_draw - 1.0) / n_draw) * w + (1.0 / n_draw) * b
        out[str(j)] = float(np.sqrt(var_hat / w)) if w > 1e-20 else float("nan")
    return out


def run_configuration(
    *,
    emu: Emulator,
    obs: ObservedProfiles,
    theta_cols: Sequence[str],
    theta_fid: np.ndarray,
    prior_lo: np.ndarray,
    prior_hi: np.ndarray,
    fit_idxs: np.ndarray,
    n_halos: int,
    warmup: int,
    samples: int,
    chains: int,
    target_accept: float,
    max_tree_depth: int,
    include_model_error: bool,
    fix_eps: float,
    seed: int,
) -> Dict[str, object]:
    prior_lo_run = prior_lo.copy()
    prior_hi_run = prior_hi.copy()

    all_idxs = np.arange(emu.theta_dim, dtype=int)
    fixed_idxs = np.setdiff1d(all_idxs, fit_idxs)
    for j in fixed_idxs:
        prior_lo_run[j] = theta_fid[j] - fix_eps
        prior_hi_run[j] = theta_fid[j] + fix_eps

    order = np.argsort(obs.valid_counts)[::-1]
    batch_idxs = order[: min(n_halos, len(order))]
    batch_names = [str(obs.names[i]) for i in batch_idxs]

    y_batch = obs.y_obs_log[batch_idxs]
    sig_batch = obs.sigma_obs_log[batch_idxs]
    m_batch = obs.m500[batch_idxs]
    r_batch = obs.r_bins[batch_idxs]

    fit_idxs_list = [int(i) for i in fit_idxs]
    fit_param_names = [theta_cols[i] for i in fit_idxs_list]

    print("-" * 80)
    print(f"Running fit with {len(fit_idxs_list)} free parameters")
    print("Free indices:", fit_idxs_list)
    print("Free names:", fit_param_names)
    print(f"Halos in batch: {len(batch_idxs)}")

    def pyro_model(y_obs_h, sigma_obs_h, m_h, r_bins_h, include_model_err: bool = False):
        theta = torch.tensor(theta_fid, dtype=torch.float32, device=emu.device).clone()

        for j, pname in zip(fit_idxs_list, fit_param_names):
            lo = torch.tensor(float(prior_lo_run[j]), dtype=torch.float32, device=emu.device)
            hi = torch.tensor(float(prior_hi_run[j]), dtype=torch.float32, device=emu.device)
            theta_j = pyro.sample(f"param_{j}_{pname}", dist.Uniform(lo, hi))
            theta[j] = theta_j

        mu_log10, std_log10 = emu.predict_log10_differentiable(
            theta=theta,
            M=m_h,
            r_bins=r_bins_h,
            field=["temperature"],
            snapnum=90,
            redshift=0.0,
            n_samples=1,
            deterministic=True,
        )

        mu = mu_log10[..., 0].to(torch.float64)
        std_model = std_log10[..., 0].to(torch.float64).clamp_min(1e-12)
        y = torch.as_tensor(y_obs_h[..., 0], dtype=torch.float64, device=emu.device)
        sigma = torch.as_tensor(sigma_obs_h[..., 0], dtype=torch.float64, device=emu.device)

        if include_model_err:
            total_var = (sigma**2 + std_model**2).clamp_min(1e-30)
        else:
            total_var = (sigma**2).clamp_min(1e-30)

        mask = torch.isfinite(y) & torch.isfinite(mu) & torch.isfinite(total_var) & (sigma < 1e5)
        y_m = y[mask]
        mu_m = mu[mask]
        sig_m = torch.sqrt(total_var[mask])

        pyro.sample("obs", dist.Normal(mu_m, sig_m).to_event(1), obs=y_m)

    chain_post: List[np.ndarray] = []
    chain_secs: List[float] = []

    for c in range(chains):
        pyro.clear_param_store()
        pyro.set_rng_seed(seed + c)

        t0 = time.time()
        kernel = NUTS(
            pyro_model,
            target_accept_prob=target_accept,
            max_tree_depth=max_tree_depth,
        )
        mcmc = MCMC(
            kernel,
            num_samples=samples,
            warmup_steps=warmup,
            num_chains=1,
            disable_progbar=False,
        )
        mcmc.run(y_batch, sig_batch, m_batch, r_batch, include_model_error)
        dt = time.time() - t0

        s = mcmc.get_samples()
        n_draw = int(next(iter(s.values())).shape[0])
        flat_chain = np.repeat(theta_fid[None, :], n_draw, axis=0).astype(np.float64)
        for j, pname in zip(fit_idxs_list, fit_param_names):
            flat_chain[:, j] = s[f"param_{j}_{pname}"].detach().cpu().numpy().astype(np.float64)

        chain_post.append(flat_chain)
        chain_secs.append(float(dt))
        print(f"Chain {c + 1}/{chains} finished in {dt:.1f} s")

    samples_all = np.stack(chain_post, axis=0)
    rhat = compute_split_rhat(samples_all, fit_idxs_list)

    return {
        "samples": samples_all,
        "fit_idxs": np.asarray(fit_idxs_list, dtype=int),
        "fit_param_names": np.asarray(fit_param_names, dtype=object),
        "batch_idxs": batch_idxs.astype(int),
        "batch_names": np.asarray(batch_names, dtype=object),
        "chain_seconds": np.asarray(chain_secs, dtype=np.float64),
        "rhat": rhat,
        "prior_lo_run": prior_lo_run,
        "prior_hi_run": prior_hi_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature-only Pyro NUTS on observed kT profiles")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("anp_training_runs/anp_all_profiles_20260325_175639"),
    )
    parser.add_argument("--kt-dir", type=Path, default=Path("/mnt/home/mlee1/CPGP_xray/Data/data/kT"))
    parser.add_argument("--catalog-path", type=Path, default=Path("/mnt/home/mlee1/CPGP_xray/Data/clusters.txt"))
    parser.add_argument(
        "--onep-csv",
        type=Path,
        default=Path("/mnt/home/mlee1/Sims/IllustrisTNG/L50n512/1P/CosmoAstroSeed_IllustrisTNG_L50n512_1P.txt"),
    )
    parser.add_argument("--rr500-nbin", type=int, default=24)
    parser.add_argument("--n-halos", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1500)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.90)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument("--include-model-error", action="store_true")
    parser.add_argument("--sigma-floor-frac-temp", type=float, default=0.25)
    parser.add_argument("--fix-eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument(
        "--free-set",
        choices=["4", "6", "35", "all"],
        default="all",
        help="Run one setup (4/6/35) or all three sequentially",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("validation_results") / "nuts_temperature_runs",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")

    os.environ.setdefault("OMP_NUM_THREADS", "1")

    emu = Emulator.from_run_dir(args.run_dir, device=args.device)

    onep = pd.read_csv(args.onep_csv, sep=r"\s+", engine="python")
    if "#Name" in onep.columns:
        onep = onep.rename(columns={"#Name": "tag"})

    theta_cols = [
        c
        for c in onep.columns
        if c != "tag" and pd.api.types.is_numeric_dtype(onep[c]) and str(c).strip().lower() != "seed"
    ][: emu.theta_dim]

    fid_mask = onep["tag"].str.contains("fiducial", case=False, na=False)
    if fid_mask.any():
        theta_fid = onep.loc[fid_mask].iloc[0][theta_cols].to_numpy(dtype=np.float64)
    else:
        theta_fid = onep[theta_cols].median().to_numpy(dtype=np.float64)

    prior_lo = np.array([onep[c].min() for c in theta_cols], dtype=np.float64)
    prior_hi = np.array([onep[c].max() for c in theta_cols], dtype=np.float64)

    obs = load_temperature_profiles(
        kt_dir=args.kt_dir,
        catalog_path=args.catalog_path,
        rr500_nbin=args.rr500_nbin,
        sigma_floor_frac=args.sigma_floor_frac_temp,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_dir / f"temp_nuts_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    free_sets = [args.free_set] if args.free_set != "all" else ["4", "6", "35"]

    run_manifest: List[Dict[str, object]] = []
    for k, free_key in enumerate(free_sets):
        fit_idxs = get_free_indices(free_key, emu.theta_dim)
        result = run_configuration(
            emu=emu,
            obs=obs,
            theta_cols=theta_cols,
            theta_fid=theta_fid,
            prior_lo=prior_lo,
            prior_hi=prior_hi,
            fit_idxs=fit_idxs,
            n_halos=args.n_halos,
            warmup=args.warmup,
            samples=args.samples,
            chains=args.chains,
            target_accept=args.target_accept,
            max_tree_depth=args.max_tree_depth,
            include_model_error=args.include_model_error,
            fix_eps=args.fix_eps,
            seed=args.seed + 1000 * k,
        )

        out_npz = out_root / f"temp_only_free{free_key}_chains.npz"
        np.savez_compressed(
            out_npz,
            samples=result["samples"],
            fit_idxs=result["fit_idxs"],
            fit_param_names=result["fit_param_names"],
            theta_cols=np.asarray(theta_cols, dtype=object),
            theta_fid=theta_fid,
            prior_lo_run=result["prior_lo_run"],
            prior_hi_run=result["prior_hi_run"],
            batch_idxs=result["batch_idxs"],
            batch_names=result["batch_names"],
            chain_seconds=result["chain_seconds"],
            warmup=np.asarray(args.warmup, dtype=np.int64),
            samples_per_chain=np.asarray(args.samples, dtype=np.int64),
            n_chains=np.asarray(args.chains, dtype=np.int64),
            target_accept=np.asarray(args.target_accept, dtype=np.float64),
            max_tree_depth=np.asarray(args.max_tree_depth, dtype=np.int64),
            include_model_error=np.asarray(int(args.include_model_error), dtype=np.int64),
            n_halos=np.asarray(args.n_halos, dtype=np.int64),
            rr500_nbin=np.asarray(args.rr500_nbin, dtype=np.int64),
            run_dir=np.asarray(str(args.run_dir), dtype=object),
            kt_dir=np.asarray(str(args.kt_dir), dtype=object),
            catalog_path=np.asarray(str(args.catalog_path), dtype=object),
        )

        summary = {
            "free_set": free_key,
            "out_file": str(out_npz),
            "fit_idxs": [int(i) for i in result["fit_idxs"]],
            "fit_param_names": [str(x) for x in result["fit_param_names"]],
            "n_halos": int(args.n_halos),
            "batch_names": [str(x) for x in result["batch_names"]],
            "chain_seconds": [float(x) for x in result["chain_seconds"]],
            "rhat_by_param_index": result["rhat"],
        }
        run_manifest.append(summary)

        print(f"Saved chains for free-set {free_key}: {out_npz}")
        print("Rhat by fitted parameter index:", result["rhat"])

    manifest_path = out_root / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print("=" * 80)
    print(f"All requested runs complete. Outputs in: {out_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
