"""
Observational data ingestion and standardization for the ANP-SBI pipeline.

Provides loaders for multiple cluster surveys and a unified interface that
produces standardized profile dictionaries ready for the SBI forward model.

Supported / planned surveys:
  - ACCEPT  (Cavagnolo+ 2009) — X-ray T, n_e profiles
  - X-COP   (Ghirardini+ 2019) — deep X-ray T, n_e, Z profiles
  - eROSITA eFEDS / eRASS1    — stacked X-ray T, n_e profiles
  - CHEX-MATE (2021)           — XMM-Newton T, n_e, Z profiles
  - SPT-SZ / SPT-ECS          — tSZ (Compton-y) stacked profiles
  - ACT DR6 (Hilton+ 2021)    — tSZ stacked profiles
  - CLoGS   (O'Sullivan+ 2017) — galaxy group X-ray profiles

Each loader returns a list of ClusterObservation dataclass instances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClusterObservation:
    """Standardized container for a single cluster's observed profiles."""

    name: str
    source: str                           # survey/catalog name
    z: float                              # redshift
    logM500: float                        # log10(M500c / Msun), estimated
    R500_kpc: float                       # R500 in kpc (estimated or measured)

    # Radial grid in kpc (common for all profiles of this cluster)
    r_kpc: np.ndarray                     # (n_r,)

    # Profiles — None if not available from this survey
    kT: Optional[np.ndarray] = None       # keV, shape (n_r,)
    kT_err: Optional[np.ndarray] = None
    ne: Optional[np.ndarray] = None       # cm^-3, shape (n_r,)
    ne_err: Optional[np.ndarray] = None
    Z: Optional[np.ndarray] = None        # solar units, shape (n_r,)
    Z_err: Optional[np.ndarray] = None
    compton_y: Optional[np.ndarray] = None  # Compton-y profile
    compton_y_err: Optional[np.ndarray] = None

    # Selection metadata (for forward-model selection function)
    selection: Dict = field(default_factory=dict)


@dataclass
class ObsCatalog:
    """A collection of cluster observations with metadata."""

    clusters: List[ClusterObservation]
    survey: str
    description: str = ""

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def logM_range(self) -> Tuple[float, float]:
        masses = [c.logM500 for c in self.clusters]
        return (min(masses), max(masses))

    @property
    def z_range(self) -> Tuple[float, float]:
        redshifts = [c.z for c in self.clusters]
        return (min(redshifts), max(redshifts))

    def filter_mass(self, logM_min: float = -np.inf,
                    logM_max: float = np.inf) -> "ObsCatalog":
        filtered = [c for c in self.clusters
                    if logM_min <= c.logM500 <= logM_max]
        return ObsCatalog(filtered, self.survey, self.description)

    def filter_redshift(self, z_min: float = 0.0,
                        z_max: float = np.inf) -> "ObsCatalog":
        filtered = [c for c in self.clusters if z_min <= c.z <= z_max]
        return ObsCatalog(filtered, self.survey, self.description)

    def to_npz(self, path: str | Path) -> None:
        """Save catalog to a standardized NPZ file (obs_expanded_v2 format).

        Format:
          cluster_names: array of names
          Per cluster (prefixed by name):
            {name}__z, {name}__logM500, {name}__R500_kpc, {name}__source
            {name}__r_kpc, {name}__kT, {name}__kT_err, {name}__ne, {name}__ne_err
            {name}__Z, {name}__Z_err, {name}__compton_y, {name}__compton_y_err
            {name}__selection (json-encoded)
        """
        import json as _json
        arrays = {"cluster_names": np.array([c.name for c in self.clusters])}
        for c in self.clusters:
            pfx = c.name
            arrays[f"{pfx}__z"] = np.float64(c.z)
            arrays[f"{pfx}__logM500"] = np.float64(c.logM500)
            arrays[f"{pfx}__logM500_est"] = np.float64(c.logM500)
            arrays[f"{pfx}__R500_kpc"] = np.float64(c.R500_kpc)
            arrays[f"{pfx}__source"] = np.array(c.source)
            arrays[f"{pfx}__R_kpc"] = c.r_kpc
            if c.kT is not None:
                arrays[f"{pfx}__kT"] = c.kT
            if c.kT_err is not None:
                arrays[f"{pfx}__kT_err"] = c.kT_err
            if c.ne is not None:
                arrays[f"{pfx}__ne"] = c.ne
            if c.ne_err is not None:
                arrays[f"{pfx}__ne_err"] = c.ne_err
            if c.Z is not None:
                arrays[f"{pfx}__Z"] = c.Z
            if c.Z_err is not None:
                arrays[f"{pfx}__Z_err"] = c.Z_err
            if c.compton_y is not None:
                arrays[f"{pfx}__compton_y"] = c.compton_y
            if c.compton_y_err is not None:
                arrays[f"{pfx}__compton_y_err"] = c.compton_y_err
            if c.selection:
                arrays[f"{pfx}__selection"] = np.array(_json.dumps(c.selection))
        np.savez_compressed(str(path), **arrays)
        logger.info("Saved %d clusters to %s", len(self.clusters), path)

    @classmethod
    def from_npz(cls, path: str | Path, survey: str = "loaded") -> "ObsCatalog":
        """Load catalog from standardized NPZ file."""
        import json as _json
        data = np.load(str(path), allow_pickle=True)
        names = list(data["cluster_names"])
        clusters = []
        for name in names:
            try:
                z_val = float(data[f"{name}__z"])
                logM = float(data.get(f"{name}__logM500",
                                      data.get(f"{name}__logM500_est", 14.0)))
            except (KeyError, ValueError):
                continue

            r_kpc = data.get(f"{name}__R_kpc", data.get(f"{name}__r_kpc"))
            if r_kpc is None:
                continue
            r_kpc = np.asarray(r_kpc, dtype=np.float64)

            R500 = float(data.get(f"{name}__R500_kpc", R500_from_M500(logM, z_val)))
            source = str(data.get(f"{name}__source", survey))

            def _get(key):
                v = data.get(f"{name}__{key}")
                return np.asarray(v, dtype=np.float64) if v is not None else None

            sel_raw = data.get(f"{name}__selection")
            sel = _json.loads(str(sel_raw)) if sel_raw is not None else {}

            clusters.append(ClusterObservation(
                name=str(name), source=source, z=z_val,
                logM500=logM, R500_kpc=R500, r_kpc=r_kpc,
                kT=_get("kT"), kT_err=_get("kT_err"),
                ne=_get("ne"), ne_err=_get("ne_err"),
                Z=_get("Z"), Z_err=_get("Z_err"),
                compton_y=_get("compton_y"), compton_y_err=_get("compton_y_err"),
                selection=sel,
            ))
        return cls(clusters, survey)


# ---------------------------------------------------------------------------
# Physical constants and helpers
# ---------------------------------------------------------------------------

MU_E = 2.0 / (1.0 + 0.76)     # mean molecular weight per electron
M_P = 1.6726e-24               # proton mass [g]
K_B = 1.3807e-16               # Boltzmann constant [erg/K]
KEV_TO_K = 1.1605e7            # 1 keV in Kelvin


def logM500_from_kT_global(kT_keV: float, z: float = 0.0) -> float:
    """Rough M-T scaling relation: M500 ∝ (kT/5keV)^{1.6}.

    Uses Arnaud+ 2005 normalization.  Returns log10(M500/Msun).
    """
    E_z = np.sqrt(0.3 * (1 + z) ** 3 + 0.7)
    M500 = 3.84e14 * (kT_keV / 5.0) ** 1.71 / E_z
    return float(np.log10(M500))


def R500_from_M500(logM500: float, z: float = 0.0) -> float:
    """Compute R500 from M500 using the critical density definition.

    Returns R500 in kpc.
    """
    H0 = 67.11  # km/s/Mpc
    H_z = H0 * np.sqrt(0.3 * (1 + z) ** 3 + 0.7)
    rho_crit = 3.0 * (H_z * 1e5 / 3.086e24) ** 2 / (8.0 * np.pi * 6.674e-8)
    M500_g = 10 ** logM500 * 1.989e33
    R500_cm = (3.0 * M500_g / (4.0 * np.pi * 500.0 * rho_crit)) ** (1.0 / 3.0)
    return float(R500_cm / 3.086e21)  # cm -> kpc


# ---------------------------------------------------------------------------
# Loader: existing obs_expanded.npz (ACCEPT + X-COP)
# ---------------------------------------------------------------------------

def load_accept_xcop(
    npz_path: str | Path,
    logM_min: float = 13.2,
    logM_max: float = 14.9,
    min_radial_points: int = 3,
    max_frac_err: float = 0.5,
) -> ObsCatalog:
    """Load the pre-existing ACCEPT + X-COP combined file."""
    data = np.load(str(npz_path), allow_pickle=True)
    names = list(data["cluster_names"])
    clusters: List[ClusterObservation] = []

    for name in names:
        try:
            z_val = float(data[f"{name}__z"])
            logM = float(data[f"{name}__logM500_est"])
        except (KeyError, ValueError):
            continue

        if not (logM_min <= logM <= logM_max):
            continue

        R_kpc = data.get(f"{name}__R_kpc")
        kT = data.get(f"{name}__kT")
        kT_err = data.get(f"{name}__kT_err")
        ne_arr = data.get(f"{name}__ne")
        ne_err = data.get(f"{name}__ne_err")
        source = str(data.get(f"{name}__source", "ACCEPT"))

        if R_kpc is None or kT is None or ne_arr is None:
            continue

        R_kpc = np.asarray(R_kpc, dtype=np.float64)
        kT = np.asarray(kT, dtype=np.float64)
        kT_err = np.asarray(kT_err, dtype=np.float64) if kT_err is not None else np.full_like(kT, np.nan)
        ne_arr = np.asarray(ne_arr, dtype=np.float64)
        ne_err = np.asarray(ne_err, dtype=np.float64) if ne_err is not None else np.full_like(ne_arr, np.nan)

        # Quality filter: enough good data points
        good_kT = np.isfinite(kT) & (kT > 0)
        good_ne = np.isfinite(ne_arr) & (ne_arr > 0)
        if good_kT.sum() < min_radial_points or good_ne.sum() < min_radial_points:
            continue

        R500 = R500_from_M500(logM, z_val)

        # Handle separate kT radial grid (X-COP)
        kT_R = data.get(f"{name}__kT_R_kpc")
        if kT_R is not None:
            kT_R = np.asarray(kT_R, dtype=np.float64)
            # Interpolate kT onto the ne radial grid
            kT_interp = np.interp(
                np.log10(np.clip(R_kpc, 1.0, None)),
                np.log10(np.clip(kT_R[good_kT], 1.0, None)),
                kT[good_kT],
                left=np.nan, right=np.nan,
            )
            kT_err_interp = np.interp(
                np.log10(np.clip(R_kpc, 1.0, None)),
                np.log10(np.clip(kT_R[good_kT], 1.0, None)),
                kT_err[good_kT],
                left=np.nan, right=np.nan,
            )
            kT = kT_interp
            kT_err = kT_err_interp

        # Metallicity (X-COP may have it)
        Z_arr = data.get(f"{name}__Z")
        Z_err_arr = data.get(f"{name}__Z_err")

        clusters.append(ClusterObservation(
            name=str(name),
            source=source,
            z=z_val,
            logM500=logM,
            R500_kpc=R500,
            r_kpc=R_kpc,
            kT=kT,
            kT_err=kT_err,
            ne=ne_arr,
            ne_err=ne_err,
            Z=np.asarray(Z_arr, dtype=np.float64) if Z_arr is not None else None,
            Z_err=np.asarray(Z_err_arr, dtype=np.float64) if Z_err_arr is not None else None,
        ))

    logger.info("ACCEPT+X-COP: loaded %d clusters (mass [%.1f, %.1f])",
                len(clusters), logM_min, logM_max)
    return ObsCatalog(clusters, "ACCEPT+XCOP",
                      f"{len(clusters)} clusters, z<0.1, X-ray kT+ne")


# ---------------------------------------------------------------------------
# Selection functions for survey-aware forward modeling
# ---------------------------------------------------------------------------

def selection_erosita_efeds(logM: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Approximate eFEDS selection: flux-limited in 0.5-2 keV band.

    Detection probability based on count-rate threshold modeled as
    logistic function of (logM, z).  Calibrated to match the eFEDS
    mass-redshift distribution (Liu+ 2022, Fig. 5).
    """
    # Simple logistic model: p = sigmoid(a*(logM - M_lim(z)))
    # M_lim(z) increases with z (harder to detect at higher z)
    M_lim = 13.6 + 0.8 * z
    x = 5.0 * (logM - M_lim)
    return 1.0 / (1.0 + np.exp(-x))


def selection_spt_sz(logM: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Approximate SPT-SZ selection: SZ signal-to-noise > 4.5.

    The SZ signal scales as M^{5/3} * E(z)^{2/3} / D_A(z)^2,
    roughly flat in z for massive clusters.
    """
    # xi_SZ ∝ M^{5/3} roughly; detection at xi > 4.5
    # This gives M_lim ~ 14.0 at z~0.3, rising slowly
    M_lim = 14.0 + 0.2 * np.log1p(z)
    x = 4.0 * (logM - M_lim)
    return 1.0 / (1.0 + np.exp(-x))


def selection_act_dr6(logM: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Approximate ACT DR6 selection (similar to SPT but wider area)."""
    M_lim = 13.9 + 0.25 * np.log1p(z)
    x = 4.0 * (logM - M_lim)
    return 1.0 / (1.0 + np.exp(-x))


def selection_chexmate(logM: np.ndarray, z: np.ndarray) -> np.ndarray:
    """CHEX-MATE: Planck SZ S/N > 6.5 selection, z < 0.6."""
    p = np.ones_like(logM)
    p[z > 0.6] = 0.0
    M_lim = 14.2 + 0.1 * z
    x = 5.0 * (logM - M_lim)
    p *= 1.0 / (1.0 + np.exp(-x))
    return p


SELECTION_FUNCTIONS = {
    "eROSITA_eFEDS": selection_erosita_efeds,
    "eROSITA_eRASS1": selection_erosita_efeds,  # similar flux limit
    "SPT-SZ": selection_spt_sz,
    "ACT_DR6": selection_act_dr6,
    "CHEX-MATE": selection_chexmate,
}


# ---------------------------------------------------------------------------
# Loader stubs for future surveys
# ---------------------------------------------------------------------------

def load_erosita_efeds(
    data_dir: str | Path,
    logM_min: float = 13.0,
    logM_max: float = 15.0,
    z_max: float = 1.2,
) -> ObsCatalog:
    """Load eROSITA eFEDS cluster profiles.

    Expected directory structure (data_dir/):
      efeds_clusters.fits  — Cluster catalog (Liu+ 2022, Table 1)
                             Required columns: Name, REDSHIFT, LAMBDA, r500_kpc
                             or M500 (from weak-lensing calibrated richness-mass)
      efeds_profiles.fits  — Profile table (Bahar+ 2022)
                             Required columns: Name, r_kpc, kT_keV, kT_err,
                             ne_cm3, ne_err

    Can also accept CSV format with the same column names.

    References: Bahar+ 2022, Liu+ 2022, Ghirardini+ 2024.
    """
    data_dir = Path(data_dir)
    clusters: List[ClusterObservation] = []

    # Try FITS first, fall back to CSV
    cat_path = data_dir / "efeds_clusters.fits"
    prof_path = data_dir / "efeds_profiles.fits"
    cat_csv = data_dir / "efeds_clusters.csv"
    prof_csv = data_dir / "efeds_profiles.csv"

    cat_data = None
    prof_data = None

    if cat_path.exists():
        try:
            from astropy.io import fits
            with fits.open(str(cat_path)) as hdul:
                cat_data = hdul[1].data
        except ImportError:
            logger.warning("astropy not available for FITS reading; trying CSV")
    if cat_data is None and cat_csv.exists():
        import csv
        with open(cat_csv) as f:
            reader = csv.DictReader(f)
            cat_data = list(reader)

    if prof_path.exists():
        try:
            from astropy.io import fits
            with fits.open(str(prof_path)) as hdul:
                prof_data = hdul[1].data
        except ImportError:
            pass
    if prof_data is None and prof_csv.exists():
        import csv
        with open(prof_csv) as f:
            reader = csv.DictReader(f)
            prof_data = list(reader)

    if cat_data is None:
        logger.info("eROSITA eFEDS: no data files found in %s", data_dir)
        return ObsCatalog(clusters, "eROSITA_eFEDS", "no data found")

    # Build profile lookup by cluster name
    prof_lookup: Dict[str, Dict[str, list]] = {}
    if prof_data is not None:
        for row in prof_data:
            name = str(row.get("Name", row.get("name", row.get("CLUSTER_NAME", ""))))
            if not name:
                continue
            if name not in prof_lookup:
                prof_lookup[name] = {"r": [], "kT": [], "kT_err": [],
                                     "ne": [], "ne_err": []}
            prof_lookup[name]["r"].append(float(row.get("r_kpc", row.get("R_KPC", 0))))
            prof_lookup[name]["kT"].append(float(row.get("kT_keV", row.get("KT", 0))))
            prof_lookup[name]["kT_err"].append(float(row.get("kT_err", row.get("KT_ERR", 0))))
            prof_lookup[name]["ne"].append(float(row.get("ne_cm3", row.get("NE", 0))))
            prof_lookup[name]["ne_err"].append(float(row.get("ne_err", row.get("NE_ERR", 0))))

    # Parse cluster catalog
    for row in cat_data:
        name = str(row.get("Name", row.get("name", row.get("CLUSTER_NAME", ""))))
        try:
            z_val = float(row.get("REDSHIFT", row.get("z", row.get("redshift", -1))))
        except (ValueError, TypeError):
            continue

        if z_val < 0 or z_val > z_max:
            continue

        # Mass: try M500 directly, or richness-mass relation
        try:
            logM = float(row.get("logM500", row.get("LOG_M500", 0)))
        except (ValueError, TypeError):
            logM = 0
        if logM == 0:
            # Richness-mass relation: logM500 ≈ 14.0 + 1.1*log10(lambda/40)
            try:
                lam = float(row.get("LAMBDA", row.get("richness", 0)))
                if lam > 0:
                    logM = 14.0 + 1.1 * np.log10(lam / 40.0)
            except (ValueError, TypeError):
                pass
        if logM == 0:
            # Fall back to M-T relation if kT is available in catalog
            try:
                kT_mean = float(row.get("kT", row.get("TEMP", 0)))
                if kT_mean > 0:
                    logM = logM500_from_kT_global(kT_mean, z_val)
            except (ValueError, TypeError):
                continue
        if not (logM_min <= logM <= logM_max):
            continue

        R500 = R500_from_M500(logM, z_val)

        # Get profiles if available
        if name in prof_lookup:
            p = prof_lookup[name]
            r_kpc = np.array(p["r"])
            kT = np.array(p["kT"])
            kT_err = np.array(p["kT_err"])
            ne = np.array(p["ne"])
            ne_err = np.array(p["ne_err"])
        else:
            continue  # skip clusters without profiles

        # Quality filter
        good = (r_kpc > 0) & (kT > 0) & (ne > 0)
        if good.sum() < 3:
            continue

        clusters.append(ClusterObservation(
            name=name, source="eROSITA_eFEDS", z=z_val,
            logM500=logM, R500_kpc=R500, r_kpc=r_kpc,
            kT=kT, kT_err=kT_err, ne=ne, ne_err=ne_err,
            selection={"flux_limit_0.5_2keV": True, "eFEDS_field": True},
        ))

    logger.info("eROSITA eFEDS: loaded %d clusters", len(clusters))
    return ObsCatalog(clusters, "eROSITA_eFEDS",
                      f"{len(clusters)} clusters, z<{z_max}, X-ray kT+ne")


def load_erosita_erass1(
    data_dir: str | Path,
    logM_min: float = 13.0,
    logM_max: float = 15.2,
    z_max: float = 1.2,
) -> ObsCatalog:
    """Load eROSITA eRASS1 stacked cluster profiles.

    Expected directory structure (data_dir/):
      erass1_stacked_profiles.fits or .csv  — Stacked T(r), n_e(r) profiles
          Required columns: mass_bin, z_bin, r_kpc, kT_keV, kT_err,
                            ne_cm3, ne_err, z_center, logM_center, n_clusters
      erass1_catalog.fits or .csv  — Optional full cluster catalog

    Stacked profiles binned by mass and redshift are the primary data
    product from the full eRASS survey.

    References: Bulbul+ 2024, Ghirardini+ 2024.
    """
    data_dir = Path(data_dir)
    clusters: List[ClusterObservation] = []

    stack_data = None
    for suffix in (".fits", ".csv"):
        stack_file = data_dir / f"erass1_stacked_profiles{suffix}"
        if stack_file.exists():
            if suffix == ".fits":
                try:
                    from astropy.io import fits
                    stack_data = fits.open(str(stack_file))[1].data
                except ImportError:
                    pass
            else:
                import csv
                with open(stack_file) as f:
                    stack_data = list(csv.DictReader(f))
            break

    if stack_data is None:
        logger.info("eROSITA eRASS1: no stacked profiles found in %s", data_dir)
        return ObsCatalog(clusters, "eROSITA_eRASS1", "no data found")

    # Group by (mass_bin, z_bin)
    stacks: Dict[Tuple[str, str], list] = {}
    for row in stack_data:
        key = (str(row.get("mass_bin", "")), str(row.get("z_bin", "")))
        if key not in stacks:
            stacks[key] = []
        stacks[key].append(row)

    for (mb, zb), rows in stacks.items():
        try:
            z_center = np.mean([float(r.get("z_center", r.get("z", 0))) for r in rows])
            logM_center = np.mean([float(r.get("logM_center", r.get("logM500", 0)))
                                   for r in rows])
            n_clusters = int(rows[0].get("n_clusters", rows[0].get("N", 1)))
        except (ValueError, TypeError):
            continue

        if z_center > z_max or not (logM_min <= logM_center <= logM_max):
            continue

        try:
            r_kpc = np.array([float(r.get("r_kpc", r.get("R_KPC", 0))) for r in rows])
            kT = np.array([float(r.get("kT_keV", r.get("KT", 0))) for r in rows])
            kT_err = np.array([float(r.get("kT_err", r.get("KT_ERR", 0))) for r in rows])
            ne = np.array([float(r.get("ne_cm3", r.get("NE", 0))) for r in rows])
            ne_err = np.array([float(r.get("ne_err", r.get("NE_ERR", 0))) for r in rows])
        except (ValueError, TypeError):
            continue

        R500 = R500_from_M500(logM_center, z_center)
        name = f"eRASS1_M{mb}_z{zb}"

        good = (r_kpc > 0) & (kT > 0) & (ne > 0)
        if good.sum() < 2:
            continue

        clusters.append(ClusterObservation(
            name=name, source="eROSITA_eRASS1", z=z_center,
            logM500=logM_center, R500_kpc=R500, r_kpc=r_kpc,
            kT=kT, kT_err=kT_err, ne=ne, ne_err=ne_err,
            selection={"erass1_flux_limit": True, "n_stacked": n_clusters},
        ))

    logger.info("eROSITA eRASS1: loaded %d cluster stacks", len(clusters))
    return ObsCatalog(clusters, "eROSITA_eRASS1",
                      f"{len(clusters)} stacked X-ray profiles")


def load_chexmate(
    data_dir: str | Path,
    logM_min: float = 13.5,
    logM_max: float = 15.0,
) -> ObsCatalog:
    """Load CHEX-MATE cluster profiles (118 Planck-selected clusters).

    Expected directory structure (data_dir/):
      chexmate_catalog.fits or .csv  — Cluster catalog
          Required columns: Name, z, M500_Msun (or logM500), R500_kpc
      profiles/  — Per-cluster profile files: {Name}_profiles.fits or .csv
          Required columns: r_kpc, kT_keV, kT_err, ne_cm3, ne_err,
                            Z_solar, Z_err

    References: CHEX-MATE Collaboration 2021, Bartalucci+ 2023.
    """
    data_dir = Path(data_dir)
    clusters: List[ClusterObservation] = []

    # Try catalog file
    cat_data = None
    for suffix in (".fits", ".csv"):
        cat_file = data_dir / f"chexmate_catalog{suffix}"
        if cat_file.exists():
            if suffix == ".fits":
                try:
                    from astropy.io import fits
                    cat_data = fits.open(str(cat_file))[1].data
                except ImportError:
                    pass
            else:
                import csv
                with open(cat_file) as f:
                    cat_data = list(csv.DictReader(f))
            break

    if cat_data is None:
        logger.info("CHEX-MATE: no catalog found in %s", data_dir)
        return ObsCatalog(clusters, "CHEX-MATE", "no data found")

    prof_dir = data_dir / "profiles"

    for row in cat_data:
        name = str(row.get("Name", row.get("name", "")))
        try:
            z_val = float(row.get("z", row.get("REDSHIFT", -1)))
        except (ValueError, TypeError):
            continue
        if z_val < 0 or z_val > 0.6:
            continue

        try:
            logM = float(row.get("logM500", 0))
        except (ValueError, TypeError):
            logM = 0
        if logM == 0:
            try:
                M500 = float(row.get("M500_Msun", row.get("M500", 0)))
                if M500 > 0:
                    logM = np.log10(M500)
            except (ValueError, TypeError):
                continue
        if not (logM_min <= logM <= logM_max):
            continue

        try:
            R500 = float(row.get("R500_kpc", R500_from_M500(logM, z_val)))
        except (ValueError, TypeError):
            R500 = R500_from_M500(logM, z_val)

        # Load per-cluster profiles
        prof_file = None
        for suffix in (".fits", ".csv"):
            candidate = prof_dir / f"{name}_profiles{suffix}"
            if candidate.exists():
                prof_file = candidate
                break
        if prof_file is None:
            continue

        try:
            if str(prof_file).endswith(".fits"):
                from astropy.io import fits
                pdata = fits.open(str(prof_file))[1].data
            else:
                import csv
                with open(prof_file) as f:
                    pdata = list(csv.DictReader(f))

            r_kpc = np.array([float(r.get("r_kpc", r.get("R_KPC", 0))) for r in pdata])
            kT = np.array([float(r.get("kT_keV", r.get("KT", 0))) for r in pdata])
            kT_err = np.array([float(r.get("kT_err", r.get("KT_ERR", 0))) for r in pdata])
            ne = np.array([float(r.get("ne_cm3", r.get("NE", 0))) for r in pdata])
            ne_err = np.array([float(r.get("ne_err", r.get("NE_ERR", 0))) for r in pdata])
            Z_arr = np.array([float(r.get("Z_solar", r.get("Z", 0))) for r in pdata])
            Z_err = np.array([float(r.get("Z_err", r.get("Z_ERR", 0))) for r in pdata])
        except Exception as e:
            logger.debug("CHEX-MATE: failed to parse %s: %s", prof_file, e)
            continue

        good = (r_kpc > 0) & (kT > 0) & (ne > 0)
        if good.sum() < 3:
            continue

        clusters.append(ClusterObservation(
            name=name, source="CHEX-MATE", z=z_val,
            logM500=logM, R500_kpc=R500, r_kpc=r_kpc,
            kT=kT, kT_err=kT_err, ne=ne, ne_err=ne_err,
            Z=Z_arr if np.any(Z_arr > 0) else None,
            Z_err=Z_err if np.any(Z_arr > 0) else None,
            selection={"planck_SZ_SNR_gt_6.5": True},
        ))

    logger.info("CHEX-MATE: loaded %d clusters", len(clusters))
    return ObsCatalog(clusters, "CHEX-MATE",
                      f"{len(clusters)} Planck-selected clusters, z<0.6, kT+ne+Z")


def load_spt_sz(
    data_dir: str | Path,
    logM_min: float = 13.8,
    logM_max: float = 15.2,
    z_max: float = 1.7,
) -> ObsCatalog:
    """Load SPT-SZ / SPT-ECS stacked Compton-y profiles.

    Expected directory structure (data_dir/):
      spt_catalog.fits or .csv  — Cluster catalog (Bocquet+ 2024)
          Required columns: Name, z, M500_Msun (or logM500, or xi for SZ S/N)
      spt_stacked_profiles.fits or .csv  — Stacked y(r) profiles
          Required columns: mass_bin, z_bin, r_arcmin, y_profile, y_err
      spt_beam.txt  — SPT beam profile (optional, for forward-model convolution)

    For stacked profiles, each stack is stored as a pseudo-cluster
    with name 'SPT_Mbin{i}_zbin{j}'.

    References: Bleem+ 2015, McDonald+ 2014, Bocquet+ 2024.
    """
    data_dir = Path(data_dir)
    clusters: List[ClusterObservation] = []

    # Try stacked profile table
    stack_data = None
    for suffix in (".fits", ".csv"):
        stack_file = data_dir / f"spt_stacked_profiles{suffix}"
        if stack_file.exists():
            if suffix == ".fits":
                try:
                    from astropy.io import fits
                    stack_data = fits.open(str(stack_file))[1].data
                except ImportError:
                    pass
            else:
                import csv
                with open(stack_file) as f:
                    stack_data = list(csv.DictReader(f))
            break

    if stack_data is None:
        logger.info("SPT-SZ: no stacked profiles found in %s", data_dir)
        return ObsCatalog(clusters, "SPT-SZ", "no data found")

    # Group by (mass_bin, z_bin)
    stacks: Dict[Tuple[str, str], list] = {}
    for row in stack_data:
        key = (str(row.get("mass_bin", "")), str(row.get("z_bin", "")))
        if key not in stacks:
            stacks[key] = []
        stacks[key].append(row)

    for (mb, zb), rows in stacks.items():
        # Parse bin center from rows or bin labels
        try:
            z_center = np.mean([float(r.get("z_center", r.get("z", 0))) for r in rows])
            logM_center = np.mean([float(r.get("logM_center", r.get("logM500", 0)))
                                   for r in rows])
            n_clusters = int(rows[0].get("n_clusters", rows[0].get("N", 1)))
        except (ValueError, TypeError):
            continue

        if z_center > z_max or not (logM_min <= logM_center <= logM_max):
            continue

        # Extract y(r) profile
        try:
            r_arcmin = np.array([float(r.get("r_arcmin", r.get("R_ARCMIN", 0)))
                                 for r in rows])
            y_prof = np.array([float(r.get("y_profile", r.get("Y", 0)))
                               for r in rows])
            y_err = np.array([float(r.get("y_err", r.get("Y_ERR", 0)))
                              for r in rows])
        except (ValueError, TypeError):
            continue

        # Convert r_arcmin to r_kpc using angular diameter distance
        # D_A(z) in Mpc, simplified flat LCDM
        from scipy.integrate import quad
        def _E_inv(zp):
            return 1.0 / np.sqrt(0.3 * (1 + zp)**3 + 0.7)
        chi, _ = quad(_E_inv, 0, z_center)
        D_A_Mpc = chi * 2997.9 / (1 + z_center)  # c/H0 * chi / (1+z)
        r_kpc = r_arcmin * (D_A_Mpc * 1000.0) * (np.pi / 180.0 / 60.0)

        R500 = R500_from_M500(logM_center, z_center)
        name = f"SPT_M{mb}_z{zb}"

        good = (r_kpc > 0) & (y_prof > 0)
        if good.sum() < 2:
            continue

        clusters.append(ClusterObservation(
            name=name, source="SPT-SZ", z=z_center,
            logM500=logM_center, R500_kpc=R500, r_kpc=r_kpc,
            compton_y=y_prof, compton_y_err=y_err,
            selection={"spt_xi_gt_4.5": True, "n_stacked": n_clusters},
        ))

    logger.info("SPT-SZ: loaded %d cluster stacks", len(clusters))
    return ObsCatalog(clusters, "SPT-SZ",
                      f"{len(clusters)} stacked tSZ profiles, z<{z_max}")


def load_act_dr6(
    data_dir: str | Path,
    logM_min: float = 13.5,
    logM_max: float = 15.2,
    z_max: float = 1.9,
) -> ObsCatalog:
    """Load ACT DR6 stacked Compton-y profiles.

    Expected directory structure (data_dir/):
      act_stacked_profiles.fits or .csv
          Required columns: mass_bin, z_bin, r_arcmin, y_profile, y_err,
                            z_center, logM_center (or logM500), n_clusters
      act_beam.txt  — ACT beam profile (optional)

    References: Hilton+ 2021, Qu+ 2024.
    """
    data_dir = Path(data_dir)
    clusters: List[ClusterObservation] = []

    stack_data = None
    for suffix in (".fits", ".csv"):
        stack_file = data_dir / f"act_stacked_profiles{suffix}"
        if stack_file.exists():
            if suffix == ".fits":
                try:
                    from astropy.io import fits
                    stack_data = fits.open(str(stack_file))[1].data
                except ImportError:
                    pass
            else:
                import csv
                with open(stack_file) as f:
                    stack_data = list(csv.DictReader(f))
            break

    if stack_data is None:
        logger.info("ACT DR6: no stacked profiles found in %s", data_dir)
        return ObsCatalog(clusters, "ACT_DR6", "no data found")

    # Group by (mass_bin, z_bin) — same logic as SPT
    stacks: Dict[Tuple[str, str], list] = {}
    for row in stack_data:
        key = (str(row.get("mass_bin", "")), str(row.get("z_bin", "")))
        if key not in stacks:
            stacks[key] = []
        stacks[key].append(row)

    for (mb, zb), rows in stacks.items():
        try:
            z_center = np.mean([float(r.get("z_center", r.get("z", 0))) for r in rows])
            logM_center = np.mean([float(r.get("logM_center", r.get("logM500", 0)))
                                   for r in rows])
            n_clusters = int(rows[0].get("n_clusters", rows[0].get("N", 1)))
        except (ValueError, TypeError):
            continue

        if z_center > z_max or not (logM_min <= logM_center <= logM_max):
            continue

        try:
            r_arcmin = np.array([float(r.get("r_arcmin", 0)) for r in rows])
            y_prof = np.array([float(r.get("y_profile", r.get("Y", 0))) for r in rows])
            y_err = np.array([float(r.get("y_err", r.get("Y_ERR", 0))) for r in rows])
        except (ValueError, TypeError):
            continue

        # Convert to kpc
        from scipy.integrate import quad
        def _E_inv(zp):
            return 1.0 / np.sqrt(0.3 * (1 + zp)**3 + 0.7)
        chi, _ = quad(_E_inv, 0, z_center)
        D_A_Mpc = chi * 2997.9 / (1 + z_center)
        r_kpc = r_arcmin * (D_A_Mpc * 1000.0) * (np.pi / 180.0 / 60.0)

        R500 = R500_from_M500(logM_center, z_center)
        name = f"ACT_M{mb}_z{zb}"

        good = (r_kpc > 0) & (y_prof > 0)
        if good.sum() < 2:
            continue

        clusters.append(ClusterObservation(
            name=name, source="ACT_DR6", z=z_center,
            logM500=logM_center, R500_kpc=R500, r_kpc=r_kpc,
            compton_y=y_prof, compton_y_err=y_err,
            selection={"act_snr_gt_4": True, "n_stacked": n_clusters},
        ))

    logger.info("ACT DR6: loaded %d cluster stacks", len(clusters))
    return ObsCatalog(clusters, "ACT_DR6",
                      f"{len(clusters)} stacked tSZ profiles, z<{z_max}")


def load_clogs_groups(
    data_dir: str | Path,
    logM_min: float = 12.5,
    logM_max: float = 14.0,
) -> ObsCatalog:
    """Load CLoGS galaxy group profiles.

    Expected directory structure (data_dir/):
      clogs_catalog.fits or .csv  — Group catalog (O'Sullivan+ 2017)
          Required columns: Name, z, logM500 (or kT for M-T relation)
      clogs_profiles.fits or .csv  — Profile table
          Required columns: Name, r_kpc, kT_keV, kT_err, ne_cm3, ne_err

    References: O'Sullivan+ 2017.
    """
    data_dir = Path(data_dir)
    clusters: List[ClusterObservation] = []

    cat_data = None
    for suffix in (".fits", ".csv"):
        cat_file = data_dir / f"clogs_catalog{suffix}"
        if cat_file.exists():
            if suffix == ".fits":
                try:
                    from astropy.io import fits
                    cat_data = fits.open(str(cat_file))[1].data
                except ImportError:
                    pass
            else:
                import csv
                with open(cat_file) as f:
                    cat_data = list(csv.DictReader(f))
            break

    if cat_data is None:
        logger.info("CLoGS: no catalog found in %s", data_dir)
        return ObsCatalog(clusters, "CLoGS", "no data found")

    # Profile lookup
    prof_data = None
    for suffix in (".fits", ".csv"):
        prof_file = data_dir / f"clogs_profiles{suffix}"
        if prof_file.exists():
            if suffix == ".csv":
                import csv
                with open(prof_file) as f:
                    prof_data = list(csv.DictReader(f))
            break

    prof_lookup: Dict[str, Dict[str, list]] = {}
    if prof_data is not None:
        for row in prof_data:
            name = str(row.get("Name", row.get("name", "")))
            if name not in prof_lookup:
                prof_lookup[name] = {"r": [], "kT": [], "kT_err": [],
                                     "ne": [], "ne_err": []}
            prof_lookup[name]["r"].append(float(row.get("r_kpc", 0)))
            prof_lookup[name]["kT"].append(float(row.get("kT_keV", row.get("kT", 0))))
            prof_lookup[name]["kT_err"].append(float(row.get("kT_err", 0)))
            prof_lookup[name]["ne"].append(float(row.get("ne_cm3", row.get("ne", 0))))
            prof_lookup[name]["ne_err"].append(float(row.get("ne_err", 0)))

    for row in cat_data:
        name = str(row.get("Name", row.get("name", "")))
        try:
            z_val = float(row.get("z", row.get("REDSHIFT", 0.02)))
        except (ValueError, TypeError):
            continue

        try:
            logM = float(row.get("logM500", 0))
        except (ValueError, TypeError):
            logM = 0
        if logM == 0:
            try:
                kT_mean = float(row.get("kT", row.get("TEMP", 0)))
                if kT_mean > 0:
                    logM = logM500_from_kT_global(kT_mean, z_val)
            except (ValueError, TypeError):
                continue
        if not (logM_min <= logM <= logM_max):
            continue

        R500 = R500_from_M500(logM, z_val)

        if name not in prof_lookup:
            continue

        p = prof_lookup[name]
        r_kpc = np.array(p["r"])
        kT = np.array(p["kT"])
        kT_err = np.array(p["kT_err"])
        ne = np.array(p["ne"])
        ne_err = np.array(p["ne_err"])

        good = (r_kpc > 0) & (kT > 0) & (ne > 0)
        if good.sum() < 2:
            continue

        clusters.append(ClusterObservation(
            name=name, source="CLoGS", z=z_val,
            logM500=logM, R500_kpc=R500, r_kpc=r_kpc,
            kT=kT, kT_err=kT_err, ne=ne, ne_err=ne_err,
            selection={"chandra_xmm_group": True},
        ))

    logger.info("CLoGS: loaded %d groups", len(clusters))
    return ObsCatalog(clusters, "CLoGS",
                      f"{len(clusters)} galaxy groups, logM {logM_min}-{logM_max}")


# ---------------------------------------------------------------------------
# Unified catalog builder
# ---------------------------------------------------------------------------

def build_unified_catalog(
    accept_xcop_npz: Optional[str | Path] = None,
    erosita_efeds_dir: Optional[str | Path] = None,
    erosita_erass1_dir: Optional[str | Path] = None,
    chexmate_dir: Optional[str | Path] = None,
    spt_dir: Optional[str | Path] = None,
    act_dir: Optional[str | Path] = None,
    clogs_dir: Optional[str | Path] = None,
    logM_min: float = 12.5,
    logM_max: float = 15.2,
    z_max: float = 2.0,
) -> ObsCatalog:
    """Combine all available survey data into a single unified catalog.

    Only loads surveys for which a path is provided and data exists.
    """
    all_clusters: List[ClusterObservation] = []

    if accept_xcop_npz and Path(accept_xcop_npz).exists():
        cat = load_accept_xcop(accept_xcop_npz, logM_min=logM_min, logM_max=logM_max)
        all_clusters.extend(cat.clusters)

    if erosita_efeds_dir and Path(erosita_efeds_dir).exists():
        cat = load_erosita_efeds(erosita_efeds_dir, logM_min=logM_min,
                                 logM_max=logM_max, z_max=z_max)
        all_clusters.extend(cat.clusters)

    if erosita_erass1_dir and Path(erosita_erass1_dir).exists():
        cat = load_erosita_erass1(erosita_erass1_dir, logM_min=logM_min,
                                  logM_max=logM_max, z_max=z_max)
        all_clusters.extend(cat.clusters)

    if chexmate_dir and Path(chexmate_dir).exists():
        cat = load_chexmate(chexmate_dir, logM_min=logM_min, logM_max=logM_max)
        all_clusters.extend(cat.clusters)

    if spt_dir and Path(spt_dir).exists():
        cat = load_spt_sz(spt_dir, logM_min=logM_min,
                          logM_max=logM_max, z_max=z_max)
        all_clusters.extend(cat.clusters)

    if act_dir and Path(act_dir).exists():
        cat = load_act_dr6(act_dir, logM_min=logM_min,
                           logM_max=logM_max, z_max=z_max)
        all_clusters.extend(cat.clusters)

    if clogs_dir and Path(clogs_dir).exists():
        cat = load_clogs_groups(clogs_dir, logM_min=logM_min, logM_max=logM_max)
        all_clusters.extend(cat.clusters)

    # Apply global mass/redshift cuts
    filtered = [c for c in all_clusters
                if logM_min <= c.logM500 <= logM_max and c.z <= z_max]

    logger.info("Unified catalog: %d clusters from %d surveys, "
                "logM [%.1f, %.1f], z [0, %.1f]",
                len(filtered),
                len({c.source for c in filtered}),
                logM_min, logM_max, z_max)

    return ObsCatalog(filtered, "unified",
                      f"{len(filtered)} clusters, multi-survey")


# ---------------------------------------------------------------------------
# Interpolation onto emulator grid
# ---------------------------------------------------------------------------

def interpolate_to_grid(
    catalog: ObsCatalog,
    ref_r_kpc: np.ndarray,
    channels: Sequence[str] = ("kT", "ne"),
    fill_value: float = np.nan,
) -> Dict[str, np.ndarray]:
    """Interpolate all cluster profiles onto a common radial grid.

    Parameters
    ----------
    catalog : ObsCatalog
    ref_r_kpc : (n_r,) reference radial grid in kpc
    channels : which profiles to interpolate
    fill_value : value for out-of-bounds

    Returns
    -------
    dict with keys:
      '{ch}_profiles' : (n_clusters, n_r) log10 profiles
      '{ch}_err'      : (n_clusters, n_r) errors (in dex for log10 profiles)
      'logM'          : (n_clusters,)
      'z'             : (n_clusters,)
      'R500_kpc'      : (n_clusters,)
      'names'         : list of cluster names
      'sources'       : list of survey names
    """
    n = catalog.n_clusters
    n_r = len(ref_r_kpc)
    log_ref = np.log10(ref_r_kpc)

    result: Dict[str, np.ndarray] = {
        "logM": np.array([c.logM500 for c in catalog.clusters]),
        "z": np.array([c.z for c in catalog.clusters]),
        "R500_kpc": np.array([c.R500_kpc for c in catalog.clusters]),
        "names": [c.name for c in catalog.clusters],
        "sources": [c.source for c in catalog.clusters],
    }

    ch_map = {
        "kT": ("kT", "kT_err"),
        "ne": ("ne", "ne_err"),
        "Z": ("Z", "Z_err"),
        "compton_y": ("compton_y", "compton_y_err"),
    }

    for ch in channels:
        if ch not in ch_map:
            raise ValueError(f"Unknown channel '{ch}'; choose from {list(ch_map)}")
        val_attr, err_attr = ch_map[ch]

        profiles = np.full((n, n_r), fill_value, dtype=np.float64)
        errors = np.full((n, n_r), fill_value, dtype=np.float64)

        for i, c in enumerate(catalog.clusters):
            val = getattr(c, val_attr)
            err = getattr(c, err_attr)
            if val is None:
                continue

            r = np.asarray(c.r_kpc, dtype=np.float64)
            val = np.asarray(val, dtype=np.float64)
            good = np.isfinite(val) & (val > 0) & np.isfinite(r) & (r > 0)

            if good.sum() < 2:
                continue

            log_r = np.log10(r[good])
            log_val = np.log10(val[good])
            profiles[i] = np.interp(log_ref, log_r, log_val,
                                    left=fill_value, right=fill_value)

            if err is not None:
                err = np.asarray(err, dtype=np.float64)
                # Convert fractional error to dex: σ_log10 ≈ err / (val * ln10)
                frac_err = np.where(good, err / np.clip(val, 1e-30, None), 0.0)
                log_err = frac_err[good] / np.log(10.0)
                errors[i] = np.interp(log_ref, log_r, log_err,
                                      left=fill_value, right=fill_value)

        result[f"{ch}_profiles"] = profiles
        result[f"{ch}_err"] = errors

    return result


# ---------------------------------------------------------------------------
# Summary statistic construction (multi-z aware)
# ---------------------------------------------------------------------------

def compute_summary_multiz(
    profiles: Dict[str, np.ndarray],
    mass_bins: Sequence[Tuple[float, float]] = ((13.2, 13.6), (13.6, 14.0),
                                                 (14.0, 14.5), (14.5, 15.0)),
    z_bins: Sequence[Tuple[float, float]] = ((0.0, 0.15), (0.15, 0.5),
                                              (0.5, 1.0)),
    channels: Sequence[str] = ("kT", "ne"),
    n_radii_per_bin: int = 5,
    max_r_by_mass: Optional[Dict[Tuple[float, float], float]] = None,
    ref_r_kpc: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a multi-z summary statistic vector from interpolated profiles.

    Bins clusters by (mass, redshift) and takes median profiles at
    adaptive radii per bin.

    Parameters
    ----------
    profiles : output of interpolate_to_grid()
    mass_bins : mass bin edges
    z_bins : redshift bin edges
    channels : profile channels to include
    n_radii_per_bin : number of radial sample points per bin
    max_r_by_mass : optional dict mapping mass_bin -> max radius in kpc
    ref_r_kpc : reference radial grid

    Returns
    -------
    summary : (n_summary,) flat summary vector
    """
    if max_r_by_mass is None:
        max_r_by_mass = {
            (13.2, 13.6): 60.0,
            (13.6, 14.0): 100.0,
            (14.0, 14.5): 300.0,
            (14.5, 15.0): 600.0,
        }

    logM = profiles["logM"]
    z_arr = profiles["z"]

    summary_parts = []

    for m_lo, m_hi in mass_bins:
        for z_lo, z_hi in z_bins:
            in_bin = ((logM >= m_lo) & (logM < m_hi) &
                      (z_arr >= z_lo) & (z_arr < z_hi))
            n_in = int(in_bin.sum())

            # Max radius for this mass bin
            max_r = max_r_by_mass.get((m_lo, m_hi), 300.0)

            # Select radial indices within range
            if ref_r_kpc is not None:
                r_mask = ref_r_kpc <= max_r
                valid_idx = np.where(r_mask)[0]
            else:
                valid_idx = np.arange(profiles[f"{channels[0]}_profiles"].shape[1])

            # Subsample to n_radii_per_bin evenly spaced points
            if len(valid_idx) > n_radii_per_bin:
                sel = np.linspace(0, len(valid_idx) - 1,
                                  n_radii_per_bin, dtype=int)
                r_idx = valid_idx[sel]
            else:
                r_idx = valid_idx
            n_r = len(r_idx)

            for ch in channels:
                prof = profiles[f"{ch}_profiles"]
                if n_in > 0 and n_r > 0:
                    bin_data = prof[in_bin][:, r_idx]
                    summary_parts.append(np.nanmedian(bin_data, axis=0))
                else:
                    summary_parts.append(np.zeros(n_r))

            # Derived: pressure = kT + ne in log space
            if "kT" in channels and "ne" in channels:
                kT_prof = profiles["kT_profiles"]
                ne_prof = profiles["ne_profiles"]
                if n_in > 0 and n_r > 0:
                    P_data = kT_prof[in_bin][:, r_idx] + ne_prof[in_bin][:, r_idx]
                    summary_parts.append(np.nanmedian(P_data, axis=0))
                else:
                    summary_parts.append(np.zeros(n_r))

            # Scalars: cluster count and median mass
            summary_parts.append(np.array([np.log1p(n_in)]))
            if n_in > 0:
                summary_parts.append(np.array([np.median(logM[in_bin])]))
                summary_parts.append(np.array([np.median(z_arr[in_bin])]))
            else:
                summary_parts.append(np.array([0.0]))
                summary_parts.append(np.array([0.0]))

    summary = np.concatenate(summary_parts)
    return np.nan_to_num(summary, nan=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Save / load unified catalog
# ---------------------------------------------------------------------------

def save_catalog_npz(catalog: ObsCatalog, path: str | Path) -> None:
    """Save a unified catalog to a compressed npz file."""
    arrays = {}
    names = []
    for i, c in enumerate(catalog.clusters):
        prefix = f"c{i:04d}"
        names.append(c.name)
        arrays[f"{prefix}__r_kpc"] = c.r_kpc
        arrays[f"{prefix}__z"] = np.array([c.z])
        arrays[f"{prefix}__logM500"] = np.array([c.logM500])
        arrays[f"{prefix}__R500_kpc"] = np.array([c.R500_kpc])
        arrays[f"{prefix}__source"] = np.array([c.source])
        for attr in ("kT", "kT_err", "ne", "ne_err", "Z", "Z_err",
                      "compton_y", "compton_y_err"):
            val = getattr(c, attr)
            if val is not None:
                arrays[f"{prefix}__{attr}"] = np.asarray(val)

    arrays["cluster_names"] = np.array(names)
    arrays["survey"] = np.array([catalog.survey])
    np.savez_compressed(str(path), **arrays)
    logger.info("Saved %d clusters to %s", len(names), path)


def load_catalog_npz(path: str | Path) -> ObsCatalog:
    """Load a unified catalog from npz."""
    data = np.load(str(path), allow_pickle=True)
    names = list(data["cluster_names"])
    survey = str(data.get("survey", ["unknown"])[0])
    clusters = []

    for i, name in enumerate(names):
        prefix = f"c{i:04d}"
        r_kpc = data[f"{prefix}__r_kpc"]
        z_val = float(data[f"{prefix}__z"][0])
        logM = float(data[f"{prefix}__logM500"][0])
        R500 = float(data[f"{prefix}__R500_kpc"][0])
        source = str(data[f"{prefix}__source"][0])

        kwargs = {}
        for attr in ("kT", "kT_err", "ne", "ne_err", "Z", "Z_err",
                      "compton_y", "compton_y_err"):
            key = f"{prefix}__{attr}"
            if key in data:
                kwargs[attr] = data[key]

        clusters.append(ClusterObservation(
            name=str(name), source=source, z=z_val,
            logM500=logM, R500_kpc=R500, r_kpc=r_kpc,
            **kwargs,
        ))

    return ObsCatalog(clusters, survey)
