#!/usr/bin/env python
"""Download publicly available cluster survey catalogs and profiles.

This script fetches data from CDS/VizieR and public archives for the
surveys used in the ANP-SBI pipeline.  It creates standardized directory
structures ready for the loaders in anp_emulator/observations.py.

Usage:
    python download_survey_data.py --output-dir ./survey_data --surveys efeds chexmate

Available surveys:
    efeds     — eROSITA eFEDS clusters (Liu+ 2022, Bahar+ 2022)
    erass1    — eROSITA eRASS1 stacked profiles (Bulbul+ 2024)
    chexmate  — CHEX-MATE clusters (CHEX-MATE 2021, Bartalucci+ 2023)
    spt       — SPT-SZ stacked profiles (McDonald+ 2014)
    act       — ACT DR6 (Hilton+ 2021)
    clogs     — CLoGS galaxy groups (O'Sullivan+ 2017)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VizieR / CDS query helpers
# ---------------------------------------------------------------------------

def _vizier_download(catalog_id: str, output_path: Path, table: str = "1",
                     columns: list[str] | None = None,
                     row_limit: int = -1) -> bool:
    """Download a VizieR catalog table as CSV via TAP query.

    Parameters
    ----------
    catalog_id : VizieR catalog ID, e.g. 'J/A+A/661/A7'
    output_path : where to save the file
    table : table number within the catalog
    columns : list of column names to fetch (None = all, but prefer specifying!)
    row_limit : max rows (-1 for unlimited)
    """
    try:
        from astroquery.vizier import Vizier
        v = Vizier(catalog=catalog_id, row_limit=row_limit,
                   columns=columns if columns else ["**"])
        tables = v.get_catalogs(catalog_id)
        if len(tables) == 0:
            logger.warning("No tables found for %s", catalog_id)
            return False
        idx = int(table) - 1 if table.isdigit() else 0
        if idx >= len(tables):
            idx = 0
        tab = tables[idx]
        tab.write(str(output_path), format="csv", overwrite=True)
        logger.info("Downloaded %s -> %s (%d rows)", catalog_id, output_path, len(tab))
        return True
    except ImportError:
        logger.warning("astroquery not installed; falling back to URL download")
        return _vizier_url_download(catalog_id, output_path, table, columns)
    except Exception as e:
        logger.error("VizieR download failed for %s: %s", catalog_id, e)
        return False


def _vizier_url_download(catalog_id: str, output_path: Path,
                         table: str = "1",
                         columns: list[str] | None = None) -> bool:
    """Fall back to direct URL download via CDS TAP service with ADQL."""
    import urllib.request
    import urllib.parse

    # Build the TAP table name: J/A+A/661/A2 table 1 → "J/A+A/661/A2/1"
    tap_table = f'"{catalog_id}/{table}"'
    col_list = ", ".join(columns) if columns else "*"
    query = f"SELECT {col_list} FROM {tap_table}"

    params = urllib.parse.urlencode({
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "QUERY": query,
    })
    url = f"https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync?{params}"

    try:
        urllib.request.urlretrieve(url, str(output_path))
        size_mb = output_path.stat().st_size / 1e6
        logger.info("Downloaded %s -> %s (%.1f MB, TAP)", catalog_id, output_path,
                     size_mb)
        return True
    except Exception as e:
        logger.error("TAP download failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Per-survey download functions
# ---------------------------------------------------------------------------

def download_efeds(output_dir: Path) -> None:
    """Download eROSITA eFEDS cluster catalog and profiles.

    Sources:
      - Cluster catalog: Liu+ 2022 (VizieR J/A+A/661/A2)
      - Temperature profiles: Bahar+ 2022 (VizieR J/A+A/661/A7)
    """
    efeds_dir = output_dir / "efeds"
    efeds_dir.mkdir(parents=True, exist_ok=True)

    # Only the columns needed for the SBI pipeline
    cluster_cols = ["Name", "RAJ2000", "DEJ2000", "z", "M500", "e_M500",
                    "R500", "L500", "T500", "e_T500", "lambda"]
    profile_cols = ["Name", "Rin", "Rout", "kT", "e_kT", "Norm",
                    "ne", "e_ne", "Z", "e_Z"]

    logger.info("Downloading eFEDS cluster catalog (Liu+ 2022)...")
    _vizier_download("J/A+A/661/A2", efeds_dir / "efeds_clusters.csv",
                     columns=cluster_cols)

    logger.info("Downloading eFEDS temperature profiles (Bahar+ 2022)...")
    _vizier_download("J/A+A/661/A7", efeds_dir / "efeds_profiles.csv",
                     columns=profile_cols)

    print(f"  eFEDS data saved to {efeds_dir}")


def download_erass1(output_dir: Path) -> None:
    """Download eROSITA eRASS1 cluster catalogs.

    Sources:
      - Cluster catalog: Bulbul+ 2024 (VizieR J/A+A/685/A106)
      - Stacked profiles: Ghirardini+ 2024
    """
    erass1_dir = output_dir / "erass1"
    erass1_dir.mkdir(parents=True, exist_ok=True)

    catalog_cols = ["Name", "RAJ2000", "DEJ2000", "z", "M500", "e_M500",
                    "R500", "T500", "e_T500", "L500"]

    logger.info("Downloading eRASS1 cluster catalog (Bulbul+ 2024)...")
    _vizier_download("J/A+A/685/A106", erass1_dir / "erass1_catalog.csv",
                     columns=catalog_cols)

    # Stacked profiles may be in supplementary material
    logger.info("Downloading eRASS1 stacked profiles (Ghirardini+ 2024)...")
    _vizier_download("J/A+A/689/A298", erass1_dir / "erass1_stacked_profiles.csv",
                     table="2")

    print(f"  eRASS1 data saved to {erass1_dir}")


def download_chexmate(output_dir: Path) -> None:
    """Download CHEX-MATE cluster catalog and profiles.

    Sources:
      - Cluster catalog: CHEX-MATE Collaboration 2021 (VizieR J/A+A/650/A104)
      - Profiles: Bartalucci+ 2023 or individual XMM data products
    """
    chex_dir = output_dir / "chexmate"
    chex_dir.mkdir(parents=True, exist_ok=True)
    (chex_dir / "profiles").mkdir(exist_ok=True)

    catalog_cols = ["Name", "RAJ2000", "DEJ2000", "z", "M500", "e_M500",
                    "R500", "T500", "e_T500", "Morph"]

    logger.info("Downloading CHEX-MATE catalog...")
    _vizier_download("J/A+A/650/A104", chex_dir / "chexmate_catalog.csv",
                     columns=catalog_cols)

    print(f"  CHEX-MATE data saved to {chex_dir}")
    print("  NOTE: Per-cluster profile FITS files must be obtained from")
    print("  the CHEX-MATE data release or XMM Science Archive.")
    print(f"  Place them in {chex_dir / 'profiles'}/{{Name}}_profiles.csv")


def download_spt(output_dir: Path) -> None:
    """Download SPT-SZ cluster catalog.

    Sources:
      - Cluster catalog: Bleem+ 2015 (VizieR J/ApJS/216/27)
      - Stacked profiles: McDonald+ 2014 (manual download)
    """
    spt_dir = output_dir / "spt"
    spt_dir.mkdir(parents=True, exist_ok=True)

    catalog_cols = ["Name", "RAJ2000", "DEJ2000", "z", "M500", "e_M500",
                    "xi", "theta"]

    logger.info("Downloading SPT-SZ cluster catalog (Bleem+ 2015)...")
    _vizier_download("J/ApJS/216/27", spt_dir / "spt_catalog.csv",
                     columns=catalog_cols)

    print(f"  SPT data saved to {spt_dir}")
    print("  NOTE: Stacked Compton-y profiles must be obtained from")
    print("  McDonald+ 2014 Table 2 or the SPT data products page.")
    print(f"  Save as {spt_dir / 'spt_stacked_profiles.csv'}")


def download_act(output_dir: Path) -> None:
    """Download ACT DR6 cluster catalog.

    Sources:
      - Cluster catalog: Hilton+ 2021 (VizieR J/ApJS/253/3)
      - Stacked profiles: Qu+ 2024
    """
    act_dir = output_dir / "act"
    act_dir.mkdir(parents=True, exist_ok=True)

    catalog_cols = ["Name", "RAJ2000", "DEJ2000", "z", "M500", "e_M500",
                    "SNR", "y0"]

    logger.info("Downloading ACT DR6 cluster catalog (Hilton+ 2021)...")
    _vizier_download("J/ApJS/253/3", act_dir / "act_catalog.csv",
                     columns=catalog_cols)

    print(f"  ACT data saved to {act_dir}")
    print("  NOTE: Stacked tSZ profiles from Qu+ 2024 require manual download.")
    print(f"  Save as {act_dir / 'act_stacked_profiles.csv'}")


def download_clogs(output_dir: Path) -> None:
    """Download CLoGS galaxy group data.

    Sources:
      - O'Sullivan+ 2017 (VizieR J/A+A/600/A127)
    """
    clogs_dir = output_dir / "clogs"
    clogs_dir.mkdir(parents=True, exist_ok=True)

    catalog_cols = ["Name", "RAJ2000", "DEJ2000", "z", "kT", "e_kT",
                    "LX", "e_LX"]

    logger.info("Downloading CLoGS catalog (O'Sullivan+ 2017)...")
    _vizier_download("J/A+A/600/A127", clogs_dir / "clogs_catalog.csv",
                     columns=catalog_cols)

    print(f"  CLoGS data saved to {clogs_dir}")
    print("  NOTE: Per-group T(r), n_e(r) profiles require manual extraction")
    print("  from O'Sullivan+ 2017 or Chandra/XMM archives.")
    print(f"  Save as {clogs_dir / 'clogs_profiles.csv'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SURVEY_MAP = {
    "efeds": download_efeds,
    "erass1": download_erass1,
    "chexmate": download_chexmate,
    "spt": download_spt,
    "act": download_act,
    "clogs": download_clogs,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download cluster survey data for the ANP-SBI pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output-dir", type=str, default="./survey_data",
                        help="Root output directory")
    parser.add_argument("--surveys", nargs="+", default=list(SURVEY_MAP.keys()),
                        choices=list(SURVEY_MAP.keys()),
                        help="Which surveys to download (default: all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading survey data to: {output_dir.resolve()}\n")

    for survey in args.surveys:
        print(f"--- {survey.upper()} ---")
        try:
            SURVEY_MAP[survey](output_dir)
        except Exception as e:
            logger.error("Failed to download %s: %s", survey, e)
        print()

    print("\nDone. To use in the SBI pipeline:")
    print("  from anp_emulator.observations import build_unified_catalog")
    print(f"  cat = build_unified_catalog(erosita_efeds_dir='{output_dir}/efeds', ...)")


if __name__ == "__main__":
    main()
