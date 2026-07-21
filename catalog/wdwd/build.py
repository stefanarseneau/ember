"""Build analysis-ready parquet catalogs for the WD+WD wide binary sample.

Full pipeline (run in order):

  1. stitch   — merges thick/thin/mixed SED fit parquets, assigns best atmosphere
                model per star, measures cooling and total ages → data/ages.pqt
  2. elbadry  — downloads El-Badry+2021 wide binary catalog from zenodo and
                caches a compact WD-pair index → data/elbadry_wdpairs.pqt
  3. pairs    — builds wdwd_pairs.pqt from tyler_wdwd.csv (base) + ages.pqt
                (XP parameters with _xp suffix) + El-Badry R_chance_align.
                Applies photometry bitmask per component, enforces M1 >= M2.

Usage:
    python -m catalog.wdwd.build            # full pipeline
    python -m catalog.wdwd.build --stitch   # stitch step only
    python -m catalog.wdwd.build --elbadry  # El-Badry download/cache only
    python -m catalog.wdwd.build --pairs    # pair-building step only
"""

import gzip
import io
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR, WDWD_STITCH_DIR, ELBADRY_URL, ELBADRY_CACHE
from .stitch import main as _stitch_main
from ..bitmasks import _make_photometry_bitmask


_TYLER_WDWD = DATA_DIR / "tyler/tyler_wdwd.csv"

# Columns to load from tyler_wdwd.csv (per component)
_TYLER_COLS = [
    "System_Name", "source_id",
    # Gaia
    "ra", "dec", "parallax", "parallax_error",
    "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "bp_rp", "ruwe",
    # Binary pair-level (same for both components; kept only from A)
    "min_sep", "wtd_par", "e_wtd_par",
    # SDSS
    "u_sdss", "g_sdss", "r_sdss", "i_sdss", "z_sdss",
    "err_u_sdss", "err_g_sdss", "err_r_sdss", "err_i_sdss", "err_z_sdss",
    "flags_u_sdss", "flags_g_sdss", "flags_r_sdss", "flags_i_sdss", "flags_z_sdss",
    # PanSTARRS
    "g_mean_psf_mag_pstarr", "r_mean_psf_mag_pstarr", "i_mean_psf_mag_pstarr",
    "z_mean_psf_mag_pstarr", "y_mean_psf_mag_pstarr",
    "g_mean_psf_mag_error_pstarr", "r_mean_psf_mag_error_pstarr",
    "i_mean_psf_mag_error_pstarr", "z_mean_psf_mag_error_pstarr",
    "y_mean_psf_mag_error_pstarr",
    "g_flags_pstarr", "r_flags_pstarr", "i_flags_pstarr",
    "z_flags_pstarr", "y_flags_pstarr",
    # 2MASS
    "j_m_tmass", "h_m_tmass", "ks_m_tmass",
    "j_msigcom_tmass", "h_msigcom_tmass", "ks_msigcom_tmass",
    # SkyMapper
    "u_psf_smap", "v_psf_smap", "g_psf_smap", "r_psf_smap", "i_psf_smap", "z_psf_smap",
    "e_u_psf_smap", "e_v_psf_smap", "e_g_psf_smap", "e_r_psf_smap",
    "e_i_psf_smap", "e_z_psf_smap",
    "u_flags_smap", "v_flags_smap", "g_flags_smap", "r_flags_smap",
    "i_flags_smap", "z_flags_smap",
    "u_nimaflags_smap", "v_nimaflags_smap", "g_nimaflags_smap", "r_nimaflags_smap",
    "i_nimaflags_smap", "z_nimaflags_smap",
    # H-atmosphere SED fit (Tyler)
    "TeffH", "e_TeffH_lower", "e_TeffH_upper",
    "LoggH", "e_LoggH_lower", "e_LoggH_upper",
    "MassH", "e_MassH_lower", "e_MassH_upper",
    "cool_ageH", "e_cool_ageH_lower", "e_cool_ageH_upper", "Chi2H",
    # IFMR ages — Cummings 2018
    "init_mass_error_lower_Cummings2018", "init_mass_Cummings2018",
    "init_mass_error_upper_Cummings2018",
    "mslife_error_lower_Cummings2018", "mslife_Cummings2018",
    "mslife_error_upper_Cummings2018",
    "tot_age_error_lower_Cummings2018", "tot_age_Cummings2018",
    "tot_age_error_upper_Cummings2018", "ifmr_flag_Cummings2018",
    # IFMR ages — MESA IFMR
    "init_mass_error_lower_MESA_IFMR", "init_mass_MESA_IFMR",
    "init_mass_error_upper_MESA_IFMR",
    "mslife_error_lower_MESA_IFMR", "mslife_MESA_IFMR",
    "mslife_error_upper_MESA_IFMR",
    "tot_age_error_lower_MESA_IFMR", "tot_age_MESA_IFMR",
    "tot_age_error_upper_MESA_IFMR", "ifmr_flag_MESA_IFMR",
]

# Zero is the missing-data sentinel for photometry — convert to NaN
_PHOT_ZERO_COLS = [
    "u_sdss", "g_sdss", "r_sdss", "i_sdss", "z_sdss",
    "err_u_sdss", "err_g_sdss", "err_r_sdss", "err_i_sdss", "err_z_sdss",
    "g_mean_psf_mag_pstarr", "r_mean_psf_mag_pstarr", "i_mean_psf_mag_pstarr",
    "z_mean_psf_mag_pstarr", "y_mean_psf_mag_pstarr",
    "g_mean_psf_mag_error_pstarr", "r_mean_psf_mag_error_pstarr",
    "i_mean_psf_mag_error_pstarr", "z_mean_psf_mag_error_pstarr",
    "y_mean_psf_mag_error_pstarr",
    "j_m_tmass", "h_m_tmass", "ks_m_tmass",
    "j_msigcom_tmass", "h_msigcom_tmass", "ks_msigcom_tmass",
    "u_psf_smap", "v_psf_smap", "g_psf_smap", "r_psf_smap", "i_psf_smap", "z_psf_smap",
    "e_u_psf_smap", "e_v_psf_smap", "e_g_psf_smap", "e_r_psf_smap",
    "e_i_psf_smap", "e_z_psf_smap",
]

# Rename tyler_wdwd.csv columns to match _make_photometry_bitmask expectations
_BITMASK_RENAME = {
    # SDSS: strip _sdss suffix
    "u_sdss": "u", "g_sdss": "g", "r_sdss": "r", "i_sdss": "i", "z_sdss": "z",
    "flags_u_sdss": "flags_u", "flags_g_sdss": "flags_g", "flags_r_sdss": "flags_r",
    "flags_i_sdss": "flags_i", "flags_z_sdss": "flags_z",
    # PanSTARRS: strip _pstarr suffix from mags; rename flag columns
    "g_mean_psf_mag_pstarr": "g_mean_psf_mag", "r_mean_psf_mag_pstarr": "r_mean_psf_mag",
    "i_mean_psf_mag_pstarr": "i_mean_psf_mag", "z_mean_psf_mag_pstarr": "z_mean_psf_mag",
    "y_mean_psf_mag_pstarr": "y_mean_psf_mag",
    "g_flags_pstarr": "g_flags_", "r_flags_pstarr": "r_flags_",
    "i_flags_pstarr": "i_flags_", "z_flags_pstarr": "z_flags_", "y_flags_pstarr": "y_flags",
    # SkyMapper: strip _smap from mags; u/v flags need renaming (g/r/i/z already match)
    "u_psf_smap": "u_psf", "v_psf_smap": "v_psf", "g_psf_smap": "g_psf",
    "r_psf_smap": "r_psf", "i_psf_smap": "i_psf", "z_psf_smap": "z_psf",
    "u_flags_smap": "u_flags", "v_flags_smap": "v_flags",
    "u_nimaflags_smap": "u_nimaflags", "v_nimaflags_smap": "v_nimaflags",
    "g_nimaflags_smap": "g_nimaflags", "r_nimaflags_smap": "r_nimaflags",
    "i_nimaflags_smap": "i_nimaflags", "z_nimaflags_smap": "z_nimaflags",
}

# Columns to load from ages.pqt (XP spectrum fits)
_XP_COLS = [
    "sourceid",
    "teff_best", "std_tt_best",
    "radius_best", "std_rr_best",
    "dist_best", "std_dist_best",
    "av_best", "std_av_best",
    "mass", "mass_hi", "mass_lo",
    "log_age_cool", "log_age_cool_hi", "log_age_cool_lo",
    "log_age", "log_age_hi", "log_age_lo",
    "SpType", "pop_type",
    "PDA", "PDB", "PDC", "PDO", "PDQ", "PDZ",
]

_XP_RENAME = {
    "teff_best":       "teff_xp",
    "std_tt_best":     "e_teff_xp",
    "radius_best":     "radius_xp",
    "std_rr_best":     "e_radius_xp",
    "dist_best":       "dist_xp",
    "std_dist_best":   "e_dist_xp",
    "av_best":         "av_xp",
    "std_av_best":     "e_av_xp",
    "mass":            "mass_xp",
    "mass_hi":         "mass_hi_xp",
    "mass_lo":         "mass_lo_xp",
    "log_age_cool":    "log_age_cool_xp",
    "log_age_cool_hi": "log_age_cool_hi_xp",
    "log_age_cool_lo": "log_age_cool_lo_xp",
    "log_age":         "log_age_xp",
    "log_age_hi":      "log_age_hi_xp",
    "log_age_lo":      "log_age_lo_xp",
}

# Pair-level columns identical for both components; kept once (from component A)
_PAIR_LEVEL_COLS = ["min_sep", "wtd_par", "e_wtd_par"]


# ── Step 1: stitch ────────────────────────────────────────────────────────

def stitch(correct_ages: bool = False) -> None:
    """Run the stitch pipeline: thick/thin/mixed parquets → data/ages.pqt."""
    _stitch_main(inpath=WDWD_STITCH_DIR, outpath=DATA_DIR / "ages.pqt",
                 correct_ages=correct_ages)


# ── Step 2: El-Badry cache ────────────────────────────────────────────────

def build_elbadry() -> None:
    """Download El-Badry+2021 and cache a compact WD-pair index.

    The full catalog is ~1.8 M rows; we keep only WD+WD and WD+MS rows with
    four columns: source_id1, source_id2, binary_type, R_chance_align.
    Skips the download if the cache already exists.
    """
    if ELBADRY_CACHE.exists():
        print(f"{ELBADRY_CACHE.name} already exists, skipping download.")
        return

    from astropy.table import Table

    print("Downloading El-Badry+2021 wide binary catalog (~200 MB)…")
    with urllib.request.urlopen(ELBADRY_URL) as resp:
        raw = resp.read()

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        with gzip.open(io.BytesIO(raw)) as gz:
            f.write(gz.read())
        tmp_path = f.name

    print("  Extracting WD pairs…")
    full = Table.read(tmp_path).to_pandas()
    Path(tmp_path).unlink()

    wd_types = {b"WDWD", b"WDMS"}
    wd_pairs = (
        full.loc[full["binary_type"].isin(wd_types),
                 ["source_id1", "source_id2", "binary_type", "R_chance_align"]]
        .copy()
    )
    wd_pairs["binary_type"] = wd_pairs["binary_type"].str.decode("utf-8").str.strip()

    wd_pairs.to_parquet(ELBADRY_CACHE, index=False)
    print(f"  Saved {ELBADRY_CACHE}  ({len(wd_pairs)} WD pairs from {len(full)} total)")


# ── Step 3: pair table ────────────────────────────────────────────────────

def build_pairs() -> None:
    """Build wdwd_pairs.pqt from tyler_wdwd.csv, ages.pqt, and the El-Badry index.

    Each row is one WD+WD pair with per-component columns suffixed _1 / _2.
    Component 1 is the more massive WD (M1 >= M2, Heintz+2024 convention).
    XP-spectrum parameters carry the _xp suffix; H-atmosphere SED parameters
    retain Tyler's naming (TeffH, MassH, cool_ageH, …).
    """
    ages_path = DATA_DIR / "ages.pqt"
    for path in (_TYLER_WDWD, ages_path, ELBADRY_CACHE):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run prerequisite steps first")

    # ── 1. Load Tyler per-component data ─────────────────────────────────
    tyler = pd.read_csv(_TYLER_WDWD, usecols=_TYLER_COLS, low_memory=False)
    print(f"Loaded {len(tyler)} WD components from {_TYLER_WDWD.name}")

    # Zero is the missing-data sentinel for photometry — convert to NaN
    for col in _PHOT_ZERO_COLS:
        tyler[col] = tyler[col].replace(0.0, np.nan)

    # ── 2. Photometry bitmask per component ───────────────────────────────
    tyler["phot_flag"] = _make_photometry_bitmask(tyler.rename(columns=_BITMASK_RENAME))

    # ── 3. Join XP parameters from ages.pqt ──────────────────────────────
    ages = pd.read_parquet(ages_path, columns=_XP_COLS).rename(columns=_XP_RENAME)
    tyler = (
        tyler
        .merge(ages, left_on="source_id", right_on="sourceid", how="left")
        .drop(columns="sourceid")
    )
    print(f"  XP matches: {tyler['teff_xp'].notna().sum()}/{len(tyler)}")

    # ── 4. Pair A and B components ────────────────────────────────────────
    tyler["_pair_key"] = tyler["System_Name"].str[:-1]  # strip trailing A/B
    comp_a = tyler[tyler["System_Name"].str.endswith("A")].copy()
    # Drop pair-level cols from B so they don't get a _2 suffix in the merged result
    comp_b = tyler[tyler["System_Name"].str.endswith("B")].drop(
        columns=_PAIR_LEVEL_COLS
    ).copy()

    pairs = comp_a.merge(comp_b, on="_pair_key", suffixes=("_1", "_2"), how="inner")
    pairs = pairs.drop(columns=["_pair_key", "System_Name_2"])
    pairs = pairs.rename(columns={"System_Name_1": "System_Name"})
    print(f"  {len(pairs)} pairs after A/B grouping")

    # ── 5. Add R_chance_align from El-Badry ──────────────────────────────
    eb = pd.read_parquet(
        ELBADRY_CACHE,
        columns=["source_id1", "source_id2", "R_chance_align", "binary_type"],
    )
    eb_wdwd = eb[eb["binary_type"] == "WDWD"][
        ["source_id1", "source_id2", "R_chance_align"]
    ]

    pairs = pairs.merge(
        eb_wdwd.rename(columns={"source_id1": "source_id_1", "source_id2": "source_id_2"}),
        on=["source_id_1", "source_id_2"], how="left",
    )
    # Try reverse component order for pairs not matched in the first pass
    unmatched = pairs["R_chance_align"].isna()
    if unmatched.any():
        rev = pairs.loc[unmatched, ["source_id_1", "source_id_2"]].merge(
            eb_wdwd.rename(columns={"source_id1": "source_id_2", "source_id2": "source_id_1"}),
            on=["source_id_1", "source_id_2"], how="left",
        )
        pairs.loc[unmatched, "R_chance_align"] = rev["R_chance_align"].values
    print(f"  R_chance_align matched: {pairs['R_chance_align'].notna().sum()}/{len(pairs)}")

    # ── 6. Enforce M1 >= M2 (XP mass preferred; fallback to MassH) ───────
    m1 = pairs["mass_xp_1"].where(pairs["mass_xp_1"].notna(), pairs["MassH_1"])
    m2 = pairs["mass_xp_2"].where(pairs["mass_xp_2"].notna(), pairs["MassH_2"])
    needs_swap = (m2 > m1).fillna(False)

    for c1 in [c for c in pairs.columns if c.endswith("_1")]:
        c2 = c1[:-2] + "_2"
        if c2 in pairs.columns:
            tmp = pairs.loc[needs_swap, c1].copy()
            pairs.loc[needs_swap, c1] = pairs.loc[needs_swap, c2]
            pairs.loc[needs_swap, c2] = tmp
    print(f"  Pairs swapped to enforce M1 >= M2: {needs_swap.sum()}")

    # ── 7. Pre-compute differences (XP) ──────────────────────────────────
    def _log_to_gyr(x):
        return 10.0 ** (x - 9.0)

    for i in ("1", "2"):
        pairs[f"e_mass_xp_{i}"] = (
            pairs[f"mass_hi_xp_{i}"] - pairs[f"mass_lo_xp_{i}"]
        ) / 2
        for base in ("log_age_cool", "log_age"):
            hi = _log_to_gyr(pairs[f"{base}_hi_xp_{i}"])
            lo = _log_to_gyr(pairs[f"{base}_lo_xp_{i}"])
            pairs[f"t_{base}_xp_{i}"]   = _log_to_gyr(pairs[f"{base}_xp_{i}"])
            pairs[f"e_t_{base}_xp_{i}"] = (hi - lo) / 2

    pairs["dM_xp"]   = pairs["mass_xp_1"] - pairs["mass_xp_2"]
    pairs["e_dM_xp"] = np.sqrt(pairs["e_mass_xp_1"] ** 2 + pairs["e_mass_xp_2"] ** 2)

    for base in ("log_age_cool", "log_age"):
        pairs[f"dt_{base}_xp"]   = pairs[f"t_{base}_xp_1"] - pairs[f"t_{base}_xp_2"]
        pairs[f"e_dt_{base}_xp"] = np.sqrt(
            pairs[f"e_t_{base}_xp_1"] ** 2 + pairs[f"e_t_{base}_xp_2"] ** 2
        )

    # ── 8. Pre-compute differences (H-atmosphere) ─────────────────────────
    for i in ("1", "2"):
        pairs[f"e_MassH_{i}"]     = (
            pairs[f"e_MassH_upper_{i}"] + pairs[f"e_MassH_lower_{i}"]
        ) / 2
        pairs[f"e_cool_ageH_{i}"] = (
            pairs[f"e_cool_ageH_upper_{i}"] + pairs[f"e_cool_ageH_lower_{i}"]
        ) / 2
        for ifmr in ("Cummings2018", "MESA_IFMR"):
            pairs[f"e_tot_age_{ifmr}_{i}"] = (
                pairs[f"tot_age_error_upper_{ifmr}_{i}"]
                + pairs[f"tot_age_error_lower_{ifmr}_{i}"]
            ) / 2

    pairs["dM_H"]           = pairs["MassH_1"]    - pairs["MassH_2"]
    pairs["e_dM_H"]         = np.sqrt(pairs["e_MassH_1"] ** 2 + pairs["e_MassH_2"] ** 2)
    pairs["dt_cool_ageH"]   = pairs["cool_ageH_1"] - pairs["cool_ageH_2"]
    pairs["e_dt_cool_ageH"] = np.sqrt(
        pairs["e_cool_ageH_1"] ** 2 + pairs["e_cool_ageH_2"] ** 2
    )
    for ifmr in ("Cummings2018", "MESA_IFMR"):
        pairs[f"dt_tot_age_{ifmr}"]   = (
            pairs[f"tot_age_{ifmr}_1"] - pairs[f"tot_age_{ifmr}_2"]
        )
        pairs[f"e_dt_tot_age_{ifmr}"] = np.sqrt(
            pairs[f"e_tot_age_{ifmr}_1"] ** 2 + pairs[f"e_tot_age_{ifmr}_2"] ** 2
        )

    # ── 9. Save ───────────────────────────────────────────────────────────
    out          = DATA_DIR / "wdwd_pairs.pqt"
    n_xp_valid   = pairs["dt_log_age_xp"].notna().sum()
    n_mesa_valid = pairs["dt_tot_age_MESA_IFMR"].notna().sum()
    pairs.to_parquet(out, index=False)
    print(
        f"Saved {out}  ({len(pairs)} pairs, "
        f"{n_xp_valid} with XP total ages, {n_mesa_valid} with MESA IFMR total ages)"
    )


# ── Full pipeline ─────────────────────────────────────────────────────────

def main(correct_ages: bool = False) -> None:
    print("=== Step 1/3: stitch ===")
    stitch(correct_ages=correct_ages)
    print("\n=== Step 2/3: El-Badry cache ===")
    build_elbadry()
    print("\n=== Step 3/3: build pairs ===")
    build_pairs()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stitch",  action="store_true", help="Run stitch step only")
    group.add_argument("--elbadry", action="store_true", help="Run El-Badry download only")
    group.add_argument("--pairs",   action="store_true", help="Run pair-building step only")
    parser.add_argument("--correct-ages", action="store_true",
                        help="Re-measure cool-WD ages using corrected Mass/Teff")
    args = parser.parse_args()

    if args.stitch:
        stitch(correct_ages=args.correct_ages)
    elif args.elbadry:
        build_elbadry()
    elif args.pairs:
        build_pairs()
    else:
        main(correct_ages=args.correct_ages)
