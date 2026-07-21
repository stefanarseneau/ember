"""Classify WD+MS companions as main-sequence, subgiant, or giant.

Two disjoint, fully hardcoded tests in (ms_bp_rp, M_G), split at
GIANT_CMD_EVOLVED_MIN_BP_RP — no isochrones, no fitting. See config.py for
the exact numbers and reasoning:

  subgiant — bp_rp < GIANT_CMD_EVOLVED_MIN_BP_RP and above a straight line
    through (0,0) and (1, GIANT_CMD_SG_LINE_AT_1).

  giant — bp_rp >= GIANT_CMD_EVOLVED_MIN_BP_RP and
    M_G < GIANT_CMD_GIANT_ABS_MAG_MAX.

build_mist_grid_cache()/load_full_isochrone_tracks() below generate and
read the MIST reference grid — used only for plotting context in
catalog.analysis.ms_evol_cmd, not by classify() itself.
"""

import numpy as np
import pandas as pd

from ..config import (
    GIANT_CMD_AGE_GRID_GYR,
    GIANT_CMD_SG_LINE_AT_1,
    GIANT_CMD_EVOLVED_MIN_BP_RP,
    GIANT_CMD_GIANT_ABS_MAG_MAX,
    MIST_EVOL_GRID_CACHE,
)


# ── MIST reference grid (cached; plotting context only) ────────────────────

def build_mist_grid_cache(feh: float) -> None:
    """Generate and cache each age's complete isochrone track (phase
    0,2,3,4,5, untruncated, original EEP order preserved via `eep`), for
    catalog.analysis.ms_evol_cmd to plot as visual context. Writes
    [age_gyr, phase, bp_rp, G_mag, eep] to MIST_EVOL_GRID_CACHE. Skips the
    (slow, ~1 min) MIST calls if the cache already exists.
    """
    if MIST_EVOL_GRID_CACHE.exists():
        return

    from isochrones.mist import MIST_Isochrone

    mist = MIST_Isochrone()
    rows = []
    for age_gyr in GIANT_CMD_AGE_GRID_GYR:
        iso = mist.isochrone(np.log10(age_gyr * 1e9), feh=feh)
        iso["bp_rp"] = iso["BP_mag"] - iso["RP_mag"]
        track = iso[iso["phase"].isin([0, 2, 3, 4, 5])].copy()
        track["age_gyr"] = age_gyr
        track["eep"] = np.arange(len(track))
        rows.append(track[["age_gyr", "phase", "bp_rp", "G_mag", "eep"]])

    grid = pd.concat(rows, ignore_index=True)
    MIST_EVOL_GRID_CACHE.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(MIST_EVOL_GRID_CACHE, index=False)
    print(f"Saved {MIST_EVOL_GRID_CACHE}  ({len(grid)} isochrone points, "
          f"{len(GIANT_CMD_AGE_GRID_GYR)} ages, [Fe/H]={feh:+.2f})")


def load_full_isochrone_tracks() -> pd.DataFrame:
    """Read the cached full isochrone tracks (see build_mist_grid_cache)."""
    return pd.read_parquet(MIST_EVOL_GRID_CACHE)


# ── Top-level classification ────────────────────────────────────────────────

def companion_abs_g(df: pd.DataFrame) -> np.ndarray:
    """Companion absolute G magnitude from ms_phot_g_mean_mag + wtd_par/parallax."""
    parallax = (
        df["wtd_par"].where(df["wtd_par"] > 0, df["parallax"]).to_numpy(dtype=float)
    )
    g_mag = df["ms_phot_g_mean_mag"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return g_mag + 5 * np.log10(parallax) - 10


def classify(df: pd.DataFrame) -> pd.Series:
    """Classify WD+MS companions: "MS", "SG" (subgiant), "RG" (giant), NaN = unclassifiable.

    Requires ms_bp_rp, ms_phot_g_mean_mag, wtd_par/parallax on `df`.
    """
    bp_rp = df["ms_bp_rp"].to_numpy(dtype=float)
    parallax = (
        df["wtd_par"].where(df["wtd_par"] > 0, df["parallax"]).to_numpy(dtype=float)
    )
    abs_g = companion_abs_g(df)

    has_photometry = (
        np.isfinite(bp_rp) & np.isfinite(abs_g) & np.isfinite(parallax) & (parallax > 0)
    )
    is_blue = bp_rp < GIANT_CMD_EVOLVED_MIN_BP_RP

    subgiant = has_photometry & is_blue & (abs_g < GIANT_CMD_SG_LINE_AT_1 * bp_rp)
    giant = has_photometry & ~is_blue & (abs_g < GIANT_CMD_GIANT_ABS_MAG_MAX)

    code = np.full(len(df), None, dtype=object)
    code[has_photometry] = "MS"
    code[subgiant] = "SG"
    code[giant] = "RG"

    n_total = len(df)
    n_ms  = int((code == "MS").sum())
    n_sub = int((code == "SG").sum())
    n_gnt = int((code == "RG").sum())
    n_unclass = n_total - n_ms - n_sub - n_gnt

    def _pct(n):
        return 100 * n / n_total if n_total else 0.0

    print("MS-companion evolutionary state:")
    print(f"  {'MS':30s} {n_ms:6d}  ({_pct(n_ms):5.1f}%)")
    print(f"  {'SG (subgiant)':30s} {n_sub:6d}  ({_pct(n_sub):5.1f}%)")
    print(f"  {'RG (giant)':30s} {n_gnt:6d}  ({_pct(n_gnt):5.1f}%)")
    print(f"  {'unclassifiable (no color/plx)':30s} {n_unclass:6d}  ({_pct(n_unclass):5.1f}%)")
    print(f"  {'total':30s} {n_total:6d}")

    return pd.Series(code, index=df.index, name="ms_evol_class")
