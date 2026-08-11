"""Stitch per-population SED fit parquets into ages.pqt.

Reads thick.pqt, thin.pqt, mixed.pqt from WDWD_STITCH_DIR, merges them, assigns
the best-fit atmosphere model per star using Vincent+2024 spectral types, applies
a cool-WD Teff correction, then runs the age interpolation for each population.

Requires:
  - WDMODELS_DIR env var pointing to the WD_models package
  - measureages package (parallel_forloop)
  - Vincent+2024 catalog at VINCENT_CATALOG (set VINCENT_CATALOG env var, or
    defaults to ~/observational/catalogs/XP-spectra/vincent2024.pqt)

Usage:
    python -m catalog.wdwd.stitch [--inpath data/build/wdwd_stitch_input] [--outpath data/catalogs/ages.pqt]
"""

import os
import sys
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR, WDWD_STITCH_DIR
from ..corrections import correct_cool_wd_mass

_COLUMNS = ["teff", "std_tt", "radius", "std_rr", "cov_rt",
            "av", "std_av", "dist", "std_dist", "nsamps"]

_VINCENT_DEFAULT = Path.home() / "observational/catalogs/XP-spectra/vincent2024.pqt"
VINCENT_CATALOG  = Path(os.environ.get("VINCENT_CATALOG", _VINCENT_DEFAULT))


def _load_wd_models():
    wdmodels_dir = os.environ["WDMODELS_DIR"]
    sys.path.append(wdmodels_dir)
    import WD_models

    mass_sun, radius_sun, newton_G = 1.9884e30, 6.957e8, 6.674e-11
    font_model = WD_models.load_model("be", "be", "be", "H")
    g_acc = (10 ** font_model["logg"]) / 100.0
    rsun  = np.sqrt(font_model["mass_array"] * mass_sun * newton_G / g_acc) / radius_sun

    rad_func  = WD_models.interp_xy_z_func(
        x=rsun, y=10 ** font_model["logteff"], z=font_model["mass_array"],
        interp_type="linear",
    )
    mass_func = WD_models.interp_xy_z_func(
        x=font_model["mass_array"], y=10 ** font_model["logteff"], z=rsun,
        interp_type="linear",
    )
    return rad_func, mass_func


def stype_to_pop_type(stype: pd.Series, teff: pd.Series) -> np.ndarray:
    """Map Vincent+2024 SpType + Teff → pop_type ('thick', 'thin', 'mixed').

    thick  — DA, DAH, DAZ; DC/DZ with Teff < 5500 K
    mixed  — DAB, DBA; DC/DZ with 5500 ≤ Teff ≤ 11 000 K
    thin   — DB, DBZ, DQ, DO; DC/DZ with Teff > 11 000 K
    """
    s = stype.fillna("").str.upper().str.replace(":", "", regex=False).str.strip()
    is_DAB = s.str.startswith("DAB")
    is_DBA = s.str.startswith("DBA")
    is_DB  = s.str.startswith("DB") & ~is_DBA
    is_DQ  = s.str.startswith("DQ")
    is_DO  = s.str.startswith("DO")
    is_DC  = s.str.startswith("DC")
    is_DZ  = s.str.startswith("DZ")

    pop = np.full(len(s), "thick", dtype=object)
    pop = np.where(is_DAB | is_DBA,                                    "mixed", pop)
    pop = np.where(is_DB | is_DQ | is_DO,                             "thin",  pop)
    pop = np.where((is_DC | is_DZ) & (teff < 5500),                   "thick", pop)
    pop = np.where((is_DC | is_DZ) & (teff >= 5500) & (teff <= 11000), "mixed", pop)
    pop = np.where((is_DC | is_DZ) & (teff > 11000),                  "thin",  pop)
    return pop


def find_best(data: pd.DataFrame) -> pd.DataFrame:
    """Assign best-fit atmosphere suffix (thick/thin/mixed) using Vincent+2024 SpType."""
    vincent = pd.read_parquet(VINCENT_CATALOG,
                              columns=["GaiaDR3", "PDA", "PDB", "PDC", "PDO", "PDQ", "PDZ", "SpType"])
    data = (data.merge(vincent, left_on="sourceid", right_on="GaiaDR3", how="left")
                .drop(columns="GaiaDR3"))

    pop_type = stype_to_pop_type(data["SpType"], data["teff_thick"])
    suffix   = np.char.add("_", pop_type)

    for c in _COLUMNS:
        data[f"{c}_best"] = data[f"{c}_thin"].to_numpy()
        data.loc[suffix == "_mixed", f"{c}_best"] = data.loc[suffix == "_mixed", f"{c}_mixed"].to_numpy()
        data.loc[suffix == "_thick", f"{c}_best"] = data.loc[suffix == "_thick", f"{c}_thick"].to_numpy()

    data["pop_type"] = pop_type
    return data


def correct_cool_ages(data: pd.DataFrame, outpath: Path) -> pd.DataFrame:
    """Re-run age measurement for cool WDs using corrected teff_best/radius_best."""
    from measureages import parallel_forloop

    age_cols = ["log_age_cool", "log_age_cool_hi", "log_age_cool_lo",
                "log_age",      "log_age_hi",      "log_age_lo",
                "mass",         "mass_hi",          "mass_lo"]
    cool_mask = data["teff_best"] < 6000

    for pop in ("thick", "thin", "mixed"):
        mask = cool_mask & (data["pop_type"] == pop)
        if not mask.any():
            continue
        print(f"Re-measuring ages for {mask.sum()} cool {pop} stars…")
        subset = data[mask].copy()
        for col in age_cols:
            subset[col] = np.nan
        ckpt = outpath.with_name(outpath.stem + f"_cool_{pop}_ckpt.pqt")
        subset = parallel_forloop(subset, ckpt, pop_type=pop)
        data.loc[mask, age_cols] = subset[age_cols].to_numpy()

    return data


def correct_teff(data: pd.DataFrame, rad_func, mass_func) -> pd.DataFrame:
    """Apply O'Brien+2024 eq. 1–3 mass and Teff correction for cool WDs (Teff < 6000 K)."""
    all_mass = rad_func(data["radius_best"].to_numpy(), data["teff_best"].to_numpy())
    mass_cor, mass_uncor, teff_cor, teff_uncor = correct_cool_wd_mass(
        all_mass, data["teff_best"].to_numpy()
    )

    mask = data["teff_best"] < 6000
    data.loc[mask, "mass_uncor"]  = mass_uncor[mask]
    data.loc[mask, "teff_best"]  *= (mass_cor[mask] / all_mass[mask]) ** (1 / 6)
    data.loc[mask, "radius_best"] = mass_func(mass_cor[mask], data.loc[mask, "teff_best"])
    return data


def main(inpath: Path = WDWD_STITCH_DIR, outpath: Path = DATA_DIR / "catalogs/ages.pqt",
         correct_ages: bool = False) -> None:
    from measureages import parallel_forloop

    rad_func, mass_func = _load_wd_models()

    dataframes = []
    for suf in ("thick", "thin", "mixed"):
        new_cols = [f"{c}_{suf}" for c in _COLUMNS]
        df = pd.read_parquet(inpath / f"{suf}.pqt")
        df["sourceid"] = df["sourceid"].astype(np.int64)
        df = df.rename(columns=dict(zip(_COLUMNS, new_cols)))
        dataframes.append(df)

    dataframe = reduce(lambda x, y: x.merge(y, on="sourceid"), dataframes)
    dataframe  = find_best(dataframe)
    dataframe  = correct_teff(dataframe, rad_func, mass_func)

    print(dataframe[["teff_thick", "teff_thin", "teff_mixed", "teff_best", "PDA"]])

    age_cols = ["log_age_cool", "log_age_cool_hi", "log_age_cool_lo",
                "log_age",      "log_age_hi",      "log_age_lo",
                "mass",         "mass_hi",         "mass_lo"]
    for col in age_cols:
        dataframe[col] = np.nan

    for pop in ("thick", "thin", "mixed"):
        mask = dataframe["pop_type"] == pop
        if not mask.any():
            continue
        print(f"Measuring ages for {mask.sum()} {pop} stars…")
        ckpt   = outpath.with_name(outpath.stem + f"_{pop}_ckpt.pqt")
        subset = parallel_forloop(dataframe[mask].copy(), ckpt, pop_type=pop)
        dataframe.loc[mask, age_cols] = subset[age_cols].to_numpy()

    if correct_ages:
        dataframe = correct_cool_ages(dataframe, outpath)

    dataframe.to_parquet(outpath)
    dataframe.to_csv(outpath.with_suffix(".csv"))
    print(f"Saved {outpath}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inpath",       type=Path, default=WDWD_STITCH_DIR)
    parser.add_argument("--outpath",      type=Path, default=DATA_DIR / "catalogs/ages.pqt")
    parser.add_argument("--correct-ages", action="store_true",
                        help="Recompute cooling and total ages from corrected mass/Teff for cool WDs")
    args = parser.parse_args()
    main(inpath=args.inpath, outpath=args.outpath, correct_ages=args.correct_ages)
