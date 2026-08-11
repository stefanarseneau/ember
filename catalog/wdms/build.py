"""Build analysis-ready parquet catalogs from raw WD+MS survey data.

Outputs written to DATA_DIR:
  combined.pqt    — one row per source passing WD quality cuts.  All main-
                    catalog columns are present; abundance columns are merged
                    in from the highest-priority survey match (NaN where none).
  metallicity.pqt — one row per (source, survey) match, abundance cols only.
                    Use this for provenance lookups and counts tables.

Run whenever raw input data changes.  Analysis scripts read the parquets;
they should not re-do any of the transformations performed here.

Survey priority for deduplication: GALAH > APOGEE > ASTRA > LAMOST.
groupby.first() (after sorting by priority) fills each abundance column from
the highest-priority survey that has a non-null value, so element abundances
unique to APOGEE (ti3_fe, …) are preserved even when GALAH wins for fe_h.
"""

import numpy as np
import pandas as pd

from ..config import DATA_DIR, SURVEY_PRIORITY, FLAG_THRESHOLD, BP_RP_A, BP_RP_B, BP_RP_EXP
from ..bitmasks import _make_photometry_bitmask
from ..corrections import correct_cool_wd_mass
from . import evolutionary_state


# ── Quality cuts ───────────────────────────────────────────────────────────

def _apply_wd_cuts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.query("Teff > 3000 & ruwe < 1.2 & R_chance_align < 0.1 & Mass > 0.21")
    excess = 3 * (BP_RP_A + BP_RP_B * df["phot_g_mean_mag"] ** BP_RP_EXP)
    return df[df["phot_bp_rp_excess_factor"] > excess]


_AGE_COLS = ["log_age_cool", "log_age_cool_hi", "log_age_cool_lo",
             "log_age",      "log_age_hi",      "log_age_lo",
             "mass",         "mass_hi",          "mass_lo"]
_CHECK_COLS = [c + "_check" for c in _AGE_COLS]


def _run_ages(df: pd.DataFrame, teff_col: str, mass_col: str, label: str,
              out_cols: list) -> pd.DataFrame:
    """Run parallel_forloop for cool WDs and store results in out_cols."""
    from sedtool.measureages import parallel_forloop
    from ..wdwd.stitch import stype_to_pop_type

    df = df.copy()
    df["_pop_type"] = stype_to_pop_type(df["spectype"], df[teff_col])
    df["_e_Teff"] = (df["e_Teff_lower"] + df["e_Teff_upper"]) / 2
    df["_e_Mass"] = (df["e_Mass_lower"] + df["e_Mass_upper"]) / 2
    for col in _AGE_COLS:
        df[col] = np.nan

    cool_mask = df[teff_col].notna() & (df[teff_col] < 6000)
    for pop in ("thick", "thin", "mixed"):
        mask = cool_mask & (df["_pop_type"] == pop)
        if not mask.any():
            continue
        print(f"{label}: measuring ages for {mask.sum()} cool {pop} WDs…")
        subset = df[mask].copy()
        ckpt = DATA_DIR / f"combined_{label}_{pop}_ckpt.parquet"
        subset = parallel_forloop(
            subset, ckpt, pop_type=pop,
            user_teff=teff_col,  user_e_teff="_e_Teff",
            user_mass=mass_col,  user_e_mass="_e_Mass",
        )
        df.loc[mask, _AGE_COLS] = subset[_AGE_COLS].to_numpy()

    df = df.rename(columns=dict(zip(_AGE_COLS, out_cols)))
    return df.drop(columns=["_pop_type", "_e_Teff", "_e_Mass"])


def _correct_cool_ages(df: pd.DataFrame) -> pd.DataFrame:
    """Re-measure cooling and total ages for cool WDs using corrected Mass/Teff."""
    return _run_ages(df, teff_col="Teff", mass_col="Mass",
                     label="correct", out_cols=_AGE_COLS)


def _check_cool_ages(df: pd.DataFrame) -> pd.DataFrame:
    """Re-measure ages using uncorrected Mass/Teff and print agreement with Tyler catalog."""
    df = _run_ages(df, teff_col="Teff_uncor", mass_col="Mass_uncor",
                   label="check", out_cols=_CHECK_COLS)

    def _stats(label, check_log_col, tyler_col):
        check_gyr = 10 ** (df[check_log_col] - 9)
        ok = check_gyr.notna() & df[tyler_col].notna()
        if not ok.any():
            print(f"  {label}: no overlapping rows")
            return
        diff = check_gyr[ok] - df.loc[ok, tyler_col]
        print(f"  {label} (N={ok.sum()}):")
        print(f"    mean Δ = {diff.mean():+.3f} Gyr,  median Δ = {diff.median():+.3f} Gyr")
        print(f"    RMS    = {(diff**2).mean()**0.5:.3f} Gyr,  |Δ| p84 = {diff.abs().quantile(0.84):.3f} Gyr")

    print("\nAge agreement (check vs Tyler):")
    _stats("cooling age", "log_age_cool_check", "cool_age")
    _stats("total age",   "log_age_check",      "tot_age")
    return df


def _apply_cool_wd_correction(df: pd.DataFrame) -> pd.DataFrame:
    mass_cor, mass_uncor, teff_cor, teff_uncor = correct_cool_wd_mass(
        df["Mass"].to_numpy(), df["Teff"].to_numpy()
    )
    df = df.copy()
    df["Mass_uncor"] = mass_uncor
    df["Mass"]       = mass_cor
    df["Teff_uncor"] = teff_uncor
    df["Teff"]       = teff_cor
    return df

# ── Per-survey processing ──────────────────────────────────────────────────

def _process_galah() -> pd.DataFrame:
    _elems = [
        "Al", "Ba", "C", "Ca", "Ce", "Co", "Cr", "Cr2", "Cu", "Eu",
        "K", "La", "Li", "Mg", "Mn", "Mo", "Na", "Nd", "Ni", "O",
        "Rb", "Ru", "Sc", "Sc2", "Si", "Sm", "Sr", "Ti", "Ti2", "V",
        "Y", "Zn", "Zr",
    ]
    rename = {
        "fe_h": "fe_h", "e_fe_h": "e_fe_h", "flag_fe_h": "flag_fe_h",
        "alpha_fe": "alpha_fe", "e_alpha_fe": "e_alpha_fe",
        "flag_alpha_fe": "flag_alpha_fe",
        "logg_2": "ms_logg", "e_logg": "e_ms_logg",
        "teff_2": "ms_teff", "e_teff": "e_ms_teff",
    }
    for el in _elems:
        lo = el.lower()
        rename[f"{el}_fe"]      = f"{lo}_fe"
        rename[f"e_{el}_fe"]    = f"e_{lo}_fe"
        rename[f"flag_{el}_fe"] = f"flag_{lo}_fe"
        rename[f"nr_{el}_fe"]   = f"nr_{lo}_fe"

    df = pd.read_csv(DATA_DIR / "tyler/tyler_wdms_GALAH.csv").query("Teff_1 > 3200")
    df = df.rename(columns=rename)
    metal_cols = [c for c in rename.values() if c in df.columns]
    return df[["source_id"] + metal_cols].copy()


def _process_apogee() -> pd.DataFrame:
    _fe_elems = [
        "AL", "CA", "CE", "CI", "CO", "CR", "CU", "C",
        "K", "MG", "MN", "NA", "NI", "N", "O", "P",
        "SI", "S", "TI", "V", "YB",
    ]
    rename = {
        "FE_H": "fe_h", "FE_H_ERR": "e_fe_h", "FE_H_FLAG": "flag_fe_h",
        "FE_H_SPEC": "fe_h_spec",
        "M_H": "m_h", "M_H_ERR": "e_m_h",
        "ALPHA_M": "alpha_m", "ALPHA_M_ERR": "e_alpha_m",
        "TIII_FE": "ti3_fe", "TIII_FE_ERR": "e_ti3_fe",
        "TIII_FE_FLAG": "flag_ti3_fe", "TIII_FE_SPEC": "ti3_fe_spec",
        "LOGG_2": "ms_logg", "LOGG_ERR": "e_ms_logg",
        "TEFF_2": "ms_teff", "TEFF_ERR": "e_ms_teff",
    }
    for el in _fe_elems:
        lo = el.lower()
        rename[f"{el}_FE"]      = f"{lo}_fe"
        rename[f"{el}_FE_ERR"]  = f"e_{lo}_fe"
        rename[f"{el}_FE_FLAG"] = f"flag_{lo}_fe"
        rename[f"{el}_FE_SPEC"] = f"{lo}_fe_spec"

    df = pd.read_csv(DATA_DIR / "tyler/tyler_wdms_APOGEE.csv").query("Teff_1 > 3200")
    df = df.rename(columns=rename)
    df["alpha_fe"]   = df["alpha_m"] + df["m_h"] - df["fe_h"]
    df["e_alpha_fe"] = np.sqrt(df["e_alpha_m"]**2 + df["e_m_h"]**2 + df["e_fe_h"]**2)
    metal_cols = list(dict.fromkeys(
        [c for c in rename.values() if c in df.columns] + ["alpha_fe", "e_alpha_fe"]
    ))
    return df[["source_id"] + metal_cols].copy()


def _process_astra() -> pd.DataFrame:
    _h_elems = [
        "al", "c", "ca", "ce", "co", "cr", "cu",
        "k", "mg", "mn", "n", "na", "nd", "ni", "o",
        "p", "s", "si", "ti", "v",
    ]
    rename = {
        "fe_h": "fe_h", "e_fe_h": "e_fe_h", "fe_h_flags": "flag_fe_h",
        "alpha_m_atm": "alpha_m_atm", "e_alpha_m_atm": "e_alpha_m_atm",
        "m_h_atm": "m_h_atm", "e_m_h_atm": "e_m_h_atm",
        "c_12_13": "c_12_13", "e_c_12_13": "e_c_12_13",
        "c_12_13_flags": "flag_c_12_13",
        "fe_h_atmo": "fe_h_atmo",
        "logg_2": "ms_logg", "e_logg": "e_ms_logg",
        "teff_2": "ms_teff", "e_teff": "e_ms_teff",
    }
    for el in _h_elems:
        rename[f"{el}_h_flags"] = f"flag_{el}_fe"

    df = (pd.read_csv(DATA_DIR / "tyler/tyler_wdms_ASTRA.csv")
            .query("Teff_1 > 3200 & flag_bad == False"))
    df = df.rename(columns=rename)

    for el in _h_elems:
        df[f"{el}_fe"]   = df[f"{el}_h"] - df["fe_h"]
        df[f"e_{el}_fe"] = np.sqrt(df[f"e_{el}_h"]**2 + df["e_fe_h"]**2)

    df["alpha_fe"]   = df["alpha_m_atm"] + df["m_h_atm"] - df["fe_h"]
    df["e_alpha_fe"] = np.sqrt(
        df["e_alpha_m_atm"]**2 + df["e_m_h_atm"]**2 + df["e_fe_h"]**2
    )

    metal_cols = list(dict.fromkeys(
        [c for c in rename.values() if c in df.columns]
        + [f"{el}_fe"   for el in _h_elems]
        + [f"e_{el}_fe" for el in _h_elems]
        + ["alpha_fe", "e_alpha_fe"]
    ))
    return df[["source_id"] + metal_cols].copy()


def _process_lamost() -> pd.DataFrame:
    """Load LAMOST WD+MS data; fetches [Fe/H] errors from VizieR and caches locally."""
    cache_path = DATA_DIR / "merge/wdms_lamost_efeh.csv"

    lamost = pd.read_csv(DATA_DIR / "merge/wdms_lamost.csv")
    lamost = lamost.rename(columns={"[Fe/H]": "fe_h", "logg": "ms_logg", "Teff.1": "ms_teff"})
    lamost = lamost[lamost.Teff > 0]
    lamost = lamost.sort_values("snrg", ascending=False).drop_duplicates(subset=["source_id"])
    lamost = lamost.query("fe_h > -900")

    if cache_path.exists():
        efeh   = pd.read_csv(cache_path)
        lamost = lamost.merge(efeh, on="ObsID", how="left")
    else:
        try:
            import pyvo
            tap     = pyvo.dal.TAPService("http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")
            id_list = ", ".join(str(int(i)) for i in lamost["ObsID"].dropna())
            result  = (
                tap.search(
                    f'SELECT "ObsID", "e_[Fe/H]" as e_FeH FROM "V/156/dr7slrs"'
                    f' WHERE "ObsID" IN ({id_list})'
                )
                .to_table()
                .to_pandas()
                .rename(columns={"e_FeH": "e_fe_h"})
            )
            result[["ObsID", "e_fe_h"]].to_csv(cache_path, index=False)
            lamost = lamost.merge(result, on="ObsID", how="left")
        except Exception as e:
            print(f"Warning: LAMOST e_fe_h query failed ({e}); using NaN")
            lamost["e_fe_h"] = np.nan

    return lamost[["source_id", "fe_h", "e_fe_h", "ms_logg", "ms_teff"]].copy()


# ── Flag thresholding ──────────────────────────────────────────────────────

def _apply_flag_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    for flag_col in flag_cols:
        base    = flag_col[len("flag_"):]
        targets = [c for c in [base, f"e_{base}"] if c in df.columns]
        if not targets:
            continue
        for survey, threshold in FLAG_THRESHOLD.items():
            bad = (df["survey"] == survey) & (df[flag_col].fillna(0) > threshold)
            df.loc[bad, targets] = np.nan
    return df


# ── Main ───────────────────────────────────────────────────────────────────

def main(correct_ages: bool = False, check_correct_ages: bool = False) -> None:
    # ── 1. Load and quality-cut main catalog ──────────────────────────────
    print("Loading main catalog…")
    main_cat = pd.read_csv(
        DATA_DIR / "tyler/WDMS_total_ages_correct_models_cut_down.csv",
        low_memory=False,
    )
    main_cat["tot_age_error_upper"]    = main_cat["tot_age_error_upper"].fillna(999)
    main_cat["total_age_lower_limit"]  = (
        main_cat["tot_age"] - main_cat["tot_age_error_lower"]
    )
    main_cat = _apply_wd_cuts(main_cat)
    main_cat = _apply_cool_wd_correction(main_cat)
    if check_correct_ages:
        main_cat = _check_cool_ages(main_cat)
    if correct_ages:
        main_cat = _correct_cool_ages(main_cat)
    main_cat["phot_flag"] = _make_photometry_bitmask(main_cat)
    print(f"  {len(main_cat)} sources after quality cuts")

    # ── 2. Process each survey ────────────────────────────────────────────
    print("Processing surveys…")
    survey_frames = {
        "GALAH":  _process_galah(),
        "APOGEE": _process_apogee(),
        "ASTRA":  _process_astra(),
        "LAMOST": _process_lamost(),
    }
    for name, df in survey_frames.items():
        df = df[df["source_id"].isin(main_cat["source_id"])].copy()
        df["survey"] = name
        survey_frames[name] = df
        print(f"  {name:6s}: {len(df)} sources")

    # ── 3. Build metallicity.pqt (long form, provenance) ─────────────────
    # Note: this also gates ms_logg on fe_h passing the sentinel check below —
    # acceptable since each survey's logg and [Fe/H] come from the same joint
    # stellar-parameter fit, so failures are correlated.
    metallicity = pd.concat(survey_frames.values(), ignore_index=True)
    metallicity = _apply_flag_thresholds(metallicity)
    metallicity = metallicity.query("fe_h > -900").copy()

    out_metal_pqt = DATA_DIR / "metallicity.pqt"
    metallicity.to_parquet(out_metal_pqt, index=False)
    print(f"\nSaved {out_metal_pqt}  ({len(metallicity)} rows)")

    out_metal_csv = DATA_DIR / "metallicity.csv"
    metallicity.to_parquet(out_metal_csv, index=False)
    print(f"\nSaved {out_metal_csv}  ({len(metallicity)} rows)")

    # ── 4. Deduplicate: GALAH > APOGEE > ASTRA > LAMOST ──────────────────
    deduped = (
        metallicity
        .assign(_rank=metallicity["survey"].map(SURVEY_PRIORITY))
        .sort_values(["source_id", "_rank"])
        .groupby("source_id", sort=False).first()
        .reset_index()
        .drop(columns="_rank")
    )

    # ── 5. Merge onto main catalog ────────────────────────────────────────
    combined = main_cat.merge(deduped, on="source_id", how="left")

    # ── 5.5 Photometry quality bitmask ────────────────────────────────────
    combined["phot_flag"] = _make_photometry_bitmask(combined)

    # ── 5.6 Age classification bitmask ────────────────────────────────────
    # 0 = normal   (totalage, age_upper_err ≤ 13 Gyr)
    # 1 = white_ll (large upper err, or 0.6 < M ≤ 0.63)
    # 2 = blue_ll  (M ≤ 0.6)
    totalage   = (combined["Mass"] > 0.63) & (combined["Teff"] > 3200)
    age_lowlim = combined["tot_age_error_upper"] > 13
    age_class  = np.ones(len(combined), dtype=np.int8)
    age_class[combined["Mass"] <= 0.6] = 2
    age_class[totalage & ~age_lowlim]  = 0
    combined["age_class"] = age_class

    # Below 0.6 Msun the IFMR-derived main-sequence lifetime is too uncertain
    # to trust (Heintz+2022, Heintz+2024), so total_age_lower_limit falls back
    # to the cooling age alone rather than tot_age - tot_age_error_lower.
    blue_ll = combined["age_class"] == 2
    combined.loc[blue_ll, "total_age_lower_limit"] = combined.loc[blue_ll, "cool_age"]

    # ── 5.7 MS-companion evolutionary state (subgiant/giant) ──────────────
    print()
    combined["ms_evol_class"] = evolutionary_state.classify(combined)

    out_combined_pqt = DATA_DIR / "combined.pqt"
    out_combined_csv = DATA_DIR / "combined.csv"
    combined.to_parquet(out_combined_pqt, index=False)
    combined.to_csv(out_combined_csv, index=False)
    print(f"Saved {out_combined_pqt}  ({len(combined)} rows, "
          f"{combined['fe_h'].notna().sum()} with metallicity)")

    # ── 6. Summary ────────────────────────────────────────────────────────
    print("\nAbundance coverage (pre-dedup, true survey provenance):")
    abund_cols = [
        ("[Fe/H]",  "fe_h"),
        ("[Li/Fe]", "li_fe"),
        ("[α/Fe]",  "alpha_fe"),
        ("[C/Fe]",  "c_fe"),
    ]
    surveys = ["APOGEE", "ASTRA", "GALAH", "LAMOST"]
    header  = (f"  {'':12}" + "".join(f"{s:>8}" for s in surveys)
               + f"{'Total':>8}{'UppLim':>8}")
    print(header)
    for label, col in abund_cols:
        counts = [metallicity.loc[metallicity.survey == s, col].notna().sum()
                  for s in surveys]
        flag_col = f"flag_{col}"
        upplim = (metallicity[flag_col] == 1).sum() if flag_col in metallicity else 0
        print(f"  {label:12}" + "".join(f"{c:>8}" for c in counts)
              + f"{sum(counts):>8}{upplim:>8}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--correct-ages", action="store_true",
                        help="Re-measure cooling and total ages for cool WDs using corrected Mass/Teff")
    parser.add_argument("--check-correct-ages", action="store_true",
                        help="Re-measure ages using uncorrected Mass/Teff to verify against original catalog")
    args = parser.parse_args()
    main(correct_ages=args.correct_ages, check_correct_ages=args.check_correct_ages)
