from functools import reduce
import pandas as pd
import numpy as np
import argparse
import os

from measureages import parallel_forloop

import sys, os, tqdm, glob
wdmodels_dir = os.environ['WDMODELS_DIR']
sys.path.append(wdmodels_dir)
import WD_models

mass_sun, radius_sun, newton_G = 1.9884e30, 6.957e8, 6.674e-11
font_model = WD_models.load_model("be", "be", "be", "H")
g_acc = (10**font_model["logg"]) / 100.0
rsun = np.sqrt(font_model["mass_array"] * mass_sun * newton_G / g_acc) / radius_sun

_G_RAD_FUNC = WD_models.interp_xy_z_func(
    x=rsun, y=10**font_model["logteff"],
    z=font_model["mass_array"],
    interp_type="linear",
)

_G_MASS_FUNC = WD_models.interp_xy_z_func(
    x=font_model["mass_array"], y=10**font_model["logteff"],
    z=rsun, interp_type="linear",
)

columns = ["teff", "std_tt", "radius", "std_rr", "cov_rt", 
           "av", "std_av", "dist", "std_dist", "nsamps"]

def find_best(data : pd.DataFrame):
    vincent = pd.read_parquet("~/observational/catalogs/XP-spectra/vincent2024.pqt")
    vincent = vincent[["GaiaDR3", "PDA", "PDB", "PDC", "PDO", "PDQ", "PDZ", "SpType"]]
    data = data.merge(vincent, left_on="sourceid", right_on="GaiaDR3", how="left")
    data = data.drop(["GaiaDR3"], axis=1)

    # Use teff_thick as the temperature reference for DC/DZ cuts
    teff = data["teff_thick"]

    # Default: treat everything as DA (thick/pure-H) until told otherwise by SpType
    suffix = np.full(len(data), "_thick", dtype=object)

    # --- SpType-based decision tree (Heintz+2024 Sec 3.1) ---
    # thick  = pure H atmosphere  (DA models,       log(H/He) → ∞)
    # mixed  = mixed H/He         (H/He = 10^-5 models)
    # thin   = pure He atmosphere (He models,        log(H/He) = -30)
    #
    # DA, DAZ, DAH, ... (but not DAB) → thick
    # DAB, DBA                        → mixed  (H/He mixed types)
    # DB (not DBA), DQ, DO            → thin
    # DC, DZ: temperature-dependent
    stype = data["SpType"].fillna("").str.upper().str.replace(":", "", regex=False).str.strip()

    is_DAB  = stype.str.startswith("DAB")
    is_DBA  = stype.str.startswith("DBA")
    is_DB   = stype.str.startswith("DB")  & ~is_DBA   # pure-He DB (excludes DBA)
    is_DQ   = stype.str.startswith("DQ")
    is_DO   = stype.str.startswith("DO")
    is_DC   = stype.str.startswith("DC")
    is_DZ   = stype.str.startswith("DZ")

    dc_dz_thick = (is_DC | is_DZ) & (teff < 5500)
    dc_dz_mixed = (is_DC | is_DZ) & (teff >= 5500) & (teff <= 11000)
    dc_dz_thin  = (is_DC | is_DZ) & (teff > 11000)

    suffix = np.where(is_DAB | is_DBA,             "_mixed", suffix)
    suffix = np.where(is_DB | is_DQ | is_DO,       "_thin",  suffix)
    suffix = np.where(dc_dz_thick,                 "_thick", suffix)
    suffix = np.where(dc_dz_mixed,                 "_mixed", suffix)
    suffix = np.where(dc_dz_thin,                  "_thin",  suffix)

    for c in columns:
        # start with thin, then overwrite where mixed/thick
        data[f"{c}_best"] = data[f"{c}_thin"].to_numpy()
        m = suffix == "_mixed"
        t = suffix == "_thick"
        data.loc[m, f"{c}_best"] = data.loc[m, f"{c}_mixed"].to_numpy()
        data.loc[t, f"{c}_best"] = data.loc[t, f"{c}_thick"].to_numpy()

    data['pop_type'] = [s.lstrip('_') for s in suffix]
    return data

def correct_teff(data : pd.DataFrame):
    mask = data["teff_best"] < 6000
    dm = lambda teff : (-4.613e-19*teff**5 + 1.726e-14*teff**4 - 2.486e-10*teff**3
                        + 1.706e-6*teff**2 - 0.005487*teff + 7.068)
    mass = _G_RAD_FUNC(data.loc[mask, "radius_best"], data.loc[mask, "teff_best"])
    mass_cor = mass + 0.63 + dm(data.loc[mask, "teff_best"])
    # apply the corrections for the cool WDs
    data.loc[mask, "teff_best"] *= (mass_cor / mass)**(1/6)
    data.loc[mask, "radius_best"] = _G_MASS_FUNC(mass, data.loc[mask, "teff_best"])
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stitch that stuff",
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inpath', type=str, default = 'data/', help='Path to input pqt file')
    parser.add_argument('outpath', type=str, default = 'ages.pqt', help='Path to output pqt file')
    args = parser.parse_args()
    
    suffixes = ["thick", "thin", "mixed"]
    dataframes = []
    for suf in suffixes:
        new_cols = [f"{col}_{suf}" for col in columns]
        data = pd.read_parquet(os.path.join(args.inpath, f"{suf}.pqt"))
        data['sourceid'] = data['sourceid'].astype(np.int64)
        data = data.rename(columns={old_col:new_col 
                for old_col, new_col in zip(columns, new_cols)})
        dataframes.append(data)
    
    dataframe = reduce(lambda x, y: x.merge(y, on='sourceid'), dataframes)
    dataframe = find_best(dataframe)

    print(len(dataframe))
    dataframe.loc[:,"mass_best"] = _G_RAD_FUNC(dataframe.loc[:, "radius_best"], dataframe.loc[:, "teff_best"])
    #dataframe = correct_teff(dataframe)
    print(len(dataframe))

    print(dataframe[["teff_thick", "teff_thin", "teff_mixed", "teff_best", "PDA"]])

    # Measure ages for each population using the appropriate Bedard model
    age_cols = ["log_age_cool", "log_age_cool_hi", "log_age_cool_lo",
                "log_age",      "log_age_hi",      "log_age_lo"]
    for col in age_cols:
        dataframe[col] = np.nan

    for pop in ['thick', 'thin', 'mixed']:
        mask = dataframe['pop_type'] == pop
        if mask.sum() == 0:
            continue
        print(f"Measuring ages for {mask.sum()} {pop} stars...")
        subset = dataframe[mask].copy()
        ckpt = args.outpath.replace('.pqt', f'_{pop}_ckpt.pqt')
        subset = parallel_forloop(subset, ckpt, pop_type=pop)
        dataframe.loc[mask, age_cols] = subset[age_cols].to_numpy()

    dataframe.to_parquet(args.outpath)
    dataframe.to_csv(f"{args.outpath[:-4]}.csv")
