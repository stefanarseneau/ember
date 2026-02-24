from functools import reduce
import pandas as pd
import numpy as np
import argparse
import os

columns = ["teff", "e_teff", "logg", "e_logg", "covar", "redchi",
            "log_age_cool", "log_age_cool_hi", "log_age_cool_lo",
            "log_age", "log_age_hi", "log_age_lo"]

def find_best(data : pd.DataFrame):
    vincent = pd.read_parquet("~/observational/catalogs/XP-spectra/vincent2024.pqt")
    vincent = vincent[["GaiaDR3", "PDA", "PDB", "PDC", "PDO", "PDQ", "PDZ"]]
    data = data.merge(vincent, left_on="gaia_dr3_source_id", right_on="GaiaDR3", how="left")
    data = data.drop(["GaiaDR3"], axis=1)

    mask = np.isnan(data["PDA"])
    data.loc[mask, "PDA"] = 1.0

    #suffix = np.where(data["PDA"] > 0.8, "_thick",
    #         np.where(data["PDA"] > 0.5 and data["teff_mixed"] >= 4000, "_mixed", "_thin"))
    
    teff = data["teff_mixed"]

    # elementwise conditions
    thick = (data["PDA"] > 0.8) | ((data["PDA"] > 0.5) & (teff < 4800))
    mixed = (data["PDA"] > 0.5) & (~thick)

    suffix = np.where(thick, "_thick",
             np.where(mixed, "_mixed", "_thin"))


    for c in columns:
        # start with thin, then overwrite where mixed/thick
        data[f"{c}_best"] = data[f"{c}_thin"].to_numpy()
        m = suffix == "_mixed"
        t = suffix == "_thick"
        data.loc[m, f"{c}_best"] = data.loc[m, f"{c}_mixed"].to_numpy()
        data.loc[t, f"{c}_best"] = data.loc[t, f"{c}_thick"].to_numpy()
    
    data.loc[mask, "PDA"] = np.nan
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stitch that stuff",
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inpath', type=str, default = 'data/', help='Path to input pqt file')
    parser.add_argument('outpath', type=str, default = 'ages.pqt', help='Path to output pqt file')
    args = parser.parse_args()
    
    suffixes = ["_thick", "_thin", "_mixed"]
    dataframes = []
    for suf in suffixes:
        new_cols = [f"{col}{suf}" for col in columns]
        data = pd.read_parquet(os.path.join(args.inpath, f"ages{suf}.pqt"))
        data = data.rename(columns={old_col:new_col 
                for old_col, new_col in zip(columns, new_cols)})
        dataframes.append(data)
    
    dataframe = reduce(lambda x, y: x.merge(y, on='gaia_dr3_source_id'), dataframes)
    dataframe = find_best(dataframe)
    print(dataframe[["teff_thick", "teff_thin", "teff_mixed", "teff_best", "PDA"]])
    dataframe.to_parquet(args.outpath)
