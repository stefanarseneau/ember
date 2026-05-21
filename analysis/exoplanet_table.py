"""LaTeX table of exoplanet host stars with WD companion ages."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load_data import DATA_DIR, load_main_data, setup_matplotlib

setup_matplotlib()

data = load_main_data()

stellarhosts_raw = pd.read_csv(DATA_DIR / "merge_data/stellarhosts.csv", comment="#")
stellarhosts_raw["gaia_dr3_id"] = (
    stellarhosts_raw["gaia_dr3_id"].str.extract(r"(\d+)$").astype("Int64")
)
sh_by_gaia = stellarhosts_raw.drop_duplicates(subset=["gaia_dr3_id"])

stellarhosts = pd.merge(
    data,
    sh_by_gaia[["gaia_dr3_id", "sy_pnum", "hostname"]],
    left_on="ms_source_id",
    right_on="gaia_dr3_id",
)
stellarhosts["reliable_age"] = stellarhosts.Mass > 0.63
print(f"{len(stellarhosts)} exoplanet host systems matched")


def latex_ci(med, hi, lo, fmt="{:.2f}"):
    if np.isnan(med):
        return "-"
    return rf"${fmt.format(med)}^{{+{fmt.format(hi)}}}_{{-{fmt.format(lo)}}}$"


df = stellarhosts.copy()

df["wd_cool_age_tex"] = [
    latex_ci(m, h, l)
    for m, h, l in zip(df.cool_age, df.e_cool_age_upper, df.e_cool_age_lower)
]
df["wd_age_tex"] = [
    latex_ci(m, h, l)
    for m, h, l in zip(df.tot_age, df.tot_age_error_upper, df.tot_age_error_lower)
]

out_path = Path(__file__).parent / "exoplanets.tex"
df[["source_id", "gaia_dr3_id", "sy_pnum", "wd_cool_age_tex", "wd_age_tex"]].to_latex(
    out_path,
    index=False,
    escape=False,
    caption="Exoplanet host stars with WD companions from the WD+MS sample.",
    label="tab:exoplanets",
)
print(f"Saved {out_path}")
