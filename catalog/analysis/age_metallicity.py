"""Age vs [Fe/H] across GALAH, APOGEE+ASTRA, and LAMOST surveys."""

import numpy as np
import pandas as pd
from scipy.odr import Model, ODR, RealData
from scipy.stats import spearmanr

from catalog.config import FIGURES_DIR
from catalog.analysis.load_data import load_catalog, setup_matplotlib


def linear(B, x):
    return B[0] * x + B[1]


def odr_fit(df, x_col, y_col, sx_col, sy_lo_col, sy_hi_col):
    d = df[[x_col, y_col, sx_col, sy_lo_col, sy_hi_col]].copy()
    d["sy"] = 0.5 * (d[sy_lo_col] + d[sy_hi_col])
    d = d.replace(0, np.nan).dropna()
    data = RealData(d[x_col].values, d[y_col].values,
                    sx=d[sx_col].values, sy=d["sy"].values)
    out = ODR(data, Model(linear), beta0=[-3.0, 6.0]).run()
    return out.beta[0], out.beta[1], out.sd_beta[0], out.sd_beta[1]


def main():
    plt, _ = setup_matplotlib()
    combined, normal, white_ll, blue_ll = load_catalog()
    totalage = normal | white_ll

    fig, ax = plt.subplots(ncols=3, figsize=(16, 6), sharey=True)

    # GALAH
    sel_g = combined.query("survey == 'GALAH'").loc[totalage]
    ax[0].errorbar(
        sel_g["fe_h"], sel_g["tot_age"],
        xerr=sel_g["e_fe_h"],
        yerr=sel_g[["tot_age_error_lower", "tot_age_error_upper"]].values.T,
        fmt="o", color="#eeeeee", markeredgecolor="k", ecolor="k", lw=2, capsize=4, markersize=7,
    )
    m_g, b_g, sm_g, sb_g = odr_fit(
        sel_g, "fe_h", "tot_age", "e_fe_h", "tot_age_error_lower", "tot_age_error_upper"
    )
    rho_g, _ = spearmanr(sel_g["fe_h"], sel_g["tot_age"], nan_policy="omit")
    ax[0].text(0.05, 0.97, f"$\\rho = {rho_g:.2f}$",
               transform=ax[0].transAxes, va="top", fontsize=16,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0))
    ax[0].set_ylim(0, 13)
    ax[0].set_xlabel("[Fe/H]")
    ax[0].set_ylabel("Age [Gyr]")
    ax[0].set_title("GALAH", fontsize=24)
    ax[0].set_xlim(-1.0, 0.8)

    # APOGEE + ASTRA
    sel_a = combined.query("survey == 'APOGEE' or survey == 'ASTRA'").loc[totalage]
    ax[1].errorbar(
        sel_a["fe_h"], sel_a["tot_age"],
        xerr=sel_a["e_fe_h"],
        yerr=sel_a[["tot_age_error_lower", "tot_age_error_upper"]].values.T,
        fmt="o", color="#eeeeee", markeredgecolor="k", ecolor="k", lw=2, capsize=4, markersize=7,
    )
    m_a, b_a, sm_a, sb_a = odr_fit(
        sel_a, "fe_h", "tot_age", "e_fe_h", "tot_age_error_lower", "tot_age_error_upper"
    )
    rho_a, _ = spearmanr(sel_a["fe_h"], sel_a["tot_age"], nan_policy="omit")
    ax[1].text(0.05, 0.97, f"$\\rho = {rho_a:.2f}$",
               transform=ax[1].transAxes, va="top", fontsize=16,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0))
    ax[1].set_xlabel("[Fe/H]")
    ax[1].set_title("APOGEE + ASTRA", fontsize=24)

    # LAMOST
    sel_l = combined.query("survey == 'LAMOST' and Mass > 0.63").dropna(
        subset=["fe_h", "tot_age", "tot_age_error_lower", "tot_age_error_upper"]
    )
    ax[2].errorbar(
        sel_l["fe_h"], sel_l["tot_age"],
        xerr=sel_l["e_fe_h"],
        yerr=sel_l[["tot_age_error_lower", "tot_age_error_upper"]].values.T,
        fmt="o", color="#eeeeee", markeredgecolor="k", ecolor="k", lw=2, capsize=4, markersize=7,
    )
    rho_l, _ = spearmanr(sel_l["fe_h"], sel_l["tot_age"], nan_policy="omit")
    ax[2].text(0.05, 0.97, f"$\\rho = {rho_l:.2f}$",
               transform=ax[2].transAxes, va="top", fontsize=16,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0))
    ax[2].set_xlabel("[Fe/H]")
    ax[2].set_title("LAMOST", fontsize=24)

    slopes = np.array([m_g, m_a])
    sigmas = np.array([sm_g, sm_a])
    w = 1.0 / sigmas**2
    m_wtd  = np.sum(w * slopes) / np.sum(w)
    sm_wtd = 1.0 / np.sqrt(np.sum(w))

    print(f"N = {len(sel_g) + len(sel_a) + len(sel_l)}")
    print(f"GALAH          slope = {m_g:.3f} +/- {sm_g:.3f} Gyr/dex")
    print(f"APOGEE + ASTRA slope = {m_a:.3f} +/- {sm_a:.3f} Gyr/dex")
    print(f"Weighted mean  slope = {m_wtd:.3f} +/- {sm_wtd:.3f} Gyr/dex")

    fig.tight_layout()
    out = FIGURES_DIR / "age_metallicity.pdf"
    fig.savefig(out)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
