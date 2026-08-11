"""Mass vs Teff diagram with crystallization sequences and IFMR forbidden region."""

import os
import sys

from scipy.interpolate import LinearNDInterpolator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd
import argparse

from catalog.config import DATA_DIR, FIGURES_DIR
from catalog.analysis.load_data import load_catalog, setup_matplotlib

wdmodels_dir = os.environ["WDMODELS_DIR"]
sys.path.append(wdmodels_dir)
import WD_models

def make_totalage_interpolator(ifmr):
    font_model = WD_models.load_model("be", "be", "be", "H")
    cooling_age_func  = WD_models.interp_xy_z_func(
        x=font_model["mass_array"], y=10 ** font_model["logteff"], 
        z=font_model["age_cool"], interp_type="linear",
    )

    step = np.diff(ifmr.M_final.values[-3:]).mean()
    extra = np.arange(ifmr.M_final.max() + step, 1.3 + step, step)
    masses = np.concatenate([ifmr.M_final.values, extra])
    teffs = np.logspace(np.log10(3000), np.log10(60_000), num=100)
    teff_grid, mass_grid = np.meshgrid(teffs, masses, indexing="ij")
    ages = (cooling_age_func(mass_grid, teff_grid)
            + np.interp(mass_grid, ifmr.M_final, ifmr.mslifetime))
    
    points = np.column_stack([teff_grid.ravel(), mass_grid.ravel()])
    return LinearNDInterpolator(points, ages.ravel())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skipline", action="store_true", help="Skip forbidden region line")
    args = parser.parse_args()

    plt, _ = setup_matplotlib()
    combined, *_ = load_catalog()

    crys = pd.read_csv(DATA_DIR / "merge_data/crystallization.csv")
    ifmr = pd.read_csv(DATA_DIR / "MESA_IFMR/ifmr_data.csv")

    total_age_interpolator = make_totalage_interpolator(ifmr)

    model = WD_models.load_model(
        low_mass_model="Bedard2020",
        middle_mass_model="Bedard2020",
        high_mass_model="Bedard2020",
        atm_type="H",
        HR_bands=("bp3-rp3", "G3"),
    )

    logg_20pct = model["HR_to_logg"](crys["bprp_20pct"].to_numpy(), crys["gabs_20pct"].to_numpy())
    mass_20pct = model["HR_to_mass"](crys["bprp_20pct"].to_numpy(), crys["gabs_20pct"].to_numpy())
    teff_20pct = 10 ** model["HR_to_logteff"](crys["bprp_20pct"].to_numpy(), crys["gabs_20pct"].to_numpy())

    logg_80pct = model["HR_to_logg"](crys["bprp_80pct"].to_numpy(), crys["gabs_80pct"].to_numpy())
    mass_80pct = model["HR_to_mass"](crys["bprp_80pct"].to_numpy(), crys["gabs_80pct"].to_numpy())
    teff_80pct = 10 ** model["HR_to_logteff"](crys["bprp_80pct"].to_numpy(), crys["gabs_80pct"].to_numpy())

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.scatter(np.log10(combined.Teff), combined.Mass,
               c="white", edgecolor="k", alpha=0.7, s=7, zorder=1, rasterized=True)
    ax.axhline(np.nanmedian(combined.Mass), c="k")

    med_teff = np.nanmedian(combined.Teff)
    x_err = [[np.log10(med_teff) - np.log10(med_teff - np.nanmedian(combined.e_Teff_lower))],
              [np.log10(med_teff + np.nanmedian(combined.e_Teff_upper)) - np.log10(med_teff)]]
    y_err = [[np.nanmedian(combined.e_Mass_lower)], [np.nanmedian(combined.e_Mass_upper)]]
    ax.errorbar(np.log10(40000), 1.2, xerr=x_err, yerr=y_err,
                fmt="o", color="k", capsize=4, zorder=10)
    ax.text(np.log10(40000), 1.2 - y_err[0][0] - 0.02, "Median\nUncertainty",
            ha="center", va="top", fontsize=8)
    #ax.plot(np.log10(teff_20pct), mass_20pct, c="lime", lw=3, label="20\\% Crystallized", zorder=2)
    #ax.plot(np.log10(teff_80pct), mass_80pct, c="lime", ls="--", lw=3, label="80\\% Crystallized", zorder=2)

    correction_data = np.load(DATA_DIR / "correction_coefficients.npy")
    mass_med, coeff = correction_data[0], correction_data[1:]
    correction_poly = np.poly1d(coeff)
    correction_teffs = np.linspace(6000, 4000, 100)
    ax.plot(np.log10(correction_teffs), correction_poly(correction_teffs),
            lw=3, c="lime", label="Ad-hoc Polynomial Correction")
    
    ax.scatter(np.log10(combined["Teff"]), combined["Mass_uncor"], c="maroon", edgecolor="k", 
               s=7, alpha=0.2, zorder=0, rasterized=True)


    teff_eval = np.linspace(3500, 60000, 300)
    mass_eval = np.linspace(0.5, 1.3, 300)
    tg, mg = np.meshgrid(teff_eval, mass_eval)
    pts = np.column_stack([tg.ravel(), mg.ravel()])
    age_grid = total_age_interpolator(pts).reshape(tg.shape)
    age_levels = [0.5, 1, 3, 5, 8, 10, 13]
    cmap = plt.get_cmap("cool", len(age_levels))
    bounds = np.concatenate([
        [age_levels[0] - (age_levels[1] - age_levels[0]) / 2],
        (np.array(age_levels[:-1]) + np.array(age_levels[1:])) / 2,
        [age_levels[-1] + (age_levels[-1] - age_levels[-2]) / 2],
    ])
    norm = BoundaryNorm(bounds, cmap.N)
    cs = ax.contour(np.log10(tg), mg, age_grid, levels=age_levels,
                    cmap=cmap, norm=norm, linewidths=1.5, zorder=3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    ax.add_patch(plt.Rectangle((0.66, 0.80), 0.32, 0.18,
                               transform=ax.transAxes, zorder=4,
                               facecolor="white", edgecolor="none", alpha=0.25))
    cax = ax.inset_axes([0.67, 0.88, 0.3, 0.05])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=14)
    cb.set_label("Total Age [Gyr]", size=16)
    cb.set_ticks(age_levels)

    ylims = ax.get_ylim()

    mask = ifmr.teff_max > 4000
    z = np.polyfit(np.log10(ifmr.loc[mask, "teff_max"]), ifmr.loc[mask, "M_final"], 1)
    p = np.poly1d(z)

    min_t = ax.get_xlim()[1]
    max_t = np.log10(ifmr.teff_max.max())
    t_fit = np.linspace(min_t, max_t, 3)
    ax.set_ylim(0.2, 1.3)

    if not args.skipline:
        ax.fill_between(np.log10(ifmr.teff_max), ifmr.M_final, ylims[0],
                        hatch="///", facecolor="none", edgecolor="0.6", linewidth=0.4, zorder=1)
        #ax.fill_between(t_fit, p(t_fit), ylims[0],
        #                hatch="///", facecolor="none", edgecolor="0.6", linewidth=0.4, zorder=1)
        #ax.plot(t_fit, p(t_fit), c="k", lw=2, zorder=7)
        ax.plot(np.log10(ifmr.teff_max), ifmr.M_final, c="k", lw=2, zorder=7)
        #ax.scatter([max_t], [ifmr.M_final.min()], s=40, c="red", edgecolor="k", zorder=100)
        #ax.text(0.02, 0.07, "Forbidden By Single Star Evolution",
        #        ha="left", va="top", fontsize=18, transform=ax.transAxes,
        #        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
        ax.set_ylim(ylims)

    ticks_kK = np.array([50, 40, 30, 20, 10, 7, 5, 4])
    tick_locs = np.log10(ticks_kK * 1e3)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([f"{t:g}" for t in ticks_kK])
    ax.set_xlabel("Effective Temperature [$10^3$ K]")
    ax.set_ylabel("Mass [$M_\\odot$]")
    ax.invert_xaxis()
    ax.set_xlim(np.log10(55000), np.log10(3500))
    ax.legend(framealpha=0, loc="lower left")

    print(f"Mean Mass = {combined.Mass.mean():.3f}")

    out = FIGURES_DIR / "mass_teff.pdf"
    fig.savefig(out)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
