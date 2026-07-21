"""Sample summary figure: HR diagram, separation, distance, magnitude histograms."""

import numpy as np

from catalog.config import FIGURES_DIR
from catalog.analysis.load_data import load_catalog, setup_matplotlib


def main():
    plt, _ = setup_matplotlib()
    combined, *_ = load_catalog()

    wd_color = "#302424"
    ms_color = "#3500b0"

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))

    ax[0, 0].scatter(combined.bp_rp, combined.phot_g_mean_mag + 5 * np.log10(combined.wtd_par) + 5,
                     s=1, c=wd_color, rasterized=True)
    ax[0, 0].scatter(combined.ms_bp_rp, combined.ms_phot_g_mean_mag + 5 * np.log10(combined.wtd_par) + 5,
                     s=1, c=ms_color, rasterized=True)
    ax[0, 0].set_xlabel("BP-RP [mag]")
    ax[0, 0].set_ylabel("$M_G$ [mag]")
    ax[0, 0].invert_yaxis()

    sep_logbins = np.logspace(np.log10(combined.sep_AU.min()), np.log10(combined.sep_AU.max()), 40)
    ax[0, 1].hist(combined.sep_AU, bins=sep_logbins, color=wd_color, histtype="step", linewidth=4)
    ax[0, 1].set_xlabel("Separation [au]")
    ax[0, 1].set_ylabel("N")
    ax[0, 1].set_xscale("log")

    dist = 1000 / combined.wtd_par
    dist_logbins = np.logspace(np.log10(dist.min()), np.log10(dist.max()), 40)
    ax[1, 0].hist(dist, bins=dist_logbins, color=wd_color, histtype="step", linewidth=4)
    ax[1, 0].set_xlabel("Distance [pc]")
    ax[1, 0].set_ylabel("N")
    ax[1, 0].set_xscale("log")

    mag_lo = min(combined.phot_g_mean_mag.min(), combined.ms_phot_g_mean_mag.min())
    mag_hi = max(combined.phot_g_mean_mag.max(), combined.ms_phot_g_mean_mag.max())
    mag_bins = np.linspace(mag_lo, mag_hi, 40)
    ax[1, 1].hist(combined.phot_g_mean_mag, bins=mag_bins, color=wd_color, histtype="step", linewidth=4)
    ax[1, 1].hist(combined.ms_phot_g_mean_mag, bins=mag_bins, color=ms_color, histtype="step", linewidth=4)
    ax[1, 1].set_xlabel("$G$ [mag]")
    ax[1, 1].set_ylabel("N")

    print(f"Mean WD G: {combined.phot_g_mean_mag.mean():.2f}")
    print(f"Mean MS G: {combined.ms_phot_g_mean_mag.mean():.2f}")

    fig.tight_layout()
    out = FIGURES_DIR / "sample_summary.pdf"
    fig.savefig(out)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
