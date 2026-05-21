"""Sample summary figure: HR diagram, separation, distance, magnitude histograms."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from load_data import OUT_DIR, load_main_data, setup_matplotlib

plt, _ = setup_matplotlib()
data = load_main_data()

wd_color = "#302424"
ms_color = "#3500b0"

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))

ax[0, 0].scatter(data.bp_rp, data.phot_g_mean_mag + 5 * np.log10(data.wtd_par) + 5,
                 s=1, c=wd_color, rasterized=True)
ax[0, 0].scatter(data.ms_bp_rp, data.ms_phot_g_mean_mag + 5 * np.log10(data.wtd_par) + 5,
                 s=1, c=ms_color, rasterized=True)
ax[0, 0].set_xlabel("BP-RP [mag]")
ax[0, 0].set_ylabel("$M_G$ [mag]")
ax[0, 0].invert_yaxis()

sep_logbins = np.logspace(np.log10(data.sep_AU.min()), np.log10(data.sep_AU.max()), 40)
ax[0, 1].hist(data.sep_AU, bins=sep_logbins, color=wd_color, histtype="step", linewidth=4)
ax[0, 1].set_xlabel("Separation [au]")
ax[0, 1].set_ylabel("N")
ax[0, 1].set_xscale("log")

dist = 1000 / data.wtd_par
dist_logbins = np.logspace(np.log10(dist.min()), np.log10(dist.max()), 40)
ax[1, 0].hist(dist, bins=dist_logbins, color=wd_color, histtype="step", linewidth=4)
ax[1, 0].set_xlabel("Distance [pc]")
ax[1, 0].set_ylabel("N")
ax[1, 0].set_xscale("log")

mag_lo = min(data.phot_g_mean_mag.min(), data.ms_phot_g_mean_mag.min())
mag_hi = max(data.phot_g_mean_mag.max(), data.ms_phot_g_mean_mag.max())
mag_bins = np.linspace(mag_lo, mag_hi, 40)
ax[1, 1].hist(data.phot_g_mean_mag, bins=mag_bins, color=wd_color, histtype="step", linewidth=4)
ax[1, 1].hist(data.ms_phot_g_mean_mag, bins=mag_bins, color=ms_color, histtype="step", linewidth=4)
ax[1, 1].set_xlabel("$G$ [mag]")
ax[1, 1].set_ylabel("N")

print(f"Mean WD G: {data.phot_g_mean_mag.mean():.2f}")
print(f"Mean MS G: {data.ms_phot_g_mean_mag.mean():.2f}")

fig.tight_layout()
fig.savefig(OUT_DIR / "sample_summary.pdf")
plt.close()
print(f"Saved {OUT_DIR / 'sample_summary.pdf'}")
