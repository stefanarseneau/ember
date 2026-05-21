"""Parallax and mass SNR improvement from wide-binary weighted parallaxes."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load_data import OUT_DIR, load_main_data, setup_matplotlib

plt, _ = setup_matplotlib()
data = load_main_data()
data = data[~np.isnan(data["total_age_lower_limit"])].copy()
data = data.query("Mass > 0.6 & 0 < total_age_lower_limit < 13").copy()

## === Figure 1: lower limit histogram ==================================

fig, ax = plt.subplots(figsize=(10,8))

mass_bins = np.arange(0.6, 1.21, 0.05)
_, bins = np.histogram(data["total_age_lower_limit"])

for ii in range(len(mass_bins) - 1):
	bin_subset = data.query(
		f"{mass_bins[ii]} < Mass & Mass <= {mass_bins[ii+1]}"
	)
	bin_name = rf"${mass_bins[ii]:1.2f}-{mass_bins[ii+1]:1.2f}~M_\odot$"
	print(f"Mass bin: {bin_name}: {len(bin_subset)}")

	ax.hist(
		bin_subset["total_age_lower_limit"], bins=bins, histtype="step", 
		label=bin_name, linewidth=3
	)

ax.legend(framealpha=0)
ax.set_xlabel("Age Lower Limit [Gyr]")
ax.set_ylabel("Count")

fig.savefig(OUT_DIR / "age_lower_limit_hist.pdf")
plt.close()
print(f"Saved {OUT_DIR / 'age_lower_limit_hist.pdf'}")

## === Figure 2: scatter plot ===========================================

fig, ax = plt.subplots(figsize=(10, 8))

ax.errorbar(
	data["total_age_lower_limit"], data["Mass"],
	yerr=[data["e_Mass_lower"], data["e_Mass_upper"]],
	mfc="white", mec="k", ecolor="k", fmt="o", capsize=4,
	rasterized=True
)

ax.set_xlabel("Age Lower Limit [Gyr]")
ax.set_ylabel(r"Mass [$M_\odot]$")
ax.set_xlim(0, 13)
ax.set_ylim(0.60, 1.31)

fig.savefig(OUT_DIR / "age_lower_limit_scatter.pdf")
plt.close()
print(f"Saved {OUT_DIR / 'age_lower_limit_scatter.pdf'}")


