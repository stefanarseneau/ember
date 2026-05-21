"""Mass vs Teff diagram with crystallization sequences and IFMR forbidden region."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load_data import DATA_DIR, OUT_DIR, load_main_data, setup_matplotlib

wdmodels_dir = os.environ["WDMODELS_DIR"]
sys.path.append(wdmodels_dir)
import WD_models

plt, _ = setup_matplotlib()
data = load_main_data()

crys = pd.read_csv(DATA_DIR / "merge_data/crystallization.csv")
ifmr = pd.read_csv(DATA_DIR / "MESA_IFMR/ifmr_data.csv")

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

ax.scatter(np.log10(data.Teff), data.Mass,
           c="white", edgecolor="k", alpha=0.7, s=7, zorder=1, rasterized=True)
ax.axhline(np.nanmedian(data.Mass), c="k")
ax.plot(np.log10(teff_20pct), mass_20pct, c="lime", lw=3, label="20\\% Crystallized", zorder=2)
ax.plot(np.log10(teff_80pct), mass_80pct, c="lime", ls="--", lw=3, label="80\\% Crystallized", zorder=2)

ylims = ax.get_ylim()

mask = ifmr.teff_max > 4000
z = np.polyfit(np.log10(ifmr.loc[mask, "teff_max"]), ifmr.loc[mask, "M_final"], 1)
p = np.poly1d(z)

min_t = ax.get_xlim()[1]
max_t = np.log10(ifmr.teff_max.max())
t_fit = np.linspace(min_t, max_t, 3)

ax.fill_between(np.log10(ifmr.teff_max), ifmr.M_final, ylims[0],
                hatch="///", facecolor="none", edgecolor="0.6", linewidth=0.4, zorder=1)
ax.fill_between(t_fit, p(t_fit), ylims[0],
                hatch="///", facecolor="none", edgecolor="0.6", linewidth=0.4, zorder=1)
ax.plot(t_fit, p(t_fit), c="k", lw=2, zorder=7)
ax.plot(np.log10(ifmr.teff_max), ifmr.M_final, c="k", lw=2, zorder=7)
ax.scatter([max_t], [ifmr.M_final.min()], s=40, c="red", edgecolor="k", zorder=100)
ax.text(0.02, 0.07, "Forbidden By Single Star Evolution",
        ha="left", va="top", fontsize=18, transform=ax.transAxes,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
ax.set_ylim(ylims)

ticks_kK = np.array([50, 40, 30, 20, 10, 7, 5, 4])
tick_locs = np.log10(ticks_kK * 1e3)
ax.set_xticks(tick_locs)
ax.set_xticklabels([f"{t:g}" for t in ticks_kK])
ax.set_xlabel("Effective Temperature [$10^3$ K]")
ax.set_ylabel("Mass [$M_\\odot$]")
ax.invert_xaxis()
ax.set_xlim(np.log10(55000), np.log10(3500))
ax.legend(framealpha=0)

print(f"Mean Mass = {data.Mass.mean():.3f}")

fig.savefig(OUT_DIR / "mass_teff.pdf")
plt.close()
print(f"Saved {OUT_DIR / 'mass_teff.pdf'}")
