"""MS lifetime vs WD mass, age isochrones, and metallicity sensitivity on total ages."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parent))
from load_data import DATA_DIR, OUT_DIR, TYLER_CODE_DIR, setup_matplotlib

wdmodels_dir = os.environ["WDMODELS_DIR"]
sys.path.append(wdmodels_dir)
import WD_models

plt, _ = setup_matplotlib()

model = WD_models.load_model(
    low_mass_model="Bedard2020",
    middle_mass_model="Bedard2020",
    high_mass_model="Bedard2020",
    atm_type="H",
    HR_bands=("bp3-rp3", "G3"),
)
interp_model = WD_models.interp_xy_z_func(
    x=model["mass_array"], y=10**model["logteff"], z=model["age_cool"]
)

ifmr = pd.read_csv(DATA_DIR / "MESA_IFMR/ifmr_data.csv")

ifmr_dir = str(TYLER_CODE_DIR / "init_mass_to_mslife")


def read_mist_grid(path: str):
    mass_i = np.load(os.path.join(path, "mi.npy"))
    mslife = np.load(os.path.join(path, "msl.npy"))
    idx = np.argsort(mass_i)
    mass_i, mslife = mass_i[idx], mslife[idx]
    mass_wd = np.interp(mass_i, ifmr.M_initial, ifmr.M_final)
    totalage = mslife + interp_model(mass_wd, 10000)
    return mass_i, mslife, mass_wd, totalage


massi_fehp000, mslife_fehp000, masswd_fehp000, totalage_fehp000 = read_mist_grid(ifmr_dir)
massi_fehp025, mslife_fehp025, masswd_fehp025, totalage_fehp025 = read_mist_grid(
    os.path.join(ifmr_dir, "mist_feh_p025")
)
massi_fehm025, mslife_fehm025, masswd_fehm025, totalage_fehm025 = read_mist_grid(
    os.path.join(ifmr_dir, "mist_feh_m025")
)
massi_fehm050, mslife_fehm050, masswd_fehm050, totalage_fehm050 = read_mist_grid(
    os.path.join(ifmr_dir, "mist_feh_m050")
)


# ── MS lifetime vs WD mass + isochrones ───────────────────────────────────
def teff_isochrone(total_age):
    masses, teffs = [], []
    for M in np.linspace(masswd_fehp000.min(), 0.90, 300):
        ms_life = np.interp(M, masswd_fehp000, mslife_fehp000)
        cool_age = total_age - ms_life
        if cool_age <= 0:
            continue
        try:
            T = brentq(lambda T: interp_model(M, T) - cool_age, 2000, 80000)
            masses.append(M)
            teffs.append(T)
        except Exception:
            pass
    return masses, teffs


# ── Metallicity sensitivity on MS lifetimes and total ages ────────────────
fehp025_color = "#2166ac"
fehm025_color = "#343434"
fehm050_color = "#b2182b"

fig, ax = plt.subplots(ncols=2, figsize=(14, 5))

with np.errstate(divide='ignore', invalid='ignore'):
    ax[0].plot(masswd_fehp025, (mslife_fehp025 - mslife_fehp000) / mslife_fehp000,
               lw=3, c=fehp025_color)
    ax[0].plot(masswd_fehm025, (mslife_fehm025 - mslife_fehp000) / mslife_fehp000,
               lw=3, c=fehm025_color)
    ax[0].plot(masswd_fehp025, (mslife_fehm050 - mslife_fehp000) / mslife_fehp000,
               lw=3, c=fehm050_color)
ax[0].axhline(0, c="k", ls="--")
ax[0].set_xlim(0.63, 0.96)
ax[0].set_ylim(-0.3, 0.2)
ax[0].set_xlabel(r"White Dwarf Mass [$M_\odot$]")
ax[0].set_ylabel(r"$\Delta \tau_\text{MS} / \tau_\text{MS}$")

ax[1].plot(masswd_fehp025, (totalage_fehp025 - totalage_fehp000) / totalage_fehp000,
           lw=3, label=r"[Fe/H] = +0.25", c=fehp025_color)
ax[1].plot(masswd_fehm025, (totalage_fehm025 - totalage_fehp000) / totalage_fehp000,
           lw=3, label=r"[Fe/H] = -0.25", c=fehm025_color)
ax[1].plot(masswd_fehp025, (totalage_fehm050 - totalage_fehp000) / totalage_fehp000,
           lw=3, label=r"[Fe/H] = -0.50", c=fehm050_color)
ax[1].axhline(0, c="k", ls="--")
ax[1].set_xlim(0.63, 0.96)
ax[1].set_ylim(-0.3, 0.2)
ax[1].set_xlabel(r"White Dwarf Mass [$M_\odot$]")
ax[1].set_ylabel(r"$\Delta \tau_\text{Total} / \tau_\text{Total}$")
ax[1].legend(framealpha=0)

fig.tight_layout()
fig.savefig(OUT_DIR / "metallicity_model.pdf")
plt.close()
print(f"Saved {OUT_DIR / 'metallicity_model.pdf'}")
