"""IFMR comparison: Bauer+2026 (from COOLING_PATH tracks) vs Fields+2016 (CSV)."""

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load_data import DATA_DIR, OUT_DIR, setup_matplotlib

plt, _ = setup_matplotlib()

coolingdir = os.environ["COOLING_PATH"]
paths = glob.glob(os.path.join(coolingdir, "feh_p000_afe_p0_wds_vvcrit0.4", "Tracks", "*"))

wd_masses = [
    float(re.search(r"(\d+\.\d+)(?=\.data$)", p).group(1))
    for p in paths
]

pattern = re.compile(r"#\s*M_in\s*=\s*([\d.]+)")
m_in_values = []
for path in paths:
    with open(path) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                m_in_values.append(float(match.group(1)))
                break

ifmr = pd.read_csv(DATA_DIR / "MESA_IFMR/ifmr_data.csv")

wd_masses   = np.array(wd_masses)
m_in_values = np.array(m_in_values)
idx = np.argsort(wd_masses)

fig, ax = plt.subplots()
ax.plot(wd_masses[idx], m_in_values[idx], c="#4978d1", lw=5, ls="--", label="Bauer+2026 IFMR")
ax.plot(ifmr.M_final, ifmr.M_initial, c="#4b4b4b", lw=5, label="Fields+2016 IFMR", zorder=0)
ax.set_xlabel(r"White Dwarf Mass [$M_\odot$]")
ax.set_ylabel(r"Progenitor Mass [$M_\odot$]")
ax.legend(framealpha=0)

interp_diff = np.mean(np.interp(ifmr.M_initial, m_in_values, wd_masses) - ifmr.M_final)
print(f"Mean IFMR M_WD offset (Bauer - Fields): {interp_diff:.4f} M_sun")

fig.savefig(OUT_DIR / "ifmr_comparison.pdf")
plt.close()
print(f"Saved {OUT_DIR / 'ifmr_comparison.pdf'}")
