from pathlib import Path

REPO_ROOT      = Path(__file__).parent.parent
DATA_DIR       = REPO_ROOT / "data"
TYLER_CODE_DIR = REPO_ROOT / "tyler_code"
FIGURES_DIR    = REPO_ROOT / "figures"

# Survey priority for deduplication (lower rank = higher priority; GALAH wins)
SURVEY_PRIORITY = {"GALAH": 0, "APOGEE": 1, "ASTRA": 2, "LAMOST": 3}

# Abundance flags are nulled when flag > threshold for that survey
FLAG_THRESHOLD = {"GALAH": 1, "APOGEE": 0, "ASTRA": 0, "LAMOST": 0}

# Gaia BP/RP excess factor quality cut: keep rows where
#   phot_bp_rp_excess_factor > 3 * (BP_RP_A + BP_RP_B * phot_g_mean_mag**BP_RP_EXP)
BP_RP_A = 0.0059898
BP_RP_B = 8.817481e-12
BP_RP_EXP = 7.618399

# MS-companion MS/subgiant/giant classification (catalog.wdms.evolutionary_state).
# Two disjoint, fully hardcoded tests in (ms_bp_rp, M_G), split at
# GIANT_CMD_EVOLVED_MIN_BP_RP — no isochrones, no fitting, deliberately
# arbitrary:
#
#   subgiant — bp_rp < GIANT_CMD_EVOLVED_MIN_BP_RP and above a straight line
#     through (0,0) and (1, GIANT_CMD_SG_LINE_AT_1), i.e.
#     M_G < GIANT_CMD_SG_LINE_AT_1 * bp_rp.
#
#   giant — bp_rp >= GIANT_CMD_EVOLVED_MIN_BP_RP and
#     M_G < GIANT_CMD_GIANT_ABS_MAG_MAX.
GIANT_CMD_SG_LINE_AT_1 = 4.0
GIANT_CMD_EVOLVED_MIN_BP_RP = 1.0
GIANT_CMD_GIANT_ABS_MAG_MAX = 4.0

# Isochrone grid — plotting context only (catalog.analysis.ms_evol_cmd), not
# used by the classification itself.
GIANT_CMD_AGE_GRID_GYR = [0.3, 0.5, 1, 2, 3, 5, 7, 10, 13.5]
MIST_EVOL_GRID_CACHE = DATA_DIR / "build/mist_evol_grid.pqt"

STYLE_FILE       = REPO_ROOT / "stefan.mplstyle"
WDWD_STITCH_DIR  = DATA_DIR / "build/wdwd_stitch_input"

# El-Badry+2021 wide binary catalog (Gaia EDR3), zenodo record 4435257.
# ELBADRY_CACHE is a compact index (WD+WD and WD+MS rows only, 4 columns)
# built by catalog.build --elbadry from the full FITS download.
ELBADRY_URL   = "https://zenodo.org/api/records/4435257/files/all_columns_catalog.fits.gz/content"
ELBADRY_CACHE = DATA_DIR / "external/elbadry_wdpairs.pqt"
