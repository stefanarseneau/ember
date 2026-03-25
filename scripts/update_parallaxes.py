"""
For every source in omnidwarf_fluxes.pqt that belongs to a wide WD+WD or
WD+MS binary (El-Badry catalog, R_chance_align < 0.1), join in ra, dec,
parallax, and parallax_error from the catalog, replace the individual parallax
with the inverse-variance weighted mean of the pair, and add two columns:
wb_companion_id and wb_binary_type.
"""
import numpy as np
import pandas as pd
FLUXES_PATH  = '../omnidwarf_fluxes.pqt'
OUT_PATH = 'widebinary_fluxes.pqt'
CATALOG_PATH = '/home/arseneau/observational/catalogs/elbadry_widebinary.pqt'
R_CHANCE_MAX = 0.1

# --------------------------------------------------------------------------
fluxes = pd.read_parquet(FLUXES_PATH)
wb     = pd.read_parquet(CATALOG_PATH)

# Restrict to high-confidence pairs
wb = wb[wb['R_chance_align'] < R_CHANCE_MAX][
    ['source_id1', 'source_id2', 'binary_type',
     'ra1', 'dec1', 'parallax1', 'parallax_error1',
     'ra2', 'dec2', 'parallax2', 'parallax_error2']
].copy()

# Keep pairs where at least one component is a WD in our flux table
s = set(fluxes['gaia_dr3_source_id'])
wb = wb[wb['source_id1'].isin(s) | wb['source_id2'].isin(s)].copy()
in1 = wb['source_id1'].isin(s)
in2 = wb['source_id2'].isin(s)
print(f'Pairs with at least one WD in fluxes: {len(wb)}')

# Flatten catalog into a per-source table with astrometry from the catalog
comp1 = wb[in1][['source_id1', 'source_id2', 'binary_type',
            'ra1', 'dec1', 'parallax1', 'parallax_error1']].rename(columns={
    'source_id1': 'gaia_dr3_source_id', 'source_id2': 'wb_companion_id',
    'ra1': 'ra', 'dec1': 'dec',
    'parallax1': 'parallax', 'parallax_error1': 'parallax_error',
})
comp2 = wb[in2][['source_id2', 'source_id1', 'binary_type',
            'ra2', 'dec2', 'parallax2', 'parallax_error2']].rename(columns={
    'source_id2': 'gaia_dr3_source_id', 'source_id1': 'wb_companion_id',
    'ra2': 'ra', 'dec2': 'dec',
    'parallax2': 'parallax', 'parallax_error2': 'parallax_error',
})
astrom = pd.concat([comp1, comp2], ignore_index=True).rename(
    columns={'binary_type': 'wb_binary_type'}
)
del comp1, comp2

# Extract source IDs and parallaxes separately to preserve int64 precision
# (mixing with float columns in a single .values array upcasts IDs to float64,
#  losing precision for ~10^18 Gaia source IDs and breaking index lookups)
pair_src = wb[['source_id1', 'source_id2']].values                              # int64
pair_plx = wb[['parallax1', 'parallax_error1', 'parallax2', 'parallax_error2']].values  # float64
del wb

# Join astrometry into fluxes, keeping only wide binary members
fluxes = fluxes.merge(astrom, on='gaia_dr3_source_id', how='inner')
del astrom
print(f'Sources retained (wide binary members): {len(fluxes)}')

# Compute weighted mean parallax for each pair and write it back
fluxes = fluxes.set_index('gaia_dr3_source_id', drop=False)

for (id1, id2), (plx1, e1, plx2, e2) in zip(pair_src, pair_plx):
    if not (np.isfinite(plx1) and np.isfinite(plx2) and e1 > 0 and e2 > 0):
        continue
    w1, w2   = 1 / e1**2, 1 / e2**2
    plx_mean = (w1 * plx1 + w2 * plx2) / (w1 + w2)
    e_mean   = 1 / np.sqrt(w1 + w2)
    for sid in [id1, id2]:
        if sid in fluxes.index:
            fluxes.at[sid, 'parallax']       = plx_mean
            fluxes.at[sid, 'parallax_error'] = e_mean

n_updated = fluxes['wb_companion_id'].notna().sum()
print(f'WDs updated with weighted mean parallax: {n_updated}')

fluxes = fluxes.reset_index(drop=True).drop(columns=[
    "Sdss_flux_u", "Sdss_flux_error_u", 
    "Sdss_flux_g", "Sdss_flux_error_g",
    "Sdss_flux_r", "Sdss_flux_error_r",])
fluxes.to_parquet(OUT_PATH)
print(f'Saved to {OUT_PATH}')
