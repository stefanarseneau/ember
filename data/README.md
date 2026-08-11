# data/

Layout follows the pipeline stage a file belongs to. Paths below are all
relative to `data/`; code refers to them via `DATA_DIR` (`catalog/config.py`).

## raw/ — unmodified upstream inputs

- `tyler/` — Tyler's original WD+MS and WD+WD age catalogs.
  - `WDMS_total_ages_correct_models_cut_down.csv` — main WD+MS catalog, read by
    `catalog/wdms/build.py` and `catalog/analysis/exoplanet_ages.py`.
  - `tyler_wdms_GALAH.csv`, `tyler_wdms_APOGEE.csv`, `tyler_wdms_ASTRA.csv` —
    per-survey abundance tables, read by `catalog/wdms/build.py`.
  - `tyler_wdwd.csv` — WD+WD per-component catalog, read by `catalog/wdwd/build.py`.
- `mesa_ifmr/` — MESA initial-final mass relation tracks.
  - `ifmr_data.csv` — read by `catalog/analysis/ifmr_comparison.py` and
    `catalog/analysis/ms_lifetimes.py`.
  - `MESA_IFMR_missing_one_point.csv` — companion raw file, not currently read
    by `catalog/` (kept alongside `ifmr_data.csv` as it's part of the same
    upstream export, unlike the files in `_archive/`).

## external/ — third-party crossmatches actually consumed by the pipeline

- `wdms_lamost.csv`, `wdms_lamost_efeh.csv` — LAMOST WD+MS crossmatch + cached
  VizieR [Fe/H] errors, read by `catalog/wdms/build.py` (`_process_lamost`).
- `stellarhosts.csv` — NASA Exoplanet Archive stellar hosts table, read by
  `catalog/analysis/exoplanet_table.py`.
- `crystallization.csv` — WD crystallization tracks, read by
  `catalog/analysis/mass_teff.py`.
- `elbadry_wdpairs.pqt` — cached compact index of El-Badry+2021 wide binaries
  (Gaia EDR3, zenodo 4435257), built by `catalog.wdwd.build --elbadry` and
  read by `catalog/wdwd/build.py` (`ELBADRY_CACHE` in `catalog/config.py`).

## build/ — intermediate build inputs/artifacts

- `wdwd_stitch_input/` (`thick.pqt`, `thin.pqt`, `mixed.pqt`) — per-population
  SED fit parquets, stitched by `catalog/wdwd/stitch.py` into `catalogs/ages.pqt`.
- `mist_evol_grid.pqt` — cached MIST isochrone grid for CMD plotting context
  (`catalog/analysis/ms_evol_cmd.py`), `MIST_EVOL_GRID_CACHE` in `catalog/config.py`.
- `correction_coefficients.npy` — cool-WD Teff/mass correction polynomial,
  written by `catalog/corrections.py`, read by `catalog/analysis/mass_teff.py`.
- Per-population checkpoint files (`combined_{label}_{pop}_ckpt.parquet`) also
  land here while `catalog/wdms/build.py` re-measures cool-WD ages.

## catalogs/ — final pipeline outputs (the actual data products)

- `combined.pqt` / `.csv` — main WD+MS catalog, output of `catalog/wdms/build.py`,
  read by everything under `catalog/analysis/` via `load_data.load_catalog()`.
- `metallicity.pqt` / `.csv` — long-form (source, survey) abundance table,
  output of `catalog/wdms/build.py`, read via `load_data.load_metallicity()`.
- `ages.pqt` — WD+WD stitched age catalog, output of `catalog/wdwd/stitch.py`,
  consumed by `catalog/wdwd/build.py`.
- `wdwd_pairs.pqt` — final WD+WD pair catalog, output of `catalog/wdwd/build.py`,
  read by `catalog/analysis/wdwd_inconsistent.py`.

## exoplanets/ — self-contained exoplanet-age sub-pipeline

Inputs and outputs of `catalog/analysis/exoplanet_ages.py`
(`EXO_DIR` in that file): `wdms_exoplanets_all.csv` (input), fitted
photometry/chains, and the resulting age tables/figures.

## _archive/ — unreferenced leftovers, kept for safety

Files with no remaining code references anywhere in `catalog/` or
`tyler_code/` as of the `data/` reorganization (2026-08-11) — mostly
byproducts of now-deleted exploratory notebooks. Not deleted outright since
they're either not worth the risk of losing something useful, or not worth
the time to verify each one individually. Safe to delete for good once
confirmed unneeded; `data/` is git-tracked so they're recoverable from history
regardless.
