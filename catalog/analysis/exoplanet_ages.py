import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from catalog.config import DATA_DIR
from catalog.analysis.load_data import load_catalog, setup_matplotlib


def _with_retries(fn, *args, retries=5, delay=30, **kwargs):
    """Call fn(*args, **kwargs), retrying on network errors up to `retries` times."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except (ConnectionResetError, ConnectionError, OSError) as exc:
            if attempt == retries - 1:
                raise
            wait = delay * (attempt + 1)
            print(f"  Network error ({exc}); retrying in {wait}s "
                  f"(attempt {attempt + 1}/{retries}) …")
            time.sleep(wait)


# ── Paths ─────────────────────────────────────────────────────────────────────
NICOLA_CAT    = Path.home() / "observational/catalogs/nicola_wds/ngf21_wds.pqt"
EXO_DIR    = DATA_DIR / "exoplanets"
EXOPLANET_CSV = EXO_DIR / "wdms_exoplanets_all.csv"


CHAINS_DIR = EXO_DIR / "chains"

PHOT_INPUT  = EXO_DIR / "phot_input.parquet"
PHOTOMETRY  = EXO_DIR / "photometry.parquet"
PHOT_FIT    = EXO_DIR / "phot_fit.parquet"
AGES_OUTPUT = EXO_DIR / "ages.parquet"

SYSTEMS = ["gaia"]


# ── Step 1: load exoplanet list ───────────────────────────────────────────────

def load_exoplanets():
	df = pd.read_csv(EXOPLANET_CSV)
	df["wd_source_id"] = df["wd_source_id"].astype(np.int64)
	print(f"Loaded {len(df)} exoplanet systems from {EXOPLANET_CSV.name}")
	return df


# ── Step 2: attach extinction from Nicola's catalog ──────────────────────────

def attach_nicola(df):
	"""Merge WD source IDs against Nicola+2021 catalog for meanAV and positions."""
	nicola = (
		pd.read_parquet(NICOLA_CAT)[["GaiaEDR3", "RA_ICRS", "DE_ICRS", "meanAV"]]
		.rename(columns={
			"GaiaEDR3": "wd_source_id",
			"RA_ICRS":  "ra",
			"DE_ICRS":  "dec",
		})
	)
	nicola["wd_source_id"] = nicola["wd_source_id"].astype(np.int64)

	df = pd.merge(df, nicola, on="wd_source_id", how="left")
	n = df["meanAV"].notna().sum()
	print(f"Matched {n}/{len(df)} WDs to Nicola's catalog (meanAV)")
	if n < len(df):
		missing = df.loc[df["meanAV"].isna(), "wd_source_id"].tolist()
		print(f"  Missing meanAV for source IDs: {missing}  →  setting meanAV = 0.0")
	return df


# ── Step 3: build fit-seds input ──────────────────────────────────────────────

def _parse_coord(ra_str, dec_str):
	"""Parse RA/Dec to decimal degrees; handles both decimal and sexagesimal strings."""
	try:
		return float(ra_str), float(dec_str)
	except ValueError:
		c = SkyCoord(ra=str(ra_str), dec=str(dec_str), unit=(u.hourangle, u.deg))
		return float(c.ra.deg), float(c.dec.deg)


def build_input(df):
	"""Construct the DataFrame expected by sedtool.photometry.process_dataframe."""
	rows = []
	for _, row in df.iterrows():
		if pd.notna(row.get("ra")) and pd.notna(row.get("dec")):
			ra_deg, dec_deg = float(row["ra"]), float(row["dec"])
		else:
			ra_deg, dec_deg = _parse_coord(row["wd_ra"], row["wd_dec"])
		rows.append({
			"gaia_dr3_source_id": int(row["wd_source_id"]),
			"ra":               ra_deg,
			"dec":              dec_deg,
			"parallax":         float(row["wtd_plx"]),
			"parallax_error":   float(row["wtd_e_plx"]),
			"meanAV":           float(row["meanAV"]) if pd.notna(row.get("meanAV")) else 0.0,
			"ms_source_id":     int(row["ms_source_id"]) if pd.notna(row.get("ms_source_id")) else -1,
			"wtd_plx":          float(row["wtd_plx"]),
			"wtd_e_plx":        float(row["wtd_e_plx"]),
			"flux_leakage":     str(row.get("flux_leakage", "")),
			"notes":            str(row.get("notes", "")),
		})
	return pd.DataFrame(rows)


# ── Step 4: photometry retrieval (cached) ─────────────────────────────────────

def get_photometry(fit_in):
	"""Return synphot, flux_dict, lambda_eff, ext_vec — from cache if available."""
	import sedtool.photometry as sed_phot

	meta_cols = ["ms_source_id", "wtd_plx", "wtd_e_plx", "flux_leakage", "notes"]

	if PHOTOMETRY.exists():
		print(f"Using cached photometry → {PHOTOMETRY}")
		synphot = pd.read_parquet(PHOTOMETRY)
		from sedtool import sed_util
		flux_dict  = sed_util.find_photocols(synphot)
		import interpolator
		ff = interpolator.atmos.sed.get_default_filters()
		band_map = {
			"gaia":      ["Gaia_G", "Gaia_BP", "Gaia_RP"],
			"sdss":      ["SDSS_u", "SDSS_g", "SDSS_r", "SDSS_i", "SDSS_z"],
			"panstarrs": ["PS1_g", "PS1_r", "PS1_i", "PS1_z", "PS1_y"],
			"skymapper": ["SkyMapper_u", "SkyMapper_v", "SkyMapper_g", "SkyMapper_r", "SkyMapper_i", "SkyMapper_z"]
		}
		lambda_eff = np.array([ff[b].lambda_eff for s in SYSTEMS for b in band_map[s]])
		ext_vec    = sed_util.fetch_extinction(lambda_eff)
	else:
		print(f"\nRetrieving photometry ({', '.join(SYSTEMS)}) for {len(fit_in)} objects …")
		synphot, flux_dict, lambda_eff, ext_vec = _with_retries(
			sed_phot.process_dataframe, fit_in, systems=SYSTEMS
		)
		synphot.to_parquet(PHOTOMETRY, index=False)
		print(f"Saved photometry → {PHOTOMETRY}  ({len(synphot)} rows, {len(flux_dict)} bands)")

	present = [c for c in meta_cols if c in fit_in.columns and c not in synphot.columns]
	if present:
		synphot = synphot.merge(
			fit_in[["gaia_dr3_source_id"] + present],
			on="gaia_dr3_source_id", how="left",
		)

	id_map = fit_in.set_index("gaia_dr3_source_id")
	for col in ("ra", "dec"):
		if col in synphot.columns and col in id_map.columns:
			mask = synphot[col].isna()
			if mask.any():
				synphot.loc[mask, col] = synphot.loc[mask, "gaia_dr3_source_id"].map(id_map[col])

	return synphot, flux_dict, lambda_eff, ext_vec


def run_mcmc(synphot, flux_dict, lambda_eff, ext_vec):
	"""Run MCMC for sources that don't yet have a saved chain file."""
	import sedtool.fitting  as sed_fit
	import sedtool.sed_util as sed_util

	CHAINS_DIR.mkdir(exist_ok=True)

	existing_ids = {
		int(p.stem) for p in CHAINS_DIR.glob("*.npy") if p.stat().st_size > 0
	}
	to_fit = synphot[~synphot["gaia_dr3_source_id"].isin(existing_ids)].copy()

	if to_fit.empty:
		print(f"All MCMC chains already exist in {CHAINS_DIR}  (skipping)")
		return

	print(f"\nRunning MCMC for {len(to_fit)}/{len(synphot)} sources "
	      f"({len(existing_ids)} already cached) …")

	flux_cols = [fc for fc, _ in flux_dict.values()]
	print(f"\n{'source_id':<22}  available bands")
	print("-" * 80)
	for _, row in synphot.iterrows():
		sid  = int(row["gaia_dr3_source_id"])
		good = [band for band, (fc, _) in flux_dict.items() if pd.notna(row.get(fc))]
		print(f"{sid:<22}  {', '.join(good)}")
	print()

	band_names = sed_util.convert_names(list(flux_dict.keys()))
	interp, _  = sed_util.make_interpolator(band_names, units="fnu")
	logg_func  = sed_util.get_logg_function()

	sed_fit.fit_mcmc(
		to_fit, flux_dict, ext_vec, interp, logg_func,
		use_gravz=False, units="fnu",
		outfile=str(CHAINS_DIR),
	)


def summarise_chains(synphot):
	"""Read all chain files and build a summary DataFrame for measure-ages."""
	if PHOT_FIT.exists():
		print(f"Using cached chain summary → {PHOT_FIT}")
		return pd.read_parquet(PHOT_FIT)

	rows = []
	for chain_file in sorted(CHAINS_DIR.glob("*.npy")):
		source_id = int(chain_file.stem)
		chain     = np.load(chain_file)
		flat      = chain.reshape(-1, chain.shape[-1]) if chain.ndim == 3 else chain
		teff_samp   = flat[:, 0]
		radius_samp = flat[:, 1]
		rows.append({
			"gaia_dr3_source_id": source_id,
			"teff_best":   float(np.median(teff_samp)),
			"std_tt_best": float(np.std(teff_samp)),
			"radius_best": float(np.median(radius_samp)),
			"std_rr_best": float(np.std(radius_samp)),
			"teff_hi":     float(np.percentile(teff_samp,   84)),
			"teff_lo":     float(np.percentile(teff_samp,   16)),
			"radius_hi":   float(np.percentile(radius_samp, 84)),
			"radius_lo":   float(np.percentile(radius_samp, 16)),
		})

	fit_df    = pd.DataFrame(rows)
	meta_cols = [c for c in synphot.columns if c not in fit_df.columns]
	fit_df    = pd.merge(
		fit_df, synphot[["gaia_dr3_source_id"] + meta_cols],
		on="gaia_dr3_source_id", how="left",
	)

	fit_df.to_parquet(PHOT_FIT, index=False)
	print(f"Saved chain summary → {PHOT_FIT}  ({len(fit_df)} rows)")
	return fit_df


def run_measure_ages():
	if AGES_OUTPUT.exists():
		print(f"Using cached ages → {AGES_OUTPUT}")
		return

	cmd = [
		"measure-ages",
		"--inpath",   str(PHOT_FIT),
		"--outpath",  str(AGES_OUTPUT),
		"--teff",     "teff_best",
		"--e_teff",   "std_tt_best",
		"--radius",   "radius_best",
		"--e_radius", "std_rr_best",
	]
	print(f"\n$ {' '.join(cmd)}")
	subprocess.run(cmd, check=True)


def compare_to_catalog(ages):
	"""Print side-by-side comparison of MCMC results vs Tyler's main catalog."""
	tyler = pd.read_csv(DATA_DIR / "raw/tyler/WDMS_total_ages_correct_models_cut_down.csv")
	tyler["source_id"] = tyler["source_id"].astype(np.int64)

	try:
		combined, *_ = load_catalog()
		sh = combined.drop_duplicates(subset=["ms_source_id"])[["ms_source_id", "hostname", "sy_pnum"]]
		tyler = pd.merge(tyler, sh.rename(columns={"ms_source_id": "ms_source_id_sh"}),
		                 left_on="ms_source_id", right_on="ms_source_id_sh", how="left")
	except Exception:
		tyler["hostname"] = ""
		tyler["sy_pnum"]  = np.nan

	comp = pd.merge(
		ages[["gaia_dr3_source_id", "teff_best", "std_tt_best",
		      "radius_best", "std_rr_best", "mass", "log_age_cool", "log_age"]],
		tyler[["source_id", "hostname", "sy_pnum", "sep_AU",
		       "Teff", "Mass",
		       "cool_age", "tot_age", "e_cool_age_upper", "e_cool_age_lower",
		       "tot_age_error_upper", "tot_age_error_lower"]].rename(columns={
			"source_id": "gaia_dr3_source_id",
			"Teff":      "tyler_teff",
			"Mass":      "tyler_mass",
		}),
		on="gaia_dr3_source_id",
	)

	if comp.empty:
		print("No overlap with Tyler's main catalog.")
		return

	W = 110
	print(f"\n{'─'*W}")
	print(f"Comparison with Tyler's catalog ({len(comp)} overlapping sources)\n")
	print(f"{'hostname':<16} {'source_id':<22}  "
	      f"{'Teff_MCMC':>10} {'Teff_Tyler':>10}  "
	      f"{'Mass_MCMC':>10} {'Mass_Tyler':>10}  "
	      f"{'CoolAge_MCMC':>13} {'CoolAge_Tyler':>13}  "
	      f"{'TotAge_MCMC':>12} {'TotAge_Tyler':>12}")
	print("─" * W)

	for _, r in comp.iterrows():
		cool_gyr = 10 ** (r.log_age_cool - 9) if np.isfinite(r.log_age_cool) else np.nan
		tot_gyr  = 10 ** (r.log_age      - 9) if np.isfinite(r.log_age)      else np.nan
		print(
			f"{r.hostname:<16} {int(r.gaia_dr3_source_id):<22}  "
			f"{r.teff_best:>10.0f} {r.tyler_teff:>10.0f}  "
			f"{r.mass:>10.3f} {r.tyler_mass:>10.3f}  "
			f"{cool_gyr:>13.3f} {r.cool_age:>13.3f}  "
			f"{tot_gyr:>12.3f} {r.tot_age:>12.3f}"
		)
	print("─" * W)


def write_latex_table(ages):
	"""Write a LaTeX table of MCMC WD parameters."""

	def _pm(val, hi, lo, fmt="{:.2f}"):
		if not np.isfinite(val):
			return r"$\cdots$"
		e_hi = hi - val if (hi is not None and np.isfinite(hi)) else np.nan
		e_lo = val - lo if (lo is not None and np.isfinite(lo)) else np.nan
		if not (np.isfinite(e_hi) and np.isfinite(e_lo)):
			return rf"${fmt.format(val)}$"
		return rf"${fmt.format(val)}^{{+{fmt.format(e_hi)}}}_{{-{fmt.format(e_lo)}}}$"

	def _log_to_gyr_pm(med, hi, lo):
		if not np.isfinite(med):
			return r"$\cdots$"
		v  = 10 ** (med - 9)
		vh = 10 ** (hi  - 9) if (hi is not None and np.isfinite(hi)) else np.nan
		vl = 10 ** (lo  - 9) if (lo is not None and np.isfinite(lo)) else np.nan
		return _pm(v, vh, vl, "{:.2f}")

	tex_rows = []
	for _, r in ages.sort_values("gaia_dr3_source_id").iterrows():
		exo_name = str(r["exoplanet_name"])
		exo_name = exo_name.replace("&", "\&") if "&" in exo_name else exo_name
		wd_id = int(r["gaia_dr3_source_id"])
		ms_id = int(r["ms_source_id"]) if pd.notna(r.get("ms_source_id", np.nan)) else r"$\cdots$"
		sep_au = float(r["sep_AU"])

		teff_str = _pm(r.get("teff_best"), r.get("teff_hi"), r.get("teff_lo"), "{:.0f}")
		mass_str = _pm(r.get("mass"),      r.get("mass_hi"), r.get("mass_lo"), "{:.3f}")
		cool_str = _log_to_gyr_pm(
			r.get("log_age_cool"), r.get("log_age_cool_hi"), r.get("log_age_cool_lo")
		)

		mass    = r.get("mass",      np.nan)
		teff    = r.get("teff_best", np.nan)
		lac     = r.get("log_age",      np.nan)
		lac_hi  = r.get("log_age_hi",   np.nan)
		lac_lo  = r.get("log_age_lo",   np.nan)

		totalage  = np.isfinite(mass) and mass > 0.63 and np.isfinite(teff) and teff > 3200
		e_upper_gyr = (10 ** (lac_hi - 9) - 10 ** (lac - 9)) if (np.isfinite(lac) and np.isfinite(lac_hi)) else np.nan
		age_lowlim  = e_upper_gyr > 13 if np.isfinite(e_upper_gyr) else True

		if totalage and not age_lowlim:
			tot_str = _log_to_gyr_pm(lac, lac_hi, lac_lo)
		elif totalage and age_lowlim or (not totalage and np.isfinite(mass) and mass > 0.6):
			lower_limit = 10 ** (lac_lo - 9) if np.isfinite(lac_lo) else np.nan
			tot_str = rf"$>{lower_limit:.2f}$" if np.isfinite(lower_limit) else r"$\cdots$"
		else:
			tot_str = r"$\cdots$"

		ra   = r.get("ra",  np.nan)
		dec  = r.get("dec", np.nan)
		plx  = r.get("wtd_plx",   np.nan)
		eplx = r.get("wtd_e_plx", np.nan)

		ra_str  = f"${ra:.4f}$"  if np.isfinite(ra)  else r"$\cdots$"
		dec_str = f"${dec:.4f}$" if np.isfinite(dec) else r"$\cdots$"
		plx_str = (rf"${plx:.2f} \pm {eplx:.2f}$"
		           if np.isfinite(plx) and np.isfinite(eplx) else r"$\cdots$")
		new_str = r"\tablenotemark{{\scriptsize a}}" if r.get("new_fit", True) else r""

		tex_rows.append(
			rf"  {exo_name} & {wd_id} & {ms_id} & {sep_au} & {plx_str} & {teff_str} & {mass_str} & {cool_str} & {tot_str}{new_str} \\"
		)

	table = "\n".join([
		r"\begin{table*}",
		r"\centering",
		r"\caption{White dwarf companion parameters for exoplanet host stars measured via SED MCMC fitting.}",
		r"\label{tab:exo_wd_ages}",
		r"\begin{tabular}{lllcccccc}",
		r"\hline\hline",
		(r"Exoplanet & WD Gaia DR3 ID & MS Gaia DR3 ID & $\varpi$ (mas)"
		 r" & $T_{\rm eff}$ (K) & Mass ($M_\odot$) & $\tau_{\rm cool}$ (Gyr) & $\tau_{\rm tot}$ [Gyr] \\"),
		r"\hline",
	] + tex_rows + [
		r"\hline",
		r"\end{tabular}",
		r"\end{table*}",
	])

	out_path = EXO_DIR / "ages_table.tex"
	out_path.write_text(table + "\n")
	print(f"\nLaTeX table → {out_path}\n")
	print(table)


def get_catalog_ids(exo):
	catalog_path = DATA_DIR / "raw/tyler/WDMS_total_ages_correct_models_cut_down.csv"
	tyler_ids = set(
		pd.read_csv(catalog_path, usecols=["source_id"])["source_id"].astype(np.int64)
	)
	return set(exo["wd_source_id"].astype(np.int64)) & tyler_ids


def load_catalog_ages(exo_full, catalog_ids):
	catalog_path = DATA_DIR / "raw/tyler/WDMS_total_ages_correct_models_cut_down.csv"
	tyler = pd.read_csv(catalog_path)
	tyler["source_id"] = tyler["source_id"].astype(np.int64)

	cat_exo = exo_full[exo_full["wd_source_id"].isin(catalog_ids)].copy()
	merged  = cat_exo.merge(
		tyler.rename(columns={"source_id": "wd_source_id"}),
		on="wd_source_id", how="inner", suffixes=("", "_tyler"),
	)

	def _log(gyr):
		return np.log10(gyr * 1e9) if (np.isfinite(gyr) and gyr > 0) else np.nan

	rows = []
	for _, r in merged.iterrows():
		ra  = r.get("ra")  if pd.notna(r.get("ra"))  else r.get("wd_ra",  np.nan)
		dec = r.get("dec") if pd.notna(r.get("dec")) else r.get("wd_dec", np.nan)
		try:
			ra, dec = float(ra), float(dec)
		except (TypeError, ValueError):
			ra, dec = _parse_coord(str(ra), str(dec))

		teff, e_tlo, e_thi = float(r.get("Teff", np.nan)), float(r.get("e_Teff_lower", np.nan)), float(r.get("e_Teff_upper", np.nan))
		mass, e_mlo, e_mhi = float(r.get("Mass", np.nan)), float(r.get("e_Mass_lower", np.nan)), float(r.get("e_Mass_upper", np.nan))
		cool, e_clo, e_chi = float(r.get("cool_age", np.nan)), float(r.get("e_cool_age_lower", np.nan)), float(r.get("e_cool_age_upper", np.nan))
		tot,  e_tolo, e_tohi = float(r.get("tot_age", np.nan)), float(r.get("tot_age_error_lower", np.nan)), float(r.get("tot_age_error_upper", np.nan))

		rows.append({
			"gaia_dr3_source_id": int(r["wd_source_id"]),
			"ms_source_id":       int(r["ms_source_id"]) if pd.notna(r.get("ms_source_id")) else -1,
			"sep_AU": 	float(r["sep_AU"]),
			"ra":        ra,  "dec": dec,
			"wtd_plx":   float(r.get("wtd_plx",   r.get("wtd_par",   np.nan))),
			"wtd_e_plx": float(r.get("wtd_e_plx", r.get("e_wtd_par", np.nan))),
			"teff_best":   teff,
			"teff_hi":     teff + e_thi if np.isfinite(e_thi) else np.nan,
			"teff_lo":     teff - e_tlo if np.isfinite(e_tlo) else np.nan,
			"std_tt_best": np.nan,
			"mass":     mass,
			"mass_hi":  mass + e_mhi if np.isfinite(e_mhi) else np.nan,
			"mass_lo":  mass - e_mlo if np.isfinite(e_mlo) else np.nan,
			"log_age_cool":    _log(cool),
			"log_age_cool_hi": _log(cool + e_chi)  if np.isfinite(e_chi)  else np.nan,
			"log_age_cool_lo": _log(cool - e_clo)  if (np.isfinite(e_clo) and cool > e_clo) else np.nan,
			"log_age":         _log(tot),
			"log_age_hi":      _log(tot + e_tohi)  if np.isfinite(e_tohi)  else np.nan,
			"log_age_lo":      _log(tot - e_tolo)  if (np.isfinite(e_tolo) and tot > e_tolo) else np.nan,
			"new_fit": False,
		})

	df = pd.DataFrame(rows)
	print(f"Loaded {len(df)} catalog systems from Tyler's data.")
	return df


def plot_sed_mosaic(synphot, flux_dict, lambda_eff, ext_vec):
	"""Plot best-fit SEDs in a 3×N mosaic."""
	import sedtool.sed_util as sed_util
	import interpolator

	if not PHOT_FIT.exists():
		print("phot_fit.parquet not found — run full pipeline first.")
		return

	plt, _ = setup_matplotlib()
	phot_fit = pd.read_parquet(PHOT_FIT)

	band_names = sed_util.convert_names(list(flux_dict.keys()))
	interp, _  = sed_util.make_interpolator(band_names, units="fnu")
	logg_func  = sed_util.get_logg_function()

	spec = interpolator.atmos.WarwickSpectrum('1d_da_nlte', units='fnu', wavl_range=(2500, 11000))

	flux_cols   = [v[0] for v in flux_dict.values()]
	e_flux_cols = [v[1] for v in flux_dict.values()]

	def _system(key):
		k = key.lower()
		if 'gaia' in k:      return 'gaia'
		if 'sdss' in k:      return 'sdss'
		if 'ps1' in k or 'panstarrs' in k: return 'ps1'
		if 'skymapper' in k: return 'skymapper'

	SYSTEM_STYLE = {
		'sdss':      ('#586994', 'o', 'SDSS'),
		'ps1':       ('#F05D5E', 's', 'PanSTARRS'),
		'skymapper': ('#BBB09B', '^', 'SkyMapper'),
		'gaia':      ('#BFAB25', 'v', 'Gaia'),
	}
	band_systems = [_system(k) for k in flux_dict.keys()]

	R_SUN, PC_M = 6.957e8, 3.086775e16
	N_DRAW = 200

	sources  = sorted(phot_fit["gaia_dr3_source_id"].values.astype(np.int64))
	n        = len(sources)
	ncols    = 3
	nrows    = (n + ncols - 1) // ncols

	fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5),
	                          constrained_layout=True, sharex=False)
	axes_flat = axes.flatten()
	rng = np.random.default_rng(42)

	for idx, source_id in enumerate(sources):
		ax  = axes_flat[idx]
		sid = int(source_id)

		chain_file = CHAINS_DIR / f"{sid}.npy"
		if not chain_file.exists():
			ax.axis("off")
			continue

		chain = np.load(chain_file)
		flat  = chain.reshape(-1, chain.shape[-1]) if chain.ndim == 3 else chain
		samp  = flat[rng.choice(len(flat), min(N_DRAW, len(flat)), replace=False)]

		syn_row = synphot[synphot["gaia_dr3_source_id"] == sid]
		if syn_row.empty:
			ax.axis("off")
			continue
		syn_row  = syn_row.iloc[0]
		obs_fl   = syn_row[flux_cols].values.astype(float)
		obs_efl  = syn_row[e_flux_cols].values.astype(float)
		obs_mask = np.isfinite(obs_fl) & np.isfinite(obs_efl) & (obs_efl > 0)

		teff_med, radius_med, dist_med, av_med = np.median(samp, axis=0)
		logg_med = float(logg_func(teff_med, radius_med))
		scale    = (radius_med * R_SUN / (dist_med * PC_M)) ** 2
		fl_spec_jy = 4 * np.pi * spec.model_spec((teff_med, logg_med)) * scale * 1e23
		fl_synth   = 4 * np.pi * interp(teff_med, logg_med, av=av_med) * scale * 1e23

		spec_draws = []
		for teff_s, radius_s, dist_s, av_s in samp:
			try:
				logg_s = float(logg_func(teff_s, radius_s))
				sc_s   = (radius_s * R_SUN / (dist_s * PC_M)) ** 2
				spec_draws.append(4 * np.pi * spec.model_spec((teff_s, logg_s)) * sc_s * 1e23)
			except Exception:
				continue

		if not spec_draws:
			ax.axis("off")
			continue

		spec_arr = np.array(spec_draws)
		spec_lo  = np.percentile(spec_arr, 16, axis=0)
		spec_hi  = np.percentile(spec_arr, 84, axis=0)

		ax.fill_between(spec.wavl, spec_lo * 1e3, spec_hi * 1e3, color='k', alpha=0.15, zorder=2)
		ax.plot(spec.wavl, fl_spec_jy * 1e3, color='k', lw=1.5, zorder=3)

		seen = set()
		for sys_key, (color, marker, label) in SYSTEM_STYLE.items():
			sys_mask  = np.array([s == sys_key for s in band_systems])
			plot_mask = sys_mask & obs_mask
			if not plot_mask.any():
				continue
			lam = lambda_eff[plot_mask]
			kw  = dict(label=label) if sys_key not in seen else {}
			seen.add(sys_key)
			ax.errorbar(lam, obs_fl[plot_mask] * 1e3, yerr=obs_efl[plot_mask] * 1e3,
			            fmt=marker, color=color, ecolor='k',
			            capsize=4, ms=6, lw=1.2, zorder=5, **kw)
			ax.plot(lam, fl_synth[plot_mask] * 1e3,
			        marker=marker, color=color, ms=8, mfc='none', mew=1.5,
			        ls='none', zorder=6)

		ax.set_xlim(2800, 10500)
		ax.set_xlabel(r'Wavelength [$\AA$]', fontsize=7)
		ax.set_ylabel(r'Flux [mJy]',         fontsize=7)
		ax.tick_params(labelsize=6)
		ax.text(0.05, 0.95, f'Gaia DR3 {sid}', transform=ax.transAxes,
		        ha='left', va='top', fontsize=5)
		if seen:
			ax.legend(fontsize=5, framealpha=0, loc='upper right')

	for ax in axes_flat[n:]:
		ax.axis("off")

	out = EXO_DIR / "sed_mosaic.pdf"
	fig.savefig(out)
	plt.close(fig)
	print(f"\nSaved SED mosaic → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
	p = argparse.ArgumentParser(
		description="Fit SED + measure ages for WD companions to exoplanet hosts."
	)
	p.add_argument(
		"--new-only", action="store_true",
		help="Only fit systems whose WD source ID is NOT already in Tyler's catalog.",
	)
	return p.parse_args()


def main():
	args = parse_args()

	print("=== Exoplanet WD companion: photometry → MCMC fit → ages ===\n")

	EXO_DIR.mkdir(parents=True, exist_ok=True)

	exo_full    = load_exoplanets()
	exo_full    = attach_nicola(exo_full)
	catalog_ids = get_catalog_ids(exo_full)

	if args.new_only:
		exo = exo_full[~exo_full["wd_source_id"].isin(catalog_ids)].reset_index(drop=True)
		print(f"--new-only: fitting {len(exo)} new systems "
		      f"({len(catalog_ids)} already in catalog will be drawn from Tyler's data).")
	else:
		exo = exo_full

	fit_in = build_input(exo)
	fit_in.to_parquet(PHOT_INPUT, index=False)
	print(f"Saved fit-seds input → {PHOT_INPUT}  ({len(fit_in)} rows)")

	synphot, flux_dict, lambda_eff, ext_vec = get_photometry(fit_in)
	run_mcmc(synphot, flux_dict, lambda_eff, ext_vec)
	summarise_chains(synphot)
	run_measure_ages()

	ages = pd.read_parquet(AGES_OUTPUT)

	if args.new_only and catalog_ids:
		ages = ages[~ages["gaia_dr3_source_id"].isin(catalog_ids)].reset_index(drop=True)
		catalog_ages = load_catalog_ages(exo_full, catalog_ids)
		ages = pd.concat([ages, catalog_ages], ignore_index=True)

	ages["new_fit"] = ~ages["gaia_dr3_source_id"].isin(catalog_ids)

	name_map = exo_full.set_index("wd_source_id")["exoplanet_name"]
	ages["exoplanet_name"] = ages["gaia_dr3_source_id"].map(name_map)
	print(list(ages.keys()))

	n_age = ages["log_age"].notna().sum()
	print(f"\nDone!  {n_age}/{len(ages)} systems have measured total ages.\n")
	cols = [c for c in ["gaia_dr3_source_id", "sep_AU", "new_fit", "teff_best", "mass", "log_age_cool", "log_age"]
	        if c in ages.columns]
	print(ages[cols].to_string(index=False))

	write_latex_table(ages)
	compare_to_catalog(ages)
	plot_sed_mosaic(synphot, flux_dict, lambda_eff, ext_vec)


if __name__ == "__main__":
	main()
