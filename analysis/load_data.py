"""Shared data-loading helpers for analysis scripts."""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
TYLER_CODE_DIR = Path(__file__).parent.parent / "tyler_code"
OUT_DIR = Path(__file__).parent / "figures"


def setup_matplotlib():
	import matplotlib as mpl
	import matplotlib.pyplot as plt

	plt.style.use("stefan.mplstyle")
	mpl.rcParams.update({
		"text.usetex": True,
		"font.family": "serif",
		"text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}",
		"pdf.fonttype": 42,
		"ps.fonttype": 42,
		"svg.fonttype": "none",
	})
	mpl.use("Agg")
	OUT_DIR.mkdir(exist_ok=True)
	return plt, mpl


def load_main_data():
	data = pd.read_csv(DATA_DIR / "tyler/WDMS_total_ages_correct_models_cut_down.csv")
	data["tot_age_error_upper"] = data["tot_age_error_upper"].fillna(999)
	data["total_age_lower_limit"] = data["tot_age"] - data["tot_age_error_lower"]
	data = data.query("Teff > 0")
	data = data[
		data.phot_bp_rp_excess_factor
		> 3 * (0.0059898 + 8.817481e-12 * data.phot_g_mean_mag**7.618399)
	]
	return data


def load_combined():
	"""Merge GALAH/APOGEE/ASTRA/LAMOST survey data into a single standardised DataFrame."""
	tyler_GALAH  = pd.read_csv(DATA_DIR / "tyler/tyler_wdms_GALAH.csv").query("Teff_1 > 3200")
	tyler_APOGEE = pd.read_csv(DATA_DIR / "tyler/tyler_wdms_APOGEE.csv").query("Teff_1 > 3200")
	tyler_ASTRA  = pd.read_csv(DATA_DIR / "tyler/tyler_wdms_ASTRA.csv").query("Teff_1 > 3200")

	for df in (tyler_GALAH, tyler_APOGEE, tyler_ASTRA):
		df["tot_age_error_upper"] = df["tot_age_error_upper"].fillna(999)

	_galah_elems = [
		"Al", "Ba", "C", "Ca", "Ce", "Co", "Cr", "Cr2", "Cu", "Eu",
		"K", "La", "Li", "Mg", "Mn", "Mo", "Na", "Nd", "Ni", "O",
		"Rb", "Ru", "Sc", "Sc2", "Si", "Sm", "Sr", "Ti", "Ti2", "V",
		"Y", "Zn", "Zr",
	]
	galah_rename = {}
	for el in _galah_elems:
		lo = el.lower()
		galah_rename[f"{el}_fe"]	  = f"{lo}_fe"
		galah_rename[f"e_{el}_fe"]	  = f"e_{lo}_fe"
		galah_rename[f"flag_{el}_fe"] = f"flag_{lo}_fe"
		galah_rename[f"nr_{el}_fe"]   = f"nr_{lo}_fe"

	_apogee_fe_elems = [
		"AL", "CA", "CE", "CI", "CO", "CR", "CU", "C",
		"K", "MG", "MN", "NA", "NI", "N", "O", "P",
		"SI", "S", "TI", "V", "YB",
	]
	apogee_rename = {
		"FE_H": "fe_h", "FE_H_ERR": "e_fe_h", "FE_H_FLAG": "flag_fe_h", "FE_H_SPEC": "fe_h_spec",
		"M_H":	"m_h",	"M_H_ERR":	"e_m_h",
		"ALPHA_M": "alpha_m", "ALPHA_M_ERR": "e_alpha_m",
		"TIII_FE": "ti3_fe", "TIII_FE_ERR": "e_ti3_fe",
		"TIII_FE_FLAG": "flag_ti3_fe", "TIII_FE_SPEC": "ti3_fe_spec",
	}
	for el in _apogee_fe_elems:
		lo = el.lower()
		apogee_rename[f"{el}_FE"]	   = f"{lo}_fe"
		apogee_rename[f"{el}_FE_ERR"]  = f"e_{lo}_fe"
		apogee_rename[f"{el}_FE_FLAG"] = f"flag_{lo}_fe"
		apogee_rename[f"{el}_FE_SPEC"] = f"{lo}_fe_spec"

	_astra_h_elems = [
		"al", "c", "ca", "ce", "co", "cr", "cu",
		"k", "mg", "mn", "n", "na", "nd", "ni", "o",
		"p", "s", "si", "ti", "v",
	]
	astra_rename = {"fe_h_flags": "flag_fe_h", "c_12_13_flags": "flag_c_12_13"}
	for el in _astra_h_elems:
		astra_rename[f"{el}_h_flags"] = f"flag_{el}_fe"

	g = tyler_GALAH.rename(columns=galah_rename).assign(survey="GALAH")
	a = tyler_APOGEE.rename(columns=apogee_rename).assign(survey="APOGEE")
	s = tyler_ASTRA.rename(columns=astra_rename).assign(survey="ASTRA")
	l = load_lamost().assign(survey="LAMOST")
	l["tot_age_error_upper"] = l["tot_age_error_upper"].fillna(999)

	def _apply_wd_cuts(df, extra=""):
		q = "ruwe < 1.2 & R_chance_align < 0.1" + (f" & {extra}" if extra else "")
		df = df.query(q)
		return df[df.phot_bp_rp_excess_factor > 3 * (0.0059898 + 8.817481e-12 * df.phot_g_mean_mag**7.618399)]

	g = _apply_wd_cuts(g)
	a = _apply_wd_cuts(a)
	s = _apply_wd_cuts(s, extra="flag_bad == False")
	l = _apply_wd_cuts(l)

	for el in _astra_h_elems:
		s[f"{el}_fe"]	= s[f"{el}_h"] - s["fe_h"]
		s[f"e_{el}_fe"] = np.sqrt(s[f"e_{el}_h"]**2 + s["e_fe_h"]**2)
		s = s.drop(columns=[f"{el}_h", f"e_{el}_h"], errors="ignore")

	metal_cols = (
		set(galah_rename.values())
		| set(apogee_rename.values())
		| {f"{el}_fe"	   for el in _astra_h_elems}
		| {f"e_{el}_fe"    for el in _astra_h_elems}
		| {f"flag_{el}_fe" for el in _astra_h_elems}
		| {
			"fe_h", "e_fe_h", "flag_fe_h",
			"alpha_fe", "e_alpha_fe", "flag_alpha_fe",
			"alpha_m", "e_alpha_m",
			"alpha_m_atm", "e_alpha_m_atm",
			"m_h", "e_m_h", "m_h_atm", "e_m_h_atm",
			"c_m_atm", "e_c_m_atm", "n_m_atm", "e_n_m_atm",
			"c_12_13", "e_c_12_13", "flag_c_12_13",
			"fe_h_atmo",
		}
	)

	all_non_metal = [
		c for df in (g, a, s, l)
		for c in df.columns if c not in metal_cols and c != "survey"
	]
	base_cols = [c for c, n in Counter(all_non_metal).items() if n >= 2]

	combined = pd.concat([g, a, s, l], ignore_index=True)
	keep = ["survey"] + base_cols + [c for c in metal_cols if c in combined.columns]
	combined = combined[[c for c in keep if c in combined.columns]]

	flag_threshold = {"GALAH": 1, "APOGEE": 0, "ASTRA": 0, "LAMOST": 0}
	for flag_col in [c for c in combined.columns if c.startswith("flag_") and c in metal_cols]:
		base = flag_col[len("flag_"):]
		cols_to_null = [c for c in [base, f"e_{base}"] if c in combined.columns]
		if not cols_to_null:
			continue
		for survey, threshold in flag_threshold.items():
			bad = (combined["survey"] == survey) & (combined[flag_col].fillna(0) > threshold)
			combined.loc[bad, cols_to_null] = np.nan

	combined = combined.query("fe_h > -900").copy()
	combined["total_age_lower_limit"] = combined["tot_age"] - combined["tot_age_error_lower"]

	totalage   = combined.Mass > 0.63
	coolingage = ~totalage
	return combined, totalage, coolingage


def age_masks(combined, totalage, coolingage):
	"""Return (normal, white_ll, blue_ll) age-axis plotting masks.

	normal	 — M > 0.63, age upper error ≤ 13 Gyr: plot tot_age ± errors
	white_ll — M > 0.63 with err_upper > 13, or 0.6 < M ≤ 0.63: total_age_lower_limit, white arrow
	blue_ll  — M ≤ 0.6: total_age_lower_limit, blue arrow
	"""
	age_lowlim = combined["tot_age_error_upper"] > 13
	normal	 = totalage & ~age_lowlim
	white_ll = (totalage & age_lowlim) | ((combined["Mass"] > 0.6) & coolingage)
	blue_ll  = combined["Mass"] <= 0.6
	return normal, white_ll, blue_ll


def avr(age):
	"""Cheng (2019) age-velocity relation. Returns (sigma_U, sigma_V, sigma_W) in km/s."""
	age = np.atleast_1d(np.asarray(age, dtype=float))
	c_u, d_u, b_u = 5.78, 26.09, 0.35
	c_v, d_v, b_v = 0.68 * c_u, 0.68 * d_u, 0.68 * b_u
	c_w, d_w, b_w = 3.81, 11.97, 0.60
	su, sv, sw = np.zeros_like(age), np.zeros_like(age), np.zeros_like(age)
	m1 = age <= 7
	su[m1] = c_u + d_u * (age[m1] / 4) ** b_u
	sv[m1] = c_v + d_v * (age[m1] / 4) ** b_v
	sw[m1] = c_w + d_w * (age[m1] / 4) ** b_w
	m2 = (age > 7) & (age <= 10)
	su[m2] = c_u + d_u*(7/4)**b_u + (66.07		- c_u - d_u*(7/4)**b_u) * (age[m2]-7)/3
	sv[m2] = c_v + d_v*(7/4)**b_v + (0.64*66.07 - c_v - d_v*(7/4)**b_v) * (age[m2]-7)/3
	sw[m2] = c_w + d_w*(7/4)**b_w + (34.07		- c_w - d_w*(7/4)**b_w) * (age[m2]-7)/3
	su[age > 10] = 66.07
	sv[age > 10] = 0.64 * 66.07
	sw[age > 10] = 34.07
	return su, sv, sw


def bensby_classify(uvw):
	"""Classify stars into thin disk / thick disk / halo using Bensby+2014 Table 1.

	Parameters
	----------
	uvw : (N, 3) array of (U, V, W) velocities in km/s, LSR frame.

	Returns
	-------
	labels : (N,) array of str — 'thin', 'thick', or 'halo'
	responsibilities : (N, 3) array of posterior probabilities P(component | UVW)
	"""
	from scipy.stats import multivariate_normal

	# Bensby+2014 kinematic selection
	keys	   = ["thin", "thick", "halo", "hercules"]
	amplitudes = np.array([0.85, 0.09, 0.0015, 0.06])
	amplitudes /= amplitudes.sum()
	means  = np.array([[0, -15, 0], [0, -46, 0], [0, -220, 0], [-40, -50, 0]], dtype=float)
	sigmas = np.array([[35, 20, 16], [67, 38, 35], [160, 90, 90], [26, 9, 17]], dtype=float)
	covs   = np.array([np.diag(s**2) for s in sigmas])

	likelihoods = np.column_stack([
		multivariate_normal(mean=means[k], cov=covs[k]).pdf(uvw)
		for k in range(amplitudes.shape[0])
	])
	weighted = amplitudes[None, :] * likelihoods
	responsibilities = weighted / weighted.sum(axis=1, keepdims=True)
	labels = np.array([keys[i] for i in np.argmax(responsibilities, axis=1)])
	return labels, responsibilities


def load_gaia_ms_rv(ms_source_ids):
	"""Fetch Gaia DR3 radial velocities for MS companions; caches to data/gaia_ms_rv.csv."""
	cache_path = DATA_DIR / "gaia_ms_rv.csv"
	if cache_path.exists():
		return pd.read_csv(cache_path, dtype={"ms_source_id": np.int64})

	from astroquery.gaia import Gaia
	ids = np.array(ms_source_ids, dtype=np.int64)
	batch_size = 2000
	rows = []
	for i in range(0, len(ids), batch_size):
		batch = ids[i : i + batch_size]
		id_str = ",".join(str(x) for x in batch)
		query = (
			"SELECT source_id, radial_velocity, radial_velocity_error "
			"FROM gaiadr3.gaia_source "
			f"WHERE source_id IN ({id_str})"
		)
		job = Gaia.launch_job(query, verbose=False)
		rows.append(job.get_results().to_pandas())
		print(f"  Gaia RV query: batch {i // batch_size + 1}/{(len(ids) + batch_size - 1) // batch_size}")

	result = pd.concat(rows, ignore_index=True)
	result = result.rename(columns={"source_id": "ms_source_id"})
	result.to_csv(cache_path, index=False)
	print(f"Cached {len(result)} Gaia MS RV rows → {cache_path}")
	return result


def load_lamost():
	"""Load LAMOST WD+MS data; fetches [Fe/H] errors from VizieR and caches locally."""
	cache_path = DATA_DIR / "merge/wdms_lamost_efeh.csv"

	lamost = pd.read_csv(DATA_DIR / "merge/wdms_lamost.csv")
	lamost = lamost.rename(columns={"[Fe/H]": "fe_h"})
	lamost = lamost[lamost.Teff > 0]
	lamost = lamost.sort_values("snrg", ascending=False).drop_duplicates(subset=["source_id"])
	lamost = lamost.query("fe_h > -900")

	if cache_path.exists():
		efeh = pd.read_csv(cache_path)
		lamost = lamost.merge(efeh, on="ObsID", how="left")
	else:
		try:
			import pyvo
			tap = pyvo.dal.TAPService("http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")
			id_list = ", ".join(str(int(i)) for i in lamost["ObsID"].dropna())
			result = (
				tap.search(
					f'SELECT "ObsID", "e_[Fe/H]" as e_FeH FROM "V/156/dr7slrs"'
					f' WHERE "ObsID" IN ({id_list})'
				)
				.to_table()
				.to_pandas()
				.rename(columns={"e_FeH": "e_fe_h"})
			)
			result[["ObsID", "e_fe_h"]].to_csv(cache_path, index=False)
			lamost = lamost.merge(result, on="ObsID", how="left")
		except Exception as e:
			print(f"Warning: LAMOST e_fe_h query failed ({e}); using NaN")
			lamost["e_fe_h"] = np.nan

	return lamost
