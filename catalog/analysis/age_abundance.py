"""Age vs abundance figures: Li, alpha, C, Ba.

Three groups on the age axis:
  normal   (age_class=0): tot_age ± errors, white
  white_ll (age_class=1): total_age_lower_limit →, white
  blue_ll  (age_class=2): total_age_lower_limit →, blue
"""

import numpy as np

from catalog.config import FIGURES_DIR
from catalog.analysis.load_data import load_catalog, load_metallicity, setup_matplotlib

LOWLIM_COLOR = "#426dfc"
LOWLIM_KW = dict(
	xlolims=True, alpha=0.5,
	fmt="o", color=LOWLIM_COLOR, markeredgecolor="k", ecolor="k",
	lw=2, capsize=4, markersize=4, zorder=0,
)
TOTAL_KW = dict(
	fmt="o", color="#eeeeee", markeredgecolor="k", ecolor="k",
	lw=2, capsize=4, markersize=7,
)


def main():
	plt, _ = setup_matplotlib()
	combined, normal, white_ll, blue_ll = load_catalog()

	combined["a_li"] = combined["li_fe"] + combined["fe_h"] + 0.96
	combined["e_a_li"] = np.sqrt(combined["e_li_fe"]**2 + combined["e_fe_h"]**2)

	# ── Li Abundance ──────────────────────────────────────────────────────────
	# flag_li_fe == 1 in GALAH is a non-detection: the abundance is an upper limit
	li_lowlim = combined["flag_li_fe"].fillna(0) == 1

	# Solar-analog box (cf. Carlos et al. 2016 solar-twin selection, loosened):
	# Teff = 5777 +/- 500 K, logg = 4.44 +/- 0.3 dex, [Fe/H] = 0.0 +/- 0.3 dex
	solar_analog = (
		combined["ms_teff"].between(5777 - 750, 5777 + 750)
		& combined["ms_logg"].between(4.44 - 0.5, 4.44 + 0.5)
		& combined["a_li"].notna()
	)
	print(f"{sum(solar_analog)} solar analog stars")

	fig, axes = plt.subplots(ncols=2, figsize=(12, 5), sharey=True)

	for panel_ax, panel_sel in zip(axes, [combined.index.notna(), solar_analog]):
		for ul in [False, True]:
			extra = {"uplims": True} if ul else {}
			sel = normal & panel_sel & (li_lowlim == ul)
			panel_ax.errorbar(
				combined.loc[sel, "tot_age"],
				combined.loc[sel, "a_li"],
				xerr=combined.loc[sel, ["tot_age_error_lower", "tot_age_error_upper"]].values.T,
				yerr=combined.loc[sel, "e_a_li"],
				**extra, **TOTAL_KW,
			)
			sel = white_ll & panel_sel & (li_lowlim == ul)
			panel_ax.errorbar(
				combined.loc[sel, "total_age_lower_limit"],
				combined.loc[sel, "a_li"],
				xerr=0.5,
				yerr=combined.loc[sel, "e_a_li"],
				xlolims=True, **extra, **TOTAL_KW,
			)
			sel = blue_ll & panel_sel & (li_lowlim == ul)
			panel_ax.errorbar(
				combined.loc[sel, "total_age_lower_limit"],
				combined.loc[sel, "a_li"],
				xerr=0.5,
				yerr=combined.loc[sel, "e_a_li"],
				**extra, **LOWLIM_KW,
			)

		panel_ax.plot(
			np.array([0, 13]),
			2.437 - 0.224 * np.array([0, 13]),
			c="k", label="Carlos et al. (2016)"
		)
		panel_ax.set_xlim(0, 7)
		panel_ax.set_xlabel("WD Age [Gyr]")

	axes[0].set_title("All Stars", fontsize=18)
	axes[1].set_title("Solar Analogs", fontsize=18)
	axes[0].set_ylim(0, 3)
	axes[0].set_ylabel("A(Li)")
	axes[0].legend(framealpha=0)

	fig.tight_layout()
	out = FIGURES_DIR / "age_li.pdf"
	fig.savefig(out)
	plt.close()
	print(f"Saved {out}")

	# ── One-sided residual test for lower-limit ages ──────────────────────────
	# A(Li) declines monotonically with age, so a lower limit on age gives an
	# upper limit on the relation's prediction (evaluated at the limit itself,
	# the youngest age still allowed). Only stars whose observed A(Li) sits
	# above that best-case prediction are in genuine tension with the relation
	# -- the true (larger) age would only push the prediction lower.
	lowlim = (white_ll | blue_ll) & combined["a_li"].notna() & combined["total_age_lower_limit"].notna()
	pred_ceiling = 2.437 - 0.224 * combined["total_age_lower_limit"]
	tension_sigma = (combined["a_li"] - pred_ceiling) / combined["e_a_li"]
	combined["li_age_tension_sigma"] = tension_sigma.where(lowlim)

	in_tension = lowlim & (tension_sigma > 3)
	print(f"{lowlim.sum()} lower-limit systems with Li/Fe")
	print(f"{in_tension.sum()} in >3sigma tension with Carlos et al. (2016), "
		  f"even under their most permissive (limit) age")
	print(f"  of which {(in_tension & solar_analog).sum()} are solar analogs "
		  f"(out of {(lowlim & solar_analog).sum()} solar-analog lower limits)")

	# Two-sided residual test for point-estimate (normal) ages -- no censoring,
	# so the ordinary signed residual against the relation applies directly.
	has_age = normal & combined["a_li"].notna() & combined["tot_age"].notna()
	pred_normal = 2.437 - 0.224 * combined["tot_age"]
	sigma_normal = (combined["a_li"] - pred_normal) / combined["e_a_li"]
	combined.loc[has_age, "li_age_tension_sigma"] = sigma_normal[has_age]

	in_tension_normal = has_age & (sigma_normal.abs() > 3)
	print(f"{has_age.sum()} normal (point-estimate age) systems with Li/Fe")
	print(f"{in_tension_normal.sum()} more than 3sigma from Carlos et al. (2016)")
	print(f"  of which {(in_tension_normal & solar_analog).sum()} are solar analogs "
		  f"(out of {(has_age & solar_analog).sum()} solar-analog point-estimate ages)")

	total_tension = in_tension | in_tension_normal
	print(f"{total_tension.sum()} systems total more than 3sigma from the relation "
		  f"({(total_tension & solar_analog).sum()} solar analogs)")

	mask = (combined["total_age_lower_limit"] > 7.5) & (combined["a_li"] > 2)
	print(combined.loc[mask, ["source_id", "Mass", "e_Mass_lower", "e_Mass_upper",
							   "Teff", "e_Teff_lower", "e_Teff_upper",
							   "total_age_lower_limit", "a_li", "e_a_li"]])
	cluster  = combined.query("ms_source_id == 4083689509602720000").iloc[0]
	pmra	 = 4.74047 * cluster.pmra / cluster.wtd_par
	e_pmra	 = np.abs(cluster.pmra / cluster.wtd_par) * np.sqrt(
		(cluster.pmra_error / cluster.pmra)**2 + (cluster.e_wtd_par / cluster.wtd_par)**2
	)
	pmdec	 = 4.74047 * cluster.pmdec / cluster.wtd_par
	e_pmdec  = np.abs(cluster.pmdec / cluster.wtd_par) * np.sqrt(
		(cluster.pmdec_error / cluster.pmdec)**2 + (cluster.e_wtd_par / cluster.wtd_par)**2
	)
	rv, e_rv = 11.41, 0.11

	pmra_cluster	= 4.74047 * -0.9733 / 3.2516
	e_pmra_cluster	= np.abs(0.0367 / 0.0038) * np.sqrt((0.0367 / -0.9733)**2 + (0.0038 / 3.2516)**2)
	pmdec_cluster	= 4.74047 * -26.6464 / 3.2516
	e_pmdec_cluster = np.abs(0.0383 / 0.0038) * np.sqrt((0.0383 / -26.6464)**2 + (0.0038 / 3.2516)**2)
	rv_cluster, e_rv_cluster = 42.18, 0.38

	relative_motion = np.sqrt(
		(pmra - pmra_cluster)**2 + (pmdec - pmdec_cluster)**2 + (rv - rv_cluster)**2
	)
	e_relative_motion = np.sqrt(
		e_pmra**2 + e_pmra_cluster**2 + e_pmdec**2 + e_pmdec_cluster**2 + e_rv**2 + e_rv_cluster**2
	)
	print(f"Relative motion of HD 179856 to NGC 6774: {relative_motion:2.2f}+/-{e_relative_motion:2.2f} km/s")

	# ── Alpha-enrichment ──────────────────────────────────────────────────────
	A_ENRICHED_CUTOFF = 0.1
	combined_reliable_alpha = combined.query("e_alpha_fe < 0.1").copy()

	fig, ax = plt.subplots(ncols=3, figsize=(18, 6), sharey=False)

	ax[0].errorbar(combined_reliable_alpha.loc[normal | white_ll, "fe_h"],
				   combined_reliable_alpha.loc[normal | white_ll, "alpha_fe"],
				   xerr=combined_reliable_alpha.loc[normal | white_ll, "e_fe_h"],
				   yerr=combined_reliable_alpha.loc[normal | white_ll, "e_alpha_fe"], **TOTAL_KW)
	ax[0].errorbar(combined_reliable_alpha.loc[blue_ll, "fe_h"],
				   combined_reliable_alpha.loc[blue_ll, "alpha_fe"],
				   xerr=combined_reliable_alpha.loc[blue_ll, "e_fe_h"],
				   yerr=combined_reliable_alpha.loc[blue_ll, "e_alpha_fe"], alpha=0.5,
				   fmt="o", color=LOWLIM_COLOR, markeredgecolor="k", ecolor="k",
				   lw=2, capsize=4, markersize=7, zorder=0)
	ax[0].axhline(y=A_ENRICHED_CUTOFF, c="k", ls="--")
	ax[0].set_xlabel("[Fe/H]")
	ax[0].set_ylabel(r"[$\alpha$/Fe]")

	ax[1].errorbar(combined_reliable_alpha.loc[normal, "tot_age"],
				   combined_reliable_alpha.loc[normal, "alpha_fe"],
				   xerr=combined_reliable_alpha.loc[normal, ["tot_age_error_lower", "tot_age_error_upper"]].values.T,
				   yerr=combined_reliable_alpha.loc[normal, "e_alpha_fe"], **TOTAL_KW)
	ax[1].errorbar(combined_reliable_alpha.loc[white_ll, "total_age_lower_limit"],
				   combined_reliable_alpha.loc[white_ll, "alpha_fe"],
				   xerr=0.5, yerr=combined_reliable_alpha.loc[white_ll, "e_alpha_fe"], xlolims=True, **TOTAL_KW)
	ax[1].errorbar(combined_reliable_alpha.loc[blue_ll, "total_age_lower_limit"],
				   combined_reliable_alpha.loc[blue_ll, "alpha_fe"],
				   xerr=0.5, yerr=combined_reliable_alpha.loc[blue_ll, "e_alpha_fe"], **LOWLIM_KW)
	ax[1].axhline(y=A_ENRICHED_CUTOFF, c="k", ls="--")
	ax[1].set_xlabel("Age [Gyr]")
	ax[1].set_xlim(0, 13)
	ax[1].sharey(ax[0])

	a_enriched	   = combined_reliable_alpha["alpha_fe"] > A_ENRICHED_CUTOFF
	not_a_enriched = combined_reliable_alpha["alpha_fe"] <= A_ENRICHED_CUTOFF
	sep_bins = np.logspace(
		min(np.log10(combined_reliable_alpha.loc[a_enriched, "sep_AU"].min()),
			np.log10(combined_reliable_alpha.loc[not_a_enriched, "sep_AU"].min())),
		max(np.log10(combined_reliable_alpha.loc[a_enriched, "sep_AU"].max()),
			np.log10(combined_reliable_alpha.loc[not_a_enriched, "sep_AU"].max())),
		5,
	)
	ax[2].hist(combined_reliable_alpha.loc[a_enriched, "sep_AU"], bins=sep_bins, histtype="step",
			   linewidth=4, color="red", label=r"$\alpha$-Enriched", density=True)
	ax[2].hist(combined_reliable_alpha.loc[not_a_enriched, "sep_AU"], bins=sep_bins, histtype="step",
			   linewidth=4, color="k", label=r"Not $\alpha$-Enriched", density=True, zorder=0)
	ax[2].set_xlabel("Separation [au]")
	ax[2].set_ylabel("Density")
	ax[2].legend(framealpha=0, loc="upper left", fontsize=14)
	ax[2].set_xscale("log")

	print(f"# alpha enriched: {len(combined_reliable_alpha.loc[a_enriched, 'sep_AU'])}")
	print(f"# not alpha enriched: {len(combined_reliable_alpha.loc[not_a_enriched, 'sep_AU'])}")
	print(combined_reliable_alpha.loc[normal | white_ll].query("alpha_fe > 0.40")[
		["source_id", "Mass", "Teff", "R_chance_align", "sep_AU", "spectype",'spec_source', 'model_used']# "ms_source_id", "alpha_fe", "e_alpha_fe", "mg_fe", "ca_fe", "ti_fe", "tot_age_error_upper"]
	])
	fig.tight_layout()
	out = FIGURES_DIR / "age_alpha.pdf"
	fig.savefig(out)
	plt.close()
	print(f"Saved {out}")

	# ── Carbon enrichment ─────────────────────────────────────────────────────
	C_ENRICHED_CUTOFF = 0.25

	fig, ax = plt.subplots(ncols=3, figsize=(18, 6), sharey=False)

	ax[0].errorbar(combined.loc[normal | white_ll, "fe_h"], combined.loc[normal | white_ll, "c_fe"],
				   xerr=combined.loc[normal | white_ll, "e_fe_h"],
				   yerr=combined.loc[normal | white_ll, "e_c_fe"], **TOTAL_KW)
	ax[0].errorbar(combined.loc[blue_ll, "fe_h"], combined.loc[blue_ll, "c_fe"],
				   xerr=combined.loc[blue_ll, "e_fe_h"],
				   yerr=combined.loc[blue_ll, "e_c_fe"], alpha=0.5,
				   fmt="o", color=LOWLIM_COLOR, markeredgecolor="k", ecolor="k",
				   lw=2, capsize=4, markersize=7, zorder=0)
	ax[0].axhline(y=C_ENRICHED_CUTOFF, c="k", ls="--")
	ax[0].set_xlabel("[Fe/H]")
	ax[0].set_ylabel("[C/Fe]")

	ax[1].errorbar(combined.loc[normal, "tot_age"], combined.loc[normal, "c_fe"],
				   xerr=combined.loc[normal, ["tot_age_error_lower", "tot_age_error_upper"]].values.T,
				   yerr=combined.loc[normal, "e_c_fe"], **TOTAL_KW)
	ax[1].errorbar(combined.loc[white_ll, "total_age_lower_limit"], combined.loc[white_ll, "c_fe"],
				   xerr=0.5, yerr=combined.loc[white_ll, "e_c_fe"], xlolims=True, **TOTAL_KW)
	ax[1].errorbar(combined.loc[blue_ll, "total_age_lower_limit"], combined.loc[blue_ll, "c_fe"],
				   xerr=0.5, yerr=combined.loc[blue_ll, "e_c_fe"], **LOWLIM_KW)
	ax[1].axhline(y=C_ENRICHED_CUTOFF, c="k", ls="--")
	ax[1].set_xlabel("Age [Gyr]")
	ax[1].set_xlim(0, 13)
	ax[1].sharey(ax[0])

	c_enriched	   = combined["c_fe"] > C_ENRICHED_CUTOFF
	not_c_enriched = combined["c_fe"] <= C_ENRICHED_CUTOFF
	sep_bins = np.logspace(
		min(np.log10(combined.loc[c_enriched, "sep_AU"].min()),
			np.log10(combined.loc[not_c_enriched, "sep_AU"].min())),
		max(np.log10(combined.loc[c_enriched, "sep_AU"].max()),
			np.log10(combined.loc[not_c_enriched, "sep_AU"].max())),
		8,
	)
	ax[2].hist(combined.loc[c_enriched, "sep_AU"], bins=sep_bins, histtype="step",
			   density=True, linewidth=4, color="red", label="C-Enriched")
	ax[2].hist(combined.loc[not_c_enriched, "sep_AU"], bins=sep_bins, histtype="step",
			   density=True, linewidth=4, color="k", label="Not C-Enriched", zorder=0)
	ax[2].set_xlabel("Separation [au]")
	ax[2].set_ylabel("Density")
	ax[2].legend(framealpha=0, fontsize=14)
	ax[2].set_xscale("log")

	fig.tight_layout()
	out = FIGURES_DIR / "age_carbon.pdf"
	fig.savefig(out)
	plt.close()
	print(f"Saved {out}")

	# ── Barium enrichment ─────────────────────────────────────────────────────
	BA_ENRICHED_CUTOFF = 1.0

	fig, ax = plt.subplots(ncols=3, figsize=(18, 6), sharey=False)

	ax[0].errorbar(combined.loc[normal | white_ll, "fe_h"], combined.loc[normal | white_ll, "ba_fe"],
				   xerr=combined.loc[normal | white_ll, "e_fe_h"],
				   yerr=combined.loc[normal | white_ll, "e_ba_fe"], **TOTAL_KW)
	ax[0].errorbar(combined.loc[blue_ll, "fe_h"], combined.loc[blue_ll, "ba_fe"],
				   xerr=combined.loc[blue_ll, "e_fe_h"],
				   yerr=combined.loc[blue_ll, "e_ba_fe"], alpha=0.5,
				   fmt="o", color=LOWLIM_COLOR, markeredgecolor="k", ecolor="k",
				   lw=2, capsize=4, markersize=7, zorder=0)
	ax[0].axhline(y=BA_ENRICHED_CUTOFF, c="k", ls="--")
	ax[0].set_xlabel("[Fe/H]")
	ax[0].set_ylabel("[Ba/Fe]")

	ax[1].errorbar(combined.loc[normal, "tot_age"], combined.loc[normal, "ba_fe"],
				   xerr=combined.loc[normal, ["tot_age_error_lower", "tot_age_error_upper"]].values.T,
				   yerr=combined.loc[normal, "e_ba_fe"], **TOTAL_KW)
	ax[1].errorbar(combined.loc[white_ll, "total_age_lower_limit"], combined.loc[white_ll, "ba_fe"],
				   xerr=0.5, yerr=combined.loc[white_ll, "e_ba_fe"], xlolims=True, **TOTAL_KW)
	ax[1].errorbar(combined.loc[blue_ll, "total_age_lower_limit"], combined.loc[blue_ll, "ba_fe"],
				   xerr=0.5, yerr=combined.loc[blue_ll, "e_ba_fe"], **LOWLIM_KW)
	ax[1].axhline(y=BA_ENRICHED_CUTOFF, c="k", ls="--")
	ax[1].set_xlabel("Age [Gyr]")
	ax[1].set_xlim(0, 13)
	ax[1].sharey(ax[0])

	ba_enriched		= combined["ba_fe"] > BA_ENRICHED_CUTOFF
	not_ba_enriched = combined["ba_fe"] <= BA_ENRICHED_CUTOFF
	sep_bins = np.logspace(
		min(np.log10(combined.loc[ba_enriched, "sep_AU"].min()),
			np.log10(combined.loc[not_ba_enriched, "sep_AU"].min())),
		max(np.log10(combined.loc[ba_enriched, "sep_AU"].max()),
			np.log10(combined.loc[not_ba_enriched, "sep_AU"].max())),
		8,
	)
	ax[2].hist(combined.loc[ba_enriched, "sep_AU"], bins=sep_bins, histtype="step",
			   density=True, linewidth=4, color="red", label="Ba-Enriched")
	ax[2].hist(combined.loc[not_ba_enriched, "sep_AU"], bins=sep_bins, histtype="step",
			   density=True, linewidth=4, color="k", label="Not Ba-Enriched", zorder=0)
	ax[2].set_xlabel("Separation [au]")
	ax[2].set_ylabel("Density")
	ax[2].legend(framealpha=0, fontsize=14)
	ax[2].set_xscale("log")

	fig.tight_layout()
	out = FIGURES_DIR / "age_barium.pdf"
	fig.savefig(out)
	plt.close()
	print(f"Saved {out}")

	# ── Survey abundance count table (AASTeX) ─────────────────────────────────
	_raw = load_metallicity()
	_abund_cols = [
		(r"$[\mathrm{Fe/H}]$",		"fe_h"),
		(r"$[\mathrm{Li/Fe}]$",		"li_fe"),
		(r"$[\alpha/\mathrm{Fe}]$", "alpha_fe"),
		(r"$[\mathrm{C/Fe}]$",		"c_fe"),
	]
	_surveys = ["APOGEE", "ASTRA", "GALAH", "LAMOST"]

	print(r"\begin{deluxetable}{lrrrrr}")
	print(r"\tablecaption{Number of sources with each abundance measurement by survey.}")
	print(r"\tablehead{")
	print(r"  \colhead{Abundance} &")
	print(r"  \colhead{APOGEE} & \colhead{ASTRA} & \colhead{GALAH} & \colhead{LAMOST} &")
	print(r"  \colhead{Total}")
	print(r"}")
	print(r"\startdata")
	for label, col in _abund_cols:
		counts = [_raw.loc[_raw.survey == s, col].notna().sum() for s in _surveys]
		print(f"  {label:<25} & " + " & ".join(f"{c:>3}" for c in counts) + f" & {sum(counts):>4} \\\\")
	print(r"\hline")
	totals = [(_raw.survey == s).sum() for s in _surveys]
	print(f"  {'Total':<25} & " + " & ".join(f"{t:>3}" for t in totals) + f" & {len(_raw):>4} \\\\")
	print(r"\enddata")
	print(r"\end{deluxetable}")

	print(f"{combined['fe_h'].notna().sum()} systems with Fe/H")
	print(f"{combined['alpha_fe'].notna().sum()} systems with alpha/Fe")
	print(f"{combined['li_fe'].notna().sum()} systems with Li/Fe ({li_lowlim.sum()} low lims)")
	print(f"{combined['c_fe'].notna().sum()} systems with C/Fe ({(combined['flag_c_fe'] == 1).sum()} low lims)")


if __name__ == "__main__":
	main()
