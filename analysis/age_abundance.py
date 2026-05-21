"""Age vs abundance figures: Li, alpha, C, Ba.

Three groups on the age axis:
  normal   (M > 0.63, err_upper ≤ 13 Gyr): tot_age ± errors, white
  white_ll (M > 0.63 with err_upper > 13, or 0.6 < M ≤ 0.63): total_age_lower_limit →, white
  blue_ll  (M ≤ 0.6): total_age_lower_limit →, blue
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from load_data import OUT_DIR, age_masks, load_combined, setup_matplotlib

plt, _ = setup_matplotlib()
combined, totalage, coolingage = load_combined()

combined["a_li"] = combined["li_fe"] + combined["fe_h"] + 0.96
combined["e_a_li"] = np.sqrt(combined["e_li_fe"]**2 + combined["e_fe_h"]**2)

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

normal, white_ll, blue_ll = age_masks(combined, totalage, coolingage)


# ── Li Abundance ──────────────────────────────────────────────────────────
# flag_li_fe == 1 in GALAH is a non-detection: the abundance is an upper limit
li_lowlim  = (combined["survey"] == "GALAH") & (combined["flag_li_fe"].fillna(0) == 1)

fig, ax = plt.subplots()

for ul in [False, True]:
    extra = {"uplims": True} if ul else {}
    sel = normal & (li_lowlim == ul)
    ax.errorbar(
        combined.loc[sel, "tot_age"],
        combined.loc[sel, "a_li"],
        xerr=combined.loc[sel, ["tot_age_error_lower", "tot_age_error_upper"]].values.T,
        yerr=combined.loc[sel, "e_a_li"],
        **extra, **TOTAL_KW,
    )
    sel = white_ll & (li_lowlim == ul)
    ax.errorbar(
        combined.loc[sel, "total_age_lower_limit"],
        combined.loc[sel, "a_li"],
        xerr=0.5,
        yerr=combined.loc[sel, "e_a_li"],
        xlolims=True, **extra, **TOTAL_KW,
    )
    sel = blue_ll & (li_lowlim == ul)
    ax.errorbar(
        combined.loc[sel, "total_age_lower_limit"],
        combined.loc[sel, "a_li"],
        xerr=0.5,
        yerr=combined.loc[sel, "e_a_li"],
        **extra, **LOWLIM_KW,
    )

ax.plot(
	np.array([0, 13]),
	2.437 - 0.224 * np.array([0, 13]),
	c = "k", label="Marília et al. (2016)"
)

ax.set_xlim(0, 13)
ax.set_ylim(0, 3)
ax.set_xlabel("Age [Gyr]")
ax.set_ylabel("A(Li)")
ax.legend(framealpha=0)

fig.savefig(OUT_DIR / "age_li.pdf")
plt.close()
print(f"Saved {OUT_DIR / 'age_li.pdf'}")


# ── Alpha-enrichment ──────────────────────────────────────────────────────
A_ENRICHED_CUTOFF = 0.1

fig, ax = plt.subplots(ncols=3, figsize=(18, 6), sharey=False)

ax[0].errorbar(combined.loc[normal | white_ll, "fe_h"],   combined.loc[normal | white_ll, "alpha_fe"],
               xerr=combined.loc[normal | white_ll, "e_fe_h"],
               yerr=combined.loc[normal | white_ll, "e_alpha_fe"], **TOTAL_KW)
ax[0].errorbar(combined.loc[blue_ll, "fe_h"], combined.loc[blue_ll, "alpha_fe"],
               xerr=combined.loc[blue_ll, "e_fe_h"],
               yerr=combined.loc[blue_ll, "e_alpha_fe"], alpha=0.5,
               fmt="o", color=LOWLIM_COLOR, markeredgecolor="k", ecolor="k",
               lw=2, capsize=4, markersize=7, zorder=0)
ax[0].axhline(y=A_ENRICHED_CUTOFF, c="k", ls="--")
ax[0].set_xlabel("[Fe/H]")
ax[0].set_ylabel(r"[$\alpha$/Fe]")

ax[1].errorbar(combined.loc[normal, "tot_age"], combined.loc[normal, "alpha_fe"],
               xerr=combined.loc[normal, ["tot_age_error_lower", "tot_age_error_upper"]].values.T,
               yerr=combined.loc[normal, "e_alpha_fe"], **TOTAL_KW)
ax[1].errorbar(combined.loc[white_ll, "total_age_lower_limit"],
               combined.loc[white_ll, "alpha_fe"],
               xerr=0.5, yerr=combined.loc[white_ll, "e_alpha_fe"], xlolims=True, **TOTAL_KW)
ax[1].errorbar(combined.loc[blue_ll, "total_age_lower_limit"],
               combined.loc[blue_ll, "alpha_fe"],
               xerr=0.5, yerr=combined.loc[blue_ll, "e_alpha_fe"], **LOWLIM_KW)
ax[1].axhline(y=A_ENRICHED_CUTOFF, c="k", ls="--")
ax[1].set_xlabel("Age [Gyr]")
ax[1].set_xlim(0, 12)
ax[1].sharey(ax[0])

a_enriched     = combined["alpha_fe"] > A_ENRICHED_CUTOFF
not_a_enriched = combined["alpha_fe"] <= A_ENRICHED_CUTOFF
sep_bins = np.logspace(
    min(np.log10(combined.loc[a_enriched, "sep_AU"].min()),
        np.log10(combined.loc[not_a_enriched, "sep_AU"].min())),
    max(np.log10(combined.loc[a_enriched, "sep_AU"].max()),
        np.log10(combined.loc[not_a_enriched, "sep_AU"].max())),
    5,
)
ax[2].hist(combined.loc[a_enriched, "sep_AU"],     bins=sep_bins, histtype="step",
           linewidth=4, color="red", label=r"$\alpha$-Enriched", density=True)
ax[2].hist(combined.loc[not_a_enriched, "sep_AU"], bins=sep_bins, histtype="step",
           linewidth=4, color="k",   label=r"Not $\alpha$-Enriched", density=True, zorder=0)
ax[2].set_xlabel("Separation [au]")
ax[2].set_ylabel("Density")
ax[2].legend(framealpha=0, loc="upper left", fontsize=14)
ax[2].set_xscale("log")

print(combined.loc[totalage].query("alpha_fe > 0.40")[
    ["ms_source_id", "alpha_fe", "e_alpha_fe", "tot_age", "tot_age_error_lower", "tot_age_error_upper"]
])
fig.tight_layout()
fig.savefig(OUT_DIR / "age_alpha.pdf")
plt.close()
print(f"Saved {OUT_DIR / 'age_alpha.pdf'}")


# ── Carbon enrichment ─────────────────────────────────────────────────────
C_ENRICHED_CUTOFF = 0.25

fig, ax = plt.subplots(ncols=3, figsize=(18, 6), sharey=False)

ax[0].errorbar(combined.loc[normal | white_ll, "fe_h"],   combined.loc[normal | white_ll, "c_fe"],
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
ax[1].errorbar(combined.loc[white_ll, "total_age_lower_limit"],
               combined.loc[white_ll, "c_fe"],
               xerr=0.5, yerr=combined.loc[white_ll, "e_c_fe"], xlolims=True, **TOTAL_KW)
ax[1].errorbar(combined.loc[blue_ll, "total_age_lower_limit"],
               combined.loc[blue_ll, "c_fe"],
               xerr=0.5, yerr=combined.loc[blue_ll, "e_c_fe"], **LOWLIM_KW)
ax[1].axhline(y=C_ENRICHED_CUTOFF, c="k", ls="--")
ax[1].set_xlabel("Age [Gyr]")
ax[1].set_xlim(0, 12)
ax[1].sharey(ax[0])

c_enriched     = combined["c_fe"] > C_ENRICHED_CUTOFF
not_c_enriched = combined["c_fe"] <= C_ENRICHED_CUTOFF
sep_bins = np.logspace(
    min(np.log10(combined.loc[c_enriched, "sep_AU"].min()),
        np.log10(combined.loc[not_c_enriched, "sep_AU"].min())),
    max(np.log10(combined.loc[c_enriched, "sep_AU"].max()),
        np.log10(combined.loc[not_c_enriched, "sep_AU"].max())),
    8,
)
ax[2].hist(combined.loc[c_enriched,     "sep_AU"], bins=sep_bins, histtype="step",
           density=True, linewidth=4, color="red", label="C-Enriched")
ax[2].hist(combined.loc[not_c_enriched, "sep_AU"], bins=sep_bins, histtype="step",
           density=True, linewidth=4, color="k",   label="Not C-Enriched", zorder=0)
ax[2].set_xlabel("Separation [au]")
ax[2].set_ylabel("Density")
ax[2].legend(framealpha=0, fontsize=14)
ax[2].set_xscale("log")

fig.tight_layout()
fig.savefig(OUT_DIR / "age_carbon.pdf")
plt.close()
print(f"Saved {OUT_DIR / 'age_carbon.pdf'}")


# ── Barium enrichment ─────────────────────────────────────────────────────
BA_ENRICHED_CUTOFF = 1.0

fig, ax = plt.subplots(ncols=3, figsize=(18, 6), sharey=False)

ax[0].errorbar(combined.loc[normal | white_ll, "fe_h"],   combined.loc[normal | white_ll, "ba_fe"],
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
ax[1].errorbar(combined.loc[white_ll, "total_age_lower_limit"],
               combined.loc[white_ll, "ba_fe"],
               xerr=0.5, yerr=combined.loc[white_ll, "e_ba_fe"], xlolims=True, **TOTAL_KW)
ax[1].errorbar(combined.loc[blue_ll, "total_age_lower_limit"],
               combined.loc[blue_ll, "ba_fe"],
               xerr=0.5, yerr=combined.loc[blue_ll, "e_ba_fe"], **LOWLIM_KW)
ax[1].axhline(y=BA_ENRICHED_CUTOFF, c="k", ls="--")
ax[1].set_xlabel("Age [Gyr]")
ax[1].set_xlim(0, 12)
ax[1].sharey(ax[0])

ba_enriched     = combined["ba_fe"] > BA_ENRICHED_CUTOFF
not_ba_enriched = combined["ba_fe"] <= BA_ENRICHED_CUTOFF
sep_bins = np.logspace(
    min(np.log10(combined.loc[ba_enriched, "sep_AU"].min()),
        np.log10(combined.loc[not_ba_enriched, "sep_AU"].min())),
    max(np.log10(combined.loc[ba_enriched, "sep_AU"].max()),
        np.log10(combined.loc[not_ba_enriched, "sep_AU"].max())),
    8,
)
ax[2].hist(combined.loc[ba_enriched,     "sep_AU"], bins=sep_bins, histtype="step",
           density=True, linewidth=4, color="red", label="Ba-Enriched")
ax[2].hist(combined.loc[not_ba_enriched, "sep_AU"], bins=sep_bins, histtype="step",
           density=True, linewidth=4, color="k",   label="Not Ba-Enriched", zorder=0)
ax[2].set_xlabel("Separation [au]")
ax[2].set_ylabel("Density")
ax[2].legend(framealpha=0, fontsize=14)
ax[2].set_xscale("log")

fig.tight_layout()
fig.savefig(OUT_DIR / "age_barium.pdf")
plt.close()
print(f"Saved {OUT_DIR / 'age_barium.pdf'}")
