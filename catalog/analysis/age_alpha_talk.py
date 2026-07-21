"""Presentation version of the age-alpha figure: left two panels, white points only."""

import numpy as np

from catalog.config import FIGURES_DIR
from catalog.analysis.load_data import load_catalog, setup_matplotlib

TOTAL_KW = dict(
    fmt="o", color="#eeeeee", markeredgecolor="k", ecolor="k",
    lw=2, capsize=4, markersize=7,
)

A_ENRICHED_CUTOFF = 0.1


def main():
    plt, _ = setup_matplotlib()
    combined, normal, white_ll, _ = load_catalog()

    combined_reliable_alpha = combined.query("e_alpha_fe < 0.1").copy()

    fig, ax = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)

    ax[0].errorbar(
        combined_reliable_alpha.loc[normal | white_ll, "fe_h"],
        combined_reliable_alpha.loc[normal | white_ll, "alpha_fe"],
        xerr=combined_reliable_alpha.loc[normal | white_ll, "e_fe_h"],
        yerr=combined_reliable_alpha.loc[normal | white_ll, "e_alpha_fe"],
        **TOTAL_KW,
    )
    ax[0].axhline(y=A_ENRICHED_CUTOFF, c="k", ls="--")
    ax[0].set_xlabel("[Fe/H]")
    ax[0].set_ylabel(r"[$\alpha$/Fe]")
    ax[0].set_ylim(-0.15, 0.5)

    ax[1].errorbar(
        combined_reliable_alpha.loc[normal, "tot_age"],
        combined_reliable_alpha.loc[normal, "alpha_fe"],
        xerr=combined_reliable_alpha.loc[normal, ["tot_age_error_lower", "tot_age_error_upper"]].values.T,
        yerr=combined_reliable_alpha.loc[normal, "e_alpha_fe"],
        **TOTAL_KW,
    )
    ax[1].errorbar(
        combined_reliable_alpha.loc[white_ll, "total_age_lower_limit"],
        combined_reliable_alpha.loc[white_ll, "alpha_fe"],
        xerr=0.5,
        yerr=combined_reliable_alpha.loc[white_ll, "e_alpha_fe"],
        xlolims=True,
        **TOTAL_KW,
    )
    ax[1].axhline(y=A_ENRICHED_CUTOFF, c="k", ls="--")
    ax[1].set_xlabel("Age [Gyr]")
    ax[1].set_xlim(0, 13)

    fig.tight_layout()
    out = FIGURES_DIR / "age_alpha_talk.pdf"
    fig.savefig(out)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
