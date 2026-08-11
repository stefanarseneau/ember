"""WD color-magnitude diagram colored by cooling-age / total-age ratio."""

import numpy as np

from catalog.config import FIGURES_DIR
from catalog.analysis.load_data import load_catalog, setup_matplotlib


def main():
    plt, _ = setup_matplotlib()
    combined, normal, white_ll, blue_ll = load_catalog()

    # tot_age is only a reliable point estimate for age_class == 0 (normal);
    # white_ll/blue_ll rows carry a lower-limit tot_age that blows up the ratio.
    sel = combined.loc[normal].dropna(
        subset=["bp_rp", "phot_g_mean_mag", "wtd_par", "cool_age", "tot_age"]
    )
    abs_g = sel["phot_g_mean_mag"] + 5 * np.log10(sel["wtd_par"]) - 10
    ratio = sel["cool_age"] / sel["tot_age"]

    fig, ax = plt.subplots(figsize=(8, 9))
    sc = ax.scatter(
        sel["bp_rp"], abs_g, c=ratio, cmap="cool", vmin=0, vmax=1,
        s=10, edgecolor="k", linewidth=0.2, rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Cooling Age / Total Age")

    ax.set_xlabel("$G_{BP} - G_{RP}$ [mag]")
    ax.set_ylabel("$M_G$ [mag]")
    ax.invert_yaxis()

    out = FIGURES_DIR / "cmd_age_ratio.pdf"
    fig.savefig(out)
    plt.close()
    print(f"Saved {out}  ({len(sel)} sources)")

    print(f"Mean ratio: {np.mean(ratio)} +/- {np.std(ratio)}")


if __name__ == "__main__":
    main()