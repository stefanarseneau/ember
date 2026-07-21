"""WD+MS companion CMD colored by evolutionary state (MS / SG / RG)."""

import numpy as np

from catalog.config import (
    FIGURES_DIR, GIANT_CMD_SG_LINE_AT_1,
    GIANT_CMD_EVOLVED_MIN_BP_RP, GIANT_CMD_GIANT_ABS_MAG_MAX,
)
from catalog.analysis.load_data import load_catalog, setup_matplotlib
from catalog.wdms.evolutionary_state import (
    companion_abs_g, build_mist_grid_cache, load_full_isochrone_tracks,
)


def main():
    plt, _ = setup_matplotlib()
    combined, *_ = load_catalog()

    abs_g = companion_abs_g(combined)
    bp_rp = combined["ms_bp_rp"].to_numpy(dtype=float)
    ms_class = combined["ms_evol_class"]

    # Isochrone tracks are plotting context only — not used by classify().
    ref_feh = float(np.nanmedian(combined["fe_h"].dropna())) if combined["fe_h"].notna().any() else 0.0
    build_mist_grid_cache(ref_feh)
    grid = load_full_isochrone_tracks()

    fig, ax = plt.subplots(figsize=(8, 9))

    ages = sorted(grid["age_gyr"].unique())
    cmap = plt.get_cmap("viridis", len(ages))
    for i, age in enumerate(ages):
        track = grid[grid["age_gyr"] == age].sort_values("eep")
        ax.plot(track["bp_rp"], track["G_mag"], c=cmap(i), lw=0.8, alpha=0.6, zorder=1,
                label=f"{age:g} Gyr isochrone")

    groups = [
        ("MS", "#9e9e9e", 3),
        ("SG", "#1f77b4", 10),
        ("RG", "#d62728", 10),
    ]
    for label, color, zorder in groups:
        mask = (ms_class == label).to_numpy()
        ax.scatter(bp_rp[mask], abs_g[mask], s=6, c=color, alpha=0.7,
                   label=f"{label} ($N={mask.sum()}$)", zorder=zorder, rasterized=True)

    sg_grid = np.linspace(0, GIANT_CMD_EVOLVED_MIN_BP_RP, 300)
    ax.plot(sg_grid, GIANT_CMD_SG_LINE_AT_1 * sg_grid, c="k", lw=1.5, ls="--",
            zorder=20, label="SG line")

    giant_grid = np.linspace(GIANT_CMD_EVOLVED_MIN_BP_RP, np.nanmax(bp_rp), 200)
    ax.plot(giant_grid, np.full_like(giant_grid, GIANT_CMD_GIANT_ABS_MAG_MAX), c="k", lw=1.5,
            ls="-.", zorder=20, label="RG threshold")
    ax.axvline(GIANT_CMD_EVOLVED_MIN_BP_RP, c="k", lw=1, ls=":", alpha=0.4)

    ax.set_xlabel("$G_{BP} - G_{RP}$ [mag]")
    ax.set_ylabel("$M_G$ [mag]")
    ax.set_xlim(bp_rp.min() - 0.2, bp_rp.max() + 0.2)
    ax.set_ylim(np.nanmax(abs_g) + 0.5, np.nanmin(abs_g) - 0.5)
    ax.legend(framealpha=0, fontsize=7, loc="lower left", ncol=2)

    out = FIGURES_DIR / "ms_evol_cmd.pdf"
    fig.savefig(out)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
