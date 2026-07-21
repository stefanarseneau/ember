"""ΔM vs Δt scatter plots for WD+WD wide binary pairs.

Saves two figures:
  inconsistent_ages_xp.pdf    — XP spectrum ages (Δt_total, ~142 pairs)
  inconsistent_ages_tyler.pdf — Tyler MESA IFMR total ages (Δt_total, ~1000 pairs)

Points are coloured by 1σ consistency: blue = consistent (|Δt| ≤ σ_Δt),
red = inconsistent.

Requires wdwd_pairs.pqt built by:
    python -m catalog.build_wdwd --elbadry   (one-time download)
    python -m catalog.build_wdwd --pairs
"""

import numpy as np

from catalog.config import FIGURES_DIR
from catalog.analysis.load_data import setup_matplotlib


def _inconsistency_stats(dt, edt, n_mc=10000, rng=None):
    """Return (inconsist_mask, f_mc_mean, f_mc_std).

    A pair is 1σ inconsistent if |Δt| > σ_Δt, i.e. the error bar does not
    overlap zero in either direction.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    inconsist = np.abs(dt) - edt > 0
    draws     = rng.normal(loc=dt, scale=edt, size=(n_mc, len(dt)))
    frac_mc   = (np.abs(draws) / edt[None, :] > 1).mean(axis=1)
    return inconsist, frac_mc.mean(), frac_mc.std()


def _make_figure(plt, sel, dm_col, dt_col, edt_col, ylabel):
    dt   = sel[dt_col].to_numpy()
    edt  = sel[edt_col].to_numpy()
    inconsist, f_mean, f_std = _inconsistency_stats(dt, edt)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        sel.loc[~inconsist, dm_col], sel.loc[~inconsist, dt_col],
        yerr=sel.loc[~inconsist, edt_col],
        fmt="o", markeredgecolor="k", ecolor="k", lw=2, capsize=3,
        markersize=7, color="#2b83ff", label="Consistent",
    )
    ax.errorbar(
        sel.loc[inconsist, dm_col], sel.loc[inconsist, dt_col],
        yerr=sel.loc[inconsist, edt_col],
        fmt="o", markeredgecolor="k", ecolor="k", lw=2, capsize=3,
        markersize=7, color="#ce0048", label="Inconsistent",
    )

    ax.axhline(0, color="k", lw=0.8, ls="--", zorder=1)
    ax.text(
        0.45, 0.95,
        rf"${100*f_mean:.0f}\pm{100*f_std:.0f}\%$ inconsistent at $1\sigma$",
        transform=ax.transAxes, va="top", fontsize=18,
    )
    ax.set_xlabel(r"$\Delta M\ (M_1 - M_2,\ M_\odot)$")
    ax.set_ylabel(ylabel)
    ax.legend(framealpha=0)

    return fig, inconsist, f_mean, f_std


def main():
    import pandas as pd
    from catalog.config import DATA_DIR

    plt, _ = setup_matplotlib()

    pairs_path = DATA_DIR / "wdwd_pairs.pqt"
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"{pairs_path} not found — run:\n"
            "  python -m catalog.build_wdwd --elbadry\n"
            "  python -m catalog.build_wdwd --pairs"
        )

    pairs = pd.read_parquet(pairs_path)

    # ── XP spectrum total ages ────────────────────────────────────────────
    xp_valid = pairs[["dM_xp", "dt_log_age_xp", "e_dt_log_age_xp"]].notna().all(axis=1)
    sel_xp   = pairs[xp_valid].copy()
    print(f"XP pairs (finite dM, dt_total): {len(sel_xp)}")

    fig_xp, inconsist_xp, f_mean_xp, f_std_xp = _make_figure(
        plt, sel_xp,
        dm_col="dM_xp", dt_col="dt_log_age_xp", edt_col="e_dt_log_age_xp",
        ylabel=r"$\Delta \tau_\mathrm{Total}\ (t_1 - t_2,\ \mathrm{Gyr})$",
    )
    out_xp = FIGURES_DIR / "inconsistent_ages_xp.pdf"
    fig_xp.savefig(out_xp)
    plt.close(fig_xp)
    print(f"Saved {out_xp}  ({inconsist_xp.sum()}/{len(sel_xp)} inconsistent, "
          f"{100*f_mean_xp:.0f}±{100*f_std_xp:.0f}%)")

    # ── Tyler MESA IFMR total ages ────────────────────────────────────────
    h_valid = pairs[["dM_H", "dt_tot_age_MESA_IFMR", "e_dt_tot_age_MESA_IFMR"]].notna().all(axis=1)
    sel_h   = pairs[h_valid].copy()
    print(f"Tyler pairs (finite dM, dt_total MESA IFMR): {len(sel_h)}")

    fig_h, inconsist_h, f_mean_h, f_std_h = _make_figure(
        plt, sel_h,
        dm_col="dM_H", dt_col="dt_tot_age_MESA_IFMR", edt_col="e_dt_tot_age_MESA_IFMR",
        ylabel=r"$\Delta \tau_\mathrm{Total}\ (t_1 - t_2,\ \mathrm{Gyr})$",
    )
    out_h = FIGURES_DIR / "inconsistent_ages_tyler.pdf"
    fig_h.savefig(out_h)
    plt.close(fig_h)
    print(f"Saved {out_h}  ({inconsist_h.sum()}/{len(sel_h)} inconsistent, "
          f"{100*f_mean_h:.0f}±{100*f_std_h:.0f}%)")


if __name__ == "__main__":
    main()
