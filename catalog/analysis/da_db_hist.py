"""[Fe/H] distribution comparison between DA and DB white dwarfs."""

import argparse

import numpy as np
from scipy import stats

from catalog.config import FIGURES_DIR
from catalog.analysis.load_data import load_catalog, setup_matplotlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skippower", action="store_true", help="Skip power analysis")
    args = parser.parse_args()

    plt, _ = setup_matplotlib()
    combined, *_ = load_catalog()

    feh_da = combined.query("spectype == 'DA'")[["fe_h", "e_fe_h", "Teff", "e_Teff_lower", "e_Teff_upper", "survey"]]
    feh_db = combined.query("spectype == 'DB'")[["fe_h", "e_fe_h", "Teff", "e_Teff_lower", "e_Teff_upper", "survey"]]

    print(f"[Total]        DA: {len(feh_da)}  DB: {len(feh_db)}")
    for survey in ["APOGEE+ASTRA", "GALAH", "LAMOST"]:
        if survey == "APOGEE+ASTRA":
            num_da = len(feh_da.query("survey == 'APOGEE' or survey == 'ASTRA'"))
            num_db = len(feh_db.query("survey == 'APOGEE' or survey == 'ASTRA'"))
        else:
            num_da = len(feh_da.query(f"survey == '{survey}'"))
            num_db = len(feh_db.query(f"survey == '{survey}'"))
        whitespace = len("APOGEE+ASTRA") - len(survey)
        print(f"[{survey}] {' '*whitespace}DA: {num_da}, DB: {num_db}")

    ks_stat, p_value = stats.ks_2samp(feh_da["fe_h"].dropna(), feh_db["fe_h"].dropna(), alternative="greater")
    print(f"Mean [Fe/H] DA: {feh_da.fe_h.mean():.2f} +/- {feh_da.fe_h.std() / len(feh_da)**0.5:.2f}")
    print(f"Mean [Fe/H] DB: {feh_db.fe_h.mean():.2f} +/- {feh_db.fe_h.std() / len(feh_db)**0.5:.2f}")
    print(f"KS statistic: {ks_stat:.3f}, p-value (DB > DA): {p_value:.3f}")

    sub = combined.query("survey in ('APOGEE', 'ASTRA')")
    feh_da_aas = sub.query("spectype == 'DA'")["fe_h"].dropna()
    feh_db_aas = sub.query("spectype == 'DB'")["fe_h"].dropna()
    ks_aas, p_aas = stats.ks_2samp(feh_da_aas, feh_db_aas, alternative="greater")
    print(f"[APOGEE+ASTRA only] DA: {len(feh_da_aas)}  DB: {len(feh_db_aas)}  KS p (DB > DA) = {p_aas:.3f}")

    da_counts, bins = np.histogram(feh_da["fe_h"].dropna())
    db_counts, _ = np.histogram(feh_db["fe_h"].dropna(), bins=bins)

    fig, ax = plt.subplots()
    ax.hist(feh_da["fe_h"], bins=bins, histtype="step", color="k", linewidth=3, label="DA")
    ax.hist(feh_db["fe_h"], bins=bins, histtype="step", color="goldenrod", linewidth=3, label="DB")
    ax.text(0.75, 0.95, f"KS $p = {p_value:.3f}$", transform=ax.transAxes, va="top", fontsize=16)
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("N")
    ax.legend(framealpha=0)
    out = FIGURES_DIR / "da_db_hist.pdf"
    fig.savefig(out)
    plt.close()
    print(f"Saved {out}")

    fig, ax = plt.subplots()
    ax.hist(feh_da["fe_h"], bins=bins, histtype="step", color="k", linewidth=3, 
                label=rf"DA ($N = {len(feh_da.dropna(subset=['fe_h']))}$)", 
                cumulative=True, density=True)
    ax.hist(feh_db["fe_h"], bins=bins, histtype="step", color="goldenrod", linewidth=3, 
                label=rf"DB ($N = {len(feh_db.dropna(subset=['fe_h']))})$", 
                cumulative=True, density=True)
    ax.text(0.05, 0.75, f"KS $p = {p_value:.3f}$", transform=ax.transAxes, va="top", fontsize=16)
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("CDF")
    ax.legend(framealpha=0)
    out = FIGURES_DIR / "da_db_cdf.pdf"
    fig.savefig(out)
    plt.close()
    print(f"Saved {out}")

    if args.skippower:
        return

    rng = np.random.default_rng(42)
    N_trials = 5000
    thresh_3s = stats.norm.sf(3)
    thresh_5s = stats.norm.sf(5)
    feh_da_vals = feh_da["fe_h"].dropna().values
    feh_db_vals = feh_db["fe_h"].dropna().values

    print(f"\n{'N_DB':>6}  {'power@3σ':>10}  {'power@5σ':>10}")
    for n_db in [26, 30, 40, 50, 75, 100, 150, 200, 300]:
        pvals = np.array([
            stats.ks_2samp(
                feh_da_vals,
                rng.choice(feh_db_vals, size=n_db, replace=True),
                alternative="greater",
            ).pvalue
            for _ in range(N_trials)
        ])
        print(f"{n_db:>6}  {np.mean(pvals < thresh_3s):>10.3f}  {np.mean(pvals < thresh_5s):>10.3f}")


if __name__ == "__main__":
    main()
