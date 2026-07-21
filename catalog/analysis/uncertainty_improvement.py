"""Parallax and mass SNR improvement from wide-binary weighted parallaxes."""

from pathlib import Path

import numpy as np
import pandas as pd

from catalog.config import FIGURES_DIR
from catalog.analysis.load_data import load_catalog, setup_matplotlib


def main():
    plt, _ = setup_matplotlib()
    combined, *_ = load_catalog()

    wd_color = "#302424"

    ngf_data = pd.read_parquet(
        Path.home() / "observational/catalogs/nicola_wds/ngf21_wds.pqt"
    )[["GaiaEDR3", "MassH", "e_MassH", "Plx", "e_Plx"]]

    data_ngf = pd.merge(combined, ngf_data, left_on="source_id", right_on="GaiaEDR3")
    data_ngf["e_Mass"] = 0.5 * (data_ngf["e_Mass_lower"] + data_ngf["e_Mass_upper"])

    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))

    all_plx_vals = np.concatenate([data_ngf.wtd_par / data_ngf.e_wtd_par,
                                    data_ngf.Plx / data_ngf.e_Plx])
    all_plx_vals = all_plx_vals[np.isfinite(all_plx_vals) & (all_plx_vals > 0) & (all_plx_vals < 1e4)]
    plx_logbins = np.logspace(np.log10(all_plx_vals.min()), np.log10(all_plx_vals.max()), 20)

    ax[0].hist(data_ngf.wtd_par / data_ngf.e_wtd_par,
               bins=plx_logbins, color=wd_color, histtype="step", linewidth=4)
    ax[0].hist(data_ngf.Plx / data_ngf.e_Plx,
               bins=plx_logbins, color=wd_color, histtype="step", ls="--", linewidth=4)
    ax[0].set_xlabel(r"$\omega / \sigma_\omega$")
    ax[0].set_ylabel("N")
    ax[0].set_xscale("log")

    all_mass_vals = np.concatenate([data_ngf.Mass / data_ngf.e_Mass,
                                     data_ngf.MassH / data_ngf.e_MassH])
    all_mass_vals = all_mass_vals[np.isfinite(all_mass_vals) & (all_mass_vals > 0) & (all_mass_vals < 1e4)]
    mass_logbins = np.logspace(np.log10(all_mass_vals.min()), np.log10(all_mass_vals.max()), 20)

    ax[1].hist(data_ngf.Mass / data_ngf.e_Mass,
               bins=mass_logbins, color=wd_color, histtype="step", linewidth=4)
    ax[1].hist(data_ngf.MassH / data_ngf.e_MassH,
               bins=mass_logbins, color=wd_color, histtype="step", ls="--", linewidth=4)
    ax[1].set_xlabel(r"$M / \sigma_M$")
    ax[1].set_ylabel("N")
    ax[1].set_xscale("log")

    wtd_psnr = data_ngf.wtd_par / data_ngf.e_wtd_par
    ngf_psnr = data_ngf.Plx / data_ngf.e_Plx
    print(f"Parallax improvement ratio: {np.nanmedian(wtd_psnr) / np.nanmedian(ngf_psnr):.3f}")

    wtd_msnr = data_ngf.Mass / data_ngf.e_Mass
    ngf_msnr = data_ngf.MassH / data_ngf.e_MassH
    print(f"Mass improvement ratio: {np.nanmedian(wtd_msnr) / np.nanmedian(ngf_msnr):.3f}")

    fig.tight_layout()
    out = FIGURES_DIR / "uncertainty_improvement.pdf"
    fig.savefig(out)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
