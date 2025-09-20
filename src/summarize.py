# wddate_summary_cli.py
from __future__ import annotations

import os
import sys
import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import typer
from wdwarfdate.wdwarfdate import WhiteDwarf

# --- your local imports (unchanged) ---
from mcmc import util  # noqa: F401
import MR_relation  # noqa: F401

app = typer.Typer(add_completion=False, help="Summarize MCMC chains and (optionally) compute WD ages with wdwarfdate.")

def make_datafile(path : str) -> pd.DataFrame:
    chains = glob.glob(os.path.join(path, "*.npy"))
    specids = [os.path.basename(chain).split('.')[0] for chain in chains]
    datafile = pd.DataFrame(columns = ['sourceid', 'radius', 'teff', 'loggH', 'loggHe', 'dist', 'av',
                                        'std_rr', 'std_tt', 'std_ggH', 'std_ggHe', 'std_dist', 'std_av',
                                        'cov_rt', 'cov_gtH', 'cov_gtHe', 'nsamps'])
    failed_files = []
    for ii, (specid, chainfile) in enumerate(zip(specids, chains)):
        try:
            chain = np.load(chainfile)
            teff_radius_component = chain[:,:2]
            loggH = MR_relation.logg_from_Teff_R(chain[:,0], chain[:,1], thickness = 'thick')
            loggHe = MR_relation.logg_from_Teff_R(chain[:,0], chain[:,1], thickness = 'thin')
            chainH = np.concatenate([chain, loggH[:, None]], axis=1)
            chainHe = np.concatenate([chain, loggHe[:, None]], axis=1)
            # extract the mean vector and covariance matrix
            maskH = ~np.any(np.isnan(chainH), axis=1)
            maskHe = ~np.any(np.isnan(chainHe), axis=1)
            meanH, covH = np.mean(chainH[maskH], axis=0), np.cov(chainH[maskH].T)
            meanHe, covHe = np.mean(chainHe[maskHe], axis=0), np.cov(chainHe[maskHe].T)
            datafile.loc[len(datafile)] = [specid, meanH[1], meanH[0], meanH[-1], meanHe[-1], meanH[2], meanH[3],
                                            np.sqrt(covH[1,1]), np.sqrt(covH[0,0]), np.sqrt(covH[-1,-1]), np.sqrt(covHe[-1,-1]),
                                            np.sqrt(covH[3,3]), np.sqrt(covH[4,4]), covH[0,1], covH[0,-1], covHe[0,-1], 
                                            chain.shape[0]]
        except ValueError as e:
            print(f"{e} || failed to read index {ii}: {chainfile}")
            failed_files.append(chainfile)
    return datafile, failed_files

def measure_ages(datafile : pd.DataFrame, method : str) -> pd.DataFrame:
    """measure WD ages using wdwarfdate"""
    assert method in ['fast_test', 'bayesian']
    teff, teff_err = datafile.teff.to_numpy(), datafile.std_tt.to_numpy()
    logg, logg_err = datafile.logg.to_numpy(), datafile.std_gg.to_numpy()
    cov = datafile.cov_gt.to_numpy()
    # compute the ages using wdwarfdate
    WD = WhiteDwarf(teff, teff_err, logg, logg_err, cov,
                    model_wd='DA', feh='p0.00', vvcrit='0.0',
                    model_ifmr='Cummings_2018_PARSEC',
                    high_perc=84, low_perc=16, method='fast_test',
                    datatype='log', save_plots=False,
                    display_plots=False)
    WD.calc_wd_age()
    # concatenate and return the results
    if method == 'fast_test':
        res = WD.results_fast_test.to_pandas()
    else:
        res = WD.results.to_pandas()
    return pd.concat([datafile, res], axis=1)

def run_summarize(
    inpath: Path,
    outpath: Path,
    wddate_fast: bool,
    wddate_bayesian: bool
):
    """
    Parse a directory of .npy chains into a summary file and optionally compute ages with wdwarfdate.
    If both --wddate-fast and --wddate-bayesian are given, 'fast' takes precedence (matches original script).
    """
    datafile, failed_files = make_datafile(inpath)

    if wddate_fast:
        typer.echo("\nMeasuring Ages (Fast)\n")
        datafile = measure_ages(datafile, method="fast_test")
        typer.echo("\nFinished Measuring Ages (Fast)\n")
    elif wddate_bayesian:
        typer.echo("\nMeasuring Ages (Bayesian)\n")
        datafile = measure_ages(datafile, method="bayesian")
        typer.echo("\nFinished Measuring Ages (Bayesian)\n")

    datafile.to_parquet(outpath)
    typer.echo(f"Saved summary to: {outpath}")
    if failed_files:
        typer.echo(f"Failed to read {len(failed_files)} files.")


if __name__ == "__main__":
    app()