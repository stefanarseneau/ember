# wddate_summary_cli.py
from __future__ import annotations

import os
import sys
import glob
from pathlib import Path
from typing import Tuple

import tqdm
import numpy as np
import pandas as pd
import typer
from wdwarfdate.wdwarfdate import WhiteDwarf

# --- local imports (unchanged) ---
from .mcmc import util  # noqa: F401
from . import MR_relation  # noqa: F401

app = typer.Typer(add_completion=False, help="Summarize MCMC chains and (optionally) compute WD ages with wdwarfdate.")

def make_datafile(path : str, discard : float = 0.3) -> pd.DataFrame:
    chains = glob.glob(os.path.join(path, "*.npy"))
    specids = [os.path.basename(chain).split('.')[0] for chain in chains]
    datafile = pd.DataFrame(columns = ['sourceid', 'radius', 'teff', 'dist', 'av',
                                        'std_rr', 'std_tt', 'std_dist', 'std_av',
                                        'cov_rt', 'nsamps'])
    failed_files = []
    for ii, (specid, chainfile) in tqdm.tqdm(enumerate(zip(specids, chains)), total=len(specids)):
        try:
            chain = np.load(chainfile)
            nn = int(discard*chain.shape[0])
            chain = chain[nn:]
            # extract the mean vector and covariance matrix
            mask = ~np.any(np.isnan(chain), axis=1)
            mean, cov = np.mean(chain[mask], axis=0), np.cov(chain[mask].T)
            datafile.loc[len(datafile)] = [specid, mean[1], mean[0], mean[2], mean[3],
                                            np.sqrt(cov[1,1]), np.sqrt(cov[0,0]), 
                                            np.sqrt(cov[2,2]), np.sqrt(cov[3,3]),
                                            cov[0,1], chain.shape[0]]
        except ValueError as e:
            print(f"{e} || failed to read index {ii}: {chainfile}")
            failed_files.append(chainfile)
    return datafile, failed_files

def measure_ages(datafile : pd.DataFrame, method : str) -> pd.DataFrame:
    """measure WD ages using wdwarfdate"""
    assert method in ['fast_test', 'bayesian']
    teff, teff_err = datafile.teff.to_numpy(), datafile.std_tt.to_numpy()
    logg, logg_err = datafile.loggH.to_numpy(), datafile.std_ggH.to_numpy()
    cov = datafile.cov_gt.to_numpy()
    # compute the ages using wdwarfdate
    WD = WhiteDwarf(teff, teff_err, logg, logg_err,
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
