# run_mcmc_typer.py
from __future__ import annotations

import os
import sys
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

# local imports
from mcmc import fitting, xpspec, util, photometry  # noqa: F401
import interpolator  # noqa: F401

app = typer.Typer(add_completion=False, help="""
Run MCMC on white dwarf photometry to infer physical parameters.
""")


def get_split(df : pd.DataFrame, num_tasks : int) -> pd.DataFrame:
    """split the dataframe by the task id"""
    if num_tasks != 0:
        chunks = np.array_split(df, num_tasks)
        task_id = os.getenv("SGE_TASK_ID")
        task_id = int(task_id) if task_id is not None else 1
        df_segment = chunks[task_id-1]
        return df_segment
    else:
        return df
    
def read_chainnames(path : str) -> dict:
    files = glob.glob(str(Path(path) / "*.npy"))
    names = [Path(file).stem for file in files]
    return np.array(names, dtype=np.int64)


@app.command("run")
def run_fitsed(
    inpath: Path,
    outpath: Path,
    rootpath: Path,
    xpphoto: bool,
    gaia: bool,
    sdss: bool,
    jplus: bool,
    synthetic: bool,
    panstarrs: bool,
    numtasks: int,
    mcmc: bool,
    source_id: str,
    ra: str,
    dec: str,
    parallax: str,
    parallax_error: str,
    meanav: str,
    gravz: str,
    gravz_error: str
):
    # Expand paths
    rootpath = rootpath.expanduser()
    outpath.mkdir(parents=True, exist_ok=True)

    logg_function = util.get_logg_function()
    sdssv_data = (
        pd.read_parquet(inpath)
        .drop_duplicates(source_id)
    )
    prev_len = len(sdssv_data)

    # Remove already-run ids by checking existing .npy chains in outpath
    try:
        names = read_chainnames(outpath)
        sdssv_data = sdssv_data[~sdssv_data[source_id].astype(np.int64).isin(names)]
    except FileNotFoundError:
        # If outpath doesn't exist yet or no files are present, proceed with full set
        pass

    # SGE chunking (optional)
    df_segment = get_split(sdssv_data, numtasks)

    # --- Band selection flags (kept as simple booleans for downstream use) ---
    photofunc = xpspec.process_dataframe if xpphoto else photometry.process_dataframe
    bands_selected = {
        "gaia": gaia,
        "sdss": sdss,
        "jplus": jplus,
        "synthetic": synthetic,
        "panstarrs": panstarrs,
    }
    selected_systems = [flag for flag, issel in bands_selected.items() if issel]
    print(selected_systems)
    assert len(selected_systems) != 0, "No systems selected!"

    typer.echo(f"Total rows before filtering: {prev_len}")
    typer.echo(f"Rows remaining after removing existing chains: {len(sdssv_data)}")
    typer.echo(f"Rows in this SGE task segment: {len(df_segment)}")
    typer.echo(f"Bands selected: {', '.join(selected_systems) or 'None'}")
    typer.echo(f"Output directory: {outpath.resolve()}")

    fittype = 'mcmc' if mcmc else 'leastsq'
    print(fittype)
    fluxdf_sdss, results_sdss = fitting.pipeline(
        df_segment, 
        selected_systems,
        photofunc, 
        logg_function = logg_function, 
        mode = fittype,
        source_id = source_id, 
        ra = ra, 
        dec = dec, 
        parallax = parallax, 
        parallax_error = parallax_error, 
        meanAV = meanav, 
        gravz = gravz if gravz != '' else None, 
        gravz_error = gravz_error if gravz_error != '' else None,
        outfile = outpath
    )

if __name__ == "__main__":
    app()
