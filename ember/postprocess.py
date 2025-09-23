import os
import sys
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from .ages import chainhandler  # noqa: E402

app = typer.Typer(add_completion=False, help="""
Measure ages or hydrogen layer masses with monte carlo post-processing.

Example:
  ember postprocess CHAINDIR OUTDIR \\
      --pqtpath ../data/catalogs/APOGEE/wdms_xmatch_ASTRA_xmatch_vincent24.pqt \\
      --feh1 fe_h --feh2 e_fe_h --logger
""")


def find_existing_files(filenames, directory):
    """Return a list of filenames that exist in the given directory."""
    existing = []
    for fname in filenames:
        full_path = os.path.join(directory, fname)
        if os.path.isfile(full_path):
            existing.append(full_path)  # or just fname if you prefer
    return existing

def get_split(chains : list[str], numtasks : int) -> pd.DataFrame:
    """split the dataframe by the task id"""
    if numtasks != 0:
        chunks = np.array_split(chains, numtasks)
        task_id = os.getenv("SGE_TASK_ID")
        task_id = int(task_id) if task_id is not None else 1
        chain_segment = chunks[task_id-1]
        return chain_segment, task_id
    else:
        return chains

# --- helper to parse string as float or column ---
def get_array(value, df=None, fallback=0.0, length=1):
    if value == '':
        return np.full(length, fallback)
    try:
        # try to interpret as float
        fval = float(value)
        return np.full(length, fval)
    except ValueError:
        # otherwise treat as column
        if df is None:
            raise ValueError(f"Tried to use column '{value}' but no dataframe provided.")
        return df[value].to_numpy()

@app.command()
def run_postprocess(
    chainpath: Path = typer.Argument(..., help="Path to read chains (directory)."),
    outpath: Path = typer.Argument(..., help="Path to save modified chains (directory)."),
    logger: bool = typer.Option(False, "--logger", "-l", help="Save the logging file."),
    pqtpath: Optional[Path] = typer.Option(None, "--pqtpath", help="Parquet file with [Fe/H] values."),
    feh1: str = typer.Option("fe_h", "--feh1", "-f", help="First [Fe/H] value: float or column name."),
    feh2: str = typer.Option("", "--feh2", "-e", help="Second [Fe/H] value: float or column name."),
    target: str = typer.Option("log_age", "--target", "-t", help="Column to interpolate onto."),
    progress: bool = typer.Option(True, "--progress/--no-progress", "-p/-P",
                                  help="Show a tqdm progress bar (default: show)."),
    uniform: bool = typer.Option(False, "--uniform/--gaussian", "-u/-g",
                                 help="Use uniform (True) vs gaussian (False) distributions."),
    numtasks: int = typer.Option(1, "--numtasks", "-n", help="Number of CPU chunks (SGE array-style)."),
):
    """
    Measure ages from WD chains. If --pqtpath is provided, chains are matched by
    basename (before .npy) to 'wd_source_id' rows in the parquet file to retrieve
    per-object [Fe/H] columns; otherwise, constants may be passed via --feh1/--feh2.
    """
    # Collect chains + per-object metadata if provided
    if pqtpath is not None:
        datafile = pd.read_parquet(pqtpath)
        candidates = [f"{file}.npy" for file in datafile.wd_source_id.astype(str)]
        chains = find_existing_files(candidates, chainpath)
        chainseg, task_id = get_split(chains, numtasks)

        # Match back to rows
        chname = [Path(ch).stem for ch in chainseg]
        matched_rows = datafile[datafile["wd_source_id"].astype(str).isin(chname)]

        feh1_arr = get_array(feh1, df=matched_rows, length=len(matched_rows))
        feh2_arr = get_array(feh2, df=matched_rows, fallback=0.0, length=len(matched_rows))
    else:
        chains = glob.glob(str(chainpath / "*.npy"))
        chainseg, task_id = get_split(chains, numtasks)

        # When not using parquet, feh1/feh2 must be floats (or '' -> fallback)
        feh1_arr = get_array(feh1, df=None, length=len(chainseg))           # raises if not float
        feh2_arr = get_array(feh2, df=None, fallback=0.0, length=len(chainseg))

    # Ensure output directory
    outpath.mkdir(parents=True, exist_ok=True)

    # Build 'uniform' mask (force uniform where feh1 is NaN)
    uniform_mask = np.full(feh1_arr.shape[0], uniform, dtype=bool)
    nan_idx = np.isnan(feh1_arr)
    uniform_mask[nan_idx] = True

    # Fill NaNs for handler expectations
    feh1_arr[nan_idx] = -1
    feh2_arr[np.isnan(feh2_arr)] = 0.5

    # Run
    result, logdf, failed = chainhandler.measure_chains(
        chainseg,
        str(outpath),
        target,
        feh1=feh1_arr,
        feh2=feh2_arr,
        uniform=uniform_mask,
        progress=progress,
    )

    # Optional logger save
    if logger and logdf is not None:
        loggerpath = outpath / (f"logger_{task_id}.pqt" if numtasks and numtasks > 1 else "logger.pqt")
        logdf.to_parquet(loggerpath)
        typer.echo(f"Saved log to: {loggerpath}")

    # Minimal completion message
    typer.echo(f"Processed {len(chainseg)} chains (task {task_id}). Failed: {len(failed) if failed is not None else 0}")


if __name__ == "__main__":
    app()
