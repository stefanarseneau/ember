# ember.py
import typer
from typing import Optional
from enum import Enum
from pathlib import Path

from pathlib import Path
from typing import List, Optional, Literal
import typer

from fitsed import run_fitsed
from postprocess import run_postprocess
from summarize import run_summarize
from submit import write_qsub_script, qsub_submit

def _detect_numtasks(argv: list[str]) -> int:
    """
    Detect --numtasks from argv. Supports:
      --numtasks N    and    --numtasks=N
    Returns 1 if not present. Last occurrence wins.
    """
    found = None
    for i, tok in enumerate(argv):
        if (tok == "--numtasks" or tok == "-nt") and i + 1 < len(argv):
            nxt = argv[i + 1]
            if nxt and not nxt.startswith("-"):
                try:
                    found = int(nxt)
                except ValueError:
                    pass
        elif tok.startswith("--numtasks=") or tok.startswith("-nt="):
            try:
                found = int(tok.split("=", 1)[1])
            except ValueError:
                pass
    return found if (found is not None and found > 0) else 1


app = typer.Typer(
    add_completion=False, 
    help="""
**EMBER** — (E)stimating (M)ass and (B)olometric properties of (E)volved (R)emnants
  by Stefan Arseneau (Boston University)

Fit white dwarf spectral energy distributions and convert
them into ages and/or hydrogen layers.

If you find this tool useful, please cite the following
papers, which EMBER is built on:
 - Huang et al. 2024, ApJS, arxiv:2401.12006
 - Bauer et al. 2025, ApJS, In Prep
""",
    rich_markup_mode="markdown",           # <-- enable Markdown in help/docstrings
    context_settings={"help_option_names": ["-h", "--help"]},  # add -h alias
)

app.__doc__ = """
**EMBER** — (E)stimating (M)ass and (B)olometric properties of (E)volved (R)emnants

Fit white dwarf spectral energy distributions and convert
them into ages and/or hydrogen layers.

If you find this tool useful, please cite the following
papers, which EMBER is built on:
 - Huang et al. 2024, ApJS, arxiv:2401.12006
 - Bauer et al. 2025, ApJS, In Prep
(see also Mark Hollands' mass-radius intepolation code: https://github.com/mahollands/MR_relation)
"""

class SubCmd(str, Enum):
    FIT_SED = "fit-sed"
    POSTPROCESS = "postprocess"

@app.command(
    "submit",
    help="Submit an EMBER job to SGE/UGE via qsub.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def submit(
    ctx: typer.Context,                                            # <— keep this
    command: SubCmd = typer.Argument(..., help="EMBER subcommand"),
    job_name: str = typer.Option("ember", "--job-name", "-N"),
    logs: Path = typer.Option(Path("./logs"), "--logs"),
    queue: Optional[str] = typer.Option(None, "--queue", "-q"),
    project: Optional[str] = typer.Option(None, "--project", "-P"),
    pe: Optional[str] = typer.Option(None, "--pe"),
    slots: int = typer.Option(1, "--slots"),
    mem: Optional[str] = typer.Option(None, "--mem"),
    hours: Optional[int] = typer.Option(None, "--hours"),
    email: Optional[str] = typer.Option(None, "--email"),
    hold_jid: Optional[str] = typer.Option(None, "--hold-jid"),
    conda_env: Optional[str] = typer.Option(None, "--conda-env"),
    script_out: Optional[Path] = typer.Option(None, "--script-out"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    # pass through everything after the subcommand name unchanged
    passthrough = list(ctx.args)

    # infer array size only from --numtasks (default 1)
    array_size = _detect_numtasks(passthrough)

    ember_cmd = ["ember", command.value, *passthrough]
    script_path = script_out or (logs / f"{job_name}.sh")

    write_qsub_script(
        script_path=script_path,
        ember_cmd=ember_cmd,
        job_name=job_name,
        logs_dir=logs,
        numtasks=array_size,  # only adds -t 1-N when > 1 (see note below)
        queue=queue,
        project=project,
        pe=pe,
        slots=slots,
        mem=mem,
        hours=hours,
        email=email,
        hold_jid=hold_jid,
        conda_env=conda_env,
    )

    typer.echo(f"Wrote job script: {script_path}")
    typer.echo(f"qsub {script_path}")
    if dry_run:
        return
    res = qsub_submit(script_path)
    if res.returncode == 0 and res.stdout.strip():
        typer.echo(res.stdout.strip())
    else:
        typer.echo(res.stderr.strip())
        raise typer.Exit(code=1)

@app.command("fit-sed",
             help = """
Run MCMC on white dwarf photometry to infer physical parameters.

This script queries external databases to get photometry which it then fits using MCMC. The photometry
can either be "real" survey photometry from Gaia, SDSS, J-PLUS, or PANSTARRs (this is default) or it
can be synthetic photometry convolved from Gaia BP/RP spectra. It reads in the Gaia source ids as well
as the information it needs (ra, dec, parallax, parallax error, extinction) from a parquet file, `inpath`.
You'll most likely need to tell it what columns to look at.

The photometry to use is controlled by the `--gaia`, `--sdss`, `--jplus`, and `--panstarrs` flags. If
the `--xpphoto` flag is passed (which tells the fitting code to use BP/RP spectra) you can also use the
`--synthetic` flag, which fits in 24 evenly-spaced rectangular passbands of width 300 angstroms.

For best results, pass only the `--xpphoto` and `--synthetic` flags. BP/RP spectra are calibrated to the
CALSPEC library using the corrections of Huang+2024 (arxiv:2401.12006).

Example:
    ember fit-sed wdms_widebinaries.pqt wdms_widebinary_chains/ \\
        --synthetic --xpphoto --inpath="/project/mesaelm/sdss-binary" \\
        --numtasks=256
""")
def fitsed(
    inpath: Path = typer.Argument(..., help="Path to the parquet with column information"),
    outpath: Path = typer.Argument(..., help="Path to save chains."),
    rootpath: Path = typer.Option(
        Path("~/observational/grav-z/gravz-hlayer"),
        "--inpath",
        help="Input root path (expanded).",
    ),
    xpphoto: bool = typer.Option(False, "--xpphoto", "-x", help="Use photometry from Gaia XP spectra."),
    gaia: bool = typer.Option(False, "--gaia", "-g", help="Fit photometry with Gaia bands."),
    sdss: bool = typer.Option(False, "--sdss", "-s", help="Fit photometry with SDSS bands."),
    jplus: bool = typer.Option(False, "--jplus", "-j", help="Fit photometry with JPLUS bands."),
    synthetic: bool = typer.Option(False, "--synthetic", "-t", help="Fit photometry with Synthetic bands."),
    panstarrs: bool = typer.Option(False, "--panstarrs", "-p", help="Fit photometry with PanSTARRS bands."),
    numtasks: int = typer.Option(0, "--numtasks", "-nt", help="Number of CPU tasks (SGE-style splitting)."),
    mcmc : bool = typer.Option(True, "--mcmc/--leastsq", help="Fit using MCMC or least squares?"),
    # --- column names ---
    source_id: str = typer.Option("wd_source_id", "--source-id", help="Column name for source IDs."),
    ra: str = typer.Option("wd_ra", "--ra", help="Column name for RA."),
    dec: str = typer.Option("wd_dec", "--dec", help="Column name for Dec."),
    parallax: str = typer.Option("ms_parallax", "--parallax", help="Column name for parallax."),
    parallax_error: str = typer.Option("ms_parallax_error", "--parallax-error", help="Column name for parallax error."),
    meanav: str = typer.Option("meanAV", "--meanav", help="Column name for extinction (A_V)."),
    gravz: str = typer.Option("", "--gravz", help="Column name for gravitational redshift (optional)."),
    gravz_error: str = typer.Option("", "--gravz-error", help="Column name for grav redshift error (optional)."),
):
    run_fitsed(
        inpath = inpath,
        outpath = outpath,
        rootpath = rootpath,
        xpphoto = xpphoto,
        gaia = gaia,
        sdss = sdss,
        jplus = jplus,
        synthetic = synthetic,
        panstarrs = panstarrs,
        numtasks = numtasks,
        mcmc = mcmc,
        source_id = source_id,
        ra = ra,
        dec = dec,
        parallax = parallax,
        parallax_error = parallax_error,
        meanav = meanav,
        gravz = gravz,
        gravz_error = gravz_error
    )
    typer.echo(f"fit-sed done")

@app.command("summarize",
            help = """
Parse a directory of .npy chains into a summary file and optionally compute ages with wdwarfdate.

Takes in a directory such as that produced by `fit-sed` and produces a parquet file which contains all 
of the mean values, their standard deviations, and the covariances of radius and effective temperature. 
It can also use `wdwarfdate` to produce age estimates, but these are not as good as the MIST ages I 
suspect. If both --wddate-fast and --wddate-bayesian are given, 'fast' takes precedence (matches original 
script).
""")
def summarize(
    inpath: Path = typer.Argument(..., help="Path from which to read chains."),
    outpath: Path = typer.Argument(..., help="Path to save summary parquet."),
    wddate_fast: bool = typer.Option(False, "--wddate-fast", "-wf", help="Calculate ages using wdwarfdate fast."),
    wddate_bayesian: bool = typer.Option(False, "--wddate-bayesian", "-wb", help="Calculate ages using wdwarfdate bayesian."),
):
    run_summarize(
        inpath = inpath,
        outpath = outpath,
        wddate_fast = wddate_fast,
        wddate_bayesian = wddate_bayesian
    )
    typer.echo(f"summarize done")

@app.command("postprocess",
             help = """
Measure ages or hydrogen layer masses with monte carlo post-processing.

This script uses MIST isochrones to measure parameters for white dwarfs. It uses Monte Carlo sampling 
on the MCMC posteriors to measure a log age posterior. It's surprisingly flexible with the [Fe/H] 
prior: it supports fitting with fixed [Fe/H] as well as uninformed and informed priors. It can measure 
anything in the MIST cooling tracks (most notably `log_age` and `log_M_H`). 

The `--feh1` and `--feh2` options work like this:

* If one of these is provided as a string, you must also supply the path to a parquet file which has a 
column of the same name as that string. The script will read the `fe_h` or `e_fe_h` arrays from those 
files.
* If one is provided as a float, the script will use that float value for every chain in the path. For 
example, passing `--feh1 -0.5 --feh2 0.1` will sample values from a Gaussian with these parameters 
(N(0.5, 0.1)) for every chain in the sample, unless the `--uniform` flag is passed in which case the 
distribution will be U(0.5, 0.1) (which doesn't really make sense, but you get the point).
* If `--feh2` is provided a float value which is less than or equal to zero, the provided values of 
`--feh1` will be fixed to that value. For example, passing `--feh1 -0.5 --feh2 -999.0` will not do Monte 
Carlo sampling with [Fe/H], and will instead fix the value of [Fe/H] to -0.5. 

In addition to this,
* If a column name is passed, the parquet must also have a `wd_source_id` column to match the chain to 
the [Fe/H] values. If any of the column values are NaNs, the program will instead sample uniformally 
from the distribution U(-1.0, 0.5).
* Passing the flag `--uniform` will treat `--feh1` and `--feh2` as respectively the lower and upper 
bounds of a uniform distribution (i.e. an uninformative prior). 

Example:
  ember postprocess CHAINDIR OUTDIR \\
      --pqtpath ../data/catalogs/APOGEE/wdms_xmatch_ASTRA_xmatch_vincent24.pqt \\
      --feh1 fe_h --feh2 e_fe_h --logger
""")
def postprocess(
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
    numtasks: int = typer.Option(1, "--num-tasks", "-n", help="Number of CPU chunks (SGE array-style)."),
):
    run_postprocess(
        chainpath = chainpath,
        outpath = outpath,
        logger = logger,
        pqtpath = pqtpath,
        feh1 = feh1,
        feh2 = feh2,
        target = target,
        progress = progress,
        uniform = uniform,
        numtasks = numtasks,
    )
    typer.echo(f"summarize done")

if __name__ == "__main__":
    app()
