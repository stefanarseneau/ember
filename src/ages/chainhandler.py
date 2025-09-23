import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm

import os, sys, glob, re
agedir = os.environ['COOLING_PATH']
data_dir = os.path.join(agedir, "data")

from . import ageinterp

def check_input(val : float | np.typing.ArrayLike, N : int):
    arr = np.asarray(val, dtype=float)
    if arr.ndim == 0:
        arr = np.full(N, arr, dtype=float)
    elif arr.shape != (N,):
        raise ValueError(f"`feh` must be scalar or length {N}; got shape {arr.shape}")
    return arr

def interp_chain(chain : np.ndarray, feh : float | np.typing.ArrayLike, interpolator : callable) -> np.array:
    """measure an age from a chain. assumes that the 0 and 1 indicies
    are teff and radius.
    """
    N = chain.shape[0]
    feh = check_input(feh, N)
    return interpolator(chain[:,0], chain[:,1], feh)

def measure_chains(
        chains : list[str], 
        savepath : str,
        outcol : str = "log_age",
        feh1 :  float | np.typing.ArrayLike = 0,
        feh2 : float | np.typing.ArrayLike = 0,
        uniform : bool | np.typing.ArrayLike = False,
        progress : bool = True      
    ):
    """measure a parameter from a chain directory. if e_feh is less than
    or equal to zero, it interprets it as a fixed value and does not treat
    it via monte carlo.
    """
    interp = ageinterp.call_interp(fe_h = None, outcol = outcol)
    outputs = {} ; failed_files = []

    feh1 = check_input(feh1, len(chains))
    feh2 = check_input(feh2, len(chains))
    uniform = check_input(uniform, len(chains))

    logger = pd.DataFrame(columns=['sourceid', 'feh1', 'feh2', 'uniform', 'mean_val', 'std_val'])
    for ii, chainfile in tqdm.tqdm(enumerate(chains), total = len(chains), disable=not progress):
        try:
            chain = np.load(chainfile)

            this_feh1, this_feh2, this_uniform = feh1[ii], feh2[ii], uniform[ii]
            if (this_feh2 > 0) and (not this_uniform):
                feharr = np.random.normal(loc=this_feh1, scale=this_feh2, size=(chain.shape[0],))
            elif not this_uniform:
                feharr = np.full(chain.shape[0], np.asarray(this_feh1))
            else:
                feharr = np.random.uniform(this_feh1, this_feh2, size=(chain.shape[0],))

            val = interp_chain(chain, feharr, interp)
            name = os.path.basename(chainfile).split('.')[0]
            np.save(os.path.join(savepath, f"{name}.npy"), np.concatenate([chain, val[:,None]], axis=1))
            logger.loc[len(logger)] = [name, this_feh1, this_feh2, this_uniform, np.mean(val), np.std(val)]
        except ValueError as e:
            print(f"{e} || failed to read index {ii}: {chainfile}")
            failed_files.append(chainfile)
    return outputs, logger, failed_files

if __name__ == "__main__":
    measure_chains("../data/wdms_widebinary_chains", outcol = "log_age", e_feh = 0.1)