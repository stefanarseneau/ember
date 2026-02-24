import numpy as np
import pandas as pd
import os

import argparse
from ember.ages import chainhandler  # noqa: E402
from ember.ages import ageinterp
from tqdm import tqdm
tqdm.pandas()

import multiprocessing as mp

# ---- per-process globals ----
_G_INTERP_TOT = None
_G_INTERP_AGE = None
_G_RAD_FUNC = None

import sys, os, tqdm, glob
wdmodels_dir = os.environ['WDMODELS_DIR']
sys.path.append(wdmodels_dir)
import WD_models

#def radius_from_loggteff(loggarray : np.array, teffarray : np.array, 
#                       low_model = 'be', mid_model = 'be', high_model = 'be',
#                       atm_type = 'H') -> np.array:
#    """compute the radial velocity from radius and effective temperature
#    """
#    mass_sun, radius_sun, newton_G, speed_light = 1.9884e30, 6.957e8, 6.674e-11, 299792458
#    font_model = WD_models.load_model(low_model, mid_model, high_model, atm_type) 
#    g_acc = (10**font_model['logg'])/100
#    rsun = np.sqrt(font_model['mass_array'] * mass_sun * newton_G / g_acc) / radius_sun
#    rad_teff_to_mass = WD_models.interp_xy_z_func(x = font_model['logg'], y = 10**font_model['logteff'],\
#                                               z = rsun, interp_type = 'linear')
#    return rad_teff_to_mass(loggarray, teffarray)

#def read_ages(row : pd.Series, chaindir : str):
#    chainpath = os.path.join(chaindir, f"{row.sourceid}.npy")
#    chain = np.load(chainpath)
#    return np.mean(chain[:,-1]), np.std(chain[:,-1])

#def measure_ages(samps, fehs, interp_tot, interp_age):
#    rng = np.random.default_rng(1000)
#    
#    try:
#        agetot = chainhandler.interp_chain(samps, np.squeeze(fehs), interp_tot)
#        agecool = chainhandler.interp_chain(samps, np.squeeze(fehs), interp_age)
#        return agetot[~np.isnan(agetot)], agecool[~np.isnan(agecool)]
#    except:
#        raise

def _init_worker():
    """Runs once per process: build interpolators + radius interpolator in the worker."""
    global _G_INTERP_TOT, _G_INTERP_AGE, _G_RAD_FUNC

    from ember.ages import ageinterp
    _G_INTERP_TOT = ageinterp.call_interp(fe_h=0, outcol="log_tot_age")
    _G_INTERP_AGE = ageinterp.call_interp(fe_h=0, outcol="log_age")

    # Build the radius-from-(logg,teff) interpolator ONCE per process (huge speedup)
    import os, sys
    wdmodels_dir = os.environ["WDMODELS_DIR"]
    if wdmodels_dir not in sys.path:
        sys.path.append(wdmodels_dir)
    import WD_models

    mass_sun, radius_sun, newton_G = 1.9884e30, 6.957e8, 6.674e-11

    font_model = WD_models.load_model("be", "be", "be", "H")
    g_acc = (10**font_model["logg"]) / 100.0
    rsun = np.sqrt(font_model["mass_array"] * mass_sun * newton_G / g_acc) / radius_sun

    _G_RAD_FUNC = WD_models.interp_xy_z_func(
        x=font_model["logg"],
        y=10**font_model["logteff"],
        z=rsun,
        interp_type="linear",
    )

def _compute_one(args):
    """
    Return (ii, lac, lac_hi, lac_lo, lat, lat_hi, lat_lo) or (ii, None,...)
    """
    ii, teff, e_teff, logg, e_logg, filled_row = args

    if np.all(np.isfinite(filled_row)):
        return (ii, None, None, None, None, None, None)

    # deterministic per-row RNG (so parallel == reproducible)
    rng = np.random.default_rng(12345 + ii)

    samps_teff = rng.normal(teff, e_teff, size=10000)
    samps_logg = rng.normal(logg, e_logg, size=10000)

    samps_radii = _G_RAD_FUNC(samps_logg, samps_teff)
    samps = np.column_stack([samps_teff, samps_radii])

    fehs = rng.uniform(-0.2, 0.1, size=10000)
    mask = np.all(np.isfinite(samps), axis=1)

    if mask.sum() < 100:
        return (ii, None, None, None, None, None, None)

    from ember.ages import chainhandler
    agetot = chainhandler.interp_chain(samps[mask], fehs[mask], _G_INTERP_TOT)
    agecool = chainhandler.interp_chain(samps[mask], fehs[mask], _G_INTERP_AGE)

    if not (isinstance(agetot, np.ndarray) and isinstance(agecool, np.ndarray)):
        return (ii, None, None, None, None, None, None)
    if agetot.size == 0 or agecool.size == 0:
        return (ii, None, None, None, None, None, None)

    return (
        ii,
        np.percentile(agecool, 50),
        np.percentile(agecool, 84),
        np.percentile(agecool, 16),
        np.percentile(agetot, 50),
        np.percentile(agetot, 84),
        np.percentile(agetot, 16),
    )

def parallel_forloop(outdata, outpath, nproc=None, chunksize=50):
    """
    Parallelizes the loop. Parent process applies results to outdata + checkpoints parquet.
    """
    cols = ["log_age_cool","log_age_cool_hi","log_age_cool_lo","log_age","log_age_hi","log_age_lo"]

    # extract only what workers need (fast + avoids pickling the DataFrame)
    teff  = outdata["teff"].to_numpy(dtype=float)
    e_teff = outdata["e_teff"].to_numpy(dtype=float)
    logg  = outdata["logg"].to_numpy(dtype=float)
    e_logg = outdata["e_logg"].to_numpy(dtype=float)
    filled = outdata[cols].to_numpy(dtype=float)

    n = len(outdata)
    if nproc is None:
        nproc = max(1, mp.cpu_count() - 1)

    ctx = mp.get_context("fork")  # Linux: avoids spawn/pickle headaches
    with ctx.Pool(processes=nproc, initializer=_init_worker) as pool:
        # imap_unordered: better load balancing
        it = pool.starmap_async  # not iterable; so use imap_unordered with an args generator

        def arggen():
            for ii in range(n):
                yield (ii, teff[ii], e_teff[ii], logg[ii], e_logg[ii], filled[ii])

        for k, res in enumerate(tqdm.tqdm(pool.imap_unordered(_compute_one, arggen(), chunksize=chunksize), total=n)):
            ii, lac, lac_hi, lac_lo, lat, lat_hi, lat_lo = res
            if lac is None:
                continue

            idx = outdata.index[ii]  # robust even if index not 0..N-1
            outdata.loc[idx, "log_age_cool"] = lac
            outdata.loc[idx, "log_age_cool_hi"] = lac_hi
            outdata.loc[idx, "log_age_cool_lo"] = lac_lo
            outdata.loc[idx, "log_age"] = lat
            outdata.loc[idx, "log_age_hi"] = lat_hi
            outdata.loc[idx, "log_age_lo"] = lat_lo

            # checkpoint occasionally (based on results applied, not ii)
            if (k + 1) % 1000 == 0:
                outdata.to_parquet(outpath)

    return outdata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure the ages with monte carlo",
                                        epilog="Example:\n"",",
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inpath', type=str, default = 'targets.pqt', help='Path to input pqt file')
    parser.add_argument('outpath', type=str, default = 'targets_out.pqt', help='Path to output pqt file')
    args = parser.parse_args()

    data = pd.read_parquet(args.inpath).dropna()
    teffs = data["teff"].to_numpy() ; e_teffs = data["e_teff"].to_numpy()
    loggs = data["logg"].to_numpy() ; e_loggs = data["e_logg"].to_numpy()
    covars = data["covar"].to_numpy()
    
    #samps_teff = np.random.normal(teffs[:,None], e_teffs[:,None], size=(teffs.shape[0], 1000))
    #samps_logg = np.random.normal(loggs[:,None], e_loggs[:,None], size=(loggs.shape[0], 1000))
    #samps_radii = radius_from_loggteff(samps_logg, samps_teff)
    #radii = np.nanmean(samps_radii, axis=1) ; e_radii = np.nanstd(samps_radii, axis=1)
    
    #samps = np.transpose(np.stack([samps_teff, samps_radii]), [1,2,0])
    #fehs = np.random.uniform(-0.2, 0.1, size=(len(data), 1000))
    #masks = np.all(~np.isnan(samps), axis=2)
    
    interp_tot = ageinterp.call_interp(fe_h = None, outcol = "log_tot_age")
    interp_age = ageinterp.call_interp(fe_h = None, outcol = "log_age")
    print(type(interp_tot))

    try:
        outdata = pd.read_parquet(args.outpath)
    except FileNotFoundError:
        outdata = data.copy()

        outdata["log_age_cool"] = np.nan       ;   outdata["log_age"] = np.nan
        outdata["log_age_cool_hi"] = np.nan    ;   outdata["log_age_hi"] = np.nan
        outdata["log_age_cool_lo"] = np.nan    ;   outdata["log_age_lo"] = np.nan

    #for ii, row in tqdm.tqdm(outdata.iterrows(), total=len(outdata)):
    #    if np.all(~np.isnan(row[["log_age_cool", "log_age_cool_hi", "log_age_cool_lo",
    #                             "log_age", "log_age_hi", "log_age_lo"]].values)):
    #        continue
    #
    #    mask = masks[ii]
    #    if mask.sum() < 100:
    #        continue
    #
    #    agetot, agecool = measure_ages(samps[ii, mask], fehs[ii, mask], interp_tot, interp_age)
    #    if isinstance(agetot, np.ndarray) and isinstance(agecool, np.ndarray):
    #        outdata.loc[ii, "log_age_cool"] = np.percentile(agecool, 50)
    #        outdata.loc[ii, "log_age_cool_hi"] = np.percentile(agecool, 84)
    #        outdata.loc[ii, "log_age_cool_lo"] = np.percentile(agecool, 16)
    #        outdata.loc[ii, "log_age"] = np.percentile(agetot, 50)
    #        outdata.loc[ii, "log_age_hi"] = np.percentile(agetot, 84)
    #        outdata.loc[ii, "log_age_lo"] = np.percentile(agetot, 16) 
    #
    #    if ii % 1000 == 0:
    #        outdata.to_parquet(args.outpath)

    outdata = parallel_forloop(outdata, args.outpath)
    outdata.to_parquet(args.outpath)
