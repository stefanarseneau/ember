import numpy as np
import pandas as pd
import os

import argparse
from ember.ages import chainhandler  # noqa: E402
from ember.ages import ageinterp
from tqdm import tqdm
tqdm.pandas()

import multiprocessing as mp
from scipy.interpolate import interp1d as _scipy_interp1d

# ---- per-process globals ----
_G_INTERP_TOT = None
_G_INTERP_AGE = None
_G_RAD_FUNC = None
_G_MASS_FUNC = None
_G_HEINTZ_COOL = None
_G_HEINTZ_IFMR = None
_G_HEINTZ_TMS = None

import sys, os, tqdm, glob
wdmodels_dir = os.environ['WDMODELS_DIR']
sys.path.append(wdmodels_dir)
import WD_models

mass_sun, radius_sun, newton_G = 1.9884e30, 6.957e8, 6.674e-11

_HEINTZ_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '..', '..', 'code_for_JJ', 'code_for_JJ', 'code_for_JJ')
)

# Maps population name → (WD_models model string, atmosphere type)
_POP_MODEL_MAP = {
    'thick': ('Bedard2020',      'H'),
    'thin':  ('Bedard2020',      'He'),
    'mixed': ('Bedard2020_thin', 'H'),
}


def _load_heintz_interps(pop_type):
    """Build cooling-age, IFMR, and MS-lifetime interpolators for high-mass WDs.

    Cooling ages use Bedard 2020 WD_models cooling tracks (model choice depends
    on population).  IFMR and MS lifetimes come from data files in
    code_for_JJ (Heintz et al. 2024).
    """
    bedard_model, atm_type = _POP_MODEL_MAP[pop_type]

    model = WD_models.load_model(bedard_model, bedard_model, bedard_model, atm_type)
    g_acc    = (10 ** model["logg"]) / 100.0      # cm s⁻² → m s⁻²
    rsun_arr = np.sqrt(model["mass_array"] * mass_sun * newton_G / g_acc) / radius_sun

    # (Teff [K], radius [R_sun]) → cooling age [Gyr]
    cool_func = WD_models.interp_xy_z_func(
        x=10 ** model["logteff"], y=rsun_arr, z=model["age_cool"],
        interp_type="linear",
    )

    # IFMR: final WD mass → initial (progenitor) mass
    # The offset a=0.06367 is taken directly from SEDs_WDMS_v5.py (Heintz+2024);
    # its origin is not documented there — verify before publication.
    ifmr_data = pd.read_csv(
        os.path.join(_HEINTZ_DIR, 'MESA_IFMR', 'MESA_IFMR_missing_one_point.csv')
    )
    a = 0.06367
    IFMR_func = _scipy_interp1d(
        ifmr_data['M_final'].to_numpy() + a,
        ifmr_data['M_initial'].to_numpy(),
        kind='linear', fill_value=np.nan, bounds_error=False,
    )

    # MS lifetime: initial mass [M_sun] → MS lifetime [Gyr]
    mi  = np.load(os.path.join(_HEINTZ_DIR, 'init_mass_to_mslife', 'mi.npy'))
    msl = np.load(os.path.join(_HEINTZ_DIR, 'init_mass_to_mslife', 'msl.npy'))
    t_ms_func = _scipy_interp1d(mi, msl, fill_value='extrapolate', bounds_error=False)

    return cool_func, IFMR_func, t_ms_func


def _init_worker(pop_type='thick'):
    """Runs once per process: build interpolators + radius interpolator in the worker."""
    global _G_INTERP_TOT, _G_INTERP_AGE, _G_RAD_FUNC, _G_LOGG_FUNC, _G_MASS_FUNC
    global _G_HEINTZ_COOL, _G_HEINTZ_IFMR, _G_HEINTZ_TMS

    from ember.ages import ageinterp
    _G_INTERP_TOT = ageinterp.call_interp(fe_h=0, outcol="log_tot_age")
    _G_INTERP_AGE = ageinterp.call_interp(fe_h=0, outcol="log_age")

    # Build the radius-from-(logg,teff) interpolator ONCE per process (huge speedup)
    import os, sys
    wdmodels_dir = os.environ["WDMODELS_DIR"]
    if wdmodels_dir not in sys.path:
        sys.path.append(wdmodels_dir)
    import WD_models


    font_model = WD_models.load_model("bet", "be", "ONe", "H")
    g_acc = (10**font_model["logg"]) / 100.0
    rsun = np.sqrt(font_model["mass_array"] * mass_sun * newton_G / g_acc) / radius_sun

    _G_RAD_FUNC = WD_models.interp_xy_z_func(
        x=font_model["logg"],
        y=10**font_model["logteff"],
        z=rsun,
        interp_type="linear",
    )

    _G_LOGG_FUNC = WD_models.interp_xy_z_func(
        x=rsun,
        y=10**font_model["logteff"],
        z=font_model["logg"],
        interp_type="linear",
    )

    # (radius [R_sun], teff [K]) → mass [M_sun], using the same population model
    bedard_model, atm_type = _POP_MODEL_MAP[pop_type]
    mass_model = WD_models.load_model(bedard_model, bedard_model, bedard_model, atm_type)
    g_acc_mass = (10 ** mass_model["logg"]) / 100.0
    rsun_mass  = np.sqrt(mass_model["mass_array"] * mass_sun * newton_G / g_acc_mass) / radius_sun
    _G_MASS_FUNC = WD_models.interp_xy_z_func(
        x=rsun_mass, y=10 ** mass_model["logteff"],
        z=mass_model["mass_array"],
        interp_type="linear",
    )

    _G_HEINTZ_COOL, _G_HEINTZ_IFMR, _G_HEINTZ_TMS = _load_heintz_interps(pop_type)

def _compute_one(args):
    """
    Return (ii, lac, lac_hi, lac_lo, lat, lat_hi, lat_lo, mass, e_mass) or (ii, None,...)
    """
    ii, teff, e_teff, radius, e_radius, cov_rt, filled_row = args

    # deterministic per-row RNG (so parallel == reproducible)
    rng = np.random.default_rng(12345 + ii)

    samps_teff  = rng.normal(teff,   e_teff,   size=10000)
    samps_radii = rng.normal(radius, e_radius, size=10000)
    samps = np.column_stack([samps_teff, samps_radii])

    fehs = rng.uniform(-0.2, 0.1, size=10000)
    mask = np.all(np.isfinite(samps), axis=1)

    # Mass MC: samps[:, 0] = teff, samps[:, 1] = radius
    samps_mass = _G_MASS_FUNC(samps[mask, 1], samps[mask, 0])
    valid_mass = np.isfinite(samps_mass)
    if valid_mass.sum() >= 100:
        ms = samps_mass[valid_mass]
        mass_val    = float(np.percentile(ms, 50))
        mass_hi_val = float(np.percentile(ms, 84))
        mass_lo_val = float(np.percentile(ms, 16))
    else:
        mass_val = mass_hi_val = mass_lo_val = np.nan

    if np.all(np.isfinite(filled_row)):
        return (ii, None, None, None, None, None, None, mass_val, mass_hi_val, mass_lo_val)

    if mask.sum() < 100:
        return (ii, None, None, None, None, None, None, mass_val, mass_hi_val, mass_lo_val)

    if (mass_val > 0.512609) and (mass_val < 1.017626):
        from ember.ages import chainhandler
        agecool = chainhandler.interp_chain(samps[mask], fehs[mask], _G_INTERP_AGE)
        agecool = agecool[~np.isnan(agecool)]
        if mass_val > 0.63:
            agetot = chainhandler.interp_chain(samps[mask], fehs[mask], _G_INTERP_TOT)
            agetot = agetot[~np.isnan(agetot)]
        else:
            agetot = np.nan*np.zeros_like(agecool)

    elif (mass_val <= 0.512609):
        # get cooling ages from Bedard 2020 (population-appropriate model)
        cool_gyr = _G_HEINTZ_COOL(samps[mask, 0], samps[mask, 1])
        valid = np.isfinite(cool_gyr) & (cool_gyr > 0)
        agecool = np.log10(cool_gyr[valid]) + 9  # log10(yr)
        agetot = np.nan * np.zeros_like(agecool)

    elif (mass_val >= 1.017626):
        # Cooling age from Bedard 2020 via Heintz+2024 method
        cool_gyr = _G_HEINTZ_COOL(samps[mask, 0], samps[mask, 1])
        valid = np.isfinite(cool_gyr) & (cool_gyr > 0)
        cool_gyr = cool_gyr[valid]
        agecool = np.log10(cool_gyr) + 9  # log10(yr)

        # Total age: cooling + MS lifetime via IFMR
        init_mass = float(_G_HEINTZ_IFMR(mass_val))
        if np.isfinite(init_mass):
            ms_life_gyr = float(_G_HEINTZ_TMS(init_mass))
            tot_gyr = np.maximum(ms_life_gyr + cool_gyr, 1e-6)
            agetot = np.log10(tot_gyr) + 9  # log10(yr)
        else:
            agetot = np.nan * np.zeros_like(agecool)

    else:
        return (ii, None, None, None, None, None, None, mass_val, mass_hi_val, mass_lo_val)

    if (not (isinstance(agetot, np.ndarray) and isinstance(agecool, np.ndarray))) or (agetot.size < 100 or agecool.size < 100):
        return (ii, None, None, None, None, None, None, mass_val, mass_hi_val, mass_lo_val)

    return (
        ii,
        np.percentile(agecool, 50),
        np.percentile(agecool, 84),
        np.percentile(agecool, 16),
        np.percentile(agetot, 50),
        np.percentile(agetot, 84),
        np.percentile(agetot, 16),
        mass_val,
        mass_hi_val,
        mass_lo_val,
    )

def parallel_forloop(outdata, outpath, pop_type='thick', nproc=None, chunksize=50):
    """
    Parallelizes the loop. Parent process applies results to outdata + checkpoints parquet.
    pop_type: 'thick', 'thin', or 'mixed' — selects the Bedard WD model for high-mass WDs.
    """
    cols = ["log_age_cool","log_age_cool_hi","log_age_cool_lo","log_age","log_age_hi","log_age_lo"]

    # extract only what workers need (fast + avoids pickling the DataFrame)
    teff     = outdata["teff_best"].to_numpy(dtype=float)
    e_teff   = outdata["std_tt_best"].to_numpy(dtype=float)
    radius   = outdata["radius_best"].to_numpy(dtype=float)
    e_radius = outdata["std_rr_best"].to_numpy(dtype=float)
    cov_rt   = outdata["cov_rt_best"].to_numpy(dtype=float)
    filled   = outdata[cols].to_numpy(dtype=float)

    n = len(outdata)
    if nproc is None:
        nproc = max(1, mp.cpu_count() - 1)

    ctx = mp.get_context("fork")  # Linux: avoids spawn/pickle headaches
    with ctx.Pool(processes=nproc, initializer=_init_worker, initargs=(pop_type,)) as pool:
        # imap_unordered: better load balancing
        it = pool.starmap_async  # not iterable; so use imap_unordered with an args generator

        def arggen():
            for ii in range(n):
                yield (ii, teff[ii], e_teff[ii], radius[ii], e_radius[ii], cov_rt[ii], filled[ii])

        for k, res in enumerate(tqdm.tqdm(pool.imap_unordered(_compute_one, arggen(), chunksize=chunksize), total=n)):
            ii, lac, lac_hi, lac_lo, lat, lat_hi, lat_lo, mass, mass_hi, mass_lo = res

            idx = outdata.index[ii]  # robust even if index not 0..N-1
            if lac is not None:
                outdata.loc[idx, "log_age_cool"]    = lac
                outdata.loc[idx, "log_age_cool_hi"] = lac_hi
                outdata.loc[idx, "log_age_cool_lo"] = lac_lo
                outdata.loc[idx, "log_age"]         = lat
                outdata.loc[idx, "log_age_hi"]      = lat_hi
                outdata.loc[idx, "log_age_lo"]      = lat_lo
            outdata.loc[idx, "mass"]    = mass
            outdata.loc[idx, "mass_hi"] = mass_hi
            outdata.loc[idx, "mass_lo"] = mass_lo

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

    # Infer population type from input filename for Bedard model selection
    inbase = os.path.basename(args.inpath).lower()
    if 'mixed' in inbase:
        pop_type = 'mixed'
    elif 'thin' in inbase:
        pop_type = 'thin'
    else:
        pop_type = 'thick'

    data = pd.read_parquet(args.inpath)#.dropna()
    teffs = data["teff_best"].to_numpy() ; e_teffs = data["std_tt_best"].to_numpy()
    loggs = data["radius_best"].to_numpy() ; e_loggs = data["std_rr_best"].to_numpy()
    covars = data["cov_rt_best"].to_numpy()

    interp_tot = ageinterp.call_interp(fe_h = None, outcol = "log_tot_age")
    interp_age = ageinterp.call_interp(fe_h = None, outcol = "log_age")

    outdata = data.copy()
    outdata["log_age_cool"]    = np.nan  ;  outdata["log_age"]    = np.nan
    outdata["log_age_cool_hi"] = np.nan  ;  outdata["log_age_hi"] = np.nan
    outdata["log_age_cool_lo"] = np.nan  ;  outdata["log_age_lo"] = np.nan
    outdata["mass"]    = np.nan
    outdata["mass_hi"] = np.nan
    outdata["mass_lo"] = np.nan

    outdata = parallel_forloop(outdata, args.outpath, pop_type=pop_type)
    outdata.to_parquet(args.outpath)
