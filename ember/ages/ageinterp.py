from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay
import pandas as pd
import numpy as np

import os, sys, glob, re
coolingdir = os.environ['COOLING_PATH']

def parse_feh(path : str) -> float:
    """Extracts feh_xxxx and converts to float.
    """
    fname = os.path.basename(path)
    m = re.search(r"feh_([mp])(\d+)", fname)
    if not m:
        return None
    sign = -1 if m.group(1) == "m" else 1
    value = int(m.group(2)) / 100
    return sign * value

def read_tracks(save : bool = False) -> pd.DataFrame:
    """read all of the tracks and generate a summary file that can be interpolated
    """
    metalfolders = glob.glob(os.path.join(coolingdir, "*"))
    datafiles = []
    for ii, metal in enumerate(metalfolders):
        coolfile = os.path.join(metal, f"{os.path.basename(metal)}.wdcool")
        data = pd.DataFrame(np.genfromtxt(coolfile, names=True))
        data['fe_h'] = parse_feh(metal)*np.ones(len(data))
        data['teff'] = 10**data['log_Teff']
        data['radius'] = 10**data['log_R']
        datafiles.append(data.drop(labels=["log_Teff", "log_R"], axis=1))
    datafile = pd.concat(datafiles).reset_index(drop=True)
    if save is not None:
        datafile.to_parquet(os.path.join(coolingdir, "summary.pqt"))
    return datafile

def make_interpolator(datafile: pd.DataFrame, fe_h=None, outcol="log_age"):
    """
    Build an interpolator that accepts scalars or arrays for teff, radius, fe_h.
    - If fe_h is None: 3D interpolation over (teff, radius, fe_h).
    - If fe_h is a number: 2D interpolation over (teff, radius) on that metallicity slice.
    Returns a function predict_log_age(teff, radius, fe_h=None|array) -> np.ndarray
    """

    def _as_array(x):
        # Ensure numpy array, allow scalars/lists
        return np.asarray(x, dtype=np.float64)

    if fe_h is not None:
        # ---------- 2D case: fixed metallicity ----------
        df = datafile.query("fe_h == @fe_h").copy()
        if len(df) == 0:
            raise ValueError(f"No rows for fe_h={fe_h}")

        pts = df[["teff", "radius"]].to_numpy(np.float64)
        vals = df[outcol].to_numpy(np.float64)

        # light scaling improves conditioning
        s_teff, s_radius = 1e-3, 1e2
        P = np.c_[pts[:, 0] * s_teff, pts[:, 1] * s_radius]

        # drop duplicate points
        Puniq, idx = np.unique(P, axis=0, return_index=True)
        V = vals[idx]

        lin = LinearNDInterpolator(Puniq, V, fill_value=np.nan)
        near = NearestNDInterpolator(Puniq, V)

        def predict_log_age(teff, radius, feh=None):
            teff = _as_array(teff)
            radius = _as_array(radius)
            # broadcast to common shape
            teff, radius = np.broadcast_arrays(teff, radius)
            X = np.column_stack([teff.ravel() * s_teff, radius.ravel() * s_radius])
            y = lin(X)
            # vectorized fallback for NaNs
            mask = np.isnan(y)
            if np.any(mask):
                y[mask] = near(X[mask])
            return y.reshape(teff.shape)

        return predict_log_age

    # ---------- 3D case: multiple metallicities ----------
    df = datafile.copy()
    if len(df) == 0:
        raise ValueError("Empty datafile")

    pts_raw = df[["teff", "radius", "fe_h"]].to_numpy(np.float64)
    vals = df[outcol].to_numpy(np.float64)

    s_teff, s_radius = 1e-3, 1e2
    P = pts_raw.copy()
    P[:, 0] *= s_teff
    P[:, 1] *= s_radius

    # drop duplicate points
    Puniq, idx = np.unique(P, axis=0, return_index=True)
    V = vals[idx]

    # robust Qhull options help with large/near-coplanar sets
    tri = Delaunay(Puniq, qhull_options="QJ Qbb Q12")
    lin = LinearNDInterpolator(tri, V, fill_value=np.nan)
    near = NearestNDInterpolator(Puniq, V)

    def predict_log_age(teff, radius, feh):
        teff = _as_array(teff)
        radius = _as_array(radius)
        feh = _as_array(feh)

        # broadcast all three to a common shape
        teff, radius, feh = np.broadcast_arrays(teff, radius, feh)
        X = np.column_stack([
            teff.ravel() * s_teff,
            radius.ravel() * s_radius,
            feh.ravel(),
        ])

        y = lin(X)
        mask = np.isnan(y)
        if np.any(mask):
            y[mask] = near(X[mask])
        return y.reshape(teff.shape)

    return predict_log_age


def call_interp(fe_h = None, outcol = "log_age"):
    """call interpolator. designed to be used as the entry point
    for external files.
    """
    if not os.path.isfile(os.path.join(coolingdir, "summary.pqt")):
        datafile = read_tracks(save = True)
    else:
        datafile = pd.read_parquet(os.path.join(coolingdir, "summary.pqt"))
    return make_interpolator(datafile, fe_h=fe_h, outcol = outcol)

    
