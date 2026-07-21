import pandas as pd
import numpy as np

flag_conditions = {
    'Gaia':      lambda flag: 0,
    'SDSS':      lambda flag: 1 if (flag & (2**2 | 2**5 | 2**18 | 2**19) != 0) else 0,
    'PANSTARRS': lambda flag: 1 if (flag & (2**10 | 2**11 | 2**12) != 0) else 0,
    'SKYMAPPER': lambda flag, nima: 1 if (flag > 0) or (nima > 0) else 0,
}


def _make_photometry_bitmask(df: pd.DataFrame) -> pd.Series:
    """Bitmask of unreliable (1) or OK (0) per-band photometry, bits 0–15.

    SDSS u/g/r/i/z   : bits 0–4   (bad quality flags OR mag == 0)
    PanSTARRS g/r/i/z/y: bits 5–9  (bad quality flags OR mag == 0)
    SkyMapper u/v/g/r/i/z: bits 10–15 (bad flag/nima OR mag == 0)
    """
    mask = np.zeros(len(df), dtype=np.int32)

    sdss_bad_bits = 2**2 | 2**5 | 2**18 | 2**19
    for bit, flag_col, mag_col in [
        (0, "flags_u", "u"), (1, "flags_g", "g"), (2, "flags_r", "r"),
        (3, "flags_i", "i"), (4, "flags_z", "z"),
    ]:
        flags = df[flag_col].fillna(0).to_numpy(dtype=np.int64)
        mags  = df[mag_col].fillna(0).to_numpy()
        mask |= (((flags & sdss_bad_bits) != 0) | (mags == 0)).astype(np.int32) << bit

    ps_bad_bits = 2**10 | 2**11 | 2**12
    for bit, flag_col, mag_col in [
        (5, "g_flags_", "g_mean_psf_mag"), (6, "r_flags_", "r_mean_psf_mag"),
        (7, "i_flags_", "i_mean_psf_mag"), (8, "z_flags_", "z_mean_psf_mag"),
        (9, "y_flags",  "y_mean_psf_mag"),
    ]:
        flags = df[flag_col].fillna(0).to_numpy(dtype=np.int64)
        mags  = df[mag_col].fillna(0).to_numpy()
        mask |= (((flags & ps_bad_bits) != 0) | (mags == 0)).astype(np.int32) << bit

    smap_bands = [
        (10, "u_flags",      "u_nimaflags", "u_psf"),
        (11, "v_flags",      "v_nimaflags", "v_psf"),
        (12, "g_flags_smap", "g_nimaflags", "g_psf"),
        (13, "r_flags_smap", "r_nimaflags", "r_psf"),
        (14, "i_flags_smap", "i_nimaflags", "i_psf"),
        (15, "z_flags_smap", "z_nimaflags", "z_psf"),
    ]
    for bit, flag_col, nima_col, mag_col in smap_bands:
        flags = df[flag_col].fillna(0).to_numpy()
        nimas = df[nima_col].fillna(0).to_numpy()
        mags  = df[mag_col].fillna(0).to_numpy()
        mask |= (((flags > 0) | (nimas > 0)) | (mags == 0)).astype(np.int32) << bit

    return pd.Series(mask, index=df.index, dtype=np.int32)