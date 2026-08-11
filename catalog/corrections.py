"""White dwarf parameter corrections applied during catalog construction."""

import numpy as np

from .config import DATA_DIR

def correct_cool_wd_mass(mass, teff):
    """Apply O'Brien+2024 eq. 1–2 mass correction for cool WDs (Teff < 6000 K).

    Mc = Mi + M_med - ΔM(Teff,i)

    where ΔM is the 5th-order polynomial fit to the running median mass of
    Gaia-photometric WD masses below 6000 K (equation 1 of arXiv:2312.02735),
    and M_med is the median mass of uncorrected WDs with Teff >= 6000 K in
    this catalog.

    Parameters
    ----------
    mass : array-like, shape (N,)
        WD masses in solar masses (all temperatures).
    teff : array-like, shape (N,)
        WD effective temperatures in K (all temperatures).

    Returns
    -------
    mass_cor : ndarray
        Corrected masses; unchanged where Teff >= 6000 K.
    mass_uncor : ndarray
        Original masses for corrected stars (NaN where Teff >= 6000 K).
    """
    mass = np.asarray(mass, dtype=float)
    teff = np.asarray(teff, dtype=float)

    stable_mask = (17_000 > teff) & (teff > 7_000)
    m_med = np.nanmedian(mass[stable_mask])

    mask  = teff < 6000
    t       = teff[mask]

    yy = mass[mask]# - m_med
    not_na  = ~np.isnan(t) & ~np.isnan(yy)
    fit = np.polyfit(t[not_na], yy[not_na], 5)
    polynomial_delta_m = np.poly1d(fit)
    delta_m = polynomial_delta_m(t)

    out_coefficients = DATA_DIR / "build/correction_coefficients.npy"
    np.save(out_coefficients, np.append([m_med], fit))

    terms = []
    for ii, coef in enumerate(fit):
        sign = "+" if coef >= 0 else "-"
        terms.append(f"{sign} {abs(coef)} T^{len(fit) - ii - 1}")
    poly = " ".join(terms)

    if poly.startswith("+ "):
        poly = poly[2:]
    print(poly)
    
    #delta_m = (-4.613e-19*t**5 + 1.726e-14*t**4 - 2.486e-10*t**3
    #           + 1.706e-6*t**2 - 0.005487*t + 7.068)

    mass_cor           = mass.copy()
    mass_cor[mask]     = mass[mask] + m_med - delta_m

    mass_uncor         = np.full(len(mass), np.nan)
    mass_uncor[mask]   = mass[mask]

    teff_cor           = teff.copy()
    teff_cor[mask]     = teff[mask] * (mass_cor[mask] / mass_uncor[mask]) ** (1/6)

    teff_uncor         = np.full(len(teff), np.nan)
    teff_uncor[mask]   = teff[mask]

    return mass_cor, mass_uncor, teff_cor, teff_uncor
