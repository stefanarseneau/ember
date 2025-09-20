import numpy as np
import pandas as pd
import scipy

import sys, os
wdmodels_dir = os.environ['WDMODELS_DIR']
sys.path.append(wdmodels_dir)
import WD_models
import interpolator

### Utility functions for the pipeline

def convert_names(bands = list[str]) -> list[str]:
    """convert between pyphot names and xp names"""
    xp_to_pyphot = {# Gaia conversions
                    'GaiaDr3Vega_G' : 'Gaia_G', 
                    'GaiaDr3Vega_BP' : 'Gaia_BP', 
                    'GaiaDr3Vega_RP' : 'Gaia_RP',
                    # SDSS conversions
                    'Sdss_u' : 'SDSS_u', 
                    'Sdss_g' : 'SDSS_g', 
                    'Sdss_r' : 'SDSS_r', 
                    'Sdss_i' : 'SDSS_i', 
                    'Sdss_z' : 'SDSS_z',
                    # PanSTARRS conversions
                    'Panstarrs1_gp' : 'PS1_g', 
                    'Panstarrs1_rp' : 'PS1_r', 
                    'Panstarrs1_ip' : 'PS1_i', 
                    'Panstarrs1_zp' : 'PS1_z', 
                    'Panstarrs1_yp' : 'PS1_y',
                    # J-PLUS conversions
                    'Jplus_uJAVA' : 'JPLUS_uJava', 
                    'Jplus_J0378' : 'JPLUS_J0378', 
                    'Jplus_J0395' : 'JPLUS_J0395', 
                    'Jplus_J0410' : 'JPLUS_J0410', 
                    'Jplus_J0430' : 'JPLUS_J0430', 
                    'Jplus_gJPLUS' : 'JPLUS_gSDSS', 
                    'Jplus_J0515' : 'JPLUS_J0515', 
                    'Jplus_rJPLUS' : 'JPLUS_rSDSS', 
                    'Jplus_J0660' : 'JPLUS_J0660', 
                    'Jplus_iJPLUS' : 'JPLUS_iSDSS', 
                    'Jplus_J0861' : 'JPLUS_J0861', 
                    'Jplus_zJPLUS' : 'JPLUS_zSDSS', 
                    }
    pyphot_to_xp = {val : key for key, val in xp_to_pyphot.items()}
    non_synth = [b for b in bands if not b.startswith("SYNTH_")]
    # perform the conversion
    if set(non_synth).issubset(set(xp_to_pyphot.keys())):
        return [xp_to_pyphot[band] if 'SYNTH_' not in band else band for band in bands]
    elif set(non_synth).issubset(set(pyphot_to_xp.keys())):
        return [pyphot_to_xp[band] if 'SYNTH_' not in band else band for band in bands]
    else:
        raise "Conversion error!"

def fetch_extinction(bands = list[str]) -> np.array:
    """return the extinction coefficients for each band in the dataset"""
    extinction_coeffs = {
                        # Gaia Dereddening
                        'GaiaDr3Vega_G' : 0.835, 
                        'GaiaDr3Vega_BP' : 1.139, 
                        'GaiaDr3Vega_RP' : 0.650,
                        # SDSS Dereddening
                        'Sdss_u' : 4.239, 
                        'Sdss_g' : 3.303, 
                        'Sdss_r' : 2.285, 
                        'Sdss_i' : 1.698, 
                        'Sdss_z' : 1.263,
                        # PanSTARRS Dereddening
                        'Panstarrs1_gp' : 3.172, 
                        'Panstarrs1_rp' : 2.271, 
                        'Panstarrs1_ip' : 1.682, 
                        'Panstarrs1_zp' : 1.322, 
                        'Panstarrs1_yp' : 1.087,
                        # J-PLUS Dereddening (https://iopscience.iop.org/article/10.3847/1538-4357/ad6b94/pdf)
                        'Jplus_uJAVA' : 4.479, 
                        'Jplus_J0378' : 4.294, 
                        'Jplus_J0395' : 4.226, 
                        'Jplus_J0410' : 4.023, 
                        'Jplus_J0430' : 3.859, 
                        'Jplus_gJPLUS' : 3.398, 
                        'Jplus_J0515' : 3.148, 
                        'Jplus_rJPLUS' : 2.383, 
                        'Jplus_J0660' : 2.161, 
                        'Jplus_iJPLUS' : 1.743, 
                        'Jplus_J0861' : 1.381, 
                        'Jplus_zJPLUS' : 1.289, 
                        }
    return np.array([extinction_coeffs[band] if 'SYNTH_' not in band else 1 for band in bands])

def check_valid(df : pd.DataFrame, use_gravz : bool = False) -> pd.DataFrame:
    """check that the dataframe passed is valid and strip out relevant parts"""
    needed_cols = ['gaia_dr3_source_id', 'ra', 'dec', 'parallax', 'parallax_error', 'meanAV']
    if use_gravz:
        needed_cols += ['gravz', 'gravz_error']
    assert all(col in df.columns for col in needed_cols), f"Missing one or more required columns: {tuple(needed_cols)}"
    df = df[needed_cols]
    return df

def extract_data(
        df : pd.DataFrame,
        source_id : str, 
        ra : str, 
        dec : str, 
        parallax : str, 
        parallax_error: str, 
        meanAV : str,
        gravz : str = None,
        gravz_error : str = None
    ) -> pd.DataFrame:
    columns = {source_id : 'gaia_dr3_source_id', ra : 'ra', dec : 'dec', parallax : 'parallax',
               parallax_error : 'parallax_error', meanAV : 'meanAV'}
    if (gravz is not None) and (gravz_error is not None):
        columns[gravz] = 'gravz'
        columns[gravz_error] = 'gravz_error'
    df = df[list(columns.keys())]
    df = df.rename(mapper=columns, axis=1)
    df['gaia_dr3_source_id'] = df['gaia_dr3_source_id'].astype(np.int64)
    return df

def find_photocols(df : pd.DataFrame):
    """find the columns which """
    # Keep only flux and flux_error columns
    flux_cols = [col for col in df.columns if '_flux_' in col]
    flux_err_cols = [col for col in df.columns if '_flux_error_' in col]
    # Match each flux column with its corresponding error column
    flux_dict = {}
    for flux in flux_cols:
        band = ('_').join(flux.split('_flux_'))
        error_col = flux.replace('_flux_', '_flux_error_')
        if error_col in flux_err_cols:
            flux_dict[band] = (flux, error_col)
    #if "GaiaDr3Vega_G" in flux_dict.keys():
    #    del flux_dict["GaiaDr3Vega_G"]
    return flux_dict

### Functions for interpolating mass-radius relation and photometry

def get_logg_function(low_model = 'f', mid_model = 'f', high_model = 'f', atm_type = 'H') -> scipy.interpolate:
    """compute the radial velocity from radius and effective temperature"""
    mass_sun, radius_sun, newton_G, speed_light = 1.9884e30, 6.957e8, 6.674e-11, 299792458
    font_model = WD_models.load_model(low_model, mid_model, high_model, atm_type)
    g_acc = (10**font_model['logg'])/100
    rsun = np.sqrt(font_model['mass_array'] * mass_sun * newton_G / g_acc) / radius_sun
    return WD_models.interp_xy_z_func(x = 10**font_model['logteff'], y = rsun,\
                                                z = font_model['logg'], interp_type = 'linear')

def get_actual_logg_function(low_model = 'f', mid_model = 'f', high_model = 'f', atm_type = 'H') -> scipy.interpolate:
    """compute the radial velocity from radius and effective temperature"""
    mass_sun, radius_sun, newton_G, speed_light = 1.9884e30, 6.957e8, 6.674e-11, 299792458
    font_model = WD_models.load_model(low_model, mid_model, high_model, atm_type)
    g_acc = (10**font_model['logg'])/100
    rsun = np.sqrt(font_model['mass_array'] * mass_sun * newton_G / g_acc) / radius_sun
    return WD_models.interp_xy_z_func(x = 10**font_model['logteff'], y = font_model['logg'],\
                                                z = rsun, interp_type = 'linear')

def make_interpolator(bands, units):
    """build the model SED using default filters"""
    defaults = interpolator.atmos.sed.get_default_filters({'zerofile' : 'alpha_lyr_mod_004'})
    sed = interpolator.atmos.WarwickPhotometry('1d_da_nlte', [defaults[band] for band in bands],
                                                units = units)
    interp, _, (T, L, A, grid_sansav, grid) = sed.make_cache(nAV=7)
    return interp, sed