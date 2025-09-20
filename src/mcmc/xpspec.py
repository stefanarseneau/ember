import pandas as pd
import numpy as np
import argparse
import pyvo

from GaiaDR3XPspectracorrectionV1 import Gaia_Correction_V1
from gaiaxpy import generate, PhotometricSystem, apply_error_correction, calibrate
from astroquery.gaia import Gaia
Gaia.ROW_LIMIT = -1

import interpolator
from . import util

def merge_ngf21(source_ids):
    tap_service = pyvo.dal.TAPService("http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")
    QUERY = f"""select GaiaEDR3 as gaia_dr3_source_id, TeffH, e_TeffH, loggH, e_loggH 
            from \"J/MNRAS/508/3877/maincat\"
            where GaiaEDR3 in {tuple(source_ids)}"""
    return tap_service.search(QUERY).table.to_pandas()

def _make_xpphotometry(source_ids : list[int], systems : list[str]) -> pd.DataFrame:
    """fetch synthetic photometry in raw Gaia format"""
    tap_service = pyvo.dal.TAPService("http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")
    QUERY = f"""select GaiaEDR3 as source_id, GmagCorr
            from \"J/MNRAS/508/3877/maincat\"
            where GaiaEDR3 in {tuple(source_ids)}"""
    ngfdata = tap_service.search(QUERY).to_table().to_pandas()
    _, sed = util.make_interpolator(systems, units='fnu')

    original, sampling = calibrate(source_ids, save_file=False, truncation=False)
    converted_spectra = Gaia_Correction_V1.correction_df(original, 
                            ngfdata.GmagCorr.to_numpy(), 
                            have_error=True, 
                            Truncation=False, 
                            absolute_correction=True
                        )
    fluxes = np.vstack(3.33564095e06*converted_spectra.flux_cor.to_numpy())*(10*sampling[None,:])**2
    e_fluxes = np.vstack(3.33564095e06*converted_spectra.flux_error.to_numpy())*(10*sampling[None,:])**2

    photo, e_photo = [], []
    for ii in range(fluxes.shape[0]):
        photo.append(sed.SED(fluxes[ii], wavl=sampling*10))
        e_photo.append(sed.SED(e_fluxes[ii], wavl=sampling*10))
    photo = np.array(photo) ; e_photo = np.array(e_photo)

    names = util.convert_names(systems)
    fluxnames = [syst.replace('_', '_flux_') for syst in names]
    e_fluxnames = [syst.replace('_', '_flux_error_') for syst in names]

    synthetic_photometry = pd.DataFrame({
        'gaia_dr3_source_id' : converted_spectra.source_id
    })

    synthetic_photometry[fluxnames] = photo
    synthetic_photometry[e_fluxnames] = e_photo
    return synthetic_photometry, sampling, converted_spectra

def _convert_gaia_flux(
        df : pd.DataFrame, 
        cols : list = ['GaiaDr3Vega_flux_G', 'GaiaDr3Vega_flux_BP', 'GaiaDr3Vega_flux_RP'],
        e_cols : list = ['GaiaDr3Vega_flux_error_G', 'GaiaDr3Vega_flux_error_BP', 'GaiaDr3Vega_flux_error_RP']
    ) -> pd.DataFrame:
    """convert gaia flux from e/s to fnu"""
    factors_flam = np.array([1.346109E-21, 3.009167E-21, 1.638483E-21])
    factors_fnu = np.array([1.736011E-33, 2.620707e-33, 3.298815e-33])    
    df[cols] *= factors_fnu / factors_flam
    df[e_cols] *= factors_fnu / factors_flam
    return df

def _watt_to_jy(df : pd.DataFrame, flux_dict : dict):
    """convert fluxes from W m^-2 Hz^-1 to Jy"""
    for ii, (flux, error) in enumerate(flux_dict.values()):
        df[flux] *= 1e26
        df[error] *= 1e26
    return df

def _remove_gaia_cols(df : pd.DataFrame) -> pd.DataFrame:
    """remove all the Gaia columns from the dataframe"""
    return df.drop(labels=[col for col in df.columns if 'Gaia' in col], axis=1)

def _get_gaia_flux(source_ids : list[int]) -> pd.DataFrame:
    """query the gaia archive and return gaia fluxes in jy"""
    QUERY = f"""select source_id as gaia_dr3_source_id, 
                  phot_g_mean_flux as gflux, 
                  phot_g_mean_flux_error as e_gflux
                from gaiadr3.gaia_source
                  where source_id in {tuple(source_ids)}"""
    results = Gaia.launch_job_async(QUERY).get_results().to_pandas()
    # perform unit conversions
    results[['gflux', 'e_gflux']] *= 1.736011E-33 * 1e26
    return results

def process_dataframe(
        df : pd.DataFrame, 
        systems : list[str] = ['gaia', 'sdss', 'panstarrs'], 
        use_gravz : bool = False,
    ) -> pd.DataFrame:
    assert set(systems).issubset(set(['gaia', 'sdss', 'panstarrs', 'jplus', 'synthetic'])), "Err: unsupported photometric system!"
    ff = interpolator.atmos.sed.get_default_filters()
    photometry_dict = {
        'gaia' : ['Gaia_G', 'Gaia_BP', 'Gaia_RP'], 
        'sdss' : ['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z'],
        'panstarrs' : ['PS1_g', 'PS1_r', 'PS1_i', 'PS1_z', 'PS1_y'],
        'jplus' : ['JPLUS_J0378', 'JPLUS_J0395', 'JPLUS_J0410', 'JPLUS_J0430',
                    'JPLUS_J0515', 'JPLUS_J0660', 'JPLUS_J0861']
    }
    photometry_dict['synthetic'] = [band for band in ff.keys() if "SYNTH_" in band]
    phot_system = np.concatenate([photometry_dict[system] for system in systems], axis=0)
    # preprocess the dataframe
    df = util.check_valid(df, use_gravz)
    # fetch xp photometry
    source_ids = df['gaia_dr3_source_id'].values.astype(int).tolist()
    synphot, _, _ = _make_xpphotometry(source_ids, systems=phot_system)
    # fix the Gaia photometry
    #synphot = _convert_gaia_flux(synphot) if 'gaia' in systems else _remove_gaia_cols(synphot)
    flux_dict = util.find_photocols(synphot)
    #synphot = _watt_to_jy(synphot, flux_dict)
    synphot = pd.merge(synphot, _get_gaia_flux(source_ids), on='gaia_dr3_source_id')
    return pd.merge(df, synphot, on='gaia_dr3_source_id'), flux_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch synthetic spectra from XP photometry.",
                                        epilog="Example:\n"",",
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inpath', type=str, default = 'targets.csv', help='Path to input CSV file')
    parser.add_argument('outpath', type=str, default = 'mcmc_info.csv', help='Path to input CSV file')
    args = parser.parse_args()
    # read the dataframe and check that we have the necessary information
    dataframe = pd.read_parquet(args.inpath)
    synphot, fluxdict = process_dataframe(dataframe)
    print(synphot)
    
