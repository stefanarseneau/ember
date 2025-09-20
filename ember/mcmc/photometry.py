import pandas as pd
import numpy as np
import argparse
import pyvo
from functools import reduce

from astroquery.gaia import Gaia
Gaia.ROW_LIMIT = -1

from . import util

def abmag_to_flux(mag : np.ndarray, e_mag : np.ndarray):
    """AB magnitude to flux in Jy"""
    flux = np.power(10, -0.4*(mag - 8.90))
    e_flux = 1.09 * flux * e_mag
    return flux, e_flux

def _get_jplus_flux(source_ids : list[np.int64]) -> pd.DataFrame:
    tap_service = pyvo.dal.TAPService("https://archive.cefca.es/catalogues/vo/tap/jplus-dr3")
    QUERY = f"""select gaia.source as gaia_dr3_source_id,
                    jfl.flux_auto[jplus::rSDSS]*1e-7 as "Jplus_flux_rSDSS", 
                    jfl.flux_relerr_auto[jplus::rSDSS]*1e-7 as "Jplus_flux_error_rSDSS",
                    jfl.flux_auto[jplus::gSDSS]*1e-7 as "Jplus_flux_gSDSS", 
                    jfl.flux_relerr_auto[jplus::gSDSS]*1e-7 as "Jplus_flux_error_gSDSS",
                    jfl.flux_auto[jplus::iSDSS]*1e-7 as "Jplus_flux_iSDSS", 
                    jfl.flux_relerr_auto[jplus::iSDSS]*1e-7 as "Jplus_flux_error_iSDSS",
                    jfl.flux_auto[jplus::zSDSS]*1e-7 as "Jplus_flux_zSDSS", 
                    jfl.flux_relerr_auto[jplus::zSDSS]*1e-7 as "Jplus_flux_error_zSDSS",
                    jfl.flux_auto[jplus::uJAVA]*1e-7 as "Jplus_flux_uJAVA", 
                    jfl.flux_relerr_auto[jplus::uJAVA]*1e-7 as "Jplus_flux_error_uJAVA",
                    jfl.flux_auto[jplus::J0378]*1e-7 as "Jplus_flux_J0378", 
                    jfl.flux_relerr_auto[jplus::J0378]*1e-7 as "Jplus_flux_error_J0378",
                    jfl.flux_auto[jplus::J0395]*1e-7 as "Jplus_flux_J0395", 
                    jfl.flux_relerr_auto[jplus::J0395]*1e-7 as "Jplus_flux_error_J0395",
                    jfl.flux_auto[jplus::J0410]*1e-7 as "Jplus_flux_J0410", 
                    jfl.flux_relerr_auto[jplus::J0410]*1e-7 as "Jplus_flux_error_J0410",
                    jfl.flux_auto[jplus::J0430]*1e-7 as "Jplus_flux_J0430", 
                    jfl.flux_relerr_auto[jplus::J0430]*1e-7 as "Jplus_flux_error_J0430",
                    jfl.flux_auto[jplus::J0515]*1e-7 as "Jplus_flux_J0515", 
                    jfl.flux_relerr_auto[jplus::J0515]*1e-7 as "Jplus_flux_error_J0515",
                    jfl.flux_auto[jplus::J0660]*1e-7 as "Jplus_flux_J0660", 
                    jfl.flux_relerr_auto[jplus::J0660]*1e-7 as "Jplus_flux_error_J0660",
                    jfl.flux_auto[jplus::J0861]*1e-7 as "Jplus_flux_J0861", 
                    jfl.flux_relerr_auto[jplus::J0861]*1e-7 as "Jplus_flux_error_J0861"
                from jplus.FNuDualObj as jfl
                join jplus.xmatch_gaia_dr3 as gaia
                    on gaia.tile_id = jfl.tile_id and gaia.NUMBER = jfl.NUMBER
                where gaia.Source in {tuple(source_ids)}"""
    table = tap_service.search(QUERY).to_table().to_pandas()
    return table.rename(columns={
                            'jplus_flux_rsdss': 'Jplus_flux_rJPLUS',
                            'jplus_flux_error_rsdss': 'Jplus_flux_error_rJPLUS',
                            'jplus_flux_gsdss': 'Jplus_flux_gJPLUS',
                            'jplus_flux_error_gsdss': 'Jplus_flux_error_gJPLUS',
                            'jplus_flux_isdss': 'Jplus_flux_iJPLUS',
                            'jplus_flux_error_isdss': 'Jplus_flux_error_iJPLUS',
                            'jplus_flux_zsdss': 'Jplus_flux_zJPLUS',
                            'jplus_flux_error_zsdss': 'Jplus_flux_error_zJPLUS',
                            'jplus_flux_ujava': 'Jplus_flux_uJAVA',
                            'jplus_flux_error_ujava': 'Jplus_flux_error_uJAVA',
                            'jplus_flux_j0378': 'Jplus_flux_J0378',
                            'jplus_flux_error_j0378': 'Jplus_flux_error_J0378',
                            'jplus_flux_j0395': 'Jplus_flux_J0395',
                            'jplus_flux_error_j0395': 'Jplus_flux_error_J0395',
                            'jplus_flux_j0410': 'Jplus_flux_J0410',
                            'jplus_flux_error_j0410': 'Jplus_flux_error_J0410',
                            'jplus_flux_j0430': 'Jplus_flux_J0430',
                            'jplus_flux_error_j0430': 'Jplus_flux_error_J0430',
                            'jplus_flux_j0515': 'Jplus_flux_J0515',
                            'jplus_flux_error_j0515': 'Jplus_flux_error_J0515',
                            'jplus_flux_j0660': 'Jplus_flux_J0660',
                            'jplus_flux_error_j0660': 'Jplus_flux_error_J0660',
                            'jplus_flux_j0861': 'Jplus_flux_J0861',
                            'jplus_flux_error_j0861': 'Jplus_flux_error_J0861'
                        })

def _get_panstarrs_flux(source_ids : list[np.int64]) -> pd.DataFrame:
    tap_service = pyvo.dal.TAPService("https://gea.esac.esa.int/tap-server/tap/")
    QUERY1 = f"""select source_id as gaia_dr3_source_id, original_ext_source_id as objid
                from gaiadr3.panstarrs1_best_neighbour
                where source_id in {tuple(source_ids)}"""
    gaianames = Gaia.launch_job_async(QUERY1).get_results().to_pandas()

    tap_service = pyvo.dal.TAPService("http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")
    QUERY2 = f"""select 
                    objID as objid,
                    gmag as Panstarrs1_mag_gp, e_gmag as Panstarrs1_mag_error_gp,
                    rmag as Panstarrs1_mag_rp, e_rmag as Panstarrs1_mag_error_rp,
                    imag as Panstarrs1_mag_ip, e_imag as Panstarrs1_mag_error_ip,
                    zmag as Panstarrs1_mag_zp, e_zmag as Panstarrs1_mag_error_zp,
                    ymag as Panstarrs1_mag_yp, e_ymag as Panstarrs1_mag_error_yp
                from \"II/349/ps1\"
                where objid in {tuple(gaianames.objid.tolist())}
                    and (f_objID & 33554432) != 0"""
    ps1photo = tap_service.search(QUERY2).to_table().to_pandas().dropna()

    for band in ["g", "r", "i", "z", "y"]:
        mag = ps1photo[f"Panstarrs1_mag_{band}p"].values
        mag_err = ps1photo[f"Panstarrs1_mag_error_{band}p"].values
        flux, flux_err = abmag_to_flux(mag, mag_err)
        ps1photo[f"Panstarrs1_flux_{band}p"] = flux
        ps1photo[f"Panstarrs1_flux_error_{band}p"] = flux_err
    return pd.merge(gaianames, ps1photo, on='objid')

def _get_sdss_flux(source_ids : list[np.int64]) -> pd.DataFrame:
    """query the SDSS archive and return fluxes in jy"""
    # query the SDSS archive
    tap_service = pyvo.dal.TAPService("http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")
    QUERY = f"""select GaiaEDR3 as gaia_dr3_source_id, *
                from \"J/MNRAS/508/3877/maincat\"
                where GaiaEDR3 in {tuple(source_ids)}
                and sdssclean = 1"""
    sdssphoto = tap_service.search(QUERY).to_table().to_pandas()

    sdssphoto['Sdss_flux_u'] = 10**(-0.4*(sdssphoto['umag'] -0.040 - 8.90))
    sdssphoto['Sdss_flux_error_u'] = 1.09 * sdssphoto['Sdss_flux_u'] * sdssphoto['e_umag']

    sdssphoto['Sdss_flux_g'] = 10**(-0.4*(sdssphoto['gmag'] - 8.90))
    sdssphoto['Sdss_flux_error_g'] = 1.09 * sdssphoto['Sdss_flux_g'] * sdssphoto['e_gmag']

    sdssphoto['Sdss_flux_r'] = 10**(-0.4*(sdssphoto['rmag'] - 8.90))
    sdssphoto['Sdss_flux_error_r'] = 1.09 * sdssphoto['Sdss_flux_r'] * sdssphoto['e_rmag']

    sdssphoto['Sdss_flux_i'] = 10**(-0.4*(sdssphoto['imag'] + 0.015 - 8.90))
    sdssphoto['Sdss_flux_error_i'] = 1.09 * sdssphoto['Sdss_flux_i'] * sdssphoto['e_imag']

    sdssphoto['Sdss_flux_z'] = 10**(-0.4*(sdssphoto['zmag'] + 0.030 - 8.90))
    sdssphoto['Sdss_flux_error_z'] = 1.09 * sdssphoto['Sdss_flux_z'] * sdssphoto['e_zmag']

    return sdssphoto[['gaia_dr3_source_id', 
                        'Sdss_flux_u', 'Sdss_flux_error_u',
                        'Sdss_flux_g', 'Sdss_flux_error_g',
                        'Sdss_flux_r', 'Sdss_flux_error_r',
                        'Sdss_flux_i', 'Sdss_flux_error_i',
                        'Sdss_flux_z', 'Sdss_flux_error_z']].dropna()

def _get_gaia_flux(source_ids : list[int]) -> pd.DataFrame:
    """query the gaia archive and return gaia fluxes in jy"""
    QUERY = f"""select source_id as gaia_dr3_source_id, 
                  phot_g_mean_flux as GaiaDr3Vega_flux_G, 
                  phot_g_mean_flux_error as GaiaDr3Vega_flux_error_G, 
                  phot_bp_mean_flux as GaiaDr3Vega_flux_BP, 
                  phot_bp_mean_flux_error as GaiaDr3Vega_flux_error_BP, 
                  phot_rp_mean_flux as GaiaDr3Vega_flux_RP, 
                  phot_rp_mean_flux_error as GaiaDr3Vega_flux_error_RP
                from gaiadr3.gaia_source
                  where source_id in {tuple(source_ids)}"""
    results = Gaia.launch_job_async(QUERY).get_results().to_pandas()
    # perform unit conversions
    results[['GaiaDr3Vega_flux_G', 'GaiaDr3Vega_flux_error_G']] *= 1.736011E-33 * 1e26
    results[['GaiaDr3Vega_flux_BP', 'GaiaDr3Vega_flux_error_BP']] *= 2.620707e-33 * 1e26
    results[['GaiaDr3Vega_flux_RP', 'GaiaDr3Vega_flux_error_RP']] *= 3.298815e-33 * 1e26
    results[['gflux', 'e_gflux']] = results[['GaiaDr3Vega_flux_G', 'GaiaDr3Vega_flux_error_G']].values
    return results

def process_dataframe(df : pd.DataFrame, systems : list[str] = ['gaia', 'sdss'], use_gravz : bool = False) -> pd.DataFrame:
    assert set(systems).issubset(set(['gaia', 'sdss', 'panstarrs', 'jplus'])), "Err: unsupported photometric system!"
    photometry_dict = {'gaia' : _get_gaia_flux, 'sdss' : _get_sdss_flux,
                       'panstarrs' : _get_panstarrs_flux,
                       'jplus' : _get_jplus_flux}
    phot_system = [photometry_dict[system] for system in systems]
    # preprocess the dataframe
    df = util.check_valid(df, use_gravz)
    source_ids = df['gaia_dr3_source_id'].values.astype(np.int64).tolist()
    synphot_list = [system(source_ids) for system in phot_system]
    synphot = reduce(lambda left, right: pd.merge(left, right, on='gaia_dr3_source_id'), synphot_list)
    # fix the Gaia photometry
    flux_dict = util.find_photocols(synphot)
    return pd.merge(df, synphot, on='gaia_dr3_source_id'), flux_dict