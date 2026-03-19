import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.io import fits
import pandas as pd
import numpy as np

import pyvo
import argparse
import os

CATALOG_DIR = "/home/arseneau/observational/catalogs"

def _read_tars():
    return pd.read_feather(os.path.join(CATALOG_DIR, 'tars_table_2.feather'))

def merge_ngf21(data : pd.DataFrame) -> pd.DataFrame:
    """merges my copy of nicola's catalog"""
    ngf = Table.read(os.path.join(CATALOG_DIR, "nicola_wds", "ngf21_maincat.fits")).to_pandas()
    merge = pd.merge(data, ngf, left_on="sourceid", right_on="GaiaEDR3")
    del ngf
    return merge

def merge_astra(data: pd.DataFrame) -> pd.DataFrame:
    astra = Table.read(os.path.join(CATALOG_DIR, "sdss", "astraAllStarASPCAP-0.8.fits.gz"), hdu=2)
    names = [name for name in astra.colnames if len(astra[name].shape) <= 1]
    astra = astra[names].to_pandas()
    print(astra.head())
    merge = pd.merge(data, astra, left_on="ms_source_id", right_on="gaia_dr3_source_id")
    del astra
    tars = _read_tars()
    merge_tars = pd.merge(merge, tars, left_on="ms_source_id", right_on="dr3_source_id")
    del tars
    return merge, merge_tars

def merge_apogee(data: pd.DataFrame) -> pd.DataFrame:
    apogee = Table.read(os.path.join(CATALOG_DIR, "sdss", "allStar-dr17-synspec_rev1.fits"))
    names = [name for name in apogee.colnames if len(apogee[name].shape) <= 1]
    apogee = apogee[names].to_pandas()
    merge = pd.merge(data, apogee, left_on="ms_source_id", right_on="GAIAEDR3_SOURCE_ID")
    del apogee
    tars = _read_tars()
    merge_tars = pd.merge(merge, tars, left_on="ms_source_id", right_on="dr3_source_id")
    del tars
    return merge, merge_tars

def merge_galah(data: pd.DataFrame) -> pd.DataFrame:
    galah = Table.read(os.path.join(CATALOG_DIR, "GALAH_DR3_main_allstar_v2.fits"))
    names = [name for name in galah.colnames if len(galah[name].shape) <= 1]
    galah = galah[names].to_pandas()
    merge = pd.merge(data, galah, left_on="ms_source_id", right_on="dr3_source_id")
    del galah
    tars = _read_tars()
    merge_tars = pd.merge(merge, tars, left_on="ms_source_id", right_on="dr3_source_id")
    del tars
    return merge, merge_tars

def merge_gyro_chiti(data : pd.DataFrame) -> pd.DataFrame:
    """merges my stuff with the gyro"""
    gyro = pd.read_csv("~/observational/software/ember/widebinaries/data/wdms_gyro_chiti.csv")
    merge = pd.merge(data, gyro, left_on="sourceid", right_on="wd_source_id")
    del gyro
    return merge

def merge_gyro_lu(data : pd.DataFrame) -> pd.DataFrame:
    """merges my stuff with the gyro"""
    tap_service = pyvo.dal.TAPService("http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")
    QUERY = f"""select *
        from \"J/AJ/167/159/table2\"
        where Gaia in {tuple(data["ms_source_id"])}
        """
    gyro = tap_service.search(QUERY).to_table().to_pandas()
    merge = pd.merge(data, gyro, left_on="ms_source_id", right_on="Gaia")
    del gyro
    return merge

def merge_heintz(data : pd.DataFrame) -> pd.DataFrame:
    """merges my stuff with the gyro"""
    heintz2022 = pd.read_csv("~/observational/software/ember/widebinaries/data/tyler_wdms.csv")
    merge = pd.merge(heintz2022, data, left_on="source_id", right_on="sourceid")
    del heintz2022
    return merge

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stitch that stuff",
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inpath', type=str, default = 'data/', help='Path to input pqt file')
    parser.add_argument('outpath', type=str, default = 'ages.pqt', help='Path to output pqt file')
    args = parser.parse_args()

    data = pd.read_parquet(args.inpath)
    wdms_wb = pd.read_parquet("~/observational/software/ember/widebinaries/wdms_widebinaries.pqt").query("R_chance_align < 0.1")
    wdms_data = pd.merge(wdms_wb, data, left_on="wd_source_id", right_on="sourceid")
    wdms_data.to_csv("data/wdms_data.csv")

    data_ngf21 = merge_ngf21(data)
    ngf_path = os.path.join(args.outpath, "ngf21_merged.pqt")
    data_ngf21.to_parquet(ngf_path)
    print(f"saved {len(data_ngf21)} rows to {ngf_path}")

    """
    data_astra, tars_astra = merge_astra(wdms_data)
    astra_path = os.path.join(args.outpath, "astra_ngf21_merged.pqt")
    data_astra.to_parquet(astra_path)
    print(f"saved {len(data_astra)} rows to {astra_path}")
    tarastra_path = os.path.join(args.outpath, "astra_tars_merged.pqt")
    tars_astra.to_parquet(tarastra_path)
    print(f"saved {len(tars_astra)} rows to {tar_path}")

    data_galah, tars_galah = merge_galah(wdms_data)
    galah_path = os.path.join(args.outpath, "galah_ngf21_merged.pqt")
    data_galah.to_parquet(galah_path)
    print(f"saved {len(data_galah)} rows to {galah_path}")
    targalah_path = os.path.join(args.outpath, "galah_tars_merged.pqt")
    tars_galah.to_parquet(targalah_path)
    print(f"saved {len(tars_galah)} rows to {tar_path}")

    data_apogee, tars_apogee = merge_apogee(wdms_data)
    apogee_path = os.path.join(args.outpath, "apogee_ngf21_merged.pqt")
    data_apogee.to_parquet(apogee_path)
    print(f"saved {len(data_apogee)} rows to {apogee_path}")
    tarapogee_path = os.path.join(args.outpath, "apogee_tars_merged.pqt")
    tars_apogee.to_parquet(tarapogee_path)
    print(f"saved {len(tars_apogee)} rows to {tar_path}")
    """

    data_gyro_chiti = merge_gyro_chiti(wdms_data)
    gyro_path_chiti = os.path.join(args.outpath, "wbinary_gyro_ages_chiti.pqt")
    data_gyro_chiti.to_parquet(gyro_path_chiti)
    print(f"saved {len(data_gyro_chiti)} rows to {gyro_path_chiti}")

    #data_gyro_lu = merge_gyro_lu(wdms_data)
    #gyro_path_lu = os.path.join(args.outpath, "wbinary_gyro_ages_lu.pqt")
    #data_gyro_lu.to_parquet(gyro_path_lu)
    #print(f"saved {len(data_gyro_lu)} rows to {gyro_path_lu}")

    data_heintz = merge_heintz(data)
    heintz_path = os.path.join(args.outpath, "wbinary_heintz2022.pqt")
    data_heintz.to_parquet(heintz_path)
    print(f"saved {len(data_heintz)} rows to {heintz_path}")
