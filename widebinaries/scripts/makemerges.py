import matplotlib.pyplot as plt
from astropy.table import Table
import pandas as pd
import numpy as np

import argparse
import os

CATALOG_DIR = "/home/arseneau/observational/catalogs"

def merge_ngf21(data : pd.DataFrame) -> pd.DataFrame:
    """merges my copy of nicola's catalog"""
    ngf = Table.read(os.path.join(CATALOG_DIR, "nicola_wds", "ngf21_maincat.fits")).to_pandas()
    merge = pd.merge(data, ngf, left_on="gaia_dr3_source_id", right_on="GaiaEDR3")
    del ngf
    return merge

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="stitch that stuff",
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inpath', type=str, default = 'data/', help='Path to input pqt file')
    parser.add_argument('outpath', type=str, default = 'ages.pqt', help='Path to output pqt file')
    args = parser.parse_args()

    data = pd.read_parquet(args.inpath)

    data_ngf21 = merge_ngf21(data)
    ngf_path = os.path.join(args.outpath, "ngf21_merged.pqt")
    data_ngf21.to_parquet(ngf_path)
    print(f"saved {len(data_ngf21)} rows to {ngf_path}")
