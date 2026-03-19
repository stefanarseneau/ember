"""
Bulk download and calibrate Gaia XP spectra for all sources in
widebinary_fluxes.pqt using gaiaxpy. Processes in chunks and checkpoints
after each chunk so the download can be resumed if interrupted.

Usage:
    python download_xp_spectra.py [--chunksize N] [--username USER]

Credentials: set GAIA_USER and GAIA_PASS environment variables, or pass
--username / --password on the command line.
"""
import argparse
import os
import numpy as np
import pandas as pd
import gaiaxpy

FLUXES_PATH  = '../widebinaries/widebinary_fluxes.pqt'
OUTPUT_PATH  = '../widebinaries/xp_spectra.pqt'
CHECKPOINT   = '../widebinaries/xp_spectra_ckpt.pqt'

# Wavelength grid: 336–1020 nm in 2 nm steps (same as gaiaxpy default range)
SAMPLING = np.arange(336, 1021, 2)


def load_done(path):
    if os.path.exists(path):
        return set(pd.read_parquet(path)['source_id'].tolist())
    return set()


def append_parquet(df, path):
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        pd.concat([existing, df], ignore_index=True).to_parquet(path)
    else:
        df.to_parquet(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--chunksize', type=int, default=200,
                        help='Number of sources per request (default: 200)')
    parser.add_argument('--username', type=str,
                        default=os.environ.get('GAIA_USER'),
                        help='Gaia Cosmos username (or set GAIA_USER)')
    parser.add_argument('--password', type=str,
                        default=os.environ.get('GAIA_PASS'),
                        help='Gaia Cosmos password (or set GAIA_PASS)')
    args = parser.parse_args()

    fluxes = pd.read_parquet(FLUXES_PATH)
    all_ids = fluxes['gaia_dr3_source_id'].tolist()

    done = load_done(CHECKPOINT)
    remaining = [sid for sid in all_ids if sid not in done]
    print(f'Total sources: {len(all_ids)}  |  already done: {len(done)}  |  remaining: {len(remaining)}')

    chunks = [remaining[i:i + args.chunksize]
              for i in range(0, len(remaining), args.chunksize)]

    for k, chunk in enumerate(chunks):
        print(f'Chunk {k+1}/{len(chunks)}  ({len(chunk)} sources)...', flush=True)
        try:
            spectra, sampling = gaiaxpy.calibrate(
                chunk,
                sampling=SAMPLING,
                save_file=False,
                username=args.username,
                password=args.password,
            )
        except Exception as e:
            print(f'  ERROR on chunk {k+1}: {e}')
            continue

        append_parquet(spectra, CHECKPOINT)
        done.update(chunk)
        print(f'  done. Total checkpointed: {len(done)}')

    # Rename checkpoint to final output
    if os.path.exists(CHECKPOINT):
        os.replace(CHECKPOINT, OUTPUT_PATH)
        print(f'\nSaved {len(done)} spectra to {OUTPUT_PATH}')
    else:
        print('No spectra downloaded.')
