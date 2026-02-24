import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from io import StringIO
from scipy.interpolate import interp1d

def read_wdcool(path: Path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.strip().startswith('#'):
                cols = re.split(r'\s+', line.strip().lstrip('#').strip())
                break
    df = pd.read_csv(
            path,
            comment="#",
            sep=r"\s+",     # replaces delim_whitespace=True
            header=None,
            names=cols,
            engine="python"
        )
    return df, cols

def parse_track_file(path: Path) -> Tuple[float, float, pd.DataFrame]:
    M_in = M_WD = None; cols = None; data = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            s = line.strip()
            if s.startswith('# M_in'):  M_in = float(re.search(r'=\s*([0-9.]+)', s).group(1))
            elif s.startswith('# M_WD'): M_WD = float(re.search(r'=\s*([0-9.]+)', s).group(1))
            elif s.startswith('# log_cool_age'):
                raw_cols = re.split(r'\s+', s.lstrip('#').strip())
                cols = [c for c in raw_cols if c != '...']  # drop literal ellipsis
            elif s:
                data.append(line.replace('...', ' '))
    if M_WD is None:
        m = re.search(r'MWD_([0-9.]+)', path.name)
        if m: M_WD = float(m.group(1))
    df = pd.read_csv(StringIO(''.join(data)), sep=r"\s+", header=None, names=cols, engine='python')
    return M_in, M_WD, df[['log_cool_age','log_tot_age']].copy()

def build_mass_to_track(tracks_dir: Path) -> Dict[float, Tuple[float, float, pd.DataFrame]]:
    mapping = {}
    for p in sorted(tracks_dir.glob("*.data")):
        try:
            M_in, M_WD, df = parse_track_file(p)
            if M_WD is not None:
                mapping[M_WD] = (M_in, M_WD, df.sort_values('log_cool_age'))
        except Exception:
            pass
    if not mapping:
        raise ValueError(f"No usable track files in {tracks_dir}")
    return mapping

def assign_log_tot_age(df_wd: pd.DataFrame, mass_to_track: Dict[float, Tuple[float, float, pd.DataFrame]],
                       mass_col='Mass', cool_col='log_age', mass_tol=0.02) -> pd.Series:
    masses = np.array(sorted(mass_to_track.keys()))
    interps = {}
    for M, (_, _, tdf) in mass_to_track.items():
        x = tdf['log_cool_age'].to_numpy()
        y = tdf['log_tot_age'].to_numpy()
        # dedup x for interp1d robustness
        x_uniq, idx = np.unique(x, return_index=True)
        interps[M] = interp1d(x_uniq, y[idx], kind='linear', fill_value='extrapolate', assume_sorted=True)
    out = np.full(len(df_wd), np.nan, float)
    for i, (m, a) in enumerate(zip(df_wd[mass_col].to_numpy(), df_wd[cool_col].to_numpy())):
        j = np.argmin(np.abs(masses - m)); Mnear = masses[j]
        if abs(Mnear - m) <= mass_tol:
            out[i] = float(interps[Mnear](a))
    return pd.Series(out, name='log_tot_age', index=df_wd.index)

def write_wdcool_with_totage(df_wd: pd.DataFrame, cols_original: list, output_path: Path) -> None:
    cols = cols_original.copy()
    if 'log_tot_age' not in cols:
        insert_at = cols.index('log_age') + 1 if 'log_age' in cols else len(cols)
        cols.insert(insert_at, 'log_tot_age')
    for c in cols:
        if c not in df_wd: df_wd[c] = np.nan
    header = '# ' + '   '.join(f'{c:>12}' for c in cols)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        for row in df_wd[cols].to_numpy():
            f.write(''.join(f'{v:13.6f}' if isinstance(v,(int,float,np.floating)) and np.isfinite(v) else f'{v:>13}' for v in row) + '\n')

def process_wdcool_with_tracks(wdcool_path: Path, tracks_dir: Path, out_suffix=".with_totage.wdcool", mass_tol=0.02) -> Path:
    df_wd, cols = read_wdcool(wdcool_path)
    mapping = build_mass_to_track(tracks_dir)
    df_wd['log_tot_age'] = assign_log_tot_age(df_wd, mapping, mass_col='Mass', cool_col='log_age', mass_tol=mass_tol)
    outpath = wdcool_path.with_suffix(out_suffix)
    write_wdcool_with_totage(df_wd, cols_original=cols + (['log_tot_age'] if 'log_tot_age' not in cols else []), output_path=outpath)
    return outpath