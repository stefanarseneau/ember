"""Shared data-loading helpers for analysis scripts."""

import pandas as pd

from catalog.config import DATA_DIR, TYLER_CODE_DIR, FIGURES_DIR, STYLE_FILE


def setup_matplotlib():
	import matplotlib as mpl
	import matplotlib.pyplot as plt

	plt.style.use(str(STYLE_FILE))
	mpl.rcParams.update({
		"text.usetex": True,
		"font.family": "serif",
		"text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}",
		"pdf.fonttype": 42,
		"ps.fonttype": 42,
		"svg.fonttype": "none",
	})
	mpl.use("Agg")
	FIGURES_DIR.mkdir(exist_ok=True)
	return plt, mpl


def load_catalog():
	"""Return (combined, normal, white_ll, blue_ll) from the built parquet.

	age_class encoding (set by catalog.build):
	  0 = normal   — totalage star, age upper error ≤ 13 Gyr
	  1 = white_ll — totalage with large upper error, or 0.6 < M ≤ 0.63
	  2 = blue_ll  — M ≤ 0.6
	"""
	combined = pd.read_parquet(DATA_DIR / "catalogs/combined.pqt")
	normal   = combined["age_class"] == 0
	white_ll = combined["age_class"] == 1
	blue_ll  = combined["age_class"] == 2
	return combined, normal, white_ll, blue_ll


def load_metallicity():
	"""Long-form parquet: one row per (source, survey). Use for provenance/count tables."""
	return pd.read_parquet(DATA_DIR / "catalogs/metallicity.pqt")
