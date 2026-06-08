"""Velocity distribution of cool (Teff < 6000 K) white dwarfs in Galactic coordinates."""

import sys
from pathlib import Path
import argparse

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

sys.path.insert(0, str(Path(__file__).parent))
from load_data import (
    OUT_DIR, load_main_data, load_gaia_ms_rv, setup_matplotlib, bensby_classify
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--isotropic-rvs", action="store_true",
    help="Fill missing Gaia RVs by drawing from the isotropic distribution",
)
args = parser.parse_args()
rv_suffix = "_iso" if args.isotropic_rvs else "_gaia"

plt, _ = setup_matplotlib()

# ── Data loading ───────────────────────────────────────────────────────────────
data = load_main_data()

U_sun, V_sun, W_sun = 11.0, 12.0, 7.0  # Schönrich et al. 2010


def get_lsr_vtransverse(dataframe):
	coords = SkyCoord(
		ra=dataframe["ra"].values * u.deg,
		dec=dataframe["dec"].values * u.deg,
		pm_ra_cosdec=dataframe["pmra"].values * u.mas / u.yr,
		pm_dec=dataframe["pmdec"].values * u.mas / u.yr,
		frame="icrs",
	)
	gal = coords.galactic
	pm_l_cosb = gal.pm_l_cosb.to(u.mas / u.yr).value
	pm_b      = gal.pm_b.to(u.mas / u.yr).value

	d_kpc     = 1.0 / dataframe["wtd_par"].values
	v_l_helio = 4.74047 * pm_l_cosb * d_kpc
	v_b_helio = 4.74047 * pm_b      * d_kpc

	l_rad = np.radians(dataframe["l"].values)
	b_rad = np.radians(dataframe["b"].values)
	v_l_sun = -U_sun * np.sin(l_rad) + V_sun * np.cos(l_rad)
	v_b_sun = (
		-U_sun * np.cos(l_rad) * np.sin(b_rad)
		- V_sun * np.sin(l_rad) * np.sin(b_rad)
		+ W_sun * np.cos(b_rad)
	)
	return v_l_helio - v_l_sun, v_b_helio - v_b_sun


data["v_l"], data["v_b"] = get_lsr_vtransverse(data)

print("Loading Gaia MS radial velocities...")
rv_gaia = load_gaia_ms_rv(data["ms_source_id"].values)
data = data.merge(rv_gaia, on="ms_source_id", how="left")

l_rad_all = np.radians(data["l"].values)
b_rad_all = np.radians(data["b"].values)
v_r_sun = (
	U_sun * np.cos(b_rad_all) * np.cos(l_rad_all)
	+ V_sun * np.cos(b_rad_all) * np.sin(l_rad_all)
	+ W_sun * np.sin(b_rad_all)
)
data["v_r_lsr"] = np.where(
	data["radial_velocity"].notna(),
	data["radial_velocity"].values - v_r_sun,
	np.nan,
)

n_rv = data["radial_velocity"].notna().sum()
print(f"Gaia RVs available: {n_rv} / {len(data)} ({100*n_rv/len(data):.1f}%)")

coolwd = data.query("Teff < 6000").copy()
n_rv_cool = coolwd["radial_velocity"].notna().sum()
print(f"Cool WDs (Teff < 6000 K): {len(coolwd)}  (RVs: {n_rv_cool})")
print(f"  sigma_l = {coolwd['v_l'].std():.1f} km/s")
print(f"  sigma_b = {coolwd['v_b'].std():.1f} km/s")

# ── UVW velocities ─────────────────────────────────────────────────────────────
def get_uvw(df):
	l_rad = np.radians(df["l"].values)
	b_rad = np.radians(df["b"].values)
	sl, cl = np.sin(l_rad), np.cos(l_rad)
	sb, cb = np.sin(b_rad), np.cos(b_rad)
	vl = df["v_l"].values
	vb = df["v_b"].values
	if args.isotropic_rvs:
		sigma_r = np.sqrt(0.5 * (vl**2 + vb**2))  # per-star MLE of sigma under isotropy
		vr_iso = np.random.normal(0.0, sigma_r)
		vr = np.where(np.isfinite(df["v_r_lsr"].values), df["v_r_lsr"].values, vr_iso)
	else:
		vr = np.where(np.isfinite(df["v_r_lsr"].values), df["v_r_lsr"].values, np.inf)
	U = sl * vl - sb * cl * vb + cb * cl * vr
	V = cl * vl - sb * sl * vb + cb * sl * vr
	W = cb * vb + sb * vr
	return U, V, W


np.random.seed(42)
U_all,  V_all,  W_all  = get_uvw(data)
U_cool, V_cool, W_cool = get_uvw(coolwd)

ok_all  = np.isfinite(U_all)  & np.isfinite(V_all)  & np.isfinite(W_all)
ok_cool = np.isfinite(U_cool) & np.isfinite(V_cool) & np.isfinite(W_cool)

X_all  = np.column_stack([U_all[ok_all],   V_all[ok_all],   W_all[ok_all]])
X_cool = np.column_stack([U_cool[ok_cool], V_cool[ok_cool], W_cool[ok_cool]])

# == Bensby classification =====================================================
clf_all,  _ = bensby_classify(X_all)
clf_cool, _ = bensby_classify(X_cool)

for pop in ("thin", "thick", "halo", "hercules"):
	n = (clf_all == pop).sum()
	print(f"  {pop}: {n}/{len(clf_all)} ({100*n/len(clf_all):.1f}%)")

# Reliable age mask: Mass > 0.63 (total age available) and fractional error < 20%
def _reliable_ages(df, ok):
	age = df["tot_age"].values[ok]
	frac_err = (
		np.maximum(
			np.where(np.isfinite(df["tot_age_error_upper"].values[ok]), df["tot_age_error_upper"].values[ok], np.inf),
			np.where(np.isfinite(df["tot_age_error_lower"].values[ok]), df["tot_age_error_lower"].values[ok], np.inf),
		) / np.where(age > 0, age, np.nan)
	)
	return (df["Mass"].values[ok] > 0.63) & (df["Teff"].values[ok] > 3200) & (frac_err < 0.20) & np.isfinite(age) & (age > 0)

reliable_all  = _reliable_ages(data,   ok_all)
reliable_cool = _reliable_ages(coolwd, ok_cool)

# == Figure 1: velocity dispersion histograms ==================================
bins = np.arange(0, 201, 10)
bc   = 0.5 * (bins[:-1] + bins[1:])

counts_l, _ = np.histogram(np.abs(coolwd["v_l"]), bins=bins)
counts_b, _ = np.histogram(np.abs(coolwd["v_b"]), bins=bins)

fig, ax = plt.subplots(ncols=2, figsize=(16, 6), sharey=True)
ax[0].errorbar(bc, counts_l, yerr=np.sqrt(counts_l), fmt="o", c="k", capsize=4)
ax[0].set_xlabel(r"$|v_l|$ [km s$^{-1}$]")
ax[0].set_ylabel("N")
ax[1].errorbar(bc, counts_b, yerr=np.sqrt(counts_b), fmt="o", c="k", capsize=4)
ax[1].set_xlabel(r"$|v_b|$ [km s$^{-1}$]")
fig.tight_layout()
fig.savefig(OUT_DIR / "coolwd_avr.pdf")
plt.close()
print(f"Saved {OUT_DIR / 'coolwd_avr.pdf'}")


# ── Figure 2: Toomre diagram coloured by Bensby classification ─────────────────
def toomre_semicircles(ax, radii, **kwargs):
	theta = np.linspace(0, np.pi, 300)
	for r in radii:
		ax.plot(r * np.cos(theta), r * np.sin(theta), **kwargs)

pop_colors = {"thin": "forestgreen", "thick": "crimson", "halo": "black", "hercules": "magenta"}
pop_labels = {"thin": "Thin Disk", "thick": "Thick Disk", "halo": "Halo", "hercules": "Hercules Stream"}

fig, ax = plt.subplots(figsize=(8, 7))
for pop, color in pop_colors.items():
	mask = clf_all == pop
	ax.scatter(X_all[mask, 1], np.sqrt(X_all[mask, 0]**2 + X_all[mask, 2]**2),
	           s=12, edgecolor="k", c=color, alpha=1, label=pop_labels[pop], zorder=1, rasterized=True)
toomre_semicircles(ax, [50, 100, 150, 200], color="k", ls="--", lw=0.8, alpha=0.4, zorder=0)
ax.set_xlabel(r"$V_{\rm LSR}$ [km s$^{-1}$]")
ax.set_ylabel(r"$\sqrt{U_{\rm LSR}^2 + W_{\rm LSR}^2}$ [km s$^{-1}$]")
ax.set_ylim(0, ax.get_ylim()[1])
ax.text(0.05, 0.05, "Gaia MS RVs" if not args.isotropic_rvs else "Isotropic RVs",
        transform=ax.transAxes, fontsize=16, va="bottom")
ax.legend(framealpha=0, loc="upper right")
fig.tight_layout()
fig.savefig(OUT_DIR / f"coolwd_toomre_classified{rv_suffix}.pdf")
plt.close()
print(f"Saved {OUT_DIR / f'coolwd_toomre_classified{rv_suffix}.pdf'}")


# ── Figure 3: Toomre diagram — main sample vs cool WDs ──────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(V_all[ok_all], np.sqrt(U_all[ok_all]**2 + W_all[ok_all]**2),
           s=4, c="k", alpha=0.4, label="Main sample", zorder=1, rasterized=True)
ax.scatter(V_cool[ok_cool], np.sqrt(U_cool[ok_cool]**2 + W_cool[ok_cool]**2),
           s=12, edgecolor="k", facecolor="cyan", alpha=0.7,
           label=r"Cool WDs ($T_{\rm eff} < 6000\,{\rm K}$)", zorder=2, rasterized=True)
toomre_semicircles(ax, [50, 100, 150, 200], color="k", ls="--", lw=0.8, alpha=0.4, zorder=0)
ax.axvline(0, color="k", lw=0.5, alpha=0.3)
ax.set_xlabel(r"$V_{\rm LSR}$ [km s$^{-1}$]")
ax.set_ylabel(r"$\sqrt{U_{\rm LSR}^2 + W_{\rm LSR}^2}$ [km s$^{-1}$]")
ax.set_ylim(0, ax.get_ylim()[1])
ax.legend(framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT_DIR / f"coolwd_toomre{rv_suffix}.pdf")
plt.close()
print(f"Saved {OUT_DIR / f'coolwd_toomre{rv_suffix}.pdf'}")

# ── Figures 4 & 5: direct ages by Bensby population ──────────────────────────
bins_age = np.arange(0, 14.25, 0.5)

print("\nDirect ages by population (full sample):")
for pop in ("thin", "thick", "halo", "hercules"):
	ages = data["tot_age"].values[ok_all][(clf_all == pop) & reliable_all]
	if len(ages):
		print(f"  {pop}: N={len(ages)}, median={np.median(ages):.1f} Gyr")

fig, ax = plt.subplots(figsize=(8, 5))
for pop, color in pop_colors.items():
	ages = data["tot_age"].values[ok_all][(clf_all == pop) & reliable_all]
	if len(ages):
		ax.hist(ages, bins=bins_age, histtype="step", lw=2, color=color,
		        label=rf"{pop_labels[pop]} ($N={len(ages)}$)")
ax.set_xlabel("Age [Gyr]")
ax.set_ylabel("$N$")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / f"allwd_pop_ages{rv_suffix}.pdf")
plt.close()
print(f"Saved {OUT_DIR / f'allwd_pop_ages{rv_suffix}.pdf'}")

print("\nDirect ages by population (cool WDs):")
for pop in ("thin", "thick", "halo", "hercules"):
	ages = coolwd["tot_age"].values[ok_cool][(clf_cool == pop) & reliable_cool]
	if len(ages):
		print(f"  {pop}: N={len(ages)}, median={np.median(ages):.1f} Gyr")

fig, ax = plt.subplots(figsize=(8, 5))
for pop, color in pop_colors.items():
	ages = coolwd["tot_age"].values[ok_cool][(clf_cool == pop) & reliable_cool]
	if len(ages):
		ax.hist(ages, bins=bins_age, histtype="step", lw=2, color=color,
		        label=rf"{pop_labels[pop]} ($N={len(ages)}$)")
ax.set_xlabel("Age [Gyr]")
ax.set_ylabel("$N$")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / f"coolwd_pop_ages{rv_suffix}.pdf")
plt.close()
print(f"Saved {OUT_DIR / f'coolwd_pop_ages{rv_suffix}.pdf'}")
