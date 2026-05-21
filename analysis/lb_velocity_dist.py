"""Galactic velocity distribution and AVR-based age distribution inference for WD+MS sample."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent))
from load_data import DATA_DIR, OUT_DIR, setup_matplotlib, avr

plt, _ = setup_matplotlib()


# ── Load data and compute Galactic velocities ──────────────────────────────
data = pd.read_csv(DATA_DIR / "tyler/WDMS_total_ages_correct_models_cut_down.csv")

coords = SkyCoord(
    ra=data["ra"].values * u.deg,
    dec=data["dec"].values * u.deg,
    pm_ra_cosdec=data["pmra"].values * u.mas / u.yr,
    pm_dec=data["pmdec"].values * u.mas / u.yr,
    frame="icrs",
)
gal       = coords.galactic
pm_l_cosb = gal.pm_l_cosb.to(u.mas / u.yr).value
pm_b      = gal.pm_b.to(u.mas / u.yr).value

d_kpc     = 1.0 / data["wtd_par"].values
v_l_helio = 4.74047 * pm_l_cosb * d_kpc
v_b_helio = 4.74047 * pm_b      * d_kpc

U_sun, V_sun, W_sun = 11.0, 12.0, 7.0  # Schönrich et al. 2010
l_rad = np.radians(data["l"].values)
b_rad = np.radians(data["b"].values)

v_l_sun = -U_sun * np.sin(l_rad) + V_sun * np.cos(l_rad)
v_b_sun = (
    -U_sun * np.cos(l_rad) * np.sin(b_rad)
    - V_sun * np.sin(l_rad) * np.sin(b_rad)
    + W_sun * np.cos(b_rad)
)

data["v_l"] = v_l_helio + v_l_sun
data["v_b"] = v_b_helio + v_b_sun

# ── Sample selection ───────────────────────────────────────────────────────
data["age_frac_err"] = (
    np.maximum(
        data["tot_age_error_upper"].fillna(np.inf),
        data["tot_age_error_lower"].fillna(np.inf),
    )
    / data["tot_age"].replace(0, np.nan)
)
mask = (
    (data["age_frac_err"] < 0.20)
    & (data["Mass"] > 0.63)
    & (data["spectype"] != "0.0")
    & data["tot_age"].notna() & (data["tot_age"] > 0)
    & data["wtd_par"].notna() & (data["wtd_par"] > 0)
)
wds = data[mask].copy().reset_index(drop=True)
wds = wds[
    wds.phot_bp_rp_excess_factor
    > 3 * (0.0059898 + 8.817481e-12 * wds.phot_g_mean_mag**7.618399)
].reset_index(drop=True)

print(f"N = {len(wds)}")
print(f"sigma_l = {wds['v_l'].std():.1f} km/s")
print(f"sigma_b = {wds['v_b'].std():.1f} km/s")

# ── Poisson errors ─────────────────────────────────────────────────────────
p_errors = pd.read_csv(DATA_DIR / "poisson_errors.csv")
p_err_low = interp1d(
    np.concatenate([p_errors["N"], [1000]]),
    np.concatenate([p_errors["e_Poisson_lower"], [np.sqrt(1000)]]),
)
p_err_up = interp1d(
    np.concatenate([p_errors["N"], [1000]]),
    np.concatenate([p_errors["e_Poisson_upper"], [np.sqrt(1000)]]),
)




# ── UVW → (l, b, r) rotation matrices for the wds sample ──────────────────
l_w = np.radians(wds["l"].values)
b_w = np.radians(wds["b"].values)
N   = len(wds)
Ms   = np.zeros((N, 3, 3))
Ms_T = np.zeros((N, 3, 3))
for i in range(N):
    sl, cl = np.sin(l_w[i]), np.cos(l_w[i])
    sb, cb = np.sin(b_w[i]), np.cos(b_w[i])
    M = np.array([
        [ sl, -sb * cl, cb * cl],
        [ cl, -sb * sl, cb * sl],
        [  0,       cb,      sb],
    ])
    Ms[i]   = M
    Ms_T[i] = M.T


def expected_sigma_lb(ages, N_merg=1, fm=0.0):
    """Per-object (sigma_l, sigma_b) averaged over N_merg Monte Carlo draws.

    Optionally adds a random merger delay to a fraction fm of objects.
    """
    zero = np.zeros(len(ages))
    sigl = np.zeros(len(ages))
    sigb = np.zeros(len(ages))
    for _ in range(N_merg):
        delays = np.zeros(len(ages))
        if fm > 0:
            idx = np.where(np.random.random(len(ages)) <= fm)[0]
            delays[idx] = np.random.random(len(idx)) * 5
            delays[ages + delays < 0] = -ages[ages + delays < 0]
        su, sv, sw = avr(ages + delays)
        M_sig = np.array([[su**2, zero, zero],
                          [zero, sv**2, zero],
                          [zero, zero,  sw**2]]).T
        covs  = (Ms_T @ M_sig @ Ms)[:, :2, :2]
        sigl += np.sqrt(covs[:, 0, 0])
        sigb += np.sqrt(covs[:, 1, 1])
    return sigl / N_merg, sigb / N_merg


# ── Observed velocity histograms ───────────────────────────────────────────
bins = np.arange(0, 101, 10)
bc   = 0.5 * (bins[:-1] + bins[1:])

nl_full, _ = np.histogram(np.abs(wds["v_l"]), bins=bins)
nb_full, _ = np.histogram(np.abs(wds["v_b"]), bins=bins)


# ── Figure 1: velocity distributions with merger fraction curves ───────────
fms    = np.arange(0.0, 1.0, 0.2)
colors = ["#7570b3", "#d95f02", "tab:cyan", "tab:purple", "tab:green"]

print("\nComputing expected distributions for merger fractions...")
xs_l, ns_l, xs_b, ns_b = [], [], [], []
for fm in fms:
    print(f"  fm = {fm:.1f}")
    sigl, sigb = expected_sigma_lb(wds["tot_age"].values, N_merg=1000, fm=fm)
    x = np.linspace(0, 130, 1000)
    xs_l.append(x)
    xs_b.append(x)
    ns_l.append(stats.norm.pdf(x, 0, np.nanmedian(sigl)))
    ns_b.append(stats.norm.pdf(x, 0, np.nanmedian(sigb)))

xs_l, ns_l = np.array(xs_l), np.array(ns_l)
xs_b, ns_b = np.array(xs_b), np.array(ns_b)

fig, ax = plt.subplots(1, 2, figsize=(18, 7))

ax[0].errorbar(bc, nl_full,
               yerr=np.array([p_err_low(nl_full), p_err_up(nl_full)]),
               fmt="k.", markersize=30, elinewidth=2.5, zorder=1)
ax[1].errorbar(bc, nb_full,
               yerr=np.array([p_err_low(nb_full), p_err_up(nb_full)]),
               fmt="k.", markersize=30, elinewidth=2.5, zorder=1)

for i, fm in enumerate(fms):
    wi = np.argmin(np.abs(xs_l[i] - bc[0]))
    ax[0].plot(xs_l[i], nl_full[0] / ns_l[i][wi] * ns_l[i],
               "--", color=colors[i], lw=5, zorder=0)
    wi = np.argmin(np.abs(xs_b[i] - bc[0]))
    ax[1].plot(xs_b[i], nb_full[0] / ns_b[i][wi] * ns_b[i],
               "--", color=colors[i], lw=5, zorder=0)

ax[0].plot([-2, -1], [0, 0], "--", color=colors[0], lw=5,
           label=r"Expectation, $f_{\rm m}$ = 0.0")
ax[0].plot([-2, -1], [0, 0], "--", color=colors[1], lw=5,
           label=r"Expectation, $f_{\rm m}$ = 0.2")
ax[0].set_xlim(0, 100)
ax[0].set_xlabel(r"$v_l$ [km/s]", fontsize=30)
ax[0].set_ylabel(r"$N$", fontsize=26)
ax[0].legend(fontsize=22)
ax[1].set_xlim(0, 100)
ax[1].set_xlabel(r"$v_b$ [km/s]", fontsize=30)
ax[1].set_ylabel(r"$N$", fontsize=26)

fig.tight_layout()

ax_inset = fig.add_axes([0.73, 0.55, 0.25, 0.4])
ax_inset.hist(wds["tot_age"], 20, histtype="step", linewidth=5.5, color="k", range=(0, 1.5))
ax_inset.set_xlabel(r"Total Age [Gyr]", fontsize=24)
ax_inset.set_ylabel(r"$N$", fontsize=22)
ax_inset.set_xlim(0, 1.05)

fig.savefig(OUT_DIR / "vel_dist_mergers.pdf")
plt.close()
print(f"Saved {OUT_DIR / 'vel_dist_mergers.pdf'}")


# ── Figure 2: inferred age distribution from velocity distribution fit ─────
print("\nComputing age templates...")
age_bins = np.arange(0.25, 10.5, 0.25)  # 41 bins, 0.25–10.25 Gyr

ns_l_age = np.zeros((len(age_bins), len(bc)))
ns_b_age = np.zeros((len(age_bins), len(bc)))
for k, a in enumerate(age_bins):
    sigl_k, sigb_k = expected_sigma_lb(np.full(N, a), N_merg=100, fm=0)
    sl = np.nanmedian(sigl_k)
    sb = np.nanmedian(sigb_k)
    # CDF differences give proper probability masses per bin
    ns_l_age[k] = 2 * (stats.norm.cdf(bins[1:], 0, sl) - stats.norm.cdf(bins[:-1], 0, sl))
    ns_b_age[k] = 2 * (stats.norm.cdf(bins[1:], 0, sb) - stats.norm.cdf(bins[:-1], 0, sb))
ns_l_age /= ns_l_age.sum(axis=1, keepdims=True)
ns_b_age /= ns_b_age.sum(axis=1, keepdims=True)

comp  = np.concatenate([nl_full, nb_full])
error = p_err_up(comp)

# Weights initialised above observed floor; young bins constrained higher
lower_bounds = np.array([1, 10, 20, 20, 20, 20, 20, 20, 10, 5] + [1] * (len(age_bins) - 10))


def age_model(xs, *weights):
    w    = np.asarray(weights)
    vl_m = (ns_l_age.T * w).T.sum(axis=0)
    vb_m = (ns_b_age.T * w).T.sum(axis=0)
    return np.concatenate([
        nl_full.sum() * vl_m / vl_m.sum(),
        nb_full.sum() * vb_m / vb_m.sum(),
    ])


print("Fitting age distribution (100 runs)...")
N_runs = 100
popts  = np.zeros((N_runs, len(age_bins)))
for i in range(N_runs):
    w0 = lower_bounds + np.random.random(len(age_bins)) * (100 - lower_bounds)
    popt, _ = curve_fit(
        age_model, np.concatenate([bc, bc]), comp,
        p0=w0, sigma=error, absolute_sigma=True, bounds=(0, 400),
    )
    popts[i] = popt

fig = plt.figure(figsize=(2.5 * 6, 2.3 * 4))
gs  = fig.add_gridspec(2, 9)
ax1 = fig.add_subplot(gs[0, 1:4])
ax2 = fig.add_subplot(gs[0, 5:8])
ax3 = fig.add_subplot(gs[1, 3:6])

ax1.errorbar(bc, nl_full,
             yerr=np.array([p_err_low(nl_full), p_err_up(nl_full)]),
             fmt="k.", elinewidth=1, zorder=1)
ax2.errorbar(bc, nb_full,
             yerr=np.array([p_err_low(nb_full), p_err_up(nb_full)]),
             fmt="k.", elinewidth=1, zorder=1)

for popt in popts:
    vl_fit = (ns_l_age.T * popt).T.sum(axis=0)
    vb_fit = (ns_b_age.T * popt).T.sum(axis=0)
    ax1.plot(bc, nl_full.sum() * vl_fit / vl_fit.sum(),
             "-", color="tab:red", alpha=0.5, lw=1, zorder=0)
    ax2.plot(bc, nb_full.sum() * vb_fit / vb_fit.sum(),
             "-", color="tab:red", alpha=0.5, lw=1, zorder=0)
    ax3.plot(age_bins, popt / popt.sum() * 100, "r-", alpha=0.5, lw=1)

n_meas, _ = np.histogram(
    wds["tot_age"], bins=np.append(age_bins - 0.125, age_bins[-1] + 0.125)
)
ax3.plot(age_bins, n_meas / n_meas.sum() * 100, "k-", label="Measured Ages")

ax1.set_xlabel(r"$v_l$ [km/s]")
ax1.set_ylabel("N")
ax2.set_xlabel(r"$v_b$ [km/s]")
ax2.set_ylabel("N")
ax3.set_xlabel("Age [Gyr]")
ax3.set_ylabel(r"\% of Sample")
ax3.legend()

fig.tight_layout()
fig.savefig(OUT_DIR / "vel_fit.pdf")
plt.close()
print(f"Saved {OUT_DIR / 'vel_fit.pdf'}")
