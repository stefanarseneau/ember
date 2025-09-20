import numpy as np
import interpolator

mass_sun, radius_sun, newton_G, speed_light = 1.9884e30, 6.957e8, 6.674e-11, 299792458

def leastsq_likelihood(params, fl, e_fl, plx, interp, logg_function):
    """likelihood function for least squares fitting"""
    pc_to_m, radius_sun = 3.08568e16, 6.957e8
    teff, logg = params.valuesdict().values()
    theta = np.array([teff, logg, 1000 / plx, 0])
    flux_model = get_model_flux(theta, interp, logg_function=logg_function)
    flux_model *= 1e23
    if ~np.any(np.isnan(flux_model)):
        return (fl - flux_model) / e_fl
    else:
        return (fl - flux_model) / e_fl
    
def get_model_flux(theta : np.array, interpolator : interpolator.atmos.WarwickPhotometry, logg_function = None) -> np.array:
    """get model photometric flux for a WD with a given radius, located a given distance away
    """     
    mass_sun, radius_sun, newton_G, speed_light = 1.9884e30, 6.957e8, 6.674e-11, 299792458
    if logg_function == None:
        # if no logg function is provided, assume mass is provided and calculate
        teff, radius, distance, av, mass = theta
        logg = np.log10(100*(newton_G * mass_sun * mass) / (radius * radius_sun)**2)
    else:
        # if logg function is provided, use it
        teff, logg, distance, av = theta
        radius = logg_function(teff, logg)
    #try:
    fl = 4 * np.pi * interpolator(teff, logg, av = av) # flux in physical units
    #except TypeError:
    #    fl = 4 * np.pi * interpolator(teff, logg)
    #convert to SI units
    pc_to_m, radius_sun = 3.086775e16, 6.957e8
    radius *= radius_sun # Rsun to meter
    distance *= pc_to_m # Parsec to meter
    return (radius / distance)**2 * fl

def mcmc_likelihood(theta, fl, e_fl, gflux, plx_prior, av_prior, likelihood, ext_vector, logg_function = None, vg_prior = None):
    """mcmc likelihood function for parameter inference"""
    # unpack the priors
    if logg_function == None:
        # if we have no logg function, fit on everything
        teff, radius, distance, av, mass = theta
        vg_th = (newton_G * mass * mass_sun) / (speed_light * radius * radius_sun) * 1e-3
        logg = np.log10(100*(newton_G * mass_sun * mass) / (radius * radius_sun)**2)
        bounds = np.array([[1000, 50000], [0.0035, 0.05], [10, 2000], [1e-4, 1], [0.1, 1.4]])
    else:
        # if we have a logg function, infer the mass and radius from the logg function
        teff, radius, distance, av = theta
        logg = logg_function(teff, radius)
        mass = (1e-2*np.power(10, logg)) * (radius * radius_sun)**2 / (newton_G * mass_sun)
        vg_th = (newton_G * mass * mass_sun) / (speed_light * radius * radius_sun) * 1e-3
        bounds = np.array([[1000, 50000], [0.0035, 0.05], [10, 2000], [1e-4, 1]])
    # compute likelihoods
    uniform = likelihood.uniform_prior(theta, bounds)
    if np.isinf(uniform):
        return -np.inf
    theta = np.array([teff, radius, distance, 0, mass]) if logg_function == None else np.array([teff, radius, distance, 0])
    ll  = likelihood.ll(theta, fl*10**(0.4*ext_vector*av), e_fl, logg_function = logg_function, use_jy = True)
    #flux_model = get_model_flux(theta, interpolator=likelihood.interp, logg_function=logg_function)
    #ll = -0.5 * np.sum((fl*10**(0.4*ext_vector*av) - flux_model)**2 / e_fl**2 + np.log(2 * np.pi * e_fl**2))
    distance_lp = likelihood.gaussian_prior(1000/distance, plx_prior[0], plx_prior[1]) + 2*np.log(distance)
    av_lp = likelihood.gaussian_prior(av, av_prior[0], av_prior[1])
    gravz_lp = likelihood.gaussian_prior(vg_th, vg_prior[0], vg_prior[1]) if vg_prior is not None else 0
    log_likelihood = ll + distance_lp + av_lp + uniform + gravz_lp
    return log_likelihood if ~np.isnan(log_likelihood) else -np.inf