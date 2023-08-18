from pyLIMA.models import USBL_model
from pyLIMA.fits import TRF_fit
from pyLIMA.fits import DE_fit
from pyLIMA.fits import MCMC_fit
from pyLIMA.models import PSBL_model
import multiprocessing as mul
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA.outputs import file_outputs
import pandas as pd

def fit_rubin_roman(event_params, path_save, path_ephemerides, algo, wfirst_lc, lsst_u, lsst_g, lsst_r, lsst_i, lsst_z, lsst_y):
    '''
    Perform fit for Rubin and Roman data
    '''
    tlsst = 60350.38482057137 + 2400000.5
    RA, DEC = 267.92497054815516, -29.152232510353276
    e = event.Event(ra=RA, dec=DEC)
    
    if len(lsst_u)+len(lsst_g)+len(lsst_r)+len(lsst_i)+len(lsst_z)+len(lsst_y)==0:
        e.name = 'Event_Roman_' + str(int(event_params['Source']))
    else:
        e.name = 'Event_RR_' + str(int(event_params['Source']))
    tel_list = []

    # Add a PyLIMA telescope object to the event with the Gaia lightcurve
    tel1 = telescopes.Telescope(name='Roman', camera_filter='W149',
                                light_curve=wfirst_lc,
                                light_curve_names=['time', 'mag', 'err_mag'],
                                light_curve_units=['JD', 'mag', 'mag'],
                                location='Space')

    ephemerides = np.loadtxt(path_ephemerides)
    ephemerides[:, 0] = ephemerides[:, 0]
    ephemerides[:, 3] *= 60 * 300000 / 150000000
    deltaT = tlsst - ephemerides[:, 0][0]
    ephemerides[:, 0] = ephemerides[:, 0] + deltaT
    tel1.spacecraft_positions = {'astrometry': [], 'photometry': ephemerides}
    e.telescopes.append(tel1)
    
    tel_list.append('Roman')
    
    if len(lsst_u) != 0:
        # Add a PyLIMA telescope object to the event with the LCO lightcurve
        tel2 = telescopes.Telescope(name='Rubin_u', camera_filter='u',
                                    light_curve=lsst_u,
                                    light_curve_names=['time', 'mag', 'err_mag'],
                                    light_curve_units=['JD', 'mag', 'mag'],
                                    location='Earth')
        e.telescopes.append(tel2)
        tel_list.append('Rubin_u')
    
    if len(lsst_g) != 0:
        # Add a PyLIMA telescope object to the event with the LCO lightcurve
        tel3 = telescopes.Telescope(name='Rubin_g', camera_filter='g',
                                    light_curve=lsst_g,
                                    light_curve_names=['time', 'mag', 'err_mag'],
                                    light_curve_units=['JD', 'mag', 'mag'],
                                    location='Earth')
        e.telescopes.append(tel3)
        tel_list.append('Rubin_g')
    
    if len(lsst_r) != 0:
        # Add a PyLIMA telescope object to the event with the LCO lightcurve
        tel4 = telescopes.Telescope(name='Rubin_r', camera_filter='r',
                                    light_curve=lsst_r,
                                    light_curve_names=['time', 'mag', 'err_mag'],
                                    light_curve_units=['JD', 'mag', 'mag'],
                                    location='Earth')
        e.telescopes.append(tel4)
        tel_list.append('Rubin_r')
    
    if len(lsst_i) != 0:
        # Add a PyLIMA telescope object to the event with the LCO lightcurve
        tel5 = telescopes.Telescope(name='Rubin_i', camera_filter='i',
                                    light_curve=lsst_i,
                                    light_curve_names=['time', 'mag', 'err_mag'],
                                    light_curve_units=['JD', 'mag', 'mag'],
                                    location='Earth')
        e.telescopes.append(tel5)
        tel_list.append('Rubin_i')
    
    if len(lsst_z) != 0:
        # Add a PyLIMA telescope object to the event with the LCO lightcurve
        tel6 = telescopes.Telescope(name='Rubin_z', camera_filter='z',
                                    light_curve=lsst_z,
                                    light_curve_names=['time', 'mag', 'err_mag'],
                                    light_curve_units=['JD', 'mag', 'mag'],
                                    location='Earth')
        e.telescopes.append(tel6)
        tel_list.append('Rubin_z')
    
    if len(lsst_y) != 0:
        # Add a PyLIMA telescope object to the event with the LCO lightcurve
        tel7 = telescopes.Telescope(name='Rubin_y', camera_filter='y',
                                    light_curve=lsst_y,
                                    light_curve_names=['time', 'mag', 'err_mag'],
                                    light_curve_units=['JD', 'mag', 'mag'],
                                    location='Earth')
        e.telescopes.append(tel7)
        tel_list.append('Rubin_y')
    
    e.check_event()

    # Give the model initial guess values somewhere near their actual values so that the fit doesn't take all day
    t0 = float(event_params['t0'])
    u0 = float(event_params['u0'])
    tE = float(event_params['te'])
    s = float(event_params['s'])
    q = float(event_params['q'])
    alpha = float(event_params['alpha'])
    piEN = float(event_params['piEN']),
    piEE = float(event_params['piEE'])

    psbl = PSBL_model.PSBLmodel(e, parallax=['Full', t0])

    if algo == 'TRF':
        fit_2 = TRF_fit.TRFfit(psbl)
    elif algo == 'MCMC':
        fit_2 = MCMC_fit.MCMCfit(psbl,MCMC_links=7000)
    elif algo == 'DE':
        fit_2 = DE_fit.DEfit(psbl, telescopes_fluxes_method='polyfit', DE_population_size=20, max_iteration=10000, display_progress=True)

    fit_2.model_parameters_guess = [t0, u0, tE, s, q, alpha, piEN, piEE]
    				#[t0 + 2, u0 + u0 * 0.05,
                                    #tE + 4,  log_s + 0.05 * ( log_s),
                                     #log_q + 0.05 * ( log_q), 0.05 * alpha + alpha,
                                    #piEN + 0.01 * piEN, piEE + 0.01 * piEE]
    rango = 0.1

    fit_2.fit_parameters['t0'][1] = [t0 - 10, t0 + 10]  # t0 limits
    fit_2.fit_parameters['u0'][1] = [u0 - abs(u0) * rango, u0 + abs(u0) * rango]  # u0 limits
    fit_2.fit_parameters['tE'][1] = [tE - tE * rango, tE + tE * rango]  # tE limits in days
    fit_2.fit_parameters['separation'][1] = [ s - abs( s) * rango,  s + abs( s) * rango]  # s limits
    fit_2.fit_parameters['mass_ratio'][1] = [ q - abs( q) * rango,  q + abs( q) * rango]  # q limits
    fit_2.fit_parameters['alpha'][1] = [alpha - abs(alpha) * rango, alpha + abs(alpha) * rango]  # alpha limits (in radians)
    fit_2.fit_parameters['piEE'][1] = [piEE - abs(piEE) * rango, piEE + abs(piEE) * rango] #parallax vector parameter boundaries
    fit_2.fit_parameters['piEN'][1] = [piEN - abs(piEN) * rango, piEN+ abs(piEN) * rango] #parallax vector parameter boundaries

    pool = mul.Pool(processes=16)

    fit_2.fit()
    
    true_values = np.array(event_params)
    time_fit = fit_2.fit_results['fit_time']
    best_fit = np.array(fit_2.fit_results["best_model"])
    if algo == 'TRF':
        cov = fit_2.fit_results["covariance_matrix"]
        chichi = fit_2.fit_results["chi2"]
        results = {'covariance_matrix': cov, 'best_model': best_fit, 'chi2': chichi, 'fit_time': time_fit, 'true_values': true_values}
        np.save(path_save + e.name + '_trf.npy', results)
    elif algo == 'MCMC':
        chains = fit_2.fit_results["MCMC_chains"]
        chains_fluxes = fit_2.fit_results["MCMC_chains_with_fluxes"]
        ln_likelihood = fit_2.fit_results['ln(likelihood)']
        results = {'MCMC_chains': chains, "MCMC_chains_with_fluxes": chains_fluxes, 'ln_likelihood': ln_likelihood, 'best_model': best_fit, 'fit_time': time_fit, 'true_values': true_values}
        np.save(path_save + e.name + '_mcmc.npy', results)
    elif algo == 'DE':
        de_population = fit_2.fit_results['DE_population']
        ln_likelihood = fit_2.fit_results['-(ln_likelihood)']
        results = {'DE_population': de_population, 'ln_likelihood': ln_likelihood, 'best_model': best_fit, 'fit_time': time_fit, 'true_values': true_values}
        np.save(path_save + e.name + '_de.npy', results)

    return fit_2, e, tel_list

