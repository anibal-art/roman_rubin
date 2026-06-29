from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA.toolbox import time_series
from pyLIMA.simulations import simulator
from pyLIMA.models import PSBL_model
from pyLIMA.models import USBL_model
from pyLIMA.models import FSPLarge_model
from pyLIMA.models import PSPL_model
from pyLIMA.fits import TRF_fit
from pyLIMA.fits import DE_fit
from pyLIMA.fits import MCMC_fit
from pyLIMA.outputs import pyLIMA_plots
from pyLIMA.outputs import file_outputs
import numpy as np
import multiprocessing as mul

def event_creation(Source, path_ephemerides, wfirst_lc, lsst_u, lsst_g,
                    lsst_r, lsst_i, lsst_z,
                    lsst_y):


    tlsst = 60350.38482057137 + 2400000.5
    RA, DEC = 267.92497054815516, -29.152232510353276
    e = event.Event(ra=RA, dec=DEC)

    if len(lsst_u) + len(lsst_g) + len(lsst_r) + len(lsst_i) + len(lsst_z) + len(lsst_y) == 0:
        e.name = 'Event_Roman_' + str(Source)
        name_roman = 'Roman' 
    else:
        e.name = 'Event_RR_' + str(Source)
        name_roman = 'Roman'
    
    tel_list = []
    if len(wfirst_lc) != 0:
    # Add a PyLIMA telescope object to the event with the Gaia lightcurve
        tel1 = telescopes.Telescope(name = name_roman, camera_filter='W149',
                                    lightcurve = wfirst_lc,
                                    lightcurve_names=['time', 'mag', 'err_mag'],
                                    lightcurve_units=['JD', 'mag', 'mag'],
                                    location='Space')
    
        tel1.spacecraft_positions = {'astrometry': [], 'photometry': np.load(path_ephemerides)}
        e.telescopes.append(tel1)
        tel_list.append('Roman')

    lsst_lc_list = [lsst_u, lsst_g, lsst_r, lsst_i, lsst_z, lsst_y]
    lsst_bands = "ugrizy"
    for j in range(len(lsst_lc_list)):
        if not len(lsst_lc_list[j]) == 0:
            
            tel = telescopes.Telescope(name = lsst_bands[j], camera_filter=lsst_bands[j],
                                       lightcurve = lsst_lc_list[j],
                                       lightcurve_names=['time', 'mag', 'err_mag'],
                                       lightcurve_units=['JD', 'mag', 'mag'],
                                       location='Earth')
            e.telescopes.append(tel)
            tel_list.append(lsst_bands[j])
    e.check_event()
    return e
    

# def model_rubin_roman(Source, true_model, event_params, path_ephemerides, model,ORIGIN, wfirst_lc, lsst_u, lsst_g, lsst_r, lsst_i, lsst_z, lsst_y):
def fit_rubin_roman(Source, event_params, path_save, path_ephemerides, model, algo, Origin, rango, wfirst_lc, lsst_u, lsst_g,
                    lsst_r, lsst_i, lsst_z,
                    lsst_y):
    '''
    Perform fit for Rubin and Roman data for fspl, usbl and pspl
    '''
    e = event_creation(Source, path_ephemerides, wfirst_lc, lsst_u, lsst_g,
                    lsst_r, lsst_i, lsst_z,
                    lsst_y)
    # Give the model initial guess values somewhere near their actual values so that the fit doesn't take all day
    if 'USBL' in model :
        t0_str = 't_center'
        u0_str = 'u_center'
    else:
        t0_str = 't0'
        u0_str = 'u0'
    
    t0 = float(event_params[t0_str])
    u0 = float(event_params[u0_str])
    tE = float(event_params['tE'])
    
    if not 'PiE' in model:
        piEN = float(event_params['piEN'])
        piEE = float(event_params['piEE'])

    
    if model == 'FSPL':
        rho = float(event_params['rho'])
        pyLIMAmodel = FSPLarge_model.FSPLargemodel(e,blend_flux_parameter='ftotal', parallax=['Full', t0])
        param_guess = [t0, u0, tE, rho, piEN, piEE]
        
    elif 'USBL' in model :
        rho = float(event_params['rho'])
        s = float(event_params['separation'])
        q = float(event_params['mass_ratio'])
        alpha = float(event_params['alpha'])
        # pyLIMAmodel = USBL_model.USBLmodel(e, blend_flux_parameter='ftotal', parallax=['Full', t0])
        if not 'PiE' in model:
            param_guess = [t0, u0, tE, rho, s, q, alpha, piEN, piEE]    
            print('Here')
            pyLIMAmodel = USBL_model.USBLmodel(e,
                                   blend_flux_parameter='ftotal',origin=Origin,
                                   parallax=['Full', t0])

        else:
            param_guess = [t0, u0, tE, rho, s, q, alpha]
            # print('Here!!')
            pyLIMAmodel = USBL_model.USBLmodel(e,
                       blend_flux_parameter='ftotal',origin=Origin)

    elif 'PSPL' in model:
        if not 'PiE' in model:
            pyLIMAmodel = PSPL_model.PSPLmodel(e, blend_flux_parameter='ftotal', parallax=['Full', t0])
            param_guess = [t0, u0, tE, piEN, piEE]
        else:
            pyLIMAmodel = PSPL_model.PSPLmodel(e, blend_flux_parameter='ftotal')
            param_guess = [t0, u0, tE]

    if algo == 'TRF':
        fit_2 = TRF_fit.TRFfit(pyLIMAmodel)
        pool = None
        
    elif algo == 'MCMC':
        fit_2 = MCMC_fit.MCMCfit(pyLIMAmodel, MCMC_links=7000)
        pool = mul.Pool(processes=36)
        
    elif algo == 'DE':
        pool = mul.Pool(processes=16)
        fit_2 = DE_fit.DEfit(pyLIMAmodel, telescopes_fluxes_method='polyfit', DE_population_size=20,
                             max_iteration=10000,
                             display_progress=True)

    

    if not rango == 0:
        rango=1
        if 'USBL' in model:
            fit_2.fit_parameters['separation'][1] = [s - rango * abs(s), s + rango * abs(s)]
            fit_2.fit_parameters['mass_ratio'][1] = [q - rango * abs(q), q + rango * abs(q)]
            if alpha ==0:
                fit_2.fit_parameters['alpha'][1] = [0,2*np.pi]
            else:
                fit_2.fit_parameters['alpha'][1] = [alpha - rango * abs(alpha), alpha + rango * abs(alpha)]

        if ('USBL' in model) or ('FSPL' in model):
            if (rho - rango * abs(rho))<0:
                fit_2.fit_parameters['rho'][1] = [0, rho + rango * abs(rho)]
            else:
                fit_2.fit_parameters['rho'][1] = [rho - rango * abs(rho), rho + rango * abs(rho)]
                
        if not 'PiE' in model:
            fit_2.fit_parameters['piEE'][1] = [piEE - rango * abs(piEE),
                                               piEE + rango * abs(piEE)]  # parallax vector parameter boundaries
            
            fit_2.fit_parameters['piEN'][1] = [piEN - rango * abs(piEN),
                                                   piEN + rango * abs(piEN)]  # parallax vector parameter boundaries
    
        fit_2.fit_parameters[t0_str][1] = [t0 - 10, t0 + 10]  # t0 limits
        fit_2.fit_parameters[u0_str][1] = [u0 - abs(u0) * rango, u0 + abs(u0) * rango]  # u0 limits
        fit_2.fit_parameters['tE'][1] = [tE - abs(tE) * rango, tE + abs(tE) * rango]  # tE limits in days
    else:
        if 'USBL' in model:
            if s < 1:
                fit_2.fit_parameters['separation'][1] = [0,1]
            elif s > 1:
                fit_2.fit_parameters['separation'][1] = [1,2*s]
            else: 
                fit_2.fit_parameters['separation'][1] = [0.5,1.5]

            fit_2.fit_parameters['mass_ratio'][1] = [1e-10,1]
            fit_2.fit_parameters['alpha'][1] = [0, 2*np.pi]
        if ('USBL' in model) or ('FSPL' in model):
            if (rho - rango * abs(rho))<0:
                fit_2.fit_parameters['rho'][1] = [0, rho + 2 * abs(rho)]
            else:
                fit_2.fit_parameters['rho'][1] = [rho - 2 * abs(rho), rho + 2 * abs(rho)]
        # rango = 0
        if not 'PiE' in model:
            fit_2.fit_parameters['piEE'][1] = [piEE - 100 * abs(piEE),
                                               piEE + 100 * abs(piEE)]  # parallax vector parameter boundaries
            fit_2.fit_parameters['piEN'][1] = [piEN - 100 * abs(piEN),
                                                   piEN + 100 * abs(piEN)]  # parallax vector parameter boundaries
        
        fit_2.fit_parameters[t0_str][1] = [t0 - 100, t0 + 100]  # t0 limits
        fit_2.fit_parameters[u0_str][1] = [-5, 5]  # u0 limits
        fit_2.fit_parameters['tE'][1] = [0,  tE * 5]  # tE limits in days




    for p in range(len(param_guess)):
        par = list(fit_2.fit_parameters.keys())[p]
        interval = fit_2.fit_parameters[par][1]
        param_guess[p] = np.random.uniform(*interval)
        
    
    fit_2.model_parameters_guess = param_guess

    
    if algo == "MCMC" or algo =='DE' :
        fit_2.fit(computational_pool=pool)
    else:
        fit_2.fit()

    true_values = np.array(event_params)
    fit_2.fit_results['true_params'] = event_params
    fit_2.fit_results['rango'] = rango
    fit_2.fit_results['method'] = algo
    fit_2.fit_results['name'] = e.name
    fit_2.fit_results['ln_likelihood'] = fit_2.likelihood_photometry(fit_2.fit_results['best_model'])
    # print(fit_2.fit_results['best_model'])
    np.save(path_save + e.name + '_' + algo +'.npy', fit_2.fit_results)
    return fit_2, e, pyLIMAmodel



def model_rubin_roman(Source, event_params, path_ephemerides, model,ORIGIN, wfirst_lc, lsst_u, lsst_g, lsst_r, lsst_i, lsst_z, lsst_y):
    '''
    Perform fit for Rubin and Roman data for fspl, usbl and pspl
    '''
    # tlsst = 60350.38482057137 + 2400000.5
    RA, DEC = 267.92497054815516, -29.152232510353276
    e = event.Event(ra=RA, dec=DEC)

    if len(lsst_u) + len(lsst_g) + len(lsst_r) + len(lsst_i) + len(lsst_z) + len(lsst_y) == 0:
        e.name = 'Event_Roman_' + str(int(Source))
        name_roman = 'W149'
    else:
        e.name = 'Event_RR_' + str(int(Source))
        name_roman = 'W149'
    tel_list = []
    # Add a PyLIMA telescope object to the event with the Gaia lightcurve
    tel1 = telescopes.Telescope(name=name_roman, camera_filter='W149',
                                lightcurve=wfirst_lc,
                                lightcurve_names=['time', 'mag', 'err_mag'],
                                lightcurve_units=['JD', 'mag', 'mag'],
                                location='Space')
    ephemerides = np.load(path_ephemerides)
    tel1.spacecraft_positions = {'astrometry': [], 'photometry': ephemerides}
    e.telescopes.append(tel1)
    tel_list.append('Roman')
    lsst_lc_list = [lsst_u, lsst_g, lsst_r, lsst_i, lsst_z, lsst_y]
    lsst_bands = "ugrizy"
    for j in range(len(lsst_lc_list)):
        if len(lsst_lc_list[j]) != 0:
            tel = telescopes.Telescope(name=lsst_bands[j], camera_filter=lsst_bands[j],
                                       lightcurve=lsst_lc_list[j],
                                       lightcurve_names=['time', 'mag', 'err_mag'],
                                       lightcurve_units=['JD', 'mag', 'mag'],
                                       location='Earth')
            e.telescopes.append(tel)
            tel_list.append(lsst_bands[j])
    e.check_event()
    # Use t_center if available; otherwise, use t0
    t_guess = float(event_params['t_center']) if 't_center' in event_params else float(event_params.get('t0', None))
    # Check if model is specified and create the appropriate model instance
    if model == 'FSPL':
        pyLIMAmodel = FSPLarge_model.FSPLargemodel(e, parallax=['Full', t_guess])
    elif model == 'USBL':
        pyLIMAmodel = USBL_model.USBLmodel(e, origin=ORIGIN,
                                           blend_flux_parameter='ftotal',
                                           parallax=['Full', t_guess])
        # else:
        #     pyLIMAmodel = USBL_model.USBLmodel(e, origin=ORIGIN,
        #                                        blend_flux_parameter='ftotal',
        #                                        parallax=['Full', t_guess])

    elif model == 'USBL_NoPiE':
        pyLIMAmodel = USBL_model.USBLmodel(e, origin=ORIGIN, blend_flux_parameter='ftotal')
    
    elif model == 'PSPL':
        pyLIMAmodel = PSPL_model.PSPLmodel(e,parallax=['Full', t_guess], blend_flux_parameter='ftotal')

    return pyLIMAmodel
