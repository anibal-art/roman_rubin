import os
parent = os.getcwd()
# print("Parent directory:", parent)
import numpy as np
import pandas as pd
import os, sys
from pyLIMA import event
from scipy.signal import find_peaks
from tqdm.auto import tqdm
from pyLIMA import telescopes
from pyLIMA.models import USBL_model
from astropy.time import Time
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.table import QTable
from astropy.time import Time
from astropy.coordinates import SkyCoord

import copy
from pathlib import Path
# script_dir = Path(__file__).parent
import sys
script_dir=parent
sys.path.append(script_dir)
sys.path.append(script_dir+'/photutils/')


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

from ulens_params import event_param


ZP = {'W149':27.615, 'u':27.03, 'g':28.38, 'r':28.16,
          'i':27.85, 'z':27.46, 'y':26.68}

def rubin_telescope(rubin_ts):
    lsst_filterlist = 'ugrizy'
    dict_tels = {}
    for band in rubin_ts:
        dict_tels[band] = telescopes.Telescope(name=band, camera_filter=band, location='Earth',
                                              lightcurve=rubin_ts[band],
                                              lightcurve_names=['time', 'mag', 'err_mag'],
                                              lightcurve_units=['JD', 'mag', 'mag'])
    return dict_tels


def Event_rubin_dp0(name,ra,dec, ts_dict):
    '''
    This function creates an Event from pyLIMA
    
    ts_dict (dict): keys 'u','g','r','i','z','y'
    l (float): galactic coordinate l
    b (float): galactic coordinate b
    '''
    my_own_creation = event.Event(ra=ra, dec=dec)
    my_own_creation.name = name
    lsst_filterlist = 'ugrizy'
    rubin_ts = {}
    for band in lsst_filterlist:
        if band in ts_dict.keys():
            mjd = ts_dict[band]
            m5 = np.ones(len(mjd))*20
            int_array = np.column_stack((mjd, m5, m5)).astype(float)
            rubin_ts[band] = int_array
        
    for band in lsst_filterlist:
        if band in ts_dict.keys():
            my_own_creation.telescopes.append(rubin_telescope(rubin_ts)[band])

    return my_own_creation


def sim_lightcurve(i, data, event, model, parallax, g=0):
    '''
    i (int): index of the TRILEGAL data set
    data (dictionary): parameters including magnitude of the stars
    path_ephemerides (str): path to the ephemeris of Gaia
    path_dataslice(str): path to the dataslice obtained from OpSims
    model(str): model desired
    g (float): blended fraction
    '''
    ZP = {'W149': 27.615, 'u': 27.03, 'g': 28.38, 'r': 28.16,
          'i': 27.85, 'z': 27.46, 'y': 26.68}
    magstar = {band: data[band] for band in ZP.keys() if band in data.keys()} 
    # adds to magstar only bands that are present on data

    new_creation = copy.deepcopy(event)
    np.random.seed(i)
    t0 = data['t0']
    tE = data['tE']
    if model == 'USBL':
        params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['tE'], 'rho': data['rho'],
                  's': data['s'], 'q': data['q'], 'alpha': data['alpha'],
                  'piEN': data['piEN'], 'piEE': data['piEE']}
        choice = np.random.choice(["central_caustic", "second_caustic", "third_caustic"])
        # usbl = pyLIMA.models.USBL_model.USBLmodel(roman_event, origin=[choice, [0, 0]],blend_flux_parameter='ftotal')
        my_own_model = USBL_model.USBLmodel(new_creation, origin=[choice, [0, 0]],
                                            blend_flux_parameter='ftotal',
                                            parallax=['Full', t0] if parallax else ['None', 0.0])
        # print(my_own_model.origin)
        # my_own_model = USBL_model.USBLmodel(new_creation,origin=[choice, [0, 0]], parallax=['Full', t0])
    elif model == 'FSPL':
        params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['tE'],
                  'rho': data['rho'], 'piEN': data['piEN'],
                  'piEE': data['piEE']}
        my_own_model = FSPLarge_model.FSPLargemodel(new_creation, parallax=['Full', t0] if parallax else ['None', 0.0])
    elif model == 'PSPL':
        params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['tE'],
                  'piEN': data['piEN'], 'piEE': data['piEE']}
        my_own_model = PSPL_model.PSPLmodel(new_creation, parallax=['Full', t0] if parallax else ['None', 0.0])

    my_own_parameters = []
    for key in params:
        my_own_parameters.append(params[key])

    my_own_flux_parameters = []
    
    fs, G, F = {}, {}, {}
    np.random.seed(i)
    for i in range(len(new_creation.telescopes)):
        band = new_creation.telescopes[i].name
        flux_baseline = 10 ** ((ZP[band] - magstar[band]) / 2.5)
        # g = 0
        something = np.random.uniform(0, 1) # only to maintain the reproducibility
        f_source = flux_baseline / (1 + g)
        fs[band] = f_source
        G[band] = g
        F[band] = f_source + g * f_source  # flux_baseline
        f_total = f_source * (1 + g)
        if my_own_model.blend_flux_parameter == "ftotal":
            my_own_flux_parameters.append(f_source)
            my_own_flux_parameters.append(f_total)
        else:
            my_own_flux_parameters.append(f_source)
            my_own_flux_parameters.append(f_source * g)
    
    my_own_parameters += my_own_flux_parameters
    pyLIMA_parameters = my_own_model.compute_pyLIMA_parameters(my_own_parameters)
    simulator.simulate_lightcurve(my_own_model, pyLIMA_parameters, add_noise=False)

    return my_own_model, pyLIMA_parameters


def mag(zp, Flux):
    '''
    Transform the flux to magnitude
    inputs
    zp: zero point
    Flux: vector that contains the lightcurve flux
    '''
    return zp - 2.5 * np.log10(abs(Flux))


def model_lightcurves_dp0(my_own_model, pyLIMA_parameters):

    for telescope in my_own_model.event.telescopes:
        magnification = my_own_model.model_magnification(telescope,
                        pyLIMA_parameters)
        model_flux = my_own_model.compute_the_microlensing_model(telescope,
                     pyLIMA_parameters)['photometry']
        telescope.lightcurve["magnification"] = magnification
        telescope.lightcurve['flux'] = model_flux
        telescope.lightcurve['mag'] = mag(ZP[telescope.name], model_flux)
        

    return my_own_model


def dp0_pyLIMA(name, event_id,  ra, dec, model, event_params, epochs, parallax, g=0):
    pylima_event = Event_rubin_dp0(name, ra, dec, epochs)
    model, pyLIMA_parameters = sim_lightcurve(event_id, event_params, pylima_event, model, parallax, g=g)
    perfect_model = model_lightcurves_dp0(model, pyLIMA_parameters)
    return perfect_model, pyLIMA_parameters

    





