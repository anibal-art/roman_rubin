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
script_dir = Path(__file__).parent
# script_dir='/home/anibal-pc/microlensing/simulation_Rubin/roman_rubin/'
sys.path.append(str(script_dir)+'/roman_rubin/photutils/')
print(str(script_dir)+'/photutils/')
from bandpass import Bandpass

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

ZP = {'W149': 27.615, 'u': 27.03, 'g': 28.38, 'r': 28.16,'i': 27.85, 'z': 27.46, 'y': 26.68}


def coords(L=0.5,B = -1.25):
    gc = SkyCoord(l=L * u.degree, b=B * u.degree, frame='galactic')
    gc.icrs.dec.value
    Ra = gc.icrs.ra.value
    Dec = gc.icrs.dec.value
    return Ra, Dec

def ts_rubin(path_dataslice):
    
    LSST_BandPass = {}
    lsst_filterlist = 'ugrizy'
    for f in lsst_filterlist:
        LSST_BandPass[f] = Bandpass()
        LSST_BandPass[f].read_throughput(str(script_dir)+'/troughputs/' + f'total_{f}.dat')
    dataSlice = np.load(path_dataslice, allow_pickle=True)
    rubin_ts = {}
    
    for fil in lsst_filterlist:
        m5 = dataSlice['fiveSigmaDepth'][np.where(dataSlice['filter'] == fil)]
        mjd = dataSlice['observationStartMJD'][np.where(dataSlice['filter'] == fil)] + 2400000.5
        int_array = np.column_stack((mjd, m5, m5)).astype(float)
        rubin_ts[fil] = int_array
    return rubin_ts


def roman_telescope(path_ephemerides):
    tstart_Roman = 2461508.763828608  # tlsst + 3*365 #Roman is expected to be launch in may 2027

    nominal_seasons = [
        {'start': '2027-02-11T00:00:00', 'end': '2027-04-24T00:00:00'},
        {'start': '2027-08-16T00:00:00', 'end': '2027-10-27T00:00:00'},
        {'start': '2028-02-11T00:00:00', 'end': '2028-04-24T00:00:00'},
        {'start': '2030-02-11T00:00:00', 'end': '2030-04-24T00:00:00'},
        {'start': '2030-08-16T00:00:00', 'end': '2030-10-27T00:00:00'},
        {'start': '2031-02-11T00:00:00', 'end': '2031-04-24T00:00:00'},
    ]
    Roman_tot = simulator.simulate_a_telescope(name='W149',
                                               time_start=tstart_Roman + 107 + 72 * 5 + 113 * 2 + 838.36 + 107,
                                               time_end=tstart_Roman + 107 + 72 * 5 + 113 * 2 + 838.36 + 107 + 72,
                                               sampling=0.25,
                                               location='Space', camera_filter='W149', uniform_sampling=True,
                                               astrometry=False)
    lightcurve_fluxes = []
    for season in nominal_seasons:
        tstart = Time(season['start'], format='isot').jd
        tend = Time(season['end'], format='isot').jd
        Roman = simulator.simulate_a_telescope(name='W149',
                                               time_start=tstart,
                                               time_end=tend,
                                               sampling=0.25,
                                               location='Space',
                                               camera_filter='W149',
                                               uniform_sampling=True,
                                               astrometry=False)
        lightcurve_fluxes.append(Roman.lightcurve)
    # Combine all the lightcurve_flux tables into one array
    combined_array = np.concatenate([lc.as_array() for lc in lightcurve_fluxes])
    
    # Convert the combined array back into a QTable
    new_table = QTable(combined_array,names=['time','mag','err_mag', 'flux', 'err_flux','inv_err_flux'], 
                       units=['JD', 'mag','mag','W/m^2', 'W/m^2','m^2/W'])
    # display(new_table)
    Roman_tot.lightcurve = new_table
    ephemerides = np.load(path_ephemerides)
    Roman_tot.location = 'Space'
    Roman_tot.spacecraft_name = 'L2'
    Roman_tot.spacecraft_positions = {'astrometry': [], 'photometry': ephemerides}
    return Roman_tot


def rubin_telescope(rubin_ts):
    lsst_filterlist = 'ugrizy'
    dict_tels = {}
    for band in rubin_ts:
        dict_tels[band] = telescopes.Telescope(name=band, camera_filter=band, location='Earth',
                                              lightcurve=rubin_ts[band],
                                              lightcurve_names=['time', 'mag', 'err_mag'],
                                              lightcurve_units=['JD', 'mag', 'mag'])
    return dict_tels


def Event_rubin_dp0(name,ra, dec, ts_dict, bands = "ugrizy"):
    '''
    This function creates an Event from pyLIMA
    
    ts_dict (dict): keys 'u','g','r','i','z','y'
    ra (float): equatorial coordinate ra (in degrees)
    dec (float): galactic coordinatedec (in degrees)
   
    '''
    # Ra, Dec = coords(l,b)
    my_own_creation = event.Event(ra=ra, dec=dec)
    my_own_creation.name = name
    # lsst_filterlist = 'ugrizy'
    rubin_ts = {}
    for band in bands:
        if band in ts_dict.keys():
            mjd = ts_dict[band]
            m5 = np.ones(len(mjd))*20
            int_array = np.column_stack((mjd, m5, m5)).astype(float)
            rubin_ts[band] = int_array
        
    for band in bands:
        if band in ts_dict.keys():
            my_own_creation.telescopes.append(rubin_telescope(rubin_ts)[band])

    return my_own_creation


def Event_roman_rubin(path_ephemerides, path_dataslice):
    '''
    :param opsim:
    :return:
    '''
    
    Ra, Dec = coords()
    rubin_ts = ts_rubin(path_dataslice)
    
    tlsst = 60413.26382860778 + 2400000.5

    my_own_creation = event.Event(ra=Ra, dec=Dec)
    my_own_creation.name = 'An event observed by Roman and Rubin'

    Roman_tot = roman_telescope(path_ephemerides)
    my_own_creation.telescopes.append(Roman_tot)
    lsst_filterlist = 'ugrizy'
    for band in lsst_filterlist:
        rubin_telescope(rubin_ts)
        my_own_creation.telescopes.append(rubin_telescope(rubin_ts)[band])

    return my_own_creation#, dataSlice, LSST_BandPass


def sim_lightcurve(i, data, event, model, parallax):
    '''
    i (int): index of the TRILEGAL data set
    data (dictionary): parameters including magnitude of the stars
    path_ephemerides (str): path to the ephemeris of Gaia
    path_dataslice(str): path to the dataslice obtained from OpSims
    model(str): model desired
    '''
    magstar = {'W149': data["W149"], 'u': data["u"], 'g': data["g"], 'r': data["r"],
               'i': data["i"], 'z': data["z"], 'y': data["Y"]}
    # ZP = {'W149': 27.615, 'u': 27.03, 'g': 28.38, 'r': 28.16,
    #       'i': 27.85, 'z': 27.46, 'y': 26.68}

    
    new_creation = copy.deepcopy(event)
    np.random.seed(i)
    t0 = data['t0']
    tE = data['te']

    if model == 'USBL':
        print(data["t0"])
        params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['te'], 'rho': data['rho'],
                  's': data['s'], 'q': data['q'], 'alpha': data['alpha'],
                  'piEN': data['piEN'], 'piEE': data['piEE']}
        choice = np.random.choice(["central_caustic", "second_caustic", "third_caustic"])
        # usbl = pyLIMA.models.USBL_model.USBLmodel(roman_event, origin=[choice, [0, 0]],blend_flux_parameter='ftotal')
        
        my_own_model = USBL_model.USBLmodel(new_creation, origin=[choice, [0, 0]],
                                            blend_flux_parameter='ftotal',
                                            parallax=['Full', t0] if parallax else ['None', 0.0])
        print(my_own_model.origin)
        # my_own_model = USBL_model.USBLmodel(new_creation,origin=[choice, [0, 0]], parallax=['Full', t0])
    elif model == 'FSPL':
        params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['te'],
                  'rho': data['rho'], 'piEN': data['piEN'],
                  'piEE': data['piEE']}
        my_own_model = FSPLarge_model.FSPLargemodel(new_creation, parallax=['Full', t0] if parallax else ['None', 0.0])
    elif model == 'PSPL':
        params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['te'],
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
        g = 0 #np.random.uniform(0, 1)
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
    simulator.simulate_lightcurve(my_own_model, pyLIMA_parameters)

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

    for k in range(len(my_own_model.event.telescopes)):
        
        model_flux = my_own_model.compute_the_microlensing_model(my_own_model.event.telescopes[k],
                                                                 pyLIMA_parameters)['photometry']
        my_own_model.event.telescopes[k].lightcurve['flux'] = model_flux
        my_own_model.event.telescopes[k].lightcurve['mag'] = mag(ZP[my_own_model.event.telescopes[k].name], model_flux)

    return my_own_model


def model_lightcurves_RR(my_own_model, pyLIMA_parameters):

    for k in range(1,len(my_own_model.event.telescopes)):
        
        model_flux = my_own_model.compute_the_microlensing_model(my_own_model.event.telescopes[k],
                                                                 pyLIMA_parameters)['photometry']
        my_own_model.event.telescopes[k].lightcurve['flux'] = model_flux
        # my_own_model.event.telescopes[k].lightcurve['mag'] = mag(ZP[my_own_model.event.telescopes[k].name], model_flux)
    return my_own_model

def dp0_pyLIMA(name, trilegal_idx, ra, dec, dict_bands_jd, event_params, model = "USBL", parallax = True):
    my_own_creation = Event_rubin_dp0(name, ra, dec, dict_bands_jd)
    model, pyLIMA_parameters = sim_lightcurve(trilegal_idx, event_params, my_own_creation, model, parallax)
    perfect_model = model_lightcurves_dp0(model, pyLIMA_parameters)
    return perfect_model
    # return perfect_model.event.telescopes[0].lightcurve['mag']


def init_pyLIMA_model(model, pylima_event, parallax = None, seed=42):
    parallax = ['Full', parallax] if parallax is not None else ['None', 0.0]
    np.random.seed(seed)
    if model == 'USBL':
        choice = np.random.choice(["central_caustic", "second_caustic", "third_caustic"])
        pylima_model = USBL_model.USBLmodel(pylima_event, origin=[choice, [0, 0]],
                                            blend_flux_parameter='ftotal',
                                            parallax=parallax) 
    elif model == 'FSPL':
        pylima_model = FSPLarge_model.FSPLargemodel(pylima_event, parallax=parallax)
    elif model == 'PSPL':
        pylima_model = PSPL_model.PSPLmodel(pylima_event, parallax=parallax)
    return pylima_model

def observed_pyLIMA_event_mags(pylima_model, pylima_event, mag_baseline, params_ulens, seed=42):
    pylima_params_list = []
    for param_name in params_ulens:
        pylima_params_list.append(params_ulens[param_name])
    pylima_flux_params = []
    fs, G, F = {}, {}, {}
    np.random.seed(seed)
    for i in range(len(pylima_event.telescopes)):
        band = pylima_event.telescopes[i].name
        flux_baseline = 10 ** ((ZP[band] - mag_baseline[band]) / 2.5)
        g = 0  # np.random.uniform(0, 1)
        f_source = flux_baseline / (1 + g)
        fs[band] = f_source
        G[band] = g
        F[band] = f_source + g * f_source  # flux_baseline
        f_total = f_source * (1 + g)
        if pylima_model.blend_flux_parameter == "ftotal":
            pylima_flux_params.append(f_source)
            pylima_flux_params.append(f_total)
        else:
            pylima_flux_params.append(f_source)
            pylima_flux_params.append(f_source * g)
    
    pylima_params_list += pylima_flux_params
    pylima_params = pylima_model.compute_pyLIMA_parameters(pylima_params_list)
    print(pylima_params_list)
    simulator.simulate_lightcurve(pylima_model, pylima_params)
    
    for k in range(len(pylima_model.event.telescopes)):
        model_flux = pylima_model.compute_the_microlensing_model(pylima_model.event.telescopes[k],
                                                                 pylima_params)['photometry']
        pylima_model.event.telescopes[k].lightcurve['flux'] = model_flux
        pylima_model.event.telescopes[k].lightcurve['mag'] = mag(ZP[pylima_model.event.telescopes[k].name], model_flux)
    
    mags = {}
    mags_err = {}
    for telesc in pylima_model.event.telescopes:
        mags[telesc.filter] = telesc.lightcurve["mag"].value
    return mags
