import numpy as np
import os, sys, re, copy, math
import pandas as pd
from pathlib import Path
# Get the directory where the script is located
script_dir = Path(__file__).parent
print(script_dir)
sys.path.append(str(script_dir)+'/photutils/')
#from bandpass import Bandpass
#from signaltonoise import calc_mag_error_m5
#from photometric_parameters import PhotometricParameters

#astropy
import astropy.units as u
from astropy.table import QTable
from astropy.time import Time
from astropy.coordinates import SkyCoord

#pyLIMA
from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA.toolbox import time_series
from pyLIMA.simulations import simulator
from pyLIMA.models import USBL_model
from pyLIMA.models import PSBL_model
from pyLIMA.models import FSPLarge_model
from pyLIMA.models import PSPL_model
from pyLIMA.fits import TRF_fit
from pyLIMA.fits import DE_fit
from pyLIMA.fits import MCMC_fit
from pyLIMA.outputs import pyLIMA_plots
from pyLIMA.outputs import file_outputs

from ulens_params import microlensing_params
import multiprocessing as mul
import h5py

def lognuniform(low=0, high=1, size=None, base=np.e):
    return np.power(base, np.random.uniform(low, high, size))
    
def fit_rubin_roman(bands_dict, pyLIMA_model, event_params):
    '''
    Perform fit for Rubin and Roman data for fspl, usbl and pspl
    '''
# Source, event_params, path_save, path_ephemerides, model, algo, Origin,rango, wfirst_lc, lsst_u, lsst_g, lsst_r, lsst_i, lsst_z,
    #                         lsst_y
    Source = self.random_seed
    
    tlsst = 60350.38482057137 + 2400000.5
    e = event.Event(ra=self.ra, dec=self.dec)
# len(lsst_u) + len(lsst_g) + len(lsst_r) + len(lsst_i) + len(lsst_z) + len(lsst_y)
    if  sum([len(bands_dict[key]) for key in 'ugrizy'])== 0:
        e.name = 'Event_Roman_' + str(int(Source))
        name_roman = 'Roman' 
    else:
        e.name = 'Event_RR_' + str(int(Source))
        name_roman = 'Roman'
    tel_list = []
    # print(len(wfirst_lc),np.size(wfirst_lc))
    
    # Add a PyLIMA telescope object to the event with the Gaia lightcurve
    tel1 = telescopes.Telescope(name = name_roman, camera_filter='W149',
                                lightcurve=bands_dict['W149'],
                                lightcurve_names=['time', 'mag', 'err_mag'],
                                lightcurve_units=['JD', 'mag', 'mag'],
                                location='Space')

    tel1.spacecraft_positions = {'astrometry': [], 'photometry': np.load(path_ephemerides)}
    e.telescopes.append(tel1)
    tel_list.append('Roman')

    # lsst_lc_list = [lsst_u, lsst_g, lsst_r, lsst_i, lsst_z, lsst_y]
    lsst_bands = "ugrizy"
    for j, band in enumerate(lsst_bands):
        if not sum([len(bands_dict[key]) for key in 'ugrizy'])== 0:
            
            tel = telescopes.Telescope(name=band, camera_filter=band,
                                       lightcurve=bands_dict[band],
                                       lightcurve_names=['time', 'mag', 'err_mag'],
                                       lightcurve_units=['JD', 'mag', 'mag'],
                                       location='Earth')
            e.telescopes.append(tel)
            tel_list.append(band)
    e.check_event()

    # Give the model initial guess values somewhere near their actual values so that the fit doesn't take all day
    if self.model == 'USBL':
        t0_str = 't_center'
        u0_str = 'u_center'
    else:
        t0_str = 't0'
        u0_str = 'u0'
        
    t0 = float(event_params[t0_str])
    u0 = float(event_params[u0_str])
    tE = float(event_params['tE'])
    piEN = float(event_params['piEN'])
    piEE = float(event_params['piEE'])

    if self.model == 'FSPL':
        rho = float(event_params['rho'])
        pyLIMAmodel = FSPLarge_model.FSPLargemodel(e,blend_flux_parameter='ftotal', parallax=['Full', t0])
        param_guess = [t0, u0, tE, rho, piEN, piEE]
    elif self.model == 'USBL':
        rho = float(event_params['rho'])
        s = float(event_params['separation'])
        q = float(event_params['mass_ratio'])
        alpha = float(event_params['alpha'])
        # pyLIMAmodel = USBL_model.USBLmodel(e, blend_flux_parameter='ftotal', parallax=['Full', t0])
        pyLIMAmodel = USBL_model.USBLmodel(e,
                                           blend_flux_parameter='ftotal',origin=model.origin,
                                           parallax=['Full', t0])
        param_guess = [t0, u0, tE, rho, s, q, alpha, piEN, piEE]
    elif self.model == 'PSPL':
        pyLIMAmodel = PSPL_model.PSPLmodel(e,blend_flux_parameter='ftotal', parallax=['Full', t0])
        param_guess = [t0, u0, tE, piEN, piEE]

    if self.algo == 'TRF':
        fit_2 = TRF_fit.TRFfit(pyLIMAmodel)
        pool = None
    elif self.algo == 'MCMC':
        fit_2 = MCMC_fit.MCMCfit(pyLIMAmodel, MCMC_links=7000)
        pool = mul.Pool(processes=36)
    elif self.algo == 'DE':
        pool = mul.Pool(processes=16)
        fit_2 = DE_fit.DEfit(pyLIMAmodel, telescopes_fluxes_method='polyfit', DE_population_size=20,
                             max_iteration=10000,
                             display_progress=True)

    fit_2.model_parameters_guess = param_guess

    if self.model == 'USBL':
        fit_2.fit_parameters['separation'][1] = [s - np.abs(s) * rango, s + np.abs(s) * rango]
        fit_2.fit_parameters['mass_ratio'][1] = [q - rango * q, q + rango * q]
        fit_2.fit_parameters['alpha'][1] = [alpha - rango * abs(alpha), alpha + rango * abs(alpha)]

    if (self.model == 'USBL') or (self.model == 'FSPL'):
        if (rho - rango * abs(rho))<0:
            fit_2.fit_parameters['rho'][1] = [0, rho + rango * abs(rho)]
        else:
            fit_2.fit_parameters['rho'][1] = [rho - rango * abs(rho), rho + rango * abs(rho)]

    fit_2.fit_parameters[t0_str][1] = [t0 - 10, t0 + 10]  # t0 limits
    fit_2.fit_parameters[u0_str][1] = [u0 - abs(u0) * rango, u0 + abs(u0) * rango]  # u0 limits

    fit_2.fit_parameters['tE'][1] = [tE - tE * rango, tE + tE * rango]  # tE limits in days
    fit_2.fit_parameters['piEE'][1] = [piEE - rango * abs(piEE),
                                       piEE + rango * abs(piEE)]  # parallax vector parameter boundaries
    fit_2.fit_parameters['piEN'][1] = [piEN - rango * abs(piEN),
                                       piEN + rango * abs(piEN)]  # parallax vector parameter boundaries
    
    if self.algo == "MCMC" or self.algo =='DE' :
        fit_2.fit(computational_pool=pool)
    else:
        fit_2.fit()

    true_values = np.array(event_params)
    fit_2.fit_results['true_params'] = event_params
    fit_2.fit_results['rango'] = rango
    fit_2.fit_results['method'] = algo
    fit_2.fit_results['name'] = e.name
    
    # save_fit(Source, path_save, fit_2.fit_results)
    np.save(path_save + e.name + '_' + algo +'.npy', fit_2.fit_results)
    return fit_2, e, pyLIMAmodel


def model_rubin_roman(Source, true_model, event_params, path_ephemerides, model,ORIGIN, wfirst_lc, lsst_u, lsst_g, lsst_r, lsst_i, lsst_z, lsst_y):
    '''
    Perform fit for Rubin and Roman data for fspl, usbl and pspl
    '''
    
    tlsst = 60350.38482057137 + 2400000.5
    RA, DEC = 267.92497054815516, -29.152232510353276
    e = event.Event(ra=RA, dec=DEC)

    if len(lsst_u) + len(lsst_g) + len(lsst_r) + len(lsst_i) + len(lsst_z) + len(lsst_y) == 0:
        e.name = 'Event_Roman_' + str(int(Source))
        name_roman = 'W149 (Roman)'
    else:
        e.name = 'Event_RR_' + str(int(Source))
        name_roman = 'W149 (Roman+Rubin)'
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
        if true_model:
            pyLIMAmodel = USBL_model.USBLmodel(e, origin=ORIGIN,
                                               blend_flux_parameter='ftotal',
                                               parallax=['Full', t_guess])
        else:
            pyLIMAmodel = USBL_model.USBLmodel(e, origin=ORIGIN, blend_flux_parameter='ftotal', parallax=['Full', t_guess])

    elif model == 'USBL_NoPiE':
        if true_model:
            pyLIMAmodel = USBL_model.USBLmodel(e, origin=ORIGIN,
                                               blend_flux_parameter='ftotal')
        else:
            pyLIMAmodel = USBL_model.USBLmodel(e, origin=ORIGIN, blend_flux_parameter='ftotal')


    
    elif model == 'PSPL':
        pyLIMAmodel = PSPL_model.PSPLmodel(e, parallax=['Full', t_guess])

    return pyLIMAmodel
def save_sim(iloc, path_TRILEGAL_set, path_to_save, my_own_model, pyLIMA_parameters, event_params):
    print('Saving Simulation...')
    # Save to an HDF5 file with specified names
    with h5py.File(path_to_save + 'Event_' + str(iloc) + '.h5', 'w') as file:
        # Save array with a specified name
        file.create_dataset('Data', data=np.array([iloc, path_TRILEGAL_set, my_own_model.origin[0]], dtype='S'))
        # Save dictionary with a specified name
        dict_group = file.create_group('pyLIMA_parameters')
        for key, value in pyLIMA_parameters.items():
            dict_group.attrs[key] = value
            
        dict_group_tril = file.create_group('TRILEGAL_params')
        for key, value in event_params.items():
            dict_group_tril.attrs[key] = value

        # Save table with a specified name
        for telo in my_own_model.event.telescopes:
            table = telo.lightcurve
            table_group = file.create_group(telo.name)
            for col in table.colnames:
                table_group.create_dataset(col, data=table[col])
    print('File saved:',path_to_save + 'Event_' + str(iloc) + '.h5' )

def save_fit(iloc , path_to_save, fit_results):
    print('Saving Fit results...')
    # Save to an HDF5 file with specified names
    with h5py.File(path_to_save + 'Event_' + str(iloc) + '.h5', 'w') as file:
        dict_group = file.create_group('fit_results_'+fit_results['name'])
        for key, value in fit_results.items():
            dict_group.attrs[key] = value

    print('File saved:',path_to_save + 'Event_' + str(iloc) + '.h5' )


def read_data(path_model):
    # Open the HDF5 file and load data using specified names
    with h5py.File(path_model, 'r') as file:
        # Load array with string with info of dataset using its name
        info_dataset = file['Data'][:]
        info_dataset = [file['Data'][:][0].decode('UTF-8'), file['Data'][:][1].decode('UTF-8'),
                        [file['Data'][:][2].decode('UTF-8'), [0, 0]]]
        # Dictionary using its name
        pyLIMA_parameters = {key: file['pyLIMA_parameters'].attrs[key] for key in file['pyLIMA_parameters'].attrs}
        # Load table using its name
        bands = {}
        for band in ("W149", "u", "g", "r", "i", "z", "y"):
            loaded_table = QTable()
            for col in file[band]:
                loaded_table[col] = file[band][col][:]
            bands[band] = loaded_table
        return info_dataset, pyLIMA_parameters, bands


def mag(zp, Flux):
    '''
    Transform the flux to magnitude
    inputs
    zp: zero point
    Flux: vector that contains the lightcurve flux
    '''
    return zp - 2.5 * np.log10(abs(Flux))
    
def has_consecutive_numbers(lst):
        """
        check if there at least 3 consecutive numbers in a list lst
        """
        sorted_lst = sorted(lst)
        for i in range(len(sorted_lst) - 2):
            if sorted_lst[i] + 1 == sorted_lst[i + 1] == sorted_lst[i + 2] - 1:
                return True
        return False
    
def set_photometric_parameters(exptime, nexp, readnoise=None):
    # readnoise = None will use the default (8.8 e/pixel). Readnoise should be in electrons/pixel.
    photParams = PhotometricParameters(exptime=exptime, nexp=nexp, readnoise=readnoise)
    return photParams


ZP = {'W149': 27.615, 'u': 27.03, 'g': 28.38, 'r': 28.16,
      'i': 27.85, 'z': 27.46, 'y': 26.68}

class simulation_event:
    def __init__(self,ra, dec,random_seed, path_ephemerides, path_dataslice,
                    data_TRILEGAL, data_Genulens, system_type, model, orbital_period, name, ZP_dict):
        self.ra = ra
        self.dec = dec
        self.random_seed = random_seed
        self.path_ephemerides = path_ephemerides
        self.path_dataslice = path_dataslice
        self.data_TRILEGAL = data_TRILEGAL
        self.data_Genulens = data_Genulens
        self.system_type = system_type
        self.model = model
        self.orbital_period = orbital_period
        self.name = name
        self.ZP_dict = ZP_dict
        
    def ts_rubin(self):
        
        LSST_BandPass = {}
        lsst_filterlist = 'ugrizy'
        for f in lsst_filterlist:
            LSST_BandPass[f] = Bandpass()
            LSST_BandPass[f].read_throughput(str(script_dir)+'/troughputs/' + f'total_{f}.dat')
        dataSlice = np.load(self.path_dataslice, allow_pickle=True)
        rubin_ts = {}
        
        for fil in lsst_filterlist:
            m5 = dataSlice['fiveSigmaDepth'][np.where(dataSlice['filter'] == fil)]
            mjd = dataSlice['observationStartMJD'][np.where(dataSlice['filter'] == fil)] + 2400000.5
            int_array = np.column_stack((mjd, m5, m5)).astype(float)
            rubin_ts[fil] = int_array
        return rubin_ts

        
    def roman_telescope(self):
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
        #ephemerides = np.load(self.path_ephemerides)
        Roman_tot.location = 'Space'
        Roman_tot.spacecraft_name = 'L2'
        #Roman_tot.spacecraft_positions = {'astrometry': [], 'photometry': ephemerides}
        return Roman_tot
        
    def roman_telescope_from_cache(self, cached_lightcurve):
        """Construye el telescopio Roman usando una lightcurve pre-calculada."""
        
    
        # Armar array igual al formato que espera Telescope:
        # columnas: time, mag, err_mag
        lc_array = np.c_[
            cached_lightcurve['time'],
            cached_lightcurve['mag'],
            cached_lightcurve['err_mag']
        ]
        
        tel = telescopes.Telescope(
            name='W149',
            camera_filter='W149',
            lightcurve=lc_array,
            lightcurve_names=['time', 'mag', 'err_mag'],
            lightcurve_units=['JD', 'mag', 'mag'],
            location='Space'
        )
        tel.spacecraft_name = 'L2'
        return tel

    def rubin_telescope(self, rubin_ts):
        
        
        lsst_filterlist = 'ugrizy'
        dict_tels = {}
        # print('rubin_ts',rubin_ts)
        for band in rubin_ts:
            # print('band',band)
            dict_tels[band] = telescopes.Telescope(name=band, camera_filter=band, location='Earth',
                                                  lightcurve=rubin_ts[band],
                                                  lightcurve_names=['time', 'mag', 'err_mag'],
                                                  lightcurve_units=['JD', 'mag', 'mag'])
        return dict_tels
    


    
    def Event_Rubin_custom_cadence(self, ts_dict):
        '''
        This function creates a odi Event from pyLIMA with a custom cadence
        
        ts_dict (dict): keys 'u','g','r','i','z','y'
        l (float): galactic coordinate l
        b (float): galactic coordinate b
        '''
        # Ra, Dec = coords(l,b)
        my_own_creation = event.Event(ra=self.ra, dec=self.dec)
        my_own_creation.name = self.name
        lsst_filterlist = 'g'
        rubin_ts = {}
        for band in lsst_filterlist:
            if band in ts_dict.keys():
                mjd = ts_dict[band]
                m5 = np.ones(len(mjd))*20
                int_array = np.column_stack((mjd, m5, m5)).astype(float)
                rubin_ts[band] = int_array
            
        for band in lsst_filterlist:
            if band in ts_dict.keys():
                my_own_creation.telescopes.append(self.rubin_telescope(rubin_ts)[band])
    
        return my_own_creation
        
    def Event_roman_rubin(self):
        '''
        :param opsim:
        :return:
        '''
        
        # Ra, Dec = coords()
        # rubin_ts = self.ts_rubin()
        rubin_ts = self.ts_rubin()
        tlsst = 60413.26382860778 + 2400000.5
    
        my_own_creation = event.Event(ra=self.ra, dec=self.dec)
        my_own_creation.name = 'An event observed by Roman and Rubin'
    
        Roman_tot = self.roman_telescope()
        my_own_creation.telescopes.append(Roman_tot)
        lsst_filterlist = 'ugrizy'
        rubin_tel = self.rubin_telescope(rubin_ts)       
        for band in lsst_filterlist:
                 
            my_own_creation.telescopes.append(rubin_tel[band])
    
        return my_own_creation#, dataSlice, LSST_BandPass

    
    def Event_roman_ODI(self, roman_lightcurve=None):
        my_own_creation = event.Event(ra=self.ra, dec=self.dec)
        Roman_tot = self.roman_telescope() if roman_lightcurve is None else self.roman_telescope_from_cache(roman_lightcurve)
        my_own_creation.telescopes.append(Roman_tot)
        return my_own_creation

    
    def event_param(self):
        # print(f'Generation of parameters: {system_type}')
        np.random.seed(self.random_seed)
        DL = self.data_Genulens['D_L']
        DS = self.data_Genulens['D_S']
        mu_rel = self.data_Genulens['mu_rel']
        logL = self.data_TRILEGAL['logL'] # log10 of the luminosity in Lsun from TRILEGAL
        logTe = self.data_TRILEGAL['logTe']  # log10 of effective temperature in K from TRILEGAL
         # = 0
        semi_major_axis =  np.random.uniform(0.1,28)  
    
        if self.system_type == "Planets_systems":      
            star_mass = np.random.uniform(1,100)
            mass_planet = np.random.uniform(1/300,13)
        
        elif self.system_type =="Binary_stars":
            star_mass = np.random.uniform(1,50)
            mass_planet = np.random.uniform(1,50)*u.M_sun.to("M_jup")
        
        elif self.system_type == "BH":
            star_mass = np.random.uniform(1,100) # mass of the BH
            mass_planet = 0
    
        elif self.system_type == "FFP":
            star_mass = 0  
            mass_planet = np.random.uniform(1/300,13)
        
        elif self.system_type == "lenses_low_mass":
            star_mass = 0  
            mass_planet =lognuniform(low=np.log10(0.01*u.M_earth.to("M_jup")), high=1.1139433523068367, size=None,base=10)# np.random.uniform(3.15e-12,6)
    
        else:
            raise ValueError(f"Unknown system_type: {self.system_type}")
         
        event_params = microlensing_params(self.system_type, self.orbital_period, semi_major_axis, DL, star_mass, 
                                                    mass_planet, DS, mu_rel, logTe, logL)
        # print("mu_rel",event_params.mu_rel)
        # print("mass_lens",event_params.mass_planet)
        # print("DL",event_params.DL)
        # print("DS",event_params.DS)
        t0 = 2462774.5545595586 #np.random.uniform(2460413.013828608, 2460413.013828608+365.25*8)  
        # print('t0',t0)
        rho = event_params.rho()       
        tE = event_params.tE()
        piE = event_params.piE()
        
        if self.system_type == "Planets_systems":
            u0 = rho*np.random.uniform(0,1)
        else:
            u0 = np.random.uniform(0,3)
        #u0 = 2
        alpha = np.random.uniform(0,np.pi)        
        angle = np.random.uniform(0,2*np.pi)    
        piEE = piE*np.cos(angle)
        piEN = piE*np.sin(angle)
        
        params_ulens = {'t0':t0,"u0":u0,"tE":tE.value,
                         'mass':mass_planet,'thetaE':event_params.theta_E().value, 
                         "thetas":event_params.thetas().value,
                         'radius': float(event_params.source_radius().value)}
        
        if self.system_type in ["FFP", "Binary_stars","Planets_systems"]:
            params_ulens["piEN"]=piEN.value
            params_ulens["piEE"]=piEE.value

        
        if self.system_type in ["FFP", "Binary_stars","Planets_systems", "lenses_low_mass"]:
            params_ulens['rho'] = float(rho)#.value
        
        if self.system_type in ["Binary_stars",'Planets_systems']:
            s = event_params.s()
            q = event_params.mass_ratio()
            params_ulens['s'] = s.value
            params_ulens['q'] = q.value
            params_ulens['alpha'] = alpha
    
        return params_ulens
         
    def sim_lightcurve(self, event, data):
        '''
        i (int): index of the TRILEGAL data set
        data (dictionary): parameters including magnitude of the stars
        path_ephemerides (str): path to the ephemeris of Gaia
        path_dataslice(str): path to the dataslice obtained from OpSims
        model(str): model desired
        '''
        # i = 
        # data = self.event_param()
        data_magstar = self.data_TRILEGAL
        magstar = {'W149': data_magstar["W149"], 'u': data_magstar["u"], 'g': data_magstar["g"],
                   'r': data_magstar["r"],'i': data_magstar["i"], 'z': data_magstar["z"], 
                   'y': data_magstar["Y"]}
        ZP = self.ZP_dict
       
        new_creation = copy.deepcopy(event)
        np.random.seed(self.random_seed)
        t0 = data['t0']
        tE = data['tE']
    
        if self.model == 'USBL':
            params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['tE'], 'rho': data['rho'],
                      's': data['s'], 'q': data['q'], 'alpha': data['alpha'],
                      'piEN': data['piEN'], 'piEE': data['piEE']}
            choice = np.random.choice(["central_caustic", "second_caustic", "third_caustic"])
            # usbl = pyLIMA.models.USBL_model.USBLmodel(roman_event, origin=[choice, [0, 0]],blend_flux_parameter='ftotal')
            my_own_model = USBL_model.USBLmodel(new_creation, origin=[choice, [0, 0]],
                                                blend_flux_parameter='ftotal',
                                                parallax=['Full', t0])
            # print(my_own_model.origin)
            # my_own_model = USBL_model.USBLmodel(new_creation,origin=[choice, [0, 0]], parallax=['Full', t0])

        elif self.model =='PSBL':
            params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['tE'],
                                  's': data['s'], 'q': data['q'], 'alpha': data['alpha']
                                  }
            choice = np.random.choice(["central_caustic", "second_caustic", "third_caustic"])
            # usbl = pyLIMA.models.USBL_model.USBLmodel(roman_event, origin=[choice, [0, 0]],blend_flux_parameter='ftotal')
            my_own_model = PSBL_model.PSBLmodel(new_creation, origin=["central_caustic", [0, 0]],
                                                blend_flux_parameter='ftotal',
                                                )

        elif self.model == 'FSPL':
            params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['tE'],
                      'rho': data['rho']}
            my_own_model = FSPLarge_model.FSPLargemodel(new_creation) #, parallax=['Full', t0])

        elif self.model == 'PSPL':
            params = {'t0': data['t0'], 'u0': data['u0'], 'tE': data['tE']}
            my_own_model = PSPL_model.PSPLmodel(new_creation,
                                                blend_flux_parameter='ftotal',
                                                parallax=['Full', t0])
    
        my_own_parameters = []
        for key in params:
            my_own_parameters.append(params[key])

        my_own_flux_parameters = []
        fs, G, F = {}, {}, {}
        np.random.seed(self.random_seed)
        for i in range(len(new_creation.telescopes)):
            band = new_creation.telescopes[i].name
            flux_baseline = 10 ** ((ZP[self.name] - magstar[band]) / 2.5)
            g = 0 
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
        simulator.simulate_lightcurve(my_own_model, pyLIMA_parameters, add_noise=True)
        return my_own_model, pyLIMA_parameters

    
    def output_perfect_lightcurves(self, event):
        my_own_model, pyLIMA_parameters = self.sim_lightcurve(event)
        for k in range(len(my_own_model.event.telescopes)):
            model_flux = my_own_model.compute_the_microlensing_model(my_own_model.event.telescopes[k],
                                                                     pyLIMA_parameters)['photometry']
            my_own_model.event.telescopes[k].lightcurve['flux'] = model_flux
            my_own_model.event.telescopes[k].lightcurve['mag'] = mag(ZP[my_own_model.event.telescopes[k].name], model_flux)   
        return my_own_model, pyLIMA_parameters


    def sim_clear_Rubin_lightcurves(self, event, data):
        my_own_model, pyLIMA_parameters = self.sim_lightcurve(event, data)
        for k in range(len(my_own_model.event.telescopes)):
            if my_own_model.event.telescopes[k].name != "W149":
                model_flux = my_own_model.compute_the_microlensing_model(my_own_model.event.telescopes[k],
                                                                         pyLIMA_parameters)['photometry']
                my_own_model.event.telescopes[k].lightcurve['flux'] = model_flux
            else:
                my_own_model.event.telescopes[k].lightcurve["mag"]=my_own_model.event.telescopes[k].lightcurve["mag"]+self.ZP_dict[my_own_model.event.telescopes[k].name]-27.4
        return my_own_model, pyLIMA_parameters


    def sim_event(self, event):
        '''
        i (int): index of the TRILEGAL data set
        data (dictionary): parameters including magnitude of the stars
        path_ephemerides (str): path to the ephemeris of Gaia
        path_dataslice(str): path to the dataslice obtained from OpSims
        model(str): model desired
        '''
        i = self.random_seed

        dataSlice = np.load(self.path_dataslice, allow_pickle=True)
        LSST_BandPass = {}
        lsst_filterlist = 'ugrizy'
        for f in lsst_filterlist:
            LSST_BandPass[f] = Bandpass()
            LSST_BandPass[f].read_throughput(str(script_dir)+'/troughputs/' + f'total_{f}.dat')
            
        photParams = set_photometric_parameters(15, 2)
        my_own_creation = self.Event_roman_rubin()#self.path_ephemerides, self.path_dataslice)
        new_creation = copy.deepcopy(my_own_creation)
        
        # model, pyLIMA_parameters = self.sim_lightcurve(event)
        my_own_model, pyLIMA_parameters = self.sim_clear_Rubin_lightcurves(event)
    
        np.random.seed(i)
        
        Roman_band = False
        Rubin_band = False
        for telo in my_own_model.event.telescopes:
            if telo.name == 'W149':
                # display(telo.lightcurve)
                telo.lightcurve['mag'] = (telo.lightcurve['mag'].value- 27.4 + ZP[telo.name])*u.mag
                m5 = np.ones(len(telo.lightcurve['mag'])) * 27.6
                telo.lightcurve = self.filter_band(telo.lightcurve, m5, telo.name)
                if not len(telo.lightcurve['mag']) == 0:
                    Roman_band = True
            else:
                # display(telo.lightcurve)
                X = telo.lightcurve['time'].value
                ym = mag(ZP[telo.name], telo.lightcurve['flux'].value)
                z, y, x, M5 = [], [], [], []
                for k in range(len(ym)):
                    m5 = dataSlice['fiveSigmaDepth'][np.where(dataSlice['filter'] == telo.name)][k]
                    magerr = calc_mag_error_m5(ym[k], LSST_BandPass[telo.name], m5, photParams)[0]
                    z.append(magerr)
                    y.append(np.random.normal(ym[k], magerr))
                    x.append(X[k])
                    M5.append(m5)
                telo.lightcurve = self.filter_band(telo.lightcurve, m5, telo.name)
    
                if not len(telo.lightcurve['mag']) == 0:
                    Rubin_band = True
    
        # This first if holds for an event with at least one Roman and Rubin band
        if Rubin_band and Roman_band:
            # This second if holds for a "detectable" event to fit
            if self.filter5points(pyLIMA_parameters, my_own_model.event.telescopes) and self.deviation_from_constant(pyLIMA_parameters, my_own_model.event.telescopes):
                print("A good event to fit")
                return my_own_model, pyLIMA_parameters, True
            else:
                print(
                    "Not a good event to fit.\nFail 5 points in t0+-tE\nNot have 3 consecutives points that deviate from constant flux in t0+-tE")
                return my_own_model, pyLIMA_parameters, False
        else:
            print("Not a good event to fit since no Rubin band")
            return my_own_model, pyLIMA_parameters, False

    
    
    def deviation_from_constant(self,pyLIMA_parameters, pyLIMA_telescopes):
        '''
         There at least 6 points in the range
         $[t_0-tE, t_0+t_E]$ with the magnification deviating from the
         constant flux by more than 3$\sigma$
        '''
        ZP = self.ZP_dict
        t0 = pyLIMA_parameters['t0']
        tE = pyLIMA_parameters['tE']
        delta_tE = 3.5*tE#*np.sqrt((1+pyLIMA_parameters['rho'])**2-pyLIMA_parameters['u0']**2)
        satis_crit = {}
        for telo in pyLIMA_telescopes:
            if not len(telo.lightcurve['mag']) == 0:
                mag_baseline = ZP[self.name] - 2.5 * np.log10(pyLIMA_parameters['ftotal_' + f'{telo.name}'])
                # print("mag_baseline",mag_baseline)
                x = telo.lightcurve['time'].value
                y = telo.lightcurve['mag'].value
                z = telo.lightcurve['err_mag'].value
                mask = (t0 - delta_tE < x) & (x < t0 + delta_tE)
                consec = []
                if len(x[mask]) >= 6:
                    combined_lists = list(zip(x[mask], y[mask], z[mask]))
                    sorted_lists = sorted(combined_lists, key=lambda item: item[0])
                    sorted_x, sorted_y, sorted_z = zip(*sorted_lists)
                    for j in range(len(sorted_y)):
                        if sorted_y[j] + 3 * sorted_z[j] < mag_baseline:
                            consec.append(j)
                    
                    result = has_consecutive_numbers(consec)
                    if result:
                        satis_crit[self.name] = True
                    else:
                        satis_crit[self.name] = False
                else:
                    satis_crit[self.name] = False
            else:
                satis_crit[self.name] = False
        return any(satis_crit.values())

    
    def filter5points(self, pyLIMA_parameters, pyLIMA_telescopes):
        '''
        Check that at least one light curve
        have at least 5 pts in the t0+-tE
        '''
        t0 = pyLIMA_parameters['t0']
        tE = pyLIMA_parameters['tE']
        crit5pts = {}
        for telo in pyLIMA_telescopes:
            if not len(telo.lightcurve['mag']) == 0:
                x = telo.lightcurve['time'].value
                mask = (t0 - tE < x) & (x < t0 + tE)
                if len(x[mask]) >= 5: #cambiar a 10, 15
                    crit5pts[telo.name] = True
                else:
                    crit5pts[telo.name] = False
        return any(crit5pts.values())
    
    
    def FilterNpoints(n, pyLIMA_parameters, pyLIMA_telescopes):
        '''
        Check that at least one light curve
        have at least 5 pts in the t0+-tE
        '''
        t0 = pyLIMA_parameters['t0']
        tE = pyLIMA_parameters['tE']
        critNpts = {}
        for telo in pyLIMA_telescopes:
            if not len(telo.lightcurve['mag']) == 0:
                x = telo.lightcurve['time'].value
                mask = (t0 - tE < x) & (x < t0 + tE)
                if len(x[mask]) >= n: #cambiar a 10, 15
                    critNpts[telo.name] = True
                else:
                    critNpts[telo.name] = False
        return any(critNpts.values())
        
    
    def filter_band(self,lightcurve, m5, fil):
        '''
        *Save the points of the lightcurve greater and smaller than
          1sigma fainter and brighter that the saturation and 5sigma_depth
        * check that the lightcurve have more than 10 points
        * check if the lightcurve have at least 1 point at 5 sigma from the 5sigma_depth
        '''
        mag_sat = {'W149': 14.8, 'u': 14.7, 'g': 15.7, 'r': 15.8, 'i': 15.8, 'z': 15.3, 'y': 13.9}
        lightcurve['m5'] =  m5
    
        b1 = lightcurve['mag'].value - lightcurve['err_mag'].value > mag_sat[fil]
        b2 = lightcurve['mag'].value + lightcurve['err_mag'].value < lightcurve['m5']
        lc_fil1 = lightcurve[b1&b2]
        return lc_fil1
