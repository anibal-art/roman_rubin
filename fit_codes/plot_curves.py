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
from pyLIMA.outputs import pyLIMA_plots
from pyLIMA.outputs import file_outputs
import pandas as pd
from fit_results import chi_dof, best_model, errors
from filter_curves import filtros
path_ephemerides = '/home/anibal/files_db/james_webb.txt' #PATH TO EPHEMERIDES
from plot_models import plot_LCmodel

def plot_fit_rr(i,path_model, path_file):
    fit_params = best_model(path_file)
    curvas, params = filtros(path_model)
    
    RA, DEC= 267.92497054815516, -29.152232510353276
    your_event = event.Event(ra=RA, dec=DEC)
    your_event.name = 'fit_rr:'+str(int(params['Source']))+'\n chi_sq='+str(round(chi_dof(path_model, path_file),4))

    telescope_1 = telescopes.Telescope(name = 'F146', 
                                       camera_filter = 'F146',
                                       light_curve = curvas['w'].astype(float),
                                       light_curve_names = ['time','mag','err_mag'],
                                       light_curve_units = ['JD','mag','mag'])

    telescope_1.location = 'Space'

    tlsst = 60350.38482057137+2400000.5
    ephemerides = np.loadtxt(path_ephemerides)
    ephemerides[:,0] = ephemerides[:,0]
    ephemerides[:,3] *=  60*300000/150000000
    deltaT = tlsst-ephemerides[:,0][0]
    ephemerides[:,0] = ephemerides[:,0]+deltaT
    telescope_1.spacecraft_positions ={'astrometry':[],'photometry':ephemerides}
    your_event.telescopes.append(telescope_1)
    if not len(curvas['u']) == 0:
        telescope_2 = telescopes.Telescope(name = 'LSST_u', 
                                           camera_filter = 'u',
                                           light_curve = curvas['u'].astype(float),
                                           light_curve_names = ['time','mag','err_mag'],
                                           light_curve_units = ['JD','mag','mag'])
        telescope_2.location = 'Earth'
        your_event.telescopes.append(telescope_2)

    if not len(curvas['g']) == 0:

        telescope_3 = telescopes.Telescope(name = 'LSST_g', 
                                           camera_filter = 'g',
                                           light_curve = curvas['g'].astype(float),
                                           light_curve_names = ['time','mag','err_mag'],
                                           light_curve_units = ['JD','mag','mag'])
        telescope_3.location = 'Earth'
        your_event.telescopes.append(telescope_3)

    if not len(curvas['r']) == 0:

        telescope_4 = telescopes.Telescope(name = 'LSST_r', 
                                           camera_filter = 'r',
                                           light_curve = curvas['r'].astype(float),
                                           light_curve_names = ['time','mag','err_mag'],
                                           light_curve_units = ['JD','mag','mag'])
        telescope_4.location = 'Earth'
        your_event.telescopes.append(telescope_4)

    if not len(curvas['i']) == 0:

        telescope_5 = telescopes.Telescope(name = 'LSST_i', 
                                           camera_filter = 'i',
                                           light_curve = curvas['i'].astype(float),
                                           light_curve_names = ['time','mag','err_mag'],
                                           light_curve_units = ['JD','mag','mag'])
        telescope_5.location = 'Earth'
        your_event.telescopes.append(telescope_5)


    if not len(curvas['z']) == 0:

        telescope_6 = telescopes.Telescope(name = 'LSST_z', 
                                           camera_filter = 'z',
                                           light_curve = curvas['z'].astype(float),
                                           light_curve_names = ['time','mag','err_mag'],
                                           light_curve_units = ['JD','mag','mag'])
        telescope_6.location = 'Earth'
        your_event.telescopes.append(telescope_6)


    if not len(curvas['y']) == 0:

        telescope_7 = telescopes.Telescope(name = 'LSST_y', 
                                           camera_filter = 'y',
                                           light_curve = curvas['y'].astype(float),
                                           light_curve_names = ['time','mag','err_mag'],
                                           light_curve_units = ['JD','mag','mag'])
        telescope_7.location = 'Earth'
        your_event.telescopes.append(telescope_7)

    model_params = [params['t0'],params['u0'],params['te'],np.log10(params['s']),np.log10(params['q']),params['alpha'],params['piEN'],params['piEE']]

    your_event.check_event()

    psbl = PSBL_model.PSBLmodel(your_event, parallax=['Full', params['t0']])

    list_of_fake_telescopes = []
    print(your_event.name)
    pyLIMA_plots.plot_lightcurves(psbl,  fit_params)
    #plot_LCmodel(psbl,  model_params)
#     plt.savefig('/home/anibal/Desktop/results_fitted/'+'fit_rr_'+str(int(params['Source']))+'.png')
    pyLIMA_plots.plot_geometry(psbl,  fit_params)
#     plt.savefig('/home/anibal/Desktop/results_fitted/'+'caustic_rr_'+str(int(params['Source']))+'.png')

def plot_fit_roman(i,path_model, path_file):
    fit_params = best_model(path_file)
    curvas, params = filtros(path_model)
         
    RA, DEC= 267.92497054815516, -29.152232510353276
    roman_event = event.Event(ra=RA, dec=DEC)
    
    roman_event.name = 'fit_Roman:'+str(int(params['Source']))+'\n chi_sq='+str(round(chi_dof(path_model, path_file),4))

    telescope_1 = telescopes.Telescope(name = 'F146', 
                                       camera_filter = 'F146',
                                       light_curve = curvas['w'].astype(float),
                                       light_curve_names = ['time','mag','err_mag'],
                                       light_curve_units = ['JD','mag','mag'])

    telescope_1.location = 'Space'

    tlsst = 60350.38482057137+2400000.5
    ephemerides = np.loadtxt(path_ephemerides)
    ephemerides[:,0] = ephemerides[:,0]
    ephemerides[:,3] *=  60*300000/150000000
    deltaT = tlsst-ephemerides[:,0][0]
    ephemerides[:,0] = ephemerides[:,0]+deltaT
    telescope_1.spacecraft_positions ={'astrometry':[],'photometry':ephemerides}
    roman_event.telescopes.append(telescope_1)
    roman_event.check_event()

    psbl_roman = PSBL_model.PSBLmodel(roman_event, parallax=['Full', params['t0']])
    model_params = [params['t0'],params['u0'],params['te'],np.log10(params['s']),np.log10(params['q']),params['alpha'],params['piEN'],params['piEE']]

    list_of_fake_telescopes = []
    print(roman_event.name)
    pyLIMA_plots.plot_lightcurves(psbl_roman, fit_params)
#     plt.savefig('/home/anibal/Desktop/results_fitted/'+'fit_roman_'+str(int(params['Source']))+'.png')
    pyLIMA_plots.plot_geometry(psbl_roman, fit_params)
#     plt.savefig('/home/anibal/Desktop/results_fitted/'+'caustic_roman_'+str(int(params['Source']))+'.png')



