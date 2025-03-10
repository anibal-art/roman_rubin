import os, sys
import numpy as np
import matplotlib.pyplot as plt
from pyLIMA.outputs import pyLIMA_plots
from cycler import cycler
import pandas as pd
sys.path.append(os.path.dirname(os.getcwd()))
from functions_roman_rubin import sim_fit,sim_event
from functions_roman_rubin import model_rubin_roman, fit_rubin_roman
from functions_roman_rubin import read_data, save

current_path = os.getcwd()
i=18 #select one event by its index in the TRILEGAL set

path_TRILEGAL_set= current_path+'/TRILEGAL/PB_planet_split_1.csv'
path_to_save_model= current_path+'/test_sim_fit/'
path_to_save_fit= current_path+'/test_interval_width/'
path_ephemerides= current_path+'/ephemerides/Gaia.txt'
path_dataslice = current_path+'/opsims/baseline/dataSlice.npy'
path_fit_rr = path_to_save_fit+f'/Event_RR_{i}_TRF.npy'
path_fit_roman =  path_to_save_fit+f'/Event_Roman_{i}_TRF.npy'
ZP = {'W149':27.615, 'u':27.03, 'g':28.38, 'r':28.16,
              'i':27.85, 'z':27.46, 'y':26.68}
colorbands={'W149':'b', 'u':'purple', 'g':'g', 'r':'red',
          'i':'yellow', 'z':'k', 'y':'cyan'}

def fit_test(index):
    # print(current_path)
    # [1e-4,1e-3,1e-2,1e-1,0.5,0.8,1,1.5,2,3,4,5]
    values = [1.00000000e-04, 1.26485522e-04, 1.59985872e-04, 2.02358965e-04,
       2.55954792e-04, 3.23745754e-04, 4.09491506e-04, 5.17947468e-04,
       6.55128557e-04, 8.28642773e-04, 1.04811313e-03, 1.32571137e-03,
       1.67683294e-03, 2.12095089e-03, 2.68269580e-03, 3.39322177e-03,
       4.29193426e-03, 5.42867544e-03, 6.86648845e-03, 8.68511374e-03,
       1.09854114e-02, 1.38949549e-02, 1.75751062e-02, 2.22299648e-02,
       2.81176870e-02, 3.55648031e-02, 4.49843267e-02, 5.68986603e-02,
       7.19685673e-02, 9.10298178e-02, 1.15139540e-01, 1.45634848e-01,
       1.84206997e-01, 2.32995181e-01, 2.94705170e-01, 3.72759372e-01,
       4.71486636e-01, 5.96362332e-01, 7.54312006e-01, 9.54095476e-01,
       1.20679264e+00, 1.52641797e+00, 1.93069773e+00, 2.44205309e+00,
       3.08884360e+00, 3.90693994e+00, 4.94171336e+00, 6.25055193e+00,
       7.90604321e+00, 1.00000000e+01]
    rango = values[index]
    model='USBL'   
    info_dataset, pyLIMA_parameters, bands = read_data(path_to_save_model+'/Event_18.h5')
    # ulens_params = []
    PAR = ['t_center','u_center','tE','rho','separation','mass_ratio','alpha','piEN','piEE']
    
    # for b in (PAR):
    #     ulens_params.append(pyLIMA_parameters[b])
    # flux_params = []
    
    # Here we change the zero point to the pyLIMA convention in order to make the alignment
    # for b in bands:
    #     if not len(bands[b])==0:
    #         zp_Rubin_to_pyLIMA = (10**((-27.4+ZP[b])/2.5))
            
    #         flux_params.append(pyLIMA_parameters['fsource_'+b]/zp_Rubin_to_pyLIMA)
    #         flux_params.append(pyLIMA_parameters['ftotal_'+b]/zp_Rubin_to_pyLIMA)
            
    # true_params = ulens_params+flux_params
    
    model_ulens = 'USBL'
    
    Source = 18
    event_params = pyLIMA_parameters
    event_params['te']=event_params['tE']
    event_params['s']=event_params['separation']
    event_params['q']=event_params['mass_ratio']
    
    f= 'W149'
    wfirst_lc = np.array([bands[f]['time'],bands[f]['mag'],bands[f]['err_mag']]).T
    f = 'u'
    lsst_u = np.array([bands[f]['time'],bands[f]['mag'],bands[f]['err_mag']]).T
    f='g'
    lsst_g = np.array([bands[f]['time'],bands[f]['mag'],bands[f]['err_mag']]).T
    f='r'
    lsst_r = np.array([bands[f]['time'],bands[f]['mag'],bands[f]['err_mag']]).T
    f='i'
    lsst_i = np.array([bands[f]['time'],bands[f]['mag'],bands[f]['err_mag']]).T
    f='z'
    lsst_z = np.array([bands[f]['time'],bands[f]['mag'],bands[f]['err_mag']]).T
    f='z'
    lsst_y = np.array([bands[f]['time'],bands[f]['mag'],bands[f]['err_mag']]).T
    
    # model_true = model_rubin_roman(Source,True,event_params, path_ephemerides,model_ulens, wfirst_lc, lsst_u, lsst_g, lsst_r, lsst_i, lsst_z,
    #                     lsst_y)
    
    algo='TRF'
    
    Source = i
    
    
    origin = 'NOTHING'
    # rango = 1
    fit_rr, event_fit_rr, pyLIMAmodel_rr = fit_rubin_roman(index, event_params, path_to_save_fit, path_ephemerides,model,algo,origin,rango,
                               wfirst_lc, lsst_u, lsst_g, lsst_r,
                                           lsst_i, lsst_z, lsst_y)
    # fit_roman, event_fit_roman, pyLIMAmodel_roman = fit_rubin_roman(Source,event_params, path_to_save_fit, path_ephemerides,model,algo,origin,rango,
    #                            wfirst_lc, [], [], [], [], [],[])