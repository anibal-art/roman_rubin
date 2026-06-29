import os, sys
import numpy as np
import matplotlib.pyplot as plt
# from pyLIMA.outputs 

from cycler import cycler
import pandas as pd
sys.path.append(os.path.dirname(os.getcwd()))
from functions_roman_rubin import sim_fit,sim_event
# from functions_roman_rubin import model_rubin_roman
from functions_roman_rubin import read_data, save_sim
from fit_lc import fit_rubin_roman, model_rubin_roman
import pyLIMA_plots
from fit_lc import fit_rubin_roman

current_path = os.getcwd()

i=18 #select one event by its index in the TRILEGAL set
model='USBL_NoPiE'

path_TRILEGAL_set= current_path+'/TRILEGAL/PB_planet_split_1.csv'
path_to_save_model= current_path+'/test_sim_fit/sim/'
path_to_save_fit= current_path+'/test_sim_fit/fit/'
path_ephemerides= current_path+'/ephemerides/Roman_positions.npy'
path_dataslice = current_path+'/opsims/baseline/dataSlice.npy'

ZP = {'W149':27.615, 'u':27.03, 'g':28.38, 'r':28.16,
          'i':27.85, 'z':27.46, 'y':26.68}
colorbands={'W149':'b', 'u':'purple', 'g':'g', 'r':'red',
          'i':'yellow', 'z':'k', 'y':'cyan'}

event_params = {    "u": 24.853,
    "g": 22.55,
    "r": 21.529,
    "i": 21.133,
    "z": 20.945,
    "Y": 20.828,
    "W149": 20.8178,
    "radius": 0.513042,
    "D_S": 5156,
    "D_L": 2817,
    "mu_rel": 5.147491,
    "m_planet": "9.462530108421484 jupiterMass",
    "m_star": "18.65259824621826 solMass",
    "t0": 2462592.427461+140,
    "tE": 74.7034363,
    "u0": (1/0.4)-0.4,
    "rho": 2.95e-3,
    "piEE": -0.032015,
    "piEN": 0.005886,
    "s": 0.4,
    "q": 0.000484,
    "alpha": -np.pi/2}#5.170221}

for i,alpha in enumerate(np.linspace(0, 2*np.pi,  100)):
    event_params['alpha']=alpha
    my_own_model, pyLIMA_parameters, decision = sim_event(i, event_params, path_ephemerides, path_dataslice,model)
    save_sim(i, path_TRILEGAL_set, path_to_save_model, my_own_model, pyLIMA_parameters, event_params)