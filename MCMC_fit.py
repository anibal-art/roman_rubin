import os, sys
import numpy as np
import matplotlib.pyplot as plt
from pyLIMA.outputs import pyLIMA_plots
from cycler import cycler

sys.path.append(os.path.dirname(os.getcwd()))
from functions_roman_rubin import sim_fit
from functions_roman_rubin import model_rubin_roman
from functions_roman_rubin import read_data

current_path = str(os.getcwd())#os.path.dirname(os.getcwd())
# print()
i=18
model='USBL'
algo='MCMC'
path_TRILEGAL_set= current_path+'/TRILEGAL/PB_planet_split_1.csv'
path_to_save_model= current_path+'/MCMC_fit/'
path_to_save_fit= current_path+'/MCMC_fit/'
path_ephemerides= current_path+'/ephemerides/Gaia.txt'
path_dataslice = current_path+'/opsims/baseline/dataSlice.npy'

sim_fit(i,model,algo,path_TRILEGAL_set,path_to_save_model,path_to_save_fit,path_ephemerides,path_dataslice)