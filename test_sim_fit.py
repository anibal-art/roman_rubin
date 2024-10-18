from functions_roman_rubin import sim_fit
import os
import numpy as np
import pandas as pd
current_path = os.getcwd()

i=0
model='USBL'
algo='TRF'
path_TRILEGAL_set= current_path+'/TRILEGAL/PB_planet_split_1.csv'
path_to_save_model= current_path+'/test_sim_fit/'
path_to_save_fit= current_path+'/test_sim_fit/'
path_ephemerides= current_path+'/ephemerides/james_webb.txt'
path_dataslice = current_path+'/opsims/baseline/dataSlice.npy'
sim_fit(i,model,algo,path_TRILEGAL_set,path_to_save_model,path_to_save_fit,path_ephemerides,path_dataslice)