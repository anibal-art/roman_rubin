#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:32:47 2026

@author: anibal
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
current_path = os.getcwd()
two_levels_up = os.path.dirname(os.path.dirname(os.getcwd()))
parent_directory = two_levels_up#os.path.abspath(os.path.join(current_path, os.pardir))
print("Parent Directory:", parent_directory)
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(os.path.dirname(os.getcwd()))

from class_analysis import create_df_to_plot
from ssh_connect import ssh_che
from connect_CHE import download_data
from read_save import read_data
from class_analysis import graph_maker_1plot


ssh = ssh_che()
sources_array = [1007]
model = "USBL"
nset = 1
system_type = 'Planets_systems'

path_run = f'/share/storage3/rubin/microlensing/romanrubin/RR2025/finalv2_set/{system_type}/'
path_save = os.getcwd()+f'/lightcurves/{system_type}_{nset}/'

if not os.path.exists(path_save):
    os.makedirs(path_save, exist_ok=True)
if not os.path.exists(path_save+f'Event_{sources_array[0]}.h5')==True:
    download_data(ssh,sources_array, nset, path_run, path_save)
else:
    print('The file already exist')

df_to_plot = create_df_to_plot(sources_array, model,path_save)

plt.close('all')

path_save = '/home/anibal/'
path_ephemerides = '/home/anibal/microlensing/simulation_Rubin/roman_rubin/ephemerides/Roman_positions.npy'
fig, axes = graph_maker_1plot(df_to_plot, model, False, path_save, path_ephemerides)

indices, strings, pyLIMA_parameters, TRILEGAL_params, bands, GENULENS_row, TRILEGAL_row= read_data(df_to_plot['path_event'].iloc[0])
axes[0].set_xlim(pyLIMA_parameters['t0']-100,pyLIMA_parameters['t0']+100)

axes[0].axvspan(pyLIMA_parameters['t0']-pyLIMA_parameters['tE'],pyLIMA_parameters['t0']+pyLIMA_parameters['tE'],color='blue',alpha=0.25)
# axes[0].axvspan(pyLIMA_parameters['t0']-pyLIMA_parameters['tE'],pyLIMA_parameters['t0']+pyLIMA_parameters['tE'],color='blue',alpha=0.25)
# axes[0].axvline(params['t0'])
# axes[0].axvspan(params['t_center']-1*params['tE']*np.sqrt(params['mass_ratio']),
#                 params['t_center']+1*params['tE']*np.sqrt(params['mass_ratio']),
#                 alpha=0.15,
#                 color='blue',label="t_{c}\pm t_{Ep}")
# axes[0].axvspan(params['t_center']-5*params['tE']*np.sqrt(params['rho']),params['t_center']+5*params['tE']*np.sqrt(params['rho']),alpha=0.25,color='red')
# plt.tight_layout()

plt.show()