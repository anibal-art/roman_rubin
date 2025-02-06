import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyLIMA.outputs import pyLIMA_plots
from cycler import cycler
sys.path.append(os.path.dirname(os.getcwd()))
from class_functions_roman_rubin import sim_events
import pickle

# with open(path_save_event+"tel_roman_18.pkl", "rb") as archivo:
#     tel_roman_rest = pickle.load(archivo)
path_save_event = '/home/anibal-pc/roman_rubin'+'/test_interval_width/'
with open(path_save_event+"sim_event/tel_rr_18.pkl", "rb") as archivo:
    tel_rr_rest = pickle.load(archivo)

with open(path_save_event+"sim_event/Event_18.pkl", "rb") as archivo:
    evento_restaurado = pickle.load(archivo)

algo = 'TRF'

import h5py
import numpy as np

def guardar_en_h5(file_name, matriz, array, rango, chi2):
    """
    Guarda una matriz, un array de NumPy, una lista y un float en un archivo .h5.

    Parameters:
    - file_name (str): Nombre del archivo .h5 donde se guardarán los datos.
    - matriz (numpy.ndarray): Matriz que se guardará.
    - array (numpy.ndarray): Array de NumPy que se guardará.
    - lista (list): Lista que se guardará.
    - flotante (float): Número flotante que se guardará.
    """
    with h5py.File(file_name, 'w') as archivo_h5:
        # Guardar la matriz
        archivo_h5.create_dataset('covariance_matrix', data=matriz)
        # Guardar el array de NumPy
        archivo_h5.create_dataset('best_model', data=array)
        # Guardar la lista como un dataset de tipo float (si es convertible)
        archivo_h5.attrs['rango'] =  rango
        # Guardar el float como un atributo
        archivo_h5.attrs['chi2'] = chi2


    # cov_matrix.append(fit_rr.fit_results['covariance_matrix'])
    # best_model.append(fit_rr.fit_results['best_model'])
    # chi_results.append(fit_rr.fit_results['chi2'])
    
    # indx0 = pkl_files[j].index('_18')
    # indx1 = pkl_files[j].index('.pkl')
    # rango = float(pkl_files[j][indx0+4:indx1])
    
    # ranges.append(rango)



def fit_test(rango):
    fit_rr = evento_restaurado.fit_event(tel_rr_rest, rango, algo)
    
    # with open(path_save_event+"fit_rr_18_"+str(rango)+".pkl", "wb") as archivo:
    #     pickle.dump(fit_rr, archivo)
    cov_matrix = fit_rr.fit_results['covariance_matrix']
    best_model = fit_rr.fit_results['best_model']
    chi2_result = fit_rr.fit_results['chi2']
    file_name = path_save_event+"fit_rr_18_"+str(rango)+".h5"
    
    guardar_en_h5(file_name, cov_matrix, best_model, rango, chi2_result)
        