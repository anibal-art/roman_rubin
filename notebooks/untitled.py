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

with open(path_save_event+"tel_rr_18.pkl", "rb") as archivo:
    tel_rr_rest = pickle.load(archivo)

with open(path_save_event+"Event_18.pkl", "rb") as archivo:
    evento_restaurado = pickle.load(archivo)

algo = 'TRF'

def fit_test(rango):
    fit_rr = evento_restaurado.fit_event(tel_rr_rest, rango, algo)
    
    with open(path_save_event+"fit_rr_18_"+str(rango)+".pkl", "wb") as archivo:
        pickle.dump(fit_rr, archivo)

        