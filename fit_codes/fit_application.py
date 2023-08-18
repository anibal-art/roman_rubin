from pyLIMA.models import USBL_model
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
import pygtc
from filter_curves import filtros
from fit_events import fit_rubin_roman

path_ephemerides = '/home/anibal/files_db/james_webb.txt'
path_save = '/home/anibal/roman_rubin/set_august/'
path_model = '/home/anibal/files_db/august_2023/' #PATH OF THE DIRECTORY OF THE LIGHT CURVE THAT I WANT TO FIT

def fit_light_curve(file_name,algo):#, path_ephemerides,path_save,path_model):
    '''
    This function take the name of the file that contains the light curves
    if pass all the selection criteria then fit the event and save the true parameters
    with the fited ones and his errors
    '''
    light_curve,PARAMS = filtros(file_name)
    if not light_curve == 0:
        array_w149, array_u, array_g, array_r, array_i, array_z, array_y = light_curve['w'],light_curve['u'],light_curve['g'],light_curve['r'],light_curve['i'],light_curve['z'],light_curve['y']
	
        fit_2, e, tel_list = fit_rubin_roman(PARAMS,path_save,path_ephemerides,algo, array_w149, array_u, array_g, array_r, array_i, array_z, array_y)
        fit_2, e, tel_list = fit_rubin_roman(PARAMS,path_save,path_ephemerides,algo, array_w149, [], [], [], [], [], [])
        return fit_2, e, tel_list
    else:
        print('This event is not selected and can´t be fitted.')
        return 0,0,0
