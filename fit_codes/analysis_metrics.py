import matplotlib.pyplot as plt
import numpy as np
from pyLIMA.fits import DE_fit
from pyLIMA.fits import TRF_fit
from pyLIMA.models import PSPL_model
from pyLIMA.models import USBL_model, pyLIMA_fancy_parameters
from pyLIMA.outputs import pyLIMA_plots
from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA.models import PSBL_model
import pandas as pd
import os
import re
from filter_curves import read_curves
from fit_results import best_model,errors,cov_matrix
from tqdm.auto import tqdm
import warnings 

def m1(path_file,path_model):
    data = np.load(path_file,allow_pickle=True)
    curvas, params = read_curves(path_model)
    true = np.array([params['t0'],params['u0'],params['te'],params['s'],params['q'],params['alpha'],params['piEN'],params['piEE']])
    fit = best_model(path_file)
    met1 = abs(fit-true)/true
    m1t0 = abs(fit-true)
    met1[0] = m1t0[0]
    return met1
def m2(path_file,path_model):
    data = np.load(path_file,allow_pickle=True)
    curvas, params = read_curves(path_model)
    true = [params['t0'],params['u0'],params['te'],params['s'],params['q'],params['alpha'],params['piEN'],params['piEE']]
    fit = best_model(path_file)
    err = errors(path_file)[0:8]
    if np.any(err == 0, axis=None):
        return np.zeros(len(true))
    else:
        return (abs(fit-np.array(true)))/err

def m3(path_file,path_model):
    data = np.load(path_file,allow_pickle=True)
    curvas, params = read_curves(path_model)
    true = [params['t0'],params['u0'],params['te'],params['s'],params['q'],params['alpha'],params['piEN'],params['piEE']]
    fit = best_model(path_file)
    err = errors(path_file)[0:8]
    if np.any(err == 0, axis=None):
        return np.zeros(len(true))
    else:
        return abs(err)/true
    
def Bins(list1,list2,n):
    num_bins = n
    bin_width = (max(max(list1), max(list2)) - min(min(list1), min(list2))) / num_bins
    Bins=np.arange(min(min(list1), min(list2)), max(max(list1), max(list2)) + bin_width, bin_width)
    return Bins

def stats_plot(lista_1,lista_2,ub,lb,LABEL):
    total_1 = len(lista_1)
    total_2 = len(lista_2)

    LISTA_1 = [value for value in lista_1 if lb<value<ub]
    LISTA_2 = [value for value in lista_2 if lb<value<ub]
    
    L1 = len(LISTA_1)
    L2 = len(LISTA_2)
    
    print('The fraction of events (RR) that converge near the true value is ', L1/total_1)
    print('The fraction of events (Roman) that converge near the true value is ', L2/total_2)
        
    nbins = 15
    plt.hist(LISTA_1,bins=Bins(LISTA_1, LISTA_2,nbins),color='red',hatch="\\",alpha=1)
    plt.hist(LISTA_2,bins=Bins(LISTA_1, LISTA_2,nbins),color='blue',hatch='/',alpha=0.5)
    plt.xlabel(LABEL)
    plt.show()
    
def m4(rr_file,roman_file):
    
    data_rr = np.load(rr_file,allow_pickle=True)
    data_roman = np.load(roman_file,allow_pickle=True)
    
    err_rr = errors(rr_file)[0:8]
    fit_roman = best_model(roman_file)
    err_roman = errors(roman_file)[0:8]
    fit_rr = best_model(rr_file)
    
    if np.any(err_rr == 0, axis=None):
        return np.zeros(len(true))
    else:
        # return (err_rr - err_roman )/abs(fit_roman)
        return err_rr/abs(fit_rr) - err_roman/abs(fit_roman) 



