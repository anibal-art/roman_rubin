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
from fit_functions import filtros
from tqdm.auto import tqdm
import warnings 
#path = '/home/anibal/Roman_Rubin/ajustes/trf/'
path = '/home/anibal/Roman_Rubin/ajustes/trf_chisq/trf/'
files_fits = os.listdir(path)
path_model = '/home/anibal/Desktop/roman-rubin/home/anibal-art/ROMAN-RUBIN/simulation/lightcurves/binarios_pylimav2/'


def m1(name_file,path):
    #path = '/home/anibal/Roman_Rubin/ajustes/trf/'
    data = np.load(path+name_file,allow_pickle=True)
    #path_model = '/home/anibal/Roman_Rubin/curvas/'
    curvas, params = filtros(path_model+'Event_'+str(int(re.findall(r'\d+', name_file)[0]))+'.txt','100',100,False)
    true = np.array([params['t0'],params['u0'],params['te'],np.log10(params['s']),np.log10(params['q']),params['alpha'],params['piEN'],params['piEE']])
    fit = best_model(name_file)
    return (abs(fit)-abs(true))/true

def m2(name_file,path):
    #path = '/home/anibal/Roman_Rubin/ajustes/trf/'
    data = np.load(path+name_file,allow_pickle=True)
    #path_model = '/home/anibal/Roman_Rubin/curvas/'
    curvas, params = filtros(path_model+'Event_'+str(int(re.findall(r'\d+', name_file)[0]))+'.txt','100',100,False)
    true = [params['t0'],params['u0'],params['te'],np.log10(params['s']),np.log10(params['q']),params['alpha'],params['piEN'],params['piEE']]
    fit = best_model(name_file)
    err = errors(name_file)[0:8]
    # Check if there is a zero in the entire array
    if np.any(err == 0, axis=None):
        return np.zeros(len(true))
    else:
        return (abs(fit)-abs(np.array(true)))/err

def m3(name_file,path):
    #path = '/home/anibal/Roman_Rubin/ajustes/trf/'
    data = np.load(path+name_file,allow_pickle=True)
    #path_model = '/home/anibal/Roman_Rubin/curvas/'
    curvas, params = filtros(path_model+'Event_'+str(int(re.findall(r'\d+', name_file)[0]))+'.txt','100',100,False)
    true = [params['t0'],params['u0'],params['te'],np.log10(params['s']),np.log10(params['q']),params['alpha'],params['piEN'],params['piEE']]
    fit = best_model(name_file)
    err = errors(name_file)[0:8]
    # Check if there is a zero in the entire array
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

