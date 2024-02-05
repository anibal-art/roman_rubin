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
from fit_results import chi_dof, best_model, event_fits, sigmas
from tqdm.auto import tqdm
import warnings


def m1(name_file):
    nset = int(name_file[name_file.index('Event_') - 2:name_file.index('Event_') - 1])
    name_set = 'set_sim' + str(nset)
    directory_simset = name_file[0:name_file.index('set')] + name_set + '/'
    nevent = re.sub(r'\D', '', name_file[name_file.index('Event_'):-1])
    model_file = directory_simset + 'Event_' + str(nevent) + '.txt'
    curves, true = read_curves(model_file)
    fit = best_model(name_file)

    met1 = {}
    met1["Source"] = int(int(nevent) + nset * 5000)
    for p in true:
        met1[p] = abs(fit[p] - float(true[p])) / abs(float(true[p]))
    return met1


def m2(name_file):
    nset = int(name_file[name_file.index('Event_') - 2:name_file.index('Event_') - 1])
    name_set = 'set_sim' + str(nset)
    directory_simset = name_file[0:name_file.index('set')] + name_set + '/'
    nevent = re.sub(r'\D', '', name_file[name_file.index('Event_'):-1])
    model_file = directory_simset + 'Event_' + str(nevent) + '.txt'
    curves, true = read_curves(model_file)
    fit = best_model(name_file)
    err = sigmas(name_file)

    met2 = {}
    met2['Source'] = int(int(nevent) + nset * 5000)
    for p in true:
        if err[p] == 0:
            met2[p] = 5000
        else:
            met2[p] = abs(fit[p] - float(true[p])) / abs(err[p])
    return met2


def m3(name_file):
    nset = int(name_file[name_file.index('Event_') - 2:name_file.index('Event_') - 1])
    name_set = 'set_sim' + str(nset)
    directory_simset = name_file[0:name_file.index('set')] + name_set + '/'
    nevent = re.sub(r'\D', '', name_file[name_file.index('Event_'):-1])
    model_file = directory_simset + 'Event_' + str(nevent) + '.txt'
    curves, true = read_curves(model_file)
    fit = best_model(name_file)
    err = sigmas(name_file)

    met3 = {}
    met3['Source'] = int(int(nevent) + nset * 5000)
    for p in true:
        if not err[p] == 0:
            met3[p] = abs(err[p]) / float(true[p])
        else:
            met3[p] = 5000
    return met3
    
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



def sigma_ratio(path_roman, path_rr):
    nset = int(path_roman[path_roman.index('Event_')-2:path_roman.index('Event_')-1])
    name_set = 'set_sim'+str(nset)
    directory_simset = path_roman[0:path_roman.index('set')]+name_set+'/'
    nevent = re.sub(r'\D', '', path_roman[path_roman.index('Event_'):-1])
    model_file = directory_simset+'Event_'+str(nevent)+'.txt'
    curves,true = read_curves(model_file)
    fit_rr = best_model(path_rr)
    err_rr = sigmas(path_rr)
    fit_roman = best_model(path_roman)
    err_roman = sigmas(path_roman)
    metric_sigma_ratio = {}
    metric_sigma_ratio['Source'] = int(int(nevent)+nset*5000)
    for p in true:
        if not err_rr[p] == 0 or not err_roman[p] == 0:
            metric_sigma_ratio[p] = abs(err_rr[p])/abs(err_roman[p])
        else:
            metric_sigma_ratio[p] = 5000
    return metric_sigma_ratio


def bias_ratio(path_roman, path_rr):
    nset = int(path_roman[path_roman.index('Event_')-2:path_roman.index('Event_')-1])
    name_set = 'set_sim'+str(nset)
    directory_simset = path_roman[0:path_roman.index('set')]+name_set+'/'
    nevent = re.sub(r'\D', '', path_roman[path_roman.index('Event_'):-1])
    model_file = directory_simset+'Event_'+str(nevent)+'.txt'
    curves,true = read_curves(model_file)
    fit_rr = best_model(path_rr)
    err_rr = sigmas(path_rr)
    fit_roman = best_model(path_roman)
    err_roman = sigmas(path_roman)
    metric_bias_ratio = {}
    metric_bias_ratio['Source'] = int(int(nevent)+nset*5000)
    for p in true:
        metric_bias_ratio[p] = abs(fit_rr[p]-float(true[p]))/abs(fit_roman[p]-float(true[p]))
    return metric_bias_ratio


def fit_values(name_file):
    nset = int(name_file[name_file.index('Event_')-2:name_file.index('Event_')-1])
    name_set = 'set_sim'+str(nset)
    directory_simset = name_file[0:name_file.index('set')]+name_set+'/'
    nevent = re.sub(r'\D', '', name_file[name_file.index('Event_'):-1])
    model_file = directory_simset+'Event_'+str(nevent)+'.txt'
    curves,true = read_curves(model_file)
    fit = best_model(name_file)
    fit_err = sigmas(name_file)
    fit_vals = {}
    fit_vals['Source'] = int(int(nevent)+nset*5000)
    for p in true:
        fit_vals[p] = fit[p]
        fit_vals[p+'_err'] = fit_err[p]
    return fit_vals


def fit_true(path):
    '''
    This function make two pandas dataframes for fit and true parameters
    inputs

    '''
    random_file = path + 'set_sim1/' + [f for f in os.listdir(path + 'set_sim1/') if 'txt' in f][0]
    curves, true = read_curves(random_file)
    cols = ['Source'] + [key for key in true]
    cols_fit = cols + [key + '_err' for key in true]
    fit_rr = pd.DataFrame(columns=cols_fit)
    fit_roman = pd.DataFrame(columns=cols_fit)
    true = pd.DataFrame(columns=cols)
    path_fits = [path + 'set_fit' + str(i) + '/' for i in range(1, 9)]
    path_model = [path + 'set_sim' + str(i) + '/' for i in range(1, 9)]

    for j in tqdm(range(0, 8)):
        Path_Fit = path_fits[j]
        Path_Model = path_model[j]
        common_elements_list = event_fits(path_fits[j])
        for e in tqdm(common_elements_list):
            roman_file = f'Event_Roman_{int(e)}_trf.npy'
            rr_file = f'Event_RR_{int(e)}_trf.npy'
            model_file = f'Event_{int(e)}.txt'
            curves, true_param = read_curves(Path_Model + model_file)
            true.loc[len(true)] = [int(e)] + [float(true_param[key]) for key in true_param]
            fit_rr.loc[len(fit_rr)] = fit_values(Path_Fit + rr_file)
            fit_roman.loc[len(fit_roman)] = fit_values(Path_Fit + roman_file)
    return fit_rr, fit_roman, true


def metrics(path):
    random_file = path + 'set_sim1/' + [f for f in os.listdir(path + 'set_sim1/') if 'txt' in f][0]
    curves, true = read_curves(random_file)
    cols = ['Source'] + [key for key in true]
    cols_fit = cols + [key + '_err' for key in true]
    fit_rr = pd.DataFrame(columns=cols_fit)
    fit_roman = pd.DataFrame(columns=cols_fit)
    true = pd.DataFrame(columns=cols)
    path_fits = [path + 'set_fit' + str(i) + '/' for i in range(1, 9)]
    path_model = [path + 'set_sim' + str(i) + '/' for i in range(1, 9)]

    chi2_rr = pd.DataFrame(columns=['source', 'chichi'])
    chi2_roman = pd.DataFrame(columns=['source', 'chichi'])

    met_1_rr = pd.DataFrame(columns=cols)
    met_2_rr = pd.DataFrame(columns=cols)
    met_3_rr = pd.DataFrame(columns=cols)
    met_1_roman = pd.DataFrame(columns=cols)
    met_2_roman = pd.DataFrame(columns=cols)
    met_3_roman = pd.DataFrame(columns=cols)
    err_ratio = pd.DataFrame(columns=cols)
    residuals_ratio = pd.DataFrame(columns=cols)
    path_fits = [path + 'set_fit' + str(i) + '/' for i in range(1, 9)]
    path_model = [path + 'set_sim' + str(i) + '/' for i in range(1, 9)]

    for j in tqdm(range(0, 8)):
        PF = path_fits[j]
        PM = path_model[j]
        common_elements_list = event_fits(path_fits[j])
        for e in tqdm(common_elements_list):
            roman_file = f'Event_Roman_{int(e)}_trf.npy'
            rr_file = f'Event_RR_{int(e)}_trf.npy'
            model_file = f'Event_{int(e)}.txt'
            curves, true_param = read_curves(PM + model_file)

            met_1_rr.loc[len(met_1_rr)] = m1(PF + rr_file)
            met_1_roman.loc[len(met_1_roman)] = m1(PF + roman_file)

            met_2_rr.loc[len(met_2_rr)] = m2(PF + rr_file)  # , index_to_insert, value_to_add)
            met_2_roman.loc[len(met_2_roman)] = m2(PF + roman_file)  # , index_to_insert, value_to_add)

            met_3_roman.loc[len(met_3_roman)] = m3(PF + roman_file)  # , index_to_insert, value_to_add)
            met_3_rr.loc[len(met_3_rr)] = m3(PF + rr_file)  # , index_to_insert, value_to_add)
            # chi2_rr.loc[len(chi2_rr)] = [value_to_add,chi_dof(path_model[0]+model_file,path_fits[0]+rr_file)]
            # chi2_roman.loc[len(chi2_roman)] = [value_to_add,chi_dof(path_model[0]+model_file,path_fits[0]+roman_file)]
            err_ratio.loc[len(err_ratio)] = sigma_ratio(PF + roman_file, PF + rr_file)
            residuals_ratio.loc[len(residuals_ratio)] = bias_ratio(PF + roman_file, PF + rr_file)
    return err_ratio, residuals_ratio, met_1_rr, met_1_roman, met_2_rr, met_2_roman, met_3_rr, met_3_roman