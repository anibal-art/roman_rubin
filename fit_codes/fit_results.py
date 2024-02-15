import pandas as pd
import numpy as np
#from filter_curves import filtros
from filter_curves import filtros, read_curves
import os
import re

def event_fits(path_fits):
    '''
    return events in common with roman and rubin
    we have events that fits only one of two for unkown reasons
    '''
    
    files_fits = os.listdir(path_fits)
    
    files_roman = [f for f in files_fits if 'Roman' in f]
    files_rr = [f for f in files_fits if not 'Roman' in f]
    
    n_rom = [] # list with the event number
    for j in files_roman:
        number = int(re.findall(r'\d+', j)[0])
        n_rom.append(number)
    
    n_rr = [] # # list with the event number
    for j in files_rr:
        number = int(re.findall(r'\d+', j)[0])
        n_rr.append(number)
    
    # Convert lists to sets
    set1 = set(n_rom)
    set2 = set(n_rr)
    # Find the common elements using intersection
    common_elements = set1.intersection(set2)
    # Convert the result back to a list (if needed)
    common_elements_list = list(common_elements)
    return common_elements_list


def cov_matrix(name_file):
    data = np.load(name_file,allow_pickle=True)
    covariance_matrix = data.item()['covariance_matrix']
    return covariance_matrix[0:8,0:8]

def chi_dof(path_model, path_file):
    data = np.load(path_file,allow_pickle=True)
    curvas, params = read_curves(path_model)
    if "Roman" in path_file:
        npoints = len(curvas['w'])
        chi2 = data.item()['chi2']
        best_model = data.item()['best_model']
        dof = npoints-len(best_model)
        return chi2/dof
    else:
        npoints = len(curvas['w'])+len(curvas['u'])+len(curvas['g'])+len(curvas['r'])+len(curvas['i'])+len(curvas['z'])+len(curvas['y'])
        chi2 = data.item()['chi2']
        best_model = data.item()['best_model']
        dof = npoints-len(best_model)
        return chi2/dof

def chi_dofRR(path_model, path_file):
    data = np.load(path_file,allow_pickle=True)
    curvas, params = read_curves(path_model)
    npoints = len(curvas['w'])+len(curvas['u'])+len(curvas['g'])+len(curvas['r'])+len(curvas['i'])+len(curvas['z'])+len(curvas['y'])
    chi2 = data.item()['chi2']
    best_model = data.item()['best_model']
    dof = npoints-len(best_model)
    return chi2/dof

def best_model(name_file):
    '''
    name_file(str):This function receives as input the name of the file
    example: /home/user/model/set_sim1/Event_RR_42_trf.npy.
    '''
    nset = int(name_file[name_file.index('Event_')-2:name_file.index('Event_')-1])
    name_set = 'set_sim'+str(nset)
    directory_simset = name_file[0:name_file.index('set')]+name_set+'/'
    nevent = re.sub(r'\D', '', name_file[name_file.index('Event_'):-1])
    model_file = directory_simset+'Event_'+str(nevent)+'.txt'
    curves,model_params = read_curves(model_file)
    data = np.load(name_file,allow_pickle=True)
    fit_params = {}
    for i,key in enumerate(model_params):
        fit_params[key] = data.item()['best_model'][i]
    return fit_params

def sigmas(name_file):
    '''
    name_file(str):This function receives as input the name of the file
    example: /home/user/model/set_sim1/Event_RR_42_trf.npy.
    '''
    nset = int(name_file[name_file.index('Event_')-2:name_file.index('Event_')-1])
    name_set = 'set_sim'+str(nset)
    directory_simset = name_file[0:name_file.index('set')]+name_set+'/'
    nevent = re.sub(r'\D', '', name_file[name_file.index('Event_'):-1])
    model_file = directory_simset+'Event_'+str(nevent)+'.txt'
    curves,model_params = read_curves(model_file)
    data = np.load(name_file,allow_pickle=True)
    covariance_matrix = data.item()['covariance_matrix']
    errors = np.sqrt(abs(np.diagonal(covariance_matrix)))
    error_params = {}
    for i,key in enumerate(model_params):
        error_params[key] = errors[i]
    return error_params
