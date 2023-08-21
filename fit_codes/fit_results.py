import pandas as pd
import numpy as np
#from filter_curves import filtros
from filter_curves import filtros, read_curves
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

def errors(name_file):
    data = np.load(name_file,allow_pickle=True)
    covariance_matrix = data.item()['covariance_matrix']
    errors = np.sqrt(abs(np.diagonal(covariance_matrix)))
    return errors

def best_model(name_file):
    data = np.load(name_file,allow_pickle=True)
    return data.item()['best_model'][0:8]

def best_full_model(name_file):
    data = np.load(name_file,allow_pickle=True)
    return data.item()['best_model']

