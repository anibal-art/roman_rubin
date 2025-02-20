import os, sys
import numpy as np
import matplotlib.pyplot as plt
from pyLIMA.outputs import pyLIMA_plots
from cycler import cycler
import pandas as pd
# sys.path.append(os.path.dirname(os.getcwd()))
from functions_roman_rubin import sim_fit,sim_event
from functions_roman_rubin import model_rubin_roman, fit_rubin_roman
from functions_roman_rubin import read_data, save, filter5points


def deviation_from_constant(pyLIMA_parameters, pyLIMA_telescopes):
    '''
     There at least four points in the range
     $[t_0-tE, t_0+t_E]$ with the magnification deviating from the
     constant flux by more than 3$\sigma$
    '''
    ZP = {'W149': 27.615, 'u': 27.03, 'g': 28.38, 'r': 28.16,
          'i': 27.85, 'z': 27.46, 'y': 26.68}
    t0 = pyLIMA_parameters['t0']
    tE = pyLIMA_parameters['tE']
    satis_crit = {}
    for telo in pyLIMA_telescopes:
        if not len(pyLIMA_telescopes[telo]['mag']) == 0:
            mag_baseline = ZP[telo] - 2.5 * np.log10(pyLIMA_parameters['ftotal_' + f'{telo}'])
            x = pyLIMA_telescopes[telo]['time'].value
            y = pyLIMA_telescopes[telo]['mag'].value
            z = pyLIMA_telescopes[telo]['err_mag'].value
            mask = (t0 - tE < x) & (x < t0 + tE)
            consec = []
            if len(x[mask]) >= 3:
                combined_lists = list(zip(x[mask], y[mask], z[mask]))
                sorted_lists = sorted(combined_lists, key=lambda item: item[0])
                sorted_x, sorted_y, sorted_z = zip(*sorted_lists)
                for j in range(len(sorted_y)):
                    if sorted_y[j] + 3 * sorted_z[j] < mag_baseline:
                        consec.append(j)
                result = has_consecutive_numbers(consec)
                if result:
                    satis_crit[telo] = True
                else:
                    satis_crit[telo] = False
            else:
                satis_crit[telo] = False
        else:
            satis_crit[telo] = False
    return any(satis_crit.values())


def has_consecutive_numbers(lst):
    """
    check if there at least 3 consecutive numbers in a list lst
    """
    sorted_lst = sorted(lst)
    for i in range(len(sorted_lst) - 2):
        if sorted_lst[i] + 1 == sorted_lst[i + 1] == sorted_lst[i + 2] - 1:
            return True
    return False


path_dir = os.getcwd()
path_storage = '/share/storage3/rubin/microlensing/romanrubin/PB/'
# path_set = 'PB/'

