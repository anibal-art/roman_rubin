#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:43:01 2024

@author: anibal
"""

from analysis import fit_true, chichi_to_fits_files, piE_cov_terms, categories_function
from pathlib import Path
#labels_params: list[str] = ['t0','u0','te','rho',"s","q","alpha",'piEN','piEE']
labels_params: list[str] = ['t0','u0','te','piEN','piEE']
script_dir = str(Path(__file__).parent)
print(script_dir)

path_ephemerides = script_dir+'/ajustes/Gaia.txt'
path = '/share/storage3/rubin/microlensing/romanrubin/BH/' # path in the CHE cluster

if len(labels_params)==5:
    save_results = script_dir+'/all_results/BH/results/'
elif len(labels_params)==6:
    save_results = script_dir+'/all_results/FFP/results/'
elif len(labels_params)==9:
    save_results = script_dir+'/all_results/PB/results/'

path_dataslice =script_dir+'/opsims/baseline/dataSlice.npy'
# nominal_seasons = [
#     {'start': '2027-02-11T00:00:00', 'end': '2027-04-24T00:00:00'},
#     {'start': '2027-08-16T00:00:00', 'end': '2027-10-27T00:00:00'},
#     {'start': '2028-02-11T00:00:00', 'end': '2028-04-24T00:00:00'},
#     {'start': '2030-02-11T00:00:00', 'end': '2030-04-24T00:00:00'},
#     {'start': '2030-08-16T00:00:00', 'end': '2030-10-27T00:00:00'},
#     {'start': '2031-02-11T00:00:00', 'end': '2031-04-24T00:00:00'},
# ]

path_model = ['set_sim'+str(i)+'/' for i in range(1,9)]
path_fit = ['set_fit'+str(i)+'/' for i in range(1,9)]
path_set_sim = [path+'set_sim'+str(i)+'/' for i in range(1,9)]
path_set_fit = [path+'set_fit'+str(i)+'/' for i in range(1,9)]


true, fit_rr, fit_roman = fit_true(path)
fit_rr1, fit_roman1 = chichi_to_fits_files(path, fit_rr, fit_roman)
fit_rr2, fit_roman2 = piE_cov_terms(path,fit_rr1,fit_roman1)
true1 = categories_function(true, path_dataslice)

fit_rr2.to_csv(save_results+'fit_rr_ffp.csv', index=False)
fit_roman2.to_csv(save_results+'fit_roman_ffp.csv', index=False)
true1.to_csv(save_results+'true_ffp.csv', index=False)
