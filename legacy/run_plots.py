import numpy as np
import pandas as pd
import os, sys, re, math, h5py

from pyLIMA.outputs import pyLIMA_plots

sys.path.append(os.path.dirname(os.getcwd()))
from functions_roman_rubin import read_data, model_rubin_roman

from bokeh.plotting import figure, show, output_file
from bokeh.layouts import gridplot, row, column
from bokeh.io import export_png
from plot_saveLC import plot_n_save

path_ephemerides = str(os.getcwd())+'/ephemerides/Roman_positions.npy'
path_run = '/share/storage3/rubin/microlensing/romanrubin/RR2025/April_set/Planets_systems/'
model = "USBL"

print(path_ephemerides)
print(model_rubin_roman)

path_save = str(os.getcwd())+'/html_plots/'
path_gb =  path_save + 'good_bias/'
path_bb = path_save + 'bad_bias/'

path_gsigma =  path_save + 'good_uncertanty/'
path_bsigma = path_save + 'bad_uncertainty/'

path_gchi2 =  path_save + 'good_chi2/'
path_bchi2 = path_save + 'bad_chi2/'

paths = [path_save,path_gb,path_bb,path_gsigma,path_bsigma,path_gchi2,path_bchi2]
for p in paths:
    os.makedirs(p, exist_ok=True)

path = str(os.getcwd())+'/all_results/USBL/RR2025/'

fit_rr = pd.read_csv(path+"/fit_rr.csv", engine='python')
fit_roman =  pd.read_csv(path+"/fit_roman.csv", engine='python')
true = pd.read_csv(path+"/true.csv", engine='python')

true['categories']=true['Category']
fit_rr['chi2']=fit_rr['chichi']
fit_roman['chi2']=fit_roman['chichi']

fit_rr['t_center']=fit_rr['t0']
fit_roman['t_center']=fit_roman['t0']
fit_rr['u_center']=fit_rr['u0']
fit_roman['u_center']=fit_roman['u0']
fit_rr['t_center_err']=fit_rr['t0_err']
fit_roman['t_center_err']=fit_roman['t0_err']
fit_rr['u_center_err']=fit_rr['u0_err']
fit_roman['u_center_err']=fit_roman['u0_err']

fit_rr["piE"]=np.sqrt(fit_rr["piEE"]**2+fit_rr["piEN"]**2)
fit_roman["piE"]=np.sqrt(fit_roman["piEE"]**2+fit_roman["piEN"]**2)
true["piE"]=np.sqrt(true["piEE"]**2+true["piEN"]**2)

fit_rr['id'] = fit_rr['Source']+5000*fit_rr['Set']
fit_roman['id'] = fit_roman['Source']+5000*fit_roman['Set']
true['id'] = true['Source']+5000*true['Set']

met_1_rr = pd.DataFrame(columns = true.columns)
met_1_roman= pd.DataFrame(columns = true.columns)
met_2_rr = pd.DataFrame(columns = true.columns)
met_2_roman= pd.DataFrame(columns = true.columns)
met_3_rr = pd.DataFrame(columns = true.columns)
met_3_roman= pd.DataFrame(columns = true.columns)
err_ratio= pd.DataFrame(columns = true.columns)
residuals_ratio= pd.DataFrame(columns = true.columns)

err_ratio['id'] = true['id']
residuals_ratio['id'] = true['id']
met_1_roman['id'] = true['id']
met_1_rr['id'] = true['id']
met_2_roman['id'] = true['id']
met_2_rr['id'] = true['id']
met_3_roman['id'] = true['id']
met_3_rr['id'] = true['id']

keys = ['t_center','u_center','tE','rho','separation',
        'mass_ratio','alpha',
        'piEN','piEE','piE']
for key in ['Source',	'Set','categories','mass','sel_crit']:
    met_1_rr[key] = true[key]
    met_1_roman[key] = true[key]
    met_2_rr[key] = true[key]
    met_2_roman[key] = true[key]
    met_3_rr[key] = true[key]
    met_3_roman[key] = true[key]
    err_ratio[key] = true[key]
    residuals_ratio[key] = true[key]
    
for key in keys:
    if key == 't_center':
        met_1_rr[key] = abs(true[key]-fit_rr[key])
        met_1_roman[key] = abs(true[key]-fit_roman[key])
        met_2_rr[key] = abs(true[key]-fit_rr[key])/fit_rr[key+'_err']
        met_2_roman[key] = abs(true[key]-fit_roman[key])/fit_roman[key+'_err']
        met_3_rr[key] = abs(fit_rr[key+'_err']/fit_rr[key])
        met_3_roman[key] = abs(fit_roman[key+'_err']/fit_roman[key])
        err_ratio[key]=abs(fit_rr[key+'_err'])/fit_roman[key+'_err']
        residuals_ratio[key]=abs(fit_rr[key+'_err'])/fit_roman[key+'_err']
        
    else:
        met_1_rr[key] = abs(true[key]-fit_rr[key])/abs(true[key])
        met_1_roman[key] = abs(true[key]-fit_roman[key])/abs(true[key])
        met_2_rr[key] = abs(true[key]-fit_rr[key])/fit_rr[key+'_err']
        met_2_roman[key] = abs(true[key]-fit_roman[key])/fit_roman[key+'_err']
        met_3_rr[key] = abs(fit_rr[key+'_err']/fit_rr[key])
        met_3_roman[key] = abs(fit_roman[key+'_err']/fit_roman[key])
        err_ratio[key]=abs(fit_rr[key+'_err'])/fit_roman[key+'_err']
        residuals_ratio[key]=abs(fit_rr[key+'_err'])/fit_roman[key+'_err']

met_1_rr['Category']=true['Category']
met_1_roman['Category']=true['Category']

bad_chi2 = fit_rr[(fit_rr['chi2']/fit_rr['dof'])>20] # in RR
bad_met1 = met_1_rr[met_1_rr['t_center']>0.99]
bad_sigma = met_3_rr[met_3_rr['t_center']>10]

good_chi2 = fit_rr[(fit_rr['chi2']/fit_rr['dof'])<1.2] # in RR
good_met1 = met_1_rr[met_1_rr['tE']<0.05]
good_sigma = met_3_rr[met_3_rr['tE']<0.1]

datas = [bad_chi2, bad_met1, bad_sigma, good_chi2,good_met1, good_sigma]
paths_to_save = [path_bchi2, path_bb, path_bsigma, path_gchi2, path_gb, path_gsigma]

for j, df in enumerate(datas):
    for i in range(4):
        data = df.iloc[i]
        nset = data['Set']
        source = data['Source']
        
        path_setsim = path_run+f'set_sim{int(nset)}/'
        path_setfit = path_run+f'set_fit{int(nset)}/'
        
        path_event = path_setsim+f'Event_{int(source)}.h5'
        
        path_fit_rr = path_setfit+f'Event_RR_{int(source)}_TRF.npy'
        path_fit_roman = path_setfit+f'Event_Roman_{int(source)}_TRF.npy'
        
        plot_n_save(int(source+nset*5000), paths_to_save[j], path_event, path_fit_rr, path_fit_roman, path_ephemerides)
