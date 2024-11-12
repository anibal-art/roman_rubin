from pathlib import Path
from class_analysis import Analysis_Event
import pandas as pd 
import os


script_dir = str(Path(__file__).parent)



# for SET in sets:
nset=1
nevent=18
path_TRILEGAL = str(Path(__file__).parent)+f'/TRILEGAL/PB_planet_split_{nset}.csv'
path_fit_rr = str(Path(__file__).parent)+"/test_sim_fit"+f"/Event_RR_{nevent}_TRF.npy"
path_fit_roman = str(Path(__file__).parent)+"/test_sim_fit"+f"/Event_Roman_{nevent}_TRF.npy"
path_model = str(Path(__file__).parent)+"/test_sim_fit"+f"/Event_{nevent}.h5"
trilegal_params = pd.read_csv(path_TRILEGAL).iloc[18]
path_dataslice = script_dir+'/opsims/baseline/dataSlice.npy'

Event = Analysis_Event("USBL", path_model, path_fit_rr, path_fit_roman,
                       path_dataslice, trilegal_params)

true, fit_rr, fit_roman = Event.fit_true()
piE_rr, err_piE_rr, piE_roman, err_piE_roman = Event.piE()
chi_rr, chi_roman, dof_rr, dof_roman = Event.chichi()

dict_mass = Event.mass_MC()
#%%

# print(fit_rr)
# print(Event.read_data())
# print(Event.chichi())
# print(Event.MC_propagation_piE())
# print(Event.piE())
# print(Event.categories_function())
# print(Event.mass_MC())
# print(Event.labels_params())

#%%
cols_true = ['Source', 'Set'] + Event.labels_params() + ['Category']
true_df = pd.DataFrame(columns=cols_true)

new_row = {}
new_row['Source']=nevent
new_row['Set']=nevent
for key in Event.labels_params():
    new_row[key]=true[key]

new_row['Category'] = Event.categories_function()
true_df = pd.concat([true_df, pd.DataFrame([new_row])], ignore_index=True)

#%%

cols_fit = ['Source', 'Set'] + Event.labels_params() + \
[f+'_err' for f in Event.labels_params()]+\
['piE', 'piE_err', 'piE_err_MC']+\
['mass','mass_err', 'mass_err_MC_rho','mass_err_MC_te', 'chichi', 'dof']

fit_rr_df = pd.DataFrame(columns=cols_fit)

new_row = {}
new_row['Source']=nevent
new_row['Set']=nevent
for key in Event.labels_params():
    new_row[key]=fit_rr[key]


new_row['piE'] = piE_rr
new_row['piE_err'] = err_piE_rr
new_row['piE_err_MC'] = err_piE_rr
new_row['chichi'] = chi_rr
new_row['dof'] = dof_rr

new_row['mass_thetaE'] = Event.fit_mass_rr1()
new_row['mass_mu'] = Event.fit_mass_rr2()
new_row['mass_thetaS'] = Event.fit_mass_rr3()

new_row['err_mass_thetaE_NotMC'] = Event.formula_mass_uncertainty_rr()
new_row['mass_err_thetaE'] = dict_mass['sigma_m_thetaE_rr']
new_row['mass_err_mu'] = dict_mass['sigma_m_mu_rr']
new_row['mass_err_thetaS'] = dict_mass['sigma_m_thetaS_rr']


new_row['mass_thetaS'] = dict_mass['sigma_m_thetaS_rr']
new_row['mass_mu'] = dict_mass['sigma_m_mu_rr']
new_row['mass_thetaE'] = dict_mass['sigma_m_thetaE_rr']


fit_rr_df = pd.concat([fit_rr_df, pd.DataFrame([new_row])], ignore_index=True)

#%%
fit_roman_df = pd.DataFrame(columns=cols_fit)

new_row = {}
new_row['Source']=nevent
new_row['Set']=nevent
for key in Event.labels_params():
    new_row[key]=fit_rr[key]


new_row['piE'] = piE_roman
new_row['piE_err'] = err_piE_roman
new_row['piE_err_MC'] = err_piE_roman
new_row['chichi'] = chi_roman
new_row['dof'] = dof_roman

new_row['mass_thetaE'] = Event.fit_mass_roman1()
new_row['mass_mu'] = Event.fit_mass_roman2()
new_row['mass_thetaS'] = Event.fit_mass_roman3()

new_row['err_mass_thetaE_NotMC'] = Event.formula_mass_uncertainty_roman()
new_row['mass_err_thetaE'] = dict_mass['sigma_m_thetaE_roman']
new_row['mass_err_mu'] = dict_mass['sigma_m_mu_roman']
new_row['mass_err_thetaS'] = dict_mass['sigma_m_thetaS_roman']

fit_roman_df = pd.concat([fit_roman_df, pd.DataFrame([new_row])], ignore_index=True)

    