from pathlib import Path
from class_analysis import Analysis_Event, labels_params, event_fits
import pandas as pd 
import os, re
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")


# Here I set the model path
model = "USBL"
path_run = '/share/storage3/rubin/microlensing/romanrubin/PB'

script_dir = str(Path(__file__).parent)
print(script_dir)
path_dataslice = script_dir+'/opsims/baseline/dataSlice.npy'

save_results = script_dir + "/all_results/"+model+"/"
os.makedirs(save_results, exist_ok=True)

# columnas de df
cols_true = ['Source', 'Set'] + labels_params(model) + ['Category','mass']


cols_fit = ['Source', 'Set'] + labels_params(model)  + \
[f+'_err' for f in labels_params(model) ]+\
['piE', 'piE_err', 'piE_err_MC']+\
['mass_thetaE','mass_mu', 'mass_thetaS','err_mass_thetaE_NotMC', 
 'mass_err_thetaE' ,'mass_err_mu','mass_err_thetaS', 'chichi', 'dof']

# filenames
filename_true = save_results+'true.csv'
filename_fit_rr = save_results+'fit_rr.csv'
filename_fit_roman = save_results+'fit_roman.csv'

if not os.path.exists(filename):
    true_df = pd.DataFrame(columns=cols_true)
    fit_rr_df = pd.DataFrame(columns=cols_fit)
    fit_roman_df = pd.DataFrame(columns=cols_fit)

    true_df.to_csv(filename_true, index=False)
    fit_roman_df.to_csv(filename_fit_roman, index=False)
    fit_rr_df.to_csv(filename_fit_rr, index=False)


for SET in tqdm(range(1,5)):
    nset=SET
    available_events = event_fits(path_run+f"/set_fit{nset}")
    for nevent in tqdm(available_events):

        
        if model == "USBL":
            path_TRILEGAL = str(Path(__file__).parent)+f'/TRILEGAL/PB_planet_split_{nset}.csv'
        elif model == "PSPL":
            path_TRILEGAL = str(Path(__file__).parent)+f'/TRILEGAL/BH_split_{nset}.csv'
        elif model == "FSPL":
            path_TRILEGAL = str(Path(__file__).parent)+f'/TRILEGAL/FFP_uni_split_{nset}.csv'
    

        path_fit_rr = path_run+f"/set_fit{nset}"+f"/Event_RR_{nevent}_TRF.npy"
        path_fit_roman = path_run+f"/set_fit{nset}"+f"/Event_Roman_{nevent}_TRF.npy"
        path_model = path_run+f"/set_sim{nset}"+f"/Event_{nevent}.h5"

        trilegal_params = pd.read_csv(path_TRILEGAL).iloc[nevent]
    
        Event = Analysis_Event(model, path_model, path_fit_rr, path_fit_roman,
                               path_dataslice, trilegal_params)
        
        true, fit_rr, fit_roman = Event.fit_true()
        piE_rr, err_piE_rr, piE_roman, err_piE_roman = Event.piE()
        chi_rr, chi_roman, dof_rr, dof_roman = Event.chichi()
        #err_piE_rr_MC , err_piE_roman_MC =  Event.MC_propagation_piE()
        #dict_mass = Event.mass_MC()
        
        new_data_true = pd.DataFrame(columns=cols_true)
        #new_row = {}
        new_data_true['Source'] = nevent
        new_data_true['Set'] = nset
        for key in Event.labels_params():
            new_data_true[key]=true[key]
        
        new_data_true['Category'] = Event.categories_function()
        new_data_true['mass'] = Event.mass_true()
        #true_df = pd.concat([true_df, pd.DataFrame([new_row])], ignore_index=True)
        new_data.to_csv(filename_true, mode="a", header=False, index=False)
        
        # df de Roman+Rubin
        new_data_rr = pd.DataFrame(columns=cols_fit)
        
        new_data_rr['Source'] = nevent
        new_data_rr['Set'] = nset
        for key in Event.labels_params():
            new_data_rr[key]=fit_rr[key]
            new_data_rr[key+'_err']=fit_rr[key+'_err']
        
        
        new_data_rr['piE'] = piE_rr
        new_data_rr['piE_err'] = err_piE_rr
        #new_row['piE_err_MC'] = err_piE_rr_MC
        new_data_rr['chichi'] = chi_rr
        new_data_rr['dof'] = dof_rr
        
        #new_row['mass_thetaE'] = Event.fit_mass_rr1()
        #new_row['mass_mu'] = Event.fit_mass_rr2()
        #if 'rho' in labels_params(model):
        #    new_row['mass_thetaS'] = Event.fit_mass_rr3()
        
            
        #new_row['err_mass_thetaE_NotMC'] = Event.formula_mass_uncertainty_rr()
        #new_row['mass_err_thetaE'] = dict_mass['sigma_m_thetaE_rr']
        #new_row['mass_err_mu'] = dict_mass['sigma_m_mu_rr']
        #new_row['mass_err_thetaS'] = dict_mass['sigma_m_thetaS_rr']
        new_data_rr.to_csv(filename_fit_rr, mode="a", header=False, index=False)
        #fit_rr_df = pd.concat([fit_rr_df, pd.DataFrame([new_row])], ignore_index=True)
        
        
        # df de Roman 
        new_data_roman = pd.DataFrame(columns=cols_fit)
        #new_row = {}
        new_data_roman['Source']=nevent
        new_data_roman['Set']=nset
        for key in Event.labels_params():
            new_data_roman[key]=fit_roman[key]
            new_data_roman[key+'_err']=fit_roman[key+'_err']
        
        
        new_data_roman['piE'] = piE_roman
        new_data_roman['piE_err'] = err_piE_roman
        #new_row['piE_err_MC'] = err_piE_roman_MC
        new_data_roman['chichi'] = chi_roman
        new_data_roman['dof'] = dof_roman
        
        #new_row['mass_thetaE'] = Event.fit_mass_roman1()
        #new_row['mass_mu'] = Event.fit_mass_roman2()
        #if 'rho' in labels_params(model):
        #    new_row['mass_thetaS'] = Event.fit_mass_roman3()
        
        #new_row['err_mass_thetaE_NotMC'] = Event.formula_mass_uncertainty_roman()
        #new_row['mass_err_thetaE'] = dict_mass['sigma_m_thetaE_roman']
        #new_row['mass_err_mu'] = dict_mass['sigma_m_mu_roman']
        #new_row['mass_err_thetaS'] = dict_mass['sigma_m_thetaS_roman']
        
        new_data_roman.to_csv(filename_fit_roman, mode="a", header=False, index=False)




