from pathlib import Path
from class_analysis import Analysis_Event
import pandas as pd 
import os, re
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")


def labels_params(model):
    if model == "USBL":
        labels_params: list[str] = ['t0','u0','te','rho',"s","q","alpha",'piEN','piEE']
    elif model == "FSPL": 
        labels_params: list[str] = ['t0','u0','te','rho','piEN','piEE']
    elif model == "PSPL":
        labels_params: list[str] = ['t0','u0','te','piEN','piEE']
    return labels_params

def event_fits(path_fits):
    '''
    return events in common with roman and rubin
    we have events that fits only one of two for unknown reasons
    '''

    files_fits = os.listdir(path_fits)

    files_roman = [f for f in files_fits if 'Roman' in f]
    files_rr = [f for f in files_fits if not 'Roman' in f]

    n_rom = []  # list with the event number
    for j in files_roman:
        number = int(re.findall(r'\d+', j)[0])
        n_rom.append(number)

    n_rr = []  # # list with the event number
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


script_dir = str(Path(__file__).parent)
path_dataslice = script_dir+'/opsims/baseline/dataSlice.npy'
model = "USBL"

save_results = script_dir + "/test_USBL/"
os.makedirs(save_results, exist_ok=True)

path_run = '/share/storage3/rubin/microlensing/romanrubin/PB'
cols_true = ['Source', 'Set'] + labels_params(model) + ['Category']
true_df = pd.DataFrame(columns=cols_true)

cols_fit = ['Source', 'Set'] + labels_params(model)  + \
[f+'_err' for f in labels_params(model) ]+\
['piE', 'piE_err', 'piE_err_MC']+\
['mass_thetaE','mass_mu', 'mass_thetaS','err_mass_thetaE_NotMC', 
 'mass_err_thetaE' ,'mass_err_mu','mass_err_thetaS', 'chichi', 'dof']

fit_rr_df = pd.DataFrame(columns=cols_fit)
fit_roman_df = pd.DataFrame(columns=cols_fit)

for SET in tqdm(range(1,5)):
    nset=SET
    available_events = event_fits(path_run+f"/set_fit{nset}")
    for nevent in tqdm(available_events):
        # nevent=18

# 
        
        if model == "USBL":
            path_TRILEGAL = str(Path(__file__).parent)+f'/TRILEGAL/PB_planet_split_{nset}.csv'
        elif model == "PSPL":
            path_TRILEGAL = str(Path(__file__).parent)+f'/TRILEGAL/BH_split_{nset}.csv'
        elif model == "FSPL":
            path_TRILEGAL = str(Path(__file__).parent)+f'/TRILEGAL/FFP_uni_split_{nset}.csv'
    
        # path_fit_rr = path_run+"/test_sim_fit"+f"/Event_RR_{nevent}_TRF.npy"
        # path_fit_roman = path_run+"/test_sim_fit"+f"/Event_Roman_{nevent}_TRF.npy"
        # path_model = path_run+"/test_sim_fit"+f"/Event_{nevent}.h5"
        path_fit_rr = path_run+f"/set_fit{nset}"+f"/Event_RR_{nevent}_TRF.npy"
        path_fit_roman = path_run+f"/set_fit{nset}"+f"/Event_Roman_{nevent}_TRF.npy"
        path_model = path_run+f"/set_sim{nset}"+f"/Event_{nevent}.h5"

        trilegal_params = pd.read_csv(path_TRILEGAL).iloc[nevent]
    
        Event = Analysis_Event(model, path_model, path_fit_rr, path_fit_roman,
                               path_dataslice, trilegal_params)
        
        true, fit_rr, fit_roman = Event.fit_true()
        piE_rr, err_piE_rr, piE_roman, err_piE_roman = Event.piE()
        chi_rr, chi_roman, dof_rr, dof_roman = Event.chichi()
        
        dict_mass = Event.mass_MC()
        
        
        new_row = {}
        new_row['Source']=nevent
        new_row['Set']=nevent
        for key in Event.labels_params():
            new_row[key]=true[key]
        
        new_row['Category'] = Event.categories_function()
        true_df = pd.concat([true_df, pd.DataFrame([new_row])], ignore_index=True)
        
        
        
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
        
        fit_rr_df = pd.concat([fit_rr_df, pd.DataFrame([new_row])], ignore_index=True)
        
        
        
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



true_df.to_csv(save_results+'true.csv', index=False)
fit_roman_df.to_csv(save_results+'fit_roman.csv', index=False)
fit_rr_df.to_csv(save_results+'fit_rr.csv', index=False)
