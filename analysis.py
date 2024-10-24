import os, sys
import re
from typing import List
from astropy.time import Time
import h5py
from astropy.table import QTable
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import warnings


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


def new_rows(camino,st):
    data_rr = np.load(camino,allow_pickle=True).item()

    fit_values = dict(zip(labels_params,
                          data_rr['best_model'][0:len(labels_params)]))
    # print(np.sqrt(np.diag(data_rr['covariance_matrix']))[0:8])
    if any(np.diag(data_rr['covariance_matrix'])<0):
        fit_error = np.zeros(len(labels_params))
    else:
        fit_error= np.sqrt(np.diag(data_rr['covariance_matrix']))[0:len(labels_params)]

    for j,key in enumerate(labels_params):
        fit_values[key+'_err']=fit_error[j]
    fit_values['Source'] = data_rr['true_params'].name+st*5000

    true_values = data_rr['true_params'][labels_params].to_dict()#.values[0:9])
    true_values['Source']=data_rr['true_params'].name+st*5000

    new_row_true = pd.DataFrame([true_values])
    new_row_fit = pd.DataFrame([fit_values])
    return new_row_true, new_row_fit

def fit_true(path):
    cols_true = ['Source']+labels_params
    cols_fit=cols_true+[t+'_err' for t in labels_params]
    true = pd.DataFrame(columns=cols_true)
    fit_rr = pd.DataFrame(columns=cols_fit)
    fit_roman = pd.DataFrame(columns=cols_fit)
    fit_completed = []
    err = []
    for st in tqdm(range(1,5)):
        PATH = path+f'set_fit{st}/'
        nevent = event_fits(PATH)
        list_files_rr = [f'Event_RR_{int(f)}_TRF.npy' for f in nevent]
        list_files_roman = [f'Event_Roman_{int(f)}_TRF.npy' for f in nevent]

        for i in range(len(nevent)):
            path_rr = PATH+list_files_rr[i]
            path_roman = PATH+list_files_roman[i]
            try:
            # if i==4:
                new_row_true, new_row_rr = new_rows(path_rr,st)
                new_row_true2, new_row_roman = new_rows(path_roman,st)
                true = pd.concat([true, new_row_true], ignore_index=True)
                fit_rr = pd.concat([fit_rr, new_row_rr], ignore_index=True)
                fit_roman = pd.concat([fit_roman, new_row_roman], ignore_index=True)

                fit_completed.append(1)

            except Exception as e:
                print(f"Error with event {nevent[i]}: {e}")
                # err.append(1)
                # print(i)
                fit_completed.append(0)
    print(len(err))
    return true, fit_rr, fit_roman


def read_data(path_model):
    # Open the HDF5 file and load data using specified names
    with h5py.File(path_model, 'r') as file:
        # Load array with string with info of dataset using its name
        info_dataset = file['Data'][:]
        info_dataset = [file['Data'][:][0].decode('UTF-8'), file['Data'][:][1].decode('UTF-8'),
                        [file['Data'][:][2].decode('UTF-8'), [0, 0]]]
        # Dictionary using its name
        pyLIMA_parameters = {key: file['pyLIMA_parameters'].attrs[key] for key in file['pyLIMA_parameters'].attrs}
        # Load table using its name
        bands = {}
        for band in ("W149", "u", "g", "r", "i", "z", "y"):
            loaded_table = QTable()
            for col in file[band]:
                loaded_table[col] = file[band][col][:]
            bands[band] = loaded_table
        return info_dataset, pyLIMA_parameters, bands


def chichi(name_file):
    '''
    name_file(str):This function receives as input the name of the file
    example: /home/user/model/set_sim1/Event_RR_42_trf.npy.
    '''
    # print(name_file[name_file.index('Event_')-2])
    nset = int(name_file[name_file.index('Event_') - 2:name_file.index('Event_') - 1])

    # print(nset)
    name_set = 'set_sim' + str(nset)
    directory_simset = name_file[0:name_file.index('set')] + name_set + '/'
    # print()
    nevent = re.sub(r'\D', '', name_file[name_file.index('Event_'):-1])

    model_file = directory_simset + 'Event_' + str(nevent) + '.h5'
    info_dataset, model_params, curves = read_data(name_file)
    # curves,model_params = read_curves(model_file)
    name_file_rr = name_file[0:name_file.index('set')] + f"/set_fit{nset}/Event_RR_{nevent}_TRF.npy"
    name_file_roman = name_file[0:name_file.index('set')] + f"/set_fit{nset}/Event_Roman_{nevent}_TRF.npy"
    data_rr = np.load(name_file_rr, allow_pickle=True)
    data_roman = np.load(name_file_roman, allow_pickle=True)
    # print(data_rr)
    try:
        chi_rr = data_rr.item()["chi2"]
        chi_roman = data_roman.item()["chi2"]

        dof_rr = sum([len(curves[key]) for key in curves]) - len(
            [len(curves[key]) for key in curves if not len(curves[key]) == 0]) * 2 - 9
        dof_roman = len(curves['W149']) - 2 - 9
        # print(model_params)
        # print(chi_roman/dof)
        return chi_rr, chi_roman, dof_rr, dof_roman
    except:
        return 0, 0,0,0

def group_consecutive_numbers(numbers):
    '''
    Defino comienzo y finalizacion de temporadas de observacion de Rubin en el campo de Roman
    '''
    numbers.sort()
    groups = []
    current_group = [numbers[0]]

    for i in range(1, len(numbers)):
        if numbers[i] - current_group[-1] < 60:
            current_group.append(numbers[i])
        else:
            groups.append(current_group)
            current_group = [numbers[i]]

    if current_group:
        groups.append(current_group)

    return groups

def intervals_overlap(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2
    return (start1 <= end2 and end1 >= start2) or (start2 <= end1 and end2 >= start1)


def chichi_to_fits_files(path,fit_rr, fit_roman):
    id_to_chi2_rr = {}
    id_to_chi2_roman = {}
    id_to_dof_rr = {}
    id_to_dof_roman = {}
    for i in tqdm(range(1,9)):
        common_elements_list = event_fits(path+f"set_fit{i}/")
        if not len(common_elements_list)==0:
            for j in range(len(common_elements_list)):
                name_file = f"Event_{common_elements_list[j]}.h5"
                chi2rr, chi2roman,dof_rr,dof_roman = chichi(path+f"set_sim{i}/"+name_file)
                # print(chi2rr, chi2roman)
                id_to_chi2_rr[int(common_elements_list[j]+i*5000)] =chi2rr
                id_to_chi2_roman[int(common_elements_list[j]+i*5000)] =chi2roman
                id_to_dof_rr[int(common_elements_list[j]+i*5000)] = dof_rr
                id_to_dof_roman[int(common_elements_list[j]+i*5000)] = dof_roman
    fit_rr['chi2'] = fit_rr['Source'].map(id_to_chi2_rr)
    fit_roman['chi2'] = fit_roman['Source'].map(id_to_chi2_roman)
    fit_roman['dof']= fit_roman['Source'].map(id_to_dof_roman)
    fit_rr['dof']= fit_rr['Source'].map(id_to_dof_rr)
    return fit_rr, fit_roman


def piE_cov_terms(path,fit_rr,fit_roman):
    cov_piEE_piEN = {}
    cov_piEE_piEN_rom = {}
    if len(labels_params)==len(['t0','u0','te','rho',"s","q","alpha",'piEN','piEE']):
        indx_piE = [7,8]
    elif len(labels_params)==len(['t0','u0','te','rho','piEN','piEE']):
        indx_piE = [4,5]
    elif len(labels_params)==len(['t0','u0','te','piEN','piEE']):
        indx_piE = [3,4]
        
        
    for i in tqdm(range(len(fit_rr))):
        nsource = fit_rr["Source"].iloc[i]
        nset = int(nsource / 5000)
        nevent = nsource - nset * 5000
        data = np.load(path + f"set_fit{nset}/Event_RR_{nevent}_TRF.npy", allow_pickle=True)
        data_rom = np.load(path + f"set_fit{nset}/Event_Roman_{nevent}_TRF.npy", allow_pickle=True)
        best_model = data.item()['best_model']
        covariance_matrix = data.item()['covariance_matrix']
        cov_piEE_piEN[nsource] = covariance_matrix[indx_piE[0], indx_piE[1]]
        best_model_rom = data_rom.item()['best_model']
        covariance_matrix_rom = data_rom.item()['covariance_matrix']
        cov_piEE_piEN_rom[nsource] = covariance_matrix_rom[indx_piE[0], indx_piE[1]]


    fit_rr["cov_piEE_piEN"] = fit_rr['Source'].map(cov_piEE_piEN)
    fit_roman["cov_piEE_piEN"] = fit_rr['Source'].map(cov_piEE_piEN_rom)

    fit_rr['piE'] = np.sqrt(fit_rr['piEN'] ** 2 + fit_rr['piEE'] ** 2)
    fit_rr['piE_err'] = (1 / fit_rr['piE']) * np.sqrt((fit_rr['piEN_err'] * fit_rr['piEN']) ** 2 + (
                fit_rr['piEE_err'] * fit_rr['piEE']) ** 2)  # +2*fit_rr['piEE']*fit_rr['piEN']*fit_rr['cov_piEE_piEN'])
    fit_roman['piE'] = np.sqrt(fit_roman['piEN'] ** 2 + fit_roman['piEE'] ** 2)
    fit_roman['piE_err'] = (1 / fit_roman['piE']) * np.sqrt((fit_roman['piEN_err'] * fit_roman['piEN']) ** 2 + (
                fit_roman['piEE_err'] * fit_roman[
            'piEE']) ** 2)  # +2*fit_roman['piEE']*fit_roman['piEN']*fit_roman['cov_piEE_piEN'])
    true['piE'] = np.sqrt(true['piEN'] ** 2 + true['piEE'] ** 2)

    return fit_rr, fit_roman
# keys = labels_params

def categories_function(true,path_dataslice):
    dataSlice = np.load(path_dataslice, allow_pickle=True)
    # dataSlice['observationStartMJD']
    consecutive_numbers = dataSlice['observationStartMJD']
    result = group_consecutive_numbers(consecutive_numbers)
    rubin_seasons = []
    roman_seasons = []
    for group in result:
        rubin_seasons.append((min(group) + 2400000.5, max(group) + 2400000.5))
    for season in nominal_seasons:
        roman_seasons.append((Time(season['start'], format='isot').jd, Time(season['end'], format='isot').jd))

    categories = {}
    for i in range(len(true)):
        Source = true['Source'].iloc[i]
        t0 = true['t0'].iloc[i]
        tE = true['te'].iloc[i]
        interval1 = (t0 - tE, t0 + tE)
        overlap_rubin = False
        for j in range(len(rubin_seasons)):
            interval2 = rubin_seasons[j]
            if intervals_overlap(interval1, rubin_seasons[j]):
                overlap_rubin = True
                break
        overlap_roman = False
        for k in range(len(roman_seasons)):
            interval2 = roman_seasons[k]
            if intervals_overlap(interval1, roman_seasons[k]):
                overlap_roman = True
                break

        if (overlap_rubin == True) and (overlap_roman == True):
            categories[Source]='A'
        if (overlap_rubin == True) and (not overlap_roman == True):
            categories[Source]='B'
        if (not overlap_rubin == True) and (not overlap_roman == True):
            categories[Source]='C'
        if (not overlap_rubin == True) and (overlap_roman == True):
            categories[Source]='D'

    true['categories'] = true['Source'].map(categories)
    return true



from pathlib import Path
#labels_params: list[str] = ['t0','u0','te','rho',"s","q","alpha",'piEN','piEE']
labels_params: list[str] = ['t0','u0','te','piEN','piEE']
script_dir = str(Path(__file__).parent)
print(script_dir)

path_ephemerides = script_dir+'/ajustes/Gaia.txt'
path = '/share/storage3/rubin/microlensing/romanrubin/BH/' # path in the CHE cluster
save_results = path+'results/'
path_dataslice =script_dir+'/opsims/baseline/dataSlice.npy'
nominal_seasons = [
    {'start': '2027-02-11T00:00:00', 'end': '2027-04-24T00:00:00'},
    {'start': '2027-08-16T00:00:00', 'end': '2027-10-27T00:00:00'},
    {'start': '2028-02-11T00:00:00', 'end': '2028-04-24T00:00:00'},
    {'start': '2030-02-11T00:00:00', 'end': '2030-04-24T00:00:00'},
    {'start': '2030-08-16T00:00:00', 'end': '2030-10-27T00:00:00'},
    {'start': '2031-02-11T00:00:00', 'end': '2031-04-24T00:00:00'},
]

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
