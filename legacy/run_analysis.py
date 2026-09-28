from analysis import fit_true, chichi_to_fits_files, piE_cov_terms, categories_function
from pathlib import Path
import os

labels_params: list[str] = ['t0','u0','te','rho',"s","q","alpha",'piEN','piEE']
#labels_params: list[str] = ['t0','u0','te','rho','piEN','piEE']
#labels_params: list[str] = ['t0','u0','te','piEN','piEE']
script_dir = str(Path(__file__).parent)
print(script_dir)


path_ephemerides = script_dir+'/ajustes/Gaia.txt'
path_storage = '/share/storage3/rubin/microlensing/romanrubin/'
path_set = 'PB/'
path = path_storage+path_set
path_dataslice = script_dir+'/opsims/baseline/dataSlice.npy'


# def save_data(labels_params): # path in the CHE cluster

if len(labels_params)==5:
    save_results = script_dir+'/all_results/BH/'+path_set
    os.makedirs(save_results, exist_ok=True)
elif len(labels_params)==6:
    save_results = script_dir+'/all_results/FFP/'+path_set
    os.makedirs(save_results, exist_ok=True)
elif len(labels_params)==9:
    save_results = script_dir+'/all_results/PB/'+'PB_MCprop'#path_set
    os.makedirs(save_results, exist_ok=True)

# path_model = ['set_sim'+str(i)+'/' for i in range(1,9)]
# path_fit = ['set_fit'+str(i)+'/' for i in range(1,9)]
# path_set_sim = [path+'set_sim'+str(i)+'/' for i in range(1,9)]
# path_set_fit = [path+'set_fit'+str(i)+'/' for i in range(1,9)]


true, fit_rr, fit_roman = fit_true(path, labels_params)
fit_rr1, fit_roman1 = chichi_to_fits_files(path, fit_rr, fit_roman)
fit_rr2, fit_roman2 = piE_cov_terms(path,fit_rr1,fit_roman1, labels_params)
true1 = categories_function(true, path_dataslice)

fit_rr2.to_csv(save_results+'fit_rr_ffp.csv', index=False)
fit_roman2.to_csv(save_results+'fit_roman_ffp.csv', index=False)
true1.to_csv(save_results+'true_ffp.csv', index=False)
