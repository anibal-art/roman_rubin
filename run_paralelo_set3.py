from sim_fit_paralelos import para_elisa
import sys, os
from pathlib import Path

# Get the directory where the script is located
script_dir = Path(__file__).parent
print(script_dir)
sys.path.append(str(script_dir)+'/photutils/')

path_ephemerides = str(script_dir)+'/ephemerides/Gaia.txt'
path_dataslice = str(script_dir)+'/opsims/baseline/dataSlice.npy'
path_storage = '/share/storage3/rubin/microlensing/romanrubin'
j=3
path_TRILEGAL_set =str(script_dir)+f'/TRILEGAL/FFP_uni_split_{j}.csv'#PB_planet_split_{j}.csv'# "/home/anibal/results_roman_rubin/PB_planet_split_1.csv"FFP_uni_split_3.csv
path_to_save_model = path_storage+f"/test/set_sim{j}/"
path_to_save_fit = path_storage+f"/test/set_fit{j}/"
# Create a directory if it doesn't exist

if not os.path.exists(path_to_save_model):
    os.makedirs(path_to_save_model)
if not os.path.exists(path_to_save_fit):
    os.makedirs(path_to_save_fit)


model='PSPL'#"USBL"
algo="TRF"


N_tr=36
para_elisa(path_ephemerides,path_dataslice,path_TRILEGAL_set,path_to_save_fit,path_to_save_model,model,algo, N_tr)
