from sim_fit_paralelos import para_elisa
import sys, os
from pathlib import Path

# Get the directory where the script is located
script_dir = Path(__file__).parent
print(script_dir)
sys.path.append(str(script_dir)+'/photutils/')

path_ephemerides = str(script_dir)+'/ephemerides/Gaia.txt'
path_dataslice = str(script_dir)+'/opsims/baseline/dataSlice.npy'
path_storage = '/share/storage3/rubin/microlensing/romanrubin/RR2025/'
#/share/storage3/rubin/microlensing/romanrubin/RR2025
j=1

model="USBL"     #'FSPL' (Free Floating Planets)#"USBL" (Binary Lens-planetary systems) #"PSPL" (Black Holes)
if model =="USBL":
	TRILEGAL_file =f"PB_planet_split_{j}.csv"
elif model=='PSPL':
	TRILEGAL_file =f"BH_split_{j}.csv"
elif model =='FSPL':
	TRILEGAL_file =f'FFP_uni_split_{j}.csv'	

path_TRILEGAL_set =str(script_dir)+f'/TRILEGAL/'+TRILEGAL_file 

path_to_save_model = path_storage+model+f"/set_sim{j}/"
path_to_save_fit = path_storage+model+f"/set_fit{j}/"

# Create a directory if it doesn't exist
if not os.path.exists(path_to_save_model):
    os.makedirs(path_to_save_model)
if not os.path.exists(path_to_save_fit):
    os.makedirs(path_to_save_fit)

algo="TRF"

N_tr=36
para_elisa(path_ephemerides,path_dataslice,path_TRILEGAL_set,path_to_save_fit,path_to_save_model,model,algo, N_tr)
