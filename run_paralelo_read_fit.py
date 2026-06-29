from sim_fit_parallelization import run_parallel_read_fit
import sys, os
from pathlib import Path
import json

script_dir = Path(__file__).parent
path_confg_file =str(script_dir)+'/fit_config_file.json'

with open(path_confg_file) as f:
        params = json.load(f)

sys.path.append(str(script_dir)+'/photutils/')
path_ephemerides = str(script_dir)+'/ephemerides/Roman_positions.npy'

path_run = params["path_storage"]+"/"+params["system_type"]+"/"
model = params["model"]
nset = params['n_db_file']
path_to_save = params['path_save_fit']
str_spec = '/double_fit/'
path_to_save_fit = path_to_save+str_spec+f"/set_fit{nset}/"
if not os.path.exists(path_to_save_fit):
    os.makedirs(path_to_save_fit)

algo = params["algo"]
N_tr=40
run_parallel_read_fit(nset, path_run, path_ephemerides, path_to_save_fit, model, algo, N_tr)
