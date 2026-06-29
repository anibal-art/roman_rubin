from create_df_analysis import create_df
import json 
from pathlib import Path

script_dir = str(Path(__file__).parent)
path_confg_file = script_dir+"/analysis_config_file.json"

with open(path_confg_file) as f:
        params = json.load(f)

# Here I set the model path
model = params["model"] # "FSPL"
system_type = params["system_type"]#'Planets_systems'


path_run = params["path_run"]+'/'#'/share/storage3/rubin/microlensing/romanrubin/RR2025/April_set/FFP/'
description = params["description"]

save_results = script_dir + "/all_results/"+model+"_"+description+"/"
create_df(path_run, save_results, model, system_type)
