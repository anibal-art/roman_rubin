from class_analysis import MicrolensingAnalysis
from pathlib import Path
import os

# Define labels for parameters
labels_params = ['t0', 'u0', 'te', 'rho', 'piEN', 'piEE']

# Determine paths
script_dir = str(Path(__file__).parent)
path_ephemerides = script_dir + '/ajustes/Gaia.txt'
path_storage = '/share/storage3/rubin/microlensing/romanrubin/'
path_set = 'test/'
path = path_storage + path_set
path_dataslice = script_dir + '/opsims/baseline/dataSlice.npy'

# Determine save_results path based on length of labels_params
if len(labels_params) == 5:
    save_results = script_dir + '/all_results/BH/' + path_set
elif len(labels_params) == 6:
    save_results = script_dir + '/all_results/FFP/' + path_set
elif len(labels_params) == 9:
    save_results = script_dir + '/all_results/PB/' + path_set

os.makedirs(save_results, exist_ok=True)

# Instantiate the class with parameters
analysis = MicrolensingAnalysis(
    path=path,
    labels_params=labels_params,
    save_results=save_results,
    path_dataslice=path_dataslice
)

# Run the analysis
analysis.run_analysis()
