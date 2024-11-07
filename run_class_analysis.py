from pathlib import Path
from class_analysis import Analysis_Event
import pandas as pd 
import os


script_dir = str(Path(__file__).parent)
path_TRILEGAL = str(Path(__file__).parent)+f'/TRILEGAL/PB_planet_split_{1}.csv'
path_fit_rr = "/home/anibal/roman_rubin/test_sim_fit/Event_RR_18_TRF.npy"
path_fit_roman = "/home/anibal/roman_rubin/test_sim_fit/Event_Roman_18_TRF.npy"
path_model = "/home/anibal/roman_rubin/test_sim_fit/Event_18.h5"

trilegal_params = pd.read_csv(path_TRILEGAL).iloc[0]

path_dataslice = script_dir+'/opsims/baseline/dataSlice.npy'

Event = Analysis_Event("USBL", path_model, path_fit_rr, path_fit_roman,
                       path_dataslice, trilegal_params)
true, fit_rr, fit_roman = Event.fit_true()

# print(fit_rr)
# print(Event.read_data())
# print(Event.chichi())
# print(Event.MC_propagation_piE())
# print(Event.piE())
# print(Event.categories_function())
print(Event.mass_MC())

