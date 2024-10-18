from functions_roman_rubin import sim_fit
import numpy as np
# path_ephemerides = '/home/anibalvarela/ajustes/james_webb.txt'
# path_dataslice = '/home/anibalvarela/opsims/baseline/dataSlice.npy'
# path_storage = '/share/storage3/rubin/microlensing/romanrubin'
# j=1
# path_TRILEGAL_set =f'/home/anibalvarela/TRILEGAL/PB_planet_split_{j}.csv'# "/home/anibal/results_roman_rubin/PB_planet_split_1.csv"FFP_uni_split_3.csv
# path_to_save_model = path_storage+f"/PB/set_sim{j}/"
# path_to_save_fit = path_storage+f"/PB/set_fit{j}/"
# model="USBL"
# algo="TRF"

i=0
model='USBL'
algo='TRF'
path_TRILEGAL_set= '/home/anibal/roman_rubin_paper/TRILEGAL/PB_planet_split_1.csv'
path_to_save_model='/home/anibal/roman_rubin_paper/test_sim_fit/'
path_to_save_fit= '/home/anibal/roman_rubin_paper/test_sim_fit/'
path_ephemerides= '/home/anibal/files_db/james_webb.txt'
path_dataslice = '/home/anibal/roman_rubin_paper/opsims/baseline/dataSlice.npy'
sim_fit(i,model,algo,path_TRILEGAL_set,path_to_save_model,path_to_save_fit,path_ephemerides,path_dataslice)