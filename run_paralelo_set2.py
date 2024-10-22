from sim_fit_paralelos import para_elisa
path_ephemerides = '/home/anibalvarela/ajustes/james_webb.txt'
path_dataslice = '/home/anibalvarela/opsims/baseline/dataSlice.npy'
path_storage = '/share/storage3/rubin/microlensing/romanrubin'
j=2
path_TRILEGAL_set =f'/home/anibalvarela/TRILEGAL/BH_split_{j}.csv'#PB_planet_split_{j}.csv'# "/home/anibal/results_roman_rubin/PB_planet_split_1.csv"FFP_uni_split_3.csv
path_to_save_model = path_storage+f"/BH/set_sim{j}/"
path_to_save_fit = path_storage+f"/BH/set_fit{j}/"
model='PSPL'#"USBL"
algo="TRF"


N_tr=36
para_elisa(path_ephemerides,path_dataslice,path_TRILEGAL_set,path_to_save_fit,path_to_save_model,model,algo, N_tr)
