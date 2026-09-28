from functions_roman_rubin import sim_fit, sim_event

i =1
system_type ="FFP"
model ="FSPL"
algo ="TRF"
path_TRILEGAL_set ="/home/anibal-pc/microlensing/simulation_Rubin/roman_rubin/chunks_TRILEGAL_GENULENS/TRILEGAL_chunk_1.csv"
path_GENULENS_set ="/home/anibal-pc/microlensing/simulation_Rubin/roman_rubin/chunks_TRILEGAL_GENULENS/Genulens_chunk_1.csv"
path_to_save_model ="/home/anibal-pc/microlensing/simulation_Rubin/roman_rubin/all_results/"
path_to_save_fit ="/home/anibal-pc/microlensing/simulation_Rubin/roman_rubin/all_results/"
path_ephemerides ="/home/anibal-pc/microlensing/simulation_Rubin/roman_rubin/ephemerides/Roman_positions.npy"
path_to_save_results ="/home/anibal-pc/microlensing/simulation_Rubin/roman_rubin/all_results/"


fit_rr, pyLIMAmodel_rr, fit_roman, pyLIMAmodel_roman = sim_fit(i, system_type, 
                                                               model, algo, path_TRILEGAL_set,
                                                               path_GENULENS_set, path_to_save_model,
                                                               path_to_save_fit, path_ephemerides, 
                                                               path_to_save_results)