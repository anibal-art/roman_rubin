import numpy as np
import pandas as pd
from read_results_ import read_results

model = 'FSPL'
system_type = "FFP"
true, fit_roman, fit_rr, met_1_roman, met_1_rr, met_2_roman, met_2_rr, met_3_roman, met_3_rr, met1_ratio, met2_ratio, met3_ratio =read_results(system_type, model)
mets_list = [
    (met_1_rr, met_1_roman),
    (met_2_rr, met_2_roman),
    (met_3_rr, met_3_roman),]
dict_cat_peak = {"A":"peak in Rubin+Roman",
                "B":"peak in Rubin",
                "C":"peak in Roman"}

p_labels = {
    "u0":r"$u_0$",
    "t0":r"$t_0$",
    "rho":r"$\rho$",
    "piE":r"$\pi_E$",
    "tE" : r"$t_E$"
}
cols = ['Source', 'Set']

from ulens_params import event_param
path_data = "/home/anibal-pc/microlensing/simulation_Rubin/roman_rubin/chunks_TRILEGAL_GENULENS/"



from tqdm import tqdm
true = true.reset_index(drop=True)
true["theta_E"] = np.nan
true["theta_source"] = np.nan
for j in tqdm(range(len(true))):
    Source = true["Source"].iloc[j]
    Set = true["Set"].iloc[j]

    data_TRILEGAL = path_data + f"/TRILEGAL_chunk_{int(Set)}.csv"
    data_Genulens = path_data+f"/Genulens_chunk_{int(Set)}.csv"
    seed = int(Source)
    i = seed
    np.random.seed(seed)
    # seed = Source
    # i = seed
    # np.random.seed(seed)
    ROW_G = np.random.randint(0, 10000)
    ROW_T = np.random.randint(0, 10000)

    TRILEGAL_row = pd.read_csv(
        data_TRILEGAL,
        skiprows=lambda x: x not in (0, ROW_T + 1)
    )
    GENULENS_row = pd.read_csv(
        data_Genulens,
        skiprows=lambda x: x not in (0, ROW_G + 1)
    )

    magstar = TRILEGAL_row[["W149", "u", "g", "r", "i", "z", "Y"]].iloc[0]

    event_params = {
        **magstar.to_dict(),
        **event_param(i, TRILEGAL_row.iloc[0], GENULENS_row.iloc[0], system_type)
    }

    thetas = event_params["thetas"]
    thetaE = event_params["thetaE"]

    true.loc[j, "theta_E"] = thetaE
    true.loc[j, "theta_source"] = thetas
    
#%%