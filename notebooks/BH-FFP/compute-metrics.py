import os, sys
import pandas as pd
from pathlib import Path

# Obtiene la ruta al home
home = Path.home()

current_path = os.getcwd()
two_levels_up = os.path.dirname(os.path.dirname(os.getcwd()))
parent_directory = two_levels_up

#%%
print("Parent Directory:", parent_directory)
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(os.path.dirname(os.getcwd()))
from class_analysis import metrics_creator
from plot_hist import compute_ratio, compute_mass_metrics
import pyarrow.dataset as ds
from filter_df import prepare_and_compute_metrics
# print(parent_directory)
model = 'PSPL'

# path= str(home) +f'microlensing/simulation/roman_rubin/all_results/{"data_BH"}/'
system_type = "BH"

dataset_true = ds.dataset(str(home)+f"/microlensing/simulation_Rubin/roman_rubin//all_results/data_{system_type}/true_ds/", format="parquet", partitioning="hive")
dataset_fit_rr = ds.dataset(str(home)+f"/microlensing/simulation_Rubin/roman_rubin//all_results/data_{system_type}/fit_rr_ds/", format="parquet", partitioning="hive")
dataset_fit_roman = ds.dataset(str(home)+f"/microlensing/simulation_Rubin/roman_rubin//all_results/data_{system_type}/fit_roman_ds/", format="parquet", partitioning="hive")

true = dataset_true.to_table().to_pandas()
fit_rr = dataset_fit_rr.to_table().to_pandas()
fit_roman =  dataset_fit_roman.to_table().to_pandas()

if model=="PSPL":
    params = ["tE", "piE", "piEE", "piEN"]
elif model=="FSPL":
    params = ["tE", "piE","rho", "piEE", "piEN"]
elif model=="USBL":
    params = ["tE","separation", "mass_ratio","alpha", "piE","rho", "piEE", "piEN"]
    

keys = ["Source", "Set"]

set_true = set(map(tuple, true[keys].values))
set_rr = set(map(tuple, fit_rr[keys].values))
set_roman = set(map(tuple, fit_roman[keys].values))

common = set_true & set_rr & set_roman

# función auxiliar
def filter_df(df):
    return df[df[keys].apply(tuple, axis=1).isin(common)]

true = filter_df(true)
fit_rr = filter_df(fit_rr)
fit_roman = filter_df(fit_roman)
#%%    
# Solo para FFP
if model == "FSPL":
    detector_6_pts = pd.read_csv("/home/anibal-pc/microlensing/simulation_Rubin/roman_rubin/all_results/deviation_results.csv")
    # detector_6_pts[detector_6_pts["decision"]==True]
    detector_6_pts["Set"] = detector_6_pts["n"]+1
    detector_6_pts["Source"] = detector_6_pts["event"]
    
    k = ["Source", "Set"]
    
    det_keys = (detector_6_pts.loc[detector_6_pts["decision"] == True, k]
                .drop_duplicates())
    
    true_detected = true.merge(det_keys, on=k, how="inner")
    true=true_detected
#%%
(
    fit_rr, fit_roman, true,
    met_1_rr, met_2_rr, met_3_rr,
    met_1_roman, met_2_roman, met_3_roman
) = prepare_and_compute_metrics(
    true=true,
    fit_rr=fit_rr,
    fit_roman=fit_roman,
    model=model,
    metrics_creator=metrics_creator,
    compute_mass_metrics=compute_mass_metrics,
    make_total_npeak_flag=True,         # opcional
    total_npeak_threshold=10,           # opcional
)

# Aplicar la función a tus métricas
met1_ratio = compute_ratio(met_1_rr, met_1_roman, params)
met2_ratio = compute_ratio(met_2_rr, met_2_roman, params)
met3_ratio = compute_ratio(met_3_rr, met_3_roman, params)


#%%
output_dir = parent_directory + f"/all_results/{system_type}_metrics/"

os.makedirs(output_dir, exist_ok=True)
dfs = {"fit_rr":fit_rr,
       "fit_roman":fit_roman,
       "true":true,
    "met_1_rr": met_1_rr,
    "met_2_rr": met_2_rr,
    "met_3_rr": met_3_rr,
    "met_1_roman": met_1_roman,
    "met_2_roman": met_2_roman,
    "met_3_roman": met_3_roman,
    "met1_ratio":met1_ratio,
    "met2_ratio":met2_ratio,
    "met3_ratio":met3_ratio
}
for name, df in dfs.items():
    df.to_parquet(output_dir + f"{name}_{system_type}.parquet")

