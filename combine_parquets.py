import pandas as pd
import pyarrow.parquet as pq
import glob
import os

# Find all parquet files
path = "/home/anibalvarela/microlensing/simulation_Rubin/roman_rubin/all_results/USBL_last/"

path_true = path + "true/"
path_fit_rr = path + "fit_rr/"
path_fit_roman = path + "fit_roman/"
print('Start combination')
parquet_files = os.listdir(path_true)
dfs_true = [pd.read_parquet(path_true+f) for f in parquet_files]
combined_df_true = pd.concat(dfs_true, ignore_index=True)
combined_df_true.to_parquet(path+'true.parquet', engine='pyarrow')

parquet_files = os.listdir(path_fit_rr)
dfs_rr = [pd.read_parquet(path_fit_rr+f) for f in parquet_files]
combined_df_rr = pd.concat(dfs_rr, ignore_index=True)
combined_df_rr.to_parquet(path+'fit_rr.parquet', engine='pyarrow')

parquet_files = os.listdir(path_fit_roman)
dfs_roman = [pd.read_parquet(path_fit_roman+f) for f in parquet_files]
combined_df_roman = pd.concat(dfs_roman, ignore_index=True)
combined_df_roman.to_parquet(path+'fit_roman.parquet', engine='pyarrow')
print('End combination')
