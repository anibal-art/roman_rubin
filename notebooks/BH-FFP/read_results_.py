import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import corner
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# Get the current working directory
current_path = os.getcwd()
two_levels_up = os.path.dirname(os.path.dirname(os.getcwd()))
parent_directory = two_levels_up#os.path.abspath(os.path.join(current_path, os.pardir))
print("Parent Directory:", parent_directory)
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(os.path.dirname(os.getcwd()))

from class_analysis import metrics_creator
from class_analysis import create_df_to_plot
from ssh_connect import ssh_che
from connect_CHE import download_data
from read_save import read_data
from class_analysis import graph_maker_1plot
from matplotlib.colors import LogNorm
from scipy.stats import chi2
from astropy import constants as C
from astropy import units as u
from ulens_params import event_param, microlensing_params
from detection_criteria import mag
import rubin_sim
import rubin_sim.maf as maf
from rubin_sim.data import get_baseline
from plot_hist import *
import pyarrow.dataset as ds
from filter_df import prepare_and_compute_metrics


def read_results(system_type, model):
    
    true = pd.read_parquet(parent_directory+f"/all_results/{system_type}_metrics/true_{system_type}.parquet")
    fit_rr = pd.read_parquet(parent_directory+f"/all_results/{system_type}_metrics/fit_rr_{system_type}.parquet")
    fit_roman = pd.read_parquet(parent_directory+f"/all_results/{system_type}_metrics/fit_roman_{system_type}.parquet")
    
    data = "rr"
    met_1_rr = pd.read_parquet(parent_directory+f"/all_results/{system_type}_metrics/met_1_{data}_{system_type}.parquet")
    met_2_rr = pd.read_parquet(parent_directory+f"/all_results/{system_type}_metrics/met_2_{data}_{system_type}.parquet")
    met_3_rr = pd.read_parquet(parent_directory+f"/all_results/{system_type}_metrics/met_3_{data}_{system_type}.parquet")
    
    data = "roman"
    met_1_roman = pd.read_parquet(parent_directory+f"/all_results/{system_type}_metrics/met_1_{data}_{system_type}.parquet")
    met_2_roman = pd.read_parquet(parent_directory+f"/all_results/{system_type}_metrics/met_2_{data}_{system_type}.parquet")
    met_3_roman = pd.read_parquet(parent_directory+f"/all_results/{system_type}_metrics/met_3_{data}_{system_type}.parquet")
    
    met1_ratio = pd.read_parquet(parent_directory+f"/all_results/{system_type}_metrics/met1_ratio_{system_type}.parquet")
    met2_ratio = pd.read_parquet(parent_directory+f"/all_results/{system_type}_metrics/met2_ratio_{system_type}.parquet")
    met3_ratio = pd.read_parquet(parent_directory+f"/all_results/{system_type}_metrics/met3_ratio_{system_type}.parquet")

    

    return true, fit_roman, fit_rr, met_1_roman, met_1_rr, met_2_roman, met_2_rr, met_3_roman, met_3_rr, met1_ratio, met2_ratio, met3_ratio
