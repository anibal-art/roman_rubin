import h5py
import math
import os, re
import pandas as pd
import numpy as np
from astropy.time import Time
from astropy.table import QTable
from astropy import constants as const
from astropy import units as u
import sys
#from pyLIMA.outputs 
import pyLIMA_plots
sys.path.append(os.getcwd())
from fit_lc import model_rubin_roman
from read_save import read_data
import warnings
from erfa import ErfaWarning
warnings.simplefilter('ignore', ErfaWarning)
import contextlib
import io
import matplotlib.pyplot as plt
from cycler import cycler
from read_save import read_data

from analysis.analysis_event import (  # noqa: F401  (moved; re-exported)
    labels_params,
    Analysis_Event,
)
from plotting.roman_rubin_plots import (  # noqa: F401  (moved; re-exported)
    create_df_to_plot,
    params_list_to_dict,
    plot_roman_rubin_plt,
    graph_maker_1plot,
    graph_maker,
)



def event_fits(path_fits):
    '''
    return events in common with roman and rubin
    we have events that fits only one of two for unknown reasons
    '''

    files_fits = os.listdir(path_fits)

    files_roman = [f for f in files_fits if 'Roman' in f]
    files_rr = [f for f in files_fits if not 'Roman' in f]

    n_rom = []  # list with the event number
    for j in files_roman:
        if not len(re.findall(r'\d+', j)) ==0:
            number = int(re.findall(r'\d+', j)[0])
            n_rom.append(number)

    n_rr = []  # # list with the event number
    for j in files_rr:
        if not len(re.findall(r'\d+', j)) ==0:
            number = int(re.findall(r'\d+', j)[0])
            n_rr.append(number)

    # Convert lists to sets
    set1 = set(n_rom)
    set2 = set(n_rr)
    # Find the common elements using intersection
    common_elements = set1.intersection(set2)
    # Convert the result back to a list (if needed)
    common_elements_list = list(common_elements)
    return common_elements_list



def get_label_params(model):
    if model == 'USBL':
        return ['t_center', 'u_center', 'tE', 'rho', 'separation',
                'mass_ratio', 'alpha', 'piEN', 'piEE', 'piE']
    elif model == 'FSPL':
        return ['t0', 'u0', 'tE', 'rho', 'piEN', 'piEE', 'piE']
    elif model == 'PSPL':
        return ['t0', 'u0', 'tE', 'piEN', 'piEE', 'piE']
    else:
        raise ValueError(f"Unknown model: {model}")

def prepare_data(true, fit):
    true = true.copy()
    fit = fit.copy()

    # Renombrar columnas excepto las claves de unión
    keys = ['Source', 'Set']
    df_true = true.rename(columns={col: col + '_true' for col in true.columns if col not in keys})
    df_fit = fit.rename(columns={col: col + '_fit' for col in fit.columns if col not in keys})

    # Merge por Source y Set
    df_merged = df_true.merge(df_fit, on=keys, how='inner')

    return df_merged



def compute_metrics(df, label_params):
    base_cols = ['Source', 'Set']
    met_1 = df[base_cols].copy()
    met_2 = df[base_cols].copy()
    met_3 = df[base_cols].copy()

    # Columnas auxiliares para marcar qué parámetro causó NaN
    invalid_param = pd.Series([None] * len(df), index=df.index, dtype="object")
    invalid_value = pd.Series([np.nan] * len(df), index=df.index)

    for key in label_params:
        true_col = key + '_true'
        fit_col = key + '_fit'
        err_col = key + '_err_fit'

        if all(col in df.columns for col in [true_col, fit_col, err_col]):
            true_vals = df[true_col].replace(0, np.nan)
            fit_vals = df[fit_col]
            err_vals = df[err_col].replace(0, np.nan)
            fit_vals_abs = fit_vals.abs().replace(0, np.nan)

            # Bias relativo o absoluto
            if key == 't_center':
                bias = (fit_vals - true_vals).abs()
            else:
                bias = ((fit_vals - true_vals).abs() / true_vals).replace([np.inf, -np.inf], np.nan)

            pull = ((fit_vals - true_vals).abs() / err_vals).replace([np.inf, -np.inf], np.nan)
            unc_rel = (err_vals / fit_vals_abs).replace([np.inf, -np.inf], np.nan)

            met_1[key] = bias
            met_2[key] = pull
            met_3[key] = unc_rel

            # Registrar la causa del NaN si no hay ninguna marcada aún
            mask_nan = bias.isna() | pull.isna() | unc_rel.isna()
            unmarked = invalid_param.isna() & mask_nan
            invalid_param[unmarked] = key

            # Guardar el valor causante del problema (prioridad: true_val luego fit_val)
            problematic_values = true_vals.where((true_vals == 0) | true_vals.isna(), np.nan)
            problematic_values = problematic_values.fillna(fit_vals.where(fit_vals == 0, np.nan))

            invalid_value[unmarked] = problematic_values[unmarked]

    # Agregar columnas de invalidación
    met_1['invalid_param'] = invalid_param
    met_1['invalid_value'] = invalid_value

    return met_1, met_2, met_3


    
def metrics_creator(true, fit, model):
    label_params = get_label_params(model)
    df_merged = prepare_data(true, fit)
    print(f"Filas después del merge: {df_merged.shape[0]}")
    return compute_metrics(df_merged, label_params)


