import os, sys
import numpy as np
import matplotlib.pyplot as plt
from pyLIMA.outputs import pyLIMA_plots
from cycler import cycler
import pandas as pd
sys.path.append(os.path.dirname(os.getcwd()))
# from functions_roman_rubin import sim_fit,sim_event
# from functions_roman_bonnie bluerubin import model_rubin_roman
from functions_roman_rubin import read_data, save_sim, sim_event
from fit_lc import fit_rubin_roman, model_rubin_roman
from class_analysis import Analysis_Event
from ulens_params import microlensing_params, event_param
import multiprocessing as mul
import h5py
from fit_lc import fit_rubin_roman
from detection_criteria import filter5points, deviation_from_constant, has_consecutive_numbers, filter_band, mag
from read_save import save_sim, save_fit, read_data
from functions_roman_rubin import sim_fit

current_path = os.getcwd()
# i=0 #select one event by its index in the TRILEGAL set
model='FSPL'
path_TRILEGAL_set = current_path+"/chunks_TRILEGAL_GENULENS/TRILEGAL_chunk_1.csv"
path_GENULENS_set = current_path+"/chunks_TRILEGAL_GENULENS/Genulens_chunk_1.csv"
path_to_save_model = os.getcwd()+'/fits_pspl/'
path_to_save_fit = os.getcwd()+'/fits_pspl/'
path_ephemerides = current_path+'/ephemerides/Roman_positions.npy'
path_dataslice = current_path
# path_fit_rr = path_to_save_fit+f'/Event_RR_{i}_TRF.npy'
# path_fit_roman =  path_to_save_fit+f'/Event_Roman_{i}_TRF.npy'
ZP = {'W149':27.615, 'u':27.03, 'g':28.38, 'r':28.16,
          'i':27.85, 'z':27.46, 'y':26.68}
colorbands={'W149':'b', 'u':'purple', 'g':'g', 'r':'red',
          'i':'yellow', 'z':'k', 'y':'cyan'}

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions_roman_rubin import sim_fit_specific_params


def extraer_resultado_rho(fit, rho_idx=3):
    """
    Extrae rho_fit, sigma_rho y sigma_rho/rho_fit desde un objeto fit de pyLIMA.
    """
    best_model = np.asarray(fit.fit_results["best_model"], dtype=float)
    cov = np.asarray(fit.fit_results["covariance_matrix"], dtype=float)

    rho_fit = best_model[rho_idx]

    var_rho = cov[rho_idx, rho_idx]

    if np.isfinite(var_rho) and var_rho >= 0:
        sigma_rho = np.sqrt(var_rho)
    else:
        sigma_rho = np.nan

    if np.isfinite(rho_fit) and rho_fit != 0:
        sigma_rho_rel = sigma_rho / np.abs(rho_fit)
    else:
        sigma_rho_rel = np.nan

    return rho_fit, sigma_rho, sigma_rho_rel


def barrido_rho(
    i,
    event_params,
    model,
    algo,
    path_to_save_model,
    path_to_save_fit,
    path_ephemerides,
    path_dataslice,
    rhos=None,
    system_type="FFP",
    rho_idx=3,
    t0=None,
):
    """
    Barre valores de rho, corre sim_fit_specific_params y guarda la incerteza
    recuperada para Roman+Rubin y Roman solamente.
    """

    if rhos is None:
        rhos = np.logspace(-2, 1, 10)

    resultados = []

    # Copia base para no modificar permanentemente el diccionario original
    event_params_base = event_params.copy()

    for k, rho_true in enumerate(rhos):

        print(f"\n=== Barrido rho {k + 1}/{len(rhos)}: rho = {rho_true:.4e} ===")

        params = event_params_base.copy()
        params["rho"] = float(rho_true)

        if t0 is not None:
            params["t0"] = float(t0)

        fit_rr, pyLIMAmodel_rr, fit_roman, pyLIMAmodel_roman = sim_fit_specific_params(
            i,
            system_type,
            model,
            algo,
            params,
            path_to_save_model,
            path_to_save_fit,
            path_ephemerides,
            path_dataslice,
        )

        rho_fit_rr, sigma_rho_rr, sigma_rho_rel_rr = extraer_resultado_rho(
            fit_rr,
            rho_idx=rho_idx,
        )

        rho_fit_roman, sigma_rho_roman, sigma_rho_rel_roman = extraer_resultado_rho(
            fit_roman,
            rho_idx=rho_idx,
        )

        resultados.append(
            {
                "rho_true": rho_true,

                "rho_fit_rr": rho_fit_rr,
                "sigma_rho_rr": sigma_rho_rr,
                "sigma_rho_rel_rr": sigma_rho_rel_rr,

                "rho_fit_roman": rho_fit_roman,
                "sigma_rho_roman": sigma_rho_roman,
                "sigma_rho_rel_roman": sigma_rho_rel_roman,

                "ratio_sigma_rho_rr_over_roman": sigma_rho_rr / sigma_rho_roman
                if np.isfinite(sigma_rho_rr) and np.isfinite(sigma_rho_roman) and sigma_rho_roman != 0
                else np.nan,
            }
        )

    df_rho = pd.DataFrame(resultados)

    return df_rho


def graficar_barrido_rho(df_rho):
    """
    Grafica cómo cambia la incerteza absoluta y relativa de rho
    en función del valor verdadero de rho.
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.loglog(
        df_rho["rho_true"],
        df_rho["sigma_rho_rr"],
        "o-",
        label="Roman + Rubin",
    )
    ax.loglog(
        df_rho["rho_true"],
        df_rho["sigma_rho_roman"],
        "s--",
        label="Roman",
    )
    ax.set_xlabel(r"$\rho_\mathrm{true}$")
    ax.set_ylabel(r"$\sigma_\rho$")
    ax.set_title("Incerteza absoluta")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.loglog(
        df_rho["rho_true"],
        df_rho["sigma_rho_rel_rr"],
        "o-",
        label="Roman + Rubin",
    )
    ax.loglog(
        df_rho["rho_true"],
        df_rho["sigma_rho_rel_roman"],
        "s--",
        label="Roman",
    )
    ax.set_xlabel(r"$\rho_\mathrm{true}$")
    ax.set_ylabel(r"$\sigma_\rho / |\rho_\mathrm{fit}|$")
    ax.set_title("Incerteza relativa")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.semilogx(
        df_rho["rho_true"],
        df_rho["ratio_sigma_rho_rr_over_roman"],
        "o-",
    )
    ax.axhline(1.0, ls="--", color="k", alpha=0.5)
    ax.set_xlabel(r"$\rho_\mathrm{true}$")
    ax.set_ylabel(r"$\sigma_{\rho,\mathrm{RR}} / \sigma_{\rho,\mathrm{Roman}}$")
    ax.set_title("Mejora relativa")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()
from ulens_params import event_param
system_type = "FFP"
np.random.seed(1)
ROW_G = 234#np.random.randint(0, 10000)
ROW_T = 8#np.random.randint(0, 10000)

TRILEGAL_row = pd.read_csv(
    path_TRILEGAL_set,
    skiprows=lambda x: x not in (0, ROW_T + 1)
)
GENULENS_row = pd.read_csv(
    path_GENULENS_set,
    skiprows=lambda x: x not in (0, ROW_G + 1)
)
magstar = TRILEGAL_row[["W149", "u", "g", "r", "i", "z", "Y"]].iloc[0]

TRILEGAL_dict = TRILEGAL_row.iloc[0].to_dict()
GENULENS_dict = GENULENS_row.iloc[0].to_dict()

event_params = {
    **magstar.to_dict(),
    **event_param(90, TRILEGAL_row.iloc[0], GENULENS_row.iloc[0], system_type),**GENULENS_dict
}
event_params["u0"]=0.5
import os
import time
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


def extraer_resultado_rho(fit, rho_idx=3):
    """
    Extrae rho_fit, sigma_rho y sigma_rho/rho_fit desde un objeto fit de pyLIMA.
    """

    best_model = np.asarray(fit.fit_results["best_model"], dtype=float)
    cov = np.asarray(fit.fit_results["covariance_matrix"], dtype=float)

    rho_fit = best_model[rho_idx]
    var_rho = cov[rho_idx, rho_idx]

    if np.isfinite(var_rho) and var_rho >= 0:
        sigma_rho = np.sqrt(var_rho)
    else:
        sigma_rho = np.nan

    if np.isfinite(rho_fit) and rho_fit != 0:
        sigma_rho_rel = sigma_rho / np.abs(rho_fit)
    else:
        sigma_rho_rel = np.nan

    return rho_fit, sigma_rho, sigma_rho_rel


def _asegurar_slash(path):
    """
    Asegura que el path termine con '/' porque sim_fit_specific_params
    parece concatenar strings del tipo path + f'Event_{i}.h5'.
    """

    if not path.endswith(os.sep):
        path = path + os.sep

    return path


def worker_barrido_rho(args):
    """
    Worker para correr un único valor de rho.

    Esta función corre en paralelo. Por eso devuelve solo un diccionario
    con números simples, no devuelve objetos fit.
    """

    from functions_roman_rubin import sim_fit_specific_params

    (
        k,
        rho_true,
        i_base,
        event_params_base,
        model,
        algo,
        path_to_save_model,
        path_to_save_fit,
        path_ephemerides,
        path_dataslice,
        system_type,
        rho_idx,
        t0,
        run_label,
    ) = args

    t_start = time.time()

    try:
        # Copia independiente para este proceso
        params = event_params_base.copy()
        params["rho"] = float(rho_true)

        if t0 is not None:
            params["t0"] = float(t0)

        # IMPORTANTE:
        # uso un i distinto por rho para que no se pisen archivos Event_i
        i_job = int(i_base * 100000 + k)

        # IMPORTANTE:
        # uso una subcarpeta distinta por rho para evitar colisiones de archivos
        path_model_job = os.path.join(
            path_to_save_model,
            run_label,
            f"rho_{k:03d}"
        )

        path_fit_job = os.path.join(
            path_to_save_fit,
            run_label,
            f"rho_{k:03d}"
        )

        os.makedirs(path_model_job, exist_ok=True)
        os.makedirs(path_fit_job, exist_ok=True)

        path_model_job = _asegurar_slash(path_model_job)
        path_fit_job = _asegurar_slash(path_fit_job)

        print(f"[rho {k:03d}] empezando rho = {rho_true:.4e}, i_job = {i_job}")

        fit_rr, pyLIMAmodel_rr, fit_roman, pyLIMAmodel_roman = sim_fit_specific_params(
            i_job,
            system_type,
            model,
            algo,
            params,
            path_model_job,
            path_fit_job,
            path_ephemerides,
            path_dataslice,
        )

        rho_fit_rr, sigma_rho_rr, sigma_rho_rel_rr = extraer_resultado_rho(
            fit_rr,
            rho_idx=rho_idx,
        )

        rho_fit_roman, sigma_rho_roman, sigma_rho_rel_roman = extraer_resultado_rho(
            fit_roman,
            rho_idx=rho_idx,
        )

        if (
            np.isfinite(sigma_rho_rr)
            and np.isfinite(sigma_rho_roman)
            and sigma_rho_roman != 0
        ):
            ratio_sigma = sigma_rho_rr / sigma_rho_roman
        else:
            ratio_sigma = np.nan

        t_end = time.time()

        return {
            "status": "ok",
            "k": k,
            "i_job": i_job,
            "rho_true": rho_true,

            "rho_fit_rr": rho_fit_rr,
            "sigma_rho_rr": sigma_rho_rr,
            "sigma_rho_rel_rr": sigma_rho_rel_rr,

            "rho_fit_roman": rho_fit_roman,
            "sigma_rho_roman": sigma_rho_roman,
            "sigma_rho_rel_roman": sigma_rho_rel_roman,

            "ratio_sigma_rho_rr_over_roman": ratio_sigma,
            "runtime_sec": t_end - t_start,
            "error": "",
        }

    except Exception:
        t_end = time.time()

        return {
            "status": "failed",
            "k": k,
            "i_job": np.nan,
            "rho_true": rho_true,

            "rho_fit_rr": np.nan,
            "sigma_rho_rr": np.nan,
            "sigma_rho_rel_rr": np.nan,

            "rho_fit_roman": np.nan,
            "sigma_rho_roman": np.nan,
            "sigma_rho_rel_roman": np.nan,

            "ratio_sigma_rho_rr_over_roman": np.nan,
            "runtime_sec": t_end - t_start,
            "error": traceback.format_exc(),
        }


def barrido_rho_parallel(
    i,
    event_params,
    model,
    algo,
    path_to_save_model,
    path_to_save_fit,
    path_ephemerides,
    path_dataslice,
    rhos=None,
    system_type="FFP",
    rho_idx=3,
    t0=None,
    n_workers=4,
    backend="process",
    run_label="rho_sweep",
):
    """
    Barre valores de rho en paralelo.

    backend:
        "process" usa ProcessPoolExecutor.
        "thread" usa ThreadPoolExecutor.

    En Linux normalmente conviene "process".
    Si en Jupyter tenés problemas de multiprocessing, probá backend="thread".
    """

    if rhos is None:
        rhos = np.logspace(-2, 1, 10)

    event_params_base = event_params.copy()

    args_list = []

    for k, rho_true in enumerate(rhos):
        args_list.append(
            (
                k,
                float(rho_true),
                int(i),
                event_params_base,
                model,
                algo,
                path_to_save_model,
                path_to_save_fit,
                path_ephemerides,
                path_dataslice,
                system_type,
                rho_idx,
                t0,
                run_label,
            )
        )

    if backend == "process":
        Executor = ProcessPoolExecutor
    elif backend == "thread":
        Executor = ThreadPoolExecutor
    else:
        raise ValueError("backend debe ser 'process' o 'thread'.")

    resultados = []

    with Executor(max_workers=n_workers) as executor:

        futures = [
            executor.submit(worker_barrido_rho, args)
            for args in args_list
        ]

        for future in as_completed(futures):
            resultado = future.result()
            resultados.append(resultado)

            if resultado["status"] == "ok":
                print(
                    f"[OK] k={resultado['k']:03d}, "
                    f"rho={resultado['rho_true']:.4e}, "
                    f"sigma_rr={resultado['sigma_rho_rr']:.4e}, "
                    f"sigma_roman={resultado['sigma_rho_roman']:.4e}, "
                    f"time={resultado['runtime_sec']:.1f} s"
                )
            else:
                print(
                    f"[FAILED] k={resultado['k']:03d}, "
                    f"rho={resultado['rho_true']:.4e}"
                )

    df_rho = pd.DataFrame(resultados)
    df_rho = df_rho.sort_values("rho_true").reset_index(drop=True)

    return df_rho


def graficar_barrido_rho(df_rho, solo_ok=True):
    """
    Grafica cómo cambia la incerteza absoluta y relativa de rho
    en función del valor verdadero de rho.
    """

    if solo_ok and "status" in df_rho.columns:
        df_plot = df_rho[df_rho["status"] == "ok"].copy()
    else:
        df_plot = df_rho.copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.loglog(
        df_plot["rho_true"],
        df_plot["sigma_rho_rr"],
        "o-",
        label="Roman + Rubin",
    )
    ax.loglog(
        df_plot["rho_true"],
        df_plot["sigma_rho_roman"],
        "s--",
        label="Roman",
    )
    ax.set_xlabel(r"$\rho_\mathrm{true}$")
    ax.set_ylabel(r"$\sigma_\rho$")
    ax.set_title("Incerteza absoluta")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.loglog(
        df_plot["rho_true"],
        df_plot["sigma_rho_rel_rr"],
        "o-",
        label="Roman + Rubin",
    )
    ax.loglog(
        df_plot["rho_true"],
        df_plot["sigma_rho_rel_roman"],
        "s--",
        label="Roman",
    )
    ax.set_xlabel(r"$\rho_\mathrm{true}$")
    ax.set_ylabel(r"$\sigma_\rho / |\rho_\mathrm{fit}|$")
    ax.set_title("Incerteza relativa")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[2]
    ax.semilogx(
        df_plot["rho_true"],
        df_plot["ratio_sigma_rho_rr_over_roman"],
        "o-",
    )
    ax.axhline(1.0, ls="--", color="k", alpha=0.5)
    ax.set_xlabel(r"$\rho_\mathrm{true}$")
    ax.set_ylabel(r"$\sigma_{\rho,\mathrm{RR}} / \sigma_{\rho,\mathrm{Roman}}$")
    ax.set_title("Mejora relativa")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()
    
rhos = np.logspace(-2, 1, 20)

i = 0

df_rho = barrido_rho_parallel(
    i=i,
    event_params=event_params,
    model=model,
    algo="TRF",
    path_to_save_model=path_to_save_model,
    path_to_save_fit=path_to_save_fit,
    path_ephemerides=path_ephemerides,
    path_dataslice=path_dataslice,
    rhos=rhos,
    system_type="FFP",
    rho_idx=3,
    t0=2461483.5,
    n_workers=4,
    backend="process",
    run_label="rho_sweep_FFPL_test",
)

print(df_rho)

graficar_barrido_rho(df_rho)
