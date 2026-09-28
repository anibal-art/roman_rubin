#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:00:04 2026

@author: anibal-pc
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from read_results_ import read_results
from ulens_params import event_param


# ============================================================
# Función completa: procesa una fila y devuelve theta_E, theta_source
# ============================================================
def compute_theta_row(row, path_data, system_type):
    Source = int(row["Source"])
    Set = int(row["Set"])

    data_TRILEGAL = path_data + f"/TRILEGAL_chunk_{Set}.csv"
    data_Genulens = path_data + f"/Genulens_chunk_{Set}.csv"

    seed = Source
    i = seed
    np.random.seed(seed)

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

    return thetaE, thetas


# ============================================================
# Función completa: agrega columnas theta_E y theta_source en paralelo
# ============================================================
def add_theta_columns_parallel(true, path_data, system_type, n_jobs=-1):
    true = true.reset_index(drop=True).copy()

    rows = [row for _, row in true.iterrows()]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(compute_theta_row)(row, path_data, system_type)
        for row in tqdm(rows, total=len(rows))
    )

    thetaE_list, theta_source_list = zip(*results)

    true["theta_E"] = thetaE_list
    true["theta_source"] = theta_source_list

    return true


# ============================================================
# Script principal
# ============================================================
model = 'FSPL'
system_type = "FFP"

true, fit_roman, fit_rr, met_1_roman, met_1_rr, met_2_roman, met_2_rr, met_3_roman, met_3_rr, met1_ratio, met2_ratio, met3_ratio = read_results(system_type, model)

mets_list = [
    (met_1_rr, met_1_roman),
    (met_2_rr, met_2_roman),
    (met_3_rr, met_3_roman),
]

dict_cat_peak = {
    "A": "peak in Rubin+Roman",
    "B": "peak in Rubin",
    "C": "peak in Roman"
}

p_labels = {
    "u0": r"$u_0$",
    "t0": r"$t_0$",
    "rho": r"$\rho$",
    "piE": r"$\pi_E$",
    "tE": r"$t_E$"
}

cols = ['Source', 'Set']

path_data = "/home/anibal-pc/microlensing/simulation_Rubin/roman_rubin/chunks_TRILEGAL_GENULENS/"

# ============================================================
# Ejecutar en paralelo
# ============================================================
true = add_theta_columns_parallel(
    true=true,
    path_data=path_data,
    system_type=system_type,
    n_jobs=-1
)

#%%
import matplotlib.pyplot as plt
plt.plot(true["theta_source"],true["theta_E"], marker="o",linestyle='',alpha=0.25)
rho_const = true[met_3_rr["rho"]<0.5]
plt.plot(rho_const["theta_source"],rho_const["theta_E"], marker="o",color='red',linestyle='',label=r"Rubin+Roman fit with $\sigma_{\rho}/\hat{\rho}<\frac{1}{2}$")
plt.xscale("log")
plt.yscale("log")
plt.ylabel(r"$\theta_E$ [mas]",fontsize=20)
plt.xlabel(r"$\theta^{\star}$ [mas]",fontsize=20)
plt.legend()
plt.grid()
plt.show()

plt.plot(true["theta_source"],true["theta_E"], marker="o",linestyle='',alpha=0.25)
rho_const = true[met_3_roman["rho"]<0.5]
plt.plot(rho_const["theta_source"],rho_const["theta_E"], marker="o",color='red',linestyle='',label=r"Roman fit with $\sigma_{\rho}/\hat{\rho}<\frac{1}{2}$")
plt.xscale("log")
plt.yscale("log")
plt.ylabel(r"$\theta_E$ [mas]",fontsize=20)
plt.xlabel(r"$\theta^{\star}$ [mas]",fontsize=20)
plt.legend()
plt.grid()
plt.show()

#%%
bins_thetaE = np.logspace(-4,0,30)
plt.hist(true["theta_E"],bins=bins_thetaE)
plt.hist(rho_const["theta_E"],bins=bins_thetaE)
plt.yscale("log")
plt.xscale("log")
plt.xlabel(r"$\theta_E [mas]$")

#%%
bins_M = np.logspace(-6,-1.5,30)
plt.hist(true["mass"],bins=bins_M)
plt.hist(rho_const["mass"],bins=bins_M)
plt.yscale("log")
plt.xscale("log")
plt.xlabel(r"$Mass [M_{\odot}]$")
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LogNorm

# ============================================================
# Verificación básica de columnas
# ============================================================
print("Columns in true:")
print(true.columns.tolist())

print("\nColumns in met_3_rr:")
print(met_3_rr.columns.tolist())

print("\nColumns in met_3_roman:")
print(met_3_roman.columns.tolist())

# ============================================================
# Renombrar rho antes del merge para evitar conflictos
# ============================================================
met_3_rr_rho = met_3_rr[['Source', 'Set', 'rho']].rename(columns={'rho': 'rho_rr'})
met_3_roman_rho = met_3_roman[['Source', 'Set', 'rho']].rename(columns={'rho': 'rho_roman'})

# ============================================================
# Alinear true con las métricas usando Source y Set
# ============================================================
df_rr = true.merge(
    met_3_rr_rho,
    on=['Source', 'Set'],
    how='inner'
)

df_roman = true.merge(
    met_3_roman_rho,
    on=['Source', 'Set'],
    how='inner'
)

# ============================================================
# Limpiar valores no físicos / no finitos
# ============================================================
df_rr = df_rr[
    np.isfinite(df_rr["theta_source"]) &
    np.isfinite(df_rr["theta_E"]) &
    np.isfinite(df_rr["rho_rr"]) &
    (df_rr["theta_source"] > 0) &
    (df_rr["theta_E"] > 0) &
    (df_rr["rho_rr"] > 0)
].copy()

df_roman = df_roman[
    np.isfinite(df_roman["theta_source"]) &
    np.isfinite(df_roman["theta_E"]) &
    np.isfinite(df_roman["rho_roman"]) &
    (df_roman["theta_source"] > 0) &
    (df_roman["theta_E"] > 0) &
    (df_roman["rho_roman"] > 0)
].copy()

# ============================================================
# Rango común para el colorbar
# ============================================================
rho_all = np.concatenate([df_rr["rho_rr"].values, df_roman["rho_roman"].values])

vmin = np.percentile(rho_all, 5)
vmax = np.percentile(rho_all, 95)

if vmin <= 0 or vmax <= 0 or np.isclose(vmin, vmax):
    vmin = np.min(rho_all[rho_all > 0])
    vmax = np.max(rho_all)

norm = LogNorm(vmin=vmin, vmax=vmax)

# ============================================================
# Figura con dos paneles
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

datasets = [
    (axes[0], df_rr, "rho_rr", r"Rubin+Roman"),
    (axes[1], df_roman, "rho_roman", r"Roman")
]

for ax, df, rho_col, title in datasets:
    # --------------------------------------------------------
    # scatter coloreado por rho
    # --------------------------------------------------------
    sc = ax.scatter(
        df["theta_source"],
        df["theta_E"],
        c=df[rho_col],
        cmap="viridis",
        norm=norm,
        s=18,
        alpha=0.7,
        edgecolors="none"
    )

    # --------------------------------------------------------
    # contorno rho = 0.5 usando triangulación
    # # --------------------------------------------------------
    # if len(df) >= 3:
    #     triang = tri.Triangulation(
    #         df["theta_source"].values,
    #         df["theta_E"].values
    #     )

    #     try:
    #         cont = ax.tricontour(
    #             triang,
    #             df[rho_col].values,
    #             levels=[0.5],
    #             colors="red",
    #             linewidths=2.0
    #         )
    #         ax.clabel(cont, fmt={0.5: r"$0.5$"}, fontsize=10)
    #     except ValueError:
    #         print(f"No se pudo dibujar el contorno rho=0.5 para {title}")

    # --------------------------------------------------------
    # puntos con rho < 0.5 remarcados
    # --------------------------------------------------------
    # detectable = df[df[rho_col] < 0.5]
    # ax.scatter(
    #     detectable["theta_source"],
    #     detectable["theta_E"],
    #     s=28,
    #     facecolors="none",
    #     edgecolors="white",
    #     linewidths=0.8,
    #     alpha=0.9,
    #     label=r"$\sigma_\rho/\hat{\rho}<0.5$"
    # )

    # # --------------------------------------------------------
    # formato
    # --------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=16)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=10, loc="best")

# ============================================================
# Etiquetas comunes
# ============================================================
axes[0].set_ylabel(r"$\theta_E$ [mas]", fontsize=18)
for ax in axes:
    ax.set_xlabel(r"$\theta^{\star}$ [mas]", fontsize=18)

# ============================================================
# Colorbar común
# ============================================================
cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), pad=0.02)
cbar.set_label(r"$\sigma_\rho/\hat{\rho}$", fontsize=16)

plt.tight_layout()
plt.show()
