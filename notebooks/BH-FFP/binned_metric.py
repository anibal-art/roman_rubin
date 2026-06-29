#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:25:35 2026

@author: anibal-pc
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from read_results_ import read_results

model = 'FSPL'
system_type = "FFP"
true, fit_roman, fit_rr, met_1_roman, met_1_rr, met_2_roman, met_2_rr, met_3_roman, met_3_rr, met1_ratio, met2_ratio, met3_ratio =read_results(system_type, model)
mets_list = [
    (met_1_rr, met_1_roman),
    (met_2_rr, met_2_roman),
    (met_3_rr, met_3_roman),]

plt.figure()
plt.scatter(true["mass"], true["tE"])
plt.yscale("log")
plt.xscale("log")
plt.show()
#%%
plt.figure()
plt.hist(met_3_roman["piE"][met_1_rr["piE"]<0.5], bins=np.logspace(-5,6,50))
plt.hist(met_3_rr["piE"][met_1_rr["piE"]<0.5], bins=np.logspace(-5,6,50),histtype="step")
plt.xscale("log")
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from read_results_ import read_results

model = 'FSPL'
system_type = "FFP"

true, fit_roman, fit_rr, met_1_roman, met_1_rr, met_2_roman, met_2_rr, met_3_roman, met_3_rr, met1_ratio, met2_ratio, met3_ratio = read_results(system_type, model)

cols_id = ["Source", "Set"]

# ============================================================
# Merge correcto
# ============================================================
df_plot = true[cols_id + ["mass", "tE"]].merge(
    met_3_rr[cols_id + ["piE"]],
    on=cols_id,
    how="inner"
)

# ============================================================
# Filtros básicos
# ============================================================
mask_valid = (
    np.isfinite(df_plot["mass"]) &
    np.isfinite(df_plot["tE"]) &
    np.isfinite(df_plot["piE"]) &
    (df_plot["mass"] > 0) &
    (df_plot["tE"] > 0)
)

df_plot = df_plot[mask_valid]

# ============================================================
# Separar poblaciones
# ============================================================
mask_good = (df_plot["piE"] > 0) & (df_plot["piE"] < 1)
mask_bad  = ~mask_good  # todo lo demás

# ============================================================
# Figura
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

# ---- puntos grises (mal medidos)
ax.scatter(
    df_plot.loc[mask_bad, "mass"],
    df_plot.loc[mask_bad, "tE"],
    color="lightgray",
    s=12,
    alpha=0.5,
    edgecolors="none",
    label=r"$NU(\pi_E) \geq 1$"
)

# ---- puntos coloreados (buenos)
sc = ax.scatter(
    df_plot.loc[mask_good, "mass"],
    df_plot.loc[mask_good, "tE"],
    c=df_plot.loc[mask_good, "piE"],
    cmap="viridis",
    norm=LogNorm(vmin=1e-3, vmax=1),
    s=18,
    alpha=0.9,
    edgecolors="none",
    label=r"$0 < NU(\pi_E) < 1$"
)

# ============================================================
# Escalas
# ============================================================
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"$Mass [M_{\odot}]$", fontsize=14)
ax.set_ylabel(r"$t_E [days] $", fontsize=14)
ax.set_title(r"$t_E$ vs Mass (colored by $NU(\pi_E)$)", fontsize=15)

# ============================================================
# Colorbar SOLO para los buenos
# ============================================================
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label(r"$NU(\pi_E)$", fontsize=14)

ax.grid(True, which="both", alpha=0.25)
ax.legend()

plt.tight_layout()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from read_results_ import read_results

model = 'FSPL'
system_type = "FFP"

true, fit_roman, fit_rr, met_1_roman, met_1_rr, met_2_roman, met_2_rr, met_3_roman, met_3_rr, met1_ratio, met2_ratio, met3_ratio = read_results(system_type, model)

# ============================================================
# Parámetros del mapa
# ============================================================
threshold = 0.5          # criterio de "buena medición"
nbins_mass = 25
nbins_tE = 25
min_count_per_bin = 5    # opcional: enmascara bins con muy pocos eventos

cols_id = ["Source", "Set"]

# ============================================================
# Merge: masa y tE desde true, métrica desde met_3_rr
# ============================================================
df_plot = true[cols_id + ["mass", "tE"]].merge(
    met_3_rr[cols_id + ["piE"]],
    on=cols_id,
    how="inner"
)

# ============================================================
# Filtrado de valores válidos
# ============================================================
mask = (
    np.isfinite(df_plot["mass"]) &
    np.isfinite(df_plot["tE"]) &
    np.isfinite(df_plot["piE"]) &
    (df_plot["mass"] > 0) &
    (df_plot["tE"] > 0) &
    (df_plot["piE"] > 0)
)

df_plot = df_plot.loc[mask].copy()

# ============================================================
# Arrays
# ============================================================
mass = df_plot["mass"].to_numpy()
tE   = df_plot["tE"].to_numpy()
nu_piE = df_plot["piE"].to_numpy()   # met_3_rr["piE"] = NU(piE)

# ============================================================
# Bins logarítmicos
# ============================================================
mass_edges = np.logspace(np.log10(mass.min()), np.log10(mass.max()), nbins_mass + 1)
tE_edges   = np.logspace(np.log10(tE.min()),   np.log10(tE.max()),   nbins_tE + 1)

# ============================================================
# Índices de bins
# ============================================================
ix = np.digitize(mass, mass_edges) - 1
iy = np.digitize(tE,   tE_edges)   - 1

valid_bin = (
    (ix >= 0) & (ix < nbins_mass) &
    (iy >= 0) & (iy < nbins_tE)
)

ix = ix[valid_bin]
iy = iy[valid_bin]
nu_piE = nu_piE[valid_bin]

# ============================================================
# Acumuladores
# ============================================================
count_total = np.zeros((nbins_tE, nbins_mass), dtype=int)
count_good  = np.zeros((nbins_tE, nbins_mass), dtype=int)

for i, j, val in zip(ix, iy, nu_piE):
    count_total[j, i] += 1
    if val < threshold:
        count_good[j, i] += 1

# ============================================================
# Fracción de sensibilidad
# ============================================================
fraction = np.full((nbins_tE, nbins_mass), np.nan, dtype=float)

nonzero = count_total > 0
fraction[nonzero] = count_good[nonzero] / count_total[nonzero]

# enmascarar bins con pocos eventos
fraction[count_total < min_count_per_bin] = np.nan

fraction_masked = np.ma.masked_invalid(fraction)

# ============================================================
# Figura
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

pcm = ax.pcolormesh(
    mass_edges,
    tE_edges,
    fraction_masked,
    cmap="viridis",
    norm=colors.Normalize(vmin=0, vmax=1),
    shading="auto",
    rasterized=True
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"Lens mass $[M_{\odot}]$", fontsize=14)
ax.set_ylabel(r"$t_E [days]$", fontsize=14)
# ax.set_title(
    # r"Fraction of events with $\sigma_{\pi_E}/\pi_E< 1/2$",
    # fontsize=15
# )

# ============================================================
# Contornos opcionales
# ============================================================
# mass_centers = np.sqrt(mass_edges[:-1] * mass_edges[1:])
# tE_centers   = np.sqrt(tE_edges[:-1] * tE_edges[1:])
# Mgrid, Tgrid = np.meshgrid(mass_centers, tE_centers)

# levels = [0.1, 0.3, 0.5, 0.7, 0.9]
# cs = ax.contour(
#     Mgrid,
#     Tgrid,
#     fraction_masked,
#     levels=levels,
#     colors="white",
#     linewidths=0.8,
#     alpha=0.9
# )

# ax.clabel(cs, fmt="%.1f", fontsize=9)

# ============================================================
# Colorbar
# ============================================================
cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(
    r"Fraction with $\sigma_{\pi_E}/\pi_E < 1/2$",
    fontsize=13
)

ax.grid(True, which="both", alpha=0.2)

plt.tight_layout()
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from read_results_ import read_results

model = 'FSPL'
system_type = "FFP"

true, fit_roman, fit_rr, met_1_roman, met_1_rr, met_2_roman, met_2_rr, met_3_roman, met_3_rr, met1_ratio, met2_ratio, met3_ratio = read_results(system_type, model)

# ============================================================
# Parámetros del mapa
# ============================================================
threshold = 0.5          # criterio de "buena medición"
nbins_mass = 25
nbins_tE = 25
min_count_per_bin = 5    # enmascara bins con muy pocos eventos
peak_category = "A"      # elegir entre "A", "B" o "C"

cols_id = ["Source", "Set"]

# ============================================================
# Filtrado por categoría peak_flag
# ============================================================
true_cat = true.loc[true["peak_flag"] == peak_category].copy()

# ============================================================
# Merge: masa y tE desde true, métrica desde met_3_rr
# ============================================================
df_plot = true_cat[cols_id + ["mass", "tE"]].merge(
    met_3_rr[cols_id + ["piE"]],
    on=cols_id,
    how="inner"
)

# ============================================================
# Filtrado de valores válidos
# ============================================================
mask = (
    np.isfinite(df_plot["mass"]) &
    np.isfinite(df_plot["tE"]) &
    np.isfinite(df_plot["piE"]) &
    (df_plot["mass"] > 0) &
    (df_plot["tE"] > 0) &
    (df_plot["piE"] > 0)
)

df_plot = df_plot.loc[mask].copy()

# ============================================================
# Verificación de cantidad de eventos
# ============================================================
if len(df_plot) == 0:
    raise ValueError(
        f"No hay eventos válidos para peak_flag = '{peak_category}' "
        "después del merge y del filtrado."
    )

# ============================================================
# Arrays
# ============================================================
mass = df_plot["mass"].to_numpy()
tE   = df_plot["tE"].to_numpy()
nu_piE = df_plot["piE"].to_numpy()   # met_3_rr["piE"] = NU(piE)

# ============================================================
# Bins logarítmicos
# ============================================================
mass_edges = np.logspace(np.log10(mass.min()), np.log10(mass.max()), nbins_mass + 1)
tE_edges   = np.logspace(np.log10(tE.min()),   np.log10(tE.max()),   nbins_tE + 1)

# ============================================================
# Índices de bins
# ============================================================
ix = np.digitize(mass, mass_edges) - 1
iy = np.digitize(tE,   tE_edges)   - 1

valid_bin = (
    (ix >= 0) & (ix < nbins_mass) &
    (iy >= 0) & (iy < nbins_tE)
)

ix = ix[valid_bin]
iy = iy[valid_bin]
nu_piE = nu_piE[valid_bin]

# ============================================================
# Acumuladores
# ============================================================
count_total = np.zeros((nbins_tE, nbins_mass), dtype=int)
count_good  = np.zeros((nbins_tE, nbins_mass), dtype=int)

for i, j, val in zip(ix, iy, nu_piE):
    count_total[j, i] += 1
    if val < threshold:
        count_good[j, i] += 1

# ============================================================
# Fracción de sensibilidad
# ============================================================
fraction = np.full((nbins_tE, nbins_mass), np.nan, dtype=float)

nonzero = count_total > 0
fraction[nonzero] = count_good[nonzero] / count_total[nonzero]

# enmascarar bins con pocos eventos
fraction[count_total < min_count_per_bin] = np.nan

fraction_masked = np.ma.masked_invalid(fraction)

# ============================================================
# Figura
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

pcm = ax.pcolormesh(
    mass_edges,
    tE_edges,
    fraction_masked,
    cmap="viridis",
    norm=colors.Normalize(vmin=0, vmax=1),
    shading="auto",
    rasterized=True
)

ax.set_xscale("log")
ax.set_yscale("log")
label_category = {"A":"peak covered by Roman+Rubin"}
ax.set_xlabel(r"Mass $[M_{\odot}$]", fontsize=14)
ax.set_ylabel(r"$t_E$ [day]", fontsize=14)
ax.set_title(
    rf"Fraction of events with $NU(\pi_E) < {threshold}$"+" \nfor "+f"{label_category[peak_category]}",
    fontsize=15)

# ============================================================
# Contornos opcionales
# ============================================================

mass_centers = np.sqrt(mass_edges[:-1] * mass_edges[1:])
tE_centers   = np.sqrt(tE_edges[:-1] * tE_edges[1:])
Mgrid, Tgrid = np.meshgrid(mass_centers, tE_centers)

levels = [0.1, 0.3, 0.5, 0.7, 0.9]
cs = ax.contour(
    Mgrid,
    Tgrid,
    fraction_masked,
    levels=levels,
    colors="white",
    linewidths=0.8,
    alpha=0.9
)

ax.clabel(cs, fmt="%.1f", fontsize=9)

# ============================================================
# Colorbar
# ============================================================
cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(
    rf"Fraction with $NU(\pi_E) < {threshold}$",
    fontsize=13
)

ax.grid(True, which="both", alpha=0.2)

plt.tight_layout()
plt.show()

#%%
