import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_results_ import read_results

model = 'FSPL'
system_type = "FFP"
true, fit_roman, fit_rr, met_1_roman, met_1_rr, met_2_roman, met_2_rr, met_3_roman, met_3_rr, met1_ratio, met2_ratio, met3_ratio =read_results(system_type, model)


npts_peak = true["W149_peak"]+true["u_peak"]+true["g_peak"]+true["r_peak"]+true["i_peak"]+true["z_peak"]+true["y_peak"]
true["npts_rubin"]=true["u_peak"]+true["g_peak"]+true["r_peak"]+true["i_peak"]+true["z_peak"]+true["y_peak"]
true["npts"]=npts_peak
true["npts/tE"]=true["npts"]/true["tE"]
#print(true.columns)


#%%
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
p ='rho'
        
df_rr_only = (
    met_3_rr.loc[met_3_rr[p] < 0.5, cols]
    .sort_values(['Source', 'Set'])
    .reset_index(drop=True))

true_filtered_rerror = true.merge(
    df_rr_only,
    on=['Source', 'Set'],
    how='inner')

plt.close("all")
p = "tE"
bines  = np.logspace(-2, 2.5, 50)
hysttype_1 = "step"
hysttype_2 = "stepfilled"

title_new = lambda x: "Category: " + f"{x}"
label2 = r"Events with $NU(\rho)<0.5$ (Roman+Rubin data)"
label1 = "All events"

categories = ["A", "B", "C"]

fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True, dpi=250)

for ax, cat in zip(axs, categories):

    df1 = true[true["peak_flag"] == cat]
    df2 = true_filtered_rerror[true_filtered_rerror["peak_flag"] == cat]
    
    N_total = len(df1)
    N_subset = len(df2)
    
    if N_total > 0:
        frac = 100 * N_subset / N_total
    else:
        frac = 0.0

    ax.hist(df1[p], bins=bines, histtype=hysttype_1,
            edgecolor='k', label=label1)

    ax.hist(df2[p], bins=bines, histtype=hysttype_2, color='royalblue',
            edgecolor='k', alpha=0.4, label=label2)

    ax.annotate(
    f"{frac:.1f}% of events\n constrained",
    xy=(0.05, 0.95),              # posición relativa dentro del eje
    xycoords='axes fraction',
    fontsize=14,
    ha='left',
    va='top',
    bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8)
)


    ax.set_title(title_new(dict_cat_peak[cat]), fontsize=20)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)
    ax.set_xlabel(p_labels[p] + " [day]", fontsize=18)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)

# Solo el primero lleva ylabel
axs[0].set_ylabel("Events per $\Delta \log (t_E)$", fontsize=18)

# Una sola leyenda
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.85])

# if outdir is None:
outname = "FFP_piE<0.5_RR"
outdir = os.getcwd()

png_path = os.path.join(outdir, f"{outname}.png")
pdf_path = os.path.join(outdir, f"{outname}.pdf")

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.show()
#%%

plt.hist(true[true["peak_flag"]=="A"]["npts/tE"], bins=np.logspace(-1.5,2.5,20),histtype="step",linewidth=2,label="Roman+Rubin")
plt.hist(true[true["peak_flag"]=="B"]["npts/tE"], bins=np.logspace(-1.5,2.5,20), histtype="step",linewidth=2,label="Rubin")
plt.hist(true[true["peak_flag"]=="C"]["npts/tE"], bins=np.logspace(-1.5,2.5,20), histtype="step",linewidth=2,label="Roman")
plt.xlabel("$N_{points}/ t_E$", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.yscale("log")
plt.xscale("log")
plt.legend()

#%%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_results_ import read_results

model = 'FSPL'
system_type = "FFP"

true, fit_roman, fit_rr, met_1_roman, met_1_rr, met_2_roman, met_2_rr, met_3_roman, met_3_rr, met1_ratio, met2_ratio, met3_ratio = read_results(system_type, model)

# ============================================================
# Construcción de npts y npts/tE
# ============================================================
npts_peak = (
    true["W149_peak"] + true["u_peak"] + true["g_peak"] +
    true["r_peak"] + true["i_peak"] + true["z_peak"] + true["y_peak"]
)

true["npts"] = npts_peak
true["npts/tE"] = true["npts"] / (true["tE"])#*true["rho"])

print(true.columns)

# ============================================================
# Filtrado del subconjunto que querés superponer
# NUEVO BLOQUE: eventos con NU(rho) < 0.5 en Roman+Rubin
# ============================================================
cols = ['Source', 'Set']
p_metric = 'rho'

df_rr_only = (
    met_3_rr.loc[met_3_rr[p_metric] < 0.5, cols]
    .sort_values(['Source', 'Set'])
    .reset_index(drop=True)
)

true_filtered_rerror = true.merge(
    df_rr_only,
    on=['Source', 'Set'],
    how='inner'
)

# ============================================================
# Diccionarios de etiquetas
# ============================================================
dict_cat_peak = {
    "A": "peak in Rubin+Roman",
    "B": "peak in Rubin",
    "C": "peak in Roman"
}

categories = ["A", "B", "C"]

# ============================================================
# PRIMER GRÁFICO REHECHO:
# histogramas de npts/tE con el subconjunto superpuesto
# ============================================================
plt.close("all")

xvar = "npts/tE"
bins_nts = np.logspace(-1,2.5, 20)

label_all = "All events"
label_subset = r"Events with $NU(\rho) < 0.5$ (Roman+Rubin data)"

fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True, dpi=250)

for ax, cat in zip(axs, categories):

    df1 = true[true["peak_flag"] == cat]
    df2 = true_filtered_rerror[true_filtered_rerror["peak_flag"] == cat]

    N_total = len(df1)
    N_subset = len(df2)

    frac = 100 * N_subset / N_total if N_total > 0 else 0.0

    ax.hist(
        df1[xvar],
        bins=bins_nts,
        histtype="step",
        linewidth=2,
        color="k",
        label=label_all
    )

    ax.hist(
        df2[xvar],
        bins=bins_nts,
        histtype="stepfilled",
        color="royalblue",
        edgecolor="k",
        alpha=0.4,
        label=label_subset
    )

    ax.annotate(
        f"{frac:.1f}% of events\nconstrained",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=14,
        ha='left',
        va='top',
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8)
    )

    ax.set_title(f"Category: {dict_cat_peak[cat]}", fontsize=18)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r"$N_{\mathrm{points}}/t_E $", fontsize=16)

    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.tick_params(axis='both', which='minor', labelsize=11)

axs[0].set_ylabel("Frequency", fontsize=16)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.88])

outname = "FFP_npts_over_tE_NUrho_lt_0p5_RR"
outdir = os.getcwd()

png_path = os.path.join(outdir, f"{outname}.png")
pdf_path = os.path.join(outdir, f"{outname}.pdf")

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.show()