import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2
# from astropy import constants as C
# from astropy import units as u

def garwood_interval(N, cl=0.68):
    """
    Intervalos exactos de Poisson (Garwood) para nivel de confianza cl.
    Devuelve (low, high) para cada bin.
    """
    N = np.asarray(N, dtype=float)
    alpha = 1.0 - cl
    low = np.zeros_like(N)
    pos = N > 0
    low[pos] = 0.5 * chi2.ppf(alpha/2.0, 2.0*N[pos])
    high = 0.5 * chi2.ppf(1.0 - alpha/2.0, 2.0*(N + 1.0))
    return low, high

def poisson_errors_exact(counts, cl=0.68):
    """
    Calcula errores de Poisson exactos (asimétricos) para todos los bins.
    Retorna matriz (2, N) lista para yerr de matplotlib.errorbar.
    """
    counts = np.asarray(counts, dtype=float)
    low, high = garwood_interval(counts, cl=cl)
    yerr_minus = counts - low
    yerr_plus = high - counts
    return np.vstack([yerr_minus, yerr_plus])

def compute_hist_ratio(hist_1, hist_2, bins, min_count=10):
    """
    Devuelve (counts(hist_1)/counts(hist_2)) bin a bin,
    forzando 0 donde el denominador tiene menos de min_count eventos.
    """
    counts1, _ = np.histogram(hist_1, bins=bins)
    counts2, _ = np.histogram(hist_2, bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(counts2 >= min_count, counts1 / counts2, 0.0)
    return ratio

def plot_piE_fraction_by_category(
    val_th,
    bins=None,
    min_count=10,
    title_suffix=""
):
    """
    Plotea la fracción (sigma/pi_E < val_th) vs t_E por categorías A, B, C.
    Usa:
      - Numeradores: true_filtered ∩ ({met_3_rr o met_3_roman} con piE < val_th)
      - Denominador: true_filtered filtrado por la categoría correspondiente.

    Parámetros
    ----------
    val_th : float
        Umbral para filtrar met_3_rr['piE'] y met_3_roman['piE'].
    bins : array-like or None
        Bins para t_E. Si None, usa np.logspace(-1, 2.3, 40).
    min_count : int
        Mínimo de eventos en el denominador para reportar el cociente en ese bin.
    title_suffix : str
        Texto extra para agregar a los títulos de los subplots (opcional).

    Requiere en el entorno:
      true_filtered (cols: 'Source','Set','peak_flag','tE')
      met_3_rr, met_3_roman (cols: 'Source','Set','piE')
    """
    if bins is None:
        bins = np.logspace(-1, 2.3, 40)
    x = bins[:-1]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # Definición de categorías y títulos
    panels = [
        ('A', 'Category peak in Roman+Rubin'),
        ('B', 'Category peak in Rubin'),
        ('C', 'Category peak in Roman'),
    ]

    for i, (cat, base_title) in enumerate(panels):
        # Denominador: todos los eventos de la categoría
        histbot = true_filtered[true_filtered['peak_flag'] == cat]['tE'].values
        
        # --- Numerador usando met_3_rr ---
        ids_rr = met_3_rr[met_3_rr['rho'] < val_th][['Source', 'Set']]
        tf_rr = true_filtered.merge(ids_rr, on=['Source', 'Set'], how='inner')
        histtop_rr = tf_rr[tf_rr['peak_flag'] == cat]['tE'].values
        axs[i].plot(
            x,
            compute_hist_ratio(histtop_rr, histbot, bins, min_count=min_count),
            marker='.',
            label='Roman+Rubin Events'
        )

        # --- Numerador usando met_3_roman ---
        ids_r = met_3_roman[met_3_roman['rho'] < val_th][['Source', 'Set']]
        tf_r = true_filtered.merge(ids_r, on=['Source', 'Set'], how='inner')
        histtop_r = tf_r[tf_r['peak_flag'] == cat]['tE'].values
        axs[i].plot(
            x,
            compute_hist_ratio(histtop_r, histbot, bins, min_count=min_count),
            marker='.',
            label='Roman Events'
        )

        axs[i].set_xscale('log')
        axs[i].set_title(f'{base_title}{(" — " + title_suffix) if title_suffix else ""}')
        axs[i].set_xlabel(r'$t_E$ [day]')
        if i == 0:
            axs[i].set_ylabel(fr'Fraction with $\sigma/\pi_E < {val_th}$')
        axs[i].legend()

    plt.tight_layout()
    return fig, axs


def plot_npts(true, title, bines):
    fig = plt.figure(figsize=(8,6))

    df = true[(true["peak_flag"]=="A")&
             (true['flag_npeak_gt10']==1)]
    npts = df["W149_peak"]+df["u_peak"]+df["g_peak"]+df["r_peak"]+df["i_peak"]+df["z_peak"]+df["y_peak"]
    
    plt.hist(npts, bins=bines,edgecolor='k',label='peak in Roman&Rubin')
    df = true[(true["peak_flag"]=="B")&
             (true['flag_npeak_gt10']==1)]
    npts = df["W149_peak"]+df["u_peak"]+df["g_peak"]+df["r_peak"]+df["i_peak"]+df["z_peak"]+df["y_peak"]
    plt.hist(npts, bins=bines,edgecolor='k',label='peak in Rubin')

    df = true[(true["peak_flag"]=="C")&
             (true['flag_npeak_gt10']==1)]
    npts = df["W149_peak"]+df["u_peak"]+df["g_peak"]+df["r_peak"]+df["i_peak"]+df["z_peak"]+df["y_peak"] 
    plt.hist(npts, bins=bines,alpha=0.5, linewidth=2, edgecolor='k',label='peak in Roman')
    
    # for b in bines:
    #     plt.axvline(b,alpha=0.2)
    plt.title(title)
    plt.legend(loc='best')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r'n points in $t0\pm0.25 t_E$',fontsize=16)
    plt.grid(True)
    plt.xlim(1,1e4)


p_labels = {
    "u0":r"$u_0$",
    "t0":r"$t_0$",
    "rho":r"$\rho$",
    "piE":r"$\pi_E$",
    "tE" : r"$t_E$"
}
    

def p_plot_hist(true, title, p, bines):
    fig = plt.figure(figsize=(8,6))
    plt.hist(true[(true["peak_flag"]=="A")&
             (true['flag_npeak_gt10']==1)][p], bins=bines,edgecolor='k',label='peak in Roman&Rubin')
    plt.hist(true[(true["peak_flag"]=="B")&
             (true['flag_npeak_gt10']==1)][p], bins=bines,edgecolor='k',label='peak in Rubin')
    plt.hist(true[(true["peak_flag"]=="C")&
             (true['flag_npeak_gt10']==1)][p], bins=bines,alpha=0.5, linewidth=2, edgecolor='k',label='peak in Roman')
    
    plt.title(title)
    plt.legend(loc='best')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(p_labels[p],fontsize=16)
    plt.grid(True)
    # plt.show()
    
def compute_hist_ratio(hist_1, hist_2, bins):
    counts1, _ = np.histogram(hist_1, bins=bins)
    counts2, _ = np.histogram(hist_2, bins=bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(counts2 > 10, counts1 / counts2, 0)
    return ratio


def compute_ratio(df_rr, df_roman, params):
    # Solo conservamos las columnas deseadas + Source y Set
    df_rr_sub = df_rr[["Source", "Set"] + params]
    df_roman_sub = df_roman[["Source", "Set"] + params]

    # Hacemos un merge por Source y Set para alinear las filas
    merged = df_rr_sub.merge(df_roman_sub, on=["Source", "Set"], suffixes=('_rr', '_roman'))

    # Calculamos los ratios
    ratio_df = merged[["Source", "Set"]].copy()
    for p in params:
        ratio_df[p] = merged[f"{p}_rr"] / merged[f"{p}_roman"]
    
    return ratio_df

def create_scatter_with_marginals(ax, x1, x2, y, labels, p, binsx, binsy, first_col=False):

    tex_label = {'t0':'t_0', 'u0':'u_0', 'tE':'t_E', 'piE':'\pi_{E}', 'piEN':'\pi_{EN}'}
    # Main scatter plot
    hb = ax.scatter(x1, y,alpha=0.5,label='Roman+Rubin')
    h2 = ax.scatter(x2, y,alpha=0.5,label='Roman')
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Create inset axes for the histograms
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.2], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.2, 1], sharey=ax)

    # Histogram settings
    binwidth = 0.1
    ax_histx.hist(x1, bins=binsx, histtype='stepfilled', edgecolor='k', fill=True, alpha=0.5, color='red',label='Roman+Rubin')
    ax_histx.hist(x2, bins=binsx, histtype='stepfilled', edgecolor='k', fill=True, alpha=0.5, color='green',label="Roman")
    ax_histy.hist(y, bins=binsy, histtype='stepfilled', edgecolor='k', fill=True, alpha=0.5, color='royalblue', orientation='horizontal')
    ax_histx.set_yscale("log")
    ax_histy.set_yscale("log")
    ax_histx.set_xscale("log")
    ax_histy.set_xscale("log")

    ax_histx.tick_params(axis = "x", labelbottom=False)
    ax_histy.tick_params(axis = "y", labelleft=False)
    ax_histx.axvline(0.5, color = 'red', ls='--',lw = 2)
    ax_histy.axhline(1, color = 'red', ls='--',lw = 2)
    ax.axvline(0.5, ymin = 0, ymax = 1, ls = '--', lw = 2, color = 'red')
    ax.axhline(1, xmin = 0, xmax = 1, ls = '--',lw = 2, color = 'red')
    labelx, labely =labels[0],labels[1]
    ax.set_xlabel(labelx, fontsize=20)
    ax.set_ylabel(labely, fontsize=20)
    x_min=0
    x_max=0.5
    y_min=0
    y_max=1
    fil_x1= x1[(y<1)&(y>0)]
    filtered1=fil_x1[(fil_x1<0.5)&(fil_x1>0)]
    number_in_square1 = len(filtered1)/len(y)
    text1 = f"Fraction of Roman+Rubin events\n in [{x_min},{x_max}] x [{y_min},{y_max}]= {round(number_in_square1 ,2)}"  # Insert the number here

    fil_x2= x2[(y<1)&(y>0)]
    filtered2=fil_x2[(fil_x2<0.5)&(fil_x2>0)]
    number_in_square2 = len(filtered2)/len(y)
    text2 = f"Fraction of Roman events\n in [{x_min},{x_max}] x [{y_min},{y_max}]= {round(number_in_square2 ,2)}"  # Insert the number here
    
    
    ax.set_title(text1+"\n"+text2, fontsize = 16)
    ax.set_ylim(binsy[0],binsy[-1])
    ax.set_xlim(binsx[0],binsx[-1])
    ax.legend(loc='best')
    ax_histx.legend(loc='best')


def plot_histogram(ax, data1, data2, xlabel, title, limit):
    ax.hist(data1, bins=np.arange(0, 1.1, 0.1), edgecolor="k",lw=0.2, alpha=0.4, label='Roman+Rubin')
    ax.hist(data2, bins=np.arange(0, 1.1, 0.1), edgecolor="k",lw=0.2, alpha=0.4, label='Roman')
    ax.set_xlabel(xlabel, fontsize=20)
    ax.axvline(limit, color='red', linestyle='--')
    ax.legend(loc='best')
    
    fraction_data1 = len(data1[data1 < limit]) / len(data1)
    fraction_data2 = len(data2[data2 < limit]) / len(data2)
    
    ax.annotate(f'Fraction of events Roman+Rubin\nwith {xlabel}<{str(limit)} = {fraction_data1:.2f}', 
                xy=(0.5, -0.3), xycoords='axes fraction',
                ha='center', va='center', fontsize=15)
    ax.annotate(f'Fraction of events Roman\nwith {xlabel}<{str(limit)} = {fraction_data2:.2f}', 
                xy=(0.5, -0.45), xycoords='axes fraction',
                ha='center', va='center', fontsize=15)


def plot_histogram(ax, data, p, colors, some_flag=False):
    # print(data)
    sources = data['Source'].values
    data = data[p]   
    lab_latex = {'te':'tE(days)','rho':'\\rho' ,'piE':'\pi_E','q':'q'}
    data_label = lab_latex[p]
    lab_latex_legend = {'te':'t_E','rho':'\\rho','piE':'\pi_E','q':'q'}
    xlabel=r'$log_{10}'+f"[{lab_latex[p]}]$"
    
    masks_rr = lambda p, label: [
        (met_1_rr[p][met_1_rr['Source'].isin(sources)] < 0.25, 'r', f"$\\alpha({label})<0.25$"),
        (met_2_rr[p][met_2_rr['Source'].isin(sources)] < 0.25, 'k', f"$\\beta({label})<0.25$"),
        (met_3_rr[p][met_3_rr['Source'].isin(sources)] < 0.25, 'g', f"$\\gamma({label})<0.25$"),
    ]
    masks_roman = lambda p, label:[
        (met_1_roman[p][met_1_roman['Source'].isin(sources)] < 0.25, 'r', f"$\\alpha({label})<0.25$"),
        (met_2_roman[p][met_2_roman['Source'].isin(sources)] < 0.25, 'k', f"$\\beta({label})<0.25$"),
        (met_3_roman[p][met_3_roman['Source'].isin(sources)] < 0.25, 'g', f"$\\gamma({label})<0.25$"),
    ]
    
    if some_flag:
        masks = masks_roman(p,lab_latex_legend[p])
        dataset = 'Roman'
        met_1 = met_1_roman[met_1_roman['Source'].isin(sources)]
        met_2 = met_2_roman[met_2_roman['Source'].isin(sources)]
        met_3 = met_3_roman[met_3_roman['Source'].isin(sources)]
    else:
        masks = masks_rr(p,lab_latex_legend[p])
        dataset = 'Roman and Rubin'
        met_1 = met_1_rr[met_1_rr['Source'].isin(sources)]
        met_2 = met_2_rr[met_2_rr['Source'].isin(sources)]
        met_3 = met_3_rr[met_3_rr['Source'].isin(sources)]

    # Plot the histogram of the true values
    ax.hist(np.log10(data), bins=30, color='royalblue', alpha=0.5, label=f'True value of ${lab_latex_legend[p]}$')

    
    # Iterate over the masks and plot the masked data
    for (mask, color, label) in masks:
        # Select the masked data based on the condition in the mask
        # print(label)
        masked_data = data[mask]
        ax.hist(np.log10(masked_data), 
                bins=30, histtype='step', color=color, lw=2, label=label)
    
    # Set labels and title
    # print(xlabel)
    ax.set_xlabel(xlabel, fontsize=20)
    

    ax.set_title('Percentage of events with:\n'+
                 f"$\\alpha({lab_latex_legend[p]})<0.25: $"+
                 f"{round(100*len(met_1[p][met_1[p]<0.25])/len(data),2)}%, "+'\n'+
                 f"$\\beta({lab_latex_legend[p]})<0.25: $"+f"{round(100*len(met_2[p][met_2[p]<0.25])/len(data),1)}%, "+'\n'+
                 f"$\\gamma({lab_latex_legend[p]})<0.25: $"+f"{round(100*len(met_3[p][met_3[p]<0.25])/len(data),2)}%",fontsize=15)
    
    ax.legend(fontsize=10)
    # ax.set_xticks([1,2,3,4],fontsize=12)
    
    ax.grid()

def make_merge(df1,df2):
    """
    merge by Source and Set
    """
    return df1.merge(df2, on=["Source", "Set"], how="inner")


def compute_mass_metrics(df_fit, true, met1, met2, met3, model):
    df1 = true[["Source","Set","mass"]]

    if not model =="PSPL":
        versions = ["v1","v2","v3"]
        df2 = df_fit[["Source","Set","mass_v1","mass_v2","mass_v3","err_mass_v2","err_mass_v1","err_mass_v3"]]
    else:
        versions = ["v1","v2"]
        df2 = df_fit[["Source","Set","mass_v1","mass_v2","err_mass_v2","err_mass_v1"]]
    mass_df = make_merge(df1, df2 )        
    for version in versions:
        met1[f"mass_{version}"] = abs(mass_df["mass"] - mass_df[f"mass_{version}"])/mass_df["mass"]
        met2[f"mass_{version}"] = abs(mass_df["mass"] - mass_df[f"mass_{version}"])/mass_df[f"err_mass_{version}"]
        met3[f"mass_{version}"] = mass_df[f"err_mass_{version}"]/mass_df[f"mass_{version}"]

    return met1,met2,met3