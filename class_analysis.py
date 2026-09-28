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



def graph_maker_1plot(dict_1, model, plot_residuals, path_save = None, path_ephemerides = None):
    # path_save = '../interesting_cases/' + model + '/'
    # path_ephemerides = '../ephemerides/Roman_positions.npy'

    # =========================
    # [NUEVO BLOQUE] Siempre creo 3 ejes para que plot_roman_rubin_plt reciba (ax0, ax1, ax2)
    # =========================
    if plot_residuals:
        fig, axes = plt.subplots(
            3, 1, figsize=(10, 6), dpi=100,
            gridspec_kw={'height_ratios': [2, 1, 1]},
            sharex=True
        )
    else:
        # Creo 3, pero "aplasto" y oculto los residuales
        fig, axes = plt.subplots(
            3, 1, figsize=(10, 6), dpi=100,
            gridspec_kw={'height_ratios': [1, 1e-6, 1e-6]},
            sharex=True
        )
        axes[1].set_visible(False)
        axes[2].set_visible(False)

        # (Opcional pero recomendable) saco espacios extra
        fig.subplots_adjust(hspace=0.0)

    # --- ploteo principal (y residuales si corresponde) ---
    plot_roman_rubin_plt(
        axes[0], axes[1], axes[2],
        dict_1['source'].iloc[0],
        dict_1['path_save'].iloc[0],
        dict_1['path_event'].iloc[0],
        dict_1['path_fit_rr'].iloc[0],
        dict_1['path_fit_roman'].iloc[0],
        path_ephemerides,
        model,
        plot_residuals
    )

    # --- formateo ---
    axes[0].invert_yaxis()
    axes[0].set_ylabel('Magnitude', fontsize=16)

    # =========================
    # [NUEVO BLOQUE] Labels condicionales: si no hay residuales, el xlabel va en axes[0]
    # =========================
    if plot_residuals:
        axes[2].set_xlabel('Time [days]', fontsize=16)
        axes[2].set_ylabel(r"$\Delta \mathrm{mag}$" + "\nRoman", fontsize=16)
        axes[1].set_ylabel(r"$\Delta \mathrm{mag}$" + "\nRoman+Rubin", fontsize=16)
    else:
        axes[0].set_xlabel('Time [days]', fontsize=16)

    return fig, axes




def graph_maker(dict_1, dict_2,dict_3, model):
    path_save = '../interesting_cases/'+model+'/'
    path_ephemerides = '../ephemerides/Roman_positions.npy'
    # f = io.StringIO()
    # with contextlib.redirect_stdout(f):
    fig, axes = plt.subplots(3, 3, figsize=(20, 9),
        gridspec_kw={'height_ratios': [2, 1, 1]}, 
                             dpi=200)
    
    plot_roman_rubin_plt(axes[0,0], axes[1,0], axes[2,0], dict_1['source'], 
                         dict_1['path_save'], dict_1['path_event'], dict_1['path_fit_rr'],
                         dict_1['path_fit_roman'], path_ephemerides, model)
    # axes[0,0].set_title('Id: '+str(dict_1['source']))
    axes[0,0].invert_yaxis()
    axes[2,0].set_xlabel('Time [days]',fontsize=16)
    axes[2,0].set_ylabel("$\Delta \mathrm{mag}$\nRoman",fontsize=16)
    axes[1,0].set_ylabel("$\Delta \mathrm{mag}$\nRoman+Rubin",fontsize=16)
    axes[0,0].set_ylabel('Magnitude',fontsize=16)
    
    plot_roman_rubin_plt(axes[0,1], axes[1,1], axes[2,1], dict_2['source'], 
                         dict_2['path_save'], dict_2['path_event'], dict_2['path_fit_rr'],
                         dict_2['path_fit_roman'], path_ephemerides, model)
    # axes[0,1].set_title('Id: '+str(dict_2['source']))
    axes[0,1].invert_yaxis()
    axes[2,1].set_xlabel('Time [days]',fontsize=16)
    # axes[2,1].set_ylabel("$\Delta \mathrm{mag}$\nRoman",fontsize=16)
    # axes[1,1].set_ylabel("$\Delta \mathrm{mag}$\nRoman+Rubin",fontsize=16)
    # axes[0,1].set_ylabel('Magnitude',fontsize=16)
    
    plot_roman_rubin_plt(axes[0,2], axes[1,2], axes[2,2], dict_3['source'], 
                         dict_3['path_save'], dict_3['path_event'], dict_3['path_fit_rr'],
                         dict_3['path_fit_roman'], path_ephemerides, model)

    axes[0,2].invert_yaxis()
    axes[2,2].set_xlabel('Time [days]',fontsize=16)

    return fig, axes


def create_df_to_plot(sources_array, model, path_event):
    # path_save = '../baseline4/'+model+'/'
    events_to_plot_df = pd.DataFrame(columns=['source', 'path_event', 'path_fit_rr', 'path_fit_roman','path_save'])
    for sce in sources_array:
        events_to_plot_df.loc[len(events_to_plot_df)] = {
            'source': sce,
            'path_event': path_event + f'Event_{sce}.h5',
            'path_fit_rr': path_event + f'Event_RR_{sce}_TRF.npy',
            'path_fit_roman': path_event + f'Event_Roman_{sce}_TRF.npy',
            'path_save':path_event
        }
    return events_to_plot_df

def labels_params(model):
    if model == "USBL":
        labels_params: list[str] = ['t_center','u_center','tE','rho',"separation","mass_ratio","alpha",'piEN','piEE']
    elif model == "FSPL": 
        labels_params: list[str] = ['t0','u0','tE','rho','piEN','piEE']
    elif model == "PSPL":
        labels_params: list[str] = ['t0','u0','tE','piEN','piEE']
    return labels_params

def params_list_to_dict(lista, model):
    dict_param = {}
    dict_param['tE'] = lista[2]
    if model=='USBL':
        dict_param['t_center'] = lista[0]
        dict_param['u_center'] = lista[1]
        dict_param['rho'] = lista[3]
        dict_param['s'] = lista[4]
        dict_param['mass_ratio'] = lista[5]
        dict_param['alpha'] = lista[6]
        dict_param['piEE'] = lista[7]
        dict_param['piEN'] = lista[8]
    elif model=='PSPL':
        dict_param['t0'] = lista[0]
        dict_param['u0'] = lista[1]
        dict_param['piEE'] = lista[3]
        dict_param['piEN'] = lista[4]
    elif model=='FSPL':
        dict_param['t0'] = lista[0]
        dict_param['u0'] = lista[1]
        dict_param['rho'] = lista[3]
        dict_param['piEE'] = lista[4]
        dict_param['piEN'] = lista[5]

    return dict_param



def plot_roman_rubin_plt(axs, axs_residual_rr, axs_residual_roman, Source,
                         path_save,path_event, path_fit_rr, path_fit_roman, 
                         path_ephemerides, model,plot_residuals):
    
    colorbands={'W149':'b', 'u':'purple', 'g':'g', 'r':'red',
          'i':'yellow', 'z':'k', 'y':'cyan'}
    colors1 = cycler(color=['purple', 'darkorange', 'forestgreen'])
    colors2 = cycler(color=['royalblue', 'crimson', 'purple'])
    ZP = {'W149':27.615, 'u':27.03, 'g':28.38, 'r':28.16,
              'i':27.85, 'z':27.46, 'y':26.68}

    Event = Analysis_Event(model, path_event,
                           path_fit_rr, 
                           path_fit_roman)
    Event.load_data_fit()
    Event.load_data_sim()

    data_fit_rr = Event.fit_rr_data
    data_fit_roman = Event.fit_roman_data
    
    pyLIMA_parameters = Event.model_params
    bands = Event.lightcurves
    
    origin = [Event.info[2],[0,0]]
    # print(Event.info)
    # print(origin)
    PAR = Event.labels_params()

    chi2_rr = Event.chichi_dof(Event.fit_rr_data)['chi2']
    DOF_rr = Event.chichi_dof(Event.fit_rr_data)["dof"]
    
    chi2_roman =  Event.chichi_dof(Event.fit_roman_data)['chi2']
    DOF_roman = Event.chichi_dof(Event.fit_roman_data)['dof']
    
    dict_fit_roman = Event.dict_fit_vals(Event.fit_roman_data) 
    dict_fit_rr = Event.dict_fit_vals(Event.fit_rr_data)
    
    
    ulens_params = []
        
    for b in (PAR):
        ulens_params.append(pyLIMA_parameters[b])
                            
    flux_params = []
    for b in bands:
        if not len(bands[b])==0:
            zp_Rubin_to_pyLIMA = (10**((-27.4+ZP[b])/2.5))            
            flux_params.append(pyLIMA_parameters['fsource_'+b]/zp_Rubin_to_pyLIMA)
            flux_params.append(pyLIMA_parameters['ftotal_'+b]/zp_Rubin_to_pyLIMA)
            
    true_params = ulens_params + flux_params

    lightcurve_dict = {}
    for f in ['W149', 'u', 'g', 'r','i','z','y']:
        lightcurve_dict[f] = np.array([bands[f]['time'],bands[f]['mag'],bands[f]['err_mag']]).T

    if model=='USBL':
        t0_str='t_center'
        u0_str='u_center'
    else:
        t0_str='t0'
        u0_str='u0'
        
    model_rr = model_rubin_roman(Source, {t0_str:data_fit_rr['best_model'][0]}, 
                                 path_ephemerides, model, origin, lightcurve_dict['W149'], 
                                  lightcurve_dict['u'], lightcurve_dict['g'], lightcurve_dict['r'],
                                 lightcurve_dict['i'], lightcurve_dict['z'],lightcurve_dict['y'])
    
    model_roman = model_rubin_roman(Source, {t0_str:data_fit_roman['best_model'][0]}, 
                                    path_ephemerides, model, origin, lightcurve_dict['W149'], [], [], [], [], [],[])
    
    
    # Call the function to plot the photometric models
    pyLIMA_plots.plot_photometric_models(
        figure_axe=axs,  # Pass None if you're only using Bokeh
        microlensing_model=model_rr,
        model_parameters=data_fit_rr['best_model'],MARKERS_COLORS=colors1,
        bokeh_plot=None,
        plot_unit='Mag'
    )
    # Call the function to plot the photometric models
    pyLIMA_plots.plot_photometric_models(
        figure_axe=axs,  # Pass None if you're only using Bokeh
        microlensing_model=model_roman,
        model_parameters=data_fit_roman['best_model'],MARKERS_COLORS=colors2,
        bokeh_plot=None,
        plot_unit='Mag'
    )

    lc = pyLIMA_plots.plot_aligned_data(figure_axe=axs,  # Pass None if you're only using Bokeh
        microlensing_model=model_rr,
        model_parameters=data_fit_rr['best_model'],
        bokeh_plot=None,
        plot_unit='Mag'
    )

    # detalles del plot
    x_shift = pyLIMA_parameters['t0']
    tE = pyLIMA_parameters['tE']
    
    xticks = np.arange(x_shift-tE,x_shift+tE,2*tE/8) 
    axs.set_xticks(xticks)
    axs.set_xticklabels([f'{x - x_shift:.3f}' for x in xticks])
    if not plot_residuals:
        axs.tick_params(axis='x', which='both', labelbottom=True, bottom=True)

    
    if model == 'USBL':
        delta_t = abs(pyLIMA_parameters['t0'] - pyLIMA_parameters['t_center'])
    else:
        delta_t = 0

    n =1
    if plot_residuals:
        pyLIMA_plots.plot_residuals(axs_residual_rr, model_rr, 
                                    data_fit_rr['best_model'],
                                    bokeh_plot=None, plot_unit='Mag')
        
        pyLIMA_plots.plot_residuals(axs_residual_roman, model_roman, 
                                    data_fit_roman['best_model'],
                                    bokeh_plot=None, plot_unit='Mag')

        
        axs_residual_rr.set_xticks(xticks)
        axs_residual_rr.set_xticklabels([f'{x - x_shift:.3f}' for x in xticks])
        axs_residual_rr.axhline(0, alpha =0.5)
        axs_residual_roman.axhline(0, alpha =0.5)
        axs_residual_roman.set_xticks(xticks)
        axs_residual_roman.set_xticklabels([f'{x - x_shift:.3f}' for x in xticks])

        
        if 2*tE>365*10:
            axs_residual_rr.set_xlim(x_shift-5*365,x_shift+5*365)
            axs_residual_roman.set_xlim(x_shift-5*365,x_shift+5*365)    
            
        elif delta_t> pyLIMA_parameters['tE']:
            axs_residual_rr.set_xlim(x_shift-delta_t*2, x_shift+delta_t*2)
            axs_residual_roman.set_xlim(x_shift-delta_t*2, x_shift+delta_t*2)    
        
        else:
            axs_residual_rr.set_xlim(x_shift-n*tE,x_shift+n*tE)
            axs_residual_roman.set_xlim(x_shift-n*tE,x_shift+n*tE)


    # if model == 'USBL':
    #     axs.axvline(pyLIMA_parameters['t_center'],color='red',linestyle='-.',alpha=0.6,label='$t_{center}$')
    # axs.axvline(x_shift,color='blue',linestyle='-.',alpha=0.6,label='$t_{0}$')
    
    if 2*tE>365*10:
        axs.set_xlim(x_shift-5*365,x_shift+5*365)
    elif delta_t> pyLIMA_parameters['tE']:
        axs.set_xlim(x_shift-delta_t*2, x_shift+delta_t*2)
    else:
        axs.set_xlim(x_shift-n*tE,x_shift+n*tE)

    xcoor = 0.1
    text = (
        r"$t_E$:"+f"{round(tE,2)} days\n"
        r"$\pi_{EE}$:"+f"{round(pyLIMA_parameters['piEE'],4)}\n"
        r"$\pi_{EN}$:"+f"{round(pyLIMA_parameters['piEN'],4)}\n"
    )
    axs.text(xcoor, 0.4, text,
        transform=axs.transAxes,
        ha='center', va='bottom',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black"))


    text_roman_fit = (
        r"Roman"+"\n"
        r"$\frac{\chi^2}{DOF} = $"+f"{round(chi2_roman/DOF_roman,3)}\n"
        r"$t_E$:" +f"{round(dict_fit_roman['tE'],2)} days  " + r"$\pm$ "+f"{round(dict_fit_roman['tE_err'],2)} days\n"
        r"$\pi_{EE}$:"+f"{round(dict_fit_roman['piEE'],4)} " + r"$\pm$"+f"{round(dict_fit_roman['piEE_err'],6)} \n"
        r"$\pi_{EN}$:"+f"{round(dict_fit_roman['piEN'],4)} " + r"$\pm$"+f"{round(dict_fit_roman['piEN_err'],6)} \n"
    )
    axs_residual_roman.text(1.2, 0.15, text_roman_fit,
        transform=axs_residual_roman.transAxes,
        ha='center', va='bottom',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black"), clip_on=False)

    text_rr_fit = (
        r"Roman & Rubin"+"\n"
        r"$\frac{\chi^2}{DOF} = $"+f"{round(chi2_rr/DOF_rr,3)}\n"
        r"$t_E$:" +f"{round(dict_fit_rr['tE'],2)} days  " + r"$\pm$ "+f"{round(dict_fit_rr['tE_err'],2)} days\n"
        r"$\pi_{EE}$:"+f"{round(dict_fit_rr['piEE'],4)} " + r"$\pm$"+f"{round(dict_fit_rr['piEE_err'],6)} \n"
        r"$\pi_{EN}$:"+f"{round(dict_fit_rr['piEN'],4)} " + r"$\pm$"+f"{round(dict_fit_rr['piEN_err'],6)} \n"
    )
    axs_residual_rr.text(1.2, 0.5, text_rr_fit,
        transform=axs_residual_rr.transAxes,
        ha='center', va='bottom',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black"),clip_on=False)

    
    axs.legend(shadow=True, fontsize='large',
                          bbox_to_anchor=(0, 1.02, 1, 0.2),
                          loc="lower left",
                          mode="expand", borderaxespad=0, ncol=3)


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



class Analysis_Event:
    """
    This is a class to analize one event.
    """
    
    def __init__(self, model, path_model=None, path_fit_rr=None, path_fit_roman=None,
                genulens_params=None, trilegal_params=None, computed_params=None,
                fit_rr_data=None, fit_roman_data = None, origin=None, lightcurves=None,
                model_params=None, info= None, indices= None):
        
        self.model = model
        self.path_model = path_model
        self.path_fit_rr   = path_fit_rr
        self.path_fit_roman   = path_fit_roman
        self.genulens_params = genulens_params
        self.trilegal_params = trilegal_params
        self.computed_params = computed_params
        self.fit_rr_data = fit_rr_data
        self.fit_roman_data = fit_roman_data
        self.origin = origin
        self.lightcurves = lightcurves
        self.model_params = model_params
        self.info = info
        self.indices = indices

    
            

        
    def refresh(self):
        self.genulens_params = None
        self.trilegal_params = None
        self.computed_params = None
        self.fit_rr_data = None
        self.fit_roman_data = None
        self.origin = None
        self.lightcurves = None
        self.model_params = None
        self.info = None
        self.indices = None
        
    def labels_params(self):
        if self.model == "USBL":
            labels_params: list[str] = ['t_center','u_center','tE','rho',"separation","mass_ratio","alpha",'piEN','piEE']
        elif self.model == "FSPL":
            labels_params: list[str] = ['t0','u0','tE','rho','piEN','piEE']
        elif self.model == "PSPL":
            labels_params: list[str] = ['t0','u0','tE','piEN','piEE']
        return labels_params
    
    def fit_model_name(self):
        """
        Modelo usado para ajustar.

        Si self.info contiene fit_model, lo usa.
        Si no, asume que el fit usó el mismo modelo simulado.
        """
        if isinstance(self.info, dict):
            return self.info.get("fit_model", self.model)

        return self.model


    def fit_has_parallax(self):
        """
        Determina si el fit incluyó paralaje.

        Si self.info contiene fit_parallax, lo usa.
        Si no existe, infiere desde el nombre del fit_model.
        """
        if isinstance(self.info, dict) and "fit_parallax" in self.info:
            return bool(self.info["fit_parallax"])

        fit_model = self.fit_model_name()

        if fit_model is None:
            return True

        return "NoPiE" not in str(fit_model)


    def fit_labels_params(self, data=None):
        """
        Parámetros físicos del modelo ajustado.

        Esto puede diferir de labels_params(), que corresponde
        al modelo simulado.

        Si data["best_model"] es un OrderedDict, extrae los nombres
        directamente desde ahí e ignora flujos y chi2.
        """

        if data is not None and isinstance(data, dict) and "best_model" in data:
            best_model = data["best_model"]

            if hasattr(best_model, "keys"):

                ignore_prefixes = (
                    "fsource_",
                    "ftotal_",
                    "fblend_",
                    "fs_",
                    "fb_",
                )

                labels = []

                for key in best_model.keys():

                    if key == "chi2":
                        continue

                    if any(str(key).startswith(prefix) for prefix in ignore_prefixes):
                        continue

                    labels.append(key)

                return labels

        fit_model = str(self.fit_model_name())

        if "USBL" in fit_model:
            labels = [
                "t_center",
                "u_center",
                "tE",
                "rho",
                "separation",
                "mass_ratio",
                "alpha",
            ]

        elif "FSPL" in fit_model:
            labels = [
                "t0",
                "u0",
                "tE",
                "rho",
            ]

        elif "PSPL" in fit_model:
            labels = [
                "t0",
                "u0",
                "tE",
            ]

        else:
            raise ValueError(f"fit_model no reconocido: {fit_model}")

        if self.fit_has_parallax():
            labels = labels + [
                "piEN",
                "piEE",
            ]

        return labels


    def _best_model_keys(self, data):
        """
        Devuelve las claves ordenadas de best_model si es dict-like.
        Si best_model es array/list, devuelve None.
        """
        best_model = data["best_model"]

        if hasattr(best_model, "keys"):
            return list(best_model.keys())

        return None


    def _best_model_value(self, data, key):
        """
        Lee un parámetro del best_model de forma robusta.
        """
        best_model = data["best_model"]

        if hasattr(best_model, "keys"):
            if key in best_model:
                return best_model[key]
            return np.nan

        labels = self.fit_labels_params(data)

        if key not in labels:
            return np.nan

        idx = labels.index(key)

        if idx >= len(best_model):
            return np.nan

        return best_model[idx]


    def _best_model_index(self, data, key):
        """
        Índice de un parámetro dentro del vector del fit.
        Sirve para leer covariance_matrix.
        """
        best_model = data["best_model"]

        if hasattr(best_model, "keys"):
            keys = list(best_model.keys())

            if key in keys:
                return keys.index(key)

            return None

        labels = self.fit_labels_params(data)

        if key in labels:
            return labels.index(key)

        return None
    

    def error_pyLIMA(self, data):
        """
        use fit_labels_params(data),
        not labels_params(), because the model fitted can be different from the
        true model.
        """

        labels = self.fit_labels_params(data)
        fit_error = {}

        if "covariance_matrix" not in data:
            for key in labels:
                fit_error[key] = np.nan
            return fit_error

        cov = data["covariance_matrix"]

        if cov is None:
            for key in labels:
                fit_error[key] = np.nan
            return fit_error

        diag = np.diag(cov)

        for key in labels:

            idx = self._best_model_index(data, key)

            if idx is None or idx >= len(diag):
                fit_error[key] = np.nan

            elif diag[idx] < 0:
                fit_error[key] = np.nan

            else:
                fit_error[key] = np.sqrt(diag[idx])

        return fit_error
        
    def load_data_fit(self):
        """
        Carga fit_rr y fit_roman si existen.

        En modo Rubin-only, path_fit_roman puede no existir.
        En ese caso deja fit_roman_data = None.
        """

        from pathlib import Path

        if self.fit_rr_data is None:
            if self.path_fit_rr is not None and Path(self.path_fit_rr).exists():
                self.fit_rr_data = np.load(
                    self.path_fit_rr,
                    allow_pickle=True,
                ).item()

        if self.fit_roman_data is None:
            if self.path_fit_roman is not None and Path(self.path_fit_roman).exists():
                self.fit_roman_data = np.load(
                    self.path_fit_roman,
                    allow_pickle=True,
                ).item()
            else:
                self.fit_roman_data = None

        return self.fit_roman_data, self.fit_rr_data

    def load_data_sim(self):
        if self.lightcurves is None:
            self.indices, self.info, self.model_params, self.computed_params, self.lightcurves, self.genulens_params, self.trilegal_params = read_data(self.path_model)

        
        return self.model_params, self.computed_params, self.lightcurves, self.genulens_params, self.trilegal_params
    

    def dict_fit_vals(self, data):
        """
        Diccionario de valores ajustados.

        Devuelve valores para:
        - los parámetros del modelo simulado, para mantener compatibilidad
          con el guardado viejo;
        - los parámetros reales del fit.

        Si un parámetro no fue ajustado, por ejemplo piEN/piEE en
        fit_parallax=False, queda como NaN.
        """

        fit_dict = {}

        fit_errors = self.error_pyLIMA(data)

        # Primero llenamos los parámetros del modelo simulado con NaN.
        # Esto mantiene compatibilidad con funciones que esperan columnas
        # como piEN/piEE aunque el fit no las tenga.
        for key in self.labels_params():
            fit_dict[key] = np.nan
            fit_dict[key + "_err"] = np.nan

        # Ahora llenamos los parámetros efectivamente ajustados.
        for key in self.fit_labels_params(data):

            fit_dict[key] = self._best_model_value(
                data,
                key,
            )

            fit_dict[key + "_err"] = fit_errors.get(
                key,
                np.nan,
            )

        return fit_dict
    
    def fit_has_piE_params(self, data):
        labels = self.fit_labels_params(data)
        return ("piEN" in labels) and ("piEE" in labels)
    def fit_values(self):
        """
        Devuelve fit_rr y fit_roman.

        En Rubin-only, fit_roman queda None.
        """

        self.load_data_fit()

        fit_rr = None
        fit_roman = None

        if self.fit_rr_data is not None:
            fit_rr = self.dict_fit_vals(self.fit_rr_data)

        if self.fit_roman_data is not None:
            fit_roman = self.dict_fit_vals(self.fit_roman_data)

        return fit_rr, fit_roman       

    def true_values(self):
        self.load_data_sim()
        model_dict = {}
        for i,key in enumerate(self.labels_params()):
            model_dict[key] = self.model_params[key]
        return model_dict    

    def chichi_dof(self, data):
        """
        Calcula chi2 y grados de libertad de forma robusta.

        Funciona para:
        - Roman+Rubin
        - Roman-only
        - Rubin-only
        - fits con o sin paralaje
        - fit_model distinto del modelo simulado
        """

        self.load_data_sim()

        chi2 = data["chi2"]

        # Número total de puntos observados en todas las bandas existentes.
        n_data = 0

        for key, lc in self.lightcurves.items():
            try:
                n_points = len(lc)
            except Exception:
                n_points = 0

            if n_points > 0:
                n_data += n_points

        best_model = data["best_model"]

        # Número de parámetros libres.
        # Si best_model es OrderedDict, contamos todo excepto chi2.
        if hasattr(best_model, "keys"):
            n_fit_params = len(
                [
                    key for key in best_model.keys()
                    if key != "chi2"
                ]
            )

        else:
            n_fit_params = len(best_model)

        dof = n_data - n_fit_params

        if dof <= 0:
            chi2_dof = np.nan
        else:
            chi2_dof = chi2 / dof

        return {
            "chi2": chi2,
            "dof": dof,
            "chi2_dof": chi2_dof,
        }

 
    def piE_MC(self, data):
        """
        Estima piE y su incertidumbre.

        Si el fit no tiene paralaje, devuelve NaN.
        """

        fit_vals = self.dict_fit_vals(data)

        if (
            "piEE" not in fit_vals
            or "piEN" not in fit_vals
            or np.isnan(fit_vals["piEE"])
            or np.isnan(fit_vals["piEN"])
        ):
            return {
                "piE": np.nan,
                "err_piE": np.nan,
            }

        dict_samples = self.samples_MC(data)

        samples_piE = np.sqrt(
            dict_samples["piEE_dist"]**2
            + dict_samples["piEN_dist"]**2
        )

        if np.all(np.isnan(samples_piE)):
            uncertainty = np.nan
        else:
            lower, upper = np.nanpercentile(
                samples_piE,
                [16, 84],
            )
            uncertainty = (upper - lower) / 2

        piE = np.sqrt(
            fit_vals["piEE"]**2
            + fit_vals["piEN"]**2
        )

        return {
            "piE": piE,
            "err_piE": uncertainty,
        }


    def thetaE(self, tE, mu_rel):
        '''
        input mu_rel in rad/years
        '''
        yr2day = 365.25
        thetaE = tE*mu_rel/yr2day
        return thetaE
    
    def mass(self, thetaE, piEN, piEE):
        aconv = (180 * 60 * 60 * 1000) / math.pi
        k= 4 * const.G / (const.c ** 2)
        mass =((thetaE/aconv**2)*u.kpc/
               (k*np.sqrt(piEN**2+piEE**2))).decompose().to('M_sun')
        return mass.value
        
    def mass_true(self):

        self.load_data_sim()
        thetaE = self.thetaE(self.model_params["tE"], self.genulens_params["mu_rel"])
        piEN = self.model_params["piEN"]
        piEE = self.model_params["piEE"]
        return self.mass(thetaE, piEN, piEE)
        
    def fit_mass_v1(self, data):
        '''
        Returns the mass computed with the true theta_E obtained from true tE
        -------
        mass_rr : float
            DESCRIPTION.

        '''
        if not self.fit_has_piE_params(data):
            return {
                "mass": np.nan,
                "err_mass": np.nan,
            }
        self.load_data_sim()
        tE = self.model_params["tE"]
        mu_rel = self.genulens_params["mu_rel"]
        thetaE = self.thetaE(tE, mu_rel)  # true thetaE with the true tE and mu_rel
        
        # fit_vals = {par:data['best_model'][i] for i,par in enumerate(self.labels_params())}
        fit_vals = self.dict_fit_vals(data)
        
        mass = self.mass(thetaE, fit_vals["piEN"], fit_vals["piEE"])
        dict_samples = self.samples_MC(data) 
        
        err_mass_v1 = self.mass(thetaE, dict_samples['piEN_dist'],
                                    dict_samples['piEE_dist'])  
        lower, upper = np.percentile(err_mass_v1, [16, 84])
        uncertainty = (upper - lower) / 2
        return {'mass':mass, 'err_mass':uncertainty}


    
    def fit_mass_v2(self, data):
        '''

        '''
        if not self.fit_has_piE_params(data):
            return {
                "mass": np.nan,
                "err_mass": np.nan,
            }
        self.load_data_sim()
        fit_vals = self.dict_fit_vals(data)
        # fit_vals = {par:data['best_model'][i] for i,par in enumerate(self.labels_params())}
        mu_rel = self.genulens_params["mu_rel"]
        thetaE = self.thetaE(fit_vals['tE'], mu_rel)
        mass = self.mass(thetaE, fit_vals["piEN"], fit_vals["piEE"])
        dict_samples = self.samples_MC(data)    
        err_mass_v2 = self.mass(dict_samples['thetaE_samples_wtE']
                                , dict_samples['piEN_dist'],
                                    dict_samples['piEE_dist'])    
            
        lower, upper = np.percentile(err_mass_v2, [16, 84])
        uncertainty = (upper - lower) / 2
        
        return {'mass':mass, 'err_mass':uncertainty}

    def fit_mass_v3(self, data):
        '''

        '''
        if not self.fit_has_piE_params(data):
            return {
                "mass": np.nan,
                "err_mass": np.nan,
            }
        # indices, strings, model_params, computed_params, curves, genulens_params, trilegal_params = read_data(self.path_model)
        self.load_data_sim()
        if (self.model == 'USBL') or (self.model == 'FSPL'):
            theta_s = np.arctan((self.computed_params["radius"]*u.R_sun/(self.genulens_params["D_S"]*u.pc))).to('mas').value      

            # fit_vals = {par:data['best_model'][i] for i,par in enumerate(self.labels_params())}
            fit_vals = self.dict_fit_vals(data)
            thetaE = theta_s/fit_vals['rho']
            mass = self.mass(thetaE, fit_vals["piEN"], fit_vals["piEE"])
            
            dict_samples = self.samples_MC(data)    
            err_mass_v3 = self.mass(dict_samples['thetaE_samples_wrho']
                                    , dict_samples['piEN_dist'],
                                    dict_samples['piEE_dist'])    
            
            lower, upper = np.percentile(err_mass_v3, [16, 84])
            uncertainty = (upper - lower) / 2
        else:
            raise Exception("Model not have rho to estimate thetaE = theta_s/rho")
        return {'mass':mass, 'err_mass':uncertainty}
        
    def samples_MC(self, data):
        """
        Estima muestras Monte Carlo usando la matriz de covarianza del fit.

        Si el fit no tiene piEN/piEE o rho, devuelve arrays de NaN
        para esas cantidades.
        """

        self.load_data_sim()

        n = 100000

        best_model = data["best_model"]

        if hasattr(best_model, "keys"):
            keys = list(best_model.keys())
            mean = np.array(
                [
                    best_model[key]
                    for key in keys
                ],
                dtype=float,
            )
        else:
            keys = self.fit_labels_params(data)
            mean = np.array(
                best_model,
                dtype=float,
            )

        if "covariance_matrix" not in data or data["covariance_matrix"] is None:
            samples = np.tile(
                mean,
                (n, 1),
            )
        else:
            samples = np.random.multivariate_normal(
                mean,
                data["covariance_matrix"],
                n,
            )

        def get_samples(key):
            if key not in keys:
                return np.full(n, np.nan)

            idx = keys.index(key)

            if idx >= samples.shape[1]:
                return np.full(n, np.nan)

            return samples[:, idx]

        piEE_dist = get_samples("piEE")
        piEN_dist = get_samples("piEN")

        tE_dist = get_samples("tE")

        if np.all(np.isnan(tE_dist)):
            thetaE_samples_wtE = np.full(n, np.nan)
        else:
            thetaE_samples_wtE = self.thetaE(
                tE_dist,
                self.genulens_params["mu_rel"],
            )

        rho_dist = get_samples("rho")

        if np.all(np.isnan(rho_dist)):
            thetaE_samples_wrho = np.full(n, np.nan)
        else:
            theta_s = np.arctan(
                (
                    self.computed_params["radius"]
                    * u.R_sun
                    / (self.genulens_params["D_S"] * u.pc)
                )
            ).to("mas").value

            thetaE_samples_wrho = theta_s / rho_dist

        return {
            "thetaE_samples_wrho": thetaE_samples_wrho,
            "thetaE_samples_wtE": thetaE_samples_wtE,
            "piEE_dist": piEE_dist,
            "piEN_dist": piEN_dist,
        }
    

    def piE_propagation(self, piEE, piEN, err_piEE, err_piEN, cov_piEE_piEN):
        '''
        
        
        '''
        piE = np.sqrt(piEE ** 2 + piEN ** 2)
        err_piE = (1 / piE) * np.sqrt((err_piEN * piEN) ** 2 + (
                    err_piEE * piEE) ** 2 +2*piEE * piEN*cov_piEE_piEN)
        return piE, err_piE

    def piE_analytical(self, data):
        """
        Versión analítica de piE.

        Si el fit no tiene piEN/piEE, devuelve NaN.
        """

        labels = self.fit_labels_params(data)

        if "piEN" not in labels or "piEE" not in labels:
            return {
                "piE": np.nan,
                "err_piE": np.nan,
            }

        piEN = self._best_model_value(data, "piEN")
        piEE = self._best_model_value(data, "piEE")

        idx_piEN = self._best_model_index(data, "piEN")
        idx_piEE = self._best_model_index(data, "piEE")

        if (
            idx_piEN is None
            or idx_piEE is None
            or "covariance_matrix" not in data
            or data["covariance_matrix"] is None
        ):
            return {
                "piE": np.sqrt(piEE**2 + piEN**2),
                "err_piE": np.nan,
            }

        cov = data["covariance_matrix"]

        err_piEN = np.sqrt(cov[idx_piEN, idx_piEN])
        err_piEE = np.sqrt(cov[idx_piEE, idx_piEE])
        cov_piEE_piEN = cov[idx_piEE, idx_piEN]

        piE, err_piE = self.piE_propagation(
            piEE,
            piEN,
            err_piEE,
            err_piEN,
            cov_piEE_piEN,
        )

        return {
            "piE": piE,
            "err_piE": err_piE,
        }
        


    def count_points(self):
        '''
        return the number of data point in each band
        '''
        self.load_data_sim()

        return {key:len(self.lightcurves[key]) for key in self.lightcurves}

    def count_points_peak(self):
        '''
        return the number of data point in each band
        '''
        self.load_data_sim()

        t0_str = 't0'
        t0 = self.model_params[t0_str]
        tE = self.model_params['tE']
        
        return {key:len(self.lightcurves[key][(self.lightcurves[key]['time']>t0-tE)&(self.lightcurves[key]['time']<t0+tE)]) for key in self.lightcurves}
        
    def count_points_secon_peak(self):
        '''
        return the number of data point in each band
        '''
        self.load_data_sim()
        if self.model =='USBL':
            t_center = self.model_params['t_center']
            tE = self.model_params['tE']
            q = self.model_params['mass_ratio']
            tEp = 5*np.sqrt(q)*tE
            return {key:len(self.lightcurves[key][(self.lightcurves[key]['time']>t_center-tEp)&(self.lightcurves[key]['time']<t_center+tEp)]) for key in self.lightcurves}

        else:
            return 

    def count_points_narrow_peak(self):
        '''
        return the number of data point in each band
        '''
        self.load_data_sim()
        t0_str = 't0'
        t0 = self.model_params[t0_str]
        tE = self.model_params['tE']
        
        return {key:len(self.lightcurves[key][(self.lightcurves[key]['time']>t0-0.25*tE)&(self.lightcurves[key]['time']<t0+0.25*tE)]) for key in self.lightcurves}

    def count_points_left_peak(self):
        '''
        return the number of data point in each band
        '''
        self.load_data_sim()
        t0_str = 't0'
        t0 = self.model_params[t0_str]
        tE = self.model_params['tE']
        
        return {key:len(self.lightcurves[key][(self.lightcurves[key]['time']>t0-tE)&(self.lightcurves[key]['time']<t0)]) for key in self.lightcurves}

    def count_points_right_peak(self):
        '''
        return the number of data point in each band
        '''
        self.load_data_sim()
        t0_str = 't0'
        t0 = self.model_params[t0_str]
        tE = self.model_params['tE']
        return {key:len(self.lightcurves[key][(self.lightcurves[key]['time']>t0)&(self.lightcurves[key]['time']<t0+tE)]) for key in self.lightcurves}
        
