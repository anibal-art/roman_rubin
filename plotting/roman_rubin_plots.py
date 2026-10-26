"""Roman+Rubin light-curve figures from saved simulations and fits.

plot_roman_rubin_plt: data, true model and the Roman+Rubin / Roman-only fit
models of one event (models rebuilt at the RA/Dec stored in each fit).
graph_maker_1plot / graph_maker: figure grids; create_df_to_plot and
params_list_to_dict: table helpers used by the notebooks.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLIMA_plots
from analysis.analysis_event import Analysis_Event
from cycler import cycler
from fit_lc import model_rubin_roman


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
                                 lightcurve_dict['i'], lightcurve_dict['z'],lightcurve_dict['y'],
                                 event_ra=data_fit_rr['event_ra'],
                                 event_dec=data_fit_rr['event_dec'])
    
    model_roman = model_rubin_roman(Source, {t0_str:data_fit_roman['best_model'][0]}, 
                                    path_ephemerides, model, origin, lightcurve_dict['W149'], [], [], [], [], [],[],
                                    event_ra=data_fit_roman['event_ra'],
                                    event_dec=data_fit_roman['event_dec'])
    
    
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
