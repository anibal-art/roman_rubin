import numpy as np
import pandas as pd
import os, sys, re, math, h5py

from pyLIMA.outputs import pyLIMA_plots

sys.path.append(os.path.dirname(os.getcwd()))
from fit_lc import model_rubin_roman
from read_save import read_data
from bokeh.plotting import figure, show, output_file, reset_output
from bokeh.layouts import gridplot, row, column
from bokeh.io import export_png
from bokeh.models import ColumnDataSource, DataTable, TableColumn
from bokeh.models import Span

from decimal import Decimal, ROUND_DOWN, InvalidOperation

# from decimal import Decimal, ROUND_DOWN

def round_nfig(numero):
    if numero == 0:
        return 0
    cifras_significativas = 3
    # Calcula el número de decimales necesarios
    exponente = int("{:e}".format(abs(numero)).split("e")[1])
    decimales = max(cifras_significativas - 1 - exponente, 2)  # Asegura al menos 2 decimales
    return round(numero, decimales)
    
# def round_with_matching_decimals(num1, num2):
#     # Convert to Decimal for precision
#     num1, num2 = Decimal(num1), Decimal(num2)

#     # Identify the smaller and larger numbers
#     smaller, larger = (num1, num2) if abs(num1) < abs(num2) else (num2, num1)

#     # Round the smaller number to 3 significant figures
#     rounded_smaller = round(smaller, 3 - smaller.adjusted() - 1)

#     # Determine the number of decimal places in the rounded smaller number
#     decimal_places = abs(Decimal(str(rounded_smaller)).as_tuple().exponent)
#     print('larger',larger)
#     print('decimal_places',decimal_places)
#     # Round the larger number to match the decimal places
#     rounded_larger = larger.quantize(Decimal(f"1E-{decimal_places}"), rounding=ROUND_DOWN)

#     return float(rounded_smaller), float(rounded_larger)


def round_with_matching_decimals(num1, num2, max_decimal_places=10):
    try:
        # Convert to Decimal for precision
        num1, num2 = Decimal(num1), Decimal(num2)

        # Identify the smaller and larger numbers
        smaller, larger = (num1, num2) if abs(num1) < abs(num2) else (num2, num1)

        # Round the smaller number to 3 significant figures
        rounded_smaller = round(smaller, 3 - smaller.adjusted() - 1)

        # Determine the number of decimal places in the rounded smaller number
        decimal_places = abs(Decimal(str(rounded_smaller)).as_tuple().exponent)
        decimal_places = min(decimal_places, max_decimal_places)  # Limitar la precisión

        print('larger', larger)
        print('decimal_places', decimal_places)

        # Round the larger number to match the decimal places
        quant = Decimal(f"1E-{decimal_places}")
        rounded_larger = larger.quantize(quant, rounding=ROUND_DOWN)

        return float(rounded_smaller), float(rounded_larger)

    except (InvalidOperation, ValueError) as e:
        print(f"[ERROR round_with_matching_decimals]: {e}")
        return float(num1), float(num2)

def round_single(x):
    return round(x, -int(np.floor(np.log10(abs(x)))))  # Example function to round to significant figures


def plot_n_save(Source,path_save,path_event, path_fit_rr, path_fit_roman, path_ephemerides, model):
    
    colorbands={'W149':'b', 'u':'purple', 'g':'g', 'r':'red',
          'i':'yellow', 'z':'k', 'y':'cyan'}

    
    ZP = {'W149':27.615, 'u':27.03, 'g':28.38, 'r':28.16,
              'i':27.85, 'z':27.46, 'y':26.68}
    # model_ulens = 'USBL'

    
    info, pyLIMA_parameters, bands = read_data(path_event)
    data_fit_rr = np.load(path_fit_rr,allow_pickle=True).item()
    data_fit_roman = np.load(path_fit_roman,allow_pickle=True).item()
    origin = info[2]
    ulens_params = []

    PAR = ['tE','piEN','piEE']
    if model == 'USBL':
        PAR = ['t_center','u_center','rho','separation','mass_ratio','alpha']+PAR
    elif model=='FSPL':
        PAR = ['t0','u0','rho']+PAR
    elif model =='PSPL':
        PAR = ['t0','u0']+PAR
        
    for b in (PAR):
        ulens_params.append(pyLIMA_parameters[b])
    flux_params = []
    for b in bands:
        if not len(bands[b])==0:
            zp_Rubin_to_pyLIMA = (10**((-27.4+ZP[b])/2.5))
            
            flux_params.append(pyLIMA_parameters['fsource_'+b]/zp_Rubin_to_pyLIMA)
            flux_params.append(pyLIMA_parameters['ftotal_'+b]/zp_Rubin_to_pyLIMA)
            
    true_params = ulens_params+flux_params
    
    
    f= 'W149'
    wfirst_lc = np.array([bands[f]['time'],bands[f]['mag'],bands[f]['err_mag']]).T
    f = 'u'
    lsst_u = np.array([bands[f]['time'],bands[f]['mag'],bands[f]['err_mag']]).T
    f='g'
    lsst_g = np.array([bands[f]['time'],bands[f]['mag'],bands[f]['err_mag']]).T
    f='r'
    lsst_r = np.array([bands[f]['time'],bands[f]['mag'],bands[f]['err_mag']]).T
    f='i'
    lsst_i = np.array([bands[f]['time'],bands[f]['mag'],bands[f]['err_mag']]).T
    f='z'
    lsst_z = np.array([bands[f]['time'],bands[f]['mag'],bands[f]['err_mag']]).T
    f='y'
    lsst_y = np.array([bands[f]['time'],bands[f]['mag'],bands[f]['err_mag']]).T
    
    # model_true = model_rubin_roman(Source,True,event_params, path_ephemerides,model, wfirst_lc, lsst_u, lsst_g, lsst_r, lsst_i, lsst_z,
    #                     lsst_y)
    if model=='USBL':
        t0_str='t_center'
        u0_str='u_center'
    else:
        t0_str='t0'
        u0_str='u0'
        
    model_rr = model_rubin_roman(Source,False,{t0_str:data_fit_rr['best_model'][0]}, path_ephemerides,model,origin, wfirst_lc, lsst_u, lsst_g, lsst_r, lsst_i, lsst_z,
                        lsst_y)
    
    model_roman = model_rubin_roman(Source,False,{t0_str:data_fit_roman['best_model'][0]}, path_ephemerides,model,origin, wfirst_lc, [], [], [], [], [],[])
    
    chi2_roman = data_fit_roman['chi2']
    chi2_rr = data_fit_rr['chi2']
    DOF_roman = len(model_roman.event.telescopes[0].lightcurve['time'])-len(data_fit_roman['best_model'])
    DOF_rr = sum([len(model_rr.event.telescopes[i].lightcurve['time']) for i in range(len(model_rr.event.telescopes))])-len(data_fit_rr['best_model'])
    print('Chi square reduced for Roman', chi2_roman/DOF_roman)
    print('Chi square reduced for RR', chi2_rr/DOF_rr)
    
    # Create a Bokeh figure
    bokeh_plot = figure(
        width=800*2, height=400*2,
        title="Microlensing Photometric Models",
        x_axis_label="Time",
        y_axis_label="Magnitude",
        tools="pan,wheel_zoom,box_zoom,reset"
    )
    
    # Call the function to plot the photometric models
    pyLIMA_plots.plot_photometric_models(
        figure_axe=None,  # Pass None if you're only using Bokeh
        microlensing_model=model_rr,
        model_parameters=data_fit_rr['best_model'],
        bokeh_plot=bokeh_plot,
        plot_unit='Mag'
    )
    # Call the function to plot the photometric models
    pyLIMA_plots.plot_photometric_models(
        figure_axe=None,  # Pass None if you're only using Bokeh
        microlensing_model=model_roman,
        model_parameters=data_fit_roman['best_model'],
        bokeh_plot=bokeh_plot,
        plot_unit='Mag'
    )
    
    pyLIMA_plots.plot_aligned_data(figure_axe=None,  # Pass None if you're only using Bokeh
        microlensing_model=model_rr,
        model_parameters=data_fit_rr['best_model'],
        bokeh_plot=bokeh_plot,
        plot_unit='Mag'
    )
    
    bokeh_plot.renderers[0].glyph.line_color = "blue"
    bokeh_plot.renderers[1].glyph.line_color = "red"
    
    bokeh_plot.y_range.flipped = True
    
    plot_residual_rr = figure(title=f"Residuals Roman+Rubin fit: chi2={chi2_rr/DOF_rr}", width=800*2, height=300)
    pyLIMA_plots.plot_residuals(None, model_rr, data_fit_rr['best_model'],bokeh_plot=plot_residual_rr, plot_unit='Mag')
    
    
    plot_residual_roman = figure(title=f"Residuals Roman fit: chi2={chi2_roman/DOF_roman}", width=800*2, height=300)
    pyLIMA_plots.plot_residuals(None, model_roman, data_fit_roman['best_model'],bokeh_plot=plot_residual_roman, plot_unit='Mag')

    # show(bokeh_plot)

    if True:
    
        
        # Disposición vertical
        layout_vertical = column(bokeh_plot, plot_residual_rr,plot_residual_roman)
        # export_png(layout_vertical, filename="grid_plot.png")
    
    
        # print('p-value of a chi squared test', chi2.sf(chi2_roman, DOF_roman))
        
        #metric alpha
        long = len(data_fit_roman['best_model'])-2
        TRUE = np.array(true_params[0:long])
        F_rr = np.array(data_fit_rr['best_model'][0:long])
        F_roman = np.array(data_fit_roman['best_model'][0:long])
        S_rr =np.array(np.sqrt(np.diag(data_fit_rr['covariance_matrix']))[0:long])
        S_roman =np.array(np.sqrt(np.diag(data_fit_roman['covariance_matrix']))[0:long])
        
        alpha_roman = abs(TRUE-F_roman)/abs(TRUE)
        alpha_rr = abs(TRUE-F_rr)/abs(TRUE)
        #metric beta
        beta_roman = abs(TRUE-F_roman)/abs(S_roman)
        beta_rr = abs(TRUE-F_rr)/abs(S_rr)
        #metric gamma
        gamma_roman = S_roman/abs(F_roman)
        gamma_rr = S_rr/abs(F_rr)
        print(alpha_roman,alpha_rr)
    
        data = dict(
            Name=["t_center", "u_center", "tE", "rho","s","q","alpha","piEE","piEN"],
            true=TRUE,
            # fit_rr=F_rr,
            # sigma_rr=S_rr,
            # fit_roman=F_roman,
            # sigma_roman=S_roman,
            fit_unc_rr=[],
            fit_unc_roman=[],
            met1_rr = [round_nfig(n) for n in alpha_rr],
            met2_rr = [round_nfig(n) for n in beta_rr],
            met3_rr = [round_nfig(n) for n in gamma_rr],
            met1_roman = [round_nfig(n) for n in alpha_roman],
            met2_roman = [round_nfig(n) for n in beta_roman],
            met3_roman = [round_nfig(n) for n in gamma_roman]
        )
    
    
    # Applying rounding function to fit and sigma values
        for fit_rr, sigma_rr, fit_roman, sigma_roman in zip(F_rr, S_rr, F_roman, S_roman):
            rounded_sigma_rr, rounded_fit_rr = round_with_matching_decimals(sigma_rr, fit_rr)
            rounded_sigma_roman, rounded_fit_roman = round_with_matching_decimals(sigma_roman, fit_roman)
            # Creating combined columns
            data["fit_unc_rr"].append(f"{rounded_fit_rr} ± {rounded_sigma_rr}")
            data["fit_unc_roman"].append(f"{rounded_fit_roman} ± {rounded_sigma_roman}")
    
        source = ColumnDataSource(data)
        
        columns = [
            TableColumn(field="Name", title="Parameters"),    # Editable text
            TableColumn(field="true", title="True"),  # Editable number
            TableColumn(field="fit_unc_rr", title="Fit ± σ (RR)"),   # Editable number
            # TableColumn(field="sigma_rr", title="σ  RR"),   # Editable number
            TableColumn(field="fit_unc_roman", title="Fit ± σ (Roman)"),   # Editable number
            # TableColumn(field="sigma_roman", title="σ  Roman"),   # Editable number
            TableColumn(field="met1_rr", title="|Fit-True|/True (RR)"),   # Editable number
            TableColumn(field="met1_roman", title="|Fit-True|/True (Roman)"),   # Editable number
            TableColumn(field="met2_rr", title="|Fit-True|/σ (RR)"),   # Editable number
            TableColumn(field="met2_roman", title="|Fit-True|/σ (Roman)"),   # Editable number
            TableColumn(field="met3_rr", title="σ/Fit (RR)"),   # Editable number
            TableColumn(field="met3_roman", title="σ/Fit (Roman)")   # Editable number
        ]
    
        data_table = DataTable(source=source, columns=columns, width= 1500, height=350, editable=True)
        
        # Combine grid and table in a row layout
        layout2 = column(layout_vertical, data_table)
    
        xticks = ['t_center', 'u_center', 'tE', 'ρ', 's', 'q', 'α', 'πEN', 'πEE']
        rango=1
        # Create the first plot (alpha) with markers only
        p1 = figure(x_range=xticks, width=800, height=250, toolbar_location=None, y_axis_type="log")
        p1.circle(xticks, alpha_roman, size=8, color="blue", legend_label="Roman")
        p1.circle(xticks, alpha_rr, size=8, color="green", legend_label="Roman + Rubin")
        p1.yaxis.axis_label = '|Fit-True|/True'
        # Add horizontal reference line
        hline = Span(location=rango, dimension='width', line_color='red', line_width=2)
        p1.add_layout(hline)
        p1.legend.orientation = "vertical"
        p1.legend.location = "top_right"  # Move to the side
        p1.legend.location = "top_left"
        p1.xaxis.major_label_orientation = 0.8
        p1.grid.visible = True
        
        # Create the second plot (beta) with markers only
        p2 = figure(x_range=xticks, width=800, height=250, toolbar_location=None, y_axis_type="log")
        p2.circle(xticks, beta_roman, size=8, color="blue", legend_label="Roman")
        p2.circle(xticks, beta_rr, size=8, color="green", legend_label="Roman + Rubin")
        p2.yaxis.axis_label = '|Fit-True|/ σ'
        p2.legend.orientation = "vertical"
        p2.legend.location = "top_right"  # Move to the side
    
        # p2.add_layout(hline)
        p2.legend.location = "top_left"
        p2.grid.visible = True
        
        # Create the third plot (gamma) with markers only and log y-axis
        p3 = figure(x_range=xticks, width=800, height=250, toolbar_location=None, y_axis_type="log")
        p3.circle(xticks, gamma_roman, size=8, color="blue", legend_label="Roman")
        p3.circle(xticks, gamma_rr, size=8, color="green", legend_label="Roman + Rubin")
        # p3.add_layout(hline)
        p3.legend.orientation = "vertical"
        p3.legend.location = "top_right"  # Move to the side
    
        p3.yaxis.axis_label = 'σ/Fit'
        p3.legend.location = "top_left"
        p3.grid.visible = True
        p3.xaxis.axis_label = "Parameters"
        
        # Arrange the plots in a vertical column
        layout = column(layout2,p1, p2, p3)
        # show(layout)
    
        output_file(path_save + f"Event_{Source}.html")

        show(layout)
        reset_output()