import sys
import cycler
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyLIMA.fits.objective_functions
from bokeh.layouts import gridplot
from bokeh.models import Arrow, OpenHead
from bokeh.models import BasicTickFormatter
from bokeh.models import ColumnDataSource, DataTable, TableColumn
from bokeh.plotting import figure
from matplotlib.ticker import MaxNLocator
from pyLIMA.astrometry import astrometric_positions
from pyLIMA.parallax import parallax
from pyLIMA.toolbox import fake_telescopes, plots

import io
from PIL import Image

plot_lightcurve_windows = 0.2
plot_residuals_windows = 0.21
MAX_PLOT_TICKS = 2
MARKER_SYMBOLS = np.array(
    [['o', '.', '*', 'v', '^', '<', '>', 's', 'p', 'd', 'x'] * 10])

MARKERS_COLORS = plt.rcParams["axes.prop_cycle"]
# this is a pointer to the module object instance itself.
thismodule = sys.modules[__name__]

thismodule.list_of_fake_telescopes = []
thismodule.saved_model = None

def update_matplotlib_colors(event):
    # Change matplotlib default colors
    n_telescopes = len(event.telescopes)
    color = plt.cm.jet(
        np.linspace(0.01, 0.99, n_telescopes))  # This returns RGBA; convert:
    # hexcolor = map(lambda rgb: '#%02x%02x%02x' % (rgb[0] * 255, rgb[1] * 255,
    # rgb[2] * 255),
    #                tuple(color[:, 0:-1]))
    hexcolor = ['#' + format(int(i[0] * 255), 'x').zfill(2) + format(int(i[1] * 255),
                                                                     'x').zfill(2) +
                format(int(i[2] * 255), 'x').zfill(2) for i in color]

    thismodule.MARKERS_COLORS = cycler.cycler(color=hexcolor)


def create_telescopes_to_plot_model(microlensing_model, pyLIMA_parameters):
    # print('BL' in microlensing_model.model_type())
    if microlensing_model == thismodule.saved_model:

        list_of_fake_telescopes = thismodule.list_of_fake_telescopes

    else:

        list_of_fake_telescopes = []

    if len(list_of_fake_telescopes) == 0:

        # Photometry first
        Earth = True

        for index,tel in enumerate(microlensing_model.event.telescopes):

            if tel.lightcurve is not None:

                if Earth and tel.location == 'Earth':

                    model_time1 = np.arange(np.min((np.min(
                        tel.lightcurve['time'].value),
                                                    pyLIMA_parameters['t0'] - 5 *
                                                    pyLIMA_parameters['tE'])),
                        np.max((np.max(
                            tel.lightcurve['time'].value),
                                pyLIMA_parameters['t0'] + 5 *
                                pyLIMA_parameters['tE'])),
                        10).round(2)
                    if 'FSPL' in microlensing_model.model_type():
                        model_time2 = np.arange(
                            pyLIMA_parameters['t0'] - 1 * pyLIMA_parameters['tE'],
                            pyLIMA_parameters['t0'] + 1 * pyLIMA_parameters['tE'],
                            1/300).round(2)
                    else:
                        model_time2 = np.arange(
                            pyLIMA_parameters['t0'] - 1 * pyLIMA_parameters['tE'],
                            pyLIMA_parameters['t0'] + 1 * pyLIMA_parameters['tE'],
                            0.25).round(2)
                    if 'BL' in microlensing_model.model_type():
                        model_time3 = np.linspace(pyLIMA_parameters['t_center']
                                                -2*np.sqrt(pyLIMA_parameters['mass_ratio'])*
                                                pyLIMA_parameters['tE'],
                                                pyLIMA_parameters['t_center'] + 2 *
                                                np.sqrt(pyLIMA_parameters['mass_ratio'])*
                                                pyLIMA_parameters['tE'],
                                                1000)#.round(5)
                                            
                        model_time = np.r_[model_time1, model_time2, model_time3]           
                    else:
                        model_time = np.r_[model_time1, model_time2]

                    
                    # model_time = np.r_[model_time1, model_time2]

                    for telescope in microlensing_model.event.telescopes:

                        if telescope.lightcurve is not None:

                            if telescope.location == 'Earth':
                                model_time = np.r_[
                                    model_time, telescope.lightcurve[
                                        'time'].value]

                                symmetric = 2 * pyLIMA_parameters['t0'] - \
                                            telescope.lightcurve['time'].value
                                model_time = np.r_[model_time, symmetric]


                    model_time.sort()

                    model_time = np.unique(model_time)

                    model_telescope = fake_telescopes.replicate_a_telescope(
                        microlensing_model, index,
                        lightcurve_time=model_time,
                        astrometry_time=None)

                    list_of_fake_telescopes.append(model_telescope)

                if tel.location == 'Space':
                    model_time1 = np.arange(np.min((np.min(
                        tel.lightcurve['time'].value),
                                                    pyLIMA_parameters['t0'] - 5 *
                                                    pyLIMA_parameters['tE'])),
                        np.max((np.max(
                            tel.lightcurve['time'].value),
                                pyLIMA_parameters['t0'] + 5 *
                                pyLIMA_parameters['tE'])),
                        10).round(2)

                    model_time2 = np.arange(
                        pyLIMA_parameters['t0'] - 1 * pyLIMA_parameters['tE'],
                        pyLIMA_parameters['t0'] + 1 * pyLIMA_parameters['tE'],
                        0.25).round(2)


                    if 'BL' in microlensing_model.model_type():
                        model_time3 = np.linspace(pyLIMA_parameters['t_center']
                                                -3*np.sqrt(pyLIMA_parameters['mass_ratio'])*
                                                pyLIMA_parameters['tE'],
                                                pyLIMA_parameters['t_center'] + 3 *
                                                np.sqrt(pyLIMA_parameters['mass_ratio'])*
                                                pyLIMA_parameters['tE'],
                                                2000)#.round(5)
                                            
                        model_time = np.r_[model_time1, model_time2, model_time3,tel.lightcurve[
                                                'time'].value]
                        
                    elif 'FSPL' in microlensing_model.model_type():
                        model_time2 = np.arange(
                            pyLIMA_parameters['t0'] - 1 * pyLIMA_parameters['tE'],
                            pyLIMA_parameters['t0'] + 1 * pyLIMA_parameters['tE'],
                            1/300).round(2)
                        model_time = np.r_[model_time1, model_time2,tel.lightcurve[
                                                'time'].value]
                    else:
                        model_time = np.r_[model_time1, model_time2]


                        model_time = np.r_[model_time1, model_time2,tel.lightcurve[
                                                'time'].value]


                    
                    
                    # model_time = np.r_[model_time1, model_time2,tel.lightcurve[
                    #     'time'].value]

                    # mask = ((model_time >= tel.lightcurve['time'].value.min()))
                            # & (
                            # model_time <= tel.lightcurve['time'].value.max()))
                    model_time = model_time#[mask]

                    model_time.sort()

                    model_time = np.unique(model_time)

                    model_telescope = fake_telescopes.replicate_a_telescope(
                            microlensing_model, index,
                            lightcurve_time=model_time,
                            astrometry_time=None)

                    list_of_fake_telescopes.append(model_telescope)

                if tel.location == 'Earth' and Earth:
                    Earth = False


        # Astrometry

        for index,tel in enumerate(microlensing_model.event.telescopes):

            if tel.astrometry is not None:

                #if tel.location == 'Space':

                #    model_time = np.arange(np.min(tel.astrometry['time'].value),
                #                           np.max(tel.astrometry['time'].value),
                #                           1).round(2)
                #else:

                model_time1 = np.arange(
                    np.min((np.min(tel.astrometry['time'].value),
                            pyLIMA_parameters['t0'] - 5 * pyLIMA_parameters['tE'])),
                    np.max((np.max(tel.astrometry['time'].value),
                            pyLIMA_parameters['t0'] + 5 * pyLIMA_parameters['tE'])),
                    10).round(2)

                model_time2 = np.arange(
                    pyLIMA_parameters['t0'] - 1 * pyLIMA_parameters['tE'],
                    pyLIMA_parameters['t0'] + 1 * pyLIMA_parameters['tE'],
                    1).round(2)

                model_time = np.r_[model_time1, model_time2]

                model_time = np.r_[model_time, tel.astrometry['time'].value]

                symmetric = 2 * pyLIMA_parameters['t0'] - tel.astrometry[
                    'time'].value
                model_time = np.r_[model_time, symmetric]

                if tel.location == 'Space':

                    mask = ((model_time >= tel.lightcurve['time'].value.min())
                            & (
                            model_time <= tel.lightcurve['time'].value.max()))
                    model_time = model_time[mask]

                model_time.sort()

                model_time = np.unique(model_time)
                model_telescope = fake_telescopes.replicate_a_telescope(
                    microlensing_model, index,
                    lightcurve_time=None,
                    astrometry_time=model_time)

                list_of_fake_telescopes.append(model_telescope)

        thismodule.saved_model = microlensing_model
        thismodule.list_of_fake_telescopes = list_of_fake_telescopes

    return list_of_fake_telescopes


def plot_lightcurves(microlensing_model, model_parameters, bokeh_plot=None):

    mat_figure, mat_figure_axes = initialize_light_curves_plot(
        event_name=microlensing_model.event.name)

    if bokeh_plot is not None:

        bokeh_lightcurves = figure(width=900, height=600, toolbar_location=None,
                                   y_axis_label=r'$$m [mag]$$')
        bokeh_residuals = figure(width=bokeh_lightcurves.width, height=200,
                                 x_range=bokeh_lightcurves.x_range,
                                 y_range=(0.18, -0.18), toolbar_location=None,
                                 x_axis_label='JD', y_axis_label=r'$$\Delta m [mag]$$')

        bokeh_lightcurves.xaxis.minor_tick_line_color = None
        bokeh_lightcurves.xaxis.major_tick_line_color = None
        bokeh_lightcurves.xaxis.major_label_text_font_size = '0pt'
        bokeh_lightcurves.y_range.flipped = True
        bokeh_lightcurves.xaxis.formatter = BasicTickFormatter(use_scientific=False)

        bokeh_residuals.xaxis.formatter = BasicTickFormatter(use_scientific=False)
        bokeh_residuals.xaxis.major_label_orientation = np.pi / 4
        bokeh_residuals.xaxis.minor_tick_line_color = None

    else:

        bokeh_lightcurves = None
        bokeh_residuals = None

    if len(model_parameters) != len(microlensing_model.model_dictionnary):
        telescopes_fluxes = microlensing_model.find_telescopes_fluxes(model_parameters)
        telescopes_fluxes = [telescopes_fluxes[key] for key in
                             telescopes_fluxes.keys()]

        model_parameters = np.r_[model_parameters, telescopes_fluxes]

    plot_photometric_models(mat_figure_axes[0], microlensing_model, model_parameters,MARKERS_COLORS,
                            plot_unit='Mag',
                            bokeh_plot=bokeh_lightcurves)

    plot_aligned_data(mat_figure_axes[0], microlensing_model, model_parameters,
                      plot_unit='Mag',
                      bokeh_plot=bokeh_lightcurves)

    plot_residuals(mat_figure_axes[1], microlensing_model, model_parameters,
                   plot_unit='Mag',
                   bokeh_plot=bokeh_residuals)

    mat_figure_axes[0].invert_yaxis()
    mat_figure_axes[1].invert_yaxis()
    mat_figure_axes[0].legend(shadow=True, fontsize='large',
                              bbox_to_anchor=(0, 1.02, 1, 0.2),
                              loc="lower left",
                              mode="expand", borderaxespad=0, ncol=3)

    try:
        bokeh_lightcurves.legend.click_policy = "mute"
        # legend = bokeh_lightcurves.legend[0]

    except AttributeError:

        pass

    figure_bokeh = gridplot([[bokeh_lightcurves], [bokeh_residuals]],
                            toolbar_location=None)

    return mat_figure, figure_bokeh


def initialize_light_curves_plot(plot_unit='Mag', event_name='A microlensing event'):
    fig_size = [10, 10]
    mat_figure, mat_figure_axes = plt.subplots(2, 1, sharex=True,
                                               gridspec_kw={'height_ratios': [3, 1]},
                                               figsize=(fig_size[0], fig_size[1]),
                                               dpi=75)
    plt.subplots_adjust(top=0.8, bottom=0.15, left=0.2, right=0.9, wspace=0.1,
                        hspace=0.1)
    mat_figure_axes[0].grid()
    mat_figure_axes[1].grid()
    # mat_figure.suptitle(event_name, fontsize=30 * fig_size[0] / len(event_name))

    mat_figure_axes[0].set_ylabel(r'$' + plot_unit + '$',
                                  fontsize=5 * fig_size[1] * 3 / 4.0)
    mat_figure_axes[0].yaxis.set_major_locator(MaxNLocator(4))
    mat_figure_axes[0].tick_params(axis='y', labelsize=3.5 * fig_size[1] * 3 / 4.0)

    mat_figure_axes[0].text(0.01, 0.96, 'provided by pyLIMA', style='italic',
                            fontsize=10,
                            transform=mat_figure_axes[0].transAxes)

    mat_figure_axes[1].set_xlabel(r'$JD$', fontsize=5 * fig_size[0] * 3 / 4.0)
    mat_figure_axes[1].xaxis.set_major_locator(MaxNLocator(3))
    mat_figure_axes[1].yaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))

    mat_figure_axes[1].ticklabel_format(useOffset=False, style='plain')
    mat_figure_axes[1].set_ylabel(r'$\Delta M$', fontsize=5 * fig_size[1] * 2 / 4.0)
    mat_figure_axes[1].tick_params(axis='x', labelsize=3.5 * fig_size[0] * 3 / 4.0)
    mat_figure_axes[1].tick_params(axis='y', labelsize=3.5 * fig_size[1] * 3 / 4.0)

    return mat_figure, mat_figure_axes


def plot_photometric_models(figure_axe, microlensing_model, model_parameters,MARKERS_COLORS,
                            bokeh_plot=None, plot_unit='Mag'):
    pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(model_parameters)

    list_of_telescopes = create_telescopes_to_plot_model(microlensing_model,
                                                         pyLIMA_parameters)
    telescopes_names = np.array([i.name for i in microlensing_model.event.telescopes])

    # plot models
    index = 0

    for tel in list_of_telescopes:

        if tel.lightcurve is not None:

            magni = microlensing_model.model_magnification(tel, pyLIMA_parameters)
            microlensing_model.derive_telescope_flux(tel, pyLIMA_parameters, magni)

            f_source = pyLIMA_parameters['fsource_' + tel.name]
            f_blend = pyLIMA_parameters['fblend_' + tel.name]

            if index == 0:
                ref_source = f_source
                ref_blend = f_blend
                index += 1

            magnitude = pyLIMA.toolbox.brightness_transformation.ZERO_POINT - 2.5 * \
                        np.log10(ref_source * magni + ref_blend)

            # delta_mag = -2.5 * np.log10(f_source + f_blend) + 2.5 * np.log10(
            ##     ref_source + ref_blend)
            # magnitude -= delta_mag

            name = tel.name

            index_color = np.where(name == telescopes_names)[0][0]
            color = MARKERS_COLORS.by_key()["color"][index_color]

            if tel.location == 'Earth':

                name = tel.location
                linestyle = '-'

            else:

                linestyle = '--'


            if bokeh_plot is not None:
                bokeh_plot.line(tel.lightcurve['time'].value, magnitude,
                                legend_label=name, color=color)
            else:
                
                plots.plot_light_curve_magnitude(tel.lightcurve['time'].value,
                                                 magnitude, figure_axe=figure_axe,
                                                 name=name, color=color,
                                                 linestyle=linestyle)


def plot_aligned_data(figure_axe, microlensing_model, model_parameters, bokeh_plot=None,
                      plot_unit='Mag'):
    pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(model_parameters)

    # plot aligned data
    index = 0

    list_of_telescopes = create_telescopes_to_plot_model(microlensing_model,
                                                         pyLIMA_parameters)

    ref_names = []
    ref_locations = []
    ref_magnification = []
    ref_fluxes = []

    for ref_tel in list_of_telescopes:
        if ref_tel.lightcurve is not None:
            model_magnification = microlensing_model.model_magnification(ref_tel,
                                                                         pyLIMA_parameters)

            microlensing_model.derive_telescope_flux(ref_tel, pyLIMA_parameters,
                                                     model_magnification)

            f_source = pyLIMA_parameters['fsource_' + ref_tel.name]
            f_blend = pyLIMA_parameters['fblend_' + ref_tel.name]

            # model_magnification = (model['photometry']-f_blend)/f_source

            ref_names.append(ref_tel.name)
            ref_locations.append(ref_tel.location)
            ref_magnification.append(model_magnification)
            ref_fluxes.append([f_source, f_blend])
    lc = {}
    for ind, tel in enumerate(microlensing_model.event.telescopes):

        if tel.lightcurve is not None:

            if tel.location == 'Earth':

                ref_index = np.where(np.array(ref_locations) == 'Earth')[0][0]

            else:

                ref_index = np.where(np.array(ref_names) == tel.name)[0][0]

            residus_in_mag = \
                pyLIMA.fits.objective_functions.photometric_residuals_in_magnitude(
                    tel, microlensing_model,
                    pyLIMA_parameters)
            if ind == 0:
                reference_source = ref_fluxes[ind][0]
                reference_blend = ref_fluxes[ind][1]
                index += 1

            # time_mask = [False for i in range(len(ref_magnification[ref_index]))]
            time_mask = []

            for time in tel.lightcurve['time'].value:
                time_index = np.where(list_of_telescopes[ref_index].lightcurve[
                                          'time'].value == time)[0][0]
                time_mask.append(time_index)

            # model_flux = ref_fluxes[ref_index][0] * ref_magnification[ref_index][
            #    time_mask] + ref_fluxes[ref_index][1]
            model_flux = reference_source * ref_magnification[ref_index][
                time_mask] + reference_blend
            magnitude = pyLIMA.toolbox.brightness_transformation.ZERO_POINT - 2.5 * \
                        np.log10(model_flux)

            ind_color = ind%len(MARKERS_COLORS.by_key()["color"])
            color = MARKERS_COLORS.by_key()["color"][ind_color]
            marker = str(MARKER_SYMBOLS[0][ind])

            lc[tel.name] = magnitude + residus_in_mag
            
            if bokeh_plot is not None:

                bokeh_plot.scatter(tel.lightcurve['time'].value,
                                   magnitude + residus_in_mag,
                                   color=color,
                                   size=5, legend_label=tel.name,
                                   muted_color=color,
                                   muted_alpha=0.2)

                err_xs = []
                err_ys = []

                for x, y, yerr in zip(tel.lightcurve['time'].value,
                                      magnitude + residus_in_mag,
                                      tel.lightcurve['err_mag'].value):
                    err_xs.append((x, x))
                    err_ys.append((y - yerr, y + yerr))

                bokeh_plot.multi_line(err_xs, err_ys, color=color,
                                      legend_label=tel.name,
                                      muted_color=color,
                                      muted_alpha=0.2)
            else:
                plots.plot_light_curve_magnitude(tel.lightcurve['time'].value,
                                 magnitude + residus_in_mag,
                                 tel.lightcurve['err_mag'].value,
                                 figure_axe=figure_axe, color=color,
                                 marker=marker, name=tel.name)
    return lc


def plot_residuals(figure_axe, microlensing_model, model_parameters, bokeh_plot=None,
                   plot_unit='Mag'):
    
    pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(model_parameters)

    # plot residuals

    for ind, tel in enumerate(microlensing_model.event.telescopes):

        if tel.lightcurve is not None:
            residus_in_mag = \
                pyLIMA.fits.objective_functions.photometric_residuals_in_magnitude(
                    tel, microlensing_model, pyLIMA_parameters)

            ind_color = ind % len(MARKERS_COLORS.by_key()["color"])
            color = MARKERS_COLORS.by_key()["color"][ind_color]
            marker = str(MARKER_SYMBOLS[0][ind])
            if bokeh_plot is None:
                plots.plot_light_curve_magnitude(tel.lightcurve['time'].value,
                                                 residus_in_mag,
                                                 tel.lightcurve['err_mag'].value,
                                                 figure_axe=figure_axe, color=color,
                                                 marker=marker, name=tel.name)

        if bokeh_plot is not None:

            bokeh_plot.scatter(tel.lightcurve['time'].value,
                               residus_in_mag,
                               color=color,
                               size=5,
                               muted_color=color,
                               muted_alpha=0.2)

            err_xs = []
            err_ys = []

            for x, y, yerr in zip(tel.lightcurve['time'].value,
                                  residus_in_mag,
                                  tel.lightcurve['err_mag'].value):
                err_xs.append((x, x))
                err_ys.append((y - yerr, y + yerr))

            bokeh_plot.multi_line(err_xs, err_ys, color=color,
                                  muted_color=color,
                                  muted_alpha=0.2)

    # figure_axe.set_ylim([-0.1, 0.1])

