import sys

import cycler
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyLIMA.fits.objective_functions
import pygtc
from bokeh.layouts import gridplot
from bokeh.models import Arrow, OpenHead
from bokeh.models import BasicTickFormatter
from bokeh.plotting import figure
from matplotlib.ticker import MaxNLocator
from pyLIMA.astrometry import astrometric_positions
from pyLIMA.parallax import parallax
from pyLIMA.toolbox import fake_telescopes, plots

plot_lightcurve_windows = 0.2
plot_residuals_windows = 0.21
MAX_PLOT_TICKS = 2
MARKER_SYMBOLS = np.array(
    [['o', '.', '*', 'v', '^', '<', '>', 's', 'p', 'd', 'x'] * 10])

# this is a pointer to the module object instance itself.
thismodule = sys.modules[__name__]

thismodule.list_of_fake_telescopes = []
thismodule.saved_model = None


def create_telescopes_to_plot_model(microlensing_model, pyLIMA_parameters):
    if microlensing_model == thismodule.saved_model:

        list_of_fake_telescopes = thismodule.list_of_fake_telescopes

    else:

        list_of_fake_telescopes = []

    if len(list_of_fake_telescopes) == 0:

        # Photometry first
        Earth = True

        for tel in microlensing_model.event.telescopes:

            if tel.lightcurve_flux is not None:

                if tel.location == 'Space':
                    model_time = np.arange(
                        np.min(tel.lightcurve_magnitude['time'].value),
                        np.max(tel.lightcurve_magnitude['time'].value),
                        0.1).round(2)

                    model_time = np.r_[
                        model_time, tel.lightcurve_magnitude['time'].value]

                    model_time.sort()

                if Earth and tel.location == 'Earth':

                    model_time1 = np.arange(np.min((np.min(
                        tel.lightcurve_magnitude['time'].value),
                                                    pyLIMA_parameters.t0 - 5 *
                                                    pyLIMA_parameters.tE)),
                        np.max((np.max(
                            tel.lightcurve_magnitude['time'].value),
                                pyLIMA_parameters.t0 + 5 *
                                pyLIMA_parameters.tE)),
                        0.05).round(2)

                    model_time2 = np.arange(
                        pyLIMA_parameters.t0 - 1 * pyLIMA_parameters.tE,
                        pyLIMA_parameters.t0 + 1 * pyLIMA_parameters.tE,
                        0.05).round(2)

                    model_time = np.r_[model_time1, model_time2]

                    for telescope in microlensing_model.event.telescopes:

                        if telescope.location == 'Earth':
                            model_time = np.r_[
                                model_time, telescope.lightcurve_magnitude[
                                    'time'].value]

                            symmetric = 2 * pyLIMA_parameters.t0 - \
                                        telescope.lightcurve_magnitude['time'].value
                            model_time = np.r_[model_time, symmetric]

                    model_time.sort()

                if (tel.location == 'Space') | (Earth and tel.location == 'Earth'):

                    model_time = np.unique(model_time)

                    model_lightcurve = np.c_[
                        model_time, [0] * len(model_time), [0.1] * len(model_time)]
                    model_telescope = fake_telescopes.create_a_fake_telescope(
                        light_curve=model_lightcurve)

                    model_telescope.name = tel.name
                    model_telescope.filter = tel.filter
                    model_telescope.location = tel.location
                    model_telescope.ld_gamma = tel.ld_gamma
                    model_telescope.ld_sigma = tel.ld_sigma
                    model_telescope.ld_a1 = tel.ld_a1
                    model_telescope.ld_a2 = tel.ld_a2
                    model_telescope.location = tel.location

                    if tel.location == 'Space':
                        model_telescope.spacecraft_name = tel.spacecraft_name
                        model_telescope.spacecraft_positions = tel.spacecraft_positions

                    if microlensing_model.parallax_model[0] != 'None':
                        model_telescope.initialize_positions()

                        model_telescope.compute_parallax(
                            microlensing_model.parallax_model,
                            microlensing_model.event.North
                            ,
                            microlensing_model.event.East)  # ,
                        # microlensing_model.event.ra/180*np.pi)

                    list_of_fake_telescopes.append(model_telescope)

                    if tel.location == 'Earth' and Earth:
                        Earth = False

        # Astrometry

        for tel in microlensing_model.event.telescopes:

            if tel.astrometry is not None:

                if tel.location == 'Space':

                    model_time = np.arange(np.min(tel.astrometry['time'].value),
                                           np.max(tel.astrometry['time'].value),
                                           0.1).round(2)
                else:

                    model_time1 = np.arange(
                        np.min((np.min(tel.lightcurve_magnitude['time'].value),
                                pyLIMA_parameters.t0 - 5 * pyLIMA_parameters.tE)),
                        np.max((np.max(tel.lightcurve_magnitude['time'].value),
                                pyLIMA_parameters.t0 + 5 * pyLIMA_parameters.tE)),
                        1).round(2)

                    model_time2 = np.arange(
                        pyLIMA_parameters.t0 - 1 * pyLIMA_parameters.tE,
                        pyLIMA_parameters.t0 + 1 * pyLIMA_parameters.tE,
                        0.1).round(2)

                    model_time = np.r_[model_time1, model_time2]

                    model_time = np.r_[model_time, telescope.astrometry['time'].value]

                    symmetric = 2 * pyLIMA_parameters.t0 - telescope.astrometry[
                        'time'].value
                    model_time = np.r_[model_time, symmetric]
                    model_time.sort()

                model_time = np.unique(model_time)
                model_astrometry = np.c_[
                    model_time, [0] * len(model_time), [0.1] * len(model_time), [
                        0] * len(model_time), [0.1] * len(model_time)]
                model_telescope = fake_telescopes.create_a_fake_telescope(
                    astrometry_curve=model_astrometry,
                    astrometry_unit=tel.astrometry['ra'].unit)

                model_telescope.name = tel.name
                model_telescope.filter = tel.filter
                model_telescope.location = tel.location
                model_telescope.ld_gamma = tel.ld_gamma
                model_telescope.ld_sigma = tel.ld_sigma
                model_telescope.ld_a1 = tel.ld_a1
                model_telescope.ld_a2 = tel.ld_a2
                model_telescope.pixel_scale = tel.pixel_scale

                if tel.location == 'Space':
                    model_telescope.spacecraft_name = tel.spacecraft_name
                    model_telescope.spacecraft_positions = tel.spacecraft_positions

                if microlensing_model.parallax_model[0] != 'None':
                    model_telescope.initialize_positions()

                    model_telescope.compute_parallax(microlensing_model.parallax_model,
                                                     microlensing_model.event.North
                                                     ,
                                                     microlensing_model.event.East)  # ,
                    # microlensing_model.event.ra / 180)# * np.pi)

                list_of_fake_telescopes.append(model_telescope)

        thismodule.saved_model = microlensing_model
        thismodule.list_of_fake_telescopes = list_of_fake_telescopes

    return list_of_fake_telescopes

def plot_light_curve_magnitude(time, mag, mag_error=None, figure_axe=None, color=None,
                               linestyle='-', marker=None, name=None):
    """
    Plot a lightcurve in magnitude

    Parameters
    ----------
    time : array, the time to plot
    mag : array, the magnitude to plot
    mag_error : array, the magnitude error
    figure_axe : matplotlib.axe, an axe to plot
    color : str, a color string
    linestyle : str, the matplotlib linestyle desired
    marker : str, the matplotlib marker
    name : str, the points name
    """
    if figure_axe:

        pass

    else:

        figure, figure_axe = plt.subplots()

    if mag_error is None:

        figure_axe.plot(time, mag, c=color, label=name, linestyle=linestyle,lw=2)

    else:

        figure_axe.errorbar(time, mag, mag_error, color=color, marker=marker,
                            label=name, linestyle='',alpha=0.5)


def plot_LCmodel(microlensing_model, model_parameters,mat_figure, mat_figure_axes, bokeh_plot=None):
    # Change matplotlib default colors
    n_telescopes = len(microlensing_model.event.telescopes)

    # mat_figure, mat_figure_axes = initialize_light_curves_plot(
        # event_name=microlensing_model.event.name)


    bokeh_lightcurves = None
    bokeh_residuals = None

    if len(model_parameters) != len(microlensing_model.model_dictionnary):
        telescopes_fluxes = microlensing_model.find_telescopes_fluxes(model_parameters)
        telescopes_fluxes = [getattr(telescopes_fluxes, key) for key in
                             telescopes_fluxes._fields]

        model_parameters = np.r_[model_parameters, telescopes_fluxes]

    dict_save = plot_aligned_data(mat_figure_axes, microlensing_model, model_parameters,
                      plot_unit='Mag',
                      bokeh_plot=bokeh_lightcurves)

    dict_model = plot_photometric_models(mat_figure_axes, microlensing_model, model_parameters,'darkblue',
                            plot_unit='Mag',
                            bokeh_plot=bokeh_lightcurves)

    #mat_figure_axes.set_title(microlensing_model.event.name)
    #mat_figure_axes.invert_yaxis()
    # mat_figure_axes.set_xlim(model_parameters[0]-5*model_parameters[2],model_parameters[0]+5*model_parameters[2])
    # mat_figure_axes.axvspan(model_parameters[0]-model_parameters[2],model_parameters[0]+model_parameters[2],color='blue',alpha=0.35)

    # mat_figure_axes.set_xlabel(r'$JD$', fontsize=20)
    # mat_figure_axes.set_ylabel(r'MAG', fontsize=20)
    # mat_figure_axes.legend(shadow=True, fontsize='large',
    #                           bbox_to_anchor=(0, 1.02, 1, 0.2),
    #                           loc="lower left",
    #                           mode="expand", borderaxespad=0, ncol=3)

    try:
        bokeh_lightcurves.legend.click_policy = "mute"
        # legend = bokeh_lightcurves.legend[0]

    except AttributeError:

        pass

    figure_bokeh = gridplot([[bokeh_lightcurves], [bokeh_residuals]],
                            toolbar_location=None)

    return mat_figure, dict_save, dict_model

def plot_only_model(microlensing_model, model_parameters,mat_figure, mat_figure_axes, bokeh_plot=None):
    # Change matplotlib default colors
    n_telescopes = len(microlensing_model.event.telescopes)

    # mat_figure, mat_figure_axes = initialize_light_curves_plot(
        # event_name=microlensing_model.event.name)


    bokeh_lightcurves = None
    bokeh_residuals = None

    if len(model_parameters) != len(microlensing_model.model_dictionnary):
        telescopes_fluxes = microlensing_model.find_telescopes_fluxes(model_parameters)
        telescopes_fluxes = [getattr(telescopes_fluxes, key) for key in
                             telescopes_fluxes._fields]

        model_parameters = np.r_[model_parameters, telescopes_fluxes]


    plot_photometric_models(mat_figure_axes, microlensing_model, model_parameters,'red',
                            plot_unit='Mag',
                            bokeh_plot=bokeh_lightcurves)

                      
    #mat_figure_axes.set_title(microlensing_model.event.name)
    #mat_figure_axes.invert_yaxis()
    # mat_figure_axes.set_xlim(model_parameters[0]-5*model_parameters[2],model_parameters[0]+5*model_parameters[2])
    # mat_figure_axes.axvspan(model_parameters[0]-model_parameters[2],model_parameters[0]+model_parameters[2],color='blue',alpha=0.35)

    #mat_figure_axes.set_xlabel(r'$JD$', fontsize=20)
    #mat_figure_axes.set_ylabel(r'MAG', fontsize=20)
    # mat_figure_axes.legend(shadow=True, fontsize='large',
    #                           bbox_to_anchor=(0, 1.02, 1, 0.2),
    #                           loc="lower left",
    #                           mode="expand", borderaxespad=0, ncol=3)

    try:
        bokeh_lightcurves.legend.click_policy = "mute"
        # legend = bokeh_lightcurves.legend[0]

    except AttributeError:

        pass

    figure_bokeh = gridplot([[bokeh_lightcurves], [bokeh_residuals]],
                            toolbar_location=None)

    return mat_figure, figure_bokeh


def plot_true_model(microlensing_model, model_parameters, mat_figure, mat_figure_axes, bokeh_plot=None):
    # Change matplotlib default colors
    n_telescopes = len(microlensing_model.event.telescopes)

    # mat_figure, mat_figure_axes = initialize_light_curves_plot(
    # event_name=microlensing_model.event.name)

    bokeh_lightcurves = None
    bokeh_residuals = None

    if len(model_parameters) != len(microlensing_model.model_dictionnary):
        telescopes_fluxes = microlensing_model.find_telescopes_fluxes(model_parameters)
        telescopes_fluxes = [getattr(telescopes_fluxes, key) for key in
                             telescopes_fluxes._fields]

        model_parameters = np.r_[model_parameters, telescopes_fluxes]

    plot_photometric_models(mat_figure_axes, microlensing_model, model_parameters, 'black',
                            plot_unit='Mag',
                            bokeh_plot=bokeh_lightcurves)

    mat_figure_axes.set_title(microlensing_model.event.name)
    mat_figure_axes.invert_yaxis()
    mat_figure_axes.set_xlim(model_parameters[0] - 5 * model_parameters[2],
                             model_parameters[0] + 5 * model_parameters[2])
    # mat_figure_axes.axvspan(model_parameters[0]-model_parameters[2],model_parameters[0]+model_parameters[2],color='blue',alpha=0.35)

    mat_figure_axes.set_xlabel(r'$JD$', fontsize=20)
    mat_figure_axes.set_ylabel(r'MAG', fontsize=20)
    # mat_figure_axes.legend(shadow=True, fontsize='large',
    #                           bbox_to_anchor=(0, 1.02, 1, 0.2),
    #                           loc="lower left",
    #                           mode="expand", borderaxespad=0, ncol=3)

    try:
        bokeh_lightcurves.legend.click_policy = "mute"
        # legend = bokeh_lightcurves.legend[0]

    except AttributeError:

        pass

    figure_bokeh = gridplot([[bokeh_lightcurves], [bokeh_residuals]],
                            toolbar_location=None)

    return mat_figure, figure_bokeh


# def initialize_light_curves_plot(plot_unit='Mag', event_name='A microlensing event'):
#     fig_size = [10, 10]

#     mat_figure, mat_figure_axes = plt.subplots(1, 1, figsize=(fig_size[0], fig_size[1]), dpi=75)
#     plt.subplots_adjust(top=0.8, bottom=0.15, left=0.2, right=0.9, wspace=0.1,
#                         hspace=0.1)
#     mat_figure_axes.grid()

#     mat_figure_axes.set_ylabel(r'$' + plot_unit + '$',
#                                   fontsize=5 * fig_size[1] * 3 / 4.0)
#     mat_figure_axes.yaxis.set_major_locator(MaxNLocator(4))
#     mat_figure_axes.tick_params(axis='y', labelsize=3.5 * fig_size[1] * 3 / 4.0)
#     mat_figure_axes.tick_params(axis='x', labelsize=3.5 * fig_size[1] * 3 / 4.0)

#     # mat_figure_axes.text(0.01, 0.96, 'provided by pyLIMA', style='italic',
#     #                         fontsize=10,
#     #                         transform=mat_figure_axes.transAxes)

#     mat_figure_axes.set_xlabel(r'$JD$', fontsize=20)


#     return mat_figure, mat_figure_axes


def plot_photometric_models(figure_axe, microlensing_model, model_parameters,Color,
                            bokeh_plot=None, plot_unit='Mag'):
    pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(model_parameters)

    list_of_telescopes = create_telescopes_to_plot_model(microlensing_model,
                                                         pyLIMA_parameters)
    telescopes_names = np.array([i.name for i in microlensing_model.event.telescopes])

    # plot models
    index = 0
    dict_save = {}
    for tel in list_of_telescopes:

        if tel.lightcurve_flux is not None:

            magni = microlensing_model.model_magnification(tel, pyLIMA_parameters)
            microlensing_model.derive_telescope_flux(tel, pyLIMA_parameters, magni)

            f_source = getattr(pyLIMA_parameters, 'fsource_' + tel.name)
            f_blend = getattr(pyLIMA_parameters, 'fblend_' + tel.name)

            if index == 0:
                ref_source = f_source
                ref_blend = f_blend
                index += 1

            magnitude = pyLIMA.toolbox.brightness_transformation.ZERO_POINT - 2.5 * \
                        np.log10(ref_source * magni + ref_blend)


            name = tel.name

            index_color = np.where(name == telescopes_names)[0][0]
            # color = plt.rcParams["axes.prop_cycle"].by_key()["color"][index_color]

            if tel.location == 'Earth':

                name = tel.location
                linestyle = '-'

            else:

                linestyle = '-'
            # print(tel.lightcurve_magnitude['time'].value)
            # plt.plot(tel.lightcurve_magnitude['time'].value, np.ones(tel.lightcurve_magnitude['time'].value), marker='o',linestlye='')
            plot_light_curve_magnitude(tel.lightcurve_magnitude['time'].value,
                                             magnitude, figure_axe=figure_axe,
                                             name=name, color=Color,
                                             linestyle=linestyle)
            dict_save[tel.name]=[tel.lightcurve_magnitude['time'].value,
                                             magnitude]
    return dict_save



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
        model_magnification = microlensing_model.model_magnification(ref_tel,
                                                                     pyLIMA_parameters)

        microlensing_model.derive_telescope_flux(ref_tel, pyLIMA_parameters,
                                                 model_magnification)

        f_source = getattr(pyLIMA_parameters, 'fsource_' + ref_tel.name)
        f_blend = getattr(pyLIMA_parameters, 'fblend_' + ref_tel.name)

        # model_magnification = (model['photometry']-f_blend)/f_source

        ref_names.append(ref_tel.name)
        ref_locations.append(ref_tel.location)
        ref_magnification.append(model_magnification)
        ref_fluxes.append([f_source, f_blend])

    dict_save = {}
    for ind, tel in enumerate(microlensing_model.event.telescopes):

        if tel.lightcurve_flux is not None:

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
            for time in tel.lightcurve_flux['time'].value:
                time_index = np.where(list_of_telescopes[ref_index].lightcurve_flux[
                                          'time'].value == time)[0][0]
                time_mask.append(time_index)

            # model_flux = ref_fluxes[ref_index][0] * ref_magnification[ref_index][
            #    time_mask] + ref_fluxes[ref_index][1]
            model_flux = reference_source * ref_magnification[ref_index][
                time_mask] + reference_blend
            magnitude = pyLIMA.toolbox.brightness_transformation.ZERO_POINT - 2.5 * \
                        np.log10(model_flux)
            
            color = {'F146':'b', 'LSST_u':'c', 'LSST_g':'g', 'LSST_r':'y', 'LSST_i':'r', 'LSST_z':'m', 'LSST_y':'k'}
#             color = plt.rcParams["axes.prop_cycle"].by_key()["color"][ind]
            
            marker = 'D'

            plot_light_curve_magnitude(tel.lightcurve_magnitude['time'].value,
                                             magnitude + residus_in_mag,
                                             tel.lightcurve_magnitude['err_mag'].value,
                                             figure_axe=figure_axe, color=color[tel.name],
                                             marker=marker, name=tel.name)

            dict_save[tel.name]=[tel.lightcurve_magnitude['time'].value,
                                             magnitude + residus_in_mag,tel.lightcurve_magnitude['err_mag'].value]
    return dict_save
