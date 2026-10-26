"""Lens geometry and astrometry plots (from pyLIMA's pyLIMA_plots, modified copy).

plot_geometry: source trajectory, caustics and lens positions.
plot_astrometry / plot_astrometric_models / plot_astrometric_data.
Model telescopes and colors come from plotting.pyLIMA_plots.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import Arrow
from bokeh.models import BasicTickFormatter
from bokeh.models import OpenHead
from bokeh.plotting import figure
from matplotlib.ticker import MaxNLocator
from pyLIMA.astrometry import astrometric_positions
from pyLIMA.parallax import parallax
from plotting.pyLIMA_plots import (
    MARKERS_COLORS,
    create_telescopes_to_plot_model,
)


def plot_geometry(microlensing_model, model_parameters, bokeh_plot=None):
    """Plot the lensing geometry (i.e source trajectory) and the table of best
    parameters.
    :param object fit: a fit object. See the microlfits for more details.
    :param list best_parameters: a list containing the model you want to plot the
    trajectory
    """


    pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(model_parameters)

    faketelescopes = create_telescopes_to_plot_model(microlensing_model,
                                                     pyLIMA_parameters)

    fig_size = [10, 10]
    figure_trajectory = plt.figure(figsize=(fig_size[0], fig_size[1]), dpi=70)

    figure_axes = figure_trajectory.add_subplot(111, aspect=1)
    figure_axes.set_aspect('equal', adjustable='box')
    plt.subplots_adjust(top=0.8, bottom=0.1, left=0.2, right=0.9, wspace=0.1,
                        hspace=0.1)
    if bokeh_plot is not None:

        bokeh_geometry = figure(width=600, height=600, x_range=(-3, 3), y_range=(-3, 3),
                                x_axis_label=r'$$x [\theta_E]$$',
                                y_axis_label=r'$$y [\theta_E]$$')

    else:

        bokeh_geometry = None

    for telescope in faketelescopes:

        if telescope.lightcurve is not None:

            platform = 'Earth'

            if telescope.location == 'Space':

                platform = telescope.name
                linestyle = '--'

            else:

                linestyle = '-'

            #reference_telescope = telescope

            telescope_index = \
                [i for i in range(len(microlensing_model.event.telescopes)) if
                 microlensing_model.event.telescopes[i].name == telescope.name][0]


            (source1_trajectory_x, source1_trajectory_y,
             source2_trajectory_x, source2_trajectory_y,
             dseparation, dalpha) = microlensing_model.sources_trajectory(
                telescope, pyLIMA_parameters,
                data_type='photometry')

            ind_color = telescope_index % len(MARKERS_COLORS.by_key()["color"])
            color = MARKERS_COLORS.by_key()["color"][ind_color]
            figure_axes.plot(source1_trajectory_x, source1_trajectory_y,
                             c=color,
                             label=platform, linestyle=linestyle)

            if bokeh_geometry is not None:

                bokeh_geometry.line(source1_trajectory_x,
                                    source1_trajectory_y,
                                    color=color,
                                    legend_label=platform)

            if source2_trajectory_y is not None:

                figure_axes.plot(source2_trajectory_x, source2_trajectory_y,
                                 c=color, alpha=0.5,
                                 linestyle=linestyle)

                if bokeh_geometry is not None:

                    bokeh_geometry.line(source2_trajectory_x,
                                    source2_trajectory_y,
                                    color=color, alpha=0.5)



            for trajectory in [[source1_trajectory_x, source1_trajectory_y],
                         [source2_trajectory_x, source2_trajectory_y] ]:

                trajectory_x,trajectory_y=trajectory

                if trajectory_x is not None:

                    for ind in [-2, -1, 0, 1, 2]:

                        try:

                            index = np.argmin(
                                np.abs(telescope.lightcurve['time'].value -
                                       (pyLIMA_parameters['t0'] + ind * pyLIMA_parameters['tE'])))
                            sign = np.sign(trajectory_x[index + 1] - trajectory_x[index])

                            delta_x = trajectory_x[index - 1] - trajectory_x[index + 1]
                            if abs(delta_x) < 1e-10:  # Avoid division by zero
                                derivative = 0
                            else:
                                derivative = (trajectory_y[index - 1] - trajectory_y[index + 1]) / delta_x
                            # derivative = (trajectory_y[index - 1] - trajectory_y[index + 1]) / (
                            #         trajectory_x[index - 1] - trajectory_x[index + 1])

                            figure_axes.annotate('',
                                                 xy=(trajectory_x[index], trajectory_y[index]),
                                                 xytext=(trajectory_x[index] - (
                                                         trajectory_x[index + 1] -
                                                         trajectory_x[index]) * 0.001,
                                                         trajectory_y[index] - (
                                                                 trajectory_x[index + 1] -
                                                                 trajectory_x[
                                                                     index]) * 0.001 *
                                                         derivative),
                                                 arrowprops=dict(arrowstyle="->",
                                                                 mutation_scale=35,
                                                                 color=color))

                            if bokeh_geometry is not None:
                                oh = OpenHead(line_color=color, line_width=1)

                                bokeh_geometry.add_layout(Arrow(end=oh,
                                                                x_start=trajectory_x[index],
                                                                y_start=trajectory_y[index],
                                                                x_end=trajectory_x[
                                                                          index] + sign * 0.001,
                                                                y_end=trajectory_y[
                                                                          index] + sign *
                                                                      0.001 * derivative))

                        except IndexError:

                            pass

    if microlensing_model.parallax_model[0] != 'None':

        telescope = faketelescopes[0]
        try:
            origin_t0par_index = np.argmin(
                np.abs(telescope.lightcurve['time'].value -
                       microlensing_model.parallax_model[1]))
            (source1_trajectory_x, source1_trajectory_y,
             source2_trajectory_x, source2_trajectory_y,
             dseparation, dalpha) = microlensing_model.sources_trajectory(
                telescope, pyLIMA_parameters,
                data_type='photometry')

        except AttributeError:
            origin_t0par_index = np.argmin(
                np.abs(telescope.astrometry['time'].value -
                       microlensing_model.parallax_model[1]))
            (source1_trajectory_x, source1_trajectory_y,
             source2_trajectory_x, source2_trajectory_y,
             dseparation, dalpha) = microlensing_model.sources_trajectory(
                telescope, pyLIMA_parameters,
                data_type='astrometry')
        # print(telescope.lightcurve['time'].value,
        #      microlensing_model.parallax_model[1],
        #      origin_t0par_index)

        # print(trajectory_x, trajectory_y)
        # import pdb;
        # pdb.set_trace()
        origin_t0par = np.array(
            (source1_trajectory_x[origin_t0par_index],
             source1_trajectory_y[origin_t0par_index]))
        # origin_t0par += 0.1

        piEN = pyLIMA_parameters['piEN']
        piEE = pyLIMA_parameters['piEE']

        EN_trajectory_angle = parallax.EN_trajectory_angle(piEN, piEE)

        plot_angle = -EN_trajectory_angle

        try:

            plot_angle += pyLIMA_parameters['alpha']

        except KeyError:

            pass

        north = [0.1, 0]
        east = [0, 0.1]

        North = [0.105, 0]
        East = [0, 0.105]

        rota_mat = np.array([[np.cos(plot_angle), -np.sin(plot_angle)],
                             [np.sin(plot_angle), np.cos(plot_angle)]])

        east = np.dot(rota_mat, east)
        north = np.dot(rota_mat, north)
        East = np.dot(rota_mat, East)
        North = np.dot(rota_mat, North)

        plt.annotate('',
                     xy=(origin_t0par[0] + east[0], origin_t0par[1] + east[1]),
                     xytext=(origin_t0par[0], origin_t0par[1]),
                     arrowprops=dict(arrowstyle="->", lw=3, alpha=0.5))
        plt.annotate('E', xy=(origin_t0par[0] + East[0], origin_t0par[1] + East[1]),
                     xytext=(origin_t0par[0] + East[0], origin_t0par[1] + East[1]),
                     weight='bold', alpha=0.5, ha='center', va='center',
                     rotation=np.rad2deg(plot_angle))

        plt.annotate('', xy=(
            origin_t0par[0] + north[0], origin_t0par[1] + north[1]),
                     xytext=(origin_t0par[0], origin_t0par[1]),
                     arrowprops=dict(arrowstyle="->", lw=3, alpha=0.5))
        plt.annotate('N',
                     xy=(origin_t0par[0] + North[0], origin_t0par[1] + North[1]),
                     xytext=(
                         origin_t0par[0] + North[0], origin_t0par[1] + North[1]),
                     weight='bold', alpha=0.5, ha='center', va='center',
                     rotation=np.rad2deg(plot_angle))

        if bokeh_geometry is not None:
            bokeh_geometry.add_layout(
                Arrow(end=OpenHead(line_color="grey", line_width=1),
                      x_start=origin_t0par[0], y_start=origin_t0par[1],
                      x_end=origin_t0par[0] + North[0],
                      y_end=origin_t0par[1] + North[1]))
            bokeh_geometry.add_layout(
                Arrow(end=OpenHead(line_color="grey", line_width=1),
                      x_start=origin_t0par[0], y_start=origin_t0par[1],
                      x_end=origin_t0par[0] + East[0],
                      y_end=origin_t0par[1] + East[1]))

    if 'BL' in microlensing_model.model_type():

        from pyLIMA.caustics import binary_caustics

        regime, caustics, cc = \
            binary_caustics.find_2_lenses_caustics_and_critical_curves(
                pyLIMA_parameters['separation'],
                pyLIMA_parameters['mass_ratio'],
                resolution=5000)

        center_of_mass = (pyLIMA_parameters['separation'] *
                          pyLIMA_parameters['mass_ratio'] / (
                1 + pyLIMA_parameters['mass_ratio']))
        plt.scatter(-center_of_mass, 0, s=10, c='k')
        plt.scatter(-center_of_mass + pyLIMA_parameters['separation'], 0, s=10, c='k')

        for count, caustic in enumerate(caustics):

            try:
                figure_axes.plot(caustic.real, caustic.imag, lw=3, c='r')
                figure_axes.plot(cc[count].real, cc[count].imag, '--k')

                if bokeh_geometry is not None:
                    bokeh_geometry.line(caustic.real, caustic.imag,
                                        color='red', line_width=3)
                    bokeh_geometry.line(cc[count].real, cc[count].imag,
                                        line_dash='dashed',
                                        color='black')

            except AttributeError:

                pass

    else:

        figure_axes.scatter(0, 0, s=10, c='r')

        einstein_ring = plt.Circle((0, 0), 1, fill=False, color='k', linestyle='--', label='Einstein Ring')
        figure_axes.add_artist(einstein_ring)

        if bokeh_geometry is not None:
            bokeh_geometry.scatter(0, 0, color='red')
            bokeh_geometry.circle(0, 0, radius=1, line_dash='dashed',
                                  line_color='black', fill_color=None)

    for telescope_index, telescope in enumerate(microlensing_model.event.telescopes):

        if telescope.lightcurve is not None:

            (source1_trajectory_x, source1_trajectory_y,
             source2_trajectory_x, source2_trajectory_y,
             dseparation, dalpha) = microlensing_model.sources_trajectory(
                telescope, pyLIMA_parameters,
                data_type='photometry')

            for ind, trajectory in enumerate([[source1_trajectory_x,
                                             source1_trajectory_y],
                               [source2_trajectory_x, source2_trajectory_y]]):

                trajectory_x, trajectory_y = trajectory

                if trajectory_x is not None:

                    if 'rho' in microlensing_model.pyLIMA_standards_dictionnary.keys():

                        rho = pyLIMA_parameters['rho']
                        if ind == 1:
                            rho = pyLIMA_parameters['rho_2']
                    else:

                        rho = 10 ** -5

                    ind_color = telescope_index % len(MARKERS_COLORS.by_key()["color"])
                    color = MARKERS_COLORS.by_key()["color"][ind_color]

                    if ind == 1:

                        patches = [plt.Circle((x, y), rho, edgecolor=color,
                                              facecolor='none',
                                              alpha=0.2) for x, y in
                                   zip(trajectory_x, trajectory_y)]
                    else:

                        patches = [plt.Circle((x, y), rho, color=color,
                                              alpha=0.2) for x, y in
                                   zip(trajectory_x, trajectory_y)]

                        figure_axes.scatter(trajectory_x, trajectory_y,
                                        c=color,
                                        alpha=0.5, label=telescope.name, s=0.1)

                    coll = matplotlib.collections.PatchCollection(patches, match_original=True)

                    figure_axes.add_collection(coll)

                    if bokeh_geometry is not None:
                        bokeh_geometry.circle(trajectory_x, trajectory_y, radius=rho,
                                              color=color,
                                              radius_dimension='max', fill_alpha=0.5)


    # legend = figure_axes.legend(numpoints=1, loc='best', fancybox=True,
    # framealpha=0.5)
    legend = figure_axes.legend(shadow=True, fontsize='large',
                                bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                                mode="expand", borderaxespad=0, ncol=3, numpoints=1)

    for handle in legend.legend_handles:

        try:

            handle.set_sizes([100])

        except AttributeError:

            pass

    figure_axes.xaxis.set_major_locator(MaxNLocator(5))
    figure_axes.yaxis.set_major_locator(MaxNLocator(5))
    figure_axes.xaxis.get_major_ticks()[0].draw = lambda *args: None
    figure_axes.yaxis.get_major_ticks()[0].draw = lambda *args: None
    figure_axes.xaxis.get_major_ticks()[-1].draw = lambda *args: None
    figure_axes.yaxis.get_major_ticks()[-1].draw = lambda *args: None

    figure_axes.set_xlabel(r'$x(\theta_E)$', fontsize=25)
    figure_axes.set_ylabel(r'$y(\theta_E)$', fontsize=25)
    figure_axes.tick_params(axis='x', labelsize=15)
    figure_axes.tick_params(axis='y', labelsize=15)

    figure_axes.axis([-5, 5, -5, 5])
    # figure_axes.axis('scaled')
    # title = microlensing_model.event.name + ' : ' + microlensing_model.model_type
    # figure_trajectory.suptitle(title, fontsize=30 * fig_size[0] / len(title))

    return figure_trajectory, bokeh_geometry


def plot_astrometry(microlensing_model, model_parameters, bokeh_plot=None):


    # Set up the geometry of the three plots
    main_plot = [0.12, 0.12, 0.48, 0.48]
    residuals_x = [0.12, 0.83, 0.83, 0.15]
    residuals_y = [0.12, 0.65, 0.83, 0.15]

    # Set up the size of the figure
    fig_size = (9.5, 9.5)
    ast_figure = plt.figure(figsize=fig_size)

    # Make the three plots
    ax_main = ast_figure.add_axes(main_plot)
    ax_res_x = ast_figure.add_axes(residuals_x)
    ax_res_y = ast_figure.add_axes(residuals_y)

    ax_main.xaxis.grid(True)
    ax_main.yaxis.grid(True)
    ax_res_x.xaxis.grid(True)
    ax_res_x.yaxis.grid(True)
    ax_res_y.xaxis.grid(True)
    ax_res_y.yaxis.grid(True)

    ax_res_x.get_shared_x_axes().joined(ax_res_x, ax_res_y)
    # ax_main.get_shared_y_axes().join(ax_main, ax_res_y)
    ax_res_y.xaxis.set_major_locator(MaxNLocator(4))
    ax_res_x.xaxis.set_major_locator(MaxNLocator(4))

    ax_res_y.ticklabel_format(useOffset=False, style='plain')

    unit = 'deg'
    for tel in microlensing_model.event.telescopes:

        if tel.astrometry is not None:
            unit = tel.astrometry['ra'].unit

    if bokeh_plot is not None:

        bokeh_main = figure(width=800, height=800, toolbar_location=None,
                            x_axis_label=r'$$E~[' + str(unit) + ']$$',
                            y_axis_label=r'$$N~[' + str(unit) + ']$$')

        bokeh_res_x = figure(width=bokeh_main.width, height=200,
                             y_range=(-0.1, 0.1), toolbar_location=None,
                             y_axis_label=r'$$\Delta_N$$')
        bokeh_res_y = figure(width=bokeh_main.width, height=200,
                             y_range=(-0.1, 0.1), toolbar_location=None,
                             y_axis_label=r'$$\Delta_E$$')

        bokeh_main.x_range.flipped = True
        bokeh_res_y.xaxis.formatter = BasicTickFormatter(use_scientific=False)
        bokeh_res_x.xaxis.minor_tick_line_color = None
        bokeh_res_x.xaxis.major_tick_line_color = None
        bokeh_res_x.xaxis.major_label_text_font_size = '0pt'

    else:

        bokeh_main = None
        bokeh_res_x = None
        bokeh_res_y = None

    if len(model_parameters) != len(microlensing_model.model_dictionnary):
        telescopes_fluxes = microlensing_model.find_telescopes_fluxes(model_parameters)

        model_parameters = np.r_[model_parameters, telescopes_fluxes]

    plot_astrometric_data(ax_main, microlensing_model, bokeh_plot=bokeh_main)

    plot_astrometric_models([ax_main, ax_res_x, ax_res_y], microlensing_model,
                            model_parameters,
                            bokeh_plots=[bokeh_main, bokeh_res_x, bokeh_res_y])

    ax_main.legend(shadow=True, fontsize='large', bbox_to_anchor=(1.05, 0.25),
                   loc="center left", borderaxespad=0, ncol=2)

    ax_main.invert_xaxis()
    ax_res_x.set_xticklabels([])

    ax_main.set_xlabel(r'$E~[' + str(unit) + ']$', fontsize=4 * fig_size[0] * 3 / 4.0)
    ax_main.set_ylabel(r'$N~[' + str(unit) + ']$', fontsize=4 * fig_size[0] * 3 / 4.0)

    ax_res_y.set_ylabel(r'$\Delta E~[' + str(unit) + ']$',
                        fontsize=4 * fig_size[0] * 3 / 4.0)
    ax_res_y.set_xlabel('$JD$',  x=0.75,fontsize=4 * fig_size[0] * 3 / 4.0)

    ax_res_x.set_ylabel(r'$\Delta N~[' + str(unit) + ']$',
                        fontsize=4 * fig_size[0] * 3 / 4.0)

    figure_bokeh = gridplot([[bokeh_res_x], [bokeh_res_y], [bokeh_main]],
                            toolbar_location='above')

    return ast_figure, figure_bokeh


def plot_astrometric_models(figure_axes, microlensing_model, model_parameters,
                            bokeh_plots=None):
    pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(model_parameters)

    list_of_telescopes = create_telescopes_to_plot_model(microlensing_model,
                                                         pyLIMA_parameters)
    telescopes_names = np.array([i.name for i in microlensing_model.event.telescopes])

    # plot models
    Earth = True

    for tel in list_of_telescopes:

        if tel.astrometry is not None:

            model = microlensing_model.compute_the_microlensing_model(tel,
                                                                      pyLIMA_parameters)

            astrometric_model = model['astrometry']
            lens_E, lens_N = astrometric_positions.lens_astrometric_positions(
                microlensing_model, tel, pyLIMA_parameters)
            name = tel.name

            index_color = np.where(name == telescopes_names)[0][0]
            color = MARKERS_COLORS.by_key()["color"][index_color]

            figure_axes[0].plot(astrometric_model[0], astrometric_model[1], c=color)

            if bokeh_plots[0] is not None:
                bokeh_plots[0].line(astrometric_model[0], astrometric_model[1],
                                    color=color)

            if Earth is True:

                source_E, source_N = \
                    astrometric_positions.astrometric_positions_of_the_source(
                        tel, pyLIMA_parameters)
                figure_axes[0].plot(source_E, source_N, c='k', label='Source')
                figure_axes[0].plot(lens_E, lens_N, c='k', linestyle='--', label='Lens')

                if bokeh_plots[0] is not None:
                    bokeh_plots[0].line(source_E, source_N, color='black',
                                        legend_label='Source')
                    bokeh_plots[0].line(lens_E, lens_N, color='black',
                                        line_dash='dotted', legend_label='Lens')

                for index in [-2, -1, 0, 1, 2]:

                    try:

                        index_time = np.argmin(np.abs(tel.astrometry['time'].value -
                                                      (
                                                              pyLIMA_parameters['t0'] +
                                                              index *
                                                              pyLIMA_parameters['tE'])))
                        derivative = (source_N[index_time - 1] - source_N[
                            index_time + 1]) / (
                                             source_E[index_time - 1] - source_E[
                                         index_time + 1])

                        figure_axes[0].annotate('', xy=(
                            source_E[index_time], source_N[index_time]),
                                                xytext=(source_E[index_time] - 0.001 * (
                                                        source_E[index_time + 1] -
                                                        source_E[index_time]),
                                                        source_N[index_time] - 0.001 * (
                                                                source_E[
                                                                    index_time +
                                                                    1] -
                                                                source_E[
                                                                    index_time])
                                                        * derivative),
                                                arrowprops=dict(arrowstyle="->",
                                                                mutation_scale=35,
                                                                color='k'))

                        index_time = np.argmin(np.abs(tel.astrometry['time'].value -
                                                      (
                                                              pyLIMA_parameters['t0'] +
                                                              index *
                                                              pyLIMA_parameters['tE'])))
                        derivative = (lens_N[index_time - 1] - lens_N[
                            index_time + 1]) / (
                                             lens_E[index_time - 1] - lens_E[
                                         index_time + 1])

                        figure_axes[0].annotate('', xy=(
                            lens_E[index_time], lens_N[index_time]),
                                                xytext=(lens_E[index_time] - 0.001 * (
                                                        lens_E[index_time + 1] -
                                                        lens_E[index_time]),
                                                        lens_N[index_time] - 0.001 * (
                                                                lens_E[
                                                                    index_time +
                                                                    1] -
                                                                lens_E[
                                                                    index_time])
                                                        * derivative),
                                                arrowprops=dict(arrowstyle="->",
                                                                mutation_scale=35,
                                                                color='k'))

                        if bokeh_plots[0] is not None:
                            oh = OpenHead(line_color='black', line_width=1)

                            bokeh_plots[0].add_layout(Arrow(end=oh,
                                                            x_start=source_E[
                                                                index_time],
                                                            y_start=source_N[
                                                                index_time],
                                                            x_end=source_E[
                                                                index_time + 1],
                                                            y_end=source_N[
                                                                index_time + 1]))

                            bokeh_plots[0].add_layout(Arrow(end=oh,
                                                            x_start=lens_E[index_time],
                                                            y_start=lens_N[index_time],
                                                            x_end=lens_E[
                                                                index_time + 1],
                                                            y_end=lens_N[
                                                                index_time + 1]))

                    except ValueError:

                        pass

                Earth = False

    # plot residuals

    for ind, tel in enumerate(microlensing_model.event.telescopes):

        if tel.astrometry is not None:

            delta_ra = tel.astrometry['ra'].value
            err_ra = tel.astrometry['err_ra'].value

            delta_dec = tel.astrometry['dec'].value
            err_dec = tel.astrometry['err_dec'].value

            ind_color = ind % len(MARKERS_COLORS.by_key()["color"])
            color = MARKERS_COLORS.by_key()["color"][ind_color]

            model = microlensing_model.compute_the_microlensing_model(tel,
                                                                      pyLIMA_parameters)

            astrometric_model = model['astrometry']

            figure_axes[1].errorbar(tel.astrometry['time'].value,
                                    delta_ra - astrometric_model[0], yerr=err_ra,
                                    fmt='.', ecolor=color, color=color,
                                    label=tel.name, alpha=0.5)

            figure_axes[2].errorbar(tel.astrometry['time'].value,
                                    delta_dec - astrometric_model[1], yerr=err_dec,
                                    fmt='.', ecolor=color,
                                    color=color,
                                    label=tel.name, alpha=0.5)

            if bokeh_plots[1] is not None:

                res_ra = delta_ra - astrometric_model[0]
                res_dec = delta_dec - astrometric_model[1]

                time = []

                err_xs = []
                err_ys = []

                for t, rex, rey, xerr, yerr in zip(tel.astrometry['time'].value, res_ra,
                                                   res_dec, err_ra, err_dec):
                    time.append((t, t))

                    err_xs.append((rex - xerr, rex + xerr))
                    err_ys.append((rey - yerr, rey + yerr))

                bokeh_plots[1].scatter(tel.astrometry['time'].value,
                                       delta_ra - astrometric_model[0],
                                       color=color,
                                       size=5,
                                       muted_color=color,
                                       muted_alpha=0.2)

                bokeh_plots[1].multi_line(time, err_xs, color=color,
                                          muted_color=color,
                                          muted_alpha=0.2)

                bokeh_plots[2].scatter(tel.astrometry['time'].value,
                                       delta_dec - astrometric_model[1],
                                       color=color,
                                       size=5,
                                       muted_color=color,
                                       muted_alpha=0.2)

                bokeh_plots[2].multi_line(time, err_ys, color=color,
                                          muted_color=color,
                                          muted_alpha=0.2)


def plot_astrometric_data(figure_ax, microlensing_model, bokeh_plot=None):
    # plot data
    for ind, tel in enumerate(microlensing_model.event.telescopes):

        if tel.astrometry is not None:

            delta_ra = tel.astrometry['ra'].value
            err_ra = tel.astrometry['err_ra'].value

            delta_dec = tel.astrometry['dec'].value
            err_dec = tel.astrometry['err_dec'].value

            ind_color = ind % len(MARKERS_COLORS.by_key()["color"])
            color = MARKERS_COLORS.by_key()["color"][ind_color]
            # marker = str(MARKER_SYMBOLS[0][ind])

            figure_ax.errorbar(delta_ra, delta_dec, xerr=err_ra, yerr=err_dec, fmt='.',
                               ecolor=color, color=color,
                               label=tel.name, alpha=0.5)

            if bokeh_plot is not None:

                bokeh_plot.scatter(delta_ra, delta_dec,
                                   color=color,
                                   size=5, legend_label=tel.name,
                                   muted_color=color,
                                   muted_alpha=0.2)

                X = []
                Y = []

                err_xs = []
                err_ys = []

                for x, y, xerr, yerr in zip(delta_ra, delta_dec, err_ra, err_dec):
                    X.append((x, x))
                    Y.append((y, y))

                    err_xs.append((x - xerr, x + xerr))
                    err_ys.append((y - yerr, y + yerr))

                bokeh_plot.multi_line(err_xs, Y, color=color,
                                      muted_color=color,
                                      muted_alpha=0.2)

                bokeh_plot.multi_line(X, err_ys, color=color,
                                      muted_color=color,
                                      muted_alpha=0.2)
