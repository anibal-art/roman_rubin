"""Posterior / sample plots (from pyLIMA's pyLIMA_plots, modified copy).

plot_distribution (corner-like plot with plot_2d_sigmas) and
plot_parameters_table.
"""

import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from bokeh.models import ColumnDataSource
from bokeh.models import DataTable
from bokeh.models import TableColumn
from bokeh.plotting import figure


def plot_distribution(samples, parameters_names=None, bokeh_plot=None):

    names = [str(i) for i in range(len(parameters_names))]


    fig_size = [10, 10]
    mat_figure, mat_figure_axes = plt.subplots(len(names), len(names),
                                               figsize=(fig_size[0], fig_size[1]),
                                               sharex='col',dpi=75)

    pos_text = mat_figure_axes[0,-2].get_position()
    eps = 0.1

    for ii in range(len(names)):
        params2 = samples[:,ii]-np.median(samples[:,ii])

        for jj in range(len(names)):

            if jj>ii:

                mat_figure.delaxes(mat_figure_axes[ii,jj])

            if jj==ii:

                mat_figure_axes[ii, jj].hist(params2,bins=50, density=True,
                                             range = [params2.min() - eps*np.abs(params2.min()),
                                                      params2.max() + eps*np.abs(params2.max())])

            if jj<ii:

                params1 = samples[:, jj]-np.median(samples[:,jj])


                plot_2d_sigmas(mat_figure_axes[ii,jj], params1, params2, bins=50, eps=eps,
                               x_range=None,
                               y_range=None)

            if jj==0:

                mat_figure_axes[ii, jj].set_ylabel(names[ii])

            if ii == len(names)-1:

                mat_figure_axes[ii, jj].set_xlabel(names[jj])

            if (jj>0) & (ii>-1):

                mat_figure_axes[ii,jj].set_yticks([])

    text = [names[i] + ' : ' + parameters_names[i] + ' - '+f'{np.median(samples[:,i]):.1}'+'\n' for i in
              range(len(parameters_names))]

    ax_text = mat_figure.add_axes(pos_text)
    ax_text.text(0.25,0.25,''.join(text), verticalalignment='top',size=15)
    ax_text.xaxis.set_visible(False)
    ax_text.yaxis.set_visible(False)  # Hide only x axis
    ax_text.spines['right'].set_visible(False)
    ax_text.spines['top'].set_visible(False)
    ax_text.spines['left'].set_visible(False)
    ax_text.spines['bottom'].set_visible(False)


    if bokeh_plot is not None:

        buf = io.BytesIO()
        mat_figure.savefig(buf, format='png', dpi=300)
        buf.seek(0)

        # Load the image into PIL, convert to RGBA and then to a numpy array
        img = Image.open(buf)
        img_array = np.array(img.convert('RGBA'))
        img_array_flipped = np.flipud(img_array)  # Flip the image vertically
        height, width, _ = img_array_flipped.shape
        dw = 14  # Width of the display area; adjust as needed
        dh = dw * (height / width)  # Calculate height to maintain aspect ratio

        # Update figure creation and image display
        p = figure(x_range=(0, dw), y_range=(0, dh))
        p.xaxis.visible = False
        p.yaxis.visible = False
        img_data = img_array_flipped.view(dtype=np.uint32).reshape((height, width))
        source = ColumnDataSource({'image': [img_data]})
        p.image_rgba(image='image', x=0, y=0, dw=dw, dh=dh, source=source)
        # Update the bokeh_plot variable with the new figure
        bokeh_plot = p
        buf.close()
        #pass
    return mat_figure, bokeh_plot


def plot_parameters_table(samples, parameters_names=None, chi2=None, bokeh_plot=None):
    # Calculate percentiles
    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    table_val = [f"{percentiles[1][i]:.2f} (+{percentiles[2][i] - percentiles[1][i]:.2f}, -{percentiles[1][i] - percentiles[0][i]:.2f})"
                 for i in range(percentiles.shape[1])]
    
    # Add chi2 if provided
    if chi2 is not None:
        table_val.append(f"{chi2:.2f}")
        parameters_names.append('chi2')
    
    # Create Bokeh table
    bokeh_table = None
    if bokeh_plot is not None:
        data = dict(names=parameters_names, values=table_val)
        source = ColumnDataSource(data)
        columns = [TableColumn(field="names", title="Parameter name"),
                   TableColumn(field="values", title="Value (uncertainty)")]
        bokeh_table = DataTable(source=source, columns=columns, width=600, height=600)
    
    # Create Matplotlib table
    cell_text = []

    for i, name in enumerate(parameters_names):
        cell_text.append([name, table_val[i]])
    
    fig, ax = plt.subplots()
    ax.axis('off')
    mpl_table = ax.table(cellText=cell_text, colLabels=['Parameter', 'Value'], loc='center', cellLoc='left', colWidths=[0.3, 0.7])
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(12)
    mpl_table.scale(1, 1.5)  
    # Alternating row colors
    for i, (row, col) in enumerate(mpl_table.get_celld()):
        if row > 0:  # Skip the header row
            color = 'lightgrey' if row % 2 == 0 else 'white'
            mpl_table.get_celld()[(row, col)].set_facecolor(color)
    
    return fig, bokeh_table


def plot_2d_sigmas(mat_ax, params1, params2, bins=50, eps=0.25, x_range=None,
                   y_range=None):
    if x_range is None:

        rangex = [params1.min() - eps*np.abs(params1.min()), params1.max() +
                  eps*np.abs(params1.max())]

    else:
        rangex = x_range

    if y_range is None:

        rangey = [params2.min() - eps*np.abs(params2.min()), params2.max() +
                  eps*np.abs(params2.max())]

    else:

        rangey = y_range

    hist = np.histogram2d(params1, params2, bins=bins, range=[rangex, rangey])
    x_center = hist[1][:-1] + np.diff(hist[1]) / 2
    y_center = hist[2][:-1] + np.diff(hist[2]) / 2

    xx, yy = np.meshgrid(x_center, y_center)
    import scipy.ndimage as snd
    hist_to_plot = snd.gaussian_filter(hist[0], 1).T

    order = np.argsort(hist_to_plot.flat)
    sorted_cumsum = np.cumsum(hist_to_plot.flat[order])

    levels = []
    for sig in [0.3934, 0.8646, 0.9888, 0.99966][::-1]:
        mask = sorted_cumsum < (1 - sig) * sorted_cumsum[-1]

        levels.append(hist_to_plot.ravel()[order][mask][-1])

    mat_ax.contourf(xx, yy, hist_to_plot, levels=levels, cmap='Blues', extend='max')
    mat_ax.contour(xx, yy, hist_to_plot, levels=levels,
                   linestyles=['dotted', 'dashdot', 'dashed', 'solid'],
                   colors='skyblue')
