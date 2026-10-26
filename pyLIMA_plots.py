"""Compatibility facade: the plotting code lives in plotting/.

* plotting/pyLIMA_plots.py          light curves, model telescopes, colors
* plotting/pyLIMA_geometry.py       lens geometry and astrometry
* plotting/pyLIMA_distributions.py  sample distributions, parameter table
"""

from plotting.pyLIMA_plots import (  # noqa: F401  (re-exported)
    update_matplotlib_colors,
    create_telescopes_to_plot_model,
    plot_lightcurves,
    initialize_light_curves_plot,
    plot_photometric_models,
    plot_aligned_data,
    plot_residuals,
    plot_lightcurve_windows,
    plot_residuals_windows,
    MAX_PLOT_TICKS,
    MARKER_SYMBOLS,
    MARKERS_COLORS,
)
from plotting.pyLIMA_geometry import (  # noqa: F401  (re-exported)
    plot_geometry,
    plot_astrometry,
    plot_astrometric_models,
    plot_astrometric_data,
)
from plotting.pyLIMA_distributions import (  # noqa: F401  (re-exported)
    plot_distribution,
    plot_parameters_table,
    plot_2d_sigmas,
)
