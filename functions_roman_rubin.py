"""Public entry point of the Roman+Rubin simulation/fit pipeline.

This module only re-exports; the code lives in:

* simulation/simulate_event.py     sim_event (one simulated event)
* simulation/event_inputs.py       catalogue row -> event_params
* simulation/roman_detection.py    Roman-only detection criteria
* simulation/telescope_arrays.py   telescope edits after the photometric filter
* photometry/roman_rubin_photometry.py   Roman/Rubin noise and filters
* fitting/fit_inputs.py            lc_to_fit, fit windows, coordinates, paths
* fitting/likelihood.py            chi2 / logL, LRT
* fitting/true_generator.py        chi2 of the true model on the fit data
* fitting/multi_fit.py             named fit specs and parquet summary
* fitting/run_fits.py              run_all_fits, run_multiple_named_fits
* pipeline/sim_fit.py              sim_fit (simulate + fit + save)
* pipeline/sim_fit_multi_fits.py   sim_fit_multi_fits (several named fits, LRT)
* analysis/simfit_records.py       output records

Importing it also configures astropy IERS (no downloads) and installs the
exact Earth-ephemerides cache used by pyLIMA's parallax.
"""

# --- Fix para IERS y sidereal_time de Astropy ---
from astropy.utils import iers
iers.conf.auto_max_age = None
iers.conf.auto_download = False  # No intenta descargar
iers.conf.iers_degraded_accuracy = 'warn'

from fit_lc import fit_rubin_roman  # noqa: F401  (re-exported)
from simulation.parallax_cache import install_earth_ephemerides_cache

# Roman/Rubin photometry (moved; re-exported for backward compatibility)
from photometry.roman_rubin_photometry import (  # noqa: F401
    set_photometric_parameters,
    _rubin_photometric_parameters_signature,
    _rubin_m5_vector_signature,
    _get_cached_rubin_gamma_vector,
    _plain_array,
    _data_slice_column,
    _as_band_limit_array,
    _append_photometry_flag_columns,
    _summarize_flag_arrays,
    summarize_photometry_flags,
    photometry_summary_columns,
    apply_roman_rubin_photometry,
    replace_flux_by_noisy_magnitude_flux_ZP,
    _RUBIN_GAMMA_VECTOR_CACHE,
    _RUBIN_GAMMA_VECTOR_CACHE_STATS,
)
from simulation.telescope_arrays import (  # noqa: F401  (moved; re-exported)
    inject_model_flux_for_ground_telescopes,
    _lightcurve_time_values,
    _capture_prefilter_telescope_times,
    _mask_time_indexed_telescope_arrays,
    remove_empty_telescopes,
    _PARALLAX_TIME_ARRAY_AXES,
)
from simulation.roman_detection import (  # noqa: F401  (moved; re-exported)
    observation_mode_passed,
    selection_data_available,
    selection_telescopes,
    roman_detection_chi2_statistics,
    roman_detection_n3sigma_statistics,
    combine_roman_detection_statistics,
)
from simulation.event_inputs import (  # noqa: F401  (moved; re-exported)
    build_event_params_from_custom_system,
)
from fitting.fit_inputs import (  # noqa: F401  (moved; re-exported)
    build_fit_time_mask,
    apply_fit_time_window_to_lc_dict,
    count_active_fit_points,
    extract_lightcurves_for_fit,
    get_model_origin,
    get_event_coordinates,
    validate_event_coordinates,
    resolve_fit_options,
    parallax_suffix_for_filename,
    expected_fit_results_path,
    fit_results_path_from_fit_object,
)
from fitting.likelihood import (  # noqa: F401  (moved; re-exported)
    _empty_likelihood_stats,
    _as_pylima_parameters_for_stats,
    compute_pylima_photometric_likelihood_stats,
    true_photometric_chi2_for_telescope,
    compute_lrt_from_multi_fits,
    add_oracle_lrt_to_results,
)
from fitting.true_generator import (  # noqa: F401  (moved; re-exported)
    _ordered_model_parameter_names,
    _get_true_pyparam_value,
    _event_param_value,
    _source_mag_for_band,
    _source_fraction_for_band,
    _true_fluxes_in_fresh_fit_system,
    true_parameter_vector_for_fresh_fit_model,
    _choose_fresh_model_for_true_generator,
    compute_true_generator_stats_on_fresh_fit_model,
    _scalar_from_summary_dict,
    true_generator_stats_fallback_from_primary_result,
)
from fitting.multi_fit import (  # noqa: F401  (moved; re-exported)
    normalize_fit_specs,
    _model_vector_from_fit_object,
    _fit_entry_from_objects,
    _fit_entry_error,
    flatten_multi_fit_results_for_parquet,
    save_multi_fit_summary_parquet,
)
from analysis.simfit_records import (  # noqa: F401  (moved; re-exported)
    fit_chi2_dof_from_n_data,
    new_data,
    extract_data_event,
    extract_nset_string,
    build_analysis_event,
    save_extracted_results,
    save_simulated_event_if_selected,
)
from utils.process import (  # noqa: F401  (moved; re-exported)
    register_clean_exit_handlers,
)

from simulation.simulate_event import (  # noqa: F401  (moved; re-exported)
    sim_event,
)
from fitting.run_fits import (  # noqa: F401  (moved; re-exported)
    _call_fit_rubin_roman_compatible,
    run_all_fits,
    run_multiple_named_fits,
)
from pipeline.sim_fit import (  # noqa: F401  (moved; re-exported)
    simulate_event_for_fit,
    sim_fit,
)
from pipeline.sim_fit_multi_fits import (  # noqa: F401  (moved; re-exported)
    sim_fit_multi_fits,
)

# Exact cache of Earth barycentric ephemerides used by pyLIMA.
# Unknown time arrays still fall back to the original Astropy-backed
# implementation. This touches no RNG state.
install_earth_ephemerides_cache()
