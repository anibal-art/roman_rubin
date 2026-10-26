import numpy as np
import os, sys, re, copy, math, time, traceback, inspect
import pandas as pd
from pathlib import Path

# Get the directory where the script is located
script_dir = Path(__file__).parent
home_dir = os.path.expanduser("~")

from rubin_sim.phot_utils.photometric_parameters import PhotometricParameters
from rubin_sim.phot_utils.signaltonoise import calc_gamma, calc_mag_error_m5

# astropy
import astropy.units as u
from astropy.table import QTable

# --- Fix para IERS y sidereal_time de Astropy ---
from astropy.utils import iers
iers.conf.auto_max_age = None
iers.conf.auto_download = False  # No intenta descargar
iers.conf.iers_degraded_accuracy = 'warn'

from pyLIMA.simulations import simulator
from class_analysis import Analysis_Event
from fit_lc import fit_rubin_roman, parallax_suffix
from timing_utils import StageTimer
from photometry.roman_photometry import apply_roman_f146_photometry
from photometry.constants import (
    PYLIMA_FIT_ZERO_POINT,
    SIMULATION_BAND_ZERO_POINTS,
)
from simulation.realization import (
    realization_from_catalog,
    data_has_materialized_blend_ratio,
    data_has_materialized_fluxes,
)
from simulation.parallax_cache import (
    install_earth_ephemerides_cache,
)
from simulation.core import simulate_light_curve
# ROMAN_F146_PANDEIA_INTEGRATION_V2
from detection_criteria import filter5points, deviation_from_constant, has_consecutive_numbers, filter_band, mag, debug_nsigma_global
from read_save import save_sim, save_fit, read_data
from set_model_pyLIMA import (
    model_choice,
    parameters_model,
    normalize_model_name,
)

# # ================================================================
# #  Guardado a Parquet
# # ================================================================
from utils.io import save_dict_as_parquet as _save_dict_as_parquet


from set_telescopes_pyLIMA import tel_roman_rubin

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

# ================================================================
#  Photometric parameters
# ================================================================
# ================================================================
# Rubin photometric gamma cache
# ================================================================
#
# calc_gamma() is expensive because Rubin Sim constructs and
# normalizes an SED and integrates it through the bandpass.
#
# For a fixed Rubin observing schedule, bandpass and photometric
# configuration, the m5 vector is identical between microlensing
# events. Therefore the corresponding gamma vector can be computed
# once per process and reused.
#
# The scientific calculation remains Rubin Sim's own calc_gamma()
# and calc_mag_error_m5().  This cache only avoids recomputing gamma
# for identical inputs.
# ================================================================

# ============================================================
# Fast synchronization of pyLIMA time-indexed geometry
# after photometric filtering.
#
# Photometric filtering only removes observation rows; it does not
# alter their times. Therefore already-computed parallax geometry can
# be restricted to the surviving rows instead of recomputed.
# ============================================================

# Exact cache of Earth barycentric ephemerides used by pyLIMA.
# Unknown time arrays still fall back to the original Astropy-backed
# implementation. This touches no RNG state.
install_earth_ephemerides_cache()

def sim_event(
    i,
    data,
    path_ephemerides,
    model,
    time_window=None,
    use_roman=True,
    use_rubin=True,
    truth_parallax=True,
    rubin_pointing_mode="source",
    rubin_cache_cell_deg=None,
    apply_detection_criteria=True,
    apply_photometric_filter=True,
    rubin_saturation_mag=None,
    roman_saturation_mag=None,
):
    """
    i (int): índice del evento
    data (dict): parámetros, incluyendo magnitudes de la estrella
    path_ephemerides (str): path a las efemérides de Roman
    model (str): modelo de microlente ('USBL','FSPL','PSPL', etc.)

    rubin_pointing_mode:
        Only "source" is supported: Rubin epochs/m5 come from OpSim at the
        event position (data["maf_ra"], data["maf_dec"], or data["ra"],
        data["dec"]), which is also the Event position used for parallax.
    """

    _sim_timer = StageTimer()
    _sim_timer.start("simulation_total")
    _sim_timer.start("simulation_prep")

    def _finish_sim_event(
        model_obj,
        params_obj,
        decision_obj,
    ):
        _sim_timer.stop("simulation_total")

        payload = _sim_timer.snapshot()

        if model_obj is not None:
            setattr(
                model_obj,
                "pipeline_timings",
                payload,
            )

        return (
            model_obj,
            params_obj,
            decision_obj,
        )

    ZP = SIMULATION_BAND_ZERO_POINTS

    t0 = data["t0"]

    # ============================================================
    # Coordenadas para Rubin/MAF
    # ============================================================
    # Every event uses its own catalogue RA/Dec (rubin_pointing_mode="source").

    source_ra = data.get("maf_ra", data.get("ra", None))
    source_dec = data.get("maf_dec", data.get("dec", None))

    if rubin_pointing_mode != "source":
        raise ValueError(
            "Only rubin_pointing_mode='source' is supported: every event "
            f"uses its own catalogue RA/Dec (got {rubin_pointing_mode!r})."
        )

    if rubin_pointing_mode == "source":

        if source_ra is None or source_dec is None:
            raise ValueError(
                "rubin_pointing_mode='source' requiere ra/dec "
                "o maf_ra/maf_dec."
            )

        rubin_pointing_mode_use = "source"
        tel_ra = float(source_ra)
        tel_dec = float(source_dec)


    _sim_timer.stop("simulation_prep")
    _sim_timer.start("simulation_maf")

    my_own_creation, dataSlice, LSST_BandPass = tel_roman_rubin(
        path_ephemerides,
        time_window=time_window,
        use_roman=use_roman,
        use_rubin=use_rubin,
        Ra=tel_ra,
        Dec=tel_dec,
        rubin_pointing_mode=rubin_pointing_mode_use,
        rubin_cache_cell_deg=rubin_cache_cell_deg,
    )
    _sim_timer.stop("simulation_maf")


    photParams = set_photometric_parameters(15, 2)
    _sim_timer.start("simulation_deepcopy")
    new_creation = copy.deepcopy(my_own_creation)
    _sim_timer.stop("simulation_deepcopy")
    np.random.seed(i)

    # ============================================================
    # Eliminar bandas sin tiempos dentro de la ventana temporal
    # ============================================================

    _sim_timer.start("simulation_remove_empty_telescopes")
    new_creation, removed_telescopes = remove_empty_telescopes(
        new_creation,
        verbose=True,
    )
    _sim_timer.stop("simulation_remove_empty_telescopes")


    # Si no quedó ninguna banda, el evento no puede simularse
    if len(new_creation.telescopes) == 0:

        print(
            "Event cannot be simulated because there are no observations "
            "inside the requested time window."
        )

        return _finish_sim_event(None, {}, False)


    # ============================================================
    # Construir el modelo solo con las bandas que tienen datos
    # ============================================================
# ============================================================
# Construir el modelo solo con las bandas que tienen datos
# ============================================================

    _sim_timer.start("simulation_model_setup")
    if truth_parallax:
        parallax_arg = ["Full", t0]
    else:
        parallax_arg = ["None", 0.0]

    band_order = [
        tel.name
        for tel in new_creation.telescopes
    ]

    # The realization (physical parameters, blend ratios, fsource/ftotal
    # and, for USBL, caustic_origin) comes only from the materialized
    # catalogue row. Nothing is sampled here; a row without materialized
    # fluxes/blend ratios for every active band is an error.
    _sim_timer.start("simulation_model_realization")

    has_materialized_blend = data_has_materialized_blend_ratio(
        data,
        band_order,
    )

    has_materialized_fluxes = data_has_materialized_fluxes(
        data,
        band_order,
    )

    if not (has_materialized_fluxes and has_materialized_blend):
        raise ValueError(
            "sim_event requires a materialized catalogue row: "
            "fsource_<band>, ftotal_<band> and blend_ratio_<band> for "
            "every active band. Generating blending at simulation time "
            "was removed (see legacy/sim_event_generated_realization.py)."
        )

    realization = realization_from_catalog(
        data,
        model,
        truth_parallax,
        t0,
        band_order,
    )

    _sim_timer.stop("simulation_model_realization")

    # USBL: caustic_origin comes from the catalogue (realization_from_catalog
    # raises if it is missing). Other models have no origin.
    if realization.caustic_origin is not None:
        caustic_origin_arg = [realization.caustic_origin, [0, 0]]
    else:
        caustic_origin_arg = None
    bl_random_origin = False

    _sim_timer.start("simulation_model_choice")

    my_own_model = model_choice(
        new_creation,
        model,
        parallax=parallax_arg,
        BL_random_origin=bl_random_origin,
        BL_origin=caustic_origin_arg,
    )

    _sim_timer.stop("simulation_model_choice")

    _sim_timer.start("simulation_model_parameters")

    params, param_order = parameters_model(
        realization.physical_params,
        my_own_model,
    )

    my_own_parameters = [
        params[key]
        for key in param_order
    ]

    _sim_timer.stop("simulation_model_parameters")

    _sim_timer.start("simulation_model_flux")

    if realization.materialized_fluxes is not None:
        # Catalog realization path.
        #
        # fsource_<band> and ftotal_<band> were already materialized
        # when the production catalog was built. They are authoritative
        # simulation inputs and MUST NOT be reconstructed from magnitude
        # or blend_ratio here.
        if my_own_model.blend_flux_parameter != "ftotal":
            raise RuntimeError(
                "Catalog flux passthrough currently requires "
                "blend_flux_parameter='ftotal'."
            )

        my_own_flux_parameters = []
        fs = {}
        G = {}
        F = {}

        for band in band_order:
            fluxes = realization.materialized_fluxes[band]

            fsource = float(fluxes["fsource"])
            ftotal = float(fluxes["ftotal"])

            my_own_flux_parameters.extend(
                [
                    fsource,
                    ftotal,
                ]
            )

            fs[band] = fsource
            F[band] = ftotal

            if fsource <= 0.0:
                raise RuntimeError(
                    f"Invalid catalog fsource for {band}: {fsource}"
                )

            G[band] = (
                ftotal - fsource
            ) / fsource

            if realization.blend_ratio is not None:
                g_catalog = float(
                    realization.blend_ratio[band]
                )

                if not np.isclose(
                    G[band],
                    g_catalog,
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        "Catalog blending inconsistency for "
                        f"{band}: flux-derived g={G[band]} "
                        f"but blend_ratio={g_catalog}"
                    )

    else:
        raise RuntimeError(
            "sim_event requires materialized catalogue fluxes "
            "(fsource_<band>, ftotal_<band>)."
        )

    my_own_parameters += my_own_flux_parameters

    _sim_timer.stop("simulation_model_flux")
    _sim_timer.stop("simulation_model_setup")

    # ============================================================
    # Simulación de la curva de luz (núcleo genérico, sin RNG,
    # sin conocimiento de catálogo ni de la encuesta)
    # ============================================================

    pyLIMA_parameters = simulate_light_curve(
        my_own_model,
        my_own_parameters,
        sim_timer=_sim_timer,
    )

    # Roman-specific: reinforce the theoretical flux for non-Roman
    # (ground) telescopes, since simulate_lightcurve(add_noise=False)
    # only fills Roman's own lightcurve table by itself. Stays here,
    # in the adapter, using the existing helper -- not in the generic
    # core.
    _sim_timer.start("simulation_ground_flux")
    new_creation = inject_model_flux_for_ground_telescopes(
        new_creation,
        my_own_model,
        pyLIMA_parameters,
        roman_band_name="W149",
    )
    _sim_timer.stop("simulation_ground_flux")

    # ============================================================
    # Fotometría Roman + Rubin
    # ============================================================
    
    _sim_timer.start("simulation_parallax_snapshot")

    if truth_parallax:
        _parallax_prefilter_times = (
            _capture_prefilter_telescope_times(
                my_own_model.event
            )
        )
    else:
        _parallax_prefilter_times = None

    _sim_timer.stop("simulation_parallax_snapshot")

    _sim_timer.start("simulation_photometry")
    new_creation, Roman_band, Rubin_band = apply_roman_rubin_photometry(
        new_creation,
        ZP,
        dataSlice,
        LSST_BandPass,
        photParams,
        apply_photometric_filter=apply_photometric_filter,
        rubin_saturation_mag=rubin_saturation_mag,
        roman_saturation_mag=roman_saturation_mag,
    )
    _sim_timer.stop("simulation_photometry")

    # ============================================================
    # Synchronize pyLIMA after photometric filtering
    # ============================================================
    # The lightcurve tables have already been filtered.
    # pyLIMA's previously computed parallax arrays may still
    # correspond to the original observation times.
    #
    # Keep the same Event and the same noisy photometry.
    # No event parameters or random numbers are regenerated.
    # ============================================================

    _sim_timer.start("simulation_parallax_resync")

    # Remove telescopes without usable photometry.
    # This simulation pipeline contains photometry only.
    my_own_model.event.telescopes = [
        tel
        for tel in my_own_model.event.telescopes
        if (
            tel.lightcurve is not None
            and len(tel.lightcurve) > 0
        )
    ]

    if truth_parallax:

        # Parallax geometry was already calculated at the original
        # observation times. Photometric filtering only removes rows,
        # so synchronize all time-indexed pyLIMA arrays by applying
        # exactly the same surviving-time mask instead of recomputing
        # the ephemeris geometry.
        _mask_time_indexed_telescope_arrays(
            my_own_model.event,
            _parallax_prefilter_times,
        )

        for tel in my_own_model.event.telescopes:

            shifts = np.asarray(
                tel.deltas_positions["photometry"]
            )

            expected = (2, len(tel.lightcurve))

            if shifts.shape != expected:
                raise RuntimeError(
                    "Parallax/photometry mismatch after masking: "
                    f"{tel.name}: "
                    f"{shifts.shape} != {expected}"
                )

    _sim_timer.stop("simulation_parallax_resync")


    # ============================================================
    # Synchronize magnitude and flux representations
    # ============================================================
    #
    # Roman and Rubin instrumental noise has already been applied
    # in magnitude space.  Make flux / err_flux consistent before
    # any detection criterion or likelihood calculation.
    #
    # PHOTOMETRY_FLUX_SYNC_BEFORE_DETECTION_V1
    replace_flux_by_noisy_magnitude_flux_ZP(
        my_own_model,
        verbose=False,
    )


    # ============================================================
    # Criterios de detección
    # ============================================================
    _sim_timer.start("simulation_validation")
    has_required_data = selection_data_available(
        Roman_band,
        Rubin_band,
        use_roman=use_roman,
        use_rubin=use_rubin,
    )
    detection_telescopes = selection_telescopes(
        new_creation.telescopes,
        use_roman=use_roman,
        use_rubin=use_rubin,
        roman_band_name="W149",
    )
    _sim_timer.stop("simulation_validation")

    # Roman-only Delta-chi2 diagnostic.
    #
    # This is computed after photometric filtering, parallax resync,
    # and magnitude/flux synchronization, on the final surviving W149
    # data. It is deliberately diagnostic-only at this stage:
    # it MUST NOT affect the detection decision below.
    setattr(
        my_own_model,
        "roman_detection_statistics",
        None,
    )

    if has_required_data:

        if use_roman:

            _sim_timer.start(
                "simulation_detection_chi2"
            )

            roman_chi2_stats = (
                roman_detection_chi2_statistics(
                    my_own_model,
                    pyLIMA_parameters,
                    roman_band_name="W149",
                )
            )

            roman_n3sigma_stats = (
                roman_detection_n3sigma_statistics(
                    my_own_model,
                    pyLIMA_parameters,
                    roman_band_name="W149",
                    nsigma=3.0,
                )
            )

            roman_detection_stats = (
                combine_roman_detection_statistics(
                    roman_chi2_stats,
                    roman_n3sigma_stats,
                    delta_chi2_min=300.0,
                    n3sigma_min=6,
                )
            )

            _sim_timer.stop(
                "simulation_detection_chi2"
            )

            setattr(
                my_own_model,
                "roman_detection_statistics",
                dict(roman_detection_stats),
            )

        # --------------------------------------------------------
        # Opción para ignorar el criterio de detección
        # --------------------------------------------------------
        if not apply_detection_criteria:
            print(
                "Detection criterion disabled. "
                "The event has valid data and will be fitted."
            )

            return _finish_sim_event(my_own_model, pyLIMA_parameters, True)

        # --------------------------------------------------------
        # Detection decision
        # --------------------------------------------------------
        _sim_timer.start("simulation_detection")

        if use_roman:

            if roman_detection_stats is None:
                raise RuntimeError(
                    "Roman detection statistics were not computed "
                    "despite use_roman=True."
                )

            decision = bool(
                roman_detection_stats[
                    "roman_detected"
                ]
            )

            res = int(
                roman_detection_stats[
                    "n3sigma_max_run"
                ]
            )

        else:
            # Preserve the historical Rubin-only behavior. The
            # production Roman-selected campaign does not use this
            # branch.
            decision, res = deviation_from_constant(
                pyLIMA_parameters,
                detection_telescopes,
                nsigma=3.0,
                nmin=6,
                window="all",
            )

        _sim_timer.stop("simulation_detection")


        if decision:

            if use_roman:
                print(
                    "A good Roman-detected event to fit: "
                    f"Delta_chi2="
                    f'{roman_detection_stats["delta_chi2"]:.6f}, '
                    f'n3sigma_max_run='
                    f'{roman_detection_stats["n3sigma_max_run"]}'
                )
            else:
                print("A good event to fit")

            return _finish_sim_event(
                my_own_model,
                pyLIMA_parameters,
                True,
            )

        else:

            if use_roman:
                print(
                    "Not a good Roman-detected event to fit. "
                    f"Delta_chi2="
                    f'{roman_detection_stats["delta_chi2"]:.6f} '
                    f'(pass={roman_detection_stats["pass_delta_chi2"]}), '
                    f'n3sigma_max_run='
                    f'{roman_detection_stats["n3sigma_max_run"]} '
                    f'(pass={roman_detection_stats["pass_n3sigma"]}).'
                )
            else:
                print(
                    "Not a good event to fit.\n"
                    "Fail deviation-from-constant criterion."
                )

            return _finish_sim_event(
                my_own_model,
                pyLIMA_parameters,
                False,
            )

    else:

        if use_roman and use_rubin:
            print(
                "Not a good event to fit since Roman "
                "and/or Rubin has no valid data"
            )

        elif use_rubin:
            print(
                "Not a good event to fit since Rubin "
                "has no valid data"
            )

        elif use_roman:
            print(
                "Not a good event to fit since Roman "
                "has no valid data"
            )

        return _finish_sim_event(my_own_model, pyLIMA_parameters, False)
# ================================================================
#  new_data y extract_data_event (sin cambios de lógica)
# ================================================================
def simulate_event_for_fit(
    i,
    event_params,
    path_ephemerides,
    model,
    time_window=None,
    use_roman=True,
    use_rubin=True,
    truth_parallax=True,
    rubin_pointing_mode="source",
    rubin_cache_cell_deg=None,
    apply_detection_criteria=True,
    apply_photometric_filter=True,
    rubin_saturation_mag=None,
    roman_saturation_mag=None,
):
    my_own_model, pyLIMA_parameters, decision = sim_event(
        i,
        event_params,
        path_ephemerides,
        model,
        time_window=time_window,
        use_roman=use_roman,
        use_rubin=use_rubin,
        truth_parallax=truth_parallax,
        rubin_pointing_mode=rubin_pointing_mode,
        rubin_cache_cell_deg=rubin_cache_cell_deg,
        apply_detection_criteria=apply_detection_criteria,
        apply_photometric_filter=apply_photometric_filter,
        rubin_saturation_mag=rubin_saturation_mag,
        roman_saturation_mag=roman_saturation_mag,
    )

    return my_own_model, pyLIMA_parameters, decision
from collections.abc import Mapping


def sim_fit(
    i,
    system_type,
    model,
    algo,
    path_TRILEGAL_set,
    path_GENULENS_set,
    path_to_save_model,
    path_to_save_fit,
    path_ephemerides,
    path_to_save_results,
    time_window=None,
    param_samplers=None,
    t0_range=[2460413.013828608, 2460413.013828608 + 365.25 * 8],
    custom_system=None,
    catalog_mode="custom_system",
    pair_catalog=None,
    path_pair_catalog=None,
    use_roman=True,
    use_rubin=True,
    truth_parallax=True,
    fit_time_window=None,
    return_data=False,
    fit_model=None,
    fit_parallax=None,
    fit_defaults=None,
    fit_bounds=None,
    initial_guess=None,
    optimizer_options=None,
    rubin_pointing_mode="source",
    rubin_cache_cell_deg=None,
    apply_detection_criteria=True,
    apply_photometric_filter=True,
    rubin_saturation_mag=None,
    roman_saturation_mag=None,
):
    """
    Simula un evento, aplica criterios de detección, guarda la simulación,
    ajusta las combinaciones solicitadas y guarda los resultados.

    Parameters
    ----------
    catalog_mode : {"custom_system"}
        Única fuente soportada: la fila del catálogo materializado.

        ``custom_system`` usa directamente el diccionario entregado en
        ``custom_system`` y no selecciona ni empareja filas de catálogos.

    custom_system : dict-like or None
        En ``catalog_mode="custom_system"`` contiene el sistema verdadero
        completo. En particular, ``tE`` se utiliza directamente.

    time_window : None, tuple or callable
        Si es None, usa la curva completa.
        Si es ``(t_min, t_max)``, se pasa a ``sim_event``.
        Si es callable, se evalúa como ``time_window(event_params)``.
        Los tiempos deben estar en JD.
    """

    tstart = time.time()

    register_clean_exit_handlers()

    seed = i
    Source = seed

    np.random.seed(seed)

    model, fit_model, fit_parallax = resolve_fit_options(
        model,
        fit_model=fit_model,
        fit_parallax=fit_parallax,
        default_fit_parallax=truth_parallax,
    )

    print(
        f"[sim_fit] sim_model={model}, "
        f"fit_model={fit_model}, "
        f"truth_parallax={truth_parallax}, "
        f"fit_parallax={fit_parallax}"
    )

    # ============================================================
    # 1. Construcción de los parámetros verdaderos
    # ============================================================

    if catalog_mode != "custom_system":
        raise ValueError(
            "Only catalog_mode='custom_system' is supported: the true "
            "parameters come from the materialized catalogue. The old "
            "'trilegal_genulens' and 'astrodatalab_pairs' modes were "
            "moved to legacy/sim_fit_generated_inputs.py."
        )

    if catalog_mode == "custom_system":

        if pair_catalog is not None or path_pair_catalog is not None:
            raise ValueError(
                "catalog_mode='custom_system' no usa pair_catalog "
                "ni path_pair_catalog."
            )

        if param_samplers not in (None, {}):
            raise ValueError(
                "catalog_mode='custom_system' recibe valores verdaderos "
                "directamente. No pases param_samplers en este modo."
            )

        event_params = build_event_params_from_custom_system(
            custom_system=custom_system,
            model=model,
            use_roman=use_roman,
            use_rubin=use_rubin,
            rubin_pointing_mode=rubin_pointing_mode,
        )

        # No existe una selección aleatoria de filas.
        ROW_T = int(i)
        ROW_G = int(i)

        # Estas tablas se crean únicamente porque las funciones actuales de
        # guardado y Analysis_Event todavía reciben ambos argumentos.
        # No representan un catálogo ni una pareja lente-fuente.
        custom_metadata_row = pd.DataFrame([event_params])
        TRILEGAL_row = custom_metadata_row.copy()
        GENULENS_row = custom_metadata_row.copy()

        # Evita que las funciones posteriores intenten operar con None.
        path_TRILEGAL_set = "custom_system"
        path_GENULENS_set = "custom_system"

        print("=" * 80)
        print("[sim_fit] catalog_mode='custom_system'")
        print("[sim_fit] No se usaron catálogos ni emparejamiento.")
        print(
            "[sim_fit] Parámetros verdaderos:",
            {
                key: event_params.get(key)
                for key in (
                    "t0",
                    "u0",
                    "tE",
                    "rho",
                    "piEN",
                    "piEE",
                )
                if key in event_params
            },
        )
        print("=" * 80)


    # ============================================================
    # 2. Ventana temporal de la simulación
    # ============================================================

    if callable(time_window):
        time_window_use = time_window(event_params)
    else:
        time_window_use = time_window

    # ============================================================
    # 3. Simulación
    # ============================================================

    my_own_model, pyLIMA_parameters, decision = simulate_event_for_fit(
        i,
        event_params,
        path_ephemerides,
        model,
        time_window=time_window_use,
        use_roman=use_roman,
        use_rubin=use_rubin,
        truth_parallax=truth_parallax,
        rubin_pointing_mode=rubin_pointing_mode,
        rubin_cache_cell_deg=rubin_cache_cell_deg,
        apply_detection_criteria=apply_detection_criteria,
        apply_photometric_filter=apply_photometric_filter,
        rubin_saturation_mag=rubin_saturation_mag,
        roman_saturation_mag=roman_saturation_mag,
    )

    # ============================================================
    # Coordenadas GEOMÉTRICAS realmente usadas por la simulación
    # ============================================================
    # No se vuelven a inferir desde catalog_mode. Se leen del Event
    # construido por sim_event, de modo que todos los modos
    # (TRILEGAL+GENULENS, AstroDataLab y custom_system) sean consistentes.

    simulation_event_ra = np.nan
    simulation_event_dec = np.nan

    if my_own_model is not None:
        simulation_event_ra, simulation_event_dec = get_event_coordinates(
            my_own_model
        )

        event_params["event_ra_used"] = simulation_event_ra
        event_params["event_dec_used"] = simulation_event_dec

        print(
            "[sim_fit] geometry used by simulation: "
            f"RA={simulation_event_ra:.10f}, "
            f"Dec={simulation_event_dec:.10f}"
        )

        if "ra" in event_params and "dec" in event_params:
            print(
                "[sim_fit] source/catalog coordinates: "
                f"RA={float(event_params['ra']):.10f}, "
                f"Dec={float(event_params['dec']):.10f}"
            )

    # ============================================================
    # Guardar la coordenada MAF realmente usada
    # ============================================================

    try:
        import set_telescopes_pyLIMA as stp

        maf_info = getattr(stp, "LAST_DATASLICE_INFO", {})

        event_params["rubin_pointing_mode"] = rubin_pointing_mode
        event_params["rubin_cache_cell_deg"] = rubin_cache_cell_deg

        event_params["maf_mode"] = maf_info.get("mode", np.nan)
        event_params["maf_cache_mode"] = maf_info.get(
            "cache_mode",
            np.nan,
        )

        event_params["maf_ra_used"] = maf_info.get(
            "maf_Ra",
            maf_info.get("Ra", np.nan),
        )

        event_params["maf_dec_used"] = maf_info.get(
            "maf_Dec",
            maf_info.get("Dec", np.nan),
        )

        event_params["maf_source_ra"] = maf_info.get(
            "source_Ra",
            event_params.get("ra", np.nan),
        )

        event_params["maf_source_dec"] = maf_info.get(
            "source_Dec",
            event_params.get("dec", np.nan),
        )

    except Exception as error:
        print(
            "[warning] No pude guardar LAST_DATASLICE_INFO:",
            repr(error),
        )

    # ============================================================
    # Guardar estadísticas continuas de detección Roman
    # ============================================================
    #
    # Preserve the raw Roman-only detection statistics in
    # event_params BEFORE any rejection return.  This makes the
    # selection auditable offline and permits threshold studies
    # without re-simulating the event.
    #
    # The names are explicitly prefixed to avoid collisions with
    # later fit/all-telescope chi2 diagnostics.
    # ============================================================

    roman_detection_stats = None

    if my_own_model is not None:
        roman_detection_stats = getattr(
            my_own_model,
            "roman_detection_statistics",
            None,
        )

    event_params[
        "roman_detection_statistics_available"
    ] = bool(
        roman_detection_stats is not None
    )

    if roman_detection_stats is not None:

        event_params.update(
            {
                "roman_detection_n_data":
                    int(
                        roman_detection_stats[
                            "n_data"
                        ]
                    ),
                "roman_detection_baseline_flux":
                    float(
                        roman_detection_stats[
                            "baseline_flux"
                        ]
                    ),
                "roman_detection_chi2_true":
                    float(
                        roman_detection_stats[
                            "chi2_true"
                        ]
                    ),
                "roman_detection_chi2_flat":
                    float(
                        roman_detection_stats[
                            "chi2_flat"
                        ]
                    ),
                "roman_detection_delta_chi2":
                    float(
                        roman_detection_stats[
                            "delta_chi2"
                        ]
                    ),
                "roman_detection_n3sigma_nsigma":
                    float(
                        roman_detection_stats[
                            "n3sigma_nsigma"
                        ]
                    ),
                "roman_detection_n3sigma_n_above":
                    int(
                        roman_detection_stats[
                            "n3sigma_n_above"
                        ]
                    ),
                "roman_detection_n3sigma_max_run":
                    int(
                        roman_detection_stats[
                            "n3sigma_max_run"
                        ]
                    ),
                "roman_detection_delta_chi2_min":
                    float(
                        roman_detection_stats[
                            "delta_chi2_min"
                        ]
                    ),
                "roman_detection_n3sigma_min":
                    int(
                        roman_detection_stats[
                            "n3sigma_min"
                        ]
                    ),
                "roman_detection_pass_delta_chi2":
                    bool(
                        roman_detection_stats[
                            "pass_delta_chi2"
                        ]
                    ),
                "roman_detection_pass_n3sigma":
                    bool(
                        roman_detection_stats[
                            "pass_n3sigma"
                        ]
                    ),
                "roman_detected":
                    bool(
                        roman_detection_stats[
                            "roman_detected"
                        ]
                    ),
            }
        )

    # ============================================================
    # Guardar diagnóstico del filtro fotométrico
    # ============================================================

    event_params["apply_photometric_filter"] = bool(apply_photometric_filter)

    try:
        event_params.update(
            summarize_photometry_flags(my_own_model)
        )
    except Exception as error:
        print(
            "[warning] No pude resumir photometry flags:",
            repr(error),
        )

    if not decision:
        print("Criteria not satisfied. Event will not be fitted.")

        return {
            "status": "rejected",
            "i": i,
            "model": model,
            "fit_model": fit_model,
            "system_type": system_type,
            "time_window": time_window_use,
            "fit_time_window": fit_time_window,
            "use_roman": use_roman,
            "use_rubin": use_rubin,
            "truth_parallax": truth_parallax,
            "fit_parallax": fit_parallax,
            "catalog_mode": catalog_mode,
            "apply_photometric_filter": apply_photometric_filter,
            "event_params": event_params,
        }

    print("Criteria satisfied. Save the simulated light-curve.")

    # ============================================================
    # 4. Flujo ruidoso y chi2 verdadero
    # ============================================================

    # ============================================================
    # True-model chi2 using the official pyLIMA implementation
    # ============================================================

    from pyLIMA.fits.objective_functions import (
        all_telescope_photometric_chi2,
    )

    n_data_true = sum(
        len(tel.lightcurve)
        for tel in my_own_model.event.telescopes
        if tel.lightcurve is not None
    )

    if n_data_true <= 0:
        raise RuntimeError(
            "No photometric observations for chi2_true."
        )

    # Fail explicitly on invalid photometry.
    for tel in my_own_model.event.telescopes:

        lc = tel.lightcurve

        if lc is None or len(lc) == 0:
            raise RuntimeError(
                f"Empty lightcurve in chi2_true: {tel.name}"
            )

        flux = np.asarray(lc["flux"], dtype=float)
        err = np.asarray(lc["err_flux"], dtype=float)

        if not (
            np.all(np.isfinite(flux))
            and np.all(np.isfinite(err))
            and np.all(err > 0)
        ):
            raise RuntimeError(
                f"Invalid photometry for {tel.name}"
            )

    # No custom residual calculation or silent skipping.
    chi2_true = float(
        all_telescope_photometric_chi2(
            my_own_model,
            pyLIMA_parameters,
        )
    )

    if not np.isfinite(chi2_true):
        raise RuntimeError(
            "pyLIMA returned non-finite chi2_true."
        )

    chi2_true_info = {
        "chi2_true": chi2_true,
        "n_data_true": int(n_data_true),
        "chi2_true_by_telescope": {},
    }

    print(
        "[pyLIMA chi2_true] "
        f"chi2={chi2_true:.6f}, "
        f"N={n_data_true}, "
        f"chi2/N={chi2_true/n_data_true:.6f}"
    )

    print("chi2_true:", chi2_true)
    print("n_data_true:", n_data_true)

    save_simulated_event_if_selected(
        i,
        ROW_G,
        ROW_T,
        path_TRILEGAL_set,
        path_GENULENS_set,
        path_to_save_model,
        my_own_model,
        pyLIMA_parameters,
        event_params,
        GENULENS_row,
        TRILEGAL_row,
    )

    lc_to_fit, lc_to_save = extract_lightcurves_for_fit(
        my_own_model,
    )

    # ============================================================
    # Aplicar ventana temporal SOLO al ajuste
    # ============================================================
    # lc_to_save conserva la curva completa simulada.
    # lc_to_fit se recorta únicamente para el fit.

    lc_to_fit = apply_fit_time_window_to_lc_dict(
        lc_to_fit,
        pyLIMA_parameters,
        fit_time_window=fit_time_window,
    )

    n_fit_points = count_active_fit_points(
        lc_to_fit,
        use_roman=use_roman,
        use_rubin=use_rubin,
    )

    print("n_fit_points after fit_time_window:", n_fit_points)

    if n_fit_points == 0:

        print(
            "Event rejected after fit_time_window: "
            "no active data points remain for the fit."
        )

        return {
            "status": "rejected_fit_window",
            "i": i,
            "model": model,
            "fit_model": fit_model,
            "system_type": system_type,
            "time_window": time_window_use,
            "fit_time_window": fit_time_window,
            "use_roman": use_roman,
            "use_rubin": use_rubin,
            "truth_parallax": truth_parallax,
            "fit_parallax": fit_parallax,
            "catalog_mode": catalog_mode,
            "apply_photometric_filter": apply_photometric_filter,
            "event_params": event_params,
        }

    origin = get_model_origin(
        my_own_model,
    )

    # ============================================================
    # 5. Ajustes
    # ============================================================

    rango = 1

    (
        fit_rr,
        event_fit_rr,
        pyLIMAmodel_rr,
        fit_roman,
        event_fit_roman,
        pyLIMAmodel_roman,
    ) = run_all_fits(
        Source,
        pyLIMA_parameters,
        path_to_save_fit,
        path_ephemerides,
        model,
        algo,
        origin,
        rango,
        lc_to_fit,
        use_roman=use_roman,
        use_rubin=use_rubin,
        fit_model=fit_model,
        fit_parallax=fit_parallax,
        fit_defaults=fit_defaults,
        fit_bounds=fit_bounds,
        initial_guess=initial_guess,
        optimizer_options=optimizer_options,
        event_ra=simulation_event_ra,
        event_dec=simulation_event_dec,
        true_model=my_own_model,
    )

    # ============================================================
    # Verificación obligatoria: simulación y fit misma geometría
    # ============================================================

    validate_event_coordinates(
        pyLIMAmodel_rr,
        simulation_event_ra,
        simulation_event_dec,
        label="Rubin+Roman/Rubin fit",
    )

    if pyLIMAmodel_roman is not None:
        validate_event_coordinates(
            pyLIMAmodel_roman,
            simulation_event_ra,
            simulation_event_dec,
            label="Roman-only fit",
        )

    # ============================================================
    # 6. Análisis y guardado
    # ============================================================

    if catalog_mode == "custom_system":
        nset_str = "custom"
    else:
        nset_str = extract_nset_string(
            path_GENULENS_set,
        )

    Event = build_analysis_event(
        i,
        model,
        algo,
        path_to_save_model,
        path_to_save_fit,
        GENULENS_row,
        TRILEGAL_row,
        event_params,
        fit_rr,
        fit_roman,
        origin,
        lc_to_save,
        pyLIMA_parameters,
        use_roman=use_roman,
        use_rubin=use_rubin,
        truth_parallax=truth_parallax,
        fit_model=fit_model,
        fit_parallax=fit_parallax,
        fit_time_window=fit_time_window,
        chi2_true=chi2_true_info,
    )

    save_roman = fit_roman is not None

    (
        new_data_true,
        new_data_rr,
        new_data_roman,
    ) = save_extracted_results(
        Event,
        model,
        i,
        system_type,
        nset_str,
        path_to_save_results,
        save_roman=save_roman,
    )

    tend = time.time()

    print("El tiempo transcurrido fue de:", tend - tstart)

    if return_data:
        return {
            "status": "fitted",
            "i": i,
            "sim_model": model,
            "fit_model": (
                fit_model
                if fit_model is not None
                else model
            ),
            "truth_parallax": truth_parallax,
            "fit_parallax": fit_parallax,
            "initial_guess": initial_guess,
            "fit_time_window": fit_time_window,
            "algo": algo,
            "system_type": system_type,
            "time_window": time_window_use,
            "use_roman": use_roman,
            "use_rubin": use_rubin,
            "catalog_mode": catalog_mode,
            "apply_photometric_filter": apply_photometric_filter,
            "event_params": event_params,
            "fit_rr": fit_rr,
            "pyLIMAmodel_rr": pyLIMAmodel_rr,
            # Modelo/parametros exactos usados para generar los datos.
            # Útiles para plots sin reconstruir el Event.
            "pyLIMAmodel_true": my_own_model,
            "pyLIMA_parameters_true": pyLIMA_parameters,
            "event_ra_used": simulation_event_ra,
            "event_dec_used": simulation_event_dec,
            "fit_roman": fit_roman,
            "pyLIMAmodel_roman": pyLIMAmodel_roman,
            "true": new_data_true,
            "fit_rr_data": new_data_rr,
            "fit_roman_data": new_data_roman,
        }

    return (
        fit_rr,
        pyLIMAmodel_rr,
        fit_roman,
        pyLIMAmodel_roman,
    )

# ============================================================================
# Multi-fit generalization: one simulation, multiple fits on the same noisy LC
# ============================================================================


def _call_fit_rubin_roman_compatible(*args, **kwargs):
    """
    Call fit_rubin_roman while passing only keyword arguments supported by the
    installed version of fit_lc.py.

    This keeps the function compatible with both older versions and the newer
    version that accepts event_ra/event_dec and random_state.
    """

    try:
        parameters = inspect.signature(fit_rubin_roman).parameters
        supported_kwargs = {
            key: val
            for key, val in kwargs.items()
            if key in parameters
        }

        dropped = sorted(set(kwargs) - set(supported_kwargs))
        if dropped:
            print(
                "[warning] fit_rubin_roman does not accept these kwargs; "
                f"dropping: {dropped}",
                flush=True,
            )

        return fit_rubin_roman(*args, **supported_kwargs)

    except Exception:
        # If inspect failed for any reason, call directly so the original error
        # is not hidden.
        return fit_rubin_roman(*args, **kwargs)


# Redefinition compatible with multi-fit runs.  Since Python resolves globals at
# call time, the existing sim_fit() defined above will also use this new version.
def run_all_fits(
    Source,
    pyLIMA_parameters,
    path_to_save_fit,
    path_ephemerides,
    model,
    algo,
    origin,
    rango,
    lc_to_fit,
    use_roman=True,
    use_rubin=True,
    fit_model=None,
    fit_parallax=None,
    fit_defaults=None,
    fit_bounds=None,
    initial_guess=None,
    event_ra=None,
    event_dec=None,
    random_state=None,
    optimizer_options=None,
    true_model=None,
):
    """
    Run Roman+Rubin/Rubin-only and optional Roman-only fits.

    Adds compatibility filtering for fit_rubin_roman keyword arguments,
    a deterministic random_state (defaulting to Source), and the same
    event_ra/event_dec geometry used for parallax fits.
    """

    model, fit_model, fit_parallax = resolve_fit_options(
        model,
        fit_model=fit_model,
        fit_parallax=fit_parallax,
        default_fit_parallax=True,
    )

    if (event_ra is None) != (event_dec is None):
        raise ValueError(
            "event_ra y event_dec deben pasarse juntos o ambos ser None."
        )

    if random_state is None:
        try:
            random_state = int(Source)
        except Exception:
            random_state = None

    lc_W149 = lc_to_fit["W149"] if use_roman else []

    lc_u = lc_to_fit["u"] if use_rubin else []
    lc_g = lc_to_fit["g"] if use_rubin else []
    lc_r = lc_to_fit["r"] if use_rubin else []
    lc_i = lc_to_fit["i"] if use_rubin else []
    lc_z = lc_to_fit["z"] if use_rubin else []
    lc_y = lc_to_fit["y"] if use_rubin else []

    print("Start the fit using active telescopes:")
    print(
        f"[run_all_fits] sim_model={model}, "
        f"fit_model={fit_model}, "
        f"fit_parallax={fit_parallax}, "
        f"event_ra={event_ra}, event_dec={event_dec}, "
        f"random_state={random_state}, "
        f"initial_guess={initial_guess!r}"
    )

    fit_rr, event_fit_rr, pyLIMAmodel_rr = _call_fit_rubin_roman_compatible(
        Source,
        pyLIMA_parameters,
        path_to_save_fit,
        path_ephemerides,
        model,
        algo,
        origin,
        rango,
        lc_W149,
        lc_u,
        lc_g,
        lc_r,
        lc_i,
        lc_z,
        lc_y,
        fit_model=fit_model,
        fit_parallax=fit_parallax,
        fit_defaults=fit_defaults,
        fit_bounds=fit_bounds,
        initial_guess=initial_guess,
        event_ra=event_ra,
        event_dec=event_dec,
        random_state=random_state,
        optimizer_options=optimizer_options,
    )

    fit_roman = None
    event_fit_roman = None
    pyLIMAmodel_roman = None

    if use_roman and len(lc_to_fit.get("W149", [])) != 0:

        # ========================================================
        # True-model chi2 on exactly the Roman data
        # ========================================================

        roman_true_info = None

        if true_model is not None:

            roman_true_info = true_photometric_chi2_for_telescope(
                true_model,
                pyLIMA_parameters,
                telescope_name="W149",
            )

            if (
                roman_true_info["n_data_true"]
                != len(lc_to_fit["W149"])
            ):
                raise RuntimeError(
                    "Roman true-chi2 data count does not match "
                    "Roman fit data count: "
                    f'{roman_true_info["n_data_true"]} vs '
                    f'{len(lc_to_fit["W149"])}'
                )

            print(
                "[pyLIMA chi2_true Roman] "
                f'chi2={roman_true_info["chi2_true"]:.6f}, '
                f'N={roman_true_info["n_data_true"]}, '
                f'chi2/N='
                f'{roman_true_info["chi2_true"] / roman_true_info["n_data_true"]:.6f}'
            )

        print("Start the fit using only the Roman data:")

        roman_random_state = None
        if random_state is not None:
            try:
                roman_random_state = int(random_state) + 1
            except Exception:
                roman_random_state = random_state

        fit_roman, event_fit_roman, pyLIMAmodel_roman = _call_fit_rubin_roman_compatible(
            Source,
            pyLIMA_parameters,
            path_to_save_fit,
            path_ephemerides,
            model,
            algo,
            origin,
            rango,
            lc_to_fit["W149"],
            [],
            [],
            [],
            [],
            [],
            [],
            fit_model=fit_model,
            fit_parallax=fit_parallax,
            fit_defaults=fit_defaults,
            fit_bounds=fit_bounds,
            initial_guess=initial_guess,
            event_ra=event_ra,
            event_dec=event_dec,
            random_state=roman_random_state,
            optimizer_options=optimizer_options,
        )

        if roman_true_info is not None:
            fit_roman.fit_results["chi2_true_roman"] = (
                roman_true_info["chi2_true"]
            )
            fit_roman.fit_results["n_data_true_roman"] = (
                roman_true_info["n_data_true"]
            )

    else:
        print("Roman-only fit skipped because Roman is off or has no data.")

    return (
        fit_rr,
        event_fit_rr,
        pyLIMAmodel_rr,
        fit_roman,
        event_fit_roman,
        pyLIMAmodel_roman,
    )


def run_multiple_named_fits(
    Source,
    pyLIMA_parameters,
    path_to_save_fit,
    path_ephemerides,
    model,
    algo,
    origin,
    rango,
    lc_to_fit,
    fit_specs,
    use_roman=True,
    use_rubin=True,
    event_ra=None,
    event_dec=None,
    existing_results=None,
    random_state_base=None,
    optimizer_options=None,
):
    """
    Run multiple named fits on the same ``lc_to_fit`` dictionary.

    Parameters
    ----------
    existing_results : dict or None
        Optional precomputed entries, for example the primary fit already run
        by sim_fit(). Keys present here are not refit.
    """

    if existing_results is None:
        results = {}
    else:
        results = dict(existing_results)

    if random_state_base is None:
        try:
            random_state_base = int(Source)
        except Exception:
            random_state_base = None

    Path(path_to_save_fit).mkdir(parents=True, exist_ok=True)

    for k, (fit_key, spec) in enumerate(fit_specs.items()):

        if fit_key in results:
            print(f"[multi-fit] {fit_key} already exists; skipping refit.")
            continue

        fit_model_use = spec.get("model", None)
        fit_parallax_use = spec.get("parallax", None)
        fit_defaults_use = spec.get("defaults", None)
        fit_bounds_use = spec.get("bounds", None)
        initial_guess_use = spec.get("initial_guess", None)

        if random_state_base is None:
            random_state = None
        else:
            random_state = int(random_state_base) + 1000 * (k + 1)

        print("=" * 80)
        print(f"[multi-fit] START {fit_key}: {spec.get('label', fit_key)}")
        print(
            f"[multi-fit] model={fit_model_use}, "
            f"parallax={fit_parallax_use}, "
            f"random_state={random_state}, "
            f"initial_guess={initial_guess_use!r}"
        )
        print("=" * 80)

        try:
            (
                fit_rr,
                event_fit_rr,
                pyLIMAmodel_rr,
                fit_roman,
                event_fit_roman,
                pyLIMAmodel_roman,
            ) = run_all_fits(
                Source,
                pyLIMA_parameters,
                path_to_save_fit,
                path_ephemerides,
                model,
                algo,
                origin,
                rango,
                lc_to_fit,
                use_roman=use_roman,
                use_rubin=use_rubin,
                fit_model=fit_model_use,
                fit_parallax=fit_parallax_use,
                fit_defaults=fit_defaults_use,
                fit_bounds=fit_bounds_use,
                initial_guess=initial_guess_use,
                optimizer_options=optimizer_options,
                event_ra=event_ra,
                event_dec=event_dec,
                random_state=random_state,
            )

            validate_event_coordinates(
                pyLIMAmodel_rr,
                event_ra,
                event_dec,
                label=f"{fit_key} Rubin+Roman/Rubin fit",
            )

            if pyLIMAmodel_roman is not None:
                validate_event_coordinates(
                    pyLIMAmodel_roman,
                    event_ra,
                    event_dec,
                    label=f"{fit_key} Roman-only fit",
                )

            results[fit_key] = _fit_entry_from_objects(
                fit_key=fit_key,
                spec=spec,
                fit_rr=fit_rr,
                event_fit_rr=event_fit_rr,
                pyLIMAmodel_rr=pyLIMAmodel_rr,
                fit_roman=fit_roman,
                event_fit_roman=event_fit_roman,
                pyLIMAmodel_roman=pyLIMAmodel_roman,
            )

        except Exception as error:
            print(
                f"[multi-fit] ERROR in {fit_key}: {repr(error)}",
                flush=True,
            )
            results[fit_key] = _fit_entry_error(
                fit_key=fit_key,
                spec=spec,
                error=error,
            )

    return results


# ================================================================
#  True-generator likelihood on a fresh fit model
# ================================================================


def sim_fit_multi_fits(
    i,
    system_type,
    model,
    algo,
    path_TRILEGAL_set,
    path_GENULENS_set,
    path_to_save_model,
    path_to_save_fit,
    path_ephemerides,
    path_to_save_results,
    time_window=None,
    param_samplers=None,
    t0_range=[2460413.013828608, 2460413.013828608 + 365.25 * 8],
    custom_system=None,
    catalog_mode="custom_system",
    pair_catalog=None,
    path_pair_catalog=None,
    use_roman=True,
    use_rubin=True,
    truth_parallax=True,
    fit_time_window=None,
    return_data=False,
    fit_model=None,
    fit_parallax=None,
    fit_defaults=None,
    fit_bounds=None,
    initial_guess=None,
    optimizer_options=None,
    fit_specs=None,
    primary_fit=None,
    lrt_config=None,
    save_multi_fit_summary=True,
    rubin_pointing_mode="source",
    rubin_cache_cell_deg=None,
    apply_detection_criteria=True,
    apply_photometric_filter=True,
    rubin_saturation_mag=None,
    roman_saturation_mag=None,
):
    """
    Generalization of sim_fit: simulate once, then run multiple fits.

    The simulated noisy light curve is generated exactly once by ``sim_fit``.
    The primary fit is also run by ``sim_fit``.  All additional fits are then
    run on the same ``lc_to_fit`` extracted from the returned true pyLIMA model.

    This is the preferred interface for hypothesis tests such as:

        H0: FSPL without parallax
        H1: FSPL with parallax

    because H0 and H1 are evaluated on the same noise realization and the same
    fit-time window.
    """

    tstart = time.time()

    _multi_timer = StageTimer()
    _multi_timer.start("multifit_total")

    fit_specs_use = normalize_fit_specs(
        fit_specs=fit_specs,
        fit_model=fit_model,
        fit_parallax=fit_parallax,
        fit_defaults=fit_defaults,
        fit_bounds=fit_bounds,
        initial_guess=initial_guess,
    )

    if primary_fit is None:
        primary_fit_key = next(iter(fit_specs_use.keys()))
    else:
        primary_fit_key = str(primary_fit)

    if primary_fit_key not in fit_specs_use:
        raise KeyError(
            f"primary_fit={primary_fit_key!r} no está en fit_specs. "
            f"Opciones: {list(fit_specs_use.keys())}"
        )

    primary_spec = fit_specs_use[primary_fit_key]

    print("=" * 80)
    print("[sim_fit_multi_fits] one simulation + multiple fits")
    print(f"[sim_fit_multi_fits] primary_fit_key = {primary_fit_key}")
    print(f"[sim_fit_multi_fits] fit keys = {list(fit_specs_use.keys())}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Run the original sim_fit once.  This simulates the noisy event and
    #    executes the primary fit.
    # ------------------------------------------------------------------
    _multi_timer.start("primary_simfit")
    primary_result = sim_fit(
        i=i,
        system_type=system_type,
        model=model,
        algo=algo,
        path_TRILEGAL_set=path_TRILEGAL_set,
        path_GENULENS_set=path_GENULENS_set,
        path_to_save_model=path_to_save_model,
        path_to_save_fit=path_to_save_fit,
        path_ephemerides=path_ephemerides,
        path_to_save_results=path_to_save_results,
        time_window=time_window,
        param_samplers=param_samplers,
        t0_range=t0_range,
        custom_system=custom_system,
        catalog_mode=catalog_mode,
        pair_catalog=pair_catalog,
        path_pair_catalog=path_pair_catalog,
        use_roman=use_roman,
        use_rubin=use_rubin,
        truth_parallax=truth_parallax,
        fit_time_window=fit_time_window,
        return_data=True,
        fit_model=primary_spec.get("model", fit_model),
        fit_parallax=primary_spec.get("parallax", fit_parallax),
        fit_defaults=primary_spec.get("defaults", fit_defaults),
        fit_bounds=primary_spec.get("bounds", fit_bounds),
        initial_guess=primary_spec.get("initial_guess", initial_guess),
        optimizer_options=optimizer_options,
        rubin_pointing_mode=rubin_pointing_mode,
        rubin_cache_cell_deg=rubin_cache_cell_deg,
        apply_detection_criteria=apply_detection_criteria,
        apply_photometric_filter=apply_photometric_filter,
        rubin_saturation_mag=rubin_saturation_mag,
        roman_saturation_mag=roman_saturation_mag,
    )
    _multi_timer.stop("primary_simfit")


    if not isinstance(primary_result, dict):
        return primary_result

    if primary_result.get("status") != "fitted":
        primary_result["multi_fit_status"] = "not_run"
        primary_result["multi_fit_reason"] = (
            "primary sim_fit did not return status='fitted'"
        )
        primary_result["fit_specs"] = fit_specs_use
        return primary_result if return_data else primary_result

    Source = int(i)
    seed = int(i)
    pyLIMA_parameters = primary_result["pyLIMA_parameters_true"]
    true_model = primary_result["pyLIMAmodel_true"]
    simulation_event_ra = primary_result.get("event_ra_used", np.nan)
    simulation_event_dec = primary_result.get("event_dec_used", np.nan)

    if not np.isfinite(simulation_event_ra) or not np.isfinite(simulation_event_dec):
        simulation_event_ra, simulation_event_dec = get_event_coordinates(
            true_model
        )

    origin = get_model_origin(true_model)
    rango = 1

    # ------------------------------------------------------------------
    # 2. Re-extract the exact same light curve for all additional fits.
    #    If the runner has patched extract_lightcurves_for_fit, this call uses
    #    the same patch as sim_fit.
    # ------------------------------------------------------------------
    _multi_timer.start("reextract_fit_lightcurve")
    lc_to_fit, lc_to_save_unused = extract_lightcurves_for_fit(true_model)

    lc_to_fit = apply_fit_time_window_to_lc_dict(
        lc_to_fit,
        pyLIMA_parameters,
        fit_time_window=fit_time_window,
    )

    n_fit_points = count_active_fit_points(
        lc_to_fit,
        use_roman=use_roman,
        use_rubin=use_rubin,
    )
    _multi_timer.stop("reextract_fit_lightcurve")


    print("[sim_fit_multi_fits] n_fit_points reused:", n_fit_points)

    # ------------------------------------------------------------------
    # 3. Register the already-run primary fit in the multi-fit result dict.
    # ------------------------------------------------------------------
    existing_results = {
        primary_fit_key: _fit_entry_from_objects(
            fit_key=primary_fit_key,
            spec=primary_spec,
            fit_rr=primary_result["fit_rr"],
            event_fit_rr=None,
            pyLIMAmodel_rr=primary_result["pyLIMAmodel_rr"],
            fit_roman=primary_result.get("fit_roman", None),
            event_fit_roman=None,
            pyLIMAmodel_roman=primary_result.get("pyLIMAmodel_roman", None),
        )
    }

    # ------------------------------------------------------------------
    # 4. Run all remaining fits on exactly the same lc_to_fit.
    # ------------------------------------------------------------------
    _multi_timer.start("additional_fits")
    multi_fit_results = run_multiple_named_fits(
        Source=Source,
        pyLIMA_parameters=pyLIMA_parameters,
        path_to_save_fit=path_to_save_fit,
        path_ephemerides=path_ephemerides,
        model=model,
        algo=algo,
        origin=origin,
        rango=rango,
        lc_to_fit=lc_to_fit,
        fit_specs=fit_specs_use,
        use_roman=use_roman,
        use_rubin=use_rubin,
        event_ra=simulation_event_ra,
        event_dec=simulation_event_dec,
        existing_results=existing_results,
        random_state_base=seed,
        optimizer_options=optimizer_options,
    )
    _multi_timer.stop("additional_fits")


    # ------------------------------------------------------------------
    # 5. True-generator likelihood on the same surviving fit-window data.
    # ------------------------------------------------------------------
    # Do NOT evaluate the original true_model directly here.  That object can
    # still carry pyLIMA parallax arrays computed before photometric filtering,
    # with lengths different from the surviving light curves.  Instead, use the
    # fresh H1-like fit model, which was rebuilt from lc_to_fit and therefore
    # has the correct times, errors and parallax arrays.  The true fluxes are
    # recomputed in the fit-model zero-point system, currently pyLIMA ZP=27.4.
    _multi_timer.start("true_generator")
    true_model_for_likelihood = None

    try:
        alternative_key = "H1"
        if isinstance(lrt_config, dict):
            alternative_key = str(lrt_config.get("alternative", alternative_key))

        true_generator_stats = compute_true_generator_stats_on_fresh_fit_model(
            multi_fit_results=multi_fit_results,
            true_pyLIMA_parameters=pyLIMA_parameters,
            event_params=primary_result.get("event_params", {}),
            alternative_key=alternative_key,
            fit_zero_point=PYLIMA_FIT_ZERO_POINT,
        )

        true_model_for_likelihood = multi_fit_results[
            true_generator_stats["fresh_model_key"]
        ]["pyLIMAmodel_rr"]

    except Exception as error:
        true_generator_stats = true_generator_stats_fallback_from_primary_result(
            primary_result,
            true_model=true_model,
        )
        true_generator_stats["error"] = repr(error)
        true_generator_stats["traceback"] = traceback.format_exc()

    # If pyLIMA returned an error/NaNs instead of raising, keep the error
    # message but fill chi2/n_data from the already computed robust true chi2.
    try:
        chi2_is_bad = not np.isfinite(float(true_generator_stats.get("chi2", np.nan)))
    except Exception:
        chi2_is_bad = True

    if chi2_is_bad:
        fallback_stats = true_generator_stats_fallback_from_primary_result(
            primary_result,
            true_model=true_model,
        )
        for key, value in fallback_stats.items():
            if key not in true_generator_stats:
                true_generator_stats[key] = value
            else:
                try:
                    current = true_generator_stats[key]
                    if isinstance(current, float) and not np.isfinite(current):
                        true_generator_stats[key] = value
                except Exception:
                    pass

    _multi_timer.stop("true_generator")

    # ------------------------------------------------------------------
    # 6. LRT if requested.
    # ------------------------------------------------------------------
    _multi_timer.start("lrt")
    lrt_results = None

    if lrt_config is not None:
        if not isinstance(lrt_config, dict):
            raise TypeError("lrt_config debe ser un diccionario o None.")

        lrt_results = compute_lrt_from_multi_fits(
            multi_fit_results,
            null_key=lrt_config.get("null", "H0"),
            alternative_key=lrt_config.get("alternative", "H1"),
            delta_k=lrt_config.get("delta_k", None),
        )

        lrt_results = add_oracle_lrt_to_results(
            lrt_results,
            true_generator_stats=true_generator_stats,
            multi_fit_results=multi_fit_results,
        )

    _multi_timer.stop("lrt")

    # ------------------------------------------------------------------
    # 7. Save one compact parquet with H0/H1/true/LRT diagnostics.
    # ------------------------------------------------------------------
    if catalog_mode == "custom_system":
        nset_str = "custom"
    else:
        nset_str = extract_nset_string(
            path_GENULENS_set,
            default="manual",
        )

    multi_fit_summary_path = None
    multi_fit_summary_record = None

    if save_multi_fit_summary:
        extra_metadata = {
            "status": primary_result.get("status", ""),
            "sim_model": model,
            "truth_parallax": bool(truth_parallax),
            "use_roman": bool(use_roman),
            "use_rubin": bool(use_rubin),
            "catalog_mode": catalog_mode,
            "apply_photometric_filter": bool(apply_photometric_filter),
            "apply_detection_criteria": bool(apply_detection_criteria),
            "n_fit_points": int(n_fit_points),
            "initial_guess_default": repr(initial_guess),
            "elapsed_sec_multi_total": float(time.time() - tstart),
        }

        simulation_timings = getattr(
            true_model,
            "pipeline_timings",
            {},
        )

        if isinstance(simulation_timings, dict):
            extra_metadata.update(
                simulation_timings
            )

        extra_metadata.update(
            _multi_timer.snapshot()
        )

        _multi_timer.start("multifit_summary_save")
        multi_fit_summary_path, multi_fit_summary_record = save_multi_fit_summary_parquet(
            i=i,
            path_to_save_results=path_to_save_results,
            primary_fit_key=primary_fit_key,
            multi_fit_results=multi_fit_results,
            true_generator_stats=true_generator_stats,
            lrt_results=lrt_results,
            nset_str=nset_str,
            extra_metadata=extra_metadata,
        )
        _multi_timer.stop("multifit_summary_save")


    _multi_timer.stop("multifit_total")

    pipeline_timings = {}

    simulation_timings = getattr(
        true_model,
        "pipeline_timings",
        {},
    )

    if isinstance(simulation_timings, dict):
        pipeline_timings.update(
            simulation_timings
        )

    pipeline_timings.update(
        _multi_timer.snapshot()
    )

    # multi_fit_summary_record is the live record returned to the runner.
    # The on-disk parquet already contains all pre-save timings; the live
    # record additionally contains save + total.
    if isinstance(multi_fit_summary_record, dict):
        multi_fit_summary_record.update(
            pipeline_timings
        )

    # ------------------------------------------------------------------
    # 8. Merge with the primary sim_fit return payload.
    # ------------------------------------------------------------------
    primary_result.update({
        "multi_fit_status": "completed",
        "pipeline_timings": pipeline_timings,
        "fit_specs": fit_specs_use,
        "primary_fit_key": primary_fit_key,
        "multi_fit_results": multi_fit_results,
        "true_generator_likelihood_stats": true_generator_stats,
        "lrt_results": lrt_results,
        "multi_fit_summary_path": (
            str(multi_fit_summary_path)
            if multi_fit_summary_path is not None
            else None
        ),
        "multi_fit_summary_record": multi_fit_summary_record,
        "pyLIMAmodel_true_fit_window": true_model_for_likelihood,
    })

    if return_data:
        return primary_result

    # Historical no-return-data mode: return the primary objects, while all
    # multi-fit diagnostics are saved on disk.
    return (
        primary_result.get("fit_rr"),
        primary_result.get("pyLIMAmodel_rr"),
        primary_result.get("fit_roman"),
        primary_result.get("pyLIMAmodel_roman"),
    )



# ================================================================
#  OVERRIDES 2026-08-26
#  Consistent chi2-based likelihoods for H0/H1/true_generator
# ================================================================

# These definitions intentionally override the earlier versions above.
# They avoid pyLIMA's absolute photometric-likelihood normalization for the
# hypothesis tests and use the Gaussian likelihood up to a common constant:
#
#     logL_chi2 = -0.5 * chi2
#     nll_chi2  =  0.5 * chi2
#
# For H0/H1/true evaluated on the same surviving light-curve points, the
# omitted normalization constant cancels exactly in likelihood-ratio tests.

