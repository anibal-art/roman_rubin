"""Simulate one Roman+Rubin microlensing event from a catalogue row.

sim_event(i, event_params, path_ephemerides, model, ...) builds the Roman and
Rubin telescopes at the event RA/Dec (tel_roman_rubin, source pointing), seeds
numpy with i, builds the true pyLIMA model from the catalogue realization
(parameters, materialized fluxes, caustic origin), simulates the noiseless
light curves, applies the Roman/Rubin photometry (noise, m5 and saturation
filter), resynchronizes the parallax arrays and applies the Roman-only
detection criteria. Returns (pyLIMA model, true parameters, decision).
"""

import copy
import numpy as np
from detection_criteria import deviation_from_constant
from photometry.constants import SIMULATION_BAND_ZERO_POINTS
from photometry.roman_rubin_photometry import apply_roman_rubin_photometry
from photometry.roman_rubin_photometry import replace_flux_by_noisy_magnitude_flux_ZP
from photometry.roman_rubin_photometry import set_photometric_parameters
from set_model_pyLIMA import model_choice
from set_model_pyLIMA import parameters_model
from set_telescopes_pyLIMA import tel_roman_rubin
from simulation.core import simulate_light_curve
from simulation.realization import data_has_materialized_blend_ratio
from simulation.realization import data_has_materialized_fluxes
from simulation.realization import realization_from_catalog
from simulation.roman_detection import combine_roman_detection_statistics
from simulation.roman_detection import roman_detection_chi2_statistics
from simulation.roman_detection import roman_detection_n3sigma_statistics
from simulation.roman_detection import selection_data_available
from simulation.roman_detection import selection_telescopes
from simulation.telescope_arrays import _capture_prefilter_telescope_times
from simulation.telescope_arrays import _mask_time_indexed_telescope_arrays
from simulation.telescope_arrays import inject_model_flux_for_ground_telescopes
from simulation.telescope_arrays import remove_empty_telescopes
from timing_utils import StageTimer


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
