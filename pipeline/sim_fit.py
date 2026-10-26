"""Simulate one catalogue event, fit it and save the outputs.

sim_fit is the function called by launch_catalog_simfit_from_config for each
event: sim_event -> lc_to_fit -> run_all_fits -> records / parquet.
"""

import numpy as np
import pandas as pd
import time
from analysis.simfit_records import build_analysis_event
from analysis.simfit_records import extract_nset_string
from analysis.simfit_records import save_extracted_results
from analysis.simfit_records import save_simulated_event_if_selected
from fitting.fit_inputs import apply_fit_time_window_to_lc_dict
from fitting.fit_inputs import count_active_fit_points
from fitting.fit_inputs import extract_lightcurves_for_fit
from fitting.fit_inputs import get_event_coordinates
from fitting.fit_inputs import get_model_origin
from fitting.fit_inputs import resolve_fit_options
from fitting.fit_inputs import validate_event_coordinates
from photometry.roman_rubin_photometry import summarize_photometry_flags
from simulation.event_inputs import build_event_params_from_custom_system
from utils.process import register_clean_exit_handlers
from fitting.run_fits import (
    run_all_fits,
)
from simulation.simulate_event import (
    sim_event,
)


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
