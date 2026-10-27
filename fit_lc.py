#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_lc.py

Rutinas para construir eventos pyLIMA, ajustar curvas Roman/Rubin y reconstruir
modelos para graficar.

Diseño:
- La construcción de modelos pyLIMA queda centralizada en set_model_pyLIMA.py.
- Este archivo NO interpreta strings tipo NoPiE.
- El uso de paralaje se controla solo con el booleano fit_parallax.
"""

from pathlib import Path
import numpy as np

from timing_utils import StageTimer

from set_model_pyLIMA import (
    build_pyLIMA_model,
    normalize_model_name,
)

from fitting.fit_event import (  # noqa: F401  (moved; re-exported)
    is_nonempty_lc,
    as_empty_if_none,
    has_rubin_data,
    get_event_name,
    resolve_event_coordinates,
    add_roman_telescope,
    add_rubin_telescopes,
    create_fit_event,
    LSST_BANDS,
)
from fitting.fit_parameters import (  # noqa: F401  (moved; re-exported)
    _param_has,
    _param_get,
    get_param,
    parallax_suffix,
    fit_trajectory_parameter_names,
    initial_params_for_fit_model,
    fit_parameter_order,
    fit_guess_values,
    build_fit_pyLIMA_model,
)
from fitting.fit_bounds import (  # noqa: F401  (moved; re-exported)
    safe_bounds,
    set_fit_bound,
    _custom_bound_candidate_keys,
    _center_for_parameter,
    get_custom_bound_spec,
    resolve_bound_spec,
    apply_custom_bounds,
    apply_narrow_bounds,
    apply_wide_bounds,
    apply_fit_bounds,
)
from fitting.initial_guess import (  # noqa: F401  (moved; re-exported)
    sample_guess_inside_bounds,
    explicit_guess_values,
    validate_initial_guess_inside_bounds,
    resolve_initial_guess,
)
from fitting.trf_optimizer import (  # noqa: F401  (moved; re-exported)
    build_fitter,
    normalize_trf_optimizer_options,
    temporary_trf_least_squares_options,
    optimizer_diagnostics_from_fit,
    run_fit,
    finalize_fit_results,
    TRF_DEFAULT_OPTIMIZER_OPTIONS,
    TRF_ALLOWED_OPTIMIZER_OPTIONS,
)
from fitting.fit_coordinates import (
    normalize_fit_coordinates,
    fancy_parameters_for,
    guess_to_fit_space,
    physical_fit_results,
)


# ============================================================
# Utilidades generales
# ============================================================

# ============================================================
# Creación del Event de pyLIMA
# ============================================================

# ============================================================
# Parámetros iniciales para el modelo de ajuste
# ============================================================

# ============================================================
# Construcción del modelo pyLIMA de ajuste
# ============================================================

# ============================================================
# Fitter
# ============================================================

# ============================================================
# Bounds
# ============================================================

# ============================================================
# Guess inicial
# ============================================================

# ============================================================
# Opciones y diagnósticos del optimizador TRF
# ============================================================

# ============================================================
# Ejecución y guardado
# ============================================================

def save_fit_results(
    path_save,
    event_name,
    algo,
    fit_model,
    fit_parallax,
    fit_results,
):
    """
    Guarda fit_results como .npy.

    El nombre codifica explícitamente si el ajuste usa paralaje.
    No se usa el string NoPiE.
    """
    path_save = Path(path_save)
    path_save.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = normalize_model_name(fit_model)
    suffix += f"_{parallax_suffix(fit_parallax)}"

    output_path = path_save / f"{event_name}_{algo}_{suffix}.npy"

    np.save(
        output_path,
        fit_results,
    )

    return output_path


# ============================================================
# Función principal: compatible con el pipeline actual
# ============================================================

def fit_rubin_roman(
    Source,
    event_params,
    path_save,
    path_ephemerides,
    model,
    algo,
    Origin,
    rango,
    wfirst_lc,
    lsst_u,
    lsst_g,
    lsst_r,
    lsst_i,
    lsst_z,
    lsst_y,
    fit_model=None,
    fit_parallax=None,
    fit_defaults=None,
    fit_bounds=None,
    random_state=None,
    initial_guess=None,
    optimizer_options=None,
    event_ra=None,
    event_dec=None,
    fit_coordinates=None,
):
    """
    Ajusta una curva Roman/Rubin con pyLIMA.

    Parameters
    ----------
    model : str
        Modelo usado para simular. Se guarda como sim_model.
        Debe ser PSPL, FSPL o USBL.

    fit_model : str or None
        Modelo usado para ajustar.
        Si None, usa model.
        Debe ser PSPL, FSPL o USBL.

    fit_parallax : bool or None
        Si True, ajusta con paralaje.
        Si False, ajusta sin paralaje.
        Si None, usa True por compatibilidad con llamadas viejas.

        Importante: ya no se infiere desde nombres tipo NoPiE.

    fit_defaults : dict or None
        Valores iniciales para parámetros que no existan en event_params.

    fit_bounds : dict or None
        Bounds custom por parámetro.

    random_state : int or None
        Semilla usada solamente cuando initial_guess=None y el guess se
        samplea aleatoriamente dentro de los bounds.

    initial_guess : None, str, dict or sequence
        Controla el punto inicial entregado al fitter.

        - None:
            conserva el comportamiento histórico y samplea uniformemente
            dentro de los bounds.
        - "truth" o "fit_params":
            usa los valores de fit_params. En el pipeline de simulaciones
            estos valores se construyen a partir de event_params.
        - dict:
            usa exactamente los valores provistos por nombre de parámetro.
        - list/tuple/ndarray:
            usa exactamente esos valores en el orden devuelto por
            fit_parameter_order().

        Ejemplo FSPL + parallax::

            initial_guess = {
                "t0": 2461000.0,
                "u0": 0.1,
                "tE": 120.0,
                "rho": 1e-3,
                "piEN": 0.2,
                "piEE": -0.1,
            }

    event_ra, event_dec : float or None
        Coordenadas del Event que debe usar el ajuste. Cuando sim_fit llama
        a esta función, deben ser las coordenadas del Event usado realmente
        durante la simulación. Son obligatorias: si falta alguna, el
        ajuste aborta.

    fit_coordinates : None, "physical" or "log_te_rho"
        Coordenadas en las que trabaja el optimizador (pyLIMA
        fancy_parameters, ver fitting.fit_coordinates). None equivale a
        "physical". Bounds, initial_guess e initial_guess_values son
        siempre físicos. fit_results["best_model"] y
        ["covariance_matrix"] quedan en espacio de ajuste;
        ["best_model_physical"] y ["covariance_matrix_physical"] en
        coordenadas físicas.
    """
    _fit_timer = StageTimer()
    _fit_timer.start("fit_total")
    _fit_timer.start("fit_setup")

    rng = np.random.default_rng(random_state)

    if fit_model is None:
        fit_model = model

    if fit_parallax is None:
        fit_parallax = True

    fit_parallax = bool(fit_parallax)
    fit_model_base = normalize_model_name(fit_model)
    sim_model_base = normalize_model_name(model)
    fit_coordinates = normalize_fit_coordinates(fit_coordinates)

    lsst_lcs = {
        "u": as_empty_if_none(lsst_u),
        "g": as_empty_if_none(lsst_g),
        "r": as_empty_if_none(lsst_r),
        "i": as_empty_if_none(lsst_i),
        "z": as_empty_if_none(lsst_z),
        "y": as_empty_if_none(lsst_y),
    }

    event_ra_use, event_dec_use = resolve_event_coordinates(
        event_ra=event_ra,
        event_dec=event_dec,
    )

    print(
        "[fit_rubin_roman] Event coordinates: "
        f"RA={event_ra_use:.10f} deg, "
        f"Dec={event_dec_use:.10f} deg"
    )

    e = create_fit_event(
        Source,
        path_ephemerides,
        as_empty_if_none(wfirst_lc),
        lsst_lcs,
        ra=event_ra_use,
        dec=event_dec_use,
        roman_name="Roman",
    )

    # Protección contra errores silenciosos de geometría/paralaje.
    if not (
        np.isclose(float(e.ra), event_ra_use, rtol=0.0, atol=1e-12)
        and np.isclose(float(e.dec), event_dec_use, rtol=0.0, atol=1e-12)
    ):
        raise RuntimeError(
            "create_fit_event creó un Event con coordenadas distintas "
            "de las solicitadas: "
            f"requested=({event_ra_use}, {event_dec_use}), "
            f"created=({e.ra}, {e.dec})."
        )

    fit_params = initial_params_for_fit_model(
        event_params,
        fit_model_base,
        fit_parallax=fit_parallax,
        fit_defaults=fit_defaults,
        Origin=Origin,
    )

    pyLIMAmodel = build_fit_pyLIMA_model(
        e,
        fit_model_base,
        fit_params,
        Origin=Origin,
        fit_parallax=fit_parallax,
        fancy_parameters=fancy_parameters_for(
            fit_coordinates,
            fit_parameter_order(
                fit_model_base,
                fit_parallax=fit_parallax,
                Origin=Origin,
            ),
        ),
    )

    fit, pool_processes = build_fitter(
        pyLIMAmodel,
        algo,
    )

    rango_used = apply_fit_bounds(
        fit,
        fit_params,
        fit_model_base,
        rango,
        fit_parallax=fit_parallax,
        fit_bounds=fit_bounds,
        Origin=Origin,
    )

    param_order = fit_parameter_order(
        fit_model_base,
        fit_parallax=fit_parallax,
        Origin=Origin,
    )

    initial_guess_used, initial_guess_source = resolve_initial_guess(
        fit,
        fit_params,
        fit_model_base,
        fit_parallax,
        param_order,
        initial_guess=initial_guess,
        rng=rng,
        Origin=Origin,
    )

    initial_guess_fit = guess_to_fit_space(
        pyLIMAmodel,
        param_order,
        initial_guess_used,
    )

    fit.model_parameters_guess = initial_guess_fit

    print(
        "[fit_rubin_roman] Initial guess "
        f"source={initial_guess_source}, "
        f"coordinates={fit_coordinates}, "
        f"order={param_order}, "
        f"values={initial_guess_used}"
    )

    _fit_timer.stop("fit_setup")
    _fit_timer.start("fit_optimizer")

    fit = run_fit(
        fit,
        algo,
        pool_processes=pool_processes,
        optimizer_options=optimizer_options,
    )

    _fit_timer.stop("fit_optimizer")
    _fit_timer.start("fit_finalize")

    fit_results = finalize_fit_results(
        fit,
        event_params,
        algo,
        rango_used,
        e.name,
        sim_model=sim_model_base,
        fit_model=fit_model_base,
        fit_parallax=fit_parallax,
        fit_bounds=fit_bounds,
    )

    # Guardar explícitamente la geometría usada por el ajuste.
    fit_results["event_ra"] = float(e.ra)
    fit_results["event_dec"] = float(e.dec)

    # Guardar exactamente qué guess recibió el optimizador.
    fit_results["initial_guess_source"] = initial_guess_source
    fit_results["initial_guess_parameter_order"] = list(param_order)
    fit_results["initial_guess_values"] = [
        float(value)
        for value in initial_guess_used
    ]
    fit_results["initial_guess_values_fit"] = [
        float(value)
        for value in initial_guess_fit
    ]

    # Coordenadas del optimizador y vista física de los resultados.
    fit_results["fit_coordinates"] = fit_coordinates

    if "best_model" in fit_results:
        fit_results.update(
            physical_fit_results(fit)
        )

    _fit_timer.stop("fit_finalize")
    fit_results.update(
        _fit_timer.snapshot()
    )

    _fit_timer.start("fit_save")

    save_fit_results(
        path_save,
        e.name,
        algo,
        fit_model_base,
        fit_parallax,
        fit_results,
    )

    _fit_timer.stop("fit_save")
    _fit_timer.stop("fit_total")

    # These final timings are available in the live fit object and
    # therefore in the multi-fit summary. We deliberately do not
    # rewrite the .npy just to include the save timing.
    fit_results.update(
        _fit_timer.snapshot()
    )

    return fit, e, pyLIMAmodel


# ============================================================
# Wrapper para construir solo el modelo sin ajustar
# ============================================================

def model_rubin_roman(
    Source,
    event_params,
    path_ephemerides,
    model,
    ORIGIN,
    wfirst_lc,
    lsst_u,
    lsst_g,
    lsst_r,
    lsst_i,
    lsst_z,
    lsst_y,
    fit_model=None,
    fit_parallax=None,
    fit_defaults=None,
    event_ra=None,
    event_dec=None,
):
    """
    Construye el modelo pyLIMA sin correr ajuste.

    Esta función sirve para graficar/evaluar un best_model ya guardado.
    No prepara guesses.
    No prepara bounds.
    No exige u0 ni tE.

    event_params solo se usa para obtener t0 si fit_parallax=True.
    Para un fit sin paralaje puede ser un diccionario vacío.

    event_ra/event_dec permiten reconstruir el modelo con exactamente la
    misma posición del cielo que se usó durante la simulación o el fit.
    Si no se pasan, se conserva el campo fijo histórico.
    """
    del fit_defaults  # argumento mantenido por compatibilidad

    if fit_model is None:
        fit_model = model

    if fit_parallax is None:
        fit_parallax = True

    fit_parallax = bool(fit_parallax)
    fit_model_base = normalize_model_name(fit_model)

    lsst_lcs = {
        "u": as_empty_if_none(lsst_u),
        "g": as_empty_if_none(lsst_g),
        "r": as_empty_if_none(lsst_r),
        "i": as_empty_if_none(lsst_i),
        "z": as_empty_if_none(lsst_z),
        "y": as_empty_if_none(lsst_y),
    }

    event_ra_use, event_dec_use = resolve_event_coordinates(
        event_ra=event_ra,
        event_dec=event_dec,
    )

    e = create_fit_event(
        Source,
        path_ephemerides,
        as_empty_if_none(wfirst_lc),
        lsst_lcs,
        ra=event_ra_use,
        dec=event_dec_use,
        roman_name="Roman",
    )

    t0_parallax = None

    if fit_parallax:
        t0_parallax = get_param(
            event_params,
            "t0",
            aliases=["t_center"],
            required=True,
        )

    pyLIMAmodel = build_pyLIMA_model(
        pyLIMA_event=e,
        model=fit_model_base,
        use_parallax=fit_parallax,
        t0_parallax=t0_parallax,
        origin=ORIGIN,
        blend_flux_parameter="ftotal",
    )

    return pyLIMAmodel
