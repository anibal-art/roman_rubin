"""Data and bookkeeping for a light-curve fit.

* extract_lightcurves_for_fit: simulated model -> lc_to_fit
  {"W149": array, "u": array, ...} with columns [time, mag, err_mag].
* build_fit_time_mask / apply_fit_time_window_to_lc_dict: optional
  fit-only time window (never affects simulation or detection).
* get_event_coordinates / validate_event_coordinates: the fit must use
  exactly the RA/Dec of the simulated Event.
* resolve_fit_options and the paths of the saved fit results.

extract_lightcurves_for_fit can be monkey-patched as
functions_roman_rubin.extract_lightcurves_for_fit (sim_fit looks it up
there).
"""

import numpy as np
from fit_lc import parallax_suffix
from pathlib import Path
from set_model_pyLIMA import normalize_model_name


def build_fit_time_mask(times, t0, tE, fit_time_window=None):
    """
    Máscara temporal usada solamente para el ajuste.

    No modifica la simulación.
    No modifica el criterio de detección.
    """

    times = np.asarray(times, dtype=float)

    if fit_time_window is None:
        return np.ones_like(times, dtype=bool)

    if not isinstance(fit_time_window, dict):
        raise TypeError("fit_time_window debe ser None o un diccionario.")

    mode = fit_time_window.get("mode", "all")

    if mode == "all":
        return np.ones_like(times, dtype=bool)

    if mode == "t0_pm_tE":
        return (times >= t0 - tE) & (times <= t0 + tE)

    if mode == "t0_pm_2tE":
        return (times >= t0 - 2.0 * tE) & (times <= t0 + 2.0 * tE)

    if mode == "t0_pm_factor_tE":
        factor = float(fit_time_window.get("factor", 3.5))
        return (times >= t0 - factor * tE) & (times <= t0 + factor * tE)

    if mode == "custom_jd":
        t_min = float(fit_time_window["t_min"])
        t_max = float(fit_time_window["t_max"])
        return (times >= t_min) & (times <= t_max)

    raise ValueError(f"fit_time_window mode no reconocido: {mode}")


def apply_fit_time_window_to_lc_dict(
    lc_to_fit,
    pyLIMA_parameters,
    fit_time_window=None,
):
    """
    Aplica la ventana temporal solo al diccionario lc_to_fit.

    lc_to_fit contiene arrays con columnas:
        time, mag, err_mag

    No toca lc_to_save ni el modelo simulado completo.
    """

    if fit_time_window is None:
        return lc_to_fit

    mode = fit_time_window.get("mode", "all")

    if mode == "all":
        return lc_to_fit

    t0 = float(pyLIMA_parameters["t0"])
    tE = float(pyLIMA_parameters["tE"])

    filtered = {}

    for band, arr in lc_to_fit.items():

        if arr is None or len(arr) == 0:
            filtered[band] = []
            continue

        arr = np.asarray(arr)

        times = arr[:, 0]

        mask = build_fit_time_mask(
            times,
            t0=t0,
            tE=tE,
            fit_time_window=fit_time_window,
        )

        filtered[band] = arr[mask]

    return filtered


def count_active_fit_points(
    lc_to_fit,
    use_roman=True,
    use_rubin=True,
):
    """
    Cuenta cuántos puntos quedan disponibles para el ajuste
    después de aplicar fit_time_window.

    Solo cuenta los telescopios activos según use_roman/use_rubin.
    """

    bands = []

    if use_roman:
        bands.append("W149")

    if use_rubin:
        bands.extend(["u", "g", "r", "i", "z", "y"])

    n_total = 0

    for band in bands:

        arr = lc_to_fit.get(band, [])

        if arr is None:
            continue

        try:
            n_total += len(arr)
        except TypeError:
            continue

    return int(n_total)


def extract_lightcurves_for_fit(pyLIMA_model):
    """
    Extrae las curvas de luz del evento en dos formatos:

    lc_to_fit:
        diccionario con arrays numpy para pasar a fit_rubin_roman.

    lc_to_save:
        diccionario con tablas para guardar dentro de Analysis_Event.
    """
    lc_to_fit = {}
    lc_to_save = {}

    for telo in pyLIMA_model.event.telescopes:

        if len(telo.lightcurve["mag"]) != 0:

            tbl = telo.lightcurve[
                [
                    "time",
                    "mag",
                    "err_mag",
                ]
            ].copy()

            if hasattr(tbl["time"], "value"):
                tbl["time"] = tbl["time"].value

            df = tbl.to_pandas()

            lc_to_fit[telo.name] = df.values
            lc_to_save[telo.name] = tbl

        else:
            lc_to_fit[telo.name] = []

    # Garantiza que existan todas las claves que fit_rubin_roman espera.
    for band in ["W149", "u", "g", "r", "i", "z", "y"]:
        if band not in lc_to_fit:
            lc_to_fit[band] = []

    return lc_to_fit, lc_to_save


def get_event_coordinates(pyLIMA_model):
    """Devuelve (RA, Dec) del Event asociado a un modelo pyLIMA."""
    if pyLIMA_model is None or getattr(pyLIMA_model, "event", None) is None:
        raise ValueError("El modelo pyLIMA no contiene un Event válido.")

    return (
        float(pyLIMA_model.event.ra),
        float(pyLIMA_model.event.dec),
    )


def validate_event_coordinates(
    pyLIMA_model,
    expected_ra,
    expected_dec,
    label="fit",
    atol=1e-10,
):
    """
    Verifica que simulación y ajuste usen exactamente la misma dirección
    del cielo. Esto es crítico cuando se usa paralaje.
    """
    got_ra, got_dec = get_event_coordinates(pyLIMA_model)

    if not (
        np.isclose(got_ra, float(expected_ra), rtol=0.0, atol=atol)
        and np.isclose(got_dec, float(expected_dec), rtol=0.0, atol=atol)
    ):
        raise RuntimeError(
            f"{label} usa coordenadas distintas de la simulación.\n"
            f"simulation = ({float(expected_ra)}, {float(expected_dec)})\n"
            f"{label} = ({got_ra}, {got_dec})"
        )

    return True


def resolve_fit_options(
    model,
    fit_model=None,
    fit_parallax=None,
    default_fit_parallax=True,
):
    """
    Normaliza las opciones de ajuste.

    Regla nueva:
    - model y fit_model son solo nombres de modelo: PSPL, FSPL o USBL.
    - fit_parallax es el único control para ajustar con/sin paralaje.
    - No se interpreta ningún sufijo de paralaje en el nombre del modelo.
    """

    model_use = normalize_model_name(model)

    if fit_model is None:
        fit_model = model_use

    fit_model_use = normalize_model_name(fit_model)

    if fit_parallax is None:
        fit_parallax_use = bool(default_fit_parallax)
    else:
        fit_parallax_use = bool(fit_parallax)

    return model_use, fit_model_use, fit_parallax_use


def expected_fit_results_path(
    path_to_save_fit,
    event_name,
    algo,
    fit_model,
    fit_parallax,
):
    """
    Reproduce el nombre que usa fit_lc.save_fit_results().
    """

    fit_model_use = normalize_model_name(fit_model)
    suffix = f"{fit_model_use}_{parallax_suffix(fit_parallax)}"

    return Path(path_to_save_fit) / f"{event_name}_{algo}_{suffix}.npy"


def fit_results_path_from_fit_object(
    fit_object,
    path_to_save_fit,
    algo,
    fit_model,
    fit_parallax,
    fallback_event_name,
):
    """
    Devuelve el path esperado del .npy de un fit ya corrido.

    Usa fit_object.fit_results["name"] si existe, porque ese nombre
    distingue automáticamente Event_RR, Event_Rubin o Event_Roman.
    """

    event_name = fallback_event_name

    try:
        event_name = fit_object.fit_results.get(
            "name",
            fallback_event_name,
        )
    except Exception:
        pass

    return expected_fit_results_path(
        path_to_save_fit,
        event_name,
        algo,
        fit_model,
        fit_parallax,
    )
