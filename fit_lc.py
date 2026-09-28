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
from contextlib import contextmanager
import multiprocessing as mul
import numpy as np

from timing_utils import StageTimer

from pyLIMA import event
from pyLIMA import telescopes

from pyLIMA.fits import TRF_fit
from pyLIMA.fits import DE_fit
from pyLIMA.fits import MCMC_fit

from set_model_pyLIMA import (
    build_pyLIMA_model,
    normalize_model_name,
)


# ============================================================
# Constantes del campo
# ============================================================

RA_FIELD = 267.92497054815516
DEC_FIELD = -29.152232510353276

LSST_BANDS = [
    "u",
    "g",
    "r",
    "i",
    "z",
    "y",
]


# ============================================================
# Utilidades generales
# ============================================================

def is_nonempty_lc(lc):
    """
    Devuelve True si la curva de luz tiene al menos un punto.
    """
    if lc is None:
        return False

    try:
        return len(lc) != 0
    except TypeError:
        return False


def as_empty_if_none(lc):
    """
    Convierte None en lista vacía.
    """
    if lc is None:
        return []

    return lc


def has_rubin_data(lsst_lcs):
    """
    Devuelve True si hay al menos una banda Rubin con datos.
    """
    return any(
        is_nonempty_lc(lc)
        for lc in lsst_lcs.values()
    )


def get_event_name(Source, wfirst_lc, lsst_lcs):
    """
    Define el nombre del evento según qué telescopios tienen datos.
    """
    has_roman = is_nonempty_lc(wfirst_lc)
    has_rubin = has_rubin_data(lsst_lcs)

    if has_roman and has_rubin:
        return f"Event_RR_{Source}"

    if has_roman and not has_rubin:
        return f"Event_Roman_{Source}"

    if (not has_roman) and has_rubin:
        return f"Event_Rubin_{Source}"

    raise ValueError(
        "No hay datos para ajustar: Roman y Rubin están vacíos."
    )


def _param_has(params, key):
    """
    Chequea si un parámetro existe en dict, pandas Series,
    pyLIMA parameters object, etc.
    """
    if params is None:
        return False

    if isinstance(params, dict):
        return key in params

    if hasattr(params, "index"):
        return key in params.index

    try:
        params[key]
        return True
    except Exception:
        pass

    return hasattr(params, key)


def _param_get(params, key):
    """
    Obtiene un parámetro desde dict, pandas Series,
    pyLIMA parameters object, etc.
    """
    if isinstance(params, dict):
        return params[key]

    if hasattr(params, "index") and key in params.index:
        return params[key]

    try:
        return params[key]
    except Exception:
        pass

    if hasattr(params, key):
        return getattr(params, key)

    raise KeyError(f"No encontré el parámetro {key}")


def get_param(
    params,
    key,
    aliases=None,
    default=None,
    required=True,
):
    """
    Lee un parámetro aceptando alias.

    Ejemplos de alias útiles:
    - t0 <-> t_center
    - u0 <-> u_center
    - s <-> separation
    - q <-> mass_ratio
    """
    if aliases is None:
        aliases = []

    if _param_has(params, key):
        return _param_get(params, key)

    for alias in aliases:
        if _param_has(params, alias):
            return _param_get(params, alias)

    if required:
        raise KeyError(
            f"No encontré '{key}' ni sus alias {aliases}."
        )

    return default


def parallax_suffix(fit_parallax):
    """
    Sufijo explícito para nombres de archivo.
    No se usa NoPiE.
    """
    if bool(fit_parallax):
        return "Parallax"

    return "NoParallax"


def resolve_event_coordinates(
    event_ra=None,
    event_dec=None,
):
    """
    Resuelve las coordenadas que usará el Event del fit/modelo.

    Reglas
    ------
    - Si no se pasan coordenadas, conserva el comportamiento histórico y
      usa RA_FIELD/DEC_FIELD.
    - Si se pasa una coordenada, deben pasarse ambas.
    - Las coordenadas explícitas deben ser finitas.

    En el pipeline nuevo, sim_fit debe pasar aquí las coordenadas del
    Event realmente usado durante la simulación (my_own_model.event).
    De esta forma simulación y ajuste usan exactamente la misma geometría
    para el cálculo del paralaje.
    """
    if event_ra is None and event_dec is None:
        return float(RA_FIELD), float(DEC_FIELD)

    if event_ra is None or event_dec is None:
        raise ValueError(
            "event_ra y event_dec deben pasarse juntos. "
            f"Recibí event_ra={event_ra!r}, event_dec={event_dec!r}."
        )

    ra = float(event_ra)
    dec = float(event_dec)

    if not np.isfinite(ra) or not np.isfinite(dec):
        raise ValueError(
            "Las coordenadas del Event deben ser finitas: "
            f"RA={ra}, Dec={dec}."
        )

    if dec < -90.0 or dec > 90.0:
        raise ValueError(
            f"Declinación fuera de rango: Dec={dec}."
        )

    return ra, dec


# ============================================================
# Creación del Event de pyLIMA
# ============================================================

def add_roman_telescope(
    e,
    wfirst_lc,
    path_ephemerides,
    roman_name="Roman",
):
    """
    Agrega Roman al evento si la curva no está vacía.
    """
    if not is_nonempty_lc(wfirst_lc):
        return e

    tel = telescopes.Telescope(
        name=roman_name,
        camera_filter="W149",
        lightcurve=wfirst_lc,
        lightcurve_names=[
            "time",
            "mag",
            "err_mag",
        ],
        lightcurve_units=[
            "JD",
            "mag",
            "mag",
        ],
        location="Space",
    )

    tel.spacecraft_positions = {
        "astrometry": [],
        "photometry": np.load(path_ephemerides),
    }

    e.telescopes.append(tel)

    return e


def add_rubin_telescopes(e, lsst_lcs):
    """
    Agrega telescopios Rubin al evento para las bandas no vacías.
    """
    for band in LSST_BANDS:

        lc = lsst_lcs.get(
            band,
            [],
        )

        if not is_nonempty_lc(lc):
            continue

        tel = telescopes.Telescope(
            name=band,
            camera_filter=band,
            lightcurve=lc,
            lightcurve_names=[
                "time",
                "mag",
                "err_mag",
            ],
            lightcurve_units=[
                "JD",
                "mag",
                "mag",
            ],
            location="Earth",
        )

        e.telescopes.append(tel)

    return e


def create_fit_event(
    Source,
    path_ephemerides,
    wfirst_lc,
    lsst_lcs,
    ra=None,
    dec=None,
    roman_name="Roman",
):
    """
    Construye el Event de pyLIMA.

    Puede representar:
    - Roman + Rubin
    - Roman only
    - Rubin only
    """
    wfirst_lc = as_empty_if_none(wfirst_lc)

    ra_use, dec_use = resolve_event_coordinates(
        event_ra=ra,
        event_dec=dec,
    )

    e = event.Event(
        ra=ra_use,
        dec=dec_use,
    )

    e.name = get_event_name(
        Source,
        wfirst_lc,
        lsst_lcs,
    )

    e = add_roman_telescope(
        e,
        wfirst_lc,
        path_ephemerides,
        roman_name=roman_name,
    )

    e = add_rubin_telescopes(
        e,
        lsst_lcs,
    )

    e.check_event()

    return e


def event_creation(
    Source,
    path_ephemerides,
    wfirst_lc,
    lsst_u,
    lsst_g,
    lsst_r,
    lsst_i,
    lsst_z,
    lsst_y,
    event_ra=None,
    event_dec=None,
):
    """
    Wrapper compatible con la versión vieja.
    """
    lsst_lcs = {
        "u": as_empty_if_none(lsst_u),
        "g": as_empty_if_none(lsst_g),
        "r": as_empty_if_none(lsst_r),
        "i": as_empty_if_none(lsst_i),
        "z": as_empty_if_none(lsst_z),
        "y": as_empty_if_none(lsst_y),
    }

    return create_fit_event(
        Source,
        path_ephemerides,
        as_empty_if_none(wfirst_lc),
        lsst_lcs,
        ra=event_ra,
        dec=event_dec,
        roman_name="Roman",
    )


# ============================================================
# Parámetros iniciales para el modelo de ajuste
# ============================================================

def initial_params_for_fit_model(
    event_params,
    fit_model,
    fit_parallax=True,
    fit_defaults=None,
):
    """
    Construye los parámetros iniciales para el modelo de ajuste.

    Esto permite, por ejemplo:
    - simular FSPL y ajustar PSPL
    - simular PSPL con paralaje y ajustar PSPL sin paralaje
    - simular USBL y ajustar PSPL

    Esta función pertenece al fitting. No debe usarse para graficar.
    """
    if fit_defaults is None:
        fit_defaults = {}

    fit_model = normalize_model_name(fit_model)

    t0 = float(
        get_param(
            event_params,
            "t0",
            aliases=["t_center"],
        )
    )

    u0 = float(
        get_param(
            event_params,
            "u0",
            aliases=["u_center"],
        )
    )

    tE = float(
        get_param(
            event_params,
            "tE",
        )
    )

    fit_params = {
        "t0": t0,
        "u0": u0,
        "tE": tE,
    }

    if fit_model in ["FSPL", "USBL"]:
        fit_params["rho"] = float(
            get_param(
                event_params,
                "rho",
                default=fit_defaults.get("rho", 1e-3),
                required=False,
            )
        )

    if fit_model == "USBL":

        fit_params["s"] = float(
            get_param(
                event_params,
                "separation",
                aliases=["s"],
                default=fit_defaults.get("s", 1.0),
                required=False,
            )
        )

        fit_params["q"] = float(
            get_param(
                event_params,
                "mass_ratio",
                aliases=["q"],
                default=fit_defaults.get("q", 1e-3),
                required=False,
            )
        )

        fit_params["alpha"] = float(
            get_param(
                event_params,
                "alpha",
                default=fit_defaults.get("alpha", 0.0),
                required=False,
            )
        )

    if fit_parallax:

        fit_params["piEN"] = float(
            get_param(
                event_params,
                "piEN",
                default=fit_defaults.get("piEN", 0.0),
                required=False,
            )
        )

        fit_params["piEE"] = float(
            get_param(
                event_params,
                "piEE",
                default=fit_defaults.get("piEE", 0.0),
                required=False,
            )
        )

    return fit_params


def fit_parameter_order(
    fit_model,
    fit_parallax=True,
):
    """
    Orden explícito de los parámetros físicos para pyLIMA.

    No incluye parámetros de flujo.
    """
    fit_model = normalize_model_name(fit_model)

    if fit_model == "PSPL":
        order = [
            "t0",
            "u0",
            "tE",
        ]

    elif fit_model == "FSPL":
        order = [
            "t0",
            "u0",
            "tE",
            "rho",
        ]

    elif fit_model == "USBL":
        order = [
            "t_center",
            "u_center",
            "tE",
            "rho",
            "separation",
            "mass_ratio",
            "alpha",
        ]

    else:
        raise ValueError(f"fit_model no reconocido: {fit_model}")

    if fit_parallax:
        order += [
            "piEN",
            "piEE",
        ]

    return order


def fit_guess_values(
    fit_params,
    fit_model,
    fit_parallax=True,
):
    """
    Devuelve los valores iniciales en el orden esperado por pyLIMA.
    """
    fit_model = normalize_model_name(fit_model)

    values = {
        "t0": fit_params["t0"],
        "u0": fit_params["u0"],
        "tE": fit_params["tE"],
        "t_center": fit_params["t0"],
        "u_center": fit_params["u0"],
    }

    if fit_model in ["FSPL", "USBL"]:
        values["rho"] = fit_params["rho"]

    if fit_model == "USBL":
        values["separation"] = fit_params["s"]
        values["mass_ratio"] = fit_params["q"]
        values["alpha"] = fit_params["alpha"]

    if fit_parallax:
        values["piEN"] = fit_params["piEN"]
        values["piEE"] = fit_params["piEE"]

    order = fit_parameter_order(
        fit_model,
        fit_parallax=fit_parallax,
    )

    return [
        values[p]
        for p in order
    ]


# ============================================================
# Construcción del modelo pyLIMA de ajuste
# ============================================================

def build_fit_pyLIMA_model(
    e,
    fit_model,
    fit_params,
    Origin=None,
    fit_parallax=True,
):
    """
    Construye el modelo pyLIMA que se va a ajustar.

    La construcción real del modelo está centralizada en set_model_pyLIMA.py.
    fit_params se usa acá solamente para t0_parallax cuando fit_parallax=True.
    """
    fit_model = normalize_model_name(fit_model)

    t0_parallax = None

    if fit_parallax:
        t0_parallax = fit_params["t0"]

    return build_pyLIMA_model(
        pyLIMA_event=e,
        model=fit_model,
        use_parallax=bool(fit_parallax),
        t0_parallax=t0_parallax,
        origin=Origin,
        random_origin=False,
        blend_flux_parameter="ftotal",
    )


# ============================================================
# Fitter
# ============================================================

def build_fitter(
    pyLIMAmodel,
    algo,
    mcmc_links=7000,
    mcmc_processes=36,
    de_processes=16,
    de_population_size=20,
    de_max_iteration=10000,
    de_display_progress=True,
):
    """
    Crea el objeto fitter y devuelve cuántos procesos usar.
    """
    if algo == "TRF":

        fit = TRF_fit.TRFfit(
            pyLIMAmodel,
        )

        pool_processes = None

    elif algo == "MCMC":

        fit = MCMC_fit.MCMCfit(
            pyLIMAmodel,
            MCMC_links=mcmc_links,
        )

        pool_processes = mcmc_processes

    elif algo == "DE":

        fit = DE_fit.DEfit(
            pyLIMAmodel,
            telescopes_fluxes_method="polyfit",
            DE_population_size=de_population_size,
            max_iteration=de_max_iteration,
            display_progress=de_display_progress,
        )

        pool_processes = de_processes

    else:
        raise ValueError(f"Algoritmo no reconocido: {algo}")

    return fit, pool_processes


# ============================================================
# Bounds
# ============================================================

def safe_bounds(
    center,
    frac=1.0,
    lower=None,
    upper=None,
    min_width=1e-8,
):
    """
    Construye bounds simétricos alrededor de center.

    Evita intervalos degenerados cuando center=0.
    """
    center = float(center)

    width = frac * abs(center)

    if width == 0:
        width = min_width

    lo = center - width
    hi = center + width

    if lower is not None:
        lo = max(lo, lower)

    if upper is not None:
        hi = min(hi, upper)

    if lo >= hi:
        hi = lo + min_width

    return [
        float(lo),
        float(hi),
    ]


def set_fit_bound(
    fit,
    parameter_name,
    bounds,
):
    """
    Setea bounds solo si el parámetro existe.
    """
    if parameter_name in fit.fit_parameters:
        fit.fit_parameters[parameter_name][1] = bounds


def _custom_bound_candidate_keys(parameter_name):
    """
    Permite usar nombres equivalentes para los bounds.

    Ejemplos:
        t0 o t_center
        u0 o u_center
        s o separation
        q o mass_ratio
    """
    aliases = {
        "t0": ["t0", "t_center"],
        "t_center": ["t_center", "t0"],
        "u0": ["u0", "u_center"],
        "u_center": ["u_center", "u0"],
        "separation": ["separation", "s"],
        "s": ["s", "separation"],
        "mass_ratio": ["mass_ratio", "q"],
        "q": ["q", "mass_ratio"],
    }

    return aliases.get(
        parameter_name,
        [parameter_name],
    )


def _center_for_parameter(parameter_name, fit_params):
    """
    Devuelve el centro natural para un parámetro usando fit_params.
    """
    if parameter_name in ["t0", "t_center"]:
        return fit_params["t0"]

    if parameter_name in ["u0", "u_center"]:
        return fit_params["u0"]

    if parameter_name == "tE":
        return fit_params["tE"]

    if parameter_name == "rho":
        return fit_params["rho"]

    if parameter_name in ["separation", "s"]:
        return fit_params["s"]

    if parameter_name in ["mass_ratio", "q"]:
        return fit_params["q"]

    if parameter_name == "alpha":
        return fit_params["alpha"]

    if parameter_name == "piEN":
        return fit_params["piEN"]

    if parameter_name == "piEE":
        return fit_params["piEE"]

    raise KeyError(
        f"No sé cómo obtener centro para {parameter_name}"
    )


def get_custom_bound_spec(
    fit_bounds,
    parameter_name,
):
    """
    Busca un bound custom para parameter_name aceptando alias.
    """
    if fit_bounds is None:
        return None

    for key in _custom_bound_candidate_keys(parameter_name):
        if key in fit_bounds:
            return fit_bounds[key]

    return None


def resolve_bound_spec(
    spec,
    center=None,
    parameter_name=None,
):
    """
    Convierte una especificación de bounds en [lo, hi].

    Formatos aceptados:

    1) Bounds absolutos:
        "tE": [0.1, 5000.0]

    2) Dict con lower/upper:
        "tE": {"lower": 0.1, "upper": 5000.0}

    3) Intervalo centrado con semi-ancho:
        "t0": {"type": "center_width", "half_width": 300.0}

    4) Intervalo relativo al valor inicial:
        "tE": {"type": "relative", "frac": 5.0, "lower": 0.0}
    """
    if spec is None:
        return None

    if isinstance(spec, (list, tuple, np.ndarray)):

        if len(spec) != 2:
            raise ValueError(
                f"Bounds inválidos para {parameter_name}: {spec}. "
                "Deben tener longitud 2."
            )

        lo = float(spec[0])
        hi = float(spec[1])

    elif isinstance(spec, dict):

        spec_type = spec.get(
            "type",
            "absolute",
        )

        if "bounds" in spec:

            bounds = spec["bounds"]

            if len(bounds) != 2:
                raise ValueError(
                    f"Bounds inválidos para {parameter_name}: {bounds}"
                )

            lo = float(bounds[0])
            hi = float(bounds[1])

        elif spec_type == "absolute":

            lo = float(spec["lower"])
            hi = float(spec["upper"])

        elif spec_type == "center_width":

            if center is None:
                raise ValueError(
                    f"Para center_width necesito center en {parameter_name}"
                )

            c = float(
                spec.get(
                    "center",
                    center,
                )
            )
            half_width = float(spec["half_width"])

            lo = c - half_width
            hi = c + half_width

        elif spec_type == "relative":

            if center is None:
                raise ValueError(
                    f"Para relative necesito center en {parameter_name}"
                )

            frac = float(
                spec.get(
                    "frac",
                    1.0,
                )
            )
            lower = spec.get("lower", None)
            upper = spec.get("upper", None)
            min_width = float(
                spec.get(
                    "min_width",
                    1e-8,
                )
            )

            return safe_bounds(
                center,
                frac=frac,
                lower=lower,
                upper=upper,
                min_width=min_width,
            )

        else:
            raise ValueError(
                f"Tipo de bound no reconocido para {parameter_name}: {spec_type}"
            )

    else:
        raise TypeError(
            f"Bound inválido para {parameter_name}: {spec}"
        )

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(
            f"Bounds no finitos para {parameter_name}: [{lo}, {hi}]"
        )

    if lo >= hi:
        raise ValueError(
            f"Bounds invertidos o degenerados para {parameter_name}: [{lo}, {hi}]"
        )

    return [
        float(lo),
        float(hi),
    ]


def apply_custom_bounds(
    fit,
    fit_params,
    fit_model,
    fit_parallax=True,
    fit_bounds=None,
):
    """
    Sobreescribe bounds usando un diccionario definido por el usuario.
    """
    if fit_bounds is None:
        return

    fit_model = normalize_model_name(fit_model)

    param_order = fit_parameter_order(
        fit_model,
        fit_parallax=fit_parallax,
    )

    for par in param_order:

        spec = get_custom_bound_spec(
            fit_bounds,
            par,
        )

        if spec is None:
            continue

        center = _center_for_parameter(
            par,
            fit_params,
        )

        bounds = resolve_bound_spec(
            spec,
            center=center,
            parameter_name=par,
        )

        set_fit_bound(
            fit,
            par,
            bounds,
        )


def apply_narrow_bounds(
    fit,
    fit_params,
    fit_model,
    fit_parallax=True,
    rango=1,
):
    """
    Bounds angostos, equivalentes al caso rango != 0.
    """
    fit_model = normalize_model_name(fit_model)

    if fit_model == "USBL":
        t0_key = "t_center"
        u0_key = "u_center"
    else:
        t0_key = "t0"
        u0_key = "u0"

    set_fit_bound(
        fit,
        t0_key,
        [
            fit_params["t0"] - 10,
            fit_params["t0"] + 10,
        ],
    )

    set_fit_bound(
        fit,
        u0_key,
        safe_bounds(
            fit_params["u0"],
            frac=rango,
        ),
    )

    set_fit_bound(
        fit,
        "tE",
        safe_bounds(
            fit_params["tE"],
            frac=rango,
            lower=0,
        ),
    )

    if fit_model in ["FSPL", "USBL"]:
        set_fit_bound(
            fit,
            "rho",
            safe_bounds(
                fit_params["rho"],
                frac=rango,
                lower=0,
            ),
        )

    if fit_model == "USBL":

        set_fit_bound(
            fit,
            "separation",
            safe_bounds(
                fit_params["s"],
                frac=rango,
                lower=0,
            ),
        )

        set_fit_bound(
            fit,
            "mass_ratio",
            safe_bounds(
                fit_params["q"],
                frac=rango,
                lower=1e-10,
                upper=1,
            ),
        )

        if fit_params["alpha"] == 0:
            alpha_bounds = [
                0,
                2 * np.pi,
            ]
        else:
            alpha_bounds = safe_bounds(
                fit_params["alpha"],
                frac=rango,
            )

        set_fit_bound(
            fit,
            "alpha",
            alpha_bounds,
        )

    if fit_parallax:

        set_fit_bound(
            fit,
            "piEN",
            safe_bounds(
                fit_params["piEN"],
                frac=rango,
            ),
        )

        set_fit_bound(
            fit,
            "piEE",
            safe_bounds(
                fit_params["piEE"],
                frac=rango,
            ),
        )


def apply_wide_bounds(
    fit,
    fit_params,
    fit_model,
    fit_parallax=True,
):
    """
    Bounds amplios, equivalentes al caso rango == 0.
    """
    fit_model = normalize_model_name(fit_model)

    if fit_model == "USBL":
        t0_key = "t_center"
        u0_key = "u_center"
    else:
        t0_key = "t0"
        u0_key = "u0"

    set_fit_bound(
        fit,
        t0_key,
        [
            fit_params["t0"] - 100,
            fit_params["t0"] + 100,
        ],
    )

    set_fit_bound(
        fit,
        u0_key,
        [
            -5,
            5,
        ],
    )

    tE_hi = 5 * abs(fit_params["tE"])

    if tE_hi == 0:
        tE_hi = 1e-6

    set_fit_bound(
        fit,
        "tE",
        [
            0,
            tE_hi,
        ],
    )

    if fit_model in ["FSPL", "USBL"]:

        rho = fit_params["rho"]

        set_fit_bound(
            fit,
            "rho",
            [
                0,
                rho + 2 * abs(rho) + 1e-12,
            ],
        )

    if fit_model == "USBL":

        s = fit_params["s"]

        if s < 1:
            s_bounds = [
                0,
                1,
            ]
        elif s > 1:
            s_bounds = [
                1,
                2 * s,
            ]
        else:
            s_bounds = [
                0.5,
                1.5,
            ]

        set_fit_bound(
            fit,
            "separation",
            s_bounds,
        )

        set_fit_bound(
            fit,
            "mass_ratio",
            [
                1e-10,
                1,
            ],
        )

        set_fit_bound(
            fit,
            "alpha",
            [
                0,
                2 * np.pi,
            ],
        )

    if fit_parallax:

        set_fit_bound(
            fit,
            "piEN",
            safe_bounds(
                fit_params["piEN"],
                frac=100,
            ),
        )

        set_fit_bound(
            fit,
            "piEE",
            safe_bounds(
                fit_params["piEE"],
                frac=100,
            ),
        )


def apply_fit_bounds(
    fit,
    fit_params,
    fit_model,
    rango,
    fit_parallax=True,
    fit_bounds=None,
):
    """
    Aplica bounds según rango y, opcionalmente, sobreescribe
    con bounds definidos por el usuario.
    """
    if rango != 0:

        rango_used = 1

        apply_narrow_bounds(
            fit,
            fit_params,
            fit_model,
            fit_parallax=fit_parallax,
            rango=rango_used,
        )

    else:

        rango_used = 0

        apply_wide_bounds(
            fit,
            fit_params,
            fit_model,
            fit_parallax=fit_parallax,
        )

    if fit_bounds is not None:

        apply_custom_bounds(
            fit,
            fit_params,
            fit_model,
            fit_parallax=fit_parallax,
            fit_bounds=fit_bounds,
        )

    return rango_used


# ============================================================
# Guess inicial
# ============================================================

def sample_guess_inside_bounds(
    fit,
    param_order,
    rng=None,
):
    """
    Samplea un guess inicial dentro de los bounds.
    """
    if rng is None:
        rng = np.random.default_rng()

    guess = []

    for par in param_order:

        if par not in fit.fit_parameters:
            raise KeyError(
                f"El parámetro {par} no está en fit.fit_parameters. "
                f"Parámetros disponibles: {list(fit.fit_parameters.keys())}"
            )

        interval = fit.fit_parameters[par][1]

        lo = float(interval[0])
        hi = float(interval[1])

        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(
                f"Bounds no finitos para {par}: {interval}"
            )

        if lo > hi:
            raise ValueError(
                f"Bounds invertidos para {par}: {interval}"
            )

        if lo == hi:
            guess.append(lo)
        else:
            guess.append(
                rng.uniform(
                    lo,
                    hi,
                )
            )

    return guess


def _initial_guess_candidate_keys(parameter_name):
    """
    Alias permitidos al pasar un initial_guess como diccionario.

    Esto permite usar indistintamente los nombres físicos cortos
    (t0, u0, s, q) o los nombres internos de pyLIMA para USBL
    (t_center, u_center, separation, mass_ratio).
    """
    aliases = {
        "t0": ["t0", "t_center"],
        "t_center": ["t_center", "t0"],
        "u0": ["u0", "u_center"],
        "u_center": ["u_center", "u0"],
        "separation": ["separation", "s"],
        "s": ["s", "separation"],
        "mass_ratio": ["mass_ratio", "q"],
        "q": ["q", "mass_ratio"],
    }

    return aliases.get(
        parameter_name,
        [parameter_name],
    )


def explicit_guess_values(
    initial_guess,
    param_order,
):
    """
    Convierte un guess explícito al orden esperado por pyLIMA.

    Parameters
    ----------
    initial_guess : dict or sequence
        Puede ser:

        1) Un diccionario, por ejemplo para FSPL + parallax::

            {
                "t0": ...,
                "u0": ...,
                "tE": ...,
                "rho": ...,
                "piEN": ...,
                "piEE": ...,
            }

        2) Una lista/tupla/array ya ordenada exactamente como param_order.

    param_order : sequence of str
        Orden de parámetros esperado por pyLIMA.

    Returns
    -------
    list of float
        Guess en el orden correcto.
    """
    if isinstance(initial_guess, dict):

        values = []

        for par in param_order:

            found = False

            for key in _initial_guess_candidate_keys(par):
                if key in initial_guess:
                    values.append(float(initial_guess[key]))
                    found = True
                    break

            if not found:
                raise KeyError(
                    f"Falta el parámetro '{par}' en initial_guess. "
                    f"Claves recibidas: {list(initial_guess.keys())}"
                )

    elif isinstance(initial_guess, (list, tuple, np.ndarray)):

        if len(initial_guess) != len(param_order):
            raise ValueError(
                "initial_guess tiene longitud incorrecta: "
                f"esperaba {len(param_order)} valores para {param_order}, "
                f"recibí {len(initial_guess)}."
            )

        values = [
            float(value)
            for value in initial_guess
        ]

    else:
        raise TypeError(
            "initial_guess debe ser None, 'truth'/'fit_params', "
            "un dict o una secuencia numérica. "
            f"Recibí {type(initial_guess).__name__}."
        )

    for par, value in zip(param_order, values):

        if not np.isfinite(value):
            raise ValueError(
                f"Guess inicial no finito para {par}: {value}"
            )

    return values


def validate_initial_guess_inside_bounds(
    fit,
    param_order,
    guess,
    atol=1e-12,
):
    """
    Verifica que un guess explícito esté dentro de los bounds del fitter.

    Para un guess explícito preferimos fallar con un mensaje claro antes que
    dejar que pyLIMA/TRF modifique silenciosamente el punto inicial.
    """
    if len(guess) != len(param_order):
        raise ValueError(
            "guess y param_order tienen longitudes distintas: "
            f"{len(guess)} != {len(param_order)}"
        )

    for par, value in zip(param_order, guess):

        if par not in fit.fit_parameters:
            raise KeyError(
                f"El parámetro {par} no está en fit.fit_parameters. "
                f"Parámetros disponibles: {list(fit.fit_parameters.keys())}"
            )

        interval = fit.fit_parameters[par][1]
        lo = float(interval[0])
        hi = float(interval[1])
        value = float(value)

        if value < lo - atol or value > hi + atol:
            raise ValueError(
                f"Guess inicial fuera de bounds para {par}: "
                f"value={value}, bounds=[{lo}, {hi}]."
            )


def resolve_initial_guess(
    fit,
    fit_params,
    fit_model,
    fit_parallax,
    param_order,
    initial_guess=None,
    rng=None,
):
    """
    Resuelve qué guess inicial usar.

    Modos soportados
    ----------------
    initial_guess is None
        Mantiene el comportamiento histórico: samplea uniformemente dentro
        de los bounds.

    initial_guess == "truth" o "fit_params"
        Usa los valores contenidos en fit_params. En el pipeline de
        simulaciones, fit_params se construye a partir de event_params,
        por lo que este modo corresponde a usar los parámetros verdaderos
        disponibles para el modelo de ajuste.

    initial_guess is dict
        Usa exactamente los valores provistos por parámetro.

    initial_guess is list/tuple/ndarray
        Usa exactamente esa secuencia, en el orden param_order.

    Returns
    -------
    guess : list of float
    source : str
        Etiqueta que describe el origen del guess.
    """
    if initial_guess is None:

        guess = sample_guess_inside_bounds(
            fit,
            param_order,
            rng=rng,
        )

        source = "random_inside_bounds"

        return guess, source

    if isinstance(initial_guess, str):

        mode = initial_guess.strip().lower()

        if mode not in {
            "truth",
            "fit_params",
        }:
            raise ValueError(
                "String initial_guess no reconocido. "
                "Usá 'truth', 'fit_params', None, un dict o una secuencia. "
                f"Recibido: {initial_guess!r}"
            )

        guess = fit_guess_values(
            fit_params,
            fit_model,
            fit_parallax=fit_parallax,
        )

        source = "fit_params"

    else:

        guess = explicit_guess_values(
            initial_guess,
            param_order,
        )

        source = "explicit"

    validate_initial_guess_inside_bounds(
        fit,
        param_order,
        guess,
    )

    return guess, source


# ============================================================
# Opciones y diagnósticos del optimizador TRF
# ============================================================

TRF_DEFAULT_OPTIMIZER_OPTIONS = {
    "xtol": 1e-10,
    "ftol": 1e-10,
    "gtol": 1e-10,
    "max_nfev": 50000,
}

TRF_ALLOWED_OPTIMIZER_OPTIONS = (
    set(TRF_DEFAULT_OPTIMIZER_OPTIONS)
    | {"x_scale"}
)


def normalize_trf_optimizer_options(optimizer_options=None):
    """
    Validate and normalize scipy.optimize.least_squares options used by
    pyLIMA TRF fits.

    Important
    ---------
    x_scale is intentionally NOT part of TRF_DEFAULT_OPTIMIZER_OPTIONS.

    If x_scale is absent, pyLIMA keeps its native scaling:

        10**floor(log10(abs(guess))) + 1

    Only an explicitly supplied x_scale overrides that behavior.
    """

    options = dict(TRF_DEFAULT_OPTIMIZER_OPTIONS)

    if optimizer_options is None:
        return options

    if not isinstance(optimizer_options, dict):
        raise TypeError(
            "optimizer_options must be a dict or None."
        )

    allowed = {
        "xtol",
        "ftol",
        "gtol",
        "max_nfev",
        "x_scale",
    }

    unknown = set(optimizer_options) - allowed

    if unknown:
        raise ValueError(
            "Unknown TRF optimizer option(s): "
            + ", ".join(sorted(unknown))
        )

    options.update(optimizer_options)

    # --------------------------------------------------------
    # Tolerances
    # --------------------------------------------------------

    for key in ("xtol", "ftol", "gtol"):

        value = options[key]

        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{key} must be a positive finite float."
            )

        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"{key} must be a positive finite float."
            )

        options[key] = value

    # --------------------------------------------------------
    # Maximum function evaluations
    # --------------------------------------------------------

    value = options["max_nfev"]

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(
            "max_nfev must be a positive integer."
        )

    options["max_nfev"] = int(value)

    # --------------------------------------------------------
    # scipy least_squares x_scale
    # --------------------------------------------------------

    if "x_scale" in optimizer_options:

        value = optimizer_options["x_scale"]

        if isinstance(value, str):

            if value.lower() != "jac":
                raise ValueError(
                    "x_scale string value must be 'jac'."
                )

            options["x_scale"] = "jac"

        else:

            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    "x_scale must be 'jac' or a positive scalar."
                )

            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "x_scale must be 'jac' or a positive scalar."
                )

            options["x_scale"] = value

    return options



@contextmanager
def temporary_trf_least_squares_options(
    optimizer_options,
):
    """
    Sobreescribe temporalmente las opciones que TRFfit.fit() pasa a
    scipy.optimize.least_squares.

    No modifica el paquete pyLIMA en disco.

    Esto es seguro para nuestro pipeline porque cada worker ejecuta un
    único fit a la vez dentro de su propio proceso.
    """

    options = normalize_trf_optimizer_options(
        optimizer_options
    )

    original = (
        TRF_fit.scipy.optimize.least_squares
    )

    def wrapped_least_squares(
        *args,
        **kwargs,
    ):
        kwargs.update(options)

        return original(
            *args,
            **kwargs,
        )

    TRF_fit.scipy.optimize.least_squares = (
        wrapped_least_squares
    )

    try:
        yield options

    finally:
        TRF_fit.scipy.optimize.least_squares = (
            original
        )


def optimizer_diagnostics_from_fit(
    fit,
):
    """
    Extrae diagnósticos escalares del scipy OptimizeResult guardado por
    pyLIMA en fit.fit_results['fit_object'].
    """

    out = {}

    fit_results = getattr(
        fit,
        "fit_results",
        {},
    )

    if not isinstance(
        fit_results,
        dict,
    ):
        return out

    fit_object = fit_results.get(
        "fit_object",
        None,
    )

    if fit_object is None:
        return out

    def get_value(key, default=None):
        try:
            return fit_object[key]
        except Exception:
            return getattr(
                fit_object,
                key,
                default,
            )

    for key in [
        "nfev",
        "njev",
        "status",
    ]:
        value = get_value(
            key,
            None,
        )

        if value is not None:
            try:
                value = int(value)
            except Exception:
                pass

            out[
                f"optimizer_{key}"
            ] = value

    success = get_value(
        "success",
        None,
    )

    if success is not None:
        out["optimizer_success"] = bool(
            success
        )

    message = get_value(
        "message",
        None,
    )

    if message is not None:
        out["optimizer_message"] = str(
            message
        )

    for key in [
        "optimality",
        "cost",
    ]:
        value = get_value(
            key,
            None,
        )

        if value is not None:
            try:
                value = float(value)
            except Exception:
                pass

            out[
                f"optimizer_{key}"
            ] = value

    active_mask = get_value(
        "active_mask",
        None,
    )

    if active_mask is not None:

        try:
            active_mask_array = np.asarray(
                active_mask,
                dtype=int,
            )

            out[
                "optimizer_n_active_bounds"
            ] = int(
                np.sum(
                    active_mask_array != 0
                )
            )

            out[
                "optimizer_active_mask"
            ] = repr(
                active_mask_array.tolist()
            )

        except Exception:
            out[
                "optimizer_active_mask"
            ] = repr(
                active_mask
            )

    effective_options = getattr(
        fit,
        "_optimizer_options_effective",
        None,
    )

    if isinstance(
        effective_options,
        dict,
    ):

        for key in [
            "xtol",
            "ftol",
            "gtol",
            "max_nfev",
            "x_scale",
        ]:
            if key in effective_options:
                out[
                    f"optimizer_{key}"
                ] = effective_options[key]

    source = getattr(
        fit,
        "_optimizer_options_source",
        None,
    )

    if source is not None:
        out[
            "optimizer_options_source"
        ] = str(source)

    try:
        out[
            "optimizer_jacobian_flag"
        ] = str(
            fit.model.Jacobian_flag
        )
    except Exception:
        pass

    return out



# ============================================================
# Ejecución y guardado
# ============================================================

def run_fit(
    fit,
    algo,
    pool_processes=None,
    optimizer_options=None,
):
    """
    Ejecuta el fit.

    Para TRF:
        optimizer_options permite sobreescribir xtol, ftol, gtol,
        max_nfev y x_scale sin modificar pyLIMA.

    Si optimizer_options=None:
        conserva exactamente el baseline de pyLIMA
        (1e-10, 1e-10, 1e-10, 50000).

    Para MCMC/DE:
        optimizer_options debe ser None.
    """

    algo_use = str(algo).upper()

    if algo_use == "TRF":

        effective_options = (
            normalize_trf_optimizer_options(
                optimizer_options
            )
        )

        fit._optimizer_options_effective = (
            effective_options
        )

        fit._optimizer_options_source = (
            "pyLIMA_default"
            if optimizer_options is None
            else "explicit"
        )

        if optimizer_options is None:

            # No interceptar scipy: reproduce literalmente pyLIMA.
            fit.fit()

        else:

            with temporary_trf_least_squares_options(
                effective_options
            ):
                fit.fit()

        return fit

    if optimizer_options is not None:
        raise ValueError(
            "optimizer_options actualmente solo se soporta para TRF. "
            f"algo={algo!r}."
        )

    if pool_processes is None:
        fit.fit()
        return fit

    with mul.Pool(
        processes=pool_processes
    ) as pool:

        fit.fit(
            computational_pool=pool,
        )

    return fit


def finalize_fit_results(
    fit,
    event_params,
    algo,
    rango_used,
    event_name,
    sim_model,
    fit_model,
    fit_parallax,
    fit_bounds=None,
):
    """
    Agrega metadatos al fit_results.
    """
    fit.fit_results["true_params"] = event_params
    fit.fit_results["sim_model"] = normalize_model_name(sim_model)
    fit.fit_results["fit_model"] = normalize_model_name(fit_model)
    fit.fit_results["fit_parallax"] = bool(fit_parallax)
    fit.fit_results["rango"] = rango_used
    fit.fit_results["method"] = algo
    fit.fit_results["name"] = event_name

    fit.fit_results["ln_likelihood"] = fit.likelihood_photometry(
        fit.fit_results["best_model"]
    )
    fit.fit_results["fit_bounds"] = fit_bounds

    # scipy/pyLIMA optimizer diagnostics.
    fit.fit_results.update(
        optimizer_diagnostics_from_fit(
            fit
        )
    )

    return fit.fit_results


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
        durante la simulación. Si ambas son None se conserva el campo fijo
        histórico RA_FIELD/DEC_FIELD por compatibilidad.
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
    )

    pyLIMAmodel = build_fit_pyLIMA_model(
        e,
        fit_model_base,
        fit_params,
        Origin=Origin,
        fit_parallax=fit_parallax,
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
    )

    param_order = fit_parameter_order(
        fit_model_base,
        fit_parallax=fit_parallax,
    )

    initial_guess_used, initial_guess_source = resolve_initial_guess(
        fit,
        fit_params,
        fit_model_base,
        fit_parallax,
        param_order,
        initial_guess=initial_guess,
        rng=rng,
    )

    fit.model_parameters_guess = initial_guess_used

    print(
        "[fit_rubin_roman] Initial guess "
        f"source={initial_guess_source}, "
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
        random_origin=False,
        blend_flux_parameter="ftotal",
    )

    return pyLIMAmodel
