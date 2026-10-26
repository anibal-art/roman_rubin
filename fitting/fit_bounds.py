"""Bounds of the fitted parameters.

apply_fit_bounds chooses custom, narrow or wide bounds; safe_bounds keeps
every interval valid; custom bound specs accept parameter aliases.
"""

import numpy as np
from set_model_pyLIMA import normalize_model_name
from fitting.fit_parameters import (
    fit_parameter_order,
    fit_trajectory_parameter_names,
)


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
    Origin=None,
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
        Origin=Origin,
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
    Origin=None,
):
    """
    Bounds angostos, equivalentes al caso rango != 0.
    """
    fit_model = normalize_model_name(fit_model)

    t0_key, u0_key = fit_trajectory_parameter_names(
        fit_model,
        Origin=Origin,
    )

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
    Origin=None,
):
    """
    Bounds amplios, equivalentes al caso rango == 0.
    """
    fit_model = normalize_model_name(fit_model)

    t0_key, u0_key = fit_trajectory_parameter_names(
        fit_model,
        Origin=Origin,
    )

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
    Origin=None,
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
            Origin=Origin,
        )

    else:

        rango_used = 0

        apply_wide_bounds(
            fit,
            fit_params,
            fit_model,
            fit_parallax=fit_parallax,
            Origin=Origin,
        )

    if fit_bounds is not None:

        apply_custom_bounds(
            fit,
            fit_params,
            fit_model,
            fit_parallax=fit_parallax,
            fit_bounds=fit_bounds,
            Origin=Origin,
        )

    return rango_used
