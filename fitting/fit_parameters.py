"""Parameters of the fit model.

* get_param: read a parameter accepting its aliases (t0/t_center, ...).
* fit_trajectory_parameter_names / fit_parameter_order: pyLIMA order of the
  fitted parameters (USBL uses t_center/u_center with the caustic origin).
* initial_params_for_fit_model / fit_guess_values: parameter values passed
  to the fit model.
* build_fit_pyLIMA_model: pyLIMA model of the fit.
* parallax_suffix: "parallax" / "noparallax" for file names.
"""

from set_model_pyLIMA import build_pyLIMA_model
from set_model_pyLIMA import normalize_model_name


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


def fit_trajectory_parameter_names(
    fit_model,
    Origin=None,
):
    """
    Return the trajectory parameter names used by pyLIMA.

    PSPL/FSPL use t0/u0.

    For USBL, pyLIMA uses t0/u0 only for center_of_mass.
    For any shifted origin (central_caustic, second_caustic,
    third_caustic, primary, secondary, etc.) pyLIMA replaces them
    with t_center/u_center.

    Origin=None preserves the historical USBL fit convention
    t_center/u_center. The production simulation-fit path passes
    the explicit realized Origin.
    """
    fit_model = normalize_model_name(fit_model)

    if fit_model != "USBL":
        return "t0", "u0"

    if Origin is None:
        return "t_center", "u_center"

    if isinstance(Origin, str):
        origin_name = Origin
    else:
        origin_name = Origin[0]

    if origin_name == "center_of_mass":
        return "t0", "u0"

    return "t_center", "u_center"


def initial_params_for_fit_model(
    event_params,
    fit_model,
    fit_parallax=True,
    fit_defaults=None,
    Origin=None,
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

    time_key, impact_key = fit_trajectory_parameter_names(
        fit_model,
        Origin=Origin,
    )

    time_alias = (
        "t0"
        if time_key == "t_center"
        else "t_center"
    )

    impact_alias = (
        "u0"
        if impact_key == "u_center"
        else "u_center"
    )

    # Keep t0/u0 as internal generic storage names in fit_params.
    # Their VALUES are the active trajectory coordinates used by
    # the fit:
    #
    #   center_of_mass -> t0, u0
    #   shifted USBL origin -> t_center, u_center
    t0 = float(
        get_param(
            event_params,
            time_key,
            aliases=[time_alias],
        )
    )

    u0 = float(
        get_param(
            event_params,
            impact_key,
            aliases=[impact_alias],
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
    Origin=None,
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

        time_key, impact_key = fit_trajectory_parameter_names(
            fit_model,
            Origin=Origin,
        )

        order = [
            time_key,
            impact_key,
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
    Origin=None,
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
        Origin=Origin,
    )

    return [
        values[p]
        for p in order
    ]


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
        blend_flux_parameter="ftotal",
    )
