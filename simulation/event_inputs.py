"""Catalogue row -> event_params dictionary simulated by sim_event.

build_event_params_from_custom_system validates and normalizes one row of
the materialized catalogue (true parameters, RA/Dec, magnitudes,
materialized fluxes, blend ratios, caustic origin). Nothing is sampled.
"""

import numpy as np
import pandas as pd
from collections.abc import Mapping


def build_event_params_from_custom_system(
    custom_system,
    model,
    use_roman=True,
    use_rubin=True,
    rubin_pointing_mode="source",
):
    """
    Construye directamente ``event_params`` sin seleccionar ni emparejar
    estrellas de catálogos.

    Parameters
    ----------
    custom_system : Mapping, pandas.Series or one-row pandas.DataFrame
        Parámetros verdaderos del evento. Para un FSPL Rubin-only típico:

        {
            "t0": ...,
            "u0": ...,
            "tE": ...,
            "rho": ...,
            "piEN": ...,
            "piEE": ...,
            "u": ...,
            "g": ...,
            "r": ...,
            "i": ...,
            "z": ...,
            "Y": ...,
            "ra": ...,
            "dec": ...,
        }

        Se conservan todas las claves adicionales como metadatos.

    model : str
        Modelo verdadero usado por ``sim_event``.

    use_roman, use_rubin : bool
        Determinan qué magnitudes son obligatorias.

    rubin_pointing_mode : str
        Si vale ``"source"``, se requieren coordenadas de la fuente.

    Returns
    -------
    dict
        Copia validada y normalizada de ``custom_system``.

    Notes
    -----
    ``tE`` se usa directamente. No se reconstruye a partir de
    ``thetaE`` ni ``mu_rel``.
    """

    if custom_system is None:
        raise ValueError(
            "catalog_mode='custom_system' requiere custom_system."
        )

    if isinstance(custom_system, pd.DataFrame):
        if len(custom_system) != 1:
            raise ValueError(
                "custom_system como DataFrame debe tener exactamente una fila."
            )
        event_params = custom_system.iloc[0].to_dict()

    elif isinstance(custom_system, pd.Series):
        event_params = custom_system.to_dict()

    elif isinstance(custom_system, Mapping):
        event_params = dict(custom_system)

    else:
        try:
            event_params = dict(custom_system)
        except Exception as error:
            raise TypeError(
                "custom_system debe ser un mapping, pandas.Series "
                "o DataFrame de una fila."
            ) from error

    # ------------------------------------------------------------
    # Normalización de nombres
    # ------------------------------------------------------------

    if "Y" not in event_params and "y" in event_params:
        event_params["Y"] = event_params["y"]

    if "y" not in event_params and "Y" in event_params:
        event_params["y"] = event_params["Y"]

    if "W149" not in event_params and "Y" in event_params:
        event_params["W149"] = event_params["Y"]

    if "maf_ra" not in event_params and "ra" in event_params:
        event_params["maf_ra"] = event_params["ra"]

    if "maf_dec" not in event_params and "dec" in event_params:
        event_params["maf_dec"] = event_params["dec"]

    # El modelo actual espera las componentes aun cuando sean nulas.
    event_params.setdefault("piEN", 0.0)
    event_params.setdefault("piEE", 0.0)

    # ------------------------------------------------------------
    # Parámetros obligatorios
    # ------------------------------------------------------------

    required = [
        "t0",
        "u0",
        "tE",
        "piEN",
        "piEE",
    ]

    model_upper = str(model).upper()

    if "FSPL" in model_upper:
        required.append("rho")

    if use_roman:
        required.append("W149")

    if use_rubin:
        required.extend(["u", "g", "r", "i", "z", "Y"])

    # Every event must carry its sky position (Rubin pointing and the
    # Event RA/Dec used for parallax), also for Roman-only runs.
    required.extend(["maf_ra", "maf_dec"])

    missing = [
        key
        for key in required
        if key not in event_params
    ]

    if missing:
        raise KeyError(
            "Faltan parámetros obligatorios en custom_system: "
            f"{missing}"
        )

    # ------------------------------------------------------------
    # Conversión y validación numérica
    # ------------------------------------------------------------

    numerical_keys = set(required)

    optional_numerical_keys = {
        "ra",
        "dec",
        "maf_ra",
        "maf_dec",
        "mass",
        "lens_mass",
        "thetaE",
        "thetaE_mas",
        "mu_rel",
        "D_L",
        "D_S",
        "D_L_kpc",
        "D_S_kpc",
        "gall",
        "galb",
        "alpha",
        "xi",
        "s",
        "q",
    }

    numerical_keys.update(
        key
        for key in optional_numerical_keys
        if (
            key in event_params
            and event_params[key] is not None
        )
    )

    for key in numerical_keys:
        try:
            value = float(event_params[key])
        except Exception as error:
            raise TypeError(
                f"custom_system[{key!r}] debe ser numérico."
            ) from error

        if not np.isfinite(value):
            raise ValueError(
                f"custom_system[{key!r}] no es finito: {value}"
            )

        event_params[key] = value

    if event_params["tE"] <= 0.0:
        raise ValueError("custom_system['tE'] debe ser positivo.")

    if "FSPL" in model_upper and event_params["rho"] <= 0.0:
        raise ValueError("custom_system['rho'] debe ser positivo.")

    # Metadatos que dejan explícito el origen directo.
    event_params["catalog_mode"] = "custom_system"
    event_params["custom_system_direct"] = True

    return event_params
