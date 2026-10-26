"""Initial guess of the fit.

resolve_initial_guess: explicit guess values, values from the event, or a
sample inside the bounds; the guess is validated against the bounds.
"""

import numpy as np
from fitting.fit_bounds import (
    _custom_bound_candidate_keys,
)
from fitting.fit_parameters import (
    fit_guess_values,
)


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

            for key in _custom_bound_candidate_keys(par):
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
    Origin=None,
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
            Origin=Origin,
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
