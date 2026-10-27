"""Fit-space coordinates for pyLIMA fits (pyLIMA fancy_parameters).

roman_rubin always describes a fit in physical parameters (t0, u0, tE,
rho, piEN, piEE, ...): fit_parameter_order, fit_params, custom bounds and
initial guesses are physical. A fit can optionally let the optimizer work
in transformed coordinates. The transformation is done by pyLIMA itself
through ``fancy_parameters`` (see pyLIMA examples/pyLIMA_example_2.py);
this module only maps between the physical description and the fit space.

Modes
-----
* "physical"   : no transformation (default, historical behaviour).
* "log_te_rho" : tE -> log_tE, rho -> log_rho (log10).

Results convention
------------------
``fit.fit_results["best_model"]`` and ``["covariance_matrix"]`` stay as
pyLIMA returns them, in fit space, so that they remain consistent with the
pyLIMA model (likelihoods, Fisher information). ``physical_fit_results``
adds ``best_model_physical`` and ``covariance_matrix_physical``.
"""

import numpy as np
from pyLIMA.models import pyLIMA_fancy_parameters

DEFAULT_FIT_COORDINATES = "physical"

FIT_COORDINATE_MODES = {
    "physical": {},
    "log_te_rho": {
        "tE": "log_tE",
        "rho": "log_rho",
    },
}

# d(physical)/d(fit) as a function of the physical value.
_LN10 = np.log(10.0)
_DPHYS_DFIT = {
    "log_tE": lambda x: _LN10 * x,
    "log_rho": lambda x: _LN10 * x,
}


def normalize_fit_coordinates(fit_coordinates=None):
    """Return a validated fit-coordinate mode name."""
    if fit_coordinates is None:
        return DEFAULT_FIT_COORDINATES

    mode = str(fit_coordinates).strip().lower()

    if mode not in FIT_COORDINATE_MODES:
        raise ValueError(
            f"fit_coordinates no reconocido: {fit_coordinates!r}. "
            f"Opciones: {sorted(FIT_COORDINATE_MODES)}"
        )

    return mode


def fancy_parameters_for(fit_coordinates, physical_parameter_names):
    """
    pyLIMA fancy_parameters object for this mode, or None for "physical".

    Only parameters present in the fitted model are transformed (e.g. a
    PSPL fit has no rho). The fancy_boundaries are pyLIMA's defaults; they
    are always replaced by per-event bounds in set_fit_bound.
    """
    mode = normalize_fit_coordinates(fit_coordinates)
    names = set(physical_parameter_names)

    mapping = {
        physical: fancy
        for physical, fancy in FIT_COORDINATE_MODES[mode].items()
        if physical in names
    }

    if not mapping:
        return None

    standard = pyLIMA_fancy_parameters.StandardFancyParameters()

    return pyLIMA_fancy_parameters.StandardFancyParameters(
        fancy_parameters=mapping,
        fancy_boundaries={
            fancy: standard.fancy_boundaries[fancy]
            for fancy in mapping.values()
        },
    )


def model_fancy_mapping(pyLIMA_model):
    """{physical_name: fit_name} actually used by a pyLIMA model."""
    fancy = getattr(pyLIMA_model, "fancy_parameters", None)

    if fancy is None:
        return {}

    keys = pyLIMA_model.model_dictionnary

    return {
        physical: name
        for physical, name in fancy.fancy_parameters.items()
        if name in keys
    }


def fit_key(pyLIMA_model, physical_name):
    """Name of a physical parameter inside fit.fit_parameters."""
    return model_fancy_mapping(pyLIMA_model).get(
        physical_name,
        physical_name,
    )


def to_fit_value(pyLIMA_model, physical_name, value):
    """Physical value -> fit-space value."""
    name = fit_key(pyLIMA_model, physical_name)

    if name == physical_name:
        return float(value)

    transform = getattr(pyLIMA_model.fancy_parameters, name)

    return float(transform({physical_name: float(value)}))


def to_physical_value(pyLIMA_model, fit_name, value):
    """Fit-space value -> physical value."""
    for physical, name in model_fancy_mapping(pyLIMA_model).items():
        if name == fit_name:
            transform = getattr(pyLIMA_model.fancy_parameters, physical)
            return float(transform({fit_name: float(value)}))

    return float(value)


def bounds_to_fit_space(pyLIMA_model, physical_name, bounds):
    """
    Physical [lo, hi] -> fit-space [lo, hi].

    The supported transformations are monotonic increasing, so the
    interval maps endpoint by endpoint. A non-finite or non-increasing
    result (e.g. lo <= 0 for a log) is an error, never silently clipped.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = to_fit_value(pyLIMA_model, physical_name, bounds[0])
        hi = to_fit_value(pyLIMA_model, physical_name, bounds[1])

    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(
            f"Bounds de {physical_name} inválidos en espacio de ajuste "
            f"{fit_key(pyLIMA_model, physical_name)}: "
            f"physical={list(bounds)}, fit=[{lo}, {hi}]."
        )

    return [lo, hi]


def guess_to_fit_space(pyLIMA_model, param_order, guess):
    """Physical guess (ordered as param_order) -> fit-space guess."""
    return [
        to_fit_value(pyLIMA_model, name, value)
        for name, value in zip(param_order, guess)
    ]


def guess_to_physical(pyLIMA_model, param_order, guess):
    """Fit-space guess (ordered as param_order) -> physical guess."""
    return [
        to_physical_value(
            pyLIMA_model,
            fit_key(pyLIMA_model, name),
            value,
        )
        for name, value in zip(param_order, guess)
    ]


def require_transformed_bounds(fit, bounded_physical_names):
    """
    Every transformed parameter must have received per-event bounds;
    otherwise pyLIMA's fixed fancy_boundaries would be used silently.
    """
    mapping = model_fancy_mapping(fit.model)
    missing = sorted(
        physical
        for physical in mapping
        if physical not in set(bounded_physical_names)
    )

    if missing:
        raise RuntimeError(
            "Parámetros transformados sin bounds por evento: "
            f"{missing} (fit keys: {[mapping[m] for m in missing]})."
        )


def stage_physical_bound(fit, physical_name, bounds):
    """
    Store physical bounds of a transformed parameter on the fitter.

    Bounds are assembled in physical space (defaults, then custom bounds
    override them) and converted only once, by write_transformed_bounds.
    """
    staged = fit.__dict__.setdefault("_physical_bounds_staged", {})
    staged[physical_name] = [float(bounds[0]), float(bounds[1])]


def write_transformed_bounds(fit):
    """
    Convert the final physical bounds of every transformed parameter and
    write them in fit.fit_parameters. No-op for physical coordinates.
    """
    mapping = model_fancy_mapping(fit.model)

    if not mapping:
        return

    staged = getattr(fit, "_physical_bounds_staged", {})
    require_transformed_bounds(fit, staged.keys())

    for physical, name in mapping.items():
        fit.fit_parameters[name][1] = bounds_to_fit_space(
            fit.model,
            physical,
            staged[physical],
        )


def physical_fit_results(fit):
    """
    Physical view of a finished fit.

    Returns best_model_physical (same order and length as
    fit.fit_parameters), covariance_matrix_physical = G C G^T with
    G = diag(d physical / d fit), and the parameter names in both spaces.
    """
    model = fit.model
    fit_names = list(fit.fit_parameters.keys())
    inverse = {
        name: physical
        for physical, name in model_fancy_mapping(model).items()
    }

    z = np.asarray(fit.fit_results["best_model"], dtype=float)
    x = np.array(
        [
            to_physical_value(model, name, value)
            for name, value in zip(fit_names, z)
        ],
        dtype=float,
    )

    G = np.ones(len(x), dtype=float)

    for index, name in enumerate(fit_names):
        if name in inverse:
            G[index] = _DPHYS_DFIT[name](x[index])

    covariance = fit.fit_results.get("covariance_matrix", None)
    covariance_physical = None

    if covariance is not None:
        covariance = np.asarray(covariance, dtype=float)
        covariance_physical = (
            G[:, np.newaxis] * covariance * G[np.newaxis, :]
        )

    return {
        "best_model_physical": x,
        "covariance_matrix_physical": covariance_physical,
        "fit_parameter_names": fit_names,
        "physical_parameter_names": [
            inverse.get(name, name) for name in fit_names
        ],
    }
