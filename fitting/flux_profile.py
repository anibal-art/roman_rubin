"""Telescope-flux handling of a pyLIMA fit.

Modes (``flux_method``)
-----------------------
* "fit" (default): pyLIMA fits fsource_<tel>, ftotal_<tel> as nonlinear
  parameters together with the microlensing parameters (historical
  behaviour of roman_rubin TRF fits).
* "bounded_profile": the fluxes are linear nuisance parameters profiled out
  at every trial point. For blend_flux_parameter="ftotal",

      flux = fsource * (A - 1) + ftotal,
      0 <= fsource <= max(flux),  0 <= ftotal <= max(flux),

  is solved exactly (weighted, box constrained). The optimizer then works
  only in the microlensing parameters. Only TRF is supported.

The profile is installed on the pyLIMA model instance (not on the pyLIMA
classes), so no other model in the process is affected.
"""

import types

import numpy as np

FLUX_METHODS = ("fit", "bounded_profile")
DEFAULT_FLUX_METHOD = "fit"


def normalize_flux_method(flux_method=None):
    """Return a validated flux method name."""
    if flux_method is None:
        return DEFAULT_FLUX_METHOD

    method = str(flux_method).strip().lower()

    if method not in FLUX_METHODS:
        raise ValueError(
            f"flux_method no reconocido: {flux_method!r}. "
            f"Opciones: {list(FLUX_METHODS)}"
        )

    return method


def bounded_profile_ftotal_2d(magnification, flux, err_flux):
    """
    Exact weighted box-constrained least squares for

        flux = fsource * (A - 1) + ftotal,
        0 <= fsource <= max(flux),  0 <= ftotal <= max(flux).

    The objective is a convex quadratic in two variables, so the minimum
    is the unconstrained solution when feasible, otherwise on an edge of
    the box. Every edge is solved in closed form and clipped to the box
    (corners included); the lowest chi2 candidate is returned.
    """
    A = np.asarray(magnification, dtype=float).reshape(-1)
    y = np.asarray(flux, dtype=float).reshape(-1)
    sigma = np.asarray(err_flux, dtype=float).reshape(-1)

    if not (len(A) == len(y) == len(sigma)):
        raise RuntimeError("bounded profile: incompatible array lengths.")

    good = (
        np.isfinite(A)
        & np.isfinite(y)
        & np.isfinite(sigma)
        & (sigma > 0.0)
    )

    if not np.all(good):
        raise RuntimeError("bounded profile: non-finite photometric data.")

    x = A - 1.0
    w = 1.0 / (sigma * sigma)
    fmax = float(np.max(y))

    if not np.isfinite(fmax) or fmax < 0.0:
        raise RuntimeError("bounded profile: invalid max(flux).")

    sw = float(np.sum(w))
    sx = float(np.sum(w * x))
    sxx = float(np.sum(w * x * x))
    sy = float(np.sum(w * y))
    sxy = float(np.sum(w * x * y))

    if sw <= 0.0:
        raise RuntimeError("bounded profile: non-positive total weight.")

    candidates = []

    def _clip(value):
        return float(np.clip(value, 0.0, fmax))

    def _add(fs, ft, source):
        fs = _clip(fs)
        ft = _clip(ft)
        residual = (y - fs * x - ft) / sigma
        candidates.append(
            (float(np.dot(residual, residual)), fs, ft, source)
        )

    # Unconstrained weighted least-squares interior.
    determinant = sxx * sw - sx * sx
    determinant_scale = max(abs(sxx * sw), abs(sx * sx), 1.0)

    if abs(determinant) > 1.0e-14 * determinant_scale:
        fs_free = (sxy * sw - sy * sx) / determinant
        ft_free = (sxx * sy - sx * sxy) / determinant

        if 0.0 <= fs_free <= fmax and 0.0 <= ft_free <= fmax:
            _add(fs_free, ft_free, "interior")

    # Edges fsource = 0 and fsource = fmax.
    for fs, label in ((0.0, "fsource_lower"), (fmax, "fsource_upper")):
        _add(fs, (sy - fs * sx) / sw, label)

    # Edges ftotal = 0 and ftotal = fmax.
    if sxx > 0.0:
        for ft, label in ((0.0, "ftotal_lower"), (fmax, "ftotal_upper")):
            _add((sxy - ft * sx) / sxx, ft, label)
    else:
        # Degenerate A ~= constant case.
        _add(0.0, sy / sw, "degenerate")

    chi2, fs, ft, source = min(candidates, key=lambda item: item[0])

    bound_tol = 1.0e-10 * max(1.0, fmax)

    def _active(value):
        if abs(value) <= bound_tol:
            return "lower"
        if abs(value - fmax) <= bound_tol:
            return "upper"
        return "free"

    return {
        "fsource": float(fs),
        "ftotal": float(ft),
        "fblend": float(ft - fs),
        "chi2": float(chi2),
        "fmax": float(fmax),
        "solution_type": str(source),
        "fsource_active": _active(fs),
        "ftotal_active": _active(ft),
    }


def _has_explicit_flux(pyLIMA_parameters, telescope_name):
    try:
        return pyLIMA_parameters["fsource_" + telescope_name] is not None
    except (TypeError, KeyError, IndexError):
        return False


def _derive_telescope_flux_bounded_profile(
    self,
    telescope,
    pyLIMA_parameters,
    magnification,
):
    """
    Instance replacement of MLmodel.derive_telescope_flux.

    Explicit fluxes (e.g. when simulating or evaluating a full parameter
    vector) keep pyLIMA's behaviour; only implicit fluxes are profiled.
    """
    if _has_explicit_flux(pyLIMA_parameters, telescope.name):
        return type(self).derive_telescope_flux(
            self,
            telescope,
            pyLIMA_parameters,
            magnification,
        )

    lightcurve = telescope.lightcurve
    result = bounded_profile_ftotal_2d(
        magnification,
        lightcurve["flux"].value,
        lightcurve["err_flux"].value,
    )

    fs = result["fsource"]
    fb = result["fblend"]

    pyLIMA_parameters["fsource_" + telescope.name] = fs
    pyLIMA_parameters["fblend_" + telescope.name] = fb
    pyLIMA_parameters["ftotal_" + telescope.name] = result["ftotal"]

    if fs != 0.0:
        gblend = fb / fs
    elif fb == 0.0:
        gblend = 0.0
    else:
        gblend = float("inf")

    pyLIMA_parameters["gblend_" + telescope.name] = gblend

    # Most recent solution per telescope, for diagnostics.
    self.__dict__.setdefault("_bounded_profile_last_fluxes", {})[
        telescope.name
    ] = dict(result)

    return None


def enable_bounded_flux_profile(pyLIMA_model):
    """Install the bounded flux profile on one pyLIMA model instance."""
    if pyLIMA_model.blend_flux_parameter != "ftotal":
        raise ValueError(
            "bounded_profile requires blend_flux_parameter='ftotal'; "
            f"got {pyLIMA_model.blend_flux_parameter!r}."
        )

    pyLIMA_model.derive_telescope_flux = types.MethodType(
        _derive_telescope_flux_bounded_profile,
        pyLIMA_model,
    )

    # The analytical residual Jacobian includes explicit flux columns and
    # is not the Jacobian of the profiled problem.
    pyLIMA_model.Jacobian_flag = "Numerical"

    return pyLIMA_model
