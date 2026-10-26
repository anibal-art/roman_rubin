"""pyLIMA fitter, TRF least-squares options and fit diagnostics.

* build_fitter: pyLIMA TRF / DE / MCMC fit object.
* normalize_trf_optimizer_options / temporary_trf_least_squares_options:
  scipy least_squares options used inside pyLIMA's TRF.
* run_fit, finalize_fit_results, optimizer_diagnostics_from_fit.
"""

import multiprocessing as mul
import numpy as np
from contextlib import contextmanager
from pyLIMA.fits import DE_fit
from pyLIMA.fits import MCMC_fit
from pyLIMA.fits import TRF_fit
from set_model_pyLIMA import normalize_model_name

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
