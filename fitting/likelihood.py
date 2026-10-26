"""Photometric chi2 / likelihood and likelihood-ratio tests.

* compute_pylima_photometric_likelihood_stats: chi2, n_data, dof,
  chi2_red and logL = -0.5 chi2 from pyLIMA's normalized residuals.
* true_photometric_chi2_for_telescope: chi2 of the true model on one
  telescope with pyLIMA's objective function.
* compute_lrt_from_multi_fits: LRT = chi2_H0 - chi2_H1 for two named fits
  of the SAME data (asymptotic p-value only as a reference).
* add_oracle_lrt_to_results: LRT of the true generator vs H0.
"""

import copy
import numpy as np


def _empty_likelihood_stats():
    """Return a standard likelihood/chi2 payload filled with NaNs."""

    return {
        "nll": np.nan,
        "logL": np.nan,
        "chi2": np.nan,
        "n_data": 0,
        "n_params": np.nan,
        "dof": np.nan,
        "chi2_red": np.nan,
        "p_value_chi2_gof": np.nan,
    }


def _as_pylima_parameters_for_stats(pyLIMA_model, model_parameters):
    """
    Accept either a model vector or pyLIMA_parameters and return
    pyLIMA_parameters.
    """

    if isinstance(model_parameters, (list, tuple, np.ndarray)):
        return pyLIMA_model.compute_pyLIMA_parameters(
            np.asarray(model_parameters, dtype=float)
        )

    return model_parameters


def compute_pylima_photometric_likelihood_stats(
    pyLIMA_model,
    model_parameters,
    rescaling_photometry_parameters=None,
):
    """
    Compute chi2 and a consistent Gaussian log-likelihood, up to a common
    additive constant, from normalized photometric residuals.

    This deliberately does NOT call
    objective_functions.all_telescope_photometric_likelihood(), because the
    absolute normalization can differ between reconstructed models and is not
    needed for LRTs.  The LRT uses only chi2 differences.
    """

    import numpy as _np
    import traceback as _traceback

    try:
        from scipy import stats as _stats
    except Exception:
        _stats = None

    from pyLIMA.fits import objective_functions as _objective_functions

    if pyLIMA_model is None:
        out = _empty_likelihood_stats()
        out["error"] = "pyLIMA_model is None"
        return out

    try:
        n_params = int(len(pyLIMA_model.model_dictionnary))
    except Exception:
        n_params = _np.nan

    try:
        pyparams = _as_pylima_parameters_for_stats(
            pyLIMA_model,
            model_parameters,
        )

        residus, errflux = _objective_functions.all_telescope_photometric_residuals(
            pyLIMA_model,
            pyparams,
            norm=True,
            rescaling_photometry_parameters=rescaling_photometry_parameters,
        )

        residus = [
            _np.asarray(r, dtype=float)
            for r in residus
            if len(r) > 0
        ]

        if len(residus) == 0:
            out = _empty_likelihood_stats()
            out.update({
                "n_params": n_params,
                "error": "no photometric residuals",
                "likelihood_definition": "logL_chi2=-0.5*chi2, no constant",
            })
            return out

        all_residus = _np.concatenate(residus)
        all_residus = all_residus[_np.isfinite(all_residus)]

        chi2 = float(_np.sum(all_residus**2))
        n_data = int(len(all_residus))
        dof = int(n_data - n_params) if _np.isfinite(n_params) else _np.nan

        logL_chi2 = -0.5 * chi2
        nll_chi2 = 0.5 * chi2

        if _np.isfinite(dof) and dof > 0:
            chi2_red = float(chi2 / dof)
            p_value_chi2_gof = (
                float(_stats.chi2.sf(chi2, int(dof)))
                if _stats is not None
                else _np.nan
            )
        else:
            chi2_red = _np.nan
            p_value_chi2_gof = _np.nan

        return {
            "nll": nll_chi2,
            "logL": logL_chi2,
            "nll_chi2": nll_chi2,
            "logL_chi2": logL_chi2,
            "chi2": chi2,
            "n_data": n_data,
            "n_params": n_params,
            "dof": dof,
            "chi2_red": chi2_red,
            "p_value_chi2_gof": p_value_chi2_gof,
            "likelihood_definition": "logL_chi2=-0.5*chi2, no constant",
        }

    except Exception as error:
        out = _empty_likelihood_stats()
        out["n_params"] = n_params
        out["error"] = repr(error)
        out["traceback"] = _traceback.format_exc()
        out["likelihood_definition"] = "failed chi2 residual evaluation"
        return out


def true_photometric_chi2_for_telescope(
    pyLIMA_model,
    pyLIMA_parameters,
    telescope_name,
):
    """
    Evaluate the true-model chi2 on one existing telescope only.

    Uses pyLIMA's official all_telescope_photometric_chi2 unchanged.
    A shallow model/event view exposes only the requested telescope;
    the original simulation model and lightcurve are not modified.
    """
    from pyLIMA.fits.objective_functions import (
        all_telescope_photometric_chi2,
    )

    matches = [
        telescope
        for telescope in pyLIMA_model.event.telescopes
        if str(telescope.name) == str(telescope_name)
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one telescope named "
            f"{telescope_name!r}; found {len(matches)}."
        )

    telescope = matches[0]

    if telescope.lightcurve is None or len(telescope.lightcurve) == 0:
        raise RuntimeError(
            f"Telescope {telescope_name!r} has no data."
        )

    model_view = copy.copy(pyLIMA_model)
    event_view = copy.copy(pyLIMA_model.event)

    event_view.telescopes = [telescope]
    model_view.event = event_view

    chi2 = float(
        all_telescope_photometric_chi2(
            model_view,
            pyLIMA_parameters,
        )
    )

    if not np.isfinite(chi2):
        raise RuntimeError(
            f"Non-finite true chi2 for {telescope_name!r}."
        )

    return {
        "chi2_true": chi2,
        "n_data_true": int(len(telescope.lightcurve)),
    }


def compute_lrt_from_multi_fits(
    multi_fit_results,
    null_key="H0",
    alternative_key="H1",
    delta_k=None,
):
    """
    Compute LRT from chi2 differences:

        LRT = chi2_H0 - chi2_H1

    This is equivalent to 2(logL_H1-logL_H0) for Gaussian errors when
    logL=-0.5*chi2 up to the same additive constant for both models.
    """

    import numpy as _np
    import traceback as _traceback

    try:
        from scipy import stats as _stats
    except Exception:
        _stats = None

    out = {
        "null_key": null_key,
        "alternative_key": alternative_key,
        "logL_H0": _np.nan,
        "logL_H1": _np.nan,
        "logL_chi2_H0": _np.nan,
        "logL_chi2_H1": _np.nan,
        "nll_H0": _np.nan,
        "nll_H1": _np.nan,
        "nll_chi2_H0": _np.nan,
        "nll_chi2_H1": _np.nan,
        "chi2_H0": _np.nan,
        "chi2_H1": _np.nan,
        "LRT": _np.nan,
        "LRT_chi2": _np.nan,
        "LRT_from_nll": _np.nan,
        "delta_chi2_H0_minus_H1": _np.nan,
        "delta_k": _np.nan,
        "p_value_LRT": _np.nan,
        "same_n_data_H0_H1": False,
        "likelihood_definition": "LRT=chi2_H0-chi2_H1",
        "error": "",
    }

    try:
        h0_entry = multi_fit_results[null_key]
        h1_entry = multi_fit_results[alternative_key]

        if h0_entry.get("status") != "fitted" or h1_entry.get("status") != "fitted":
            out["error"] = (
                f"Fits not both fitted: "
                f"{null_key}={h0_entry.get('status')}, "
                f"{alternative_key}={h1_entry.get('status')}"
            )
            return out

        h0 = h0_entry["likelihood_stats"]
        h1 = h1_entry["likelihood_stats"]

        chi2_H0 = float(h0["chi2"])
        chi2_H1 = float(h1["chi2"])

        logL_H0 = -0.5 * chi2_H0
        logL_H1 = -0.5 * chi2_H1
        nll_H0 = 0.5 * chi2_H0
        nll_H1 = 0.5 * chi2_H1

        if delta_k is None:
            delta_k_use = int(h1["n_params"] - h0["n_params"])
        else:
            delta_k_use = int(delta_k)

        LRT = chi2_H0 - chi2_H1
        LRT_from_nll = 2.0 * (nll_H0 - nll_H1)
        delta_chi2 = LRT

        if _stats is not None and _np.isfinite(LRT) and delta_k_use > 0:
            p_value_LRT = float(_stats.chi2.sf(LRT, df=delta_k_use))
        else:
            p_value_LRT = _np.nan

        out.update({
            "logL_H0": logL_H0,
            "logL_H1": logL_H1,
            "logL_chi2_H0": logL_H0,
            "logL_chi2_H1": logL_H1,
            "nll_H0": nll_H0,
            "nll_H1": nll_H1,
            "nll_chi2_H0": nll_H0,
            "nll_chi2_H1": nll_H1,
            "chi2_H0": chi2_H0,
            "chi2_H1": chi2_H1,
            "LRT": LRT,
            "LRT_chi2": LRT,
            "LRT_from_nll": LRT_from_nll,
            "delta_chi2_H0_minus_H1": delta_chi2,
            "delta_k": delta_k_use,
            "p_value_LRT": p_value_LRT,
            "same_n_data_H0_H1": int(h0.get("n_data", -1)) == int(h1.get("n_data", -2)),
        })

    except Exception as error:
        out["error"] = repr(error)
        out["traceback"] = _traceback.format_exc()

    return out


def add_oracle_lrt_to_results(lrt_results, true_generator_stats, multi_fit_results):
    """
    Add oracle simulation diagnostics using the same chi2 convention.
    """

    import numpy as _np
    import traceback as _traceback

    if lrt_results is None:
        return None

    out = dict(lrt_results)

    try:
        null_key = out.get("null_key", "H0")
        h0_stats = multi_fit_results[null_key]["likelihood_stats"]

        h0_chi2 = float(h0_stats.get("chi2", _np.nan))
        true_chi2 = float(true_generator_stats.get("chi2", _np.nan))

        true_logL_chi2 = -0.5 * true_chi2
        true_nll_chi2 = 0.5 * true_chi2

        out["true_generator_logL"] = true_logL_chi2
        out["true_generator_logL_chi2"] = true_logL_chi2
        out["true_generator_nll"] = true_nll_chi2
        out["true_generator_nll_chi2"] = true_nll_chi2
        out["true_generator_chi2"] = true_chi2
        out["true_generator_n_data"] = int(true_generator_stats.get("n_data", 0))
        out["true_generator_n_params"] = true_generator_stats.get("n_params", _np.nan)
        out["true_generator_dof"] = true_generator_stats.get("dof", _np.nan)
        out["true_generator_likelihood_definition"] = "logL_chi2=-0.5*chi2, no constant"

        out["oracle_LRT_true_vs_H0"] = h0_chi2 - true_chi2
        out["oracle_LRT_chi2_true_vs_H0"] = h0_chi2 - true_chi2
        out["delta_chi2_H0_minus_true_generator"] = h0_chi2 - true_chi2

        try:
            out["same_n_data_H0_true_generator"] = (
                int(h0_stats.get("n_data", -1))
                == int(true_generator_stats.get("n_data", -2))
            )
        except Exception:
            out["same_n_data_H0_true_generator"] = False

    except Exception as error:
        out["oracle_error"] = repr(error)
        out["oracle_traceback"] = _traceback.format_exc()

    return out
