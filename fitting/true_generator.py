"""chi2 / logL of the TRUE simulated model on the fitted data.

The true parameter vector (with the true fluxes expressed in the fit
zero-point system, PYLIMA_FIT_ZERO_POINT) is evaluated on a fresh fit model
built from lc_to_fit. A fallback recovers the same numbers from the primary
sim_fit result.
"""

import numpy as np
import pandas as pd
from photometry.constants import PYLIMA_FIT_ZERO_POINT
from photometry.constants import SIMULATION_BAND_ZERO_POINTS
from fitting.likelihood import (
    compute_pylima_photometric_likelihood_stats,
)


def _ordered_model_parameter_names(pyLIMA_model):
    """
    Return model parameters in the exact vector order expected by pyLIMA.
    """

    return [
        name
        for name, _ in sorted(
            pyLIMA_model.model_dictionnary.items(),
            key=lambda item: item[1],
        )
    ]


def _get_true_pyparam_value(true_pyLIMA_parameters, name):
    """
    Extract a value from pyLIMA_parameters, whether it is dict-like or has
    attributes.  Raises a clear KeyError if unavailable.
    """

    if isinstance(true_pyLIMA_parameters, dict):
        if name in true_pyLIMA_parameters:
            return true_pyLIMA_parameters[name]

    if hasattr(true_pyLIMA_parameters, name):
        return getattr(true_pyLIMA_parameters, name)

    try:
        return true_pyLIMA_parameters[name]
    except Exception as error:
        raise KeyError(
            f"No pude extraer el parámetro verdadero {name!r}."
        ) from error


def _event_param_value(event_params, names, default=None):
    """Return the first existing event_params value among names."""

    if event_params is None:
        return default

    for name in names:
        if name in event_params:
            value = event_params[name]
            try:
                if value is None:
                    continue
                value_float = float(value)
                if np.isfinite(value_float):
                    return value_float
            except Exception:
                return value

    return default


def _source_mag_for_band(event_params, band):
    """
    Get the catalog/source magnitude for the requested band.

    The Sedighe LSSTMONTS rows keep source_mag_<band>.  Older/custom inputs may
    keep the magnitude directly as r/i/y or Y/W149.  We support all of them.
    """

    band = str(band)

    candidates = [
        f"source_mag_{band}",
        band,
    ]

    if band == "y":
        candidates.extend(["source_mag_Y", "Y"])
    elif band == "Y":
        candidates.extend(["source_mag_y", "y"])
    elif band == "W149":
        candidates.extend(["source_mag_W149"])

    value = _event_param_value(event_params, candidates, default=None)

    if value is None:
        raise KeyError(
            f"No encuentro magnitud de fuente para banda {band!r}. "
            f"Probé {candidates}."
        )

    return float(value)


def _source_fraction_for_band(event_params, band, default=1.0):
    """
    Get f_s = F_source/F_total for a band.

    In the current LSSTMONTS pipeline this is stored as source_fraction_<band>.
    For compatibility we also accept blend_<band>, because in older configs that
    field was used to carry the same source-fraction quantity.
    """

    band = str(band)

    candidates = [
        f"source_fraction_{band}",
        f"blend_{band}",
        f"fs_{band}",
        f"fsource_fraction_{band}",
    ]

    if band == "y":
        candidates.extend([
            "source_fraction_Y",
            "blend_Y",
        ])
    elif band == "Y":
        candidates.extend([
            "source_fraction_y",
            "blend_y",
        ])

    value = _event_param_value(event_params, candidates, default=default)

    try:
        value = float(value)
    except Exception:
        value = float(default)

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"source_fraction inválida para banda {band!r}: {value}. "
            "Para evaluar el modelo verdadero se requiere f_s > 0."
        )

    return value


def _true_fluxes_in_fresh_fit_system(
    band,
    event_params,
    true_pyLIMA_parameters=None,
    fit_zero_point=PYLIMA_FIT_ZERO_POINT,
):
    """
    Compute true fsource/ftotal in the flux system used by fit_lc.py.

    Important convention:
    - The original simulator uses Rubin/Roman band-dependent zero points.
    - fit_lc.py builds fitting telescopes from magnitudes and pyLIMA converts
      those magnitudes using its internal zero point, here 27.4.

    Therefore the true generator fluxes used for oracle likelihoods must be in
    the fresh fit-model system, not in the original simulation ZP system.
    """

    band = str(band)
    band_for_catalog = "y" if band == "Y" else band

    try:
        mag_source = _source_mag_for_band(event_params, band_for_catalog)
        source_fraction = _source_fraction_for_band(
            event_params,
            band_for_catalog,
            default=1.0,
        )

        fsource = 10.0 ** (0.4 * (float(fit_zero_point) - mag_source))
        ftotal = fsource / source_fraction

        return float(fsource), float(ftotal)

    except Exception:
        # Fallback: rescale fluxes already present in true_pyLIMA_parameters
        # from the simulator band zero point into the fit-model zero point.
        if true_pyLIMA_parameters is None:
            raise

        fsource_name = f"fsource_{band}"
        ftotal_name = f"ftotal_{band}"

        try:
            fsource_original = float(
                _get_true_pyparam_value(true_pyLIMA_parameters, fsource_name)
            )
            ftotal_original = float(
                _get_true_pyparam_value(true_pyLIMA_parameters, ftotal_name)
            )
        except Exception:
            # Last y/Y compatibility fallback.
            alt_band = "Y" if band == "y" else "y" if band == "Y" else band
            fsource_original = float(
                _get_true_pyparam_value(true_pyLIMA_parameters, f"fsource_{alt_band}")
            )
            ftotal_original = float(
                _get_true_pyparam_value(true_pyLIMA_parameters, f"ftotal_{alt_band}")
            )

        simulation_zp = SIMULATION_BAND_ZERO_POINTS.get(
            band,
            SIMULATION_BAND_ZERO_POINTS.get("y" if band == "Y" else band, fit_zero_point),
        )

        scale = 10.0 ** (0.4 * (float(fit_zero_point) - float(simulation_zp)))

        return (
            float(fsource_original * scale),
            float(ftotal_original * scale),
        )


def true_parameter_vector_for_fresh_fit_model(
    fresh_parallax_model,
    true_pyLIMA_parameters,
    event_params,
    fit_zero_point=PYLIMA_FIT_ZERO_POINT,
):
    """
    Build the true-generator parameter vector in the order required by a fresh
    fit model.

    This is the correct object for evaluating logL_true:
    - The model is fresh and was built only from surviving fit points.
    - It includes parallax when the alternative model includes parallax.
    - Flux parameters are recomputed in the same zero-point convention used by
      fit_lc.py, not copied from the original simulator model.
    """

    values = []
    flux_debug = {}

    for name in _ordered_model_parameter_names(fresh_parallax_model):

        if name.startswith("fsource_"):
            band = name.replace("fsource_", "", 1)
            fsource, ftotal = _true_fluxes_in_fresh_fit_system(
                band=band,
                event_params=event_params,
                true_pyLIMA_parameters=true_pyLIMA_parameters,
                fit_zero_point=fit_zero_point,
            )
            values.append(fsource)
            flux_debug.setdefault(band, {})["fsource"] = fsource
            flux_debug.setdefault(band, {})["ftotal"] = ftotal
            continue

        if name.startswith("ftotal_"):
            band = name.replace("ftotal_", "", 1)
            fsource, ftotal = _true_fluxes_in_fresh_fit_system(
                band=band,
                event_params=event_params,
                true_pyLIMA_parameters=true_pyLIMA_parameters,
                fit_zero_point=fit_zero_point,
            )
            values.append(ftotal)
            flux_debug.setdefault(band, {})["fsource"] = fsource
            flux_debug.setdefault(band, {})["ftotal"] = ftotal
            continue

        if name.startswith("fblend_"):
            band = name.replace("fblend_", "", 1)
            fsource, ftotal = _true_fluxes_in_fresh_fit_system(
                band=band,
                event_params=event_params,
                true_pyLIMA_parameters=true_pyLIMA_parameters,
                fit_zero_point=fit_zero_point,
            )
            values.append(ftotal - fsource)
            flux_debug.setdefault(band, {})["fsource"] = fsource
            flux_debug.setdefault(band, {})["ftotal"] = ftotal
            flux_debug.setdefault(band, {})["fblend"] = ftotal - fsource
            continue

        values.append(
            float(_get_true_pyparam_value(true_pyLIMA_parameters, name))
        )

    return np.asarray(values, dtype=float), flux_debug


def _choose_fresh_model_for_true_generator(
    multi_fit_results,
    alternative_key="H1",
):
    """
    Choose a fresh model for evaluating the true generator.

    Prefer the LRT alternative model because it should have the same physical
    structure as the simulated truth, i.e. FSPL + parallax.  If unavailable,
    fall back to any successful parallax fit.
    """

    if alternative_key in multi_fit_results:
        entry = multi_fit_results[alternative_key]
        if entry.get("pyLIMAmodel_rr", None) is not None:
            return str(alternative_key), entry["pyLIMAmodel_rr"]

    for key, entry in multi_fit_results.items():
        if bool(entry.get("fit_parallax", False)) and entry.get("pyLIMAmodel_rr", None) is not None:
            return str(key), entry["pyLIMAmodel_rr"]

    raise RuntimeError(
        "No hay modelo fresco con paralaje para evaluar true_generator_logL. "
        "Necesito un fit alternativo tipo H1 con parallax=True."
    )


def compute_true_generator_stats_on_fresh_fit_model(
    multi_fit_results,
    true_pyLIMA_parameters,
    event_params,
    alternative_key="H1",
    fit_zero_point=PYLIMA_FIT_ZERO_POINT,
):
    """
    Correct oracle likelihood for the true generator.

    This evaluates the true microlensing parameters on a fresh pyLIMA model
    built from the same surviving data points used by H0/H1.  It avoids the
    common failure mode where the original simulation model still carries
    parallax arrays with pre-filter lengths.
    """

    fresh_key, fresh_model = _choose_fresh_model_for_true_generator(
        multi_fit_results=multi_fit_results,
        alternative_key=alternative_key,
    )

    true_vector, flux_debug = true_parameter_vector_for_fresh_fit_model(
        fresh_parallax_model=fresh_model,
        true_pyLIMA_parameters=true_pyLIMA_parameters,
        event_params=event_params,
        fit_zero_point=fit_zero_point,
    )

    stats_out = compute_pylima_photometric_likelihood_stats(
        fresh_model,
        true_vector,
    )

    stats_out["fresh_model_key"] = fresh_key
    stats_out["fit_zero_point"] = float(fit_zero_point)
    stats_out["true_vector"] = true_vector
    stats_out["true_flux_debug"] = flux_debug

    return stats_out


def _scalar_from_summary_dict(dct, key, default=np.nan):
    """Return the first scalar from summary dict values stored as [value]."""

    try:
        value = dct.get(key, default)
    except Exception:
        return default

    try:
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            if len(value) == 0:
                return default
            value = value[0]
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return value


def true_generator_stats_fallback_from_primary_result(primary_result, true_model=None):
    """
    Fallback diagnostics for the true generator when the formal pyLIMA
    likelihood cannot be evaluated on the filtered light curves.

    This can happen if pyLIMA parallax arrays were computed before a
    photometric filter shortened one telescope light curve.  The formal H0/H1
    LRT is not affected, because H0 and H1 are fit models rebuilt directly on
    the same lc_to_fit.  For the generator we keep chi2_true/n_data_true as a
    simulation-only diagnostic and set nll/logL to NaN.
    """

    chi2_true = np.nan
    n_data_true = 0

    true_summary = None

    try:
        true_summary = primary_result.get("true", None)
    except Exception:
        true_summary = None

    if isinstance(true_summary, dict):
        chi2_true = _scalar_from_summary_dict(
            true_summary,
            "chi2_true",
            default=np.nan,
        )
        n_data_true = _scalar_from_summary_dict(
            true_summary,
            "n_data_true",
            default=0,
        )

    if not np.isfinite(float(chi2_true)):
        try:
            event_params = primary_result.get("event_params", {})
            chi2_true = event_params.get("chi2_true", np.nan)
        except Exception:
            chi2_true = np.nan

    try:
        n_data_true = int(n_data_true)
    except Exception:
        n_data_true = 0

    try:
        n_params = int(len(true_model.model_dictionnary))
    except Exception:
        n_params = np.nan

    try:
        dof = int(n_data_true - n_params) if np.isfinite(n_params) else np.nan
    except Exception:
        dof = np.nan

    if np.isfinite(chi2_true) and np.isfinite(dof) and dof > 0:
        chi2_red = float(chi2_true) / float(dof)
    else:
        chi2_red = np.nan

    return {
        "nll": np.nan,
        "logL": np.nan,
        "chi2": float(chi2_true) if np.isfinite(float(chi2_true)) else np.nan,
        "n_data": n_data_true,
        "n_params": n_params,
        "dof": dof,
        "chi2_red": chi2_red,
        "p_value_chi2_gof": np.nan,
        "error": "formal true-generator logL unavailable; using chi2_true fallback",
    }
