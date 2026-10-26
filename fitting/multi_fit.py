"""Several named fits on the same simulated data: specs, entries, summary.

* normalize_fit_specs: dict or list of fit specs -> ordered, validated dict.
* _fit_entry_from_objects / _fit_entry_error: standard result entry per fit.
* flatten_multi_fit_results_for_parquet / save_multi_fit_summary_parquet:
  one-row summary per event.
The fits are run by functions_roman_rubin.run_multiple_named_fits.
"""

import numpy as np
import pandas as pd
import traceback
from pathlib import Path
from fitting.likelihood import (
    _empty_likelihood_stats,
    compute_pylima_photometric_likelihood_stats,
)


def normalize_fit_specs(
    fit_specs=None,
    fit_model=None,
    fit_parallax=None,
    fit_defaults=None,
    fit_bounds=None,
    initial_guess=None,
):
    """
    Normalize a set of requested fits.

    If ``fit_specs`` is None, this reproduces the old behavior with one fit.

    Accepted forms
    --------------
    dict:
        {
          "H0": {"model": "FSPL", "parallax": False, "bounds": {...}},
          "H1": {"model": "FSPL", "parallax": True,  "bounds": {...}},
        }

    list/tuple:
        [
          {"key": "H0", "model": "FSPL", "parallax": False},
          {"key": "H1", "model": "FSPL", "parallax": True},
        ]
    """

    from collections import OrderedDict

    if fit_specs is None:
        return OrderedDict({
            "fit": {
                "key": "fit",
                "label": "fit",
                "model": fit_model,
                "parallax": fit_parallax,
                "defaults": fit_defaults,
                "bounds": fit_bounds,
                "initial_guess": initial_guess,
            }
        })

    if isinstance(fit_specs, dict):
        out = OrderedDict()

        for key, spec in fit_specs.items():
            if spec is None:
                spec = {}
            if not isinstance(spec, dict):
                raise TypeError(
                    f"fit_specs[{key!r}] debe ser un diccionario."
                )

            key = str(key)
            label = str(spec.get("label", key))

            out[key] = {
                "key": key,
                "label": label,
                "model": spec.get("model", fit_model),
                "parallax": spec.get("parallax", fit_parallax),
                "defaults": spec.get("defaults", fit_defaults),
                "bounds": spec.get("bounds", fit_bounds),
                "initial_guess": spec.get("initial_guess", initial_guess),
            }

        if len(out) == 0:
            raise ValueError("fit_specs no puede estar vacío.")

        return out

    if isinstance(fit_specs, (list, tuple)):
        out = OrderedDict()

        for k, spec in enumerate(fit_specs):
            if spec is None:
                spec = {}
            if not isinstance(spec, dict):
                raise TypeError(
                    f"fit_specs[{k}] debe ser un diccionario."
                )

            key = str(spec.get("key", f"fit_{k}"))
            label = str(spec.get("label", key))

            if key in out:
                raise ValueError(f"fit_specs contiene key duplicada: {key!r}")

            out[key] = {
                "key": key,
                "label": label,
                "model": spec.get("model", fit_model),
                "parallax": spec.get("parallax", fit_parallax),
                "defaults": spec.get("defaults", fit_defaults),
                "bounds": spec.get("bounds", fit_bounds),
                "initial_guess": spec.get("initial_guess", initial_guess),
            }

        if len(out) == 0:
            raise ValueError("fit_specs no puede estar vacío.")

        return out

    raise TypeError("fit_specs debe ser None, dict, list o tuple.")


def _model_vector_from_fit_object(fit_rr):
    """Extract the best-fit parameter vector from a pyLIMA fit object."""

    return np.asarray(
        fit_rr.fit_results["best_model"],
        dtype=float,
    )


def _fit_entry_from_objects(
    fit_key,
    spec,
    fit_rr,
    event_fit_rr,
    pyLIMAmodel_rr,
    fit_roman=None,
    event_fit_roman=None,
    pyLIMAmodel_roman=None,
):
    """Build a standard multi-fit result entry from fit objects."""

    best_model = _model_vector_from_fit_object(fit_rr)

    likelihood_stats = compute_pylima_photometric_likelihood_stats(
        pyLIMAmodel_rr,
        best_model,
    )

    optimizer_diagnostics = {
        str(key): value
        for key, value in fit_rr.fit_results.items()
        if str(key).startswith("optimizer_")
    }

    fit_timings = {
        str(key): value
        for key, value in fit_rr.fit_results.items()
        if str(key).startswith("timing_")
    }

    return {
        "status": "fitted",
        "key": str(fit_key),
        "label": str(spec.get("label", fit_key)),
        "fit_model": spec.get("model", None),
        "fit_parallax": bool(spec.get("parallax", False)),
        "fit_bounds": spec.get("bounds", None),
        "fit_defaults": spec.get("defaults", None),
        "initial_guess": spec.get("initial_guess", None),
        "initial_guess_source": fit_rr.fit_results.get("initial_guess_source", None),
        "initial_guess_parameter_order": fit_rr.fit_results.get("initial_guess_parameter_order", None),
        "initial_guess_values": fit_rr.fit_results.get("initial_guess_values", None),
        "timings": fit_timings,
        "optimizer_diagnostics": optimizer_diagnostics,
        "fit_rr": fit_rr,
        "event_fit_rr": event_fit_rr,
        "pyLIMAmodel_rr": pyLIMAmodel_rr,
        "fit_roman": fit_roman,
        "event_fit_roman": event_fit_roman,
        "pyLIMAmodel_roman": pyLIMAmodel_roman,
        "best_model": best_model,
        "likelihood_stats": likelihood_stats,
    }


def _fit_entry_error(fit_key, spec, error):
    """Build a standard multi-fit result entry for a failed fit."""

    return {
        "status": "error",
        "key": str(fit_key),
        "label": str(spec.get("label", fit_key)),
        "fit_model": spec.get("model", None),
        "fit_parallax": bool(spec.get("parallax", False)),
        "fit_bounds": spec.get("bounds", None),
        "fit_defaults": spec.get("defaults", None),
        "initial_guess": spec.get("initial_guess", None),
        "initial_guess_source": None,
        "initial_guess_parameter_order": None,
        "initial_guess_values": None,
        "fit_rr": None,
        "event_fit_rr": None,
        "pyLIMAmodel_rr": None,
        "fit_roman": None,
        "event_fit_roman": None,
        "pyLIMAmodel_roman": None,
        "best_model": None,
        "likelihood_stats": _empty_likelihood_stats(),
        "error": repr(error),
        "traceback": traceback.format_exc(),
    }


def flatten_multi_fit_results_for_parquet(
    i,
    primary_fit_key,
    multi_fit_results,
    true_generator_stats=None,
    lrt_results=None,
    extra_metadata=None,
):
    """Flatten multi-fit output into a single-row parquet-friendly dict."""

    record = {
        "Source": int(i),
        "primary_fit_key": str(primary_fit_key),
    }

    if extra_metadata:
        for key, value in extra_metadata.items():
            record[key] = value

    for fit_key, entry in multi_fit_results.items():
        prefix = str(fit_key)
        record[f"{prefix}_status"] = entry.get("status", "")
        record[f"{prefix}_label"] = entry.get("label", "")
        record[f"{prefix}_fit_model"] = entry.get("fit_model", "")
        record[f"{prefix}_fit_parallax"] = entry.get("fit_parallax", np.nan)
        record[f"{prefix}_initial_guess"] = repr(entry.get("initial_guess", None))
        record[f"{prefix}_initial_guess_source"] = entry.get("initial_guess_source", None)
        record[f"{prefix}_initial_guess_parameter_order"] = repr(
            entry.get("initial_guess_parameter_order", None)
        )
        record[f"{prefix}_initial_guess_values"] = repr(
            entry.get("initial_guess_values", None)
        )

        stats_payload = entry.get("likelihood_stats", {})
        for key, value in stats_payload.items():
            if key in {"traceback"}:
                continue
            record[f"{prefix}_{key}"] = value

        timing_payload = entry.get(
            "timings",
            {},
        )

        for key, value in timing_payload.items():
            record[f"{prefix}_{key}"] = value


        optimizer_payload = entry.get(
            "optimizer_diagnostics",
            {},
        )

        for key, value in optimizer_payload.items():
            record[f"{prefix}_{key}"] = value

        if entry.get("best_model") is not None:
            try:
                best_model = np.asarray(entry.get("best_model"), dtype=float)
                record[f"{prefix}_best_model"] = repr(best_model.tolist())
            except Exception:
                pass

        if entry.get("status") == "error":
            record[f"{prefix}_error"] = entry.get("error", "")

    if true_generator_stats is not None:
        for key, value in true_generator_stats.items():
            # Do not store large/non-scalar internals in the parquet row.
            # Keep a compact flux-debug repr because it is useful for checking
            # the zero-point convention.
            if key in {"traceback", "true_vector"}:
                continue
            if key == "true_flux_debug":
                record[f"true_generator_{key}"] = repr(value)
            else:
                record[f"true_generator_{key}"] = value

    if lrt_results is not None:
        for key, value in lrt_results.items():
            if key in {"traceback"}:
                continue
            record[f"lrt_{key}"] = value

        # Convenience aliases without the lrt_ prefix.
        for key in [
            "LRT",
            "LRT_from_nll",
            "delta_chi2_H0_minus_H1",
            "delta_k",
            "p_value_LRT",
            "oracle_LRT_true_vs_H0",
            "delta_chi2_H0_minus_true_generator",
        ]:
            if key in lrt_results:
                record[key] = lrt_results[key]

    return record


def save_multi_fit_summary_parquet(
    i,
    path_to_save_results,
    primary_fit_key,
    multi_fit_results,
    true_generator_stats=None,
    lrt_results=None,
    nset_str="manual",
    extra_metadata=None,
):
    """Save one multi-fit summary parquet for this simulated event."""

    base_dir = Path(path_to_save_results)
    out_dir = base_dir / "multi_fit"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"multi_fit_{nset_str}_{i}.parquet"

    record = flatten_multi_fit_results_for_parquet(
        i=i,
        primary_fit_key=primary_fit_key,
        multi_fit_results=multi_fit_results,
        true_generator_stats=true_generator_stats,
        lrt_results=lrt_results,
        extra_metadata=extra_metadata,
    )

    pd.DataFrame([record]).to_parquet(
        path,
        engine="pyarrow",
        index=False,
    )

    print(f"→ Guardado multi-fit summary: {path}")

    return path, record
