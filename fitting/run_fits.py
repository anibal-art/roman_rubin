"""Run the pyLIMA fits of one simulated event.

* run_all_fits: Roman+Rubin fit and Roman-only fit (fit_lc.fit_rubin_roman),
  plus the chi2 of the true model on the Roman data.
* run_multiple_named_fits: several named fit specs on the same data.
"""

import inspect
from fit_lc import fit_rubin_roman
from fitting.fit_inputs import resolve_fit_options
from fitting.fit_inputs import validate_event_coordinates
from fitting.likelihood import true_photometric_chi2_for_telescope
from fitting.multi_fit import _fit_entry_error
from fitting.multi_fit import _fit_entry_from_objects
from pathlib import Path


def _call_fit_rubin_roman_compatible(*args, **kwargs):
    """
    Call fit_rubin_roman while passing only keyword arguments supported by the
    installed version of fit_lc.py.

    This keeps the function compatible with both older versions and the newer
    version that accepts event_ra/event_dec and random_state.
    """

    try:
        parameters = inspect.signature(fit_rubin_roman).parameters
        supported_kwargs = {
            key: val
            for key, val in kwargs.items()
            if key in parameters
        }

        dropped = sorted(set(kwargs) - set(supported_kwargs))
        if dropped:
            print(
                "[warning] fit_rubin_roman does not accept these kwargs; "
                f"dropping: {dropped}",
                flush=True,
            )

        return fit_rubin_roman(*args, **supported_kwargs)

    except Exception:
        # If inspect failed for any reason, call directly so the original error
        # is not hidden.
        return fit_rubin_roman(*args, **kwargs)


def run_all_fits(
    Source,
    pyLIMA_parameters,
    path_to_save_fit,
    path_ephemerides,
    model,
    algo,
    origin,
    rango,
    lc_to_fit,
    use_roman=True,
    use_rubin=True,
    fit_model=None,
    fit_parallax=None,
    fit_defaults=None,
    fit_bounds=None,
    initial_guess=None,
    event_ra=None,
    event_dec=None,
    random_state=None,
    optimizer_options=None,
    true_model=None,
):
    """
    Run Roman+Rubin/Rubin-only and optional Roman-only fits.

    Adds compatibility filtering for fit_rubin_roman keyword arguments,
    a deterministic random_state (defaulting to Source), and the same
    event_ra/event_dec geometry used for parallax fits.
    """

    model, fit_model, fit_parallax = resolve_fit_options(
        model,
        fit_model=fit_model,
        fit_parallax=fit_parallax,
        default_fit_parallax=True,
    )

    if (event_ra is None) != (event_dec is None):
        raise ValueError(
            "event_ra y event_dec deben pasarse juntos o ambos ser None."
        )

    if random_state is None:
        try:
            random_state = int(Source)
        except Exception:
            random_state = None

    lc_W149 = lc_to_fit["W149"] if use_roman else []

    lc_u = lc_to_fit["u"] if use_rubin else []
    lc_g = lc_to_fit["g"] if use_rubin else []
    lc_r = lc_to_fit["r"] if use_rubin else []
    lc_i = lc_to_fit["i"] if use_rubin else []
    lc_z = lc_to_fit["z"] if use_rubin else []
    lc_y = lc_to_fit["y"] if use_rubin else []

    print("Start the fit using active telescopes:")
    print(
        f"[run_all_fits] sim_model={model}, "
        f"fit_model={fit_model}, "
        f"fit_parallax={fit_parallax}, "
        f"event_ra={event_ra}, event_dec={event_dec}, "
        f"random_state={random_state}, "
        f"initial_guess={initial_guess!r}"
    )

    fit_rr, event_fit_rr, pyLIMAmodel_rr = _call_fit_rubin_roman_compatible(
        Source,
        pyLIMA_parameters,
        path_to_save_fit,
        path_ephemerides,
        model,
        algo,
        origin,
        rango,
        lc_W149,
        lc_u,
        lc_g,
        lc_r,
        lc_i,
        lc_z,
        lc_y,
        fit_model=fit_model,
        fit_parallax=fit_parallax,
        fit_defaults=fit_defaults,
        fit_bounds=fit_bounds,
        initial_guess=initial_guess,
        event_ra=event_ra,
        event_dec=event_dec,
        random_state=random_state,
        optimizer_options=optimizer_options,
    )

    fit_roman = None
    event_fit_roman = None
    pyLIMAmodel_roman = None

    if use_roman and len(lc_to_fit.get("W149", [])) != 0:

        # ========================================================
        # True-model chi2 on exactly the Roman data
        # ========================================================

        roman_true_info = None

        if true_model is not None:

            roman_true_info = true_photometric_chi2_for_telescope(
                true_model,
                pyLIMA_parameters,
                telescope_name="W149",
            )

            if (
                roman_true_info["n_data_true"]
                != len(lc_to_fit["W149"])
            ):
                raise RuntimeError(
                    "Roman true-chi2 data count does not match "
                    "Roman fit data count: "
                    f'{roman_true_info["n_data_true"]} vs '
                    f'{len(lc_to_fit["W149"])}'
                )

            print(
                "[pyLIMA chi2_true Roman] "
                f'chi2={roman_true_info["chi2_true"]:.6f}, '
                f'N={roman_true_info["n_data_true"]}, '
                f'chi2/N='
                f'{roman_true_info["chi2_true"] / roman_true_info["n_data_true"]:.6f}'
            )

        print("Start the fit using only the Roman data:")

        roman_random_state = None
        if random_state is not None:
            try:
                roman_random_state = int(random_state) + 1
            except Exception:
                roman_random_state = random_state

        fit_roman, event_fit_roman, pyLIMAmodel_roman = _call_fit_rubin_roman_compatible(
            Source,
            pyLIMA_parameters,
            path_to_save_fit,
            path_ephemerides,
            model,
            algo,
            origin,
            rango,
            lc_to_fit["W149"],
            [],
            [],
            [],
            [],
            [],
            [],
            fit_model=fit_model,
            fit_parallax=fit_parallax,
            fit_defaults=fit_defaults,
            fit_bounds=fit_bounds,
            initial_guess=initial_guess,
            event_ra=event_ra,
            event_dec=event_dec,
            random_state=roman_random_state,
            optimizer_options=optimizer_options,
        )

        if roman_true_info is not None:
            fit_roman.fit_results["chi2_true_roman"] = (
                roman_true_info["chi2_true"]
            )
            fit_roman.fit_results["n_data_true_roman"] = (
                roman_true_info["n_data_true"]
            )

    else:
        print("Roman-only fit skipped because Roman is off or has no data.")

    return (
        fit_rr,
        event_fit_rr,
        pyLIMAmodel_rr,
        fit_roman,
        event_fit_roman,
        pyLIMAmodel_roman,
    )


def run_multiple_named_fits(
    Source,
    pyLIMA_parameters,
    path_to_save_fit,
    path_ephemerides,
    model,
    algo,
    origin,
    rango,
    lc_to_fit,
    fit_specs,
    use_roman=True,
    use_rubin=True,
    event_ra=None,
    event_dec=None,
    existing_results=None,
    random_state_base=None,
    optimizer_options=None,
):
    """
    Run multiple named fits on the same ``lc_to_fit`` dictionary.

    Parameters
    ----------
    existing_results : dict or None
        Optional precomputed entries, for example the primary fit already run
        by sim_fit(). Keys present here are not refit.
    """

    if existing_results is None:
        results = {}
    else:
        results = dict(existing_results)

    if random_state_base is None:
        try:
            random_state_base = int(Source)
        except Exception:
            random_state_base = None

    Path(path_to_save_fit).mkdir(parents=True, exist_ok=True)

    for k, (fit_key, spec) in enumerate(fit_specs.items()):

        if fit_key in results:
            print(f"[multi-fit] {fit_key} already exists; skipping refit.")
            continue

        fit_model_use = spec.get("model", None)
        fit_parallax_use = spec.get("parallax", None)
        fit_defaults_use = spec.get("defaults", None)
        fit_bounds_use = spec.get("bounds", None)
        initial_guess_use = spec.get("initial_guess", None)

        if random_state_base is None:
            random_state = None
        else:
            random_state = int(random_state_base) + 1000 * (k + 1)

        print("=" * 80)
        print(f"[multi-fit] START {fit_key}: {spec.get('label', fit_key)}")
        print(
            f"[multi-fit] model={fit_model_use}, "
            f"parallax={fit_parallax_use}, "
            f"random_state={random_state}, "
            f"initial_guess={initial_guess_use!r}"
        )
        print("=" * 80)

        try:
            (
                fit_rr,
                event_fit_rr,
                pyLIMAmodel_rr,
                fit_roman,
                event_fit_roman,
                pyLIMAmodel_roman,
            ) = run_all_fits(
                Source,
                pyLIMA_parameters,
                path_to_save_fit,
                path_ephemerides,
                model,
                algo,
                origin,
                rango,
                lc_to_fit,
                use_roman=use_roman,
                use_rubin=use_rubin,
                fit_model=fit_model_use,
                fit_parallax=fit_parallax_use,
                fit_defaults=fit_defaults_use,
                fit_bounds=fit_bounds_use,
                initial_guess=initial_guess_use,
                optimizer_options=optimizer_options,
                event_ra=event_ra,
                event_dec=event_dec,
                random_state=random_state,
            )

            validate_event_coordinates(
                pyLIMAmodel_rr,
                event_ra,
                event_dec,
                label=f"{fit_key} Rubin+Roman/Rubin fit",
            )

            if pyLIMAmodel_roman is not None:
                validate_event_coordinates(
                    pyLIMAmodel_roman,
                    event_ra,
                    event_dec,
                    label=f"{fit_key} Roman-only fit",
                )

            results[fit_key] = _fit_entry_from_objects(
                fit_key=fit_key,
                spec=spec,
                fit_rr=fit_rr,
                event_fit_rr=event_fit_rr,
                pyLIMAmodel_rr=pyLIMAmodel_rr,
                fit_roman=fit_roman,
                event_fit_roman=event_fit_roman,
                pyLIMAmodel_roman=pyLIMAmodel_roman,
            )

        except Exception as error:
            print(
                f"[multi-fit] ERROR in {fit_key}: {repr(error)}",
                flush=True,
            )
            results[fit_key] = _fit_entry_error(
                fit_key=fit_key,
                spec=spec,
                error=error,
            )

    return results
