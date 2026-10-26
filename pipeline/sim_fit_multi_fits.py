"""Simulate one catalogue event and run several named fits on it.

sim_fit_multi_fits runs the primary sim_fit and then the named fit specs
(e.g. H1/H0) on the same simulated data, with LRT and oracle LRT summaries.
"""

import numpy as np
import time
import traceback
from analysis.simfit_records import extract_nset_string
from fitting.fit_inputs import apply_fit_time_window_to_lc_dict
from fitting.fit_inputs import count_active_fit_points
from fitting.fit_inputs import extract_lightcurves_for_fit
from fitting.fit_inputs import get_event_coordinates
from fitting.likelihood import add_oracle_lrt_to_results
from fitting.likelihood import compute_lrt_from_multi_fits
from fitting.multi_fit import _fit_entry_from_objects
from fitting.multi_fit import normalize_fit_specs
from fitting.multi_fit import save_multi_fit_summary_parquet
from fitting.true_generator import compute_true_generator_stats_on_fresh_fit_model
from fitting.true_generator import true_generator_stats_fallback_from_primary_result
from photometry.constants import PYLIMA_FIT_ZERO_POINT
from timing_utils import StageTimer
from fitting.run_fits import (
    run_multiple_named_fits,
)
from pipeline.sim_fit import (
    sim_fit,
)


def sim_fit_multi_fits(
    i,
    system_type,
    model,
    algo,
    path_TRILEGAL_set,
    path_GENULENS_set,
    path_to_save_model,
    path_to_save_fit,
    path_ephemerides,
    path_to_save_results,
    time_window=None,
    param_samplers=None,
    t0_range=[2460413.013828608, 2460413.013828608 + 365.25 * 8],
    custom_system=None,
    catalog_mode="custom_system",
    pair_catalog=None,
    path_pair_catalog=None,
    use_roman=True,
    use_rubin=True,
    truth_parallax=True,
    fit_time_window=None,
    return_data=False,
    fit_model=None,
    fit_parallax=None,
    fit_defaults=None,
    fit_bounds=None,
    initial_guess=None,
    optimizer_options=None,
    fit_specs=None,
    primary_fit=None,
    lrt_config=None,
    save_multi_fit_summary=True,
    rubin_pointing_mode="source",
    rubin_cache_cell_deg=None,
    apply_detection_criteria=True,
    apply_photometric_filter=True,
    rubin_saturation_mag=None,
    roman_saturation_mag=None,
):
    """
    Generalization of sim_fit: simulate once, then run multiple fits.

    The simulated noisy light curve is generated exactly once by ``sim_fit``.
    The primary fit is also run by ``sim_fit``.  All additional fits are then
    run on the same ``lc_to_fit`` extracted from the returned true pyLIMA model.

    This is the preferred interface for hypothesis tests such as:

        H0: FSPL without parallax
        H1: FSPL with parallax

    because H0 and H1 are evaluated on the same noise realization and the same
    fit-time window.
    """

    tstart = time.time()

    _multi_timer = StageTimer()
    _multi_timer.start("multifit_total")

    fit_specs_use = normalize_fit_specs(
        fit_specs=fit_specs,
        fit_model=fit_model,
        fit_parallax=fit_parallax,
        fit_defaults=fit_defaults,
        fit_bounds=fit_bounds,
        initial_guess=initial_guess,
    )

    if primary_fit is None:
        primary_fit_key = next(iter(fit_specs_use.keys()))
    else:
        primary_fit_key = str(primary_fit)

    if primary_fit_key not in fit_specs_use:
        raise KeyError(
            f"primary_fit={primary_fit_key!r} no está en fit_specs. "
            f"Opciones: {list(fit_specs_use.keys())}"
        )

    primary_spec = fit_specs_use[primary_fit_key]

    print("=" * 80)
    print("[sim_fit_multi_fits] one simulation + multiple fits")
    print(f"[sim_fit_multi_fits] primary_fit_key = {primary_fit_key}")
    print(f"[sim_fit_multi_fits] fit keys = {list(fit_specs_use.keys())}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Run the original sim_fit once.  This simulates the noisy event and
    #    executes the primary fit.
    # ------------------------------------------------------------------
    _multi_timer.start("primary_simfit")
    primary_result = sim_fit(
        i=i,
        system_type=system_type,
        model=model,
        algo=algo,
        path_TRILEGAL_set=path_TRILEGAL_set,
        path_GENULENS_set=path_GENULENS_set,
        path_to_save_model=path_to_save_model,
        path_to_save_fit=path_to_save_fit,
        path_ephemerides=path_ephemerides,
        path_to_save_results=path_to_save_results,
        time_window=time_window,
        param_samplers=param_samplers,
        t0_range=t0_range,
        custom_system=custom_system,
        catalog_mode=catalog_mode,
        pair_catalog=pair_catalog,
        path_pair_catalog=path_pair_catalog,
        use_roman=use_roman,
        use_rubin=use_rubin,
        truth_parallax=truth_parallax,
        fit_time_window=fit_time_window,
        return_data=True,
        fit_model=primary_spec.get("model", fit_model),
        fit_parallax=primary_spec.get("parallax", fit_parallax),
        fit_defaults=primary_spec.get("defaults", fit_defaults),
        fit_bounds=primary_spec.get("bounds", fit_bounds),
        initial_guess=primary_spec.get("initial_guess", initial_guess),
        optimizer_options=optimizer_options,
        rubin_pointing_mode=rubin_pointing_mode,
        rubin_cache_cell_deg=rubin_cache_cell_deg,
        apply_detection_criteria=apply_detection_criteria,
        apply_photometric_filter=apply_photometric_filter,
        rubin_saturation_mag=rubin_saturation_mag,
        roman_saturation_mag=roman_saturation_mag,
    )
    _multi_timer.stop("primary_simfit")


    if not isinstance(primary_result, dict):
        return primary_result

    if primary_result.get("status") != "fitted":
        primary_result["multi_fit_status"] = "not_run"
        primary_result["multi_fit_reason"] = (
            "primary sim_fit did not return status='fitted'"
        )
        primary_result["fit_specs"] = fit_specs_use
        return primary_result if return_data else primary_result

    Source = int(i)
    seed = int(i)
    pyLIMA_parameters = primary_result["pyLIMA_parameters_true"]
    true_model = primary_result["pyLIMAmodel_true"]
    simulation_event_ra = primary_result.get("event_ra_used", np.nan)
    simulation_event_dec = primary_result.get("event_dec_used", np.nan)

    if not np.isfinite(simulation_event_ra) or not np.isfinite(simulation_event_dec):
        simulation_event_ra, simulation_event_dec = get_event_coordinates(
            true_model
        )

    origin = getattr(true_model, "origin", None)
    rango = 1

    # ------------------------------------------------------------------
    # 2. Re-extract the exact same light curve for all additional fits.
    #    If the runner has patched extract_lightcurves_for_fit, this call uses
    #    the same patch as sim_fit.
    # ------------------------------------------------------------------
    _multi_timer.start("reextract_fit_lightcurve")
    lc_to_fit, lc_to_save_unused = extract_lightcurves_for_fit(true_model)

    lc_to_fit = apply_fit_time_window_to_lc_dict(
        lc_to_fit,
        pyLIMA_parameters,
        fit_time_window=fit_time_window,
    )

    n_fit_points = count_active_fit_points(
        lc_to_fit,
        use_roman=use_roman,
        use_rubin=use_rubin,
    )
    _multi_timer.stop("reextract_fit_lightcurve")


    print("[sim_fit_multi_fits] n_fit_points reused:", n_fit_points)

    # ------------------------------------------------------------------
    # 3. Register the already-run primary fit in the multi-fit result dict.
    # ------------------------------------------------------------------
    existing_results = {
        primary_fit_key: _fit_entry_from_objects(
            fit_key=primary_fit_key,
            spec=primary_spec,
            fit_rr=primary_result["fit_rr"],
            event_fit_rr=None,
            pyLIMAmodel_rr=primary_result["pyLIMAmodel_rr"],
            fit_roman=primary_result.get("fit_roman", None),
            event_fit_roman=None,
            pyLIMAmodel_roman=primary_result.get("pyLIMAmodel_roman", None),
        )
    }

    # ------------------------------------------------------------------
    # 4. Run all remaining fits on exactly the same lc_to_fit.
    # ------------------------------------------------------------------
    _multi_timer.start("additional_fits")
    multi_fit_results = run_multiple_named_fits(
        Source=Source,
        pyLIMA_parameters=pyLIMA_parameters,
        path_to_save_fit=path_to_save_fit,
        path_ephemerides=path_ephemerides,
        model=model,
        algo=algo,
        origin=origin,
        rango=rango,
        lc_to_fit=lc_to_fit,
        fit_specs=fit_specs_use,
        use_roman=use_roman,
        use_rubin=use_rubin,
        event_ra=simulation_event_ra,
        event_dec=simulation_event_dec,
        existing_results=existing_results,
        random_state_base=seed,
        optimizer_options=optimizer_options,
    )
    _multi_timer.stop("additional_fits")


    # ------------------------------------------------------------------
    # 5. True-generator likelihood on the same surviving fit-window data.
    # ------------------------------------------------------------------
    # Do NOT evaluate the original true_model directly here.  That object can
    # still carry pyLIMA parallax arrays computed before photometric filtering,
    # with lengths different from the surviving light curves.  Instead, use the
    # fresh H1-like fit model, which was rebuilt from lc_to_fit and therefore
    # has the correct times, errors and parallax arrays.  The true fluxes are
    # recomputed in the fit-model zero-point system, currently pyLIMA ZP=27.4.
    _multi_timer.start("true_generator")
    true_model_for_likelihood = None

    try:
        alternative_key = "H1"
        if isinstance(lrt_config, dict):
            alternative_key = str(lrt_config.get("alternative", alternative_key))

        true_generator_stats = compute_true_generator_stats_on_fresh_fit_model(
            multi_fit_results=multi_fit_results,
            true_pyLIMA_parameters=pyLIMA_parameters,
            event_params=primary_result.get("event_params", {}),
            alternative_key=alternative_key,
            fit_zero_point=PYLIMA_FIT_ZERO_POINT,
        )

        true_model_for_likelihood = multi_fit_results[
            true_generator_stats["fresh_model_key"]
        ]["pyLIMAmodel_rr"]

    except Exception as error:
        true_generator_stats = true_generator_stats_fallback_from_primary_result(
            primary_result,
            true_model=true_model,
        )
        true_generator_stats["error"] = repr(error)
        true_generator_stats["traceback"] = traceback.format_exc()

    # If pyLIMA returned an error/NaNs instead of raising, keep the error
    # message but fill chi2/n_data from the already computed robust true chi2.
    try:
        chi2_is_bad = not np.isfinite(float(true_generator_stats.get("chi2", np.nan)))
    except Exception:
        chi2_is_bad = True

    if chi2_is_bad:
        fallback_stats = true_generator_stats_fallback_from_primary_result(
            primary_result,
            true_model=true_model,
        )
        for key, value in fallback_stats.items():
            if key not in true_generator_stats:
                true_generator_stats[key] = value
            else:
                try:
                    current = true_generator_stats[key]
                    if isinstance(current, float) and not np.isfinite(current):
                        true_generator_stats[key] = value
                except Exception:
                    pass

    _multi_timer.stop("true_generator")

    # ------------------------------------------------------------------
    # 6. LRT if requested.
    # ------------------------------------------------------------------
    _multi_timer.start("lrt")
    lrt_results = None

    if lrt_config is not None:
        if not isinstance(lrt_config, dict):
            raise TypeError("lrt_config debe ser un diccionario o None.")

        lrt_results = compute_lrt_from_multi_fits(
            multi_fit_results,
            null_key=lrt_config.get("null", "H0"),
            alternative_key=lrt_config.get("alternative", "H1"),
            delta_k=lrt_config.get("delta_k", None),
        )

        lrt_results = add_oracle_lrt_to_results(
            lrt_results,
            true_generator_stats=true_generator_stats,
            multi_fit_results=multi_fit_results,
        )

    _multi_timer.stop("lrt")

    # ------------------------------------------------------------------
    # 7. Save one compact parquet with H0/H1/true/LRT diagnostics.
    # ------------------------------------------------------------------
    if catalog_mode == "custom_system":
        nset_str = "custom"
    else:
        nset_str = extract_nset_string(
            path_GENULENS_set,
            default="manual",
        )

    multi_fit_summary_path = None
    multi_fit_summary_record = None

    if save_multi_fit_summary:
        extra_metadata = {
            "status": primary_result.get("status", ""),
            "sim_model": model,
            "truth_parallax": bool(truth_parallax),
            "use_roman": bool(use_roman),
            "use_rubin": bool(use_rubin),
            "catalog_mode": catalog_mode,
            "apply_photometric_filter": bool(apply_photometric_filter),
            "apply_detection_criteria": bool(apply_detection_criteria),
            "n_fit_points": int(n_fit_points),
            "initial_guess_default": repr(initial_guess),
            "elapsed_sec_multi_total": float(time.time() - tstart),
        }

        simulation_timings = getattr(
            true_model,
            "pipeline_timings",
            {},
        )

        if isinstance(simulation_timings, dict):
            extra_metadata.update(
                simulation_timings
            )

        extra_metadata.update(
            _multi_timer.snapshot()
        )

        _multi_timer.start("multifit_summary_save")
        multi_fit_summary_path, multi_fit_summary_record = save_multi_fit_summary_parquet(
            i=i,
            path_to_save_results=path_to_save_results,
            primary_fit_key=primary_fit_key,
            multi_fit_results=multi_fit_results,
            true_generator_stats=true_generator_stats,
            lrt_results=lrt_results,
            nset_str=nset_str,
            extra_metadata=extra_metadata,
        )
        _multi_timer.stop("multifit_summary_save")


    _multi_timer.stop("multifit_total")

    pipeline_timings = {}

    simulation_timings = getattr(
        true_model,
        "pipeline_timings",
        {},
    )

    if isinstance(simulation_timings, dict):
        pipeline_timings.update(
            simulation_timings
        )

    pipeline_timings.update(
        _multi_timer.snapshot()
    )

    # multi_fit_summary_record is the live record returned to the runner.
    # The on-disk parquet already contains all pre-save timings; the live
    # record additionally contains save + total.
    if isinstance(multi_fit_summary_record, dict):
        multi_fit_summary_record.update(
            pipeline_timings
        )

    # ------------------------------------------------------------------
    # 8. Merge with the primary sim_fit return payload.
    # ------------------------------------------------------------------
    primary_result.update({
        "multi_fit_status": "completed",
        "pipeline_timings": pipeline_timings,
        "fit_specs": fit_specs_use,
        "primary_fit_key": primary_fit_key,
        "multi_fit_results": multi_fit_results,
        "true_generator_likelihood_stats": true_generator_stats,
        "lrt_results": lrt_results,
        "multi_fit_summary_path": (
            str(multi_fit_summary_path)
            if multi_fit_summary_path is not None
            else None
        ),
        "multi_fit_summary_record": multi_fit_summary_record,
        "pyLIMAmodel_true_fit_window": true_model_for_likelihood,
    })

    if return_data:
        return primary_result

    # Historical no-return-data mode: return the primary objects, while all
    # multi-fit diagnostics are saved on disk.
    return (
        primary_result.get("fit_rr"),
        primary_result.get("pyLIMAmodel_rr"),
        primary_result.get("fit_roman"),
        primary_result.get("pyLIMAmodel_roman"),
    )
