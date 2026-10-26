"""Roman-only event detection used by the simulation.

Johnson et al. (2020) criteria on Roman W149 only:
* Delta chi2 = chi2(flat baseline) - chi2(true model) >= 300;
* at least n3sigma (default 6) consecutive points >= 3 sigma above the
  baseline flux.
The baseline flux is the catalogue ftotal_W149 of the true model.
selection_* / observation_mode_passed choose the telescopes used.
"""

import numpy as np


def observation_mode_passed(
    Roman_band,
    Rubin_band,
    use_roman=True,
    use_rubin=True,
):
    """
    Decide si hay datos válidos según los telescopios activos.
    """

    if use_roman and use_rubin:
        return Roman_band and Rubin_band

    if use_roman and not use_rubin:
        return Roman_band

    if use_rubin and not use_roman:
        return Rubin_band

    return False


def selection_data_available(
    Roman_band,
    Rubin_band,
    use_roman=True,
    use_rubin=True,
):
    """
    Data-availability gate for scientific event selection.

    Whenever Roman is active, the scientific sample is defined by
    Roman alone. Rubin availability must not determine whether the
    event enters the Roman-selected sample.

    Rubin-only workflows retain the historical generic behavior.
    """

    if use_roman:
        return bool(Roman_band)

    return observation_mode_passed(
        Roman_band,
        Rubin_band,
        use_roman=use_roman,
        use_rubin=use_rubin,
    )


def selection_telescopes(
    pyLIMA_telescopes,
    use_roman=True,
    use_rubin=True,
    roman_band_name="W149",
):
    """
    Telescopes allowed to contribute to the event-selection criterion.

    If Roman is active, selection uses W149 only.
    Rubin cannot strengthen or rescue Roman detectability.

    Rubin-only workflows use the non-Roman telescopes.
    """

    telescopes = list(pyLIMA_telescopes)

    if use_roman:
        return [
            tel
            for tel in telescopes
            if tel.name == roman_band_name
        ]

    if use_rubin:
        return [
            tel
            for tel in telescopes
            if tel.name != roman_band_name
        ]

    return []


def roman_detection_chi2_statistics(
    pyLIMA_model,
    pyLIMA_parameters,
    *,
    roman_band_name="W149",
):
    """
    Compute the Roman-only detection Delta-chi2 on one telescope.

    Definition:

        chi2_true =
            sum_i [
                (F_obs,i - F_true,i) / sigma_F,i
            ]**2

        chi2_flat =
            sum_i [
                (F_obs,i - F_baseline) / sigma_F,i
            ]**2

        delta_chi2 =
            chi2_flat - chi2_true

    where every sum uses EXACTLY the same surviving Roman points.

    The baseline is the already-materialized total baseline flux
    carried by the true pyLIMA parameter realization:

        pyLIMA_parameters[f"ftotal_{roman_band_name}"]

    This helper deliberately:

    - evaluates only the requested Roman telescope;
    - never includes Rubin data;
    - never fits either hypothesis;
    - never applies a t0/tE time window;
    - never masks or silently drops invalid points;
    - never reads ftotal from a catalog row;
    - never calls LRT/oracle machinery.

    Any malformed/non-finite Roman data are pipeline-integrity errors
    and raise rather than being silently ignored.
    """

    telescope_matches = [
        telescope
        for telescope in pyLIMA_model.event.telescopes
        if str(telescope.name) == str(roman_band_name)
    ]

    if len(telescope_matches) != 1:
        raise RuntimeError(
            "Roman detection chi2 requires exactly one telescope "
            f"named {roman_band_name!r}; found "
            f"{len(telescope_matches)}."
        )

    telescope = telescope_matches[0]

    if telescope.lightcurve is None:
        raise RuntimeError(
            f"Roman telescope {roman_band_name!r} has no lightcurve."
        )

    if len(telescope.lightcurve) == 0:
        raise RuntimeError(
            f"Roman telescope {roman_band_name!r} has zero data points."
        )

    lightcurve = telescope.lightcurve

    required_columns = (
        "flux",
        "err_flux",
    )

    missing_columns = [
        name
        for name in required_columns
        if name not in lightcurve.colnames
    ]

    if missing_columns:
        raise RuntimeError(
            "Roman detection chi2 is missing required lightcurve "
            f"columns for {roman_band_name!r}: {missing_columns}"
        )

    observed_flux = np.asarray(
        lightcurve["flux"].value,
        dtype=float,
    )

    err_flux = np.asarray(
        lightcurve["err_flux"].value,
        dtype=float,
    )

    if observed_flux.ndim != 1:
        raise RuntimeError(
            "Roman observed flux must be one-dimensional; "
            f"shape={observed_flux.shape}."
        )

    if err_flux.ndim != 1:
        raise RuntimeError(
            "Roman flux uncertainty must be one-dimensional; "
            f"shape={err_flux.shape}."
        )

    n_data = int(observed_flux.size)

    if n_data == 0:
        raise RuntimeError(
            f"Roman telescope {roman_band_name!r} has zero flux points."
        )

    if err_flux.size != n_data:
        raise RuntimeError(
            "Roman observed-flux/error length mismatch: "
            f"{n_data} vs {err_flux.size}."
        )

    if not np.all(np.isfinite(observed_flux)):
        raise RuntimeError(
            "Roman observed flux contains non-finite values."
        )

    if not np.all(np.isfinite(err_flux)):
        raise RuntimeError(
            "Roman flux uncertainty contains non-finite values."
        )

    if not np.all(err_flux > 0.0):
        raise RuntimeError(
            "Roman flux uncertainty must be strictly positive."
        )

    baseline_key = f"ftotal_{roman_band_name}"

    if baseline_key not in pyLIMA_parameters:
        raise RuntimeError(
            "Roman detection chi2 is missing baseline parameter "
            f"{baseline_key!r}."
        )

    baseline_flux = float(
        pyLIMA_parameters[baseline_key]
    )

    if not np.isfinite(baseline_flux):
        raise RuntimeError(
            f"Roman baseline flux {baseline_key!r} is non-finite."
        )

    microlensing_model = (
        pyLIMA_model.compute_the_microlensing_model(
            telescope,
            pyLIMA_parameters,
        )
    )

    if "photometry" not in microlensing_model:
        raise RuntimeError(
            "True microlensing model did not return "
            "'photometry'."
        )

    true_flux = np.asarray(
        microlensing_model["photometry"],
        dtype=float,
    )

    if true_flux.ndim != 1:
        raise RuntimeError(
            "Roman true-model flux must be one-dimensional; "
            f"shape={true_flux.shape}."
        )

    if true_flux.size != n_data:
        raise RuntimeError(
            "Roman observed/true-model length mismatch: "
            f"{n_data} vs {true_flux.size}."
        )

    if not np.all(np.isfinite(true_flux)):
        raise RuntimeError(
            "Roman true-model flux contains non-finite values."
        )

    residual_true = (
        observed_flux - true_flux
    ) / err_flux

    residual_flat = (
        observed_flux - baseline_flux
    ) / err_flux

    chi2_true = float(
        np.sum(
            residual_true * residual_true,
            dtype=np.float64,
        )
    )

    chi2_flat = float(
        np.sum(
            residual_flat * residual_flat,
            dtype=np.float64,
        )
    )

    delta_chi2 = float(
        chi2_flat - chi2_true
    )

    if not np.isfinite(chi2_true):
        raise RuntimeError(
            "Roman true-model chi2 is non-finite."
        )

    if not np.isfinite(chi2_flat):
        raise RuntimeError(
            "Roman flat-model chi2 is non-finite."
        )

    if not np.isfinite(delta_chi2):
        raise RuntimeError(
            "Roman detection Delta-chi2 is non-finite."
        )

    return {
        "roman_band_name": str(roman_band_name),
        "n_data": n_data,
        "baseline_flux": baseline_flux,
        "chi2_true": chi2_true,
        "chi2_flat": chi2_flat,
        "delta_chi2": delta_chi2,
    }


def roman_detection_n3sigma_statistics(
    pyLIMA_model,
    pyLIMA_parameters,
    *,
    roman_band_name="W149",
    nsigma=3.0,
):
    """
    Compute the Roman-only consecutive-nsigma detection statistic.

    A point is above threshold when

        F_obs >= F_baseline + nsigma * sigma_F

    where F_baseline is ftotal_<roman_band_name> from the true pyLIMA
    parameter realization.

    Only the requested Roman telescope is used. Points are ordered by
    observation time before the maximum consecutive run is computed.

    Invalid/non-finite inputs are pipeline-integrity errors and raise;
    points are never silently masked.
    """

    nsigma = float(nsigma)

    if not np.isfinite(nsigma) or nsigma <= 0.0:
        raise ValueError(
            f"nsigma must be finite and positive; got {nsigma!r}."
        )

    matches = [
        telescope
        for telescope in pyLIMA_model.event.telescopes
        if str(telescope.name) == str(roman_band_name)
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Roman n3sigma statistic requires exactly one telescope "
            f"named {roman_band_name!r}; found {len(matches)}."
        )

    telescope = matches[0]

    if telescope.lightcurve is None:
        raise RuntimeError(
            f"Roman telescope {roman_band_name!r} has no lightcurve."
        )

    if len(telescope.lightcurve) == 0:
        raise RuntimeError(
            f"Roman telescope {roman_band_name!r} has zero data points."
        )

    lightcurve = telescope.lightcurve

    required = (
        "time",
        "flux",
        "err_flux",
    )

    missing = [
        name
        for name in required
        if name not in lightcurve.colnames
    ]

    if missing:
        raise RuntimeError(
            "Roman n3sigma statistic is missing required columns: "
            f"{missing}"
        )

    times = np.asarray(
        lightcurve["time"].value,
        dtype=float,
    )

    observed_flux = np.asarray(
        lightcurve["flux"].value,
        dtype=float,
    )

    err_flux = np.asarray(
        lightcurve["err_flux"].value,
        dtype=float,
    )

    if (
        times.ndim != 1
        or observed_flux.ndim != 1
        or err_flux.ndim != 1
    ):
        raise RuntimeError(
            "Roman n3sigma inputs must be one-dimensional."
        )

    n_data = int(times.size)

    if n_data == 0:
        raise RuntimeError(
            "Roman n3sigma statistic has zero data points."
        )

    if (
        observed_flux.size != n_data
        or err_flux.size != n_data
    ):
        raise RuntimeError(
            "Roman n3sigma input-length mismatch."
        )

    if not np.all(np.isfinite(times)):
        raise RuntimeError(
            "Roman times contain non-finite values."
        )

    if not np.all(np.isfinite(observed_flux)):
        raise RuntimeError(
            "Roman observed flux contains non-finite values."
        )

    if not np.all(np.isfinite(err_flux)):
        raise RuntimeError(
            "Roman flux uncertainty contains non-finite values."
        )

    if not np.all(err_flux > 0.0):
        raise RuntimeError(
            "Roman flux uncertainty must be strictly positive."
        )

    baseline_key = f"ftotal_{roman_band_name}"

    if baseline_key not in pyLIMA_parameters:
        raise RuntimeError(
            "Roman n3sigma statistic is missing baseline parameter "
            f"{baseline_key!r}."
        )

    baseline_flux = float(
        pyLIMA_parameters[baseline_key]
    )

    if not np.isfinite(baseline_flux):
        raise RuntimeError(
            "Roman baseline flux is non-finite."
        )

    # Stable sort so the definition is explicitly chronological.
    order = np.argsort(
        times,
        kind="mergesort",
    )

    flux_sorted = observed_flux[order]
    err_sorted = err_flux[order]

    above = (
        flux_sorted
        >= baseline_flux + nsigma * err_sorted
    )

    max_run = 0
    current_run = 0

    for flag in above:

        if bool(flag):
            current_run += 1
            max_run = max(
                max_run,
                current_run,
            )

        else:
            current_run = 0

    return {
        "roman_band_name": str(roman_band_name),
        "n_data": n_data,
        "baseline_flux": baseline_flux,
        "n3sigma_nsigma": nsigma,
        "n3sigma_n_above": int(
            np.count_nonzero(above)
        ),
        "n3sigma_max_run": int(max_run),
    }


def combine_roman_detection_statistics(
    chi2_stats,
    n3sigma_stats,
    *,
    delta_chi2_min=300.0,
    n3sigma_min=6,
):
    """
    Combine the two Roman-only forecast detection criteria.

    Detection requires BOTH:

        Delta chi2 >= delta_chi2_min
        max consecutive n-sigma points >= n3sigma_min

    Boundaries are inclusive ("at least"), following the survey
    forecast definition.
    """

    delta_chi2_min = float(
        delta_chi2_min
    )

    n3sigma_min = int(
        n3sigma_min
    )

    if (
        not np.isfinite(delta_chi2_min)
        or delta_chi2_min < 0.0
    ):
        raise ValueError(
            "delta_chi2_min must be finite and non-negative."
        )

    if n3sigma_min < 1:
        raise ValueError(
            "n3sigma_min must be >= 1."
        )

    if (
        chi2_stats["roman_band_name"]
        != n3sigma_stats["roman_band_name"]
    ):
        raise RuntimeError(
            "Roman chi2/n3sigma band mismatch."
        )

    if (
        int(chi2_stats["n_data"])
        != int(n3sigma_stats["n_data"])
    ):
        raise RuntimeError(
            "Roman chi2/n3sigma data-count mismatch."
        )

    if (
        float(chi2_stats["baseline_flux"])
        != float(n3sigma_stats["baseline_flux"])
    ):
        raise RuntimeError(
            "Roman chi2/n3sigma baseline mismatch."
        )

    delta_chi2 = float(
        chi2_stats["delta_chi2"]
    )

    max_run = int(
        n3sigma_stats["n3sigma_max_run"]
    )

    pass_delta_chi2 = bool(
        delta_chi2 >= delta_chi2_min
    )

    pass_n3sigma = bool(
        max_run >= n3sigma_min
    )

    result = dict(
        chi2_stats
    )

    result.update(
        n3sigma_stats
    )

    result.update(
        {
            "delta_chi2_min": delta_chi2_min,
            "n3sigma_min": n3sigma_min,
            "pass_delta_chi2": pass_delta_chi2,
            "pass_n3sigma": pass_n3sigma,
            "roman_detected": bool(
                pass_delta_chi2
                and pass_n3sigma
            ),
        }
    )

    return result
