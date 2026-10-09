import numpy as np
import os, sys, re, copy, math, time, traceback, inspect
import pandas as pd
from pathlib import Path

# Get the directory where the script is located
script_dir = Path(__file__).parent
home_dir = os.path.expanduser("~")

from rubin_sim.phot_utils.photometric_parameters import PhotometricParameters
from rubin_sim.phot_utils.signaltonoise import calc_gamma, calc_mag_error_m5

# astropy
import astropy.units as u
from astropy.table import QTable

# --- Fix para IERS y sidereal_time de Astropy ---
from astropy.utils import iers
iers.conf.auto_max_age = None
iers.conf.auto_download = False  # No intenta descargar
iers.conf.iers_degraded_accuracy = 'warn'

from pyLIMA.simulations import simulator
from class_analysis import Analysis_Event
from ulens_params import microlensing_params, event_param, sample_from_spec
from ulens_params import event_param_from_pair_row
from fit_lc import fit_rubin_roman, parallax_suffix
from timing_utils import StageTimer
from photometry.roman_photometry import apply_roman_f146_photometry
from photometry.constants import (
    PYLIMA_FIT_ZERO_POINT,
    SIMULATION_BAND_ZERO_POINTS,
)
from simulation.realization import (
    realization_from_catalog,
    realization_from_generated_parameters,
    data_has_materialized_blend_ratio,
)
from simulation.parallax_cache import (
    install_earth_ephemerides_cache,
)
from simulation.core import simulate_light_curve
# ROMAN_F146_PANDEIA_INTEGRATION_V2
from detection_criteria import filter5points, deviation_from_constant, has_consecutive_numbers, filter_band, mag, debug_nsigma_global
from read_save import save_sim, save_fit, read_data
from set_model_pyLIMA import (
    model_choice,
    parameters_model,
    flux_parameters_model,
    flux_parameters_from_blend_ratio,
    normalize_model_name,
)

# # ================================================================
# #  Guardado a Parquet
# # ================================================================
from utils.io import save_dict_as_parquet as _save_dict_as_parquet


from set_telescopes_pyLIMA import tel_roman_rubin

# ================================================================
#  Photometric parameters
# ================================================================
def set_photometric_parameters(exptime, nexp, readnoise=None):
    # readnoise = None will use the default (8.8 e/pixel). Readnoise should be in electrons/pixel.
    photParams = PhotometricParameters(exptime=exptime, nexp=nexp, readnoise=readnoise)
    return photParams





# ================================================================
# Rubin photometric gamma cache
# ================================================================
#
# calc_gamma() is expensive because Rubin Sim constructs and
# normalizes an SED and integrates it through the bandpass.
#
# For a fixed Rubin observing schedule, bandpass and photometric
# configuration, the m5 vector is identical between microlensing
# events. Therefore the corresponding gamma vector can be computed
# once per process and reused.
#
# The scientific calculation remains Rubin Sim's own calc_gamma()
# and calc_mag_error_m5().  This cache only avoids recomputing gamma
# for identical inputs.
# ================================================================

_RUBIN_GAMMA_VECTOR_CACHE = {}

_RUBIN_GAMMA_VECTOR_CACHE_STATS = {
    "vector_hits": 0,
    "vector_misses": 0,
    "gamma_values_computed": 0,
}


def _rubin_photometric_parameters_signature(
    phot_params,
):
    """
    Stable process-local signature of PhotometricParameters state.

    A new PhotometricParameters object is created by sim_event()
    for each event, so object identity cannot be used for caching.
    """

    return tuple(
        sorted(
            (
                str(name),
                type(value).__module__,
                type(value).__qualname__,
                repr(value),
            )
            for name, value
            in vars(phot_params).items()
        )
    )


def _rubin_m5_vector_signature(
    m5_values,
):
    """
    Exact float64 signature of a Rubin m5 vector.
    """

    values = np.ascontiguousarray(
        np.asarray(
            m5_values,
            dtype=np.float64,
        )
    )

    return (
        values.shape,
        values.tobytes(),
    )


def _get_cached_rubin_gamma_vector(
    band_name,
    bandpass,
    m5_values,
    phot_params,
):
    """
    Return the exact Rubin-Sim gamma vector for one band/schedule.

    On a cache miss, every gamma value is calculated with the
    official rubin_sim.phot_utils.signaltonoise.calc_gamma().
    Subsequent events with exactly the same inputs reuse that vector.

    The cached value retains a strong reference to the Bandpass
    object, so Python object-id reuse cannot produce a false hit.
    """

    m5_array = np.asarray(
        m5_values,
        dtype=np.float64,
    )

    key = (
        str(band_name),
        id(bandpass),
        _rubin_photometric_parameters_signature(
            phot_params
        ),
        _rubin_m5_vector_signature(
            m5_array
        ),
    )

    cached = (
        _RUBIN_GAMMA_VECTOR_CACHE.get(
            key
        )
    )

    if cached is not None:

        cached_bandpass, gamma_values = (
            cached
        )

        if cached_bandpass is bandpass:

            _RUBIN_GAMMA_VECTOR_CACHE_STATS[
                "vector_hits"
            ] += 1

            return gamma_values

    gamma_values = np.empty(
        len(m5_array),
        dtype=float,
    )

    for index, m5_value in enumerate(
        m5_array
    ):

        gamma_values[index] = calc_gamma(
            bandpass,
            float(m5_value),
            phot_params,
        )

    # Protect cached data against accidental mutation.
    gamma_values.setflags(
        write=False
    )

    _RUBIN_GAMMA_VECTOR_CACHE[
        key
    ] = (
        bandpass,
        gamma_values,
    )

    _RUBIN_GAMMA_VECTOR_CACHE_STATS[
        "vector_misses"
    ] += 1

    _RUBIN_GAMMA_VECTOR_CACHE_STATS[
        "gamma_values_computed"
    ] += len(
        gamma_values
    )

    return gamma_values


def get_rubin_gamma_vector_cache_stats():
    """
    Return process-local Rubin gamma-cache diagnostics.
    """

    return {
        **_RUBIN_GAMMA_VECTOR_CACHE_STATS,
        "cached_vectors":
            len(
                _RUBIN_GAMMA_VECTOR_CACHE
            ),
        "cached_gamma_values":
            sum(
                len(value[1])
                for value
                in _RUBIN_GAMMA_VECTOR_CACHE.values()
            ),
    }


def reset_rubin_gamma_vector_cache():
    """
    Clear the process-local Rubin gamma-vector cache.
    """

    _RUBIN_GAMMA_VECTOR_CACHE.clear()

    _RUBIN_GAMMA_VECTOR_CACHE_STATS[
        "vector_hits"
    ] = 0

    _RUBIN_GAMMA_VECTOR_CACHE_STATS[
        "vector_misses"
    ] = 0

    _RUBIN_GAMMA_VECTOR_CACHE_STATS[
        "gamma_values_computed"
    ] = 0


def _plain_array(x, dtype=None):
    """
    Convierte columnas astropy con unidades, listas o arrays a np.array.
    """
    if hasattr(x, "value"):
        x = x.value
    return np.asarray(x, dtype=dtype)


def _data_slice_column(dataSlice, candidates):
    """
    Devuelve la primera columna existente en dataSlice entre candidates.
    Soporta numpy structured arrays y astropy tables.
    """
    if dataSlice is None:
        return None

    names = []

    if hasattr(dataSlice, "dtype") and getattr(dataSlice.dtype, "names", None) is not None:
        names = list(dataSlice.dtype.names)
    elif hasattr(dataSlice, "colnames"):
        names = list(dataSlice.colnames)
    elif hasattr(dataSlice, "columns"):
        names = list(dataSlice.columns)

    for name in candidates:
        if name in names:
            return dataSlice[name]

    return None


def _as_band_limit_array(limit, band_name, n, default=np.nan):
    """
    Convierte un límite de saturación a array por punto.

    limit puede ser:
        None
        float
        dict por banda, por ejemplo {"g": 16.0, "r": 15.8}
        array-like de longitud n
    """
    if limit is None:
        return np.full(n, default, dtype=float)

    if isinstance(limit, dict):
        value = limit.get(band_name, default)
        return np.full(n, float(value), dtype=float)

    try:
        arr = np.asarray(limit, dtype=float)
        if arr.ndim == 0:
            return np.full(n, float(arr), dtype=float)
        if len(arr) == n:
            return arr.astype(float)
    except Exception:
        pass

    return np.full(n, float(limit), dtype=float)


def _append_photometry_flag_columns(
    data_tbl,
    model_mag,
    m5,
    saturation_mag=None,
):
    """
    Agrega columnas diagnósticas de fotometría.

    Convención:
        too_faint_5sigma: model_mag > m5
        saturated: model_mag < saturation_mag, si saturation_mag es finito
        photometry_keep: no saturado y no más débil que m5
    """
    model_mag = np.asarray(model_mag, dtype=float)
    m5 = np.asarray(m5, dtype=float)

    if saturation_mag is None:
        saturation_mag = np.full(len(model_mag), np.nan, dtype=float)
    else:
        saturation_mag = np.asarray(saturation_mag, dtype=float)

    too_faint = model_mag > m5
    saturated = (
        np.isfinite(saturation_mag)
        & (model_mag < saturation_mag)
    )
    keep = (~too_faint) & (~saturated)

    data_tbl["mag_model"] = model_mag
    data_tbl["photometry_m5"] = m5
    data_tbl["photometry_saturation_mag"] = saturation_mag
    data_tbl["photometry_too_faint_5sigma"] = too_faint
    data_tbl["photometry_saturated"] = saturated
    data_tbl["photometry_keep"] = keep
    data_tbl["photometry_rejected"] = ~keep

    reason = np.full(len(model_mag), "ok", dtype=object)
    reason[too_faint] = "too_faint_5sigma"
    reason[saturated] = "saturated"
    reason[too_faint & saturated] = "saturated_and_too_faint"
    data_tbl["photometry_flag_reason"] = reason

    return data_tbl


def _summarize_flag_arrays(prefix, data_tbl):
    """
    Resumen de flags para una curva individual.
    """
    if data_tbl is None or len(data_tbl) == 0:
        return {
            f"{prefix}_n_total": 0,
            f"{prefix}_n_keep": 0,
            f"{prefix}_n_too_faint_5sigma": 0,
            f"{prefix}_n_saturated": 0,
            f"{prefix}_n_rejected": 0,
        }

    names = getattr(data_tbl, "colnames", [])

    n_total = len(data_tbl)

    def count_col(col):
        if col not in names:
            return 0
        return int(np.sum(_plain_array(data_tbl[col], dtype=bool)))

    n_keep = count_col("photometry_keep")
    n_faint = count_col("photometry_too_faint_5sigma")
    n_sat = count_col("photometry_saturated")
    n_rej = count_col("photometry_rejected")

    return {
        f"{prefix}_n_total": int(n_total),
        f"{prefix}_n_keep": int(n_keep),
        f"{prefix}_n_too_faint_5sigma": int(n_faint),
        f"{prefix}_n_saturated": int(n_sat),
        f"{prefix}_n_rejected": int(n_rej),
    }


def summarize_photometry_flags(pyLIMA_model):
    """
    Resume los flags de fotometría guardados en las curvas de luz.
    Devuelve columnas listas para sumar a event_params / parquet.
    """
    summary = {
        "phot_n_total": 0,
        "phot_n_keep": 0,
        "phot_n_too_faint_5sigma": 0,
        "phot_n_saturated": 0,
        "phot_n_rejected": 0,
    }

    if pyLIMA_model is None:
        return summary

    for tel in pyLIMA_model.event.telescopes:
        band = str(tel.name)
        band_prefix = f"phot_{band}"
        band_summary = _summarize_flag_arrays(
            band_prefix,
            getattr(tel, "lightcurve", None),
        )
        summary.update(band_summary)

        summary["phot_n_total"] += band_summary[f"{band_prefix}_n_total"]
        summary["phot_n_keep"] += band_summary[f"{band_prefix}_n_keep"]
        summary["phot_n_too_faint_5sigma"] += band_summary[f"{band_prefix}_n_too_faint_5sigma"]
        summary["phot_n_saturated"] += band_summary[f"{band_prefix}_n_saturated"]
        summary["phot_n_rejected"] += band_summary[f"{band_prefix}_n_rejected"]

    n_total = summary["phot_n_total"]
    if n_total > 0:
        summary["phot_frac_too_faint_5sigma"] = summary["phot_n_too_faint_5sigma"] / n_total
        summary["phot_frac_saturated"] = summary["phot_n_saturated"] / n_total
        summary["phot_frac_rejected"] = summary["phot_n_rejected"] / n_total
    else:
        summary["phot_frac_too_faint_5sigma"] = np.nan
        summary["phot_frac_saturated"] = np.nan
        summary["phot_frac_rejected"] = np.nan

    return summary


def photometry_summary_columns():
    """
    Columnas que conviene propagar al parquet true.
    """
    base = [
        "apply_photometric_filter",
        "phot_n_total",
        "phot_n_keep",
        "phot_n_too_faint_5sigma",
        "phot_n_saturated",
        "phot_n_rejected",
        "phot_frac_too_faint_5sigma",
        "phot_frac_saturated",
        "phot_frac_rejected",
    ]

    bands = ["W149", "u", "g", "r", "i", "z", "y"]
    suffixes = [
        "n_total",
        "n_keep",
        "n_too_faint_5sigma",
        "n_saturated",
        "n_rejected",
    ]

    for band in bands:
        for suffix in suffixes:
            base.append(f"phot_{band}_{suffix}")

    return base


def apply_roman_rubin_photometry(
    new_creation,
    ZP,
    dataSlice,
    LSST_BandPass,
    photParams,
    roman_band_name="W149",
    roman_original_zp=27.4,
    roman_m5=27.6,
    apply_photometric_filter=True,
    rubin_saturation_mag=None,
    roman_saturation_mag=None,
):
    """
    Aplica fotometría simulada a los telescopios activos.

    Si apply_photometric_filter=True:
        mantiene el comportamiento viejo y aplica filter_band().

    Si apply_photometric_filter=False:
        no elimina puntos por m5/saturación; conserva todos los puntos
        y agrega flags diagnósticos:
            photometry_too_faint_5sigma
            photometry_saturated
            photometry_keep
            photometry_rejected
            photometry_flag_reason

    Nota:
        - too_faint_5sigma se calcula con la magnitud del modelo sin ruido.
        - saturated usa dataSlice si contiene una columna de saturación reconocida,
          o rubin_saturation_mag si se pasa explícitamente.
    """

    Roman_band = False
    Rubin_band = False

    for telo in new_creation.telescopes:

        if telo.name == roman_band_name:

            telo = apply_roman_f146_photometry(
                telescope=telo,
                zero_point=ZP[telo.name],
                apply_photometric_filter=apply_photometric_filter,
            )

            if len(telo.lightcurve["mag"]) != 0:
                Roman_band = True

        else:

            if dataSlice is None:
                telo.lightcurve = telo.lightcurve[:0]
                continue

            band_name = telo.name

            X = _plain_array(telo.lightcurve["time"], dtype=float)
            ym = mag(ZP[band_name], _plain_array(telo.lightcurve["flux"], dtype=float))

            mask = dataSlice["filter"] == band_name
            m5_all = np.asarray(dataSlice["fiveSigmaDepth"][mask], dtype=float)

            if len(m5_all) == 0:
                telo.lightcurve = telo.lightcurve[:0]
                continue

            sat_col = _data_slice_column(
                dataSlice,
                [
                    "saturation_mag",
                    "saturationMag",
                    "sat_mag",
                    "satMag",
                ],
            )

            if sat_col is not None:
                sat_all = np.asarray(sat_col[mask], dtype=float)
            else:
                sat_all = None

            err_mag = []
            obs_mag = []
            obs_time = []
            M5 = []
            SAT = []

            n_pts = len(ym)
            n_m5 = len(m5_all)
            n_sat = len(sat_all) if sat_all is not None else 0

            gamma_all = _get_cached_rubin_gamma_vector(
                band_name,
                LSST_BandPass[band_name],
                m5_all,
                photParams,
            )

            saturation_fallback = _as_band_limit_array(
                rubin_saturation_mag,
                band_name,
                n_pts,
                default=np.nan,
            )

            for k in range(n_pts):

                idx_m5 = min(k, n_m5 - 1)
                m5_k = float(m5_all[idx_m5])

                if sat_all is not None and n_sat > 0:
                    idx_sat = min(k, n_sat - 1)
                    sat_k = float(sat_all[idx_sat])
                else:
                    sat_k = float(saturation_fallback[k])

                magerr_k = calc_mag_error_m5(
                    float(ym[k]),
                    LSST_BandPass[band_name],
                    m5_k,
                    photParams,
                    gamma=float(gamma_all[idx_m5]),
                )[0]

                err_mag.append(magerr_k)
                obs_mag.append(np.random.normal(ym[k], magerr_k))
                obs_time.append(X[k])
                M5.append(m5_k)
                SAT.append(sat_k)

            data_tbl = QTable(
                [
                    telo.lightcurve["err_flux"].value,
                    np.array(err_mag),
                    telo.lightcurve["flux"].value,
                    telo.lightcurve["inv_err_flux"].value,
                    np.array(M5),
                    np.array(obs_mag),
                    np.array(obs_time),
                ],
                names=(
                    "err_flux",
                    "err_mag",
                    "flux",
                    "inv_err_flux",
                    "m5",
                    "mag",
                    "time",
                ),
            )

            data_tbl = _append_photometry_flag_columns(
                data_tbl,
                model_mag=ym,
                m5=np.array(M5),
                saturation_mag=np.array(SAT),
            )

            if apply_photometric_filter:
                keep = _plain_array(
                    data_tbl["photometry_keep"],
                    dtype=bool,
                )
                telo.lightcurve = data_tbl[keep]
            else:
                telo.lightcurve = data_tbl

            if len(telo.lightcurve["mag"]) != 0:
                Rubin_band = True

    return new_creation, Roman_band, Rubin_band

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


def inject_model_flux_for_ground_telescopes(
    new_creation,
    pyLIMA_model,
    pyLIMA_parameters,
    roman_band_name="W149",
):
    """
    Refuerza el flujo teórico del modelo para telescopios Rubin.

    Roman también contiene ahora flujo teórico porque
    simulator.simulate_lightcurve() se llama con add_noise=False.
    Su ruido instrumental se agrega después con RomanF146Noise.
    """

    for tel in new_creation.telescopes:

        if tel.name == roman_band_name:
            continue

        model_flux = pyLIMA_model.compute_the_microlensing_model(
            tel,
            pyLIMA_parameters,
        )["photometry"]

        tel.lightcurve["flux"] = model_flux

    return new_creation


# ============================================================
# Fast synchronization of pyLIMA time-indexed geometry
# after photometric filtering.
#
# Photometric filtering only removes observation rows; it does not
# alter their times. Therefore already-computed parallax geometry can
# be restricted to the surviving rows instead of recomputed.
# ============================================================

_PARALLAX_TIME_ARRAY_AXES = {
    "Earth_positions": 0,
    "Earth_speeds": 0,
    "sidereal_times": 0,
    "telescope_positions": 0,
    "Earth_positions_projected": 1,
    "Earth_speeds_projected": 1,
    "deltas_positions": 1,
}


def _lightcurve_time_values(telescope):
    times = telescope.lightcurve["time"]

    if hasattr(times, "value"):
        times = times.value

    return np.asarray(times, dtype=float)


def _capture_prefilter_telescope_times(event):
    """Save only observation times needed to reconstruct filter masks."""

    out = {}

    for tel in event.telescopes:

        if tel.lightcurve is None:
            continue

        out[tel.name] = _lightcurve_time_values(
            tel
        ).copy()

    return out


def _mask_time_indexed_telescope_arrays(
    event,
    prefilter_times,
):
    """
    Restrict already-computed pyLIMA time-indexed arrays to the
    photometric rows that survived filtering.

    This is mathematically equivalent to recomputing parallax at the
    surviving times, but avoids the expensive ephemeris calculation.

    Fail loudly if filtering changed times rather than only removing
    rows.
    """

    for tel in event.telescopes:

        if tel.lightcurve is None or len(tel.lightcurve) == 0:
            continue

        if tel.name not in prefilter_times:
            raise RuntimeError(
                "Missing pre-filter telescope times: "
                f"{tel.name}"
            )

        before = np.asarray(
            prefilter_times[tel.name],
            dtype=float,
        )

        after = _lightcurve_time_values(tel)

        n_before = len(before)
        n_after = len(after)

        if n_after > n_before:
            raise RuntimeError(
                "Photometric filtering increased point count: "
                f"{tel.name}: {n_before} -> {n_after}"
            )

        # Fast no-op when nothing was removed.
        if (
            n_before == n_after
            and np.array_equal(before, after)
        ):
            continue

        # Filtering preserves order, so surviving rows are recovered
        # exactly by indexing into the original sorted timestamps.
        idx = np.searchsorted(before, after)

        if np.any(idx < 0) or np.any(idx >= n_before):
            raise RuntimeError(
                "Filtered times are not a subset of original times: "
                f"{tel.name}"
            )

        if not np.array_equal(before[idx], after):
            max_diff = float(
                np.max(np.abs(before[idx] - after))
            )

            raise RuntimeError(
                "Photometric filtering modified observation times "
                f"for {tel.name}; max difference={max_diff}"
            )

        for attr, axis in _PARALLAX_TIME_ARRAY_AXES.items():

            if not hasattr(tel, attr):
                continue

            value = getattr(tel, attr)

            if value is None:
                continue

            if isinstance(value, dict):

                for key, item in list(value.items()):

                    if item is None:
                        continue

                    shape = np.shape(item)

                    if len(shape) <= axis:
                        continue

                    n_axis = shape[axis]

                    if n_axis == n_before:
                        value[key] = np.take(
                            item,
                            idx,
                            axis=axis,
                        )

                    elif n_axis == n_after:
                        # Already synchronized by another operation.
                        continue

                    else:
                        raise RuntimeError(
                            "Unexpected time-axis length: "
                            f"{tel.name}.{attr}[{key!r}] "
                            f"has {n_axis}; expected "
                            f"{n_before} or {n_after}"
                        )

            else:

                shape = np.shape(value)

                if len(shape) <= axis:
                    continue

                n_axis = shape[axis]

                if n_axis == n_before:
                    setattr(
                        tel,
                        attr,
                        np.take(
                            value,
                            idx,
                            axis=axis,
                        ),
                    )

                elif n_axis == n_after:
                    continue

                else:
                    raise RuntimeError(
                        "Unexpected time-axis length: "
                        f"{tel.name}.{attr} has {n_axis}; "
                        f"expected {n_before} or {n_after}"
                    )



# Exact cache of Earth barycentric ephemerides used by pyLIMA.
# Unknown time arrays still fall back to the original Astropy-backed
# implementation. This touches no RNG state.
install_earth_ephemerides_cache()

def sim_event(
    i,
    data,
    path_ephemerides,
    model,
    time_window=None,
    use_roman=True,
    use_rubin=True,
    truth_parallax=True,
    rubin_pointing_mode="fixed",
    rubin_cache_cell_deg=None,
    apply_detection_criteria=True,
    apply_photometric_filter=True,
    rubin_saturation_mag=None,
    roman_saturation_mag=None,
):
    """
    i (int): índice del evento
    data (dict): parámetros, incluyendo magnitudes de la estrella
    path_ephemerides (str): path a las efemérides de Roman
    model (str): modelo de microlente ('USBL','FSPL','PSPL', etc.)

    rubin_pointing_mode:
        "fixed"  -> usa el campo fijo Roman/Rubin.
        "source" -> usa data["maf_ra"], data["maf_dec"] o data["ra"], data["dec"].
                    Pensado para Rubin-only + AstroDataLab pairs.
    """

    _sim_timer = StageTimer()
    _sim_timer.start("simulation_total")
    _sim_timer.start("simulation_prep")

    def _finish_sim_event(
        model_obj,
        params_obj,
        decision_obj,
    ):
        _sim_timer.stop("simulation_total")

        payload = _sim_timer.snapshot()

        if model_obj is not None:
            setattr(
                model_obj,
                "pipeline_timings",
                payload,
            )

        return (
            model_obj,
            params_obj,
            decision_obj,
        )

    # ============================================================
    # Magnitudes de la fuente
    # ============================================================
    # Solo exigimos las bandas necesarias según use_roman/use_rubin.
    # Esto permite correr Roman-only con solo W149, o Rubin-only sin W149
    # en modos custom_system.

    magstar = {}

    if use_roman:

        if "W149" not in data:
            raise KeyError(
                "use_roman=True pero data no contiene magnitud 'W149'."
            )

        magstar["W149"] = data["W149"]

    if use_rubin:

        for band in ["u", "g", "r", "i", "z"]:

            if band not in data:
                raise KeyError(
                    f"use_rubin=True pero data no contiene magnitud '{band}'."
                )

            magstar[band] = data[band]

        if "Y" in data:
            magstar["y"] = data["Y"]

        elif "y" in data:
            magstar["y"] = data["y"]

        else:
            raise KeyError(
                "use_rubin=True pero data no contiene magnitud 'Y' ni 'y'."
            )

    ZP = SIMULATION_BAND_ZERO_POINTS

    t0 = data["t0"]

    # ============================================================
    # Coordenadas para Rubin/MAF
    # ============================================================
    # Regla:
    # - Roman+Rubin: siempre campo fijo.
    # - Rubin-only + rubin_pointing_mode="source": usa coordenadas de la fuente.
    # - Default: campo fijo.

    source_ra = data.get("maf_ra", data.get("ra", None))
    source_dec = data.get("maf_dec", data.get("dec", None))

    if rubin_pointing_mode == "source":

        if source_ra is None or source_dec is None:
            raise ValueError(
                "rubin_pointing_mode='source' requiere ra/dec "
                "o maf_ra/maf_dec."
            )

        rubin_pointing_mode_use = "source"
        tel_ra = float(source_ra)
        tel_dec = float(source_dec)

    elif rubin_pointing_mode == "fixed":

        rubin_pointing_mode_use = "fixed"
        tel_ra = None
        tel_dec = None

    else:
        raise ValueError(
            f"rubin_pointing_mode inválido: {rubin_pointing_mode}"
        )

    _sim_timer.stop("simulation_prep")
    _sim_timer.start("simulation_maf")

    my_own_creation, dataSlice, LSST_BandPass = tel_roman_rubin(
        path_ephemerides,
        time_window=time_window,
        use_roman=use_roman,
        use_rubin=use_rubin,
        Ra=tel_ra,
        Dec=tel_dec,
        rubin_pointing_mode=rubin_pointing_mode_use,
        rubin_cache_cell_deg=rubin_cache_cell_deg,
    )
    _sim_timer.stop("simulation_maf")


    photParams = set_photometric_parameters(15, 2)
    _sim_timer.start("simulation_deepcopy")
    new_creation = copy.deepcopy(my_own_creation)
    _sim_timer.stop("simulation_deepcopy")
    np.random.seed(i)

    # ============================================================
    # Eliminar bandas sin tiempos dentro de la ventana temporal
    # ============================================================

    _sim_timer.start("simulation_remove_empty_telescopes")
    new_creation, removed_telescopes = remove_empty_telescopes(
        new_creation,
        verbose=True,
    )
    _sim_timer.stop("simulation_remove_empty_telescopes")


    # Si no quedó ninguna banda, el evento no puede simularse
    if len(new_creation.telescopes) == 0:

        print(
            "Event cannot be simulated because there are no observations "
            "inside the requested time window."
        )

        return _finish_sim_event(None, {}, False)


    # ============================================================
    # Construir el modelo solo con las bandas que tienen datos
    # ============================================================
# ============================================================
# Construir el modelo solo con las bandas que tienen datos
# ============================================================

    _sim_timer.start("simulation_model_setup")
    if truth_parallax:
        parallax_arg = ["Full", t0]
    else:
        parallax_arg = ["None", 0.0]

    band_order = [
        tel.name
        for tel in new_creation.telescopes
    ]

    # Event realization: blending and caustic_origin are decided here,
    # from `data`, before any pyLIMA model is built. Two independent
    # decisions (do not couple them -- see simulation/realization.py):
    #
    # 1. Blending: data_has_materialized_blend_ratio is purely about
    #    whether blend_ratio_<band> is already materialized in `data`
    #    -- it says nothing about `catalog_mode` or "is this a catalog
    #    event". `data` without it for every band -- true for every
    #    caller today -- goes through realization_from_generated_
    #    parameters, which leaves blending undecided (sampled below,
    #    exactly as before this refactor).
    # 2. caustic_origin (USBL only): resolved independently by each
    #    producer. realization_from_catalog requires it explicit,
    #    always. realization_from_generated_parameters uses it if
    #    `data` happens to provide it, and otherwise leaves it None --
    #    "not yet decided" -- handled right below by falling back to
    #    the exact historical random-origin mechanism.
    _sim_timer.start("simulation_model_realization")

    if data_has_materialized_blend_ratio(data, band_order):
        realization = realization_from_catalog(
            data,
            model,
            truth_parallax,
            t0,
            band_order,
        )
    else:
        realization = realization_from_generated_parameters(
            data,
            model,
            truth_parallax,
            t0,
            band_order,
        )

    _sim_timer.stop("simulation_model_realization")

    if realization.caustic_origin is not None:
        # Explicit origin (from either producer): never sampled.
        caustic_origin_arg = [realization.caustic_origin, [0, 0]]
        bl_random_origin = False
    else:
        # Not USBL, or legacy/generated USBL with no explicit origin
        # in `data`: preserve EXACTLY the historical random-origin
        # mechanism (np.random.choice inside
        # set_model_pyLIMA.choose_usbl_origin, via model_choice),
        # unmoved, at the same point in the RNG sequence it has always
        # occupied -- before flux_parameters_model's blending draws.
        caustic_origin_arg = None
        bl_random_origin = True

    _sim_timer.start("simulation_model_choice")

    my_own_model = model_choice(
        new_creation,
        model,
        parallax=parallax_arg,
        BL_random_origin=bl_random_origin,
        BL_origin=caustic_origin_arg,
    )

    _sim_timer.stop("simulation_model_choice")

    _sim_timer.start("simulation_model_parameters")

    params, param_order = parameters_model(
        realization.physical_params,
        my_own_model,
    )

    my_own_parameters = [
        params[key]
        for key in param_order
    ]

    _sim_timer.stop("simulation_model_parameters")

    _sim_timer.start("simulation_model_flux")

    if realization.blend_ratio is not None:
        # Explicit realization path: blend_ratio already decided (by
        # catalog.blending). Not the LRT monkey-patch point.
        my_own_flux_parameters, fs, G, F = flux_parameters_from_blend_ratio(
            magstar,
            ZP,
            my_own_model,
            band_order,
            realization.blend_ratio,
        )
    else:
        # Legacy/generated/LRT path: EXACT historical signature and
        # call site. This global is what LRT monkey-patches
        # (functions_roman_rubin.flux_parameters_model = ...), and its
        # replacement does not accept blend_ratio or **kwargs -- do
        # not add any keyword here.
        my_own_flux_parameters, fs, G, F = flux_parameters_model(
            magstar,
            ZP,
            my_own_model,
            band_order=band_order,
        )

    my_own_parameters += my_own_flux_parameters

    _sim_timer.stop("simulation_model_flux")
    _sim_timer.stop("simulation_model_setup")

    # ============================================================
    # Simulación de la curva de luz (núcleo genérico, sin RNG,
    # sin conocimiento de catálogo ni de la encuesta)
    # ============================================================

    pyLIMA_parameters = simulate_light_curve(
        my_own_model,
        my_own_parameters,
        sim_timer=_sim_timer,
    )

    # Roman-specific: reinforce the theoretical flux for non-Roman
    # (ground) telescopes, since simulate_lightcurve(add_noise=False)
    # only fills Roman's own lightcurve table by itself. Stays here,
    # in the adapter, using the existing helper -- not in the generic
    # core.
    _sim_timer.start("simulation_ground_flux")
    new_creation = inject_model_flux_for_ground_telescopes(
        new_creation,
        my_own_model,
        pyLIMA_parameters,
        roman_band_name="W149",
    )
    _sim_timer.stop("simulation_ground_flux")

    # ============================================================
    # Fotometría Roman + Rubin
    # ============================================================
    
    _sim_timer.start("simulation_parallax_snapshot")

    if truth_parallax:
        _parallax_prefilter_times = (
            _capture_prefilter_telescope_times(
                my_own_model.event
            )
        )
    else:
        _parallax_prefilter_times = None

    _sim_timer.stop("simulation_parallax_snapshot")

    _sim_timer.start("simulation_photometry")
    new_creation, Roman_band, Rubin_band = apply_roman_rubin_photometry(
        new_creation,
        ZP,
        dataSlice,
        LSST_BandPass,
        photParams,
        apply_photometric_filter=apply_photometric_filter,
        rubin_saturation_mag=rubin_saturation_mag,
        roman_saturation_mag=roman_saturation_mag,
    )
    _sim_timer.stop("simulation_photometry")

    # ============================================================
    # Synchronize pyLIMA after photometric filtering
    # ============================================================
    # The lightcurve tables have already been filtered.
    # pyLIMA's previously computed parallax arrays may still
    # correspond to the original observation times.
    #
    # Keep the same Event and the same noisy photometry.
    # No event parameters or random numbers are regenerated.
    # ============================================================

    _sim_timer.start("simulation_parallax_resync")

    # Remove telescopes without usable photometry.
    # This simulation pipeline contains photometry only.
    my_own_model.event.telescopes = [
        tel
        for tel in my_own_model.event.telescopes
        if (
            tel.lightcurve is not None
            and len(tel.lightcurve) > 0
        )
    ]

    if truth_parallax:

        # Parallax geometry was already calculated at the original
        # observation times. Photometric filtering only removes rows,
        # so synchronize all time-indexed pyLIMA arrays by applying
        # exactly the same surviving-time mask instead of recomputing
        # the ephemeris geometry.
        _mask_time_indexed_telescope_arrays(
            my_own_model.event,
            _parallax_prefilter_times,
        )

        for tel in my_own_model.event.telescopes:

            shifts = np.asarray(
                tel.deltas_positions["photometry"]
            )

            expected = (2, len(tel.lightcurve))

            if shifts.shape != expected:
                raise RuntimeError(
                    "Parallax/photometry mismatch after masking: "
                    f"{tel.name}: "
                    f"{shifts.shape} != {expected}"
                )

    _sim_timer.stop("simulation_parallax_resync")


    # ============================================================
    # Synchronize magnitude and flux representations
    # ============================================================
    #
    # Roman and Rubin instrumental noise has already been applied
    # in magnitude space.  Make flux / err_flux consistent before
    # any detection criterion or likelihood calculation.
    #
    # PHOTOMETRY_FLUX_SYNC_BEFORE_DETECTION_V1
    replace_flux_by_noisy_magnitude_flux_ZP(
        my_own_model,
        verbose=False,
    )


    # ============================================================
    # Criterios de detección
    # ============================================================
    _sim_timer.start("simulation_validation")
    has_required_data = observation_mode_passed(
        Roman_band,
        Rubin_band,
        use_roman=use_roman,
        use_rubin=use_rubin,
    )
    _sim_timer.stop("simulation_validation")


    if has_required_data:

        # --------------------------------------------------------
        # Opción para ignorar el criterio de detección
        # --------------------------------------------------------
        if not apply_detection_criteria:
            print(
                "Detection criterion disabled. "
                "The event has valid data and will be fitted."
            )

            return _finish_sim_event(my_own_model, pyLIMA_parameters, True)

        # --------------------------------------------------------
        # Criterio de detección original
        # --------------------------------------------------------
        _sim_timer.start("simulation_detection")
        decision, res = deviation_from_constant(
            pyLIMA_parameters,
            new_creation.telescopes,
            nsigma=3.0,
            nmin=6,
            window="all",
        )
        _sim_timer.stop("simulation_detection")


        if decision:
            print("A good event to fit")

            return _finish_sim_event(my_own_model, pyLIMA_parameters, True)

        else:
            print(
                "Not a good event to fit.\n"
                "Fail deviation-from-constant criterion."
            )

            return _finish_sim_event(my_own_model, pyLIMA_parameters, False)

    else:

        if use_roman and use_rubin:
            print(
                "Not a good event to fit since Roman "
                "and/or Rubin has no valid data"
            )

        elif use_rubin:
            print(
                "Not a good event to fit since Rubin "
                "has no valid data"
            )

        elif use_roman:
            print(
                "Not a good event to fit since Roman "
                "has no valid data"
            )

        return _finish_sim_event(my_own_model, pyLIMA_parameters, False)
# ================================================================
#  new_data y extract_data_event (sin cambios de lógica)
# ================================================================
def new_data(Event, nset, nevent, cols_fit, data):
    """
    Extrae parámetros del fit de forma robusta.

    Si falla piE_MC o alguna estimación de masa por una matriz de
    covarianza mal condicionada, se guardan NaNs en lugar de detener
    toda la corrida.
    """

    import numpy as np

    new_row = dict.fromkeys(cols_fit)

    new_row["Source"] = [nevent]
    new_row["Set"] = [nset]

    # ------------------------------------------------------------
    # Parámetros ajustados y errores
    # ------------------------------------------------------------

    try:
        fit_vals = Event.dict_fit_vals(data)

        for key in fit_vals:
            new_row[key] = [fit_vals[key]]

    except Exception as error:
        print(
            "[warning] Event.dict_fit_vals failed "
            f"for Source={nevent}, Set={nset}: {repr(error)}"
        )

    # ------------------------------------------------------------
    # piE por Monte Carlo
    # ------------------------------------------------------------

    try:
        piemc = Event.piE_MC(data)

        new_row["piE"] = [piemc.get("piE", np.nan)]
        new_row["piE_err"] = [piemc.get("err_piE", np.nan)]

    except Exception as error:
        print(
            "[warning] Event.piE_MC failed "
            f"for Source={nevent}, Set={nset}: {repr(error)}"
        )

        new_row["piE"] = [np.nan]
        new_row["piE_err"] = [np.nan]

    # ------------------------------------------------------------
    # Chi2 / dof
    # ------------------------------------------------------------

    try:
        chichidof = Event.chichi_dof(data)

        new_row["chichi"] = [chichidof.get("chi2", np.nan)]
        new_row["dof"] = [chichidof.get("dof", np.nan)]

    except Exception as error:
        print(
            "[warning] Event.chichi_dof failed "
            f"for Source={nevent}, Set={nset}: {repr(error)}"
        )

        new_row["chichi"] = [np.nan]
        new_row["dof"] = [np.nan]

    # ------------------------------------------------------------
    # Masas derivadas
    # ------------------------------------------------------------

    if not Event.model == "PSPL":

        try:
            fitmassv3 = Event.fit_mass_v3(data)

            new_row["err_mass_v3"] = [
                fitmassv3.get("err_mass", np.nan)
            ]

            new_row["mass_v3"] = [
                fitmassv3.get("mass", np.nan)
            ]

        except Exception as error:
            print(
                "[warning] Event.fit_mass_v3 failed "
                f"for Source={nevent}, Set={nset}: {repr(error)}"
            )

            new_row["err_mass_v3"] = [np.nan]
            new_row["mass_v3"] = [np.nan]

    try:
        fitmassv2 = Event.fit_mass_v2(data)

        new_row["err_mass_v2"] = [
            fitmassv2.get("err_mass", np.nan)
        ]

        new_row["mass_v2"] = [
            fitmassv2.get("mass", np.nan)
        ]

    except Exception as error:
        print(
            "[warning] Event.fit_mass_v2 failed "
            f"for Source={nevent}, Set={nset}: {repr(error)}"
        )

        new_row["err_mass_v2"] = [np.nan]
        new_row["mass_v2"] = [np.nan]

    try:
        fitmassv1 = Event.fit_mass_v1(data)

        new_row["err_mass_v1"] = [
            fitmassv1.get("err_mass", np.nan)
        ]

        new_row["mass_v1"] = [
            fitmassv1.get("mass", np.nan)
        ]

    except Exception as error:
        print(
            "[warning] Event.fit_mass_v1 failed "
            f"for Source={nevent}, Set={nset}: {repr(error)}"
        )

        new_row["err_mass_v1"] = [np.nan]
        new_row["mass_v1"] = [np.nan]

    # ------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------

    try:
        new_row["ln_likelihood"] = [data.get("ln_likelihood", np.nan)]
    except Exception:
        new_row["ln_likelihood"] = [np.nan]

    return new_row
    
    
def extract_data_event(Event, model, nevent, system_type, nset):
    """
    Extrae datos true, fit_rr y opcionalmente fit_roman.

    Esta versión:
    - no fuerza cargar fit_roman si Roman está apagado;
    - guarda chi2_true y n_data_true desde Event.info;
    - agrega delta_chi2_true = chi2_fit - chi2_true;
    - funciona para Rubin-only;
    - tolera fits sin paralaje;
    - guarda coordenadas de fuente y metadata MAF/Rubin para mapas espaciales.
    """

    import numpy as np
    from pathlib import Path

    # ============================================================
    # Cargar simulación
    # ============================================================

    Event.load_data_sim()

    # ============================================================
    # Cargar fit_rr si no viene en memoria
    # ============================================================

    if Event.fit_rr_data is None:
        if Event.path_fit_rr is not None and Path(Event.path_fit_rr).exists():
            Event.fit_rr_data = np.load(
                Event.path_fit_rr,
                allow_pickle=True,
            ).item()
        else:
            raise FileNotFoundError(
                f"No encuentro fit_rr_data ni archivo en {Event.path_fit_rr}"
            )

    # ============================================================
    # Cargar fit_roman solo si existe
    # ============================================================

    if Event.fit_roman_data is None:
        if Event.path_fit_roman is not None and Path(Event.path_fit_roman).exists():
            Event.fit_roman_data = np.load(
                Event.path_fit_roman,
                allow_pickle=True,
            ).item()
        else:
            Event.fit_roman_data = None

    labels_params = Event.labels_params()

    # ============================================================
    # Columnas espaciales / metadata MAF
    # ============================================================

    cols_spatial = [
        "ra",
        "dec",
        "maf_ra",
        "maf_dec",
        "maf_ra_used",
        "maf_dec_used",
        "maf_source_ra",
        "maf_source_dec",
        "rubin_pointing_mode",
        "rubin_cache_cell_deg",
        "maf_mode",
        "maf_cache_mode",
        "maf_cache_source",
        "maf_n_obs",
        "D_L",
        "D_S",
        "D_L_kpc",
        "D_S_kpc",
        "mu_rel",
        "lens_ra",
        "lens_dec",
        "galb",
        "gall",
        "lens_galb",
        "lens_gall",
    ]

    cols_photometry = photometry_summary_columns()

    # ============================================================
    # Columnas true
    # ============================================================

    if "dfit" not in system_type:

        if model == "USBL":

            cols_true = (
                ["Source", "Set"]
                + labels_params
                + [
                    "Category",
                    "Category_p",
                    "mass",
                    "sel_crit",
                    "piE",
                ]
            )

        elif model == "FSPL":

            cols_true = (
                ["Source", "Set"]
                + labels_params
                + [
                    "Category",
                    "mass",
                    "sel_crit",
                    "piE",
                    "crit_FFP_Rubin",
                ]
            )

        else:

            cols_true = (
                ["Source", "Set"]
                + labels_params
                + [
                    "Category",
                    "mass",
                    "sel_crit",
                    "piE",
                    "W149",
                    "u",
                    "g",
                    "r",
                    "i",
                    "z",
                    "y",
                ]
            )

        cols_true = (
            cols_true
            + [
                "chi2_true",
                "n_data_true",
            ]
            + cols_spatial
            + cols_photometry
        )

    # ============================================================
    # Construir true table
    # ============================================================

    if "dfit" not in system_type:

        true = Event.true_values()

        new_data_true = dict.fromkeys(cols_true)

        new_data_true["Source"] = [nevent]
        new_data_true["Set"] = [nset]

        for key in Event.labels_params():
            new_data_true[key] = [true[key]]

        new_data_true["piE"] = [
            np.sqrt(
                true["piEN"] ** 2
                + true["piEE"] ** 2
            )
        ]

        if model == "USBL":

            npts_speak = Event.count_points_secon_peak()

            for f in npts_speak:
                new_data_true["anomaly_" + f] = [
                    npts_speak[f]
                ]

        new_data_true["mass"] = [
            Event.mass_true()
        ]

        npts = Event.count_points()

        for f in npts:
            new_data_true[f] = [
                npts[f]
            ]

        npts_pk = Event.count_points_peak()

        for f in npts_pk:
            new_data_true[f + "_peak"] = [
                npts_pk[f]
            ]

        npts_pk_narrow = Event.count_points_narrow_peak()

        for f in npts_pk:
            new_data_true[f + "_npeak"] = [
                npts_pk_narrow[f]
            ]

        npts_pk_left = Event.count_points_left_peak()

        for f in npts_pk:
            new_data_true[f + "_lpeak"] = [
                npts_pk_left[f]
            ]

        npts_pk_right = Event.count_points_right_peak()

        for f in npts_pk:
            new_data_true[f + "_rpeak"] = [
                npts_pk_right[f]
            ]

        # ========================================================
        # Chi2 del modelo verdadero
        # ========================================================

        if isinstance(Event.info, dict):

            chi2_true = Event.info.get(
                "chi2_true",
                np.nan,
            )

            n_data_true = Event.info.get(
                "n_data_true",
                np.nan,
            )

        else:

            chi2_true = np.nan
            n_data_true = np.nan

        new_data_true["chi2_true"] = [
            chi2_true
        ]

        new_data_true["n_data_true"] = [
            n_data_true
        ]

        # ========================================================
        # Coordenadas y metadata espacial
        # ========================================================

        computed_params = getattr(Event, "computed_params", {})

        if computed_params is None:
            computed_params = {}

        for key in cols_spatial + cols_photometry:

            try:
                value = computed_params.get(key, np.nan)
            except Exception:
                value = np.nan

            new_data_true[key] = [
                value
            ]

    else:

        new_data_true = []

    # ============================================================
    # Columnas fit
    # ============================================================

    cols_fit = (
        ["Source", "Set"]
        + labels_params
        + [f + "_err" for f in labels_params]
        + ["piE", "piE_err"]
        + ["chichi", "dof"]
        + [
            "mass_v1",
            "mass_v2",
            "err_mass_v1",
            "err_mass_v2",
        ]
    )

    if model != "PSPL":

        cols_fit = cols_fit + [
            "mass_v3",
            "err_mass_v3",
        ]

    cols_fit = cols_fit + [
        "chi2_true",
        "n_data_true",
        "delta_chi2_true",
    ]

    # ============================================================
    # Extraer info true desde Event.info
    # ============================================================

    if isinstance(Event.info, dict):

        chi2_true = Event.info.get(
            "chi2_true",
            np.nan,
        )

        n_data_true = Event.info.get(
            "n_data_true",
            np.nan,
        )

    else:

        chi2_true = np.nan
        n_data_true = np.nan

    # ============================================================
    # Fit RR / telescopios activos
    # ============================================================

    new_data_rr = new_data(
        Event,
        nset,
        nevent,
        cols_fit,
        Event.fit_rr_data,
    )

    new_data_rr["chi2_true"] = [
        chi2_true
    ]

    new_data_rr["n_data_true"] = [
        n_data_true
    ]

    try:
        new_data_rr["delta_chi2_true"] = [
            new_data_rr["chichi"][0] - chi2_true
        ]
    except Exception:
        new_data_rr["delta_chi2_true"] = [
            np.nan
        ]

    # ============================================================
    # Fit Roman-only, si existe
    # ============================================================
    if Event.fit_roman_data is not None:
        try:

            new_data_roman = new_data(
                Event,
                nset,
                nevent,
                cols_fit,
                Event.fit_roman_data,
            )

            new_data_roman["chi2_true"] = [
                chi2_true
            ]

            new_data_roman["n_data_true"] = [
                n_data_true
            ]

            try:
                new_data_roman["delta_chi2_true"] = [
                    new_data_roman["chichi"][0] - chi2_true
                ]
            except Exception:
                new_data_roman["delta_chi2_true"] = [
                    np.nan
                ]

        except Exception as error:

            print(
                "[warning] Could not extract Roman-only fit data "
                f"for Source={nevent}, Set={nset}: {repr(error)}"
            )

            new_data_roman = None

    else:

        new_data_roman = None

    return new_data_true, new_data_rr, new_data_roman

def register_clean_exit_handlers():
    """
    Permite que el proceso salga limpiamente ante SIGTERM/SIGINT.
    """
    import signal
    import sys

    try:
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
        signal.signal(signal.SIGINT,  lambda s, f: sys.exit(0))
    except ValueError:
        # Puede pasar si se llama desde un thread que no es el principal.
        pass
    
def choose_catalog_rows(seed, n_rows=10000):
    """
    Reproduce la selección aleatoria original de filas TRILEGAL y GENULENS.

    Antes era:

        np.random.seed(i)
        ROW_G = np.random.randint(0, 10000)
        ROW_T = np.random.randint(0, 10000)

    Usamos RandomState para no contaminar el estado global de numpy.
    """
    rng = np.random.RandomState(seed)

    ROW_G = rng.randint(0, n_rows)
    ROW_T = rng.randint(0, n_rows)

    return ROW_G, ROW_T

def read_catalog_row(path_catalog, row_index):
    """
    Lee una única fila de un catálogo CSV, preservando el header.

    row_index es 0-based respecto de los datos, no contando el header.
    """
    row = pd.read_csv(
        path_catalog,
        skiprows=lambda x: x not in (0, row_index + 1),
    )

    return row

def load_trilegal_genulens_rows(
    path_TRILEGAL_set,
    path_GENULENS_set,
    ROW_T,
    ROW_G,
    verbose=True,
):
    """
    Carga una fila de TRILEGAL y una fila de GENULENS.
    """
    TRILEGAL_row = read_catalog_row(
        path_TRILEGAL_set,
        ROW_T,
    )

    GENULENS_row = read_catalog_row(
        path_GENULENS_set,
        ROW_G,
    )

    if verbose:
        print(TRILEGAL_row)

    return TRILEGAL_row, GENULENS_row


# def build_event_params_from_rows(
#     i,
#     TRILEGAL_row,
#     GENULENS_row,
#     system_type,
# ):
#     """
#     Construye el diccionario event_params a partir de TRILEGAL + GENULENS.
#     """
#     magstar = TRILEGAL_row[
#         [
#             "W149",
#             "u",
#             "g",
#             "r",
#             "i",
#             "z",
#             "Y",
#         ]
#     ].iloc[0]

#     event_params = {
#         **magstar.to_dict(),
#         **event_param(
#             i,
#             TRILEGAL_row.iloc[0],
#             GENULENS_row.iloc[0],
#             system_type,
#         ),
#     }

#     return event_params
def build_event_params_from_rows(
    i,
    TRILEGAL_row,
    GENULENS_row,
    system_type,
    param_samplers=None,
    t0_range=[2460413.013828608, 2460413.013828608 + 365.25 * 8],
    custom_system=None,
):
    """
    Construye event_params a partir de TRILEGAL + GENULENS,
    permitiendo modificar las distribuciones de parámetros.
    """
    # Roman2024 TRILEGAL uses F146mag in Vega.
    # Keep W149 as the legacy internal Roman channel name.
    TRILEGAL_row = TRILEGAL_row.copy()

    roman_catalog_mag_column = None

    for _candidate in (
        "F146mag",
        "F146",
        "W149",
    ):
        if _candidate in TRILEGAL_row.columns:
            roman_catalog_mag_column = _candidate
            break

    if roman_catalog_mag_column is None:
        raise KeyError(
            "No Roman magnitude column found in TRILEGAL row. "
            "Expected one of: F146mag, F146, W149."
        )

    if "W149" not in TRILEGAL_row.columns:
        TRILEGAL_row["W149"] = (
            TRILEGAL_row[
                roman_catalog_mag_column
            ]
        )

    magstar = TRILEGAL_row[
        [
            "W149",
            "u",
            "g",
            "r",
            "i",
            "z",
            "Y",
        ]
    ].iloc[0]

    event_params = {
        **magstar.to_dict(),
        **event_param(
            i,
            TRILEGAL_row.iloc[0],
            GENULENS_row.iloc[0],
            system_type,
            t0_range=t0_range,
            custom_system=custom_system,
            param_samplers=param_samplers,
        ),
    }

    event_params[
        "roman_catalog_mag_column"
    ] = roman_catalog_mag_column

    if roman_catalog_mag_column in {
        "F146mag",
        "F146",
    }:
        event_params["F146mag"] = float(
            TRILEGAL_row[
                roman_catalog_mag_column
            ].iloc[0]
        )

    return event_params
def simulate_event_for_fit(
    i,
    event_params,
    path_ephemerides,
    model,
    time_window=None,
    use_roman=True,
    use_rubin=True,
    truth_parallax=True,
    rubin_pointing_mode="fixed",
    rubin_cache_cell_deg=None,
    apply_detection_criteria=True,
    apply_photometric_filter=True,
    rubin_saturation_mag=None,
    roman_saturation_mag=None,
):
    my_own_model, pyLIMA_parameters, decision = sim_event(
        i,
        event_params,
        path_ephemerides,
        model,
        time_window=time_window,
        use_roman=use_roman,
        use_rubin=use_rubin,
        truth_parallax=truth_parallax,
        rubin_pointing_mode=rubin_pointing_mode,
        rubin_cache_cell_deg=rubin_cache_cell_deg,
        apply_detection_criteria=apply_detection_criteria,
        apply_photometric_filter=apply_photometric_filter,
        rubin_saturation_mag=rubin_saturation_mag,
        roman_saturation_mag=roman_saturation_mag,
    )

    return my_own_model, pyLIMA_parameters, decision
def build_fit_time_mask(times, t0, tE, fit_time_window=None):
    """
    Máscara temporal usada solamente para el ajuste.

    No modifica la simulación.
    No modifica el criterio de detección.
    """

    times = np.asarray(times, dtype=float)

    if fit_time_window is None:
        return np.ones_like(times, dtype=bool)

    if not isinstance(fit_time_window, dict):
        raise TypeError("fit_time_window debe ser None o un diccionario.")

    mode = fit_time_window.get("mode", "all")

    if mode == "all":
        return np.ones_like(times, dtype=bool)

    if mode == "t0_pm_tE":
        return (times >= t0 - tE) & (times <= t0 + tE)

    if mode == "t0_pm_2tE":
        return (times >= t0 - 2.0 * tE) & (times <= t0 + 2.0 * tE)

    if mode == "t0_pm_factor_tE":
        factor = float(fit_time_window.get("factor", 3.5))
        return (times >= t0 - factor * tE) & (times <= t0 + factor * tE)

    if mode == "custom_jd":
        t_min = float(fit_time_window["t_min"])
        t_max = float(fit_time_window["t_max"])
        return (times >= t_min) & (times <= t_max)

    raise ValueError(f"fit_time_window mode no reconocido: {mode}")
def apply_fit_time_window_to_lc_dict(
    lc_to_fit,
    pyLIMA_parameters,
    fit_time_window=None,
):
    """
    Aplica la ventana temporal solo al diccionario lc_to_fit.

    lc_to_fit contiene arrays con columnas:
        time, mag, err_mag

    No toca lc_to_save ni el modelo simulado completo.
    """

    if fit_time_window is None:
        return lc_to_fit

    mode = fit_time_window.get("mode", "all")

    if mode == "all":
        return lc_to_fit

    t0 = float(pyLIMA_parameters["t0"])
    tE = float(pyLIMA_parameters["tE"])

    filtered = {}

    for band, arr in lc_to_fit.items():

        if arr is None or len(arr) == 0:
            filtered[band] = []
            continue

        arr = np.asarray(arr)

        times = arr[:, 0]

        mask = build_fit_time_mask(
            times,
            t0=t0,
            tE=tE,
            fit_time_window=fit_time_window,
        )

        filtered[band] = arr[mask]

    return filtered
def count_active_fit_points(
    lc_to_fit,
    use_roman=True,
    use_rubin=True,
):
    """
    Cuenta cuántos puntos quedan disponibles para el ajuste
    después de aplicar fit_time_window.

    Solo cuenta los telescopios activos según use_roman/use_rubin.
    """

    bands = []

    if use_roman:
        bands.append("W149")

    if use_rubin:
        bands.extend(["u", "g", "r", "i", "z", "y"])

    n_total = 0

    for band in bands:

        arr = lc_to_fit.get(band, [])

        if arr is None:
            continue

        try:
            n_total += len(arr)
        except TypeError:
            continue

    return int(n_total)


def save_simulated_event_if_selected(
    i,
    ROW_G,
    ROW_T,
    path_TRILEGAL_set,
    path_GENULENS_set,
    path_to_save_model,
    my_own_model,
    pyLIMA_parameters,
    event_params,
    GENULENS_row,
    TRILEGAL_row,
):
    """
    Guarda la curva simulada usando tu función save_sim().
    """
    save_sim(
        i,
        ROW_G,
        ROW_T,
        path_TRILEGAL_set,
        path_GENULENS_set,
        path_to_save_model,
        my_own_model,
        pyLIMA_parameters,
        event_params,
        GENULENS_row,
        TRILEGAL_row,
    )
    
    
def extract_lightcurves_for_fit(pyLIMA_model):
    """
    Extrae las curvas de luz del evento en dos formatos:

    lc_to_fit:
        diccionario con arrays numpy para pasar a fit_rubin_roman.

    lc_to_save:
        diccionario con tablas para guardar dentro de Analysis_Event.
    """
    lc_to_fit = {}
    lc_to_save = {}

    for telo in pyLIMA_model.event.telescopes:

        if len(telo.lightcurve["mag"]) != 0:

            tbl = telo.lightcurve[
                [
                    "time",
                    "mag",
                    "err_mag",
                ]
            ].copy()

            if hasattr(tbl["time"], "value"):
                tbl["time"] = tbl["time"].value

            df = tbl.to_pandas()

            lc_to_fit[telo.name] = df.values
            lc_to_save[telo.name] = tbl

        else:
            lc_to_fit[telo.name] = []

    # Garantiza que existan todas las claves que fit_rubin_roman espera.
    for band in ["W149", "u", "g", "r", "i", "z", "y"]:
        if band not in lc_to_fit:
            lc_to_fit[band] = []

    return lc_to_fit, lc_to_save

def get_model_origin(pyLIMA_model):
    """
    Devuelve el origen del modelo si existe.

    Para USBL suele existir my_own_model.origin.
    Para PSPL/FSPL puede no ser relevante.
    """
    return getattr(pyLIMA_model, "origin", None)


def get_event_coordinates(pyLIMA_model):
    """Devuelve (RA, Dec) del Event asociado a un modelo pyLIMA."""
    if pyLIMA_model is None or getattr(pyLIMA_model, "event", None) is None:
        raise ValueError("El modelo pyLIMA no contiene un Event válido.")

    return (
        float(pyLIMA_model.event.ra),
        float(pyLIMA_model.event.dec),
    )


def validate_event_coordinates(
    pyLIMA_model,
    expected_ra,
    expected_dec,
    label="fit",
    atol=1e-10,
):
    """
    Verifica que simulación y ajuste usen exactamente la misma dirección
    del cielo. Esto es crítico cuando se usa paralaje.
    """
    got_ra, got_dec = get_event_coordinates(pyLIMA_model)

    if not (
        np.isclose(got_ra, float(expected_ra), rtol=0.0, atol=atol)
        and np.isclose(got_dec, float(expected_dec), rtol=0.0, atol=atol)
    ):
        raise RuntimeError(
            f"{label} usa coordenadas distintas de la simulación.\n"
            f"simulation = ({float(expected_ra)}, {float(expected_dec)})\n"
            f"{label} = ({got_ra}, {got_dec})"
        )

    return True


def resolve_fit_options(
    model,
    fit_model=None,
    fit_parallax=None,
    default_fit_parallax=True,
):
    """
    Normaliza las opciones de ajuste.

    Regla nueva:
    - model y fit_model son solo nombres de modelo: PSPL, FSPL o USBL.
    - fit_parallax es el único control para ajustar con/sin paralaje.
    - No se interpreta ningún sufijo de paralaje en el nombre del modelo.
    """

    model_use = normalize_model_name(model)

    if fit_model is None:
        fit_model = model_use

    fit_model_use = normalize_model_name(fit_model)

    if fit_parallax is None:
        fit_parallax_use = bool(default_fit_parallax)
    else:
        fit_parallax_use = bool(fit_parallax)

    return model_use, fit_model_use, fit_parallax_use


def parallax_suffix_for_filename(use_parallax):
    """
    Sufijo explícito para nombres de archivos de fit.

    Wrapper de compatibilidad sobre `fit_lc.parallax_suffix` (misma lógica).
    """
    return parallax_suffix(use_parallax)


def expected_fit_results_path(
    path_to_save_fit,
    event_name,
    algo,
    fit_model,
    fit_parallax,
):
    """
    Reproduce el nombre que usa fit_lc.save_fit_results().
    """

    fit_model_use = normalize_model_name(fit_model)
    suffix = f"{fit_model_use}_{parallax_suffix_for_filename(fit_parallax)}"

    return Path(path_to_save_fit) / f"{event_name}_{algo}_{suffix}.npy"


def fit_results_path_from_fit_object(
    fit_object,
    path_to_save_fit,
    algo,
    fit_model,
    fit_parallax,
    fallback_event_name,
):
    """
    Devuelve el path esperado del .npy de un fit ya corrido.

    Usa fit_object.fit_results["name"] si existe, porque ese nombre
    distingue automáticamente Event_RR, Event_Rubin o Event_Roman.
    """

    event_name = fallback_event_name

    try:
        event_name = fit_object.fit_results.get(
            "name",
            fallback_event_name,
        )
    except Exception:
        pass

    return expected_fit_results_path(
        path_to_save_fit,
        event_name,
        algo,
        fit_model,
        fit_parallax,
    )


def extract_nset_string(path_GENULENS_set, default="manual"):
    """
    Extrae el identificador numérico del path GENULENS.
    Si no encuentra números, devuelve default.
    """
    matches = re.findall(r"\d+", str(path_GENULENS_set))

    if len(matches) == 0:
        return default

    return matches[0]

def build_analysis_event(
    i,
    model,
    algo,
    path_to_save_model,
    path_to_save_fit,
    GENULENS_row,
    TRILEGAL_row,
    event_params,
    fit_rr,
    fit_roman,
    origin,
    lc_to_save,
    pyLIMA_parameters,
    use_roman=True,
    use_rubin=True,
    truth_parallax=True,
    fit_model=None,
    fit_parallax=None,
    fit_time_window=None,
    chi2_true=None,
):
    """
    Construye el objeto Analysis_Event.

    Mantiene model y fit_model separados.
    La opción de paralaje se guarda solamente en fit_parallax,
    no en el nombre del modelo.
    """

    model, fit_model, fit_parallax = resolve_fit_options(
        model,
        fit_model=fit_model,
        fit_parallax=fit_parallax,
        default_fit_parallax=True,
    )

    if chi2_true is None:
        chi2_true = {
            "chi2_true": np.nan,
            "n_data_true": np.nan,
            "chi2_true_by_telescope": {},
        }

    path_model = Path(path_to_save_model) / f"Event_{i}.h5"

    path_fit_rr = fit_results_path_from_fit_object(
        fit_rr,
        path_to_save_fit,
        algo,
        fit_model,
        fit_parallax,
        fallback_event_name=f"Event_RR_{i}",
    )

    if fit_roman is not None:
        path_fit_roman = fit_results_path_from_fit_object(
            fit_roman,
            path_to_save_fit,
            algo,
            fit_model,
            fit_parallax,
            fallback_event_name=f"Event_Roman_{i}",
        )
    else:
        path_fit_roman = None

    Event = Analysis_Event(
        model,
        path_model=str(path_model),
        path_fit_rr=str(path_fit_rr),
        path_fit_roman=(
            str(path_fit_roman)
            if path_fit_roman is not None
            else None
        ),
        genulens_params=GENULENS_row.iloc[0],
        trilegal_params=TRILEGAL_row.iloc[0],
        computed_params=event_params,
        fit_rr_data=fit_rr.fit_results,
        fit_roman_data=fit_roman.fit_results if fit_roman is not None else None,
        origin=origin,
        lightcurves=lc_to_save,
        model_params=pyLIMA_parameters,
        info={
            "use_roman": use_roman,
            "use_rubin": use_rubin,
            "sim_model": model,
            "fit_model": fit_model,
            "fit_parallax": fit_parallax,
            "chi2_true": chi2_true["chi2_true"],
            "n_data_true": chi2_true["n_data_true"],
            "chi2_true_by_telescope": chi2_true["chi2_true_by_telescope"],
            "truth_parallax": bool(truth_parallax),
            "fit_time_window": fit_time_window,
            "apply_photometric_filter": event_params.get("apply_photometric_filter", np.nan),
            "phot_n_total": event_params.get("phot_n_total", np.nan),
            "phot_n_keep": event_params.get("phot_n_keep", np.nan),
            "phot_n_too_faint_5sigma": event_params.get("phot_n_too_faint_5sigma", np.nan),
            "phot_n_saturated": event_params.get("phot_n_saturated", np.nan),
            "phot_n_rejected": event_params.get("phot_n_rejected", np.nan),
            "path_fit_rr_actual": str(path_fit_rr),
            "path_fit_roman_actual": (
                str(path_fit_roman)
                if path_fit_roman is not None
                else None
            ),
        },
        indices=None,
    )

    return Event
def save_extracted_results(
    Event,
    model,
    i,
    system_type,
    nset_str,
    path_to_save_results,
    save_roman=True,
):
    """
    Extrae y guarda true, fit_rr y, opcionalmente, fit_roman en parquet.

    Si save_roman=False, no intenta guardar fit_roman.
    """

    new_data_true, new_data_rr, new_data_roman = extract_data_event(
        Event,
        model,
        i,
        system_type,
        nset_str,
    )

    print(new_data_true, new_data_rr, new_data_roman)

    base_dir = Path(path_to_save_results)

    path_true_dir = base_dir / "true"
    path_fitrr_dir = base_dir / "fit_rr"
    path_fitro_dir = base_dir / "fit_roman"

    path_true_file = path_true_dir / f"true_rr_{nset_str}_{i}.parquet"
    path_fitrr_file = path_fitrr_dir / f"fit_rr_{nset_str}_{i}.parquet"
    path_fitro_file = path_fitro_dir / f"fit_roman_{nset_str}_{i}.parquet"

    _save_dict_as_parquet(
        new_data_true,
        path_true_file,
        append=False,
    )

    _save_dict_as_parquet(
        new_data_rr,
        path_fitrr_file,
        append=False,
    )

    if save_roman and new_data_roman is not None:
        _save_dict_as_parquet(
            new_data_roman,
            path_fitro_file,
            append=False,
        )
        roman_msg = f"  {path_fitro_file}"
    else:
        roman_msg = "  fit_roman no guardado"

    print(
        "→ Guardado Parquet:\n"
        f"  {path_true_file}\n"
        f"  {path_fitrr_file}\n"
        f"{roman_msg}"
    )

    return new_data_true, new_data_rr, new_data_roman


def build_event_params_from_pair_row(
    i,
    pair_row,
    system_type,
    param_samplers=None,
    t0_range=[2460413.013828608, 2460413.013828608 + 365.25 * 8],
    custom_system=None,
):
    """
    Construye event_params usando una fila del catálogo pareado
    fuente-lente de AstroDataLab.

    Importante:
    - ra, dec son las coordenadas de la fuente.
    - lens_ra, lens_dec son las coordenadas de la lente.
    - maf_ra, maf_dec son las coordenadas que debe usar MAF/Rubin.
    """

    magstar = pair_row[
        [
            "W149",
            "u",
            "g",
            "r",
            "i",
            "z",
            "Y",
        ]
    ]

    event_params = {
        **magstar.to_dict(),
        **event_param_from_pair_row(
            i,
            pair_row,
            system_type,
            t0_range=t0_range,
            custom_system=custom_system,
            param_samplers=param_samplers,
        ),
    }

    # ------------------------------------------------------------
    # Guardar información de coordenadas y cinemática del pair row
    # ------------------------------------------------------------

    keys_to_keep = [
        "ra",
        "dec",
        "lens_ra",
        "lens_dec",
        "D_S",
        "D_L",
        "D_S_kpc",
        "D_L_kpc",
        "mu_rel",
        "theta_rad",
        "mu_source",
        "mu_lens",
        "gc",
        "galb",
        "gall",
        "lens_gc",
        "lens_galb",
        "lens_gall",
    ]

    for key in keys_to_keep:
        if key in pair_row.index:
            try:
                event_params[key] = float(pair_row[key])
            except Exception:
                event_params[key] = pair_row[key]

    if "ra" in pair_row.index and "dec" in pair_row.index:
        event_params["maf_ra"] = float(pair_row["ra"])
        event_params["maf_dec"] = float(pair_row["dec"])

    return event_params

def load_pair_catalog_row(
    pair_catalog=None,
    path_pair_catalog=None,
    row_index=None,
):
    """
    Lee una fila del catálogo pareado.

    Se puede pasar:
    - pair_catalog como DataFrame ya cargado;
    - path_pair_catalog como parquet/csv.

    row_index es 0-based.
    """

    if pair_catalog is None and path_pair_catalog is None:
        raise ValueError(
            "Tenés que pasar pair_catalog o path_pair_catalog."
        )

    if pair_catalog is None:
        path_pair_catalog = Path(path_pair_catalog)

        if path_pair_catalog.suffix == ".parquet":
            pair_catalog = pd.read_parquet(path_pair_catalog)

        elif path_pair_catalog.suffix == ".csv":
            pair_catalog = pd.read_csv(path_pair_catalog)

        else:
            raise ValueError(
                "path_pair_catalog debe ser .parquet o .csv"
            )

    if row_index is None:
        raise ValueError("Tenés que pasar row_index.")

    if row_index >= len(pair_catalog):
        raise IndexError(
            f"row_index={row_index} fuera de rango para catálogo "
            f"con {len(pair_catalog)} filas."
        )

    pair_row = pair_catalog.iloc[row_index]

    return pair_row

def all_telescope_photometric_chi2_robust(
    model,
    pyLIMA_parameters,
    rescaling_parameters=None,
    verbose=False,
):
    """
    Versión robusta de all_telescope_photometric_chi2 de pyLIMA.

    Hace lo mismo:
        chi2 = sum(((flux - model_flux) / err_flux)**2)

    pero saltea telescopios/bandas vacías o inconsistentes.
    """

    import numpy as np

    residuals_norm = []
    chi2_by_tel = {}

    ind = 0

    for telescope in model.event.telescopes:

        tel_name = telescope.name

        if telescope.lightcurve is None:
            if verbose:
                print(f"[chi2_true] {tel_name}: lightcurve is None, skip")
            continue

        if len(telescope.lightcurve) == 0:
            if verbose:
                print(f"[chi2_true] {tel_name}: empty lightcurve, skip")
            continue

        lightcurve = telescope.lightcurve

        if "flux" not in lightcurve.colnames:
            if verbose:
                print(f"[chi2_true] {tel_name}: no flux column, skip")
            continue

        if "err_flux" not in lightcurve.colnames:
            if verbose:
                print(f"[chi2_true] {tel_name}: no err_flux column, skip")
            continue

        flux = np.asarray(
            lightcurve["flux"].value,
            dtype=float,
        )

        err_flux = np.asarray(
            lightcurve["err_flux"].value,
            dtype=float,
        )

        if len(flux) == 0:
            if verbose:
                print(f"[chi2_true] {tel_name}: flux has zero length, skip")
            continue

        try:
            microlensing_model = model.compute_the_microlensing_model(
                telescope,
                pyLIMA_parameters,
            )

            photometric_model = np.asarray(
                microlensing_model["photometry"],
                dtype=float,
            )

        except Exception as e:
            if verbose:
                print(f"[chi2_true] {tel_name}: model failed: {e}")
            continue

        if len(photometric_model) == 0:
            if verbose:
                print(f"[chi2_true] {tel_name}: model has zero length, skip")
            continue

        if len(flux) != len(photometric_model):
            if verbose:
                print(
                    f"[chi2_true] {tel_name}: length mismatch, skip. "
                    f"len(flux)={len(flux)}, "
                    f"len(model)={len(photometric_model)}"
                )
            continue

        if len(err_flux) != len(flux):
            if verbose:
                print(
                    f"[chi2_true] {tel_name}: err length mismatch, skip. "
                    f"len(err_flux)={len(err_flux)}, "
                    f"len(flux)={len(flux)}"
                )
            continue

        if rescaling_parameters is not None:
            err_flux = err_flux * rescaling_parameters[ind]

        mask = (
            np.isfinite(flux)
            & np.isfinite(err_flux)
            & np.isfinite(photometric_model)
            & (err_flux > 0)
        )

        if np.sum(mask) == 0:
            if verbose:
                print(f"[chi2_true] {tel_name}: no valid points, skip")
            continue

        residus_norm = (
            flux[mask] - photometric_model[mask]
        ) / err_flux[mask]

        residuals_norm.append(residus_norm)

        chi2_tel = float(
            np.sum(residus_norm**2)
        )

        chi2_by_tel[tel_name] = {
            "chi2": chi2_tel,
            "n_data": int(np.sum(mask)),
        }

        if verbose:
            print(
                f"[chi2_true] {tel_name}: "
                f"n={np.sum(mask)}, chi2={chi2_tel:.3f}"
            )

        ind += 1

    if len(residuals_norm) == 0:
        return {
            "chi2_true": np.nan,
            "n_data_true": 0,
            "chi2_true_by_telescope": chi2_by_tel,
        }

    all_residuals = np.concatenate(residuals_norm)

    return {
        "chi2_true": float(np.sum(all_residuals**2)),
        "n_data_true": int(len(all_residuals)),
        "chi2_true_by_telescope": chi2_by_tel,
    }

def replace_flux_by_noisy_magnitude_flux_ZP(pyLIMA_model, verbose=False):
    """
    Reemplaza flux y err_flux usando las magnitudes ruidosas simuladas.

    Asume que los telescopios se llaman exactamente:
        "W149", "u", "g", "r", "i", "z", "y"

    Convención:
        mag = ZP - 2.5 log10(flux)

    Entonces:
        flux = 10**(0.4 * (ZP - mag))
        err_flux = flux * ln(10)/2.5 * err_mag
    """

    import numpy as np

    ZP = SIMULATION_BAND_ZERO_POINTS

    for telescope in pyLIMA_model.event.telescopes:

        band = telescope.name

        if band not in ZP:
            if verbose:
                print(f"[replace_flux_ZP] {band}: banda no reconocida, skip")
            continue

        if telescope.lightcurve is None:
            if verbose:
                print(f"[replace_flux_ZP] {band}: lightcurve is None, skip")
            continue

        if len(telescope.lightcurve) == 0:
            if verbose:
                print(f"[replace_flux_ZP] {band}: lightcurve vacía, skip")
            continue

        lc = telescope.lightcurve

        mag = np.asarray(
            lc["mag"].value,
            dtype=float,
        )

        err_mag = np.asarray(
            lc["err_mag"].value,
            dtype=float,
        )

        flux = 10.0 ** (0.4 * (ZP[band] - mag))

        err_flux = (
            flux
            * np.log(10.0)
            / 2.5
            * err_mag
        )

        lc["flux"] = flux
        lc["err_flux"] = err_flux

        if "inv_err_flux" in lc.colnames:
            lc["inv_err_flux"] = 1.0 / err_flux

        if verbose:
            print(
                f"[replace_flux_ZP] {band}: "
                f"N={len(flux)}, "
                f"median mag={np.nanmedian(mag):.4f}, "
                f"median err_mag={np.nanmedian(err_mag):.4g}, "
                f"median flux={np.nanmedian(flux):.4g}, "
                f"median err_flux={np.nanmedian(err_flux):.4g}"
            )
            
from collections.abc import Mapping


def build_event_params_from_custom_system(
    custom_system,
    model,
    use_roman=True,
    use_rubin=True,
    rubin_pointing_mode="fixed",
):
    """
    Construye directamente ``event_params`` sin seleccionar ni emparejar
    estrellas de catálogos.

    Parameters
    ----------
    custom_system : Mapping, pandas.Series or one-row pandas.DataFrame
        Parámetros verdaderos del evento. Para un FSPL Rubin-only típico:

        {
            "t0": ...,
            "u0": ...,
            "tE": ...,
            "rho": ...,
            "piEN": ...,
            "piEE": ...,
            "u": ...,
            "g": ...,
            "r": ...,
            "i": ...,
            "z": ...,
            "Y": ...,
            "ra": ...,
            "dec": ...,
        }

        Se conservan todas las claves adicionales como metadatos.

    model : str
        Modelo verdadero usado por ``simulate_event_for_fit``.

    use_roman, use_rubin : bool
        Determinan qué magnitudes son obligatorias.

    rubin_pointing_mode : str
        Si vale ``"source"``, se requieren coordenadas de la fuente.

    Returns
    -------
    dict
        Copia validada y normalizada de ``custom_system``.

    Notes
    -----
    ``tE`` se usa directamente. No se reconstruye a partir de
    ``thetaE`` ni ``mu_rel``.
    """

    if custom_system is None:
        raise ValueError(
            "catalog_mode='custom_system' requiere custom_system."
        )

    if isinstance(custom_system, pd.DataFrame):
        if len(custom_system) != 1:
            raise ValueError(
                "custom_system como DataFrame debe tener exactamente una fila."
            )
        event_params = custom_system.iloc[0].to_dict()

    elif isinstance(custom_system, pd.Series):
        event_params = custom_system.to_dict()

    elif isinstance(custom_system, Mapping):
        event_params = dict(custom_system)

    else:
        try:
            event_params = dict(custom_system)
        except Exception as error:
            raise TypeError(
                "custom_system debe ser un mapping, pandas.Series "
                "o DataFrame de una fila."
            ) from error

    # ------------------------------------------------------------
    # Normalización de nombres
    # ------------------------------------------------------------

    if "Y" not in event_params and "y" in event_params:
        event_params["Y"] = event_params["y"]

    if "y" not in event_params and "Y" in event_params:
        event_params["y"] = event_params["Y"]

    if "W149" not in event_params and "Y" in event_params:
        event_params["W149"] = event_params["Y"]

    if "maf_ra" not in event_params and "ra" in event_params:
        event_params["maf_ra"] = event_params["ra"]

    if "maf_dec" not in event_params and "dec" in event_params:
        event_params["maf_dec"] = event_params["dec"]

    # El modelo actual espera las componentes aun cuando sean nulas.
    event_params.setdefault("piEN", 0.0)
    event_params.setdefault("piEE", 0.0)

    # ------------------------------------------------------------
    # Parámetros obligatorios
    # ------------------------------------------------------------

    required = [
        "t0",
        "u0",
        "tE",
        "piEN",
        "piEE",
    ]

    model_upper = str(model).upper()

    if "FSPL" in model_upper:
        required.append("rho")

    if use_roman:
        required.append("W149")

    if use_rubin:
        required.extend(["u", "g", "r", "i", "z", "Y"])

    if (
        use_rubin
        and str(rubin_pointing_mode).lower() == "source"
    ):
        required.extend(["maf_ra", "maf_dec"])

    missing = [
        key
        for key in required
        if key not in event_params
    ]

    if missing:
        raise KeyError(
            "Faltan parámetros obligatorios en custom_system: "
            f"{missing}"
        )

    # ------------------------------------------------------------
    # Conversión y validación numérica
    # ------------------------------------------------------------

    numerical_keys = set(required)

    optional_numerical_keys = {
        "ra",
        "dec",
        "maf_ra",
        "maf_dec",
        "mass",
        "lens_mass",
        "thetaE",
        "thetaE_mas",
        "mu_rel",
        "D_L",
        "D_S",
        "D_L_kpc",
        "D_S_kpc",
        "gall",
        "galb",
        "alpha",
        "xi",
        "s",
        "q",
    }

    numerical_keys.update(
        key
        for key in optional_numerical_keys
        if (
            key in event_params
            and event_params[key] is not None
        )
    )

    for key in numerical_keys:
        try:
            value = float(event_params[key])
        except Exception as error:
            raise TypeError(
                f"custom_system[{key!r}] debe ser numérico."
            ) from error

        if not np.isfinite(value):
            raise ValueError(
                f"custom_system[{key!r}] no es finito: {value}"
            )

        event_params[key] = value

    if event_params["tE"] <= 0.0:
        raise ValueError("custom_system['tE'] debe ser positivo.")

    if "FSPL" in model_upper and event_params["rho"] <= 0.0:
        raise ValueError("custom_system['rho'] debe ser positivo.")

    # Metadatos que dejan explícito el origen directo.
    event_params["catalog_mode"] = "custom_system"
    event_params["custom_system_direct"] = True

    return event_params


def sim_fit(
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
    catalog_mode="trilegal_genulens",
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
    rubin_pointing_mode="fixed",
    rubin_cache_cell_deg=None,
    apply_detection_criteria=True,
    apply_photometric_filter=True,
    rubin_saturation_mag=None,
    roman_saturation_mag=None,
):
    """
    Simula un evento, aplica criterios de detección, guarda la simulación,
    ajusta las combinaciones solicitadas y guarda los resultados.

    Parameters
    ----------
    catalog_mode : {
        "trilegal_genulens",
        "astrodatalab_pairs",
        "custom_system",
    }
        Fuente de los parámetros verdaderos.

        ``custom_system`` usa directamente el diccionario entregado en
        ``custom_system`` y no selecciona ni empareja filas de catálogos.

    custom_system : dict-like or None
        En ``catalog_mode="custom_system"`` contiene el sistema verdadero
        completo. En particular, ``tE`` se utiliza directamente.

    time_window : None, tuple or callable
        Si es None, usa la curva completa.
        Si es ``(t_min, t_max)``, se pasa a ``sim_event``.
        Si es callable, se evalúa como ``time_window(event_params)``.
        Los tiempos deben estar en JD.
    """

    tstart = time.time()

    register_clean_exit_handlers()

    seed = i
    Source = seed

    np.random.seed(seed)

    model, fit_model, fit_parallax = resolve_fit_options(
        model,
        fit_model=fit_model,
        fit_parallax=fit_parallax,
        default_fit_parallax=truth_parallax,
    )

    print(
        f"[sim_fit] sim_model={model}, "
        f"fit_model={fit_model}, "
        f"truth_parallax={truth_parallax}, "
        f"fit_parallax={fit_parallax}"
    )

    # ============================================================
    # 1. Construcción de los parámetros verdaderos
    # ============================================================

    if catalog_mode == "trilegal_genulens":

        ROW_G = np.random.randint(0, 10000)
        ROW_T = np.random.randint(0, 10000)

        TRILEGAL_row = pd.read_csv(
            path_TRILEGAL_set,
            skiprows=lambda x: x not in (0, ROW_T + 1),
        )

        GENULENS_row = pd.read_csv(
            path_GENULENS_set,
            skiprows=lambda x: x not in (0, ROW_G + 1),
        )

        print(TRILEGAL_row)

        event_params = build_event_params_from_rows(
            i,
            TRILEGAL_row,
            GENULENS_row,
            system_type,
            param_samplers=param_samplers,
            t0_range=t0_range,
            custom_system=custom_system,
        )

    elif catalog_mode == "custom_system":

        if pair_catalog is not None or path_pair_catalog is not None:
            raise ValueError(
                "catalog_mode='custom_system' no usa pair_catalog "
                "ni path_pair_catalog."
            )

        if param_samplers not in (None, {}):
            raise ValueError(
                "catalog_mode='custom_system' recibe valores verdaderos "
                "directamente. No pases param_samplers en este modo."
            )

        event_params = build_event_params_from_custom_system(
            custom_system=custom_system,
            model=model,
            use_roman=use_roman,
            use_rubin=use_rubin,
            rubin_pointing_mode=rubin_pointing_mode,
        )

        # No existe una selección aleatoria de filas.
        ROW_T = int(i)
        ROW_G = int(i)

        # Estas tablas se crean únicamente porque las funciones actuales de
        # guardado y Analysis_Event todavía reciben ambos argumentos.
        # No representan un catálogo ni una pareja lente-fuente.
        custom_metadata_row = pd.DataFrame([event_params])
        TRILEGAL_row = custom_metadata_row.copy()
        GENULENS_row = custom_metadata_row.copy()

        # Evita que las funciones posteriores intenten operar con None.
        path_TRILEGAL_set = "custom_system"
        path_GENULENS_set = "custom_system"

        print("=" * 80)
        print("[sim_fit] catalog_mode='custom_system'")
        print("[sim_fit] No se usaron catálogos ni emparejamiento.")
        print(
            "[sim_fit] Parámetros verdaderos:",
            {
                key: event_params.get(key)
                for key in (
                    "t0",
                    "u0",
                    "tE",
                    "rho",
                    "piEN",
                    "piEE",
                )
                if key in event_params
            },
        )
        print("=" * 80)

    elif catalog_mode == "astrodatalab_pairs":

        # Para mantener reproducibilidad:
        # se usa el mismo seed=i y se elige una fila del catálogo pareado.
        if pair_catalog is not None:
            n_pair_rows = len(pair_catalog)
        else:
            if path_pair_catalog is None:
                raise ValueError(
                    "Si catalog_mode='astrodatalab_pairs', pasá "
                    "pair_catalog o path_pair_catalog."
                )

            path_pair_catalog_obj = Path(path_pair_catalog)

            if path_pair_catalog_obj.suffix == ".parquet":
                n_pair_rows = len(
                    pd.read_parquet(path_pair_catalog_obj)
                )
            elif path_pair_catalog_obj.suffix == ".csv":
                n_pair_rows = (
                    sum(1 for _ in open(path_pair_catalog_obj)) - 1
                )
            else:
                raise ValueError(
                    "path_pair_catalog debe ser .parquet o .csv."
                )

        if n_pair_rows <= 0:
            raise ValueError("El catálogo pareado está vacío.")

        ROW_P = np.random.randint(0, n_pair_rows)

        pair_row = load_pair_catalog_row(
            pair_catalog=pair_catalog,
            path_pair_catalog=path_pair_catalog,
            row_index=ROW_P,
        )

        # Compatibilidad con Analysis_Event y save_sim.
        TRILEGAL_row = pd.DataFrame([pair_row])
        GENULENS_row = pd.DataFrame([pair_row])

        ROW_T = ROW_P
        ROW_G = ROW_P

        event_params = build_event_params_from_pair_row(
            i,
            pair_row,
            system_type,
            param_samplers=param_samplers,
            t0_range=t0_range,
            custom_system=custom_system,
        )

        if path_TRILEGAL_set is None:
            path_TRILEGAL_set = str(path_pair_catalog)

        if path_GENULENS_set is None:
            path_GENULENS_set = str(path_pair_catalog)

    else:
        raise ValueError(
            "catalog_mode debe ser 'trilegal_genulens', "
            "'custom_system' o 'astrodatalab_pairs'."
        )

    # ============================================================
    # 2. Ventana temporal de la simulación
    # ============================================================

    if callable(time_window):
        time_window_use = time_window(event_params)
    else:
        time_window_use = time_window

    # ============================================================
    # 3. Simulación
    # ============================================================

    my_own_model, pyLIMA_parameters, decision = simulate_event_for_fit(
        i,
        event_params,
        path_ephemerides,
        model,
        time_window=time_window_use,
        use_roman=use_roman,
        use_rubin=use_rubin,
        truth_parallax=truth_parallax,
        rubin_pointing_mode=rubin_pointing_mode,
        rubin_cache_cell_deg=rubin_cache_cell_deg,
        apply_detection_criteria=apply_detection_criteria,
        apply_photometric_filter=apply_photometric_filter,
        rubin_saturation_mag=rubin_saturation_mag,
        roman_saturation_mag=roman_saturation_mag,
    )

    # ============================================================
    # Coordenadas GEOMÉTRICAS realmente usadas por la simulación
    # ============================================================
    # No se vuelven a inferir desde catalog_mode. Se leen del Event
    # construido por sim_event, de modo que todos los modos
    # (TRILEGAL+GENULENS, AstroDataLab y custom_system) sean consistentes.

    simulation_event_ra = np.nan
    simulation_event_dec = np.nan

    if my_own_model is not None:
        simulation_event_ra, simulation_event_dec = get_event_coordinates(
            my_own_model
        )

        event_params["event_ra_used"] = simulation_event_ra
        event_params["event_dec_used"] = simulation_event_dec

        print(
            "[sim_fit] geometry used by simulation: "
            f"RA={simulation_event_ra:.10f}, "
            f"Dec={simulation_event_dec:.10f}"
        )

        if "ra" in event_params and "dec" in event_params:
            print(
                "[sim_fit] source/catalog coordinates: "
                f"RA={float(event_params['ra']):.10f}, "
                f"Dec={float(event_params['dec']):.10f}"
            )

    # ============================================================
    # Guardar la coordenada MAF realmente usada
    # ============================================================

    try:
        import set_telescopes_pyLIMA as stp

        maf_info = getattr(stp, "LAST_DATASLICE_INFO", {})

        event_params["rubin_pointing_mode"] = rubin_pointing_mode
        event_params["rubin_cache_cell_deg"] = rubin_cache_cell_deg

        event_params["maf_mode"] = maf_info.get("mode", np.nan)
        event_params["maf_cache_mode"] = maf_info.get(
            "cache_mode",
            np.nan,
        )

        event_params["maf_ra_used"] = maf_info.get(
            "maf_Ra",
            maf_info.get("Ra", np.nan),
        )

        event_params["maf_dec_used"] = maf_info.get(
            "maf_Dec",
            maf_info.get("Dec", np.nan),
        )

        event_params["maf_source_ra"] = maf_info.get(
            "source_Ra",
            event_params.get("ra", np.nan),
        )

        event_params["maf_source_dec"] = maf_info.get(
            "source_Dec",
            event_params.get("dec", np.nan),
        )

    except Exception as error:
        print(
            "[warning] No pude guardar LAST_DATASLICE_INFO:",
            repr(error),
        )

    # ============================================================
    # Guardar diagnóstico del filtro fotométrico
    # ============================================================

    event_params["apply_photometric_filter"] = bool(apply_photometric_filter)

    try:
        event_params.update(
            summarize_photometry_flags(my_own_model)
        )
    except Exception as error:
        print(
            "[warning] No pude resumir photometry flags:",
            repr(error),
        )

    if not decision:
        print("Criteria not satisfied. Event will not be fitted.")

        return {
            "status": "rejected",
            "i": i,
            "model": model,
            "fit_model": fit_model,
            "system_type": system_type,
            "time_window": time_window_use,
            "fit_time_window": fit_time_window,
            "use_roman": use_roman,
            "use_rubin": use_rubin,
            "truth_parallax": truth_parallax,
            "fit_parallax": fit_parallax,
            "catalog_mode": catalog_mode,
            "apply_photometric_filter": apply_photometric_filter,
            "event_params": event_params,
        }

    print("Criteria satisfied. Save the simulated light-curve.")

    # ============================================================
    # 4. Flujo ruidoso y chi2 verdadero
    # ============================================================

    # ============================================================
    # True-model chi2 using the official pyLIMA implementation
    # ============================================================

    from pyLIMA.fits.objective_functions import (
        all_telescope_photometric_chi2,
    )

    n_data_true = sum(
        len(tel.lightcurve)
        for tel in my_own_model.event.telescopes
        if tel.lightcurve is not None
    )

    if n_data_true <= 0:
        raise RuntimeError(
            "No photometric observations for chi2_true."
        )

    # Fail explicitly on invalid photometry.
    for tel in my_own_model.event.telescopes:

        lc = tel.lightcurve

        if lc is None or len(lc) == 0:
            raise RuntimeError(
                f"Empty lightcurve in chi2_true: {tel.name}"
            )

        flux = np.asarray(lc["flux"], dtype=float)
        err = np.asarray(lc["err_flux"], dtype=float)

        if not (
            np.all(np.isfinite(flux))
            and np.all(np.isfinite(err))
            and np.all(err > 0)
        ):
            raise RuntimeError(
                f"Invalid photometry for {tel.name}"
            )

    # No custom residual calculation or silent skipping.
    chi2_true = float(
        all_telescope_photometric_chi2(
            my_own_model,
            pyLIMA_parameters,
        )
    )

    if not np.isfinite(chi2_true):
        raise RuntimeError(
            "pyLIMA returned non-finite chi2_true."
        )

    chi2_true_info = {
        "chi2_true": chi2_true,
        "n_data_true": int(n_data_true),
        "chi2_true_by_telescope": {},
    }

    print(
        "[pyLIMA chi2_true] "
        f"chi2={chi2_true:.6f}, "
        f"N={n_data_true}, "
        f"chi2/N={chi2_true/n_data_true:.6f}"
    )

    print("chi2_true:", chi2_true)
    print("n_data_true:", n_data_true)

    save_simulated_event_if_selected(
        i,
        ROW_G,
        ROW_T,
        path_TRILEGAL_set,
        path_GENULENS_set,
        path_to_save_model,
        my_own_model,
        pyLIMA_parameters,
        event_params,
        GENULENS_row,
        TRILEGAL_row,
    )

    lc_to_fit, lc_to_save = extract_lightcurves_for_fit(
        my_own_model,
    )

    # ============================================================
    # Aplicar ventana temporal SOLO al ajuste
    # ============================================================
    # lc_to_save conserva la curva completa simulada.
    # lc_to_fit se recorta únicamente para el fit.

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

    print("n_fit_points after fit_time_window:", n_fit_points)

    if n_fit_points == 0:

        print(
            "Event rejected after fit_time_window: "
            "no active data points remain for the fit."
        )

        return {
            "status": "rejected_fit_window",
            "i": i,
            "model": model,
            "fit_model": fit_model,
            "system_type": system_type,
            "time_window": time_window_use,
            "fit_time_window": fit_time_window,
            "use_roman": use_roman,
            "use_rubin": use_rubin,
            "truth_parallax": truth_parallax,
            "fit_parallax": fit_parallax,
            "catalog_mode": catalog_mode,
            "apply_photometric_filter": apply_photometric_filter,
            "event_params": event_params,
        }

    origin = get_model_origin(
        my_own_model,
    )

    # ============================================================
    # 5. Ajustes
    # ============================================================

    rango = 1

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
        fit_model=fit_model,
        fit_parallax=fit_parallax,
        fit_defaults=fit_defaults,
        fit_bounds=fit_bounds,
        initial_guess=initial_guess,
        optimizer_options=optimizer_options,
        event_ra=simulation_event_ra,
        event_dec=simulation_event_dec,
    )

    # ============================================================
    # Verificación obligatoria: simulación y fit misma geometría
    # ============================================================

    validate_event_coordinates(
        pyLIMAmodel_rr,
        simulation_event_ra,
        simulation_event_dec,
        label="Rubin+Roman/Rubin fit",
    )

    if pyLIMAmodel_roman is not None:
        validate_event_coordinates(
            pyLIMAmodel_roman,
            simulation_event_ra,
            simulation_event_dec,
            label="Roman-only fit",
        )

    # ============================================================
    # 6. Análisis y guardado
    # ============================================================

    if catalog_mode == "custom_system":
        nset_str = "custom"
    else:
        nset_str = extract_nset_string(
            path_GENULENS_set,
        )

    Event = build_analysis_event(
        i,
        model,
        algo,
        path_to_save_model,
        path_to_save_fit,
        GENULENS_row,
        TRILEGAL_row,
        event_params,
        fit_rr,
        fit_roman,
        origin,
        lc_to_save,
        pyLIMA_parameters,
        use_roman=use_roman,
        use_rubin=use_rubin,
        truth_parallax=truth_parallax,
        fit_model=fit_model,
        fit_parallax=fit_parallax,
        fit_time_window=fit_time_window,
        chi2_true=chi2_true_info,
    )

    save_roman = fit_roman is not None

    (
        new_data_true,
        new_data_rr,
        new_data_roman,
    ) = save_extracted_results(
        Event,
        model,
        i,
        system_type,
        nset_str,
        path_to_save_results,
        save_roman=save_roman,
    )

    tend = time.time()

    print("El tiempo transcurrido fue de:", tend - tstart)

    if return_data:
        return {
            "status": "fitted",
            "i": i,
            "sim_model": model,
            "fit_model": (
                fit_model
                if fit_model is not None
                else model
            ),
            "truth_parallax": truth_parallax,
            "fit_parallax": fit_parallax,
            "initial_guess": initial_guess,
            "fit_time_window": fit_time_window,
            "algo": algo,
            "system_type": system_type,
            "time_window": time_window_use,
            "use_roman": use_roman,
            "use_rubin": use_rubin,
            "catalog_mode": catalog_mode,
            "apply_photometric_filter": apply_photometric_filter,
            "event_params": event_params,
            "fit_rr": fit_rr,
            "pyLIMAmodel_rr": pyLIMAmodel_rr,
            # Modelo/parametros exactos usados para generar los datos.
            # Útiles para plots sin reconstruir el Event.
            "pyLIMAmodel_true": my_own_model,
            "pyLIMA_parameters_true": pyLIMA_parameters,
            "event_ra_used": simulation_event_ra,
            "event_dec_used": simulation_event_dec,
            "fit_roman": fit_roman,
            "pyLIMAmodel_roman": pyLIMAmodel_roman,
            "true": new_data_true,
            "fit_rr_data": new_data_rr,
            "fit_roman_data": new_data_roman,
        }

    return (
        fit_rr,
        pyLIMAmodel_rr,
        fit_roman,
        pyLIMAmodel_roman,
    )

def remove_empty_telescopes(event, verbose=True):
    """
    Elimina del evento los telescopios/bandas sin observaciones.

    pyLIMA construye un parámetro de flujo por telescopio y ejecuta
    np.max(flux). Si la curva está vacía, la creación del modelo falla.
    """

    valid_telescopes = []
    removed_telescopes = []

    for telescope in event.telescopes:

        lightcurve = getattr(
            telescope,
            "lightcurve",
            None,
        )

        if lightcurve is None:
            n_points = 0
        else:
            try:
                n_points = len(lightcurve)
            except TypeError:
                n_points = 0

        if n_points > 0:
            valid_telescopes.append(telescope)
        else:
            removed_telescopes.append(telescope.name)

    event.telescopes = valid_telescopes

    if verbose and removed_telescopes:
        print(
            "Removed empty telescopes before model creation:",
            removed_telescopes,
        )

    return event, removed_telescopes

def read_fit(
    nsource,
    nset,
    path_run,
    model,
    algo,
    path_to_save_fit,
    path_ephemerides,
    fit_model=None,
    fit_parallax=True,
):

    path_event = path_run + f'/set_sim{nset}/Event_{nsource}.h5'

    print('path_event', path_event)
    print('os.path.getsize(path_model):', os.path.getsize(path_event))
    if os.path.getsize(path_event) == 0:
        return
    else:
        model, fit_model, fit_parallax = resolve_fit_options(
            model,
            fit_model=fit_model,
            fit_parallax=fit_parallax,
            default_fit_parallax=True,
        )

        info_event, pyLIMA_parameters, curves = read_data(path_event)

        lc_to_fit = {}
        for telo in curves:
            if len(curves[telo]['mag']) != 0:
                df = curves[telo][['time', 'mag', 'err_mag']].to_pandas()
                lc_to_fit[telo] = df.values
            else:
                lc_to_fit[telo] = []

        origin = info_event[2]
        rango = 1

        print("Start the fit using Roman and Rubin data:")
        fit_rr, event_fit_rr, pyLIMAmodel_rr = fit_rubin_roman(
            nsource, pyLIMA_parameters, path_to_save_fit, path_ephemerides,
            model, algo, origin, rango,
            lc_to_fit["W149"], lc_to_fit["u"], lc_to_fit["g"], lc_to_fit["r"],
            lc_to_fit["i"], lc_to_fit["z"], lc_to_fit["y"],
            fit_model=fit_model,
            fit_parallax=fit_parallax,
        )
        print("Start the fit using only the Roman data:")
        fit_roman, event_fit_roman, pyLIMAmodel_roman = fit_rubin_roman(
            nsource, pyLIMA_parameters, path_to_save_fit, path_ephemerides,
            model, algo, origin, rango,
            lc_to_fit["W149"], [], [], [], [], [], [],
            fit_model=fit_model,
            fit_parallax=fit_parallax,
        )

# ============================================================================
# Multi-fit generalization: one simulation, multiple fits on the same noisy LC
# ============================================================================


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


# Redefinition compatible with multi-fit runs.  Since Python resolves globals at
# call time, the existing sim_fit() defined above will also use this new version.
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


def _time_mask_from_fit_times(original_times, fit_times, atol=1.0e-9):
    """Return a mask selecting original_times that are in fit_times."""

    original_times = np.asarray(original_times, dtype=float)
    fit_times = np.asarray(fit_times, dtype=float)

    if len(original_times) == 0 or len(fit_times) == 0:
        return np.zeros(len(original_times), dtype=bool)

    # The times normally come from the same table, so exact matching should
    # work.  The tolerance fallback protects against astropy/pandas round-off.
    rounded_original = np.round(original_times / atol).astype(np.int64)
    rounded_fit = np.round(fit_times / atol).astype(np.int64)

    return np.isin(rounded_original, rounded_fit)


def copy_true_model_on_fit_lightcurves(pyLIMA_model_true, lc_to_fit):
    """
    Copy the true model and crop its telescopes to exactly the same timestamps
    used by ``lc_to_fit``.

    This is needed for the oracle true-generator likelihood to be evaluated on
    the same data as H0/H1, especially when a fit-only time window is used.
    """

    model_copy = copy.deepcopy(pyLIMA_model_true)

    for telescope in model_copy.event.telescopes:
        tel_name = str(telescope.name)

        if telescope.lightcurve is None or len(telescope.lightcurve) == 0:
            continue

        arr = lc_to_fit.get(tel_name, [])

        if arr is None or len(arr) == 0:
            telescope.lightcurve = telescope.lightcurve[:0]
            continue

        arr = np.asarray(arr, dtype=float)
        fit_times = arr[:, 0]

        original_times = _plain_array(
            telescope.lightcurve["time"],
            dtype=float,
        )

        mask = _time_mask_from_fit_times(
            original_times,
            fit_times,
        )

        telescope.lightcurve = telescope.lightcurve[mask]

    return model_copy



# ================================================================
#  True-generator likelihood on a fresh fit model
# ================================================================


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
    catalog_mode="trilegal_genulens",
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
    rubin_pointing_mode="fixed",
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

    origin = get_model_origin(true_model)
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



# ================================================================
#  OVERRIDES 2026-08-26
#  Consistent chi2-based likelihoods for H0/H1/true_generator
# ================================================================

# These definitions intentionally override the earlier versions above.
# They avoid pyLIMA's absolute photometric-likelihood normalization for the
# hypothesis tests and use the Gaussian likelihood up to a common constant:
#
#     logL_chi2 = -0.5 * chi2
#     nll_chi2  =  0.5 * chi2
#
# For H0/H1/true evaluated on the same surviving light-curve points, the
# omitted normalization constant cancels exactly in likelihood-ratio tests.


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
