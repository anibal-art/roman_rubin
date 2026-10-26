"""Roman + Rubin photometric noise stage of the simulation.

Turns the noiseless pyLIMA light curves into observed photometry:

* Roman W149: delegated to ``photometry.roman_photometry``
  (Pandeia F146 error model and saturation flags).
* Rubin u,g,r,i,z,y: rubin_sim m5-based error per visit
  (``calc_mag_error_m5``) using the event's OpSim ``dataSlice``
  (``filter``, ``fiveSigmaDepth`` and an optional saturation column).
  The per-band gamma vector is cached in ``_RUBIN_GAMMA_VECTOR_CACHE``.

Invariants (changing them changes every simulated event):

* Noise is drawn for every epoch BEFORE the photometric filter
  (model mag fainter than m5, or saturated), so a filtered light curve
  is a subset of one noise realization.
* The order of the ``np.random.normal`` calls defines the realization
  for a given seed.

Also holds the per-point ``photometry_*`` flag bookkeeping and the
summary columns written to the output tables.

Moved verbatim from ``functions_roman_rubin.py``, which re-exports
every name defined here.
"""

import numpy as np
from astropy.table import QTable
from detection_criteria import mag
from photometry.constants import SIMULATION_BAND_ZERO_POINTS
from photometry.roman_photometry import apply_roman_f146_photometry
from rubin_sim.phot_utils.photometric_parameters import PhotometricParameters
from rubin_sim.phot_utils.signaltonoise import calc_gamma
from rubin_sim.phot_utils.signaltonoise import calc_mag_error_m5


_RUBIN_GAMMA_VECTOR_CACHE = {}

_RUBIN_GAMMA_VECTOR_CACHE_STATS = {
    "vector_hits": 0,
    "vector_misses": 0,
    "gamma_values_computed": 0,
}


def set_photometric_parameters(exptime, nexp, readnoise=None):
    # readnoise = None will use the default (8.8 e/pixel). Readnoise should be in electrons/pixel.
    photParams = PhotometricParameters(exptime=exptime, nexp=nexp, readnoise=readnoise)
    return photParams


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
