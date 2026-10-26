"""Helpers for the Rubin OpSim dataslice.

* canonical Rubin filter names (u, g, r, i, z, y) and the Rubin time series;
* _quantize_sky_position: sky cell of the per-position dataslice cache.
"""

import numpy as np


def _canonical_rubin_filter_array(values):
    """
    Convierte nombres de filtro de OpSim a bandas Rubin canónicas.

    Ejemplos:
        'g'    -> 'g'
        'g_6'  -> 'g'
        'r_57' -> 'r'
        b'i_39' -> 'i'
    """

    arr = np.asarray(values)

    if arr.dtype.kind == "S":
        arr = np.char.decode(arr, "utf-8")
    else:
        arr = arr.astype(str)

    out = []

    for value in arr:
        value = str(value).strip()

        if len(value) >= 4 and value.startswith("b'") and value.endswith("'"):
            value = value[2:-1]

        if len(value) >= 4 and value.startswith('b"') and value.endswith('"'):
            value = value[2:-1]

        # Caso nuevo de baseline v5.3.5:
        # g_6, i_39, r_57, u_24, y_10, z_20
        band = value.split("_")[0].lower()

        if band not in {"u", "g", "r", "i", "z", "y"}:
            raise ValueError(
                f"No pude interpretar el filtro Rubin {value!r} como banda ugrizy."
            )

        out.append(band)

    return np.asarray(out, dtype=str)


def _canonicalize_dataslice_filter_column(dataSlice):
    """
    Devuelve una copia del dataSlice con la columna de filtro canonicalizada.

    Esto permite que el resto del pipeline, incluido functions_roman_rubin.py,
    siga usando comparaciones antiguas como:

        dataSlice["filter"] == "r"

    aunque la OpSim nueva tenga valores como:

        r_57, g_6, i_39, u_24, y_10, z_20

    Los telescopios pyLIMA se siguen llamando u,g,r,i,z,y.
    """

    if dataSlice is None:
        return dataSlice

    names = getattr(getattr(dataSlice, "dtype", None), "names", None)

    if names is None:
        return dataSlice

    if "filter" in names:
        column = "filter"
    elif "band" in names:
        column = "band"
    else:
        return dataSlice

    canonical = _canonical_rubin_filter_array(dataSlice[column])

    # Copia explícita: no modifica el objeto original de MAF in-place.
    out = dataSlice.copy()

    try:
        out[column] = canonical
    except Exception:
        # Fallback robusto para dtypes muy restrictivos.
        out[column] = canonical.astype(out[column].dtype, copy=False)

    return out


def _build_rubin_ts(dataSlice, lsst_filterlist="ugrizy"):
    """
    Construye las curvas base de Rubin a partir del dataSlice.

    Cada telescopio Rubin recibe columnas:
        time, mag, err_mag

    donde mag y err_mag se cargan inicialmente con m5.
    """

    rubin_ts = {}

    if "filter" in dataSlice.dtype.names:
        filter_values = _canonical_rubin_filter_array(dataSlice["filter"])
    elif "band" in dataSlice.dtype.names:
        filter_values = _canonical_rubin_filter_array(dataSlice["band"])
    else:
        raise KeyError(
            "dataSlice no tiene columna 'filter' ni 'band'. "
            f"Columnas disponibles: {dataSlice.dtype.names}"
        )

    print(
        "[MAF] Rubin observations by canonical band:",
        {
            fil: int(np.sum(filter_values == fil))
            for fil in lsst_filterlist
        },
        flush=True,
    )

    for fil in lsst_filterlist:
        mask = filter_values == fil

        m5 = dataSlice["fiveSigmaDepth"][mask]
        mjd = dataSlice["observationStartMJD"][mask] + 2400000.5

        int_array = np.column_stack(
            (
                mjd,
                m5,
                m5,
            )
        ).astype(float)

        rubin_ts[fil] = int_array

    return rubin_ts


def _quantize_sky_position(Ra, Dec, cache_cell_deg=None):
    """
    Devuelve una coordenada para cachear/consultar MAF.

    Si cache_cell_deg is None:
        usa la coordenada exacta.

    Si cache_cell_deg tiene valor, por ejemplo 0.05:
        agrupa las coordenadas en celdas de 0.05 deg.
    """

    Ra = float(Ra)
    Dec = float(Dec)

    if cache_cell_deg is None:
        return Ra, Dec, "exact"

    cache_cell_deg = float(cache_cell_deg)

    if cache_cell_deg <= 0:
        raise ValueError("cache_cell_deg debe ser positivo o None.")

    Ra_q = np.round(Ra / cache_cell_deg) * cache_cell_deg
    Dec_q = np.round(Dec / cache_cell_deg) * cache_cell_deg

    Ra_q = Ra_q % 360.0
    Dec_q = np.clip(Dec_q, -90.0, 90.0)

    return float(Ra_q), float(Dec_q), "cell"
