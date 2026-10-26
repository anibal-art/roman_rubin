"""Optional time window applied to telescopes and to the OpSim dataslice.
"""

import numpy as np


def _get_col_values(col):
    """
    Devuelve valores numéricos desde una columna astropy Quantity/Column
    o desde un array numpy.
    """
    return col.value if hasattr(col, "value") else np.asarray(col)


def _validate_time_window(time_window):
    """
    Normaliza una ventana temporal en JD.
    """
    if time_window is None:
        return None

    if len(time_window) != 2:
        raise ValueError("time_window debe ser None o una tupla (time_min, time_max).")

    time_min, time_max = time_window

    if time_min is None and time_max is None:
        return None

    if time_min is not None:
        time_min = float(time_min)

    if time_max is not None:
        time_max = float(time_max)

    if time_min is not None and time_max is not None:
        if time_min > time_max:
            raise ValueError("time_min no puede ser mayor que time_max.")

    return time_min, time_max


def slice_lightcurve_by_time(lightcurve, time_window=None):
    """
    Recorta una lightcurve usando la columna 'time'.
    """
    time_window = _validate_time_window(time_window)

    if time_window is None:
        return lightcurve

    time_min, time_max = time_window

    t = _get_col_values(lightcurve["time"])

    mask = np.ones(len(t), dtype=bool)

    if time_min is not None:
        mask &= t >= time_min

    if time_max is not None:
        mask &= t <= time_max

    return lightcurve[mask]


def slice_dataslice_by_time(dataSlice, time_window=None):
    """
    Recorta el dataSlice de Rubin usando observationStartMJD.

    Las curvas de luz están en JD, mientras que OpSim/MAF usa MJD.
    Por eso se convierte:

        JD = MJD + 2400000.5
    """
    time_window = _validate_time_window(time_window)

    if time_window is None:
        return dataSlice

    time_min, time_max = time_window

    jd = dataSlice["observationStartMJD"] + 2400000.5

    mask = np.ones(len(jd), dtype=bool)

    if time_min is not None:
        mask &= jd >= time_min

    if time_max is not None:
        mask &= jd <= time_max

    return dataSlice[mask]


def apply_time_window_to_event(my_event, time_window=None):
    """
    Aplica una ventana temporal a todos los telescopios del evento.
    """
    time_window = _validate_time_window(time_window)

    if time_window is None:
        return my_event

    for tel in my_event.telescopes:
        tel.lightcurve = slice_lightcurve_by_time(
            tel.lightcurve,
            time_window=time_window,
        )

    return my_event
