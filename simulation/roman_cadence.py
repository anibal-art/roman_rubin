"""Roman GBTDS time sampling.

6 nominal seasons at 12.1 min cadence plus 3 off-seasons at 72 h; the
cached template is built by set_telescopes_pyLIMA._load_or_build_roman_template.
"""

import numpy as np
from astropy.time import Time
from pyLIMA.simulations import simulator


def _structured_to_matrix(arr, fields):
    """Convierte un array estructurado 1D a matriz 2D por campos; si ya es 2D, lo retorna."""
    if hasattr(arr, "dtype") and getattr(arr.dtype, "names", None):
        # asegurar que todos los campos existen
        fields = [f for f in fields if f in arr.dtype.names]
        return np.column_stack([arr[f] for f in fields])
    # si no es estructurado pero es 1D, forzamos 2D si se puede
    arr = np.asarray(arr)
    if arr.ndim == 1:
        # no sabemos columnas: devolvemos con shape (N,1) para que falle antes que silently wrong
        return arr.reshape(-1, 1)
    return arr


def _roman_nominal_and_off_seasons():
    """
    Define las temporadas Roman usadas en la simulación.
    """
    nominal_seasons = [
        {"start": "2027-02-11T00:00:00", "end": "2027-04-24T00:00:00"},
        {"start": "2027-08-16T00:00:00", "end": "2027-10-27T00:00:00"},
        {"start": "2028-02-11T00:00:00", "end": "2028-04-24T00:00:00"},
        {"start": "2030-02-11T00:00:00", "end": "2030-04-24T00:00:00"},
        {"start": "2030-08-16T00:00:00", "end": "2030-10-27T00:00:00"},
        {"start": "2031-02-11T00:00:00", "end": "2031-04-24T00:00:00"},
    ]

    off_seasons = [
        {"start": "2028-08-15T00:00:00", "end": "2028-10-27T00:00:00"},
        {"start": "2029-02-11T00:00:00", "end": "2029-04-24T00:00:00"},
        {"start": "2029-08-16T00:00:00", "end": "2029-10-27T00:00:00"},
    ]

    return nominal_seasons, off_seasons


def _simulate_roman_season(season, sampling):
    """
    Simula una temporada Roman W149.
    """
    tstart = Time(season["start"], format="isot").jd
    tend = Time(season["end"], format="isot").jd

    Roman = simulator.simulate_a_telescope(
        name="W149",
        time_start=tstart,
        time_end=tend,
        sampling=sampling,
        location="Space",
        camera_filter="W149",
        uniform_sampling=True,
        astrometry=False,
    )

    return Roman.lightcurve
