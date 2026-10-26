"""Build the pyLIMA Event that is fitted.

lc_to_fit arrays -> pyLIMA Event with a Roman telescope ("W149") and one
telescope per Rubin band with data, at the RA/Dec of the simulated event
(resolve_event_coordinates raises if they are missing).
"""

import numpy as np
from pyLIMA import event
from pyLIMA import telescopes

LSST_BANDS = [
    "u",
    "g",
    "r",
    "i",
    "z",
    "y",
]


def is_nonempty_lc(lc):
    """
    Devuelve True si la curva de luz tiene al menos un punto.
    """
    if lc is None:
        return False

    try:
        return len(lc) != 0
    except TypeError:
        return False


def as_empty_if_none(lc):
    """
    Convierte None en lista vacía.
    """
    if lc is None:
        return []

    return lc


def has_rubin_data(lsst_lcs):
    """
    Devuelve True si hay al menos una banda Rubin con datos.
    """
    return any(
        is_nonempty_lc(lc)
        for lc in lsst_lcs.values()
    )


def get_event_name(Source, wfirst_lc, lsst_lcs):
    """
    Define el nombre del evento según qué telescopios tienen datos.
    """
    has_roman = is_nonempty_lc(wfirst_lc)
    has_rubin = has_rubin_data(lsst_lcs)

    if has_roman and has_rubin:
        return f"Event_RR_{Source}"

    if has_roman and not has_rubin:
        return f"Event_Roman_{Source}"

    if (not has_roman) and has_rubin:
        return f"Event_Rubin_{Source}"

    raise ValueError(
        "No hay datos para ajustar: Roman y Rubin están vacíos."
    )


def resolve_event_coordinates(
    event_ra=None,
    event_dec=None,
):
    """
    Resuelve las coordenadas que usará el Event del fit/modelo.

    Reglas
    ------
    - event_ra y event_dec son obligatorias: no hay coordenadas por
      defecto (cada evento usa su RA/Dec del catálogo).
    - Las coordenadas explícitas deben ser finitas.

    En el pipeline nuevo, sim_fit debe pasar aquí las coordenadas del
    Event realmente usado durante la simulación (my_own_model.event).
    De esta forma simulación y ajuste usan exactamente la misma geometría
    para el cálculo del paralaje.
    """
    if event_ra is None or event_dec is None:
        raise ValueError(
            "event_ra y event_dec son obligatorias (RA/Dec del evento). "
            f"Recibí event_ra={event_ra!r}, event_dec={event_dec!r}."
        )

    ra = float(event_ra)
    dec = float(event_dec)

    if not np.isfinite(ra) or not np.isfinite(dec):
        raise ValueError(
            "Las coordenadas del Event deben ser finitas: "
            f"RA={ra}, Dec={dec}."
        )

    if dec < -90.0 or dec > 90.0:
        raise ValueError(
            f"Declinación fuera de rango: Dec={dec}."
        )

    return ra, dec


def add_roman_telescope(
    e,
    wfirst_lc,
    path_ephemerides,
    roman_name="Roman",
):
    """
    Agrega Roman al evento si la curva no está vacía.
    """
    if not is_nonempty_lc(wfirst_lc):
        return e

    tel = telescopes.Telescope(
        name=roman_name,
        camera_filter="W149",
        lightcurve=wfirst_lc,
        lightcurve_names=[
            "time",
            "mag",
            "err_mag",
        ],
        lightcurve_units=[
            "JD",
            "mag",
            "mag",
        ],
        location="Space",
    )

    tel.spacecraft_positions = {
        "astrometry": [],
        "photometry": np.load(path_ephemerides),
    }

    e.telescopes.append(tel)

    return e


def add_rubin_telescopes(e, lsst_lcs):
    """
    Agrega telescopios Rubin al evento para las bandas no vacías.
    """
    for band in LSST_BANDS:

        lc = lsst_lcs.get(
            band,
            [],
        )

        if not is_nonempty_lc(lc):
            continue

        tel = telescopes.Telescope(
            name=band,
            camera_filter=band,
            lightcurve=lc,
            lightcurve_names=[
                "time",
                "mag",
                "err_mag",
            ],
            lightcurve_units=[
                "JD",
                "mag",
                "mag",
            ],
            location="Earth",
        )

        e.telescopes.append(tel)

    return e


def create_fit_event(
    Source,
    path_ephemerides,
    wfirst_lc,
    lsst_lcs,
    ra=None,
    dec=None,
    roman_name="Roman",
):
    """
    Construye el Event de pyLIMA.

    Puede representar:
    - Roman + Rubin
    - Roman only
    - Rubin only
    """
    wfirst_lc = as_empty_if_none(wfirst_lc)

    ra_use, dec_use = resolve_event_coordinates(
        event_ra=ra,
        event_dec=dec,
    )

    e = event.Event(
        ra=ra_use,
        dec=dec_use,
    )

    e.name = get_event_name(
        Source,
        wfirst_lc,
        lsst_lcs,
    )

    e = add_roman_telescope(
        e,
        wfirst_lc,
        path_ephemerides,
        roman_name=roman_name,
    )

    e = add_rubin_telescopes(
        e,
        lsst_lcs,
    )

    e.check_event()

    return e
