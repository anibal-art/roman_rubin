"""Functions of fit_lc.py that nothing calls (moved in step 7; not imported)."""


def event_creation(
    Source,
    path_ephemerides,
    wfirst_lc,
    lsst_u,
    lsst_g,
    lsst_r,
    lsst_i,
    lsst_z,
    lsst_y,
    event_ra=None,
    event_dec=None,
):
    """
    Wrapper compatible con la versión vieja.
    """
    lsst_lcs = {
        "u": as_empty_if_none(lsst_u),
        "g": as_empty_if_none(lsst_g),
        "r": as_empty_if_none(lsst_r),
        "i": as_empty_if_none(lsst_i),
        "z": as_empty_if_none(lsst_z),
        "y": as_empty_if_none(lsst_y),
    }

    return create_fit_event(
        Source,
        path_ephemerides,
        as_empty_if_none(wfirst_lc),
        lsst_lcs,
        ra=event_ra,
        dec=event_dec,
        roman_name="Roman",
    )
