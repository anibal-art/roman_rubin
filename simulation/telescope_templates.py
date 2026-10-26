"""In-process cache of the Roman and Rubin pyLIMA telescope templates.

Building a pyLIMA telescope (and Roman's ephemerides) is the same for every
event with the same time array; _build_event_template reuses copies of
cached templates. The caches are dicts mutated in place.
"""

import copy
import hashlib
import numpy as np
import os
from pyLIMA import event
from pyLIMA import telescopes

_ROMAN_TELESCOPE_TEMPLATE_CACHE = {}

_RUBIN_TELESCOPE_TEMPLATE_CACHE = {}

_ROMAN_EPHEMERIDES_TEMPLATE_CACHE = {}

_TELESCOPE_TEMPLATE_CACHE_STATS = {
    "roman_hits": 0,
    "roman_misses": 0,
    "rubin_hits": 0,
    "rubin_misses": 0,
}


def _telescope_array_signature(values):
    """
    Exact SHA256 signature of an input numerical array.
    """
    arr = np.ascontiguousarray(
        np.asarray(values)
    )

    if arr.dtype.hasobject:
        raise TypeError(
            "Object arrays are not valid Telescope-cache inputs."
        )

    h = hashlib.sha256()
    h.update(arr.dtype.str.encode("ascii"))
    h.update(repr(arr.shape).encode("ascii"))
    h.update(arr.tobytes(order="C"))

    return h.hexdigest()


def _load_cached_roman_ephemerides(path_ephemerides):
    """
    Load one unchanged Roman ephemerides file once per process.
    """
    path = os.path.abspath(
        os.path.expanduser(
            str(path_ephemerides)
        )
    )

    st = os.stat(path)

    file_key = (
        path,
        int(st.st_size),
        int(st.st_mtime_ns),
    )

    cached = (
        _ROMAN_EPHEMERIDES_TEMPLATE_CACHE
        .get(file_key)
    )

    if cached is None:

        ephemerides = np.load(path)

        cached = (
            ephemerides,
            _telescope_array_signature(
                ephemerides
            ),
        )

        _ROMAN_EPHEMERIDES_TEMPLATE_CACHE[
            file_key
        ] = cached

    return cached


def _get_cached_roman_telescope(
    roman_mag,
    path_ephemerides,
):
    """
    Return a fresh Roman Telescope copied from a pristine template.
    """
    ephemerides, ephemerides_signature = (
        _load_cached_roman_ephemerides(
            path_ephemerides
        )
    )

    key = (
        _telescope_array_signature(
            roman_mag
        ),
        ephemerides_signature,
    )

    template = (
        _ROMAN_TELESCOPE_TEMPLATE_CACHE
        .get(key)
    )

    if template is None:

        template = telescopes.Telescope(
            name="W149",
            camera_filter="W149",
            location="Space",
            lightcurve=roman_mag,
            lightcurve_names=[
                "time",
                "mag",
                "err_mag",
            ],
            lightcurve_units=[
                "d",
                "mag",
                "mag",
            ],
        )

        template.spacecraft_name = "L2"

        template.spacecraft_positions = {
            "astrometry": [],
            "photometry": ephemerides,
        }

        _ROMAN_TELESCOPE_TEMPLATE_CACHE[
            key
        ] = template

        _TELESCOPE_TEMPLATE_CACHE_STATS[
            "roman_misses"
        ] += 1

    else:

        _TELESCOPE_TEMPLATE_CACHE_STATS[
            "roman_hits"
        ] += 1

    return copy.deepcopy(
        template
    )


def _get_cached_rubin_telescope(
    band,
    lightcurve,
):
    """
    Return a fresh Rubin Telescope copied from a pristine template.

    The cache key uses the complete input [time, mag, err_mag]
    array, not only its timestamps.
    """
    band = str(band)

    key = (
        band,
        _telescope_array_signature(
            lightcurve
        ),
    )

    template = (
        _RUBIN_TELESCOPE_TEMPLATE_CACHE
        .get(key)
    )

    if template is None:

        template = telescopes.Telescope(
            name=band,
            camera_filter=band,
            location="Earth",
            lightcurve=lightcurve,
            lightcurve_names=[
                "time",
                "mag",
                "err_mag",
            ],
            lightcurve_units=[
                "d",
                "mag",
                "mag",
            ],
        )

        _RUBIN_TELESCOPE_TEMPLATE_CACHE[
            key
        ] = template

        _TELESCOPE_TEMPLATE_CACHE_STATS[
            "rubin_misses"
        ] += 1

    else:

        _TELESCOPE_TEMPLATE_CACHE_STATS[
            "rubin_hits"
        ] += 1

    return copy.deepcopy(
        template
    )


def _build_event_template(
    Ra,
    Dec,
    roman_mag,
    rubin_ts,
    path_ephemerides,
    lsst_filterlist="ugrizy",
    use_roman=True,
    use_rubin=True,
):
    """
    Construye el Event base con Roman y/o Rubin.
    """

    my_own_creation = event.Event(
        ra=Ra,
        dec=Dec,
    )

    my_own_creation.name = "An event observed by Roman/Rubin"

    if use_roman:

        Roman_tot = _get_cached_roman_telescope(
            roman_mag,
            path_ephemerides,
        )

        my_own_creation.telescopes.append(
            Roman_tot
        )

    if use_rubin:

        for band in lsst_filterlist:

            lsst_telescope = _get_cached_rubin_telescope(
                band,
                rubin_ts[band],
            )

            my_own_creation.telescopes.append(
                lsst_telescope
            )


    if len(my_own_creation.telescopes) == 0:
        raise ValueError("No hay telescopios activos: use_roman=False y use_rubin=False.")

    return my_own_creation
