"""Deterministic, precomputed blending for Roman-Rubin catalogs."""

import hashlib
import numpy as np

BLENDING_SCHEME = "independent_uniform_g_v1"

BAND_MAG = {
    "W149": "W149",
    "u": "u",
    "g": "g",
    "r": "r",
    "i": "i",
    "z": "z",
    "y": "Y",
}

ZERO_POINTS = {
    "W149": 27.615,
    "u": 27.03,
    "g": 28.38,
    "r": 28.16,
    "i": 27.85,
    "z": 27.46,
    "y": 26.68,
}


def _uniform_from_seeds(seeds, band):
    """Stateless reproducible U(0,1) draw per event and band.

    SplitMix64 mixing: changing the number or order of events
    does not change their realizations.
    """
    key = f"{BLENDING_SCHEME}:{band}".encode()
    salt = int.from_bytes(
        hashlib.blake2b(key, digest_size=8).digest(),
        "little",
    )

    x = np.asarray(seeds, dtype=np.uint64).copy()

    with np.errstate(over="ignore"):
        x ^= np.uint64(salt)
        x += np.uint64(0x9E3779B97F4A7C15)
        x = (
            (x ^ (x >> 30))
            * np.uint64(0xBF58476D1CE4E5B9)
        )
        x = (
            (x ^ (x >> 27))
            * np.uint64(0x94D049BB133111EB)
        )
        x ^= x >> 31

    return (
        (x >> 11).astype(np.float64) + 0.5
    ) / (2.0 ** 53)


def blending_columns(data):
    """Return all photometric realization columns.

    data must expose event_seed and the existing magnitude
    columns. Works with dictionaries of arrays or DataFrames.
    """
    seeds = np.asarray(data["event_seed"], dtype=np.uint64)

    result = {}

    for band, mag_column in BAND_MAG.items():
        magnitude = np.asarray(
            data[mag_column], dtype=np.float64
        )

        if not np.isfinite(magnitude).all():
            raise ValueError(
                f"Non-finite baseline magnitude in {mag_column}"
            )

        g = _uniform_from_seeds(seeds, band)

        fsource = 10.0 ** (
            (ZERO_POINTS[band] - magnitude) / 2.5
        )
        ftotal = fsource * (1.0 + g)

        if not (
            np.isfinite(ftotal).all()
            and (ftotal > 0).all()
            and np.isfinite(fsource).all()
            and (fsource > 0).all()
        ):
            raise ValueError(
                f"Invalid flux in band {band}"
            )

        result[f"blend_ratio_{band}"] = g
        result[f"fsource_{band}"] = fsource
        result[f"ftotal_{band}"] = ftotal

    return result
