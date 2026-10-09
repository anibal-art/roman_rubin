"""Exact process-local cache for pyLIMA Earth ephemerides.

The expensive operation is

    pyLIMA.parallax.parallax.Earth_ephemerides(times)

which ultimately calls Astropy's
get_body_barycentric_posvel("Earth", ...).

This module caches only vector evaluations. Scalar evaluations such as
the parallax reference epoch are always delegated to pyLIMA/Astropy.

A cached value is used only when the requested time array is exactly
identical to a cached time array. Unknown arrays fall back to the
original pyLIMA implementation and are then cached in memory.

No RNG state is touched.
"""

from pathlib import Path
import threading

import numpy as np

from pyLIMA.parallax import parallax


_REPO_ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_PRELOAD = (
    _REPO_ROOT
    / ".rr_cache"
    / "earth_ephemerides_fixed_survey.npz"
)


_ORIGINAL_EARTH_EPHEMERIDES = parallax.Earth_ephemerides

_CACHE = []
_INSTALLED = False
_PRELOADED = False
_LOCK = threading.RLock()

_STATS = {
    "vector_hits": 0,
    "vector_misses": 0,
    "scalar_fallbacks": 0,
    "preloaded_vectors": 0,
}


def _as_times(time_to_treat):
    return np.asarray(
        time_to_treat,
        dtype=float,
    )


def _find_exact(times):
    for item in _CACHE:
        cached_times = item["times"]

        if (
            times.shape == cached_times.shape
            and np.array_equal(
                times,
                cached_times,
            )
        ):
            return item

    return None


def _store(times, positions, speeds):
    if _find_exact(times) is not None:
        return

    _CACHE.append(
        {
            "times": np.asarray(
                times,
                dtype=float,
            ).copy(),
            "positions": np.asarray(
                positions,
                dtype=float,
            ).copy(),
            "speeds": np.asarray(
                speeds,
                dtype=float,
            ).copy(),
        }
    )


def preload_cache(path=None):
    """Load an exact precomputed cache if present."""

    global _PRELOADED

    with _LOCK:
        if _PRELOADED:
            return

        cache_path = (
            Path(path)
            if path is not None
            else _DEFAULT_PRELOAD
        )

        if not cache_path.exists():
            _PRELOADED = True
            return

        with np.load(
            cache_path,
            allow_pickle=False,
        ) as z:

            n_vectors = int(
                z["n_vectors"][0]
            )

            for i in range(n_vectors):
                _store(
                    z[f"times_{i}"],
                    z[f"positions_{i}"],
                    z[f"speeds_{i}"],
                )

        _STATS["preloaded_vectors"] = len(_CACHE)
        _PRELOADED = True


def cached_earth_ephemerides(time_to_treat):
    """Drop-in replacement for pyLIMA Earth_ephemerides."""

    times = _as_times(
        time_to_treat
    )

    # Reference epochs are cheap and remain entirely in pyLIMA/Astropy.
    if times.ndim == 0 or times.size <= 1:
        _STATS["scalar_fallbacks"] += 1

        return _ORIGINAL_EARTH_EPHEMERIDES(
            time_to_treat
        )

    with _LOCK:
        item = _find_exact(times)

        if item is not None:
            _STATS["vector_hits"] += 1

            # Independent arrays: downstream pyLIMA cannot mutate
            # the cache itself.
            return (
                item["positions"].copy(),
                item["speeds"].copy(),
            )

    _STATS["vector_misses"] += 1

    positions, speeds = (
        _ORIGINAL_EARTH_EPHEMERIDES(
            time_to_treat
        )
    )

    with _LOCK:
        _store(
            times,
            positions,
            speeds,
        )

    return positions, speeds


def install_earth_ephemerides_cache(
    preload_path=None,
):
    """Install the exact cache once in this Python process."""

    global _INSTALLED

    with _LOCK:
        if _INSTALLED:
            return

        preload_cache(
            preload_path
        )

        parallax.Earth_ephemerides = (
            cached_earth_ephemerides
        )

        _INSTALLED = True


def earth_ephemerides_cache_stats():
    """Return a copy of cache diagnostics."""

    with _LOCK:
        return {
            **_STATS,
            "cached_vectors": len(_CACHE),
            "installed": bool(_INSTALLED),
        }
