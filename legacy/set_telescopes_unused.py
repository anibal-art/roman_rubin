"""Functions of set_telescopes_pyLIMA.py that nothing calls (moved in step 7; not imported)."""


def get_telescope_template_cache_stats():
    """
    Process-local Telescope-template cache diagnostics.
    """
    out = dict(
        _TELESCOPE_TEMPLATE_CACHE_STATS
    )

    out["roman_cached_templates"] = len(
        _ROMAN_TELESCOPE_TEMPLATE_CACHE
    )

    out["rubin_cached_templates"] = len(
        _RUBIN_TELESCOPE_TEMPLATE_CACHE
    )

    out["roman_ephemerides_cached_files"] = len(
        _ROMAN_EPHEMERIDES_TEMPLATE_CACHE
    )

    return out


def reset_telescope_template_cache():
    """
    Clear only Telescope-template optimization caches.
    """
    _ROMAN_TELESCOPE_TEMPLATE_CACHE.clear()
    _RUBIN_TELESCOPE_TEMPLATE_CACHE.clear()
    _ROMAN_EPHEMERIDES_TEMPLATE_CACHE.clear()

    for key in (
        "roman_hits",
        "roman_misses",
        "rubin_hits",
        "rubin_misses",
    ):
        _TELESCOPE_TEMPLATE_CACHE_STATS[
            key
        ] = 0


def _opsim_info(opsim_db_path=None):
    path = _resolve_opsim_db_path(opsim_db_path)
    return {
        "opsim_db_path": str(path),
        "opsim_cache_tag": _opsim_cache_tag(path),
    }


def make_t0_window(data, window_days=None, window_tE=None):
    """
    Construye una ventana simétrica alrededor de t0.

    Ejemplos
    --------
    window_tE=5    -> (t0 - 5*tE, t0 + 5*tE)
    window_days=50 -> (t0 - 50, t0 + 50)
    """
    t0 = data["t0"]
    tE = data["tE"]

    if window_days is not None and window_tE is not None:
        raise ValueError("Usá window_days o window_tE, no ambos.")

    if window_days is not None:
        delta_t = float(window_days)

    elif window_tE is not None:
        delta_t = float(window_tE) * float(tE)

    else:
        raise ValueError("Tenés que pasar window_days o window_tE.")

    return t0 - delta_t, t0 + delta_t


def make_asymmetric_t0_window(data, before_days=None, after_days=None,
                              before_tE=None, after_tE=None):
    """
    Construye una ventana asimétrica alrededor de t0.

    Sirve para probar, por ejemplo:
    - solo datos antes del pico;
    - solo datos después del pico;
    - más baseline de un lado que del otro.
    """
    t0 = data["t0"]
    tE = data["tE"]

    if before_days is not None and before_tE is not None:
        raise ValueError("Usá before_days o before_tE, no ambos.")

    if after_days is not None and after_tE is not None:
        raise ValueError("Usá after_days o after_tE, no ambos.")

    if before_days is None and before_tE is None:
        before = 0.0
    elif before_days is not None:
        before = float(before_days)
    else:
        before = float(before_tE) * float(tE)

    if after_days is None and after_tE is None:
        after = 0.0
    elif after_days is not None:
        after = float(after_days)
    else:
        after = float(after_tE) * float(tE)

    return t0 - before, t0 + after
