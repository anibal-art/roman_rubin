"""Roman GBTDS time sampling.

Season structure from the GBTDS definition (Roman Observations Time Allocation
Committee report, arXiv:2505.10574, and roman-docs.stsci.edu, "Galactic Bulge
Time-Domain Survey"): the five-year mission has 10 bulge seasons of ~72 days
(spring ~Feb-Apr, autumn ~Aug-Oct); the first three and the last three are
high-cadence seasons (F146 every 12.1 min, 70.5 days of monitoring, the first
one shortened by 2 days) and the middle four are low-cadence seasons (a
~1.5 h microlensing unit, i.e. five F146 epochs 12.1 min apart, every 5 days,
over the whole season). Calendar dates are not part of the definition: the
first season is the first bulge window after commissioning. Here it is spring
2027. The cached template is built by
set_telescopes_pyLIMA._load_or_build_roman_template under a name that
includes ROMAN_SCHEDULE_VERSION, so a schedule change never reuses an old
template.

History: until 2026-11-17 the template had 6 high-cadence seasons and only 3
low-cadence seasons sampled every 72 h (the spring 2030 low-cadence season was
missing and the last high-cadence season was spring 2031 instead of autumn
2031).
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


ROMAN_SCHEDULE_VERSION = "gbtds_rotac2025_v1"

# Bulge visibility windows (~72 d). High-cadence monitoring lasts 70.5 d,
# centred in the window; the first season is shortened by 2 d at its start.
HIGH_CADENCE_DAYS = 0.084027777777778   # 12.1 min, F146 epoch spacing (days)
HIGH_SEASON_MONITORING_DAYS = 70.5
FIRST_SEASON_SHORTENING_DAYS = 2.0
LOW_CADENCE_PERIOD_DAYS = 5.0           # one microlensing unit every 5 days
LOW_CADENCE_EPOCHS_PER_UNIT = 5         # F146 epochs per unit, 12.1 min apart

_SPRING = ("02-11T00:00:00", "04-24T00:00:00")
_AUTUMN = ("08-16T00:00:00", "10-27T00:00:00")


def _window(year, half):
    a, b = _SPRING if half == "spring" else _AUTUMN
    return {"start": f"{year}-{a}", "end": f"{year}-{b}"}


def _roman_nominal_and_off_seasons():
    """
    Roman GBTDS seasons: (high-cadence seasons, low-cadence seasons), each a
    list of {"start", "end"} ISO dates. High-cadence entries are already
    trimmed to the 70.5 d of monitoring (68.5 d for the first one).
    """
    windows = [(2027, "spring"), (2027, "autumn"), (2028, "spring"),     # high 1-3
               (2028, "autumn"), (2029, "spring"), (2029, "autumn"),     # low 1-4
               (2030, "spring"),
               (2030, "autumn"), (2031, "spring"), (2031, "autumn")]     # high 4-6
    nominal_seasons, off_seasons = [], []
    for k, (year, half) in enumerate(windows):
        w = _window(year, half)
        if 3 <= k <= 6:
            off_seasons.append(w)
            continue
        t_a = Time(w["start"], format="isot").jd
        t_b = Time(w["end"], format="isot").jd
        mid = 0.5 * (t_a + t_b)
        start = mid - 0.5 * HIGH_SEASON_MONITORING_DAYS
        end = mid + 0.5 * HIGH_SEASON_MONITORING_DAYS
        if k == 0:
            start += FIRST_SEASON_SHORTENING_DAYS
        nominal_seasons.append({"start": Time(start, format="jd").isot,
                                "end": Time(end, format="jd").isot})
    return nominal_seasons, off_seasons


def _simulate_roman_low_cadence_season(season):
    """
    Low-cadence season: LOW_CADENCE_EPOCHS_PER_UNIT F146 epochs 12.1 min
    apart, repeated every LOW_CADENCE_PERIOD_DAYS over the whole window.
    """
    lc = _simulate_roman_season(season, sampling=HIGH_CADENCE_DAYS * 24.0)
    t = np.asarray(lc["time"].value, dtype=float)
    phase = (t - t[0]) % LOW_CADENCE_PERIOD_DAYS
    # the first epoch of a unit has phase in [0, 12.1 min): exactly
    # LOW_CADENCE_EPOCHS_PER_UNIT epochs have phase < that many spacings
    keep = phase < LOW_CADENCE_EPOCHS_PER_UNIT * HIGH_CADENCE_DAYS
    return lc[keep]


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
