"""Shared photometric constants.

Leaf module: no imports from ``functions_roman_rubin``, ``simulation``,
``catalog``, or any other simulation/catalog code, so it can be imported
from anywhere without creating a cycle.
"""

SIMULATION_BAND_ZERO_POINTS = {
    "W149": 27.615,
    "u": 27.03,
    "g": 28.38,
    "r": 28.16,
    "i": 27.85,
    "z": 27.46,
    "y": 26.68,
}

PYLIMA_FIT_ZERO_POINT = 27.4
