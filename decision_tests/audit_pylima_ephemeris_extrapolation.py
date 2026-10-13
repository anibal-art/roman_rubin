#!/usr/bin/env python3

import inspect
import warnings

import numpy as np

from astropy.utils.exceptions import AstropyWarning
from pyLIMA import telescopes


EPHEMERIDES = "ephemerides/Roman_positions.npy"


def header(title):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def values(x):
    return np.asarray(
        x.value if hasattr(x, "value") else x,
        dtype=float,
    )


def main():

    # ============================================================
    # 1. Source code actually used by this pyLIMA installation
    # ============================================================

    header("PYLIMA VERSION / SOURCE")

    import pyLIMA

    print(
        "pyLIMA version =",
        getattr(pyLIMA, "__version__", "unknown"),
    )

    print(
        "Telescope source =",
        inspect.getsourcefile(
            telescopes.Telescope
        ),
    )

    header("Telescope.find_space_positions SOURCE")

    print(
        inspect.getsource(
            telescopes.Telescope.find_space_positions
        )
    )

    # ============================================================
    # 2. Ephemeris support
    # ============================================================

    eph = np.asarray(
        np.load(EPHEMERIDES),
        dtype=float,
    )

    tmin = float(eph[:, 0].min())
    tmax = float(eph[:, 0].max())

    header("EPHEMERIS SUPPORT")

    print("tmin =", repr(tmin))
    print("tmax =", repr(tmax))
    print("span =", repr(tmax - tmin), "days")

    # ============================================================
    # 3. Test epochs
    #
    # Includes:
    # - first Roman season well BEFORE ephemeris support
    # - one day before support
    # - exactly at support start
    # - inside support
    # - exactly at support end
    # - one day after support
    # ============================================================

    test_times = np.array(
        [
            2461448.5,   # ~2027-02-12, S1
            tmin - 30.0,
            tmin - 1.0,
            tmin,
            tmin + 1.0,
            0.5 * (tmin + tmax),
            tmax - 1.0,
            tmax,
            tmax + 1.0,
        ],
        dtype=float,
    )

    dummy_lc = np.column_stack(
        [
            test_times,
            np.full_like(test_times, 20.0),
            np.full_like(test_times, 0.01),
        ]
    )

    tel = telescopes.Telescope(
        name="Roman_test",
        camera_filter="W149",
        location="Space",
        lightcurve=dummy_lc,
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
    )

    tel.spacecraft_name = "L2"

    tel.spacecraft_positions = {
        "astrometry": [],
        "photometry": eph,
    }

    # ============================================================
    # 4. Run exactly the pyLIMA method
    # ============================================================

    header("RUN find_space_positions")

    with warnings.catch_warnings(record=True) as caught:

        warnings.simplefilter("always")

        try:
            tel.find_space_positions()
            status = "SUCCESS"

        except Exception as error:
            status = "ERROR"
            print("ERROR TYPE =", type(error).__name__)
            print("ERROR      =", repr(error))

    print("status =", status)

    if caught:
        print()
        print("WARNINGS:")

        for w in caught:
            print(
                type(w.message).__name__,
                ":",
                str(w.message),
            )

    # ============================================================
    # 5. Inspect resulting positions
    # ============================================================

    header("RESULTING SPACE POSITIONS")

    print(
        "spacecraft_positions keys =",
        list(tel.spacecraft_positions.keys()),
    )

    print(
        "telescope_positions keys =",
        list(tel.telescope_positions.keys()),
    )

    if "photometry" not in tel.telescope_positions:
        print(
            "No photometry entry was created in "
            "telescope_positions."
        )
        return

    positions = np.asarray(
        tel.telescope_positions["photometry"],
        dtype=float,
    )

    print(
        "positions shape =",
        positions.shape,
    )

    print()
    print(
        "time, relative_to_support, x, y, z, norm_AU"
    )

    for k, t in enumerate(test_times):

        if t < tmin:
            relation = "BEFORE"
        elif t > tmax:
            relation = "AFTER"
        else:
            relation = "INSIDE"

        if positions.ndim == 2:

            # pyLIMA may store either (N,3) or (3,N).
            if positions.shape == (
                len(test_times),
                3,
            ):
                p = positions[k]

            elif positions.shape == (
                3,
                len(test_times),
            ):
                p = positions[:, k]

            else:
                print(
                    "Unexpected position shape:",
                    positions.shape,
                )
                break

            norm = float(
                np.linalg.norm(p)
            )

            print(
                f"{t:.9f}, "
                f"{relation:6s}, "
                f"{p[0]:+.12e}, "
                f"{p[1]:+.12e}, "
                f"{p[2]:+.12e}, "
                f"{norm:.12e}"
            )

    # ============================================================
    # 6. Finite-value checks
    # ============================================================

    header("FINITE CHECK")

    print(
        "all finite =",
        bool(
            np.all(
                np.isfinite(positions)
            )
        ),
    )

    print()
    print(
        "If positions exist for BEFORE/AFTER epochs, "
        "pyLIMA is extrapolating or otherwise extending "
        "the supplied ephemeris beyond its nominal support."
    )


if __name__ == "__main__":
    main()
