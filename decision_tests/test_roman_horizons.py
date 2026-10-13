#!/usr/bin/env python3

import inspect
import numpy as np

from astropy.time import Time
from pyLIMA.parallax import JPL_ephemerides


ROMAN_ID = "-211"

TEST_DATES = [
    "2027-02-11T00:00:00",
    "2027-04-24T00:00:00",
    "2027-08-16T00:00:00",
    "2028-02-11T00:00:00",
    "2030-02-11T00:00:00",
    "2031-04-24T00:00:00",
]


def header(title):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def main():

    header("PYLIMA JPL HELPER")

    print(
        "source file =",
        inspect.getsourcefile(
            JPL_ephemerides
        ),
    )

    print()
    print(
        "horizons_API signature =",
        inspect.signature(
            JPL_ephemerides.horizons_API
        ),
    )

    print()
    print(
        inspect.getsource(
            JPL_ephemerides.horizons_API
        )
    )

    header("TEST EPOCHS")

    times = Time(
        TEST_DATES,
        format="isot",
        scale="utc",
    )

    jd = np.asarray(
        times.jd,
        dtype=float,
    )

    for iso, value in zip(
        TEST_DATES,
        jd,
    ):
        print(
            f"{iso}  JD={value:.9f}"
        )

    header("QUERY JPL HORIZONS")

    print("target      =", ROMAN_ID)
    print("observatory = Geocentric")

    try:
        result = (
            JPL_ephemerides.horizons_API(
                ROMAN_ID,
                jd,
                observatory="Geocentric",
            )
        )

    except Exception as error:

        print()
        print("QUERY FAILED")
        print(
            "type =",
            type(error).__name__,
        )
        print(
            "error =",
            repr(error),
        )

        return

    header("RAW RETURN")

    print(
        "return type =",
        type(result),
    )

    print(
        "repr =",
        repr(result),
    )

    # pyLIMA typically returns:
    # (target_name, ephemeris)
    if (
        isinstance(result, tuple)
        and len(result) == 2
    ):

        target_name, eph = result

        print()
        print(
            "target_name =",
            target_name,
        )

        eph = np.asarray(
            eph,
            dtype=float,
        )

    else:
        eph = np.asarray(
            result,
            dtype=float,
        )

    header("PARSED EPHEMERIS")

    print(
        "shape =",
        eph.shape,
    )

    print(eph)

    if (
        eph.ndim != 2
        or eph.shape[1] != 4
    ):
        print()
        print(
            "Unexpected shape. "
            "Do not proceed to production ephemeris."
        )
        return

    print()
    print(
        "Expected columns from pyLIMA:"
    )
    print(
        "[JD, RA_deg, Dec_deg, distance_AU]"
    )

    print()
    print("distance statistics:")

    distance = eph[:, 3]

    print(
        "  min    =",
        repr(float(np.min(distance))),
        "AU",
    )
    print(
        "  median =",
        repr(float(np.median(distance))),
        "AU",
    )
    print(
        "  max    =",
        repr(float(np.max(distance))),
        "AU",
    )

    header("BASIC SANITY")

    finite = bool(
        np.all(
            np.isfinite(eph)
        )
    )

    print(
        "all finite =",
        finite,
    )

    plausible_distance = bool(
        np.all(
            (distance > 0.005)
            &
            (distance < 0.02)
        )
    )

    print(
        "distance 0.005--0.02 AU =",
        plausible_distance,
    )

    if finite and plausible_distance:
        print()
        print(
            "HORIZONS QUERY LOOKS USABLE."
        )
        print(
            "Next step: generate the full Roman "
            "2027--2031 ephemeris and validate its "
            "L2 geometry before replacing any file."
        )


if __name__ == "__main__":
    main()
