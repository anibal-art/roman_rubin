#!/usr/bin/env python3

import numpy as np

from astropy.time import Time
from astroquery.jplhorizons import Horizons
from pyLIMA.parallax import JPL_ephemerides


ROMAN_ID = "-211"
GEOCENTER = "500@399"

TEST_DATES = [
    # Control cercano a la época actual
    "2026-09-15T00:00:00",
    "2026-10-10T00:00:00",

    # Fechas de nuestra campaña
    "2027-02-11T00:00:00",
    "2027-04-24T00:00:00",
    "2027-08-16T00:00:00",
    "2028-02-11T00:00:00",
    "2030-02-11T00:00:00",
    "2031-04-24T00:00:00",
]


def header(title):
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def test_pylima_mapping():
    header("PYLIMA WRAPPER MAPPING")

    for func_name in (
        "horizons_obscodes",
        "horizons_obstimes",
    ):
        func = getattr(
            JPL_ephemerides,
            func_name,
            None,
        )

        print()
        print(func_name)

        if func is None:
            print("  NOT PRESENT")
            continue

        try:
            value = func(ROMAN_ID)

            print(
                "  value =",
                repr(value),
            )

            print(
                "  type  =",
                type(value).__name__,
            )

        except Exception as error:
            print(
                "  ERROR =",
                type(error).__name__,
                repr(error),
            )


def query_one_date(iso):
    jd = float(
        Time(
            iso,
            format="isot",
            scale="utc",
        ).jd
    )

    print()
    print("-" * 92)
    print("DATE =", iso)
    print("JD   =", repr(jd))
    print("-" * 92)

    try:
        obj = Horizons(
            id=ROMAN_ID,
            id_type=None,
            location=GEOCENTER,
            epochs=jd,
        )

        eph = obj.ephemerides()

    except Exception as error:

        print(
            "STATUS = FAILED"
        )

        print(
            "TYPE   =",
            type(error).__name__,
        )

        print(
            "ERROR  =",
            repr(error),
        )

        return {
            "iso": iso,
            "jd": jd,
            "status": "FAILED",
        }

    print(
        "STATUS = SUCCESS"
    )

    print(
        "rows   =",
        len(eph),
    )

    print(
        "columns =",
        list(eph.colnames),
    )

    if len(eph) == 0:
        return {
            "iso": iso,
            "jd": jd,
            "status": "EMPTY",
        }

    row = eph[0]

    wanted = [
        "targetname",
        "datetime_str",
        "datetime_jd",
        "RA",
        "DEC",
        "delta",
    ]

    print()

    for key in wanted:

        if key in eph.colnames:
            print(
                f"{key:14s} =",
                row[key],
            )

    result = {
        "iso": iso,
        "jd": jd,
        "status": "SUCCESS",
    }

    for key in (
        "datetime_jd",
        "RA",
        "DEC",
        "delta",
    ):

        if key in eph.colnames:

            try:
                result[key] = float(
                    row[key]
                )
            except Exception:
                pass

    return result


def main():

    header("QUERY CONFIGURATION")

    print(
        "Roman Horizons ID =",
        ROMAN_ID,
    )

    print(
        "observer center    =",
        GEOCENTER,
    )

    print(
        "meaning            = Earth geocenter"
    )

    test_pylima_mapping()

    header("DIRECT ASTROQUERY TEST")

    results = []

    for iso in TEST_DATES:
        results.append(
            query_one_date(iso)
        )

    header("SUMMARY")

    for result in results:

        print(
            f"{result['iso']}  "
            f"{result['status']}"
        )

    successes = [
        r
        for r in results
        if r["status"] == "SUCCESS"
    ]

    failures = [
        r
        for r in results
        if r["status"] != "SUCCESS"
    ]

    print()
    print(
        "N success =",
        len(successes),
    )

    print(
        "N failed  =",
        len(failures),
    )

    if successes:

        distances = [
            r["delta"]
            for r in successes
            if "delta" in r
        ]

        if distances:

            print()
            print(
                "successful Earth-Roman distances [AU]:"
            )

            print(
                "  min    =",
                repr(
                    float(
                        np.min(distances)
                    )
                ),
            )

            print(
                "  median =",
                repr(
                    float(
                        np.median(distances)
                    )
                ),
            )

            print(
                "  max    =",
                repr(
                    float(
                        np.max(distances)
                    )
                ),
            )

    header("DECISION")

    campaign_dates = results[2:]

    campaign_ok = all(
        r["status"] == "SUCCESS"
        for r in campaign_dates
    )

    print(
        "all campaign epochs available =",
        campaign_ok,
    )

    if campaign_ok:

        print()
        print(
            "RESULT: Horizons currently has Roman "
            "ephemerides at all tested campaign epochs."
        )

        print(
            "Next step: request the complete daily "
            "2027--2031 ephemeris."
        )

    else:

        print()
        print(
            "RESULT: Roman target -211 does NOT currently "
            "provide all epochs required by the simulation."
        )

        print(
            "Do not extrapolate the spacecraft trajectory."
        )

        print(
            "Next step would be to identify a documented "
            "future Roman/L2 trajectory source for the "
            "missing epochs."
        )


if __name__ == "__main__":
    main()
