#!/usr/bin/env python3

import warnings

import numpy as np
import pandas as pd

from astropy.time import Time
from astropy.coordinates import get_sun
from astropy.utils.exceptions import AstropyWarning
from astroquery.jplhorizons import Horizons


ROMAN_ID = "-211"
GEOCENTER = "500@399"

SOURCE_RA_DEG = 267.92497054815516
SOURCE_DEC_DEG = -29.152232510353276

IDEAL_L2_DISTANCE_AU = 0.01

SEASONS = [
    ("S1", "2027-02-11", "2027-04-24"),
    ("S2", "2027-08-16", "2027-10-27"),
    ("S3", "2028-02-11", "2028-04-24"),
]


def header(title):
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def unit_from_radec(ra_deg, dec_deg):

    ra = np.deg2rad(
        np.asarray(ra_deg, dtype=float)
    )

    dec = np.deg2rad(
        np.asarray(dec_deg, dtype=float)
    )

    return np.column_stack(
        [
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        ]
    )


def normalize(v):

    v = np.asarray(v, dtype=float)

    norm = np.linalg.norm(
        v,
        axis=1,
        keepdims=True,
    )

    return v / norm


def projected_norm(vectors, line_of_sight):

    vectors = np.asarray(
        vectors,
        dtype=float,
    )

    los = np.asarray(
        line_of_sight,
        dtype=float,
    )

    parallel = (
        vectors @ los
    )[:, None] * los[None, :]

    perpendicular = (
        vectors - parallel
    )

    return np.linalg.norm(
        perpendicular,
        axis=1,
    )


def angle_deg(a, b):

    a = normalize(a)
    b = normalize(b)

    dot = np.sum(
        a * b,
        axis=1,
    )

    dot = np.clip(
        dot,
        -1.0,
        1.0,
    )

    return np.rad2deg(
        np.arccos(dot)
    )


def daily_jd(start, stop):

    t0 = Time(
        start + "T00:00:00",
        format="isot",
        scale="utc",
    )

    t1 = Time(
        stop + "T00:00:00",
        format="isot",
        scale="utc",
    )

    return np.arange(
        float(t0.jd),
        float(t1.jd) + 0.1,
        1.0,
    )


def query_roman(jd):

    pieces = []

    # Keep requests small and predictable.
    for start in range(
        0,
        len(jd),
        50,
    ):

        chunk = jd[
            start:start + 50
        ]

        obj = Horizons(
            id=ROMAN_ID,
            id_type=None,
            location=GEOCENTER,
            epochs=chunk,
        )

        eph = obj.ephemerides()

        pieces.append(
            pd.DataFrame(
                {
                    "jd": np.asarray(
                        eph["datetime_jd"],
                        dtype=float,
                    ),
                    "ra_deg": np.asarray(
                        eph["RA"],
                        dtype=float,
                    ),
                    "dec_deg": np.asarray(
                        eph["DEC"],
                        dtype=float,
                    ),
                    "distance_au": np.asarray(
                        eph["delta"],
                        dtype=float,
                    ),
                }
            )
        )

    return pd.concat(
        pieces,
        ignore_index=True,
    )


def summarize(label, x):

    x = np.asarray(
        x,
        dtype=float,
    )

    print(
        f"{label:34s}"
        f" min={np.nanmin(x):.8f}"
        f" median={np.nanmedian(x):.8f}"
        f" max={np.nanmax(x):.8f}"
    )


def main():

    source_los = unit_from_radec(
        [SOURCE_RA_DEG],
        [SOURCE_DEC_DEG],
    )[0]

    all_results = []

    for (
        season,
        start,
        stop,
    ) in SEASONS:

        header(
            f"{season}: {start} -> {stop}"
        )

        jd = daily_jd(
            start,
            stop,
        )

        print(
            "N requested epochs =",
            len(jd),
        )

        roman = query_roman(
            jd
        )

        if len(roman) != len(jd):
            raise RuntimeError(
                f"{season}: Horizons returned "
                f"{len(roman)} rows for "
                f"{len(jd)} requested epochs."
            )

        # --------------------------------------------------------
        # Actual Roman geocentric vector from Horizons.
        # --------------------------------------------------------

        roman_direction = unit_from_radec(
            roman["ra_deg"],
            roman["dec_deg"],
        )

        roman_vector = (
            roman[
                "distance_au"
            ].to_numpy()[:, None]
            * roman_direction
        )

        # --------------------------------------------------------
        # Antisolar direction.
        # --------------------------------------------------------

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore",
                category=AstropyWarning,
            )

            sun = get_sun(
                Time(
                    roman["jd"].to_numpy(),
                    format="jd",
                    scale="utc",
                )
            )

        sun_direction = unit_from_radec(
            sun.ra.deg,
            sun.dec.deg,
        )

        antisolar_direction = (
            -sun_direction
        )

        # --------------------------------------------------------
        # Two ideal-L2 controls:
        #
        # A) same radial Earth-Roman distance as Horizons,
        #    but exactly anti-solar.
        #
        # B) fixed 0.01 AU and exactly anti-solar.
        # --------------------------------------------------------

        ideal_same_r = (
            roman[
                "distance_au"
            ].to_numpy()[:, None]
            * antisolar_direction
        )

        ideal_fixed = (
            IDEAL_L2_DISTANCE_AU
            * antisolar_direction
        )

        dperp_roman = projected_norm(
            roman_vector,
            source_los,
        )

        dperp_same_r = projected_norm(
            ideal_same_r,
            source_los,
        )

        dperp_fixed = projected_norm(
            ideal_fixed,
            source_los,
        )

        halo_angle = angle_deg(
            roman_direction,
            antisolar_direction,
        )

        frac_same_r = (
            dperp_roman
            / dperp_same_r
            - 1.0
        )

        frac_fixed = (
            dperp_roman
            / dperp_fixed
            - 1.0
        )

        summarize(
            "Earth-Roman distance [AU]",
            roman["distance_au"],
        )

        summarize(
            "angle Roman vs antisolar [deg]",
            halo_angle,
        )

        summarize(
            "D_perp Horizons [AU]",
            dperp_roman,
        )

        summarize(
            "D_perp antisolar same-r [AU]",
            dperp_same_r,
        )

        summarize(
            "D_perp ideal 0.01 AU [AU]",
            dperp_fixed,
        )

        summarize(
            "frac diff vs same-r",
            frac_same_r,
        )

        summarize(
            "frac diff vs 0.01 AU",
            frac_fixed,
        )

        out = roman.copy()

        out["season"] = season
        out["halo_angle_deg"] = halo_angle
        out["dperp_horizons_au"] = dperp_roman
        out["dperp_antisolar_same_r_au"] = (
            dperp_same_r
        )
        out["dperp_ideal_0p01_au"] = (
            dperp_fixed
        )
        out["frac_diff_same_r"] = (
            frac_same_r
        )
        out["frac_diff_0p01"] = (
            frac_fixed
        )

        all_results.append(
            out
        )

    result = pd.concat(
        all_results,
        ignore_index=True,
    )

    header("ALL THREE SEASONS")

    summarize(
        "Earth-Roman distance [AU]",
        result["distance_au"],
    )

    summarize(
        "angle Roman vs antisolar [deg]",
        result["halo_angle_deg"],
    )

    summarize(
        "D_perp Horizons [AU]",
        result["dperp_horizons_au"],
    )

    summarize(
        "D_perp ideal 0.01 AU [AU]",
        result["dperp_ideal_0p01_au"],
    )

    summarize(
        "frac diff vs same-r",
        result["frac_diff_same_r"],
    )

    summarize(
        "frac diff vs 0.01 AU",
        result["frac_diff_0p01"],
    )

    output = (
        "decision_tests/"
        "horizons_vs_ideal_l2.csv"
    )

    result.to_csv(
        output,
        index=False,
    )

    print()
    print(
        "saved =",
        output,
    )

    header("DECISION NUMBERS")

    abs_same = np.abs(
        result[
            "frac_diff_same_r"
        ].to_numpy()
    )

    abs_fixed = np.abs(
        result[
            "frac_diff_0p01"
        ].to_numpy()
    )

    for label, x in (
        (
            "same radial distance",
            abs_same,
        ),
        (
            "fixed 0.01 AU",
            abs_fixed,
        ),
    ):

        print()
        print(label)

        print(
            "  median |fractional difference| =",
            float(
                np.median(x)
            ),
        )

        print(
            "  90th percentile             =",
            float(
                np.quantile(
                    x,
                    0.90,
                )
            ),
        )

        print(
            "  max                         =",
            float(
                np.max(x)
            ),
        )


if __name__ == "__main__":
    main()
