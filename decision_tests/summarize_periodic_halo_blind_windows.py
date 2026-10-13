#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from astropy.time import Time


INPUT = Path(
    "decision_tests/horizons_periodic_halo_validation.csv"
)

WINDOWS = [
    (
        "S3-like Feb-Apr 2028",
        "2028-02-11T00:00:00",
        "2028-04-24T23:59:59",
    ),
    (
        "Aug-Oct proxy 2028",
        "2028-08-16T00:00:00",
        "2028-10-03T23:59:59",
    ),
]


def header(title):
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def summarize(label, x):

    x = np.asarray(
        x,
        dtype=float,
    )

    print(
        f"{label:42s}"
        f" min={np.nanmin(x):.8g}"
        f" median={np.nanmedian(x):.8g}"
        f" p90={np.nanquantile(x, 0.90):.8g}"
        f" max={np.nanmax(x):.8g}"
    )


def jd_from_iso(value):

    return float(
        Time(
            value,
            format="isot",
            scale="utc",
        ).jd
    )


def main():

    header("INPUT")

    if not INPUT.exists():
        raise FileNotFoundError(
            INPUT
        )

    df = pd.read_csv(
        INPUT
    )

    required = {
        "jd",
        "relative_projected_vector_error",
        "Dperp_fractional_error",
        "projected_direction_error_deg",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing columns: "
            + repr(
                sorted(missing)
            )
        )

    print(
        "rows =",
        len(df),
    )

    if "best_period_days" in df.columns:

        periods = df[
            "best_period_days"
        ].dropna().unique()

        print(
            "best_period_days =",
            periods,
        )

    for name, start, stop in WINDOWS:

        header(name)

        start_jd = jd_from_iso(
            start
        )

        stop_jd = jd_from_iso(
            stop
        )

        mask = (
            (df["jd"] >= start_jd)
            &
            (df["jd"] <= stop_jd)
        )

        part = df.loc[
            mask
        ].copy()

        print(
            "start =",
            start,
        )

        print(
            "stop  =",
            stop,
        )

        print(
            "rows  =",
            len(part),
        )

        if len(part) == 0:
            print(
                "NO DATA"
            )
            continue

        summarize(
            "relative projected-vector error",
            part[
                "relative_projected_vector_error"
            ],
        )

        summarize(
            "|Dperp fractional error|",
            np.abs(
                part[
                    "Dperp_fractional_error"
                ]
            ),
        )

        summarize(
            "projected direction error [deg]",
            part[
                "projected_direction_error_deg"
            ],
        )

    header("LATE-2028 TREND")

    blind = df[
        df["jd"]
        >= jd_from_iso(
            "2028-01-01T00:00:00"
        )
    ].copy()

    blind[
        "days_since_2028"
    ] = (
        blind["jd"]
        - jd_from_iso(
            "2028-01-01T00:00:00"
        )
    )

    for column in (
        "relative_projected_vector_error",
        "Dperp_fractional_error",
        "projected_direction_error_deg",
    ):

        y = blind[
            column
        ].to_numpy(
            dtype=float
        )

        if column == "Dperp_fractional_error":
            y = np.abs(y)

        x = blind[
            "days_since_2028"
        ].to_numpy(
            dtype=float
        )

        finite = (
            np.isfinite(x)
            &
            np.isfinite(y)
        )

        slope = np.polyfit(
            x[finite],
            y[finite],
            1,
        )[0]

        print(
            f"{column:42s}"
            f" slope_per_day={slope:.8g}"
            f" slope_per_year={365.25*slope:.8g}"
        )

    print()
    print(
        "No fitting was performed."
    )

    print(
        "No ephemeris or production file was modified."
    )


if __name__ == "__main__":
    main()
