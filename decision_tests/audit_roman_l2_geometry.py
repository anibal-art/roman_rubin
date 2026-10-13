#!/usr/bin/env python3

from pathlib import Path
import argparse
import warnings

import numpy as np

from astropy.time import Time
from astropy.coordinates import get_sun
from astropy.utils.exceptions import AstropyWarning


SEASONS = [
    ("S1", "2027-02-11", "2027-04-24"),
    ("S2", "2027-08-16", "2027-10-27"),
    ("S3", "2028-02-11", "2028-04-24"),
    ("S4", "2030-02-11", "2030-04-24"),
    ("S5", "2030-08-16", "2030-10-27"),
    ("S6", "2031-02-11", "2031-04-24"),
]


def header(title):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def unit_from_radec(ra_deg, dec_deg):
    ra = np.deg2rad(
        np.asarray(ra_deg, dtype=float)
    )
    dec = np.deg2rad(
        np.asarray(dec_deg, dtype=float)
    )

    cos_dec = np.cos(dec)

    return np.column_stack(
        [
            cos_dec * np.cos(ra),
            cos_dec * np.sin(ra),
            np.sin(dec),
        ]
    )


def normalize(v):
    v = np.asarray(v, dtype=float)

    norm = np.linalg.norm(
        v,
        axis=-1,
        keepdims=True,
    )

    return v / norm


def projected_norm(vectors, los):
    vectors = np.asarray(
        vectors,
        dtype=float,
    )

    los = np.asarray(
        los,
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


def angular_separation_deg(v1, v2):
    a = normalize(v1)
    b = normalize(v2)

    dots = np.sum(
        a * b,
        axis=1,
    )

    dots = np.clip(
        dots,
        -1.0,
        1.0,
    )

    return np.rad2deg(
        np.arccos(dots)
    )


def stats(label, values, unit=""):
    x = np.asarray(
        values,
        dtype=float,
    )

    finite = np.isfinite(x)

    if not finite.any():
        print(f"{label}: no finite values")
        return

    x = x[finite]

    suffix = (
        f" {unit}"
        if unit
        else ""
    )

    print(
        f"{label:28s} "
        f"min={np.min(x):.8f}{suffix}  "
        f"median={np.median(x):.8f}{suffix}  "
        f"max={np.max(x):.8f}{suffix}"
    )


def calculate_geometry(
    rows,
    source_ra_deg,
    source_dec_deg,
):
    jd = np.asarray(
        rows[:, 0],
        dtype=float,
    )

    sat_ra = np.asarray(
        rows[:, 1],
        dtype=float,
    )

    sat_dec = np.asarray(
        rows[:, 2],
        dtype=float,
    )

    distance = np.asarray(
        rows[:, 3],
        dtype=float,
    )

    # ----------------------------------------------------------
    # Earth -> Roman vector directly from the Horizons-like
    # ephemeris representation used by pyLIMA.
    # ----------------------------------------------------------

    sat_direction = unit_from_radec(
        sat_ra,
        sat_dec,
    )

    earth_to_roman = (
        distance[:, None]
        * sat_direction
    )

    # ----------------------------------------------------------
    # Source line of sight.
    # ----------------------------------------------------------

    source_los = unit_from_radec(
        [source_ra_deg],
        [source_dec_deg],
    )[0]

    d_perp = projected_norm(
        earth_to_roman,
        source_los,
    )

    # ----------------------------------------------------------
    # Geocentric Sun direction at the same epochs.
    #
    # L2 should be approximately in the ANTI-solar direction.
    # ----------------------------------------------------------

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=AstropyWarning,
        )

        sun = get_sun(
            Time(
                jd,
                format="jd",
                scale="utc",
            )
        )

    sun_ra = np.asarray(
        sun.ra.deg,
        dtype=float,
    )

    sun_dec = np.asarray(
        sun.dec.deg,
        dtype=float,
    )

    sun_direction = unit_from_radec(
        sun_ra,
        sun_dec,
    )

    antisolar_direction = (
        -sun_direction
    )

    angle_from_antisolar = (
        angular_separation_deg(
            sat_direction,
            antisolar_direction,
        )
    )

    # ----------------------------------------------------------
    # Counterfactual:
    # same Earth-Roman distance, but exactly anti-solar.
    # ----------------------------------------------------------

    ideal_l2_vector = (
        distance[:, None]
        * antisolar_direction
    )

    d_perp_antisolar = projected_norm(
        ideal_l2_vector,
        source_los,
    )

    ratio = np.full_like(
        d_perp,
        np.nan,
    )

    good = (
        d_perp_antisolar > 0
    )

    ratio[good] = (
        d_perp[good]
        / d_perp_antisolar[good]
    )

    return {
        "jd": jd,
        "distance": distance,
        "d_perp": d_perp,
        "angle_from_antisolar": angle_from_antisolar,
        "d_perp_antisolar": d_perp_antisolar,
        "d_perp_ratio": ratio,
    }


def season_bounds(start_iso, end_iso):
    start = Time(
        f"{start_iso}T00:00:00",
        scale="utc",
    ).jd

    end = Time(
        f"{end_iso}T23:59:59",
        scale="utc",
    ).jd

    return (
        float(start),
        float(end),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ephemerides",
        default="ephemerides/Roman_positions.npy",
        type=Path,
    )

    parser.add_argument(
        "--source-ra",
        type=float,
        default=267.92497054815516,
        help="Bulge/Roman field RA in deg.",
    )

    parser.add_argument(
        "--source-dec",
        type=float,
        default=-29.152232510353276,
        help="Bulge/Roman field Dec in deg.",
    )

    parser.add_argument(
        "--output",
        default=(
            "decision_tests/"
            "roman_l2_geometry_daily.csv"
        ),
        type=Path,
    )

    args = parser.parse_args()

    path = args.ephemerides.resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    eph = np.asarray(
        np.load(path),
        dtype=float,
    )

    if (
        eph.ndim != 2
        or eph.shape[1] != 4
    ):
        raise ValueError(
            "Expected ephemeris shape (N,4): "
            "[JD, RA_deg, Dec_deg, distance_AU]. "
            f"Got {eph.shape}."
        )

    if not np.all(
        np.diff(eph[:, 0]) > 0
    ):
        raise ValueError(
            "Ephemeris times are not strictly increasing."
        )

    header("INPUT")

    print("ephemerides =", path)
    print("shape       =", eph.shape)
    print(
        "source RA   =",
        args.source_ra,
        "deg",
    )
    print(
        "source Dec  =",
        args.source_dec,
        "deg",
    )

    print()
    print(
        "ephemeris first JD =",
        repr(float(eph[0, 0])),
        Time(
            eph[0, 0],
            format="jd",
        ).iso,
    )

    print(
        "ephemeris last JD  =",
        repr(float(eph[-1, 0])),
        Time(
            eph[-1, 0],
            format="jd",
        ).iso,
    )

    all_geometry = calculate_geometry(
        eph,
        args.source_ra,
        args.source_dec,
    )

    header("FULL EPHEMERIS GEOMETRY")

    stats(
        "Earth-Roman distance",
        all_geometry["distance"],
        "AU",
    )

    stats(
        "D_perp actual",
        all_geometry["d_perp"],
        "AU",
    )

    stats(
        "angle from antisolar",
        all_geometry[
            "angle_from_antisolar"
        ],
        "deg",
    )

    stats(
        "D_perp if antisolar",
        all_geometry[
            "d_perp_antisolar"
        ],
        "AU",
    )

    stats(
        "actual / antisolar D_perp",
        all_geometry[
            "d_perp_ratio"
        ],
    )

    rows_out = []

    header("ROMAN NOMINAL SEASONS")

    for (
        name,
        start_iso,
        end_iso,
    ) in SEASONS:

        start_jd, end_jd = season_bounds(
            start_iso,
            end_iso,
        )

        season_duration = (
            end_jd - start_jd
        )

        overlap_start = max(
            start_jd,
            float(eph[0, 0]),
        )

        overlap_end = min(
            end_jd,
            float(eph[-1, 0]),
        )

        overlap_days = max(
            0.0,
            overlap_end - overlap_start,
        )

        coverage_fraction = (
            overlap_days
            / season_duration
            if season_duration > 0
            else np.nan
        )

        mask = (
            (eph[:, 0] >= start_jd)
            &
            (eph[:, 0] <= end_jd)
        )

        subset = eph[mask]

        print()
        print("-" * 88)
        print(
            f"{name}: {start_iso} -> {end_iso}"
        )
        print("-" * 88)

        print(
            "season duration [d] =",
            f"{season_duration:.3f}",
        )

        print(
            "ephemeris rows in season =",
            len(subset),
        )

        print(
            "temporal coverage fraction =",
            f"{coverage_fraction:.6f}",
        )

        print(
            "days outside ephemeris =",
            f"{season_duration - overlap_days:.3f}",
        )

        if len(subset) == 0:
            print(
                "NO EPHEMERIS ROWS INSIDE THIS SEASON"
            )
            continue

        geom = calculate_geometry(
            subset,
            args.source_ra,
            args.source_dec,
        )

        stats(
            "Earth-Roman distance",
            geom["distance"],
            "AU",
        )

        stats(
            "D_perp actual",
            geom["d_perp"],
            "AU",
        )

        stats(
            "angle from antisolar",
            geom[
                "angle_from_antisolar"
            ],
            "deg",
        )

        stats(
            "D_perp if antisolar",
            geom[
                "d_perp_antisolar"
            ],
            "AU",
        )

        stats(
            "actual / antisolar D_perp",
            geom[
                "d_perp_ratio"
            ],
        )

        for i in range(
            len(subset)
        ):
            rows_out.append(
                [
                    name,
                    geom["jd"][i],
                    geom["distance"][i],
                    geom["d_perp"][i],
                    geom[
                        "angle_from_antisolar"
                    ][i],
                    geom[
                        "d_perp_antisolar"
                    ][i],
                    geom[
                        "d_perp_ratio"
                    ][i],
                ]
            )

    header("SAVE")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = (
        "season,jd,distance_au,"
        "d_perp_actual_au,"
        "angle_from_antisolar_deg,"
        "d_perp_antisolar_au,"
        "d_perp_ratio"
    )

    if rows_out:
        arr = np.asarray(
            rows_out,
            dtype=object,
        )

        with args.output.open(
            "w"
        ) as f:
            f.write(columns + "\n")

            for row in arr:
                f.write(
                    ",".join(
                        str(x)
                        for x in row
                    )
                    + "\n"
                )

        print(
            "written =",
            args.output.resolve(),
        )
        print(
            "rows    =",
            len(rows_out),
        )

    else:
        print(
            "No rows to write."
        )

    print()
    print(
        "IMPORTANT: this test evaluates the geometry encoded "
        "by the current ephemeris. It does not modify it."
    )


if __name__ == "__main__":
    main()
