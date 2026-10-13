#!/usr/bin/env python3

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from astropy.time import Time
from astropy.coordinates import get_sun
from astropy.utils.exceptions import AstropyWarning


CANDIDATE = Path(
    "decision_tests/data/"
    "Roman_positions_periodic_candidate.npy"
)

HORIZONS = Path(
    "decision_tests/oem_phase_fit_validation.csv"
)

SOURCE_RA_DEG = 267.92497054815516
SOURCE_DEC_DEG = -29.152232510353276

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
    print("=" * 96)
    print(title)
    print("=" * 96)


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


def source_basis(ra_deg, dec_deg):

    ra = np.deg2rad(float(ra_deg))
    dec = np.deg2rad(float(dec_deg))

    north = np.array(
        [
            -np.sin(dec) * np.cos(ra),
            -np.sin(dec) * np.sin(ra),
            np.cos(dec),
        ]
    )

    east = np.array(
        [
            -np.sin(ra),
            np.cos(ra),
            0.0,
        ]
    )

    return north, east


def project_NE(vectors, north, east):

    vectors = np.asarray(
        vectors,
        dtype=float,
    )

    return np.column_stack(
        [
            vectors @ north,
            vectors @ east,
        ]
    )


def angle_3d_deg(a, b):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = (
        a
        / np.linalg.norm(
            a,
            axis=1,
            keepdims=True,
        )
    )

    b = (
        b
        / np.linalg.norm(
            b,
            axis=1,
            keepdims=True,
        )
    )

    cosine = np.sum(
        a * b,
        axis=1,
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return np.rad2deg(
        np.arccos(cosine)
    )


def angle_2d_deg(a, b):

    dot = np.sum(
        a * b,
        axis=1,
    )

    cross = (
        a[:, 0] * b[:, 1]
        - a[:, 1] * b[:, 0]
    )

    return np.rad2deg(
        np.arctan2(
            np.abs(cross),
            dot,
        )
    )


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


def jd_start(date):

    return float(
        Time(
            date + "T00:00:00",
            format="isot",
            scale="utc",
        ).jd
    )


def jd_stop(date):

    return float(
        Time(
            date + "T23:59:59",
            format="isot",
            scale="utc",
        ).jd
    )


def main():

    # ============================================================
    # Load candidate.
    # ============================================================

    header("CANDIDATE")

    if not CANDIDATE.exists():
        raise FileNotFoundError(
            CANDIDATE
        )

    candidate = np.load(
        CANDIDATE
    )

    if (
        candidate.ndim != 2
        or candidate.shape[1] != 4
    ):
        raise RuntimeError(
            "Candidate must have shape (N, 4)."
        )

    jd_c = candidate[:, 0]
    ra_c = candidate[:, 1]
    dec_c = candidate[:, 2]
    dist_c = candidate[:, 3]

    vector_c = (
        dist_c[:, None]
        * unit_from_radec(
            ra_c,
            dec_c,
        )
    )

    print(
        "shape =",
        candidate.shape,
    )

    print(
        "first JD =",
        repr(float(jd_c[0])),
    )

    print(
        "last JD  =",
        repr(float(jd_c[-1])),
    )

    print(
        "cadence min/median/max =",
        float(np.min(np.diff(jd_c))),
        float(np.median(np.diff(jd_c))),
        float(np.max(np.diff(jd_c))),
    )

    print(
        "all finite =",
        bool(np.all(np.isfinite(candidate))),
    )

    summarize(
        "Earth-Roman distance [AU]",
        dist_c,
    )

    # Cartesian interpolators for comparison only.
    interp_xyz = [
        interp1d(
            jd_c,
            vector_c[:, axis],
            kind="linear",
            bounds_error=True,
        )
        for axis in range(3)
    ]

    # ============================================================
    # Compare with every stored Horizons epoch.
    # ============================================================

    header("CANDIDATE vs ALL AVAILABLE HORIZONS")

    if not HORIZONS.exists():
        raise FileNotFoundError(
            HORIZONS
        )

    h = pd.read_csv(
        HORIZONS
    )

    required = {
        "jd",
        "ra_deg",
        "dec_deg",
        "distance_au",
    }

    missing = required - set(h.columns)

    if missing:
        raise RuntimeError(
            "Missing Horizons columns: "
            + repr(sorted(missing))
        )

    h = (
        h.sort_values("jd")
        .drop_duplicates(
            subset="jd",
            keep="first",
        )
        .reset_index(drop=True)
    )

    jd_h = h["jd"].to_numpy(
        dtype=float
    )

    h_vector = (
        h[
            "distance_au"
        ].to_numpy(
            dtype=float
        )[:, None]
        * unit_from_radec(
            h["ra_deg"],
            h["dec_deg"],
        )
    )

    pred_vector = np.column_stack(
        [
            f(jd_h)
            for f in interp_xyz
        ]
    )

    position_error = np.linalg.norm(
        pred_vector - h_vector,
        axis=1,
    )

    position_relative = (
        position_error
        / np.linalg.norm(
            h_vector,
            axis=1,
        )
    )

    direction_error = angle_3d_deg(
        pred_vector,
        h_vector,
    )

    north, east = source_basis(
        SOURCE_RA_DEG,
        SOURCE_DEC_DEG,
    )

    true_NE = project_NE(
        h_vector,
        north,
        east,
    )

    pred_NE = project_NE(
        pred_vector,
        north,
        east,
    )

    true_Dperp = np.linalg.norm(
        true_NE,
        axis=1,
    )

    pred_Dperp = np.linalg.norm(
        pred_NE,
        axis=1,
    )

    projected_vector_error = (
        np.linalg.norm(
            pred_NE - true_NE,
            axis=1,
        )
        / true_Dperp
    )

    Dperp_fractional_error = (
        pred_Dperp
        / true_Dperp
        - 1.0
    )

    projected_direction_error = (
        angle_2d_deg(
            pred_NE,
            true_NE,
        )
    )

    print(
        "Horizons rows =",
        len(h),
    )

    print(
        "Horizons first JD =",
        float(jd_h.min()),
    )

    print(
        "Horizons last JD  =",
        float(jd_h.max()),
    )

    print()

    summarize(
        "3D position error [AU]",
        position_error,
    )

    summarize(
        "relative 3D position error",
        position_relative,
    )

    summarize(
        "3D direction error [deg]",
        direction_error,
    )

    summarize(
        "relative projected-vector error",
        projected_vector_error,
    )

    summarize(
        "|Dperp fractional error|",
        np.abs(
            Dperp_fractional_error
        ),
    )

    summarize(
        "projected direction error [deg]",
        projected_direction_error,
    )

    # ============================================================
    # Candidate geometry versus anti-solar direction.
    # ============================================================

    header("FULL CANDIDATE GEOMETRY")

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            category=AstropyWarning,
        )

        sun = get_sun(
            Time(
                jd_c,
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

    antisolar_angle = angle_3d_deg(
        vector_c,
        antisolar_direction,
    )

    candidate_NE = project_NE(
        vector_c,
        north,
        east,
    )

    candidate_Dperp = np.linalg.norm(
        candidate_NE,
        axis=1,
    )

    summarize(
        "Earth-Roman distance [AU]",
        dist_c,
    )

    summarize(
        "angle from antisolar [deg]",
        antisolar_angle,
    )

    summarize(
        "D_perp toward source [AU]",
        candidate_Dperp,
    )

    # ============================================================
    # Six configured Roman seasons.
    # ============================================================

    header("CONFIGURED ROMAN SEASONS")

    season_rows = []

    for name, start, stop in SEASONS:

        start_jd = jd_start(start)
        stop_jd = jd_stop(stop)

        supported = bool(
            start_jd >= jd_c.min()
            and stop_jd <= jd_c.max()
        )

        mask = (
            (jd_c >= start_jd)
            &
            (jd_c <= stop_jd)
        )

        print()
        print(
            name,
            start,
            "->",
            stop,
        )

        print(
            "supported =",
            supported,
        )

        print(
            "rows =",
            int(mask.sum()),
        )

        if mask.sum() == 0:
            continue

        summarize(
            "distance [AU]",
            dist_c[mask],
        )

        summarize(
            "angle from antisolar [deg]",
            antisolar_angle[mask],
        )

        summarize(
            "D_perp [AU]",
            candidate_Dperp[mask],
        )

        season_rows.append(
            {
                "season": name,
                "start": start,
                "stop": stop,
                "supported":
                    supported,
                "n_rows":
                    int(mask.sum()),
                "distance_median_au":
                    float(
                        np.median(
                            dist_c[mask]
                        )
                    ),
                "antisolar_angle_median_deg":
                    float(
                        np.median(
                            antisolar_angle[
                                mask
                            ]
                        )
                    ),
                "dperp_median_au":
                    float(
                        np.median(
                            candidate_Dperp[
                                mask
                            ]
                        )
                    ),
            }
        )

    # ============================================================
    # Save diagnostics.
    # ============================================================

    comparison = h.copy()

    comparison[
        "candidate_position_error_au"
    ] = position_error

    comparison[
        "candidate_relative_position_error"
    ] = position_relative

    comparison[
        "candidate_direction_error_deg"
    ] = direction_error

    comparison[
        "candidate_relative_projected_vector_error"
    ] = projected_vector_error

    comparison[
        "candidate_Dperp_fractional_error"
    ] = Dperp_fractional_error

    comparison[
        "candidate_projected_direction_error_deg"
    ] = projected_direction_error

    comparison_path = Path(
        "decision_tests/"
        "roman_periodic_candidate_vs_horizons.csv"
    )

    seasons_path = Path(
        "decision_tests/"
        "roman_periodic_candidate_seasons.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    pd.DataFrame(
        season_rows
    ).to_csv(
        seasons_path,
        index=False,
    )

    header("FILES")

    print(
        "comparison =",
        comparison_path.resolve(),
    )

    print(
        "seasons    =",
        seasons_path.resolve(),
    )

    header("FINAL CHECKS")

    print(
        "all six seasons supported =",
        bool(
            len(season_rows) == 6
            and all(
                row["supported"]
                for row in season_rows
            )
        ),
    )

    print(
        "candidate all finite =",
        bool(
            np.all(
                np.isfinite(candidate)
            )
        ),
    )

    print()
    print(
        "No fitting was performed."
    )

    print(
        "No file under ephemerides/ was modified."
    )

    print(
        "No production config was modified."
    )


if __name__ == "__main__":
    main()
