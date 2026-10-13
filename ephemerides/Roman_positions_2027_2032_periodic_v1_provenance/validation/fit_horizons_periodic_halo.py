#!/usr/bin/env python3

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from astropy.time import Time
from astropy.coordinates import get_sun
from astropy.utils.exceptions import AstropyWarning


INPUT = Path(
    "decision_tests/oem_phase_fit_validation.csv"
)

SOURCE_RA_DEG = 267.92497054815516
SOURCE_DEC_DEG = -29.152232510353276

TRAIN_END = "2027-12-31T23:59:59"

P_MIN_DAYS = 140.0
P_MAX_DAYS = 220.0
P_STEP_DAYS = 0.05

EPSILON_DEG = 23.439291111

S3_START = "2028-02-11T00:00:00"
S3_END = "2028-04-24T23:59:59"


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


def normalize_rows(v):

    v = np.asarray(
        v,
        dtype=float,
    )

    return (
        v
        / np.linalg.norm(
            v,
            axis=1,
            keepdims=True,
        )
    )


def rotating_basis(jd):

    jd = np.asarray(
        jd,
        dtype=float,
    )

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

    sun_direction = unit_from_radec(
        sun.ra.deg,
        sun.dec.deg,
    )

    # +X = anti-solar.
    xhat = -sun_direction

    epsilon = np.deg2rad(
        EPSILON_DEG
    )

    ecliptic_north = np.array(
        [
            0.0,
            -np.sin(epsilon),
            np.cos(epsilon),
        ]
    )

    zref = np.repeat(
        ecliptic_north[None, :],
        len(jd),
        axis=0,
    )

    yhat = normalize_rows(
        np.cross(
            zref,
            xhat,
        )
    )

    zhat = normalize_rows(
        np.cross(
            xhat,
            yhat,
        )
    )

    return (
        xhat,
        yhat,
        zhat,
    )


def inertial_to_rotating(vectors, basis):

    xhat, yhat, zhat = basis

    return np.column_stack(
        [
            np.sum(
                vectors * xhat,
                axis=1,
            ),
            np.sum(
                vectors * yhat,
                axis=1,
            ),
            np.sum(
                vectors * zhat,
                axis=1,
            ),
        ]
    )


def rotating_to_inertial(vectors, basis):

    xhat, yhat, zhat = basis

    return (
        vectors[:, 0, None] * xhat
        + vectors[:, 1, None] * yhat
        + vectors[:, 2, None] * zhat
    )


def source_basis(ra_deg, dec_deg):

    ra = np.deg2rad(
        float(ra_deg)
    )

    dec = np.deg2rad(
        float(dec_deg)
    )

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

    return np.column_stack(
        [
            vectors @ north,
            vectors @ east,
        ]
    )


def design_matrix(t, period, t_ref):

    phase = (
        2.0
        * np.pi
        * (t - t_ref)
        / period
    )

    return np.column_stack(
        [
            np.ones_like(phase),
            np.cos(phase),
            np.sin(phase),
        ]
    )


def fit_model(
    t_train,
    xyz_train,
    period,
    t_ref,
):

    A = design_matrix(
        t_train,
        period,
        t_ref,
    )

    coefficients = np.empty(
        (3, 3),
        dtype=float,
    )

    for axis in range(3):

        coeff, _, _, _ = (
            np.linalg.lstsq(
                A,
                xyz_train[:, axis],
                rcond=None,
            )
        )

        coefficients[
            axis
        ] = coeff

    return coefficients


def predict_model(
    t,
    period,
    t_ref,
    coefficients,
):

    A = design_matrix(
        t,
        period,
        t_ref,
    )

    out = np.empty(
        (len(t), 3),
        dtype=float,
    )

    for axis in range(3):

        out[:, axis] = (
            A
            @ coefficients[axis]
        )

    return out


def summarize(label, x):

    x = np.asarray(
        x,
        dtype=float,
    )

    print(
        f"{label:40s}"
        f" min={np.nanmin(x):.8g}"
        f" median={np.nanmedian(x):.8g}"
        f" p90={np.nanquantile(x, 0.90):.8g}"
        f" max={np.nanmax(x):.8g}"
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
        "ra_deg",
        "dec_deg",
        "distance_au",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing columns: "
            + repr(sorted(missing))
        )

    df = (
        df.sort_values("jd")
        .drop_duplicates(
            subset="jd",
            keep="first",
        )
        .reset_index(drop=True)
    )

    jd = df[
        "jd"
    ].to_numpy(
        dtype=float
    )

    direction = unit_from_radec(
        df["ra_deg"],
        df["dec_deg"],
    )

    vector = (
        df[
            "distance_au"
        ].to_numpy(
            dtype=float
        )[:, None]
        * direction
    )

    basis = rotating_basis(
        jd
    )

    rotating = inertial_to_rotating(
        vector,
        basis,
    )

    train_end_jd = float(
        Time(
            TRAIN_END,
            format="isot",
            scale="utc",
        ).jd
    )

    train = (
        jd <= train_end_jd
    )

    blind = (
        jd > train_end_jd
    )

    s3_start_jd = float(
        Time(
            S3_START,
            format="isot",
            scale="utc",
        ).jd
    )

    s3_end_jd = float(
        Time(
            S3_END,
            format="isot",
            scale="utc",
        ).jd
    )

    s3 = (
        (jd >= s3_start_jd)
        &
        (jd <= s3_end_jd)
    )

    print(
        "rows total =",
        len(df),
    )

    print(
        "train 2027 =",
        int(train.sum()),
    )

    print(
        "blind 2028 =",
        int(blind.sum()),
    )

    print(
        "S3 blind   =",
        int(s3.sum()),
    )

    t_ref = float(
        np.mean(
            jd[train]
        )
    )

    # ============================================================
    # Search common quasi-halo period using 2027 ONLY.
    # ============================================================

    header("PERIOD SEARCH — TRAINING 2027 ONLY")

    period_grid = np.arange(
        P_MIN_DAYS,
        P_MAX_DAYS
        + 0.5 * P_STEP_DAYS,
        P_STEP_DAYS,
    )

    rows = []

    for period in period_grid:

        coeff = fit_model(
            jd[train],
            rotating[train],
            period,
            t_ref,
        )

        pred = predict_model(
            jd[train],
            period,
            t_ref,
            coeff,
        )

        residual = (
            pred
            - rotating[train]
        )

        rms = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        residual**2,
                        axis=1,
                    )
                )
            )
        )

        rows.append(
            {
                "period_days":
                    float(period),

                "train_rms_au":
                    rms,
            }
        )

    search = pd.DataFrame(
        rows
    )

    best = search.loc[
        search[
            "train_rms_au"
        ].idxmin()
    ]

    best_period = float(
        best["period_days"]
    )

    print(
        "best period [d] =",
        best_period,
    )

    print(
        "training RMS [AU] =",
        float(
            best[
                "train_rms_au"
            ]
        ),
    )

    print()
    print("Ten best periods:")

    print(
        search
        .sort_values(
            "train_rms_au"
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    # ============================================================
    # Final model still fitted ONLY to 2027.
    # ============================================================

    coeff = fit_model(
        jd[train],
        rotating[train],
        best_period,
        t_ref,
    )

    pred_rotating = predict_model(
        jd,
        best_period,
        t_ref,
        coeff,
    )

    pred_vector = rotating_to_inertial(
        pred_rotating,
        basis,
    )

    # ============================================================
    # Full 3D errors.
    # ============================================================

    position_error = np.linalg.norm(
        pred_vector
        - vector,
        axis=1,
    )

    radial_true = np.linalg.norm(
        vector,
        axis=1,
    )

    radial_pred = np.linalg.norm(
        pred_vector,
        axis=1,
    )

    radial_fractional_error = (
        radial_pred
        / radial_true
        - 1.0
    )

    # ============================================================
    # Scientific projected baseline.
    # ============================================================

    north, east = source_basis(
        SOURCE_RA_DEG,
        SOURCE_DEC_DEG,
    )

    true_NE = project_NE(
        vector,
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
            pred_NE
            - true_NE,
            axis=1,
        )
        / true_Dperp
    )

    Dperp_fractional_error = (
        pred_Dperp
        / true_Dperp
        - 1.0
    )

    projected_angle_error = angle_2d_deg(
        pred_NE,
        true_NE,
    )

    # ============================================================
    # Reports.
    # ============================================================

    for label, mask in (
        (
            "TRAIN 2027",
            train,
        ),
        (
            "BLIND ALL 2028",
            blind,
        ),
        (
            "BLIND ROMAN S3",
            s3,
        ),
    ):

        header(label)

        summarize(
            "3D position error [AU]",
            position_error[
                mask
            ],
        )

        summarize(
            "|radial fractional error|",
            np.abs(
                radial_fractional_error[
                    mask
                ]
            ),
        )

        summarize(
            "relative projected-vector error",
            projected_vector_error[
                mask
            ],
        )

        summarize(
            "|Dperp fractional error|",
            np.abs(
                Dperp_fractional_error[
                    mask
                ]
            ),
        )

        summarize(
            "projected direction error [deg]",
            projected_angle_error[
                mask
            ],
        )

    header("BLIND S3 DECISION NUMBERS")

    x = projected_vector_error[
        s3
    ]

    mag = np.abs(
        Dperp_fractional_error[
            s3
        ]
    )

    ang = projected_angle_error[
        s3
    ]

    print(
        "best period [days] =",
        best_period,
    )

    print()
    print(
        "median relative vector error =",
        float(
            np.median(x)
        ),
    )

    print(
        "p90 relative vector error    =",
        float(
            np.quantile(
                x,
                0.90,
            )
        ),
    )

    print()
    print(
        "median |Dperp error| =",
        float(
            np.median(mag)
        ),
    )

    print(
        "p90 |Dperp error|    =",
        float(
            np.quantile(
                mag,
                0.90,
            )
        ),
    )

    print()
    print(
        "median direction error [deg] =",
        float(
            np.median(ang)
        ),
    )

    print(
        "p90 direction error [deg]    =",
        float(
            np.quantile(
                ang,
                0.90,
            )
        ),
    )

    # ============================================================
    # Save.
    # ============================================================

    output = df.copy()

    output[
        "set"
    ] = np.where(
        train,
        "train_2027",
        "blind_2028",
    )

    output.loc[
        s3,
        "set",
    ] = "blind_S3"

    output[
        "best_period_days"
    ] = best_period

    output[
        "pred_x_au"
    ] = pred_vector[:, 0]

    output[
        "pred_y_au"
    ] = pred_vector[:, 1]

    output[
        "pred_z_au"
    ] = pred_vector[:, 2]

    output[
        "relative_projected_vector_error"
    ] = projected_vector_error

    output[
        "Dperp_fractional_error"
    ] = Dperp_fractional_error

    output[
        "projected_direction_error_deg"
    ] = projected_angle_error

    output_path = Path(
        "decision_tests/"
        "horizons_periodic_halo_validation.csv"
    )

    search_path = Path(
        "decision_tests/"
        "horizons_period_search.csv"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    search.to_csv(
        search_path,
        index=False,
    )

    header("FILES")

    print(
        "validation =",
        output_path.resolve(),
    )

    print(
        "period search =",
        search_path.resolve(),
    )

    print()
    print(
        "2028 was never used to fit either "
        "the period or the orbital coefficients."
    )

    print(
        "No production ephemeris was created "
        "or modified."
    )


if __name__ == "__main__":
    main()
