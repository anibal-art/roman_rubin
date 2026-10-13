#!/usr/bin/env python3

from pathlib import Path
import json
import hashlib
import warnings

import numpy as np
import pandas as pd

from astropy.time import Time
from astropy.coordinates import get_sun
from astropy.utils.exceptions import AstropyWarning


INPUT = Path(
    "decision_tests/oem_phase_fit_validation.csv"
)

OUTPUT_NPY = Path(
    "decision_tests/data/"
    "Roman_positions_periodic_candidate.npy"
)

OUTPUT_JSON = Path(
    "decision_tests/data/"
    "Roman_positions_periodic_candidate.json"
)

OUTPUT_CSV = Path(
    "decision_tests/data/"
    "Roman_positions_periodic_candidate.csv"
)

# Full support comfortably enclosing all Roman seasons.
START_ISO = "2027-01-01T00:00:00"
STOP_ISO = "2032-01-01T00:00:00"

CADENCE_DAYS = 1.0

P_MIN_DAYS = 175.0
P_MAX_DAYS = 190.0
P_STEP_DAYS = 0.01

EPSILON_DEG = 23.439291111


def header(title):
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def unit_from_radec(
    ra_deg,
    dec_deg,
):

    ra = np.deg2rad(
        np.asarray(
            ra_deg,
            dtype=float,
        )
    )

    dec = np.deg2rad(
        np.asarray(
            dec_deg,
            dtype=float,
        )
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
        ],
        dtype=float,
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


def inertial_to_rotating(
    vectors,
    basis,
):

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


def rotating_to_inertial(
    vectors,
    basis,
):

    xhat, yhat, zhat = basis

    return (
        vectors[:, 0, None] * xhat
        + vectors[:, 1, None] * yhat
        + vectors[:, 2, None] * zhat
    )


def design_matrix(
    jd,
    period,
    t_ref,
):

    phase = (
        2.0
        * np.pi
        * (jd - t_ref)
        / period
    )

    return np.column_stack(
        [
            np.ones_like(phase),
            np.cos(phase),
            np.sin(phase),
        ]
    )


def fit_coefficients(
    jd,
    rotating,
    period,
    t_ref,
):

    A = design_matrix(
        jd,
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
                rotating[:, axis],
                rcond=None,
            )
        )

        coefficients[
            axis
        ] = coeff

    return coefficients


def predict_rotating(
    jd,
    period,
    t_ref,
    coefficients,
):

    A = design_matrix(
        jd,
        period,
        t_ref,
    )

    out = np.empty(
        (len(jd), 3),
        dtype=float,
    )

    for axis in range(3):

        out[:, axis] = (
            A
            @ coefficients[axis]
        )

    return out


def cartesian_to_radec_distance(
    vectors,
):

    vectors = np.asarray(
        vectors,
        dtype=float,
    )

    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]

    distance = np.linalg.norm(
        vectors,
        axis=1,
    )

    ra = np.rad2deg(
        np.arctan2(
            y,
            x,
        )
    )

    ra = np.mod(
        ra,
        360.0,
    )

    dec = np.rad2deg(
        np.arcsin(
            z / distance
        )
    )

    return (
        ra,
        dec,
        distance,
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
            + repr(
                sorted(missing)
            )
        )

    df = (
        df.sort_values("jd")
        .drop_duplicates(
            subset="jd",
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        "input rows =",
        len(df),
    )

    print(
        "input first JD =",
        df.iloc[0]["jd"],
    )

    print(
        "input last JD  =",
        df.iloc[-1]["jd"],
    )

    print(
        "input sha256 =",
        sha256(INPUT),
    )

    jd = df[
        "jd"
    ].to_numpy(
        dtype=float,
    )

    direction = unit_from_radec(
        df["ra_deg"],
        df["dec_deg"],
    )

    inertial = (
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
        inertial,
        basis,
    )

    t_ref = float(
        np.mean(jd)
    )

    # ============================================================
    # Refit period using ALL validated Horizons support.
    # ============================================================

    header("PERIOD SEARCH")

    periods = np.arange(
        P_MIN_DAYS,
        P_MAX_DAYS
        + 0.5 * P_STEP_DAYS,
        P_STEP_DAYS,
    )

    rows = []

    for period in periods:

        coeff = fit_coefficients(
            jd,
            rotating,
            period,
            t_ref,
        )

        pred = predict_rotating(
            jd,
            period,
            t_ref,
            coeff,
        )

        residual = (
            pred
            - rotating
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
            (
                float(period),
                rms,
            )
        )

    search = pd.DataFrame(
        rows,
        columns=[
            "period_days",
            "rms_au",
        ],
    )

    best = search.loc[
        search[
            "rms_au"
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
        "fit RMS [AU]    =",
        float(
            best["rms_au"]
        ),
    )

    print()
    print(
        search
        .sort_values("rms_au")
        .head(10)
        .to_string(
            index=False
        )
    )

    coefficients = fit_coefficients(
        jd,
        rotating,
        best_period,
        t_ref,
    )

    header("COEFFICIENTS")

    axis_names = [
        "X_antisolar",
        "Y_tangent",
        "Z_ecliptic",
    ]

    for axis, name in enumerate(
        axis_names
    ):

        print()
        print(name)

        print(
            "  C =",
            repr(
                float(
                    coefficients[
                        axis,
                        0,
                    ]
                )
            ),
        )

        print(
            "  A_cos =",
            repr(
                float(
                    coefficients[
                        axis,
                        1,
                    ]
                )
            ),
        )

        print(
            "  B_sin =",
            repr(
                float(
                    coefficients[
                        axis,
                        2,
                    ]
                )
            ),
        )

    # ============================================================
    # Generate daily candidate.
    # ============================================================

    header("GENERATE CANDIDATE")

    start_jd = float(
        Time(
            START_ISO,
            format="isot",
            scale="utc",
        ).jd
    )

    stop_jd = float(
        Time(
            STOP_ISO,
            format="isot",
            scale="utc",
        ).jd
    )

    candidate_jd = np.arange(
        start_jd,
        stop_jd
        + 0.5 * CADENCE_DAYS,
        CADENCE_DAYS,
    )

    candidate_rot = predict_rotating(
        candidate_jd,
        best_period,
        t_ref,
        coefficients,
    )

    candidate_basis = rotating_basis(
        candidate_jd
    )

    candidate_inertial = (
        rotating_to_inertial(
            candidate_rot,
            candidate_basis,
        )
    )

    (
        candidate_ra,
        candidate_dec,
        candidate_distance,
    ) = cartesian_to_radec_distance(
        candidate_inertial
    )

    candidate = np.column_stack(
        [
            candidate_jd,
            candidate_ra,
            candidate_dec,
            candidate_distance,
        ]
    )

    print(
        "shape =",
        candidate.shape,
    )

    print(
        "first =",
        candidate[0],
    )

    print(
        "last  =",
        candidate[-1],
    )

    print()
    print(
        "distance min    =",
        float(
            np.min(
                candidate_distance
            )
        ),
    )

    print(
        "distance median =",
        float(
            np.median(
                candidate_distance
            )
        ),
    )

    print(
        "distance max    =",
        float(
            np.max(
                candidate_distance
            )
        ),
    )

    # ============================================================
    # Internal reconstruction check.
    # ============================================================

    reconstructed = (
        candidate_distance[:, None]
        * unit_from_radec(
            candidate_ra,
            candidate_dec,
        )
    )

    reconstruction_error = (
        np.max(
            np.abs(
                reconstructed
                - candidate_inertial
            )
        )
    )

    print()
    print(
        "max Cartesian reconstruction "
        "error [AU] =",
        reconstruction_error,
    )

    if not np.all(
        np.isfinite(
            candidate
        )
    ):

        raise RuntimeError(
            "Candidate contains non-finite values."
        )

    if reconstruction_error > 1e-12:

        raise RuntimeError(
            "RA/Dec conversion reconstruction "
            "error is unexpectedly large."
        )

    # ============================================================
    # Save only inside decision_tests/data.
    # ============================================================

    OUTPUT_NPY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        OUTPUT_NPY,
        candidate,
    )

    out_df = pd.DataFrame(
        candidate,
        columns=[
            "jd",
            "ra_deg",
            "dec_deg",
            "distance_au",
        ],
    )

    out_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    metadata = {
        "status":
            "candidate_not_production",

        "model":
            (
                "first harmonic in Sun-Earth "
                "rotating frame"
            ),

        "source":
            str(INPUT),

        "source_sha256":
            sha256(INPUT),

        "fit_rows":
            int(len(df)),

        "fit_start_jd":
            float(jd.min()),

        "fit_stop_jd":
            float(jd.max()),

        "fit_reference_jd":
            float(t_ref),

        "best_period_days":
            float(best_period),

        "coefficients_AU": {
            axis_names[i]: {
                "C":
                    float(
                        coefficients[i, 0]
                    ),

                "cos":
                    float(
                        coefficients[i, 1]
                    ),

                "sin":
                    float(
                        coefficients[i, 2]
                    ),
            }
            for i in range(3)
        },

        "candidate_start_iso":
            START_ISO,

        "candidate_stop_iso":
            STOP_ISO,

        "candidate_cadence_days":
            CADENCE_DAYS,

        "candidate_rows":
            int(len(candidate)),

        "columns": [
            "JD_UTC",
            "RA_deg_geocentric",
            "Dec_deg_geocentric",
            "Earth_Roman_distance_AU",
        ],

        "validation_note": (
            "Model class was selected after blind "
            "2028 validation using a fit performed "
            "only on 2027 data. This final candidate "
            "refits the validated model class using "
            "all available Horizons support."
        ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    header("OUTPUT")

    for path in (
        OUTPUT_NPY,
        OUTPUT_CSV,
        OUTPUT_JSON,
    ):

        print(
            path.resolve()
        )

        print(
            "  sha256 =",
            sha256(path),
        )

    print()
    print(
        "IMPORTANT: candidate only."
    )

    print(
        "No file under ephemerides/ "
        "was modified."
    )

    print(
        "No production configuration "
        "was modified."
    )


if __name__ == "__main__":
    main()
