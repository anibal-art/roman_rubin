#!/usr/bin/env python3

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import get_sun
from astropy.utils.exceptions import AstropyWarning


OEM_PATH = Path(
    "decision_tests/data/RST_103026.oem"
)

HORIZONS_CSV = Path(
    "decision_tests/oem_phase_fit_validation.csv"
)

SOURCE_RA_DEG = 267.92497054815516
SOURCE_DEC_DEG = -29.152232510353276

PHASE_MIN_DAYS = -365.0
PHASE_MAX_DAYS = +365.0
PHASE_STEP_DAYS = 0.25

AU_KM = u.au.to(u.km)
EPSILON_DEG = 23.439291111

SEASONS = {
    "S1": (
        "2027-02-11T00:00:00",
        "2027-04-24T23:59:59",
    ),
    "S2": (
        "2027-08-16T00:00:00",
        "2027-10-27T23:59:59",
    ),
    "S3": (
        "2028-02-11T00:00:00",
        "2028-04-24T23:59:59",
    ),
}


def header(title):
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def parse_oem_time(token):

    date_part, time_part = token.split("T")
    year, doy = date_part.split("-")

    return Time(
        f"{year}:{doy}:{time_part}",
        format="yday",
        scale="utc",
    )


def load_oem(path):

    records = []

    with path.open() as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if not (
                len(line) >= 5
                and line[:4].isdigit()
                and line[4] == "-"
            ):
                continue

            parts = line.split()

            if len(parts) != 7:
                continue

            t = parse_oem_time(
                parts[0]
            )

            records.append(
                {
                    "jd": float(t.jd),
                    "x_km": float(parts[1]),
                    "y_km": float(parts[2]),
                    "z_km": float(parts[3]),
                }
            )

    df = pd.DataFrame(records)

    return (
        df.sort_values("jd")
        .drop_duplicates(
            subset="jd",
            keep="first",
        )
        .reset_index(drop=True)
    )


def unit_from_radec(ra_deg, dec_deg):

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

    norm = np.linalg.norm(
        v,
        axis=1,
        keepdims=True,
    )

    return v / norm


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

    return xhat, yhat, zhat


def inertial_to_rotating(v, basis):

    xhat, yhat, zhat = basis

    return np.column_stack(
        [
            np.sum(v * xhat, axis=1),
            np.sum(v * yhat, axis=1),
            np.sum(v * zhat, axis=1),
        ]
    )


def rotating_to_inertial(v, basis):

    xhat, yhat, zhat = basis

    return (
        v[:, 0, None] * xhat
        + v[:, 1, None] * yhat
        + v[:, 2, None] * zhat
    )


def source_basis(ra_deg, dec_deg):
    """
    Unit vectors on the sky.

    North: increasing Dec.
    East : increasing RA.
    """

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
        ],
        dtype=float,
    )

    east = np.array(
        [
            -np.sin(ra),
            np.cos(ra),
            0.0,
        ],
        dtype=float,
    )

    los = np.array(
        [
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        ],
        dtype=float,
    )

    return (
        north,
        east,
        los,
    )


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


def season_mask(jd, season):

    start_iso, end_iso = SEASONS[
        season
    ]

    start = float(
        Time(
            start_iso,
            format="isot",
            scale="utc",
        ).jd
    )

    end = float(
        Time(
            end_iso,
            format="isot",
            scale="utc",
        ).jd
    )

    return (
        (jd >= start)
        &
        (jd <= end)
    )


def angle_2d_deg(a, b):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

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
        f"{label:38s}"
        f" min={np.nanmin(x):.8g}"
        f" median={np.nanmedian(x):.8g}"
        f" p90={np.nanquantile(x, 0.90):.8g}"
        f" max={np.nanmax(x):.8g}"
    )


def main():

    header("INPUT")

    if not OEM_PATH.exists():
        raise FileNotFoundError(
            OEM_PATH
        )

    if not HORIZONS_CSV.exists():
        raise FileNotFoundError(
            HORIZONS_CSV
        )

    oem = load_oem(
        OEM_PATH
    )

    horizons = pd.read_csv(
        HORIZONS_CSV
    )

    required = {
        "jd",
        "ra_deg",
        "dec_deg",
        "distance_au",
    }

    missing = (
        required
        - set(horizons.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing Horizons columns: "
            + repr(sorted(missing))
        )

    horizons = (
        horizons
        .sort_values("jd")
        .drop_duplicates(
            subset="jd",
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        "OEM rows      =",
        len(oem),
    )

    print(
        "Horizons rows =",
        len(horizons),
    )

    # ============================================================
    # OEM -> rotating frame.
    # ============================================================

    oem_jd = oem[
        "jd"
    ].to_numpy(
        dtype=float
    )

    oem_vector = (
        oem[
            [
                "x_km",
                "y_km",
                "z_km",
            ]
        ].to_numpy(
            dtype=float
        )
        / AU_KM
    )

    oem_rot = inertial_to_rotating(
        oem_vector,
        rotating_basis(
            oem_jd
        ),
    )

    interpolators = [
        interp1d(
            oem_jd,
            oem_rot[:, axis],
            kind="cubic",
            bounds_error=False,
            fill_value=np.nan,
        )
        for axis in range(3)
    ]

    def evaluate_rotating(jd):

        jd = np.asarray(
            jd,
            dtype=float,
        )

        return np.column_stack(
            [
                f(jd)
                for f in interpolators
            ]
        )

    # ============================================================
    # Horizons actual Earth->Roman vectors.
    # ============================================================

    h_jd = horizons[
        "jd"
    ].to_numpy(
        dtype=float
    )

    h_direction = unit_from_radec(
        horizons["ra_deg"],
        horizons["dec_deg"],
    )

    h_vector = (
        horizons[
            "distance_au"
        ].to_numpy(
            dtype=float
        )[:, None]
        * h_direction
    )

    h_basis = rotating_basis(
        h_jd
    )

    north, east, los = source_basis(
        SOURCE_RA_DEG,
        SOURCE_DEC_DEG,
    )

    true_NE = project_NE(
        h_vector,
        north,
        east,
    )

    true_Dperp = np.linalg.norm(
        true_NE,
        axis=1,
    )

    # ============================================================
    # TRAIN = S1 + S2.
    # BLIND = S3.
    # ============================================================

    mask_s1 = season_mask(
        h_jd,
        "S1",
    )

    mask_s2 = season_mask(
        h_jd,
        "S2",
    )

    mask_s3 = season_mask(
        h_jd,
        "S3",
    )

    train_mask = (
        mask_s1
        | mask_s2
    )

    blind_mask = mask_s3

    header("SEASON COUNTS")

    print(
        "S1 rows =",
        int(mask_s1.sum()),
    )

    print(
        "S2 rows =",
        int(mask_s2.sum()),
    )

    print(
        "S3 rows =",
        int(mask_s3.sum()),
    )

    print(
        "TRAIN S1+S2 =",
        int(train_mask.sum()),
    )

    print(
        "BLIND S3    =",
        int(blind_mask.sum()),
    )

    if train_mask.sum() == 0:
        raise RuntimeError(
            "Training season mask is empty."
        )

    if blind_mask.sum() == 0:
        raise RuntimeError(
            "Blind season mask is empty."
        )

    # ============================================================
    # PHASE SEARCH.
    #
    # Objective = RMS of the two-dimensional projected
    # Earth-Roman baseline vector in the source plane.
    # ============================================================

    header("PROJECTED-BASELINE PHASE SEARCH")

    phase_grid = np.arange(
        PHASE_MIN_DAYS,
        PHASE_MAX_DAYS
        + 0.5 * PHASE_STEP_DAYS,
        PHASE_STEP_DAYS,
    )

    results = []

    for shift in phase_grid:

        rot = evaluate_rotating(
            h_jd + shift
        )

        finite = np.all(
            np.isfinite(rot),
            axis=1,
        )

        valid_train = (
            train_mask
            & finite
        )

        if (
            valid_train.sum()
            != train_mask.sum()
        ):
            continue

        pred_vector = rotating_to_inertial(
            rot,
            h_basis,
        )

        pred_NE = project_NE(
            pred_vector,
            north,
            east,
        )

        residual_NE = (
            pred_NE[train_mask]
            - true_NE[train_mask]
        )

        rms_vector_au = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        residual_NE**2,
                        axis=1,
                    )
                )
            )
        )

        relative_vector_error = (
            np.linalg.norm(
                residual_NE,
                axis=1,
            )
            /
            true_Dperp[
                train_mask
            ]
        )

        results.append(
            {
                "phase_shift_days":
                    float(shift),

                "projected_rms_au":
                    rms_vector_au,

                "median_relative_vector_error":
                    float(
                        np.median(
                            relative_vector_error
                        )
                    ),

                "p90_relative_vector_error":
                    float(
                        np.quantile(
                            relative_vector_error,
                            0.90,
                        )
                    ),
            }
        )

    search = pd.DataFrame(
        results
    )

    if len(search) == 0:
        raise RuntimeError(
            "No valid phase shifts."
        )

    best = search.loc[
        search[
            "projected_rms_au"
        ].idxmin()
    ]

    best_shift = float(
        best[
            "phase_shift_days"
        ]
    )

    print(
        "best phase shift [d] =",
        best_shift,
    )

    print(
        "projected RMS [AU]   =",
        float(
            best[
                "projected_rms_au"
            ]
        ),
    )

    print(
        "projected RMS [km]   =",
        float(
            best[
                "projected_rms_au"
            ]
        ) * AU_KM,
    )

    print()
    print(
        "Ten best phase solutions:"
    )

    print(
        search
        .sort_values(
            "projected_rms_au"
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    # ============================================================
    # Evaluate best phase at every Horizons epoch.
    # ============================================================

    pred_rot = evaluate_rotating(
        h_jd + best_shift
    )

    if not np.all(
        np.isfinite(
            pred_rot
        )
    ):
        raise RuntimeError(
            "Best shift has epochs outside OEM support."
        )

    pred_vector = rotating_to_inertial(
        pred_rot,
        h_basis,
    )

    pred_NE = project_NE(
        pred_vector,
        north,
        east,
    )

    pred_Dperp = np.linalg.norm(
        pred_NE,
        axis=1,
    )

    residual_NE = (
        pred_NE
        - true_NE
    )

    absolute_vector_error = np.linalg.norm(
        residual_NE,
        axis=1,
    )

    relative_vector_error = (
        absolute_vector_error
        / true_Dperp
    )

    fractional_Dperp_error = (
        pred_Dperp
        / true_Dperp
        - 1.0
    )

    projected_angle_error = angle_2d_deg(
        pred_NE,
        true_NE,
    )

    north_error = (
        pred_NE[:, 0]
        - true_NE[:, 0]
    )

    east_error = (
        pred_NE[:, 1]
        - true_NE[:, 1]
    )

    # ============================================================
    # Report each scientific season separately.
    # ============================================================

    for season, mask in (
        ("S1 TRAIN", mask_s1),
        ("S2 TRAIN", mask_s2),
        ("S3 BLIND", mask_s3),
    ):

        header(season)

        summarize(
            "true D_perp [AU]",
            true_Dperp[
                mask
            ],
        )

        summarize(
            "|projected vector error| [AU]",
            absolute_vector_error[
                mask
            ],
        )

        summarize(
            "relative projected vector error",
            relative_vector_error[
                mask
            ],
        )

        summarize(
            "|D_perp fractional error|",
            np.abs(
                fractional_Dperp_error[
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

        summarize(
            "|North component error| [AU]",
            np.abs(
                north_error[
                    mask
                ]
            ),
        )

        summarize(
            "|East component error| [AU]",
            np.abs(
                east_error[
                    mask
                ]
            ),
        )

    # ============================================================
    # Blind decision metrics.
    # ============================================================

    header("BLIND S3 DECISION METRICS")

    blind_vector = (
        relative_vector_error[
            blind_mask
        ]
    )

    blind_mag = np.abs(
        fractional_Dperp_error[
            blind_mask
        ]
    )

    blind_angle = (
        projected_angle_error[
            blind_mask
        ]
    )

    print(
        "phase shift [d] =",
        best_shift,
    )

    print()
    print(
        "median relative projected-vector error =",
        float(
            np.median(
                blind_vector
            )
        ),
    )

    print(
        "p90 relative projected-vector error    =",
        float(
            np.quantile(
                blind_vector,
                0.90,
            )
        ),
    )

    print(
        "max relative projected-vector error    =",
        float(
            np.max(
                blind_vector
            )
        ),
    )

    print()
    print(
        "median |Dperp magnitude error| =",
        float(
            np.median(
                blind_mag
            )
        ),
    )

    print(
        "p90 |Dperp magnitude error|    =",
        float(
            np.quantile(
                blind_mag,
                0.90,
            )
        ),
    )

    print()
    print(
        "median projected direction error [deg] =",
        float(
            np.median(
                blind_angle
            )
        ),
    )

    print(
        "p90 projected direction error [deg]    =",
        float(
            np.quantile(
                blind_angle,
                0.90,
            )
        ),
    )

    # ============================================================
    # Save.
    # ============================================================

    output = horizons.copy()

    output["season"] = "outside"

    output.loc[
        mask_s1,
        "season",
    ] = "S1_train"

    output.loc[
        mask_s2,
        "season",
    ] = "S2_train"

    output.loc[
        mask_s3,
        "season",
    ] = "S3_blind"

    output[
        "phase_shift_days"
    ] = best_shift

    output[
        "true_DN_au"
    ] = true_NE[:, 0]

    output[
        "true_DE_au"
    ] = true_NE[:, 1]

    output[
        "pred_DN_au"
    ] = pred_NE[:, 0]

    output[
        "pred_DE_au"
    ] = pred_NE[:, 1]

    output[
        "true_Dperp_au"
    ] = true_Dperp

    output[
        "pred_Dperp_au"
    ] = pred_Dperp

    output[
        "relative_vector_error"
    ] = relative_vector_error

    output[
        "Dperp_fractional_error"
    ] = fractional_Dperp_error

    output[
        "projected_direction_error_deg"
    ] = projected_angle_error

    output_path = Path(
        "decision_tests/"
        "oem_projected_phase_validation.csv"
    )

    search_path = Path(
        "decision_tests/"
        "oem_projected_phase_search.csv"
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
        "phase grid =",
        search_path.resolve(),
    )

    print()
    print(
        "No production ephemeris was created "
        "or modified."
    )


if __name__ == "__main__":
    main()
