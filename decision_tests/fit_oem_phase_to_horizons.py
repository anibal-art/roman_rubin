#!/usr/bin/env python3

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from astropy.time import Time
from astropy.coordinates import get_sun
from astropy.utils.exceptions import AstropyWarning
from astropy import units as u

from astroquery.jplhorizons import Horizons


OEM_PATH = Path(
    "decision_tests/data/RST_103026.oem"
)

ROMAN_ID = "-211"
GEOCENTER = "500@399"

SOURCE_RA_DEG = 267.92497054815516
SOURCE_DEC_DEG = -29.152232510353276

HORIZONS_START = "2027-02-11"
HORIZONS_END = "2028-10-03"

# Fit phase only on 2027.
TRAIN_END = "2027-12-31"

PHASE_MIN_DAYS = -365.0
PHASE_MAX_DAYS = +365.0
PHASE_STEP_DAYS = 0.25

AU_KM = u.au.to(u.km)

# Mean obliquity, sufficient to define the
# Sun-Earth rotating reference frame.
EPSILON_DEG = 23.439291111


def header(title):
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


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

            values = [
                float(x)
                for x in parts[1:]
            ]

            records.append(
                {
                    "time_oem": parts[0],
                    "jd_utc": float(t.jd),
                    "x_km": values[0],
                    "y_km": values[1],
                    "z_km": values[2],
                }
            )

    df = pd.DataFrame(
        records
    )

    df = (
        df.sort_values("jd_utc")
        .drop_duplicates(
            subset="jd_utc",
            keep="first",
        )
        .reset_index(drop=True)
    )

    return df


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

    norm = np.linalg.norm(
        v,
        axis=1,
        keepdims=True,
    )

    return v / norm


def sun_earth_rotating_basis(jd):

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

    sun_dir = unit_from_radec(
        sun.ra.deg,
        sun.dec.deg,
    )

    # +X = anti-solar direction.
    xhat = -sun_dir

    eps = np.deg2rad(
        EPSILON_DEG
    )

    ecliptic_north = np.array(
        [
            0.0,
            -np.sin(eps),
            np.cos(eps),
        ],
        dtype=float,
    )

    z_reference = np.repeat(
        ecliptic_north[None, :],
        len(jd),
        axis=0,
    )

    # Right-handed basis:
    # X anti-solar
    # Y approximately along Earth's orbital motion
    # Z approximately ecliptic north.
    yhat = normalize_rows(
        np.cross(
            z_reference,
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

    vectors = np.asarray(
        vectors,
        dtype=float,
    )

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
    rotating,
    basis,
):

    rotating = np.asarray(
        rotating,
        dtype=float,
    )

    xhat, yhat, zhat = basis

    return (
        rotating[:, 0, None] * xhat
        + rotating[:, 1, None] * yhat
        + rotating[:, 2, None] * zhat
    )


def daily_jd(
    start_iso,
    stop_iso,
):

    t0 = Time(
        start_iso + "T00:00:00",
        format="isot",
        scale="utc",
    )

    t1 = Time(
        stop_iso + "T00:00:00",
        format="isot",
        scale="utc",
    )

    return np.arange(
        float(t0.jd),
        float(t1.jd) + 0.1,
        1.0,
    )


def query_horizons(jd):

    pieces = []

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


def projected_norm(
    vectors,
    los,
):

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

    return np.linalg.norm(
        vectors - parallel,
        axis=1,
    )


def vector_angle_deg(
    a,
    b,
):

    a = normalize_rows(a)
    b = normalize_rows(b)

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


def summarize(
    label,
    x,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    print(
        f"{label:30s}"
        f" min={np.nanmin(x):.8g}"
        f" median={np.nanmedian(x):.8g}"
        f" p90={np.nanquantile(x, 0.90):.8g}"
        f" max={np.nanmax(x):.8g}"
    )


def main():

    # ============================================================
    # OEM in Sun-Earth rotating coordinates
    # ============================================================

    header("LOAD OEM")

    oem = load_oem(
        OEM_PATH
    )

    print(
        "OEM rows =",
        len(oem),
    )

    print(
        "first =",
        oem.iloc[0]["time_oem"],
    )

    print(
        "last  =",
        oem.iloc[-1]["time_oem"],
    )

    oem_jd = oem[
        "jd_utc"
    ].to_numpy()

    oem_vector_au = (
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

    oem_basis = (
        sun_earth_rotating_basis(
            oem_jd
        )
    )

    oem_rot = inertial_to_rotating(
        oem_vector_au,
        oem_basis,
    )

    header("OEM ROTATING-FRAME RANGE")

    for i, name in enumerate(
        ("X_anti_solar", "Y_tangent", "Z_ecliptic")
    ):

        summarize(
            name + " [AU]",
            oem_rot[:, i],
        )

    # Interpolate the quasi-halo trajectory
    # in the rotating frame.
    interpolators = []

    for axis in range(3):

        interpolators.append(
            interp1d(
                oem_jd,
                oem_rot[:, axis],
                kind="cubic",
                bounds_error=False,
                fill_value=np.nan,
            )
        )

    def evaluate_oem_rotating(
        jd_query
    ):

        jd_query = np.asarray(
            jd_query,
            dtype=float,
        )

        return np.column_stack(
            [
                f(jd_query)
                for f in interpolators
            ]
        )

    # ============================================================
    # Horizons actual/predicted Roman trajectory
    # ============================================================

    header("QUERY HORIZONS")

    jd = daily_jd(
        HORIZONS_START,
        HORIZONS_END,
    )

    horizons = query_horizons(
        jd
    )

    print(
        "Horizons rows =",
        len(horizons),
    )

    print(
        "first JD =",
        horizons.iloc[0]["jd"],
    )

    print(
        "last JD  =",
        horizons.iloc[-1]["jd"],
    )

    h_direction = unit_from_radec(
        horizons["ra_deg"],
        horizons["dec_deg"],
    )

    h_vector = (
        horizons[
            "distance_au"
        ].to_numpy()[:, None]
        * h_direction
    )

    h_basis = (
        sun_earth_rotating_basis(
            horizons[
                "jd"
            ].to_numpy()
        )
    )

    h_rot = inertial_to_rotating(
        h_vector,
        h_basis,
    )

    # ============================================================
    # Train / validation split
    # ============================================================

    train_end_jd = float(
        Time(
            TRAIN_END + "T23:59:59",
            format="isot",
            scale="utc",
        ).jd
    )

    train_mask = (
        horizons["jd"].to_numpy()
        <= train_end_jd
    )

    test_mask = (
        ~train_mask
    )

    header("TRAIN / TEST")

    print(
        "training rows =",
        int(
            train_mask.sum()
        ),
    )

    print(
        "test rows     =",
        int(
            test_mask.sum()
        ),
    )

    # ============================================================
    # Fit one phase offset using TRAINING DATA ONLY.
    # ============================================================

    header("PHASE SEARCH")

    phase_grid = np.arange(
        PHASE_MIN_DAYS,
        PHASE_MAX_DAYS
        + 0.5 * PHASE_STEP_DAYS,
        PHASE_STEP_DAYS,
    )

    train_jd = horizons[
        "jd"
    ].to_numpy()[train_mask]

    train_truth = h_rot[
        train_mask
    ]

    results = []

    for shift in phase_grid:

        prediction = (
            evaluate_oem_rotating(
                train_jd + shift
            )
        )

        finite = np.all(
            np.isfinite(prediction),
            axis=1,
        )

        if (
            finite.sum()
            < 0.95 * len(train_jd)
        ):
            continue

        residual = (
            prediction[finite]
            - train_truth[finite]
        )

        rms_au = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        residual**2,
                        axis=1,
                    )
                )
            )
        )

        results.append(
            (
                float(shift),
                rms_au,
            )
        )

    phase_results = pd.DataFrame(
        results,
        columns=[
            "phase_shift_days",
            "rms_au",
        ],
    )

    if len(phase_results) == 0:
        raise RuntimeError(
            "No valid phase offsets."
        )

    best_row = phase_results.loc[
        phase_results[
            "rms_au"
        ].idxmin()
    ]

    best_shift = float(
        best_row[
            "phase_shift_days"
        ]
    )

    best_rms = float(
        best_row[
            "rms_au"
        ]
    )

    print(
        "best phase shift [d] =",
        best_shift,
    )

    print(
        "training RMS [AU]    =",
        best_rms,
    )

    print(
        "training RMS [km]    =",
        best_rms * AU_KM,
    )

    print()
    print("Ten best phase-grid rows:")

    print(
        phase_results
        .sort_values("rms_au")
        .head(10)
        .to_string(index=False)
    )

    # ============================================================
    # Evaluate the phase-calibrated OEM on ALL data.
    # ============================================================

    pred_rot = evaluate_oem_rotating(
        horizons[
            "jd"
        ].to_numpy()
        + best_shift
    )

    finite = np.all(
        np.isfinite(pred_rot),
        axis=1,
    )

    if not finite.all():
        raise RuntimeError(
            "Best phase offset leaves "
            "Horizons epochs outside OEM support."
        )

    pred_inertial = (
        rotating_to_inertial(
            pred_rot,
            h_basis,
        )
    )

    source_los = unit_from_radec(
        [SOURCE_RA_DEG],
        [SOURCE_DEC_DEG],
    )[0]

    actual_dperp = projected_norm(
        h_vector,
        source_los,
    )

    pred_dperp = projected_norm(
        pred_inertial,
        source_los,
    )

    pos_error_km = (
        np.linalg.norm(
            pred_inertial
            - h_vector,
            axis=1,
        )
        * AU_KM
    )

    angle_error = vector_angle_deg(
        pred_inertial,
        h_vector,
    )

    radial_error_km = (
        np.linalg.norm(
            pred_inertial,
            axis=1,
        )
        - np.linalg.norm(
            h_vector,
            axis=1,
        )
    ) * AU_KM

    dperp_fractional_error = (
        pred_dperp
        / actual_dperp
        - 1.0
    )

    # ============================================================
    # Report train and blind validation separately.
    # ============================================================

    for name, mask in (
        (
            "TRAIN 2027",
            train_mask,
        ),
        (
            "BLIND TEST 2028",
            test_mask,
        ),
    ):

        header(name)

        summarize(
            "position error [km]",
            pos_error_km[mask],
        )

        summarize(
            "direction error [deg]",
            angle_error[mask],
        )

        summarize(
            "radial error [km]",
            np.abs(
                radial_error_km[
                    mask
                ]
            ),
        )

        summarize(
            "|D_perp fractional error|",
            np.abs(
                dperp_fractional_error[
                    mask
                ]
            ),
        )

        print()
        print(
            "signed D_perp fractional:"
        )

        summarize(
            "Dperp pred/true - 1",
            dperp_fractional_error[
                mask
            ],
        )

    # ============================================================
    # Save diagnostics
    # ============================================================

    output = horizons.copy()

    output[
        "set"
    ] = np.where(
        train_mask,
        "train_2027",
        "blind_test_2028",
    )

    output[
        "phase_shift_days"
    ] = best_shift

    output[
        "position_error_km"
    ] = pos_error_km

    output[
        "direction_error_deg"
    ] = angle_error

    output[
        "radial_error_km"
    ] = radial_error_km

    output[
        "dperp_true_au"
    ] = actual_dperp

    output[
        "dperp_pred_au"
    ] = pred_dperp

    output[
        "dperp_fractional_error"
    ] = dperp_fractional_error

    output_path = Path(
        "decision_tests/"
        "oem_phase_fit_validation.csv"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    phase_path = Path(
        "decision_tests/"
        "oem_phase_search.csv"
    )

    phase_results.to_csv(
        phase_path,
        index=False,
    )

    header("FILES")

    print(
        "validation =",
        output_path.resolve(),
    )

    print(
        "phase grid =",
        phase_path.resolve(),
    )

    header("SCIENTIFIC DECISION METRICS")

    blind = np.abs(
        dperp_fractional_error[
            test_mask
        ]
    )

    print(
        "Blind-test median |Dperp error| =",
        float(
            np.median(blind)
        ),
    )

    print(
        "Blind-test p90 |Dperp error|    =",
        float(
            np.quantile(
                blind,
                0.90,
            )
        ),
    )

    print(
        "Blind-test max |Dperp error|    =",
        float(
            np.max(blind)
        ),
    )

    print()
    print(
        "No production ephemeris was created "
        "or modified by this test."
    )


if __name__ == "__main__":
    main()
