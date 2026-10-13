#!/usr/bin/env python3

from pathlib import Path

import numpy as np

from scipy.interpolate import interp1d

from pyLIMA import telescopes


CANDIDATE_PATH = Path(
    "decision_tests/data/"
    "Roman_positions_periodic_candidate.npy"
)

SEASON_TEST_TIMES = [
    # S1
    2461447.5,   # 2027-02-11
    2461483.5,
    2461519.5,   # 2027-04-24

    # S2
    2461633.5,   # 2027-08-16
    2461669.5,
    2461705.5,   # 2027-10-27

    # S3
    2461812.5,   # 2028-02-11
    2461849.0,   # half-day tests interpolation
    2461885.5,   # 2028-04-24

    # S4
    2462543.5,   # 2030-02-11
    2462579.5,
    2462615.5,   # 2030-04-24

    # S5
    2462729.5,   # 2030-08-16
    2462765.5,
    2462801.5,   # 2030-10-27

    # S6
    2462908.5,   # 2031-02-11
    2462944.5,
    2462980.5,   # 2031-04-24
]


def header(title):
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def spherical_to_cartesian(
    ra_deg,
    dec_deg,
    distance,
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

    distance = np.asarray(
        distance,
        dtype=float,
    )

    x = (
        distance
        * np.cos(dec)
        * np.cos(ra)
    )

    y = (
        distance
        * np.cos(dec)
        * np.sin(ra)
    )

    z = (
        distance
        * np.sin(dec)
    )

    return np.column_stack(
        [x, y, z]
    )


def main():

    header("LOAD CANDIDATE")

    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(
            CANDIDATE_PATH
        )

    candidate = np.load(
        CANDIDATE_PATH
    )

    print(
        "shape =",
        candidate.shape,
    )

    print(
        "first JD =",
        candidate[0, 0],
    )

    print(
        "last JD  =",
        candidate[-1, 0],
    )

    test_times = np.asarray(
        SEASON_TEST_TIMES,
        dtype=float,
    )

    print()
    print(
        "test min JD =",
        test_times.min(),
    )

    print(
        "test max JD =",
        test_times.max(),
    )

    inside = (
        test_times.min()
        >= candidate[:, 0].min()
        and
        test_times.max()
        <= candidate[:, 0].max()
    )

    print(
        "all test times inside support =",
        inside,
    )

    if not inside:
        raise RuntimeError(
            "Test times outside candidate support."
        )

    # ============================================================
    # Independent expected Earth -> Roman Cartesian vectors.
    # ============================================================

    header("INDEPENDENT CARTESIAN INTERPOLATION")

    candidate_xyz = spherical_to_cartesian(
        candidate[:, 1],
        candidate[:, 2],
        candidate[:, 3],
    )

    interpolation = [
        interp1d(
            candidate[:, 0],
            candidate_xyz[:, axis],
            kind="linear",
            bounds_error=True,
        )
        for axis in range(3)
    ]

    expected_earth_to_roman = np.column_stack(
        [
            f(test_times)
            for f in interpolation
        ]
    )

    expected_norm = np.linalg.norm(
        expected_earth_to_roman,
        axis=1,
    )

    print(
        "expected norms [AU]:"
    )

    for t, norm in zip(
        test_times,
        expected_norm,
    ):
        print(
            f"  {t:.6f}  {norm:.12f}"
        )

    # ============================================================
    # pyLIMA telescope.
    # ============================================================

    header("PYLIMA")

    light_curve = np.column_stack(
        [
            test_times,
            np.full(
                len(test_times),
                20.0,
            ),
            np.full(
                len(test_times),
                0.01,
            ),
        ]
    )

    telescope = telescopes.Telescope(
        name="Roman_candidate_audit",
        camera_filter="W149",
        lightcurve=light_curve,
        lightcurve_names=[
            "time",
            "mag",
            "err_mag",
        ],
        lightcurve_units=[
            "JD",
            "mag",
            "mag",
        ],
        location="Space",
    )

    telescope.spacecraft_name = "L2"

    telescope.spacecraft_positions = {
        "astrometry": [],
        "photometry": candidate,
    }

    telescope.find_space_positions()

    pylima_position = np.asarray(
        telescope.telescope_positions[
            "photometry"
        ],
        dtype=float,
    )

    print(
        "pyLIMA position shape =",
        pylima_position.shape,
    )

    print(
        "all finite =",
        bool(
            np.all(
                np.isfinite(
                    pylima_position
                )
            )
        ),
    )

    # ============================================================
    # pyLIMA source code uses:
    #
    # satellite_positions = -[x, y, z]
    #
    # Therefore its internal vector is expected to be the negative
    # of the geocentric Earth -> spacecraft vector supplied in the
    # ephemeris.
    # ============================================================

    header("SIGN / NUMERICAL CHECK")

    expected_pylima = (
        -expected_earth_to_roman
    )

    difference = (
        pylima_position
        - expected_pylima
    )

    wrong_sign_difference = (
        pylima_position
        - expected_earth_to_roman
    )

    error = np.linalg.norm(
        difference,
        axis=1,
    )

    wrong_sign_error = np.linalg.norm(
        wrong_sign_difference,
        axis=1,
    )

    pylima_norm = np.linalg.norm(
        pylima_position,
        axis=1,
    )

    for i, t in enumerate(
        test_times
    ):

        print()
        print(
            f"JD = {t:.6f}"
        )

        print(
            "  Earth->Roman expected =",
            expected_earth_to_roman[i],
        )

        print(
            "  pyLIMA internal       =",
            pylima_position[i],
        )

        print(
            "  pyLIMA + Earth->Roman =",
            difference[i],
        )

        print(
            "  error expected sign   =",
            error[i],
        )

        print(
            "  error opposite sign   =",
            wrong_sign_error[i],
        )

        print(
            "  norm expected / pyLIMA =",
            expected_norm[i],
            pylima_norm[i],
        )

    header("SUMMARY")

    print(
        "max expected-sign error [AU] =",
        float(
            np.max(
                error
            )
        ),
    )

    print(
        "median expected-sign error [AU] =",
        float(
            np.median(
                error
            )
        ),
    )

    print(
        "median opposite-sign error [AU] =",
        float(
            np.median(
                wrong_sign_error
            )
        ),
    )

    norm_error = np.abs(
        pylima_norm
        - expected_norm
    )

    print(
        "max norm error [AU] =",
        float(
            np.max(
                norm_error
            )
        ),
    )

    sign_ok = bool(
        np.max(error)
        < 1e-12
    )

    norm_ok = bool(
        np.max(norm_error)
        < 1e-12
    )

    print()
    print(
        "pyLIMA sign convention reproduced =",
        sign_ok,
    )

    print(
        "pyLIMA distance reproduced =",
        norm_ok,
    )

    # ============================================================
    # Confirm pyLIMA did not mutate the original supplied table.
    # ============================================================

    header("SUPPLIED EPHEMERIS")

    stored = np.asarray(
        telescope.spacecraft_positions[
            "photometry"
        ],
        dtype=float,
    )

    print(
        "stored shape =",
        stored.shape,
    )

    print(
        "stored candidate unchanged =",
        bool(
            np.array_equal(
                stored,
                candidate,
            )
        ),
    )

    header("FINAL")

    passed = (
        sign_ok
        and norm_ok
        and np.all(
            np.isfinite(
                pylima_position
            )
        )
    )

    print(
        "PYLIMA CANDIDATE AUDIT PASSED =",
        bool(passed),
    )

    print()
    print(
        "No production ephemeris was created."
    )

    print(
        "No file under ephemerides/ was modified."
    )

    print(
        "No production configuration was modified."
    )


if __name__ == "__main__":
    main()
