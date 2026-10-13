#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np

try:
    from astropy.time import Time
except Exception:
    Time = None


def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def describe_array(label, arr):
    print_header(label)

    print("shape =", arr.shape)
    print("dtype =", arr.dtype)
    print("ndim  =", arr.ndim)

    if arr.ndim != 2:
        print("WARNING: expected a 2D array.")
        return

    print("n_rows =", arr.shape[0])
    print("n_cols =", arr.shape[1])

    if arr.shape[0] == 0:
        return

    print()
    print("first row:")
    print(arr[0])

    print()
    print("last row:")
    print(arr[-1])

    if arr.shape[1] >= 1:
        t = np.asarray(arr[:, 0], dtype=float)

        finite = np.isfinite(t)

        print()
        print("time column:")
        print("  finite =", int(finite.sum()), "/", len(t))

        if finite.any():
            tf = t[finite]

            print("  min =", repr(float(np.min(tf))))
            print("  max =", repr(float(np.max(tf))))
            print("  span_days =", repr(float(np.max(tf) - np.min(tf))))

            if len(tf) > 1:
                dt = np.diff(tf)

                print("  cadence median [d] =", repr(float(np.median(dt))))
                print("  cadence min    [d] =", repr(float(np.min(dt))))
                print("  cadence max    [d] =", repr(float(np.max(dt))))

            if (
                Time is not None
                and np.min(tf) > 2_000_000
                and np.max(tf) < 3_000_000
            ):
                print(
                    "  first ISO =",
                    Time(float(tf[0]), format="jd").iso,
                )
                print(
                    "  last ISO  =",
                    Time(float(tf[-1]), format="jd").iso,
                )

    if arr.shape[1] > 1:
        xyz = np.asarray(arr[:, 1:], dtype=float)

        print()
        print("non-time columns:")

        for j in range(xyz.shape[1]):
            x = xyz[:, j]
            finite = np.isfinite(x)

            if finite.any():
                print(
                    f"  col {j + 1}: "
                    f"min={np.min(x[finite]):.16g} "
                    f"max={np.max(x[finite]):.16g}"
                )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--current",
        default="ephemerides/Roman_positions.npy",
        type=Path,
    )

    parser.add_argument(
        "--legacy",
        default="ephemerides/Roman_positionsv0.npy",
        type=Path,
    )

    args = parser.parse_args()

    current_path = args.current.resolve()
    legacy_path = args.legacy.resolve()

    print_header("FILES")

    print("current =", current_path)
    print("legacy  =", legacy_path)

    if not current_path.exists():
        raise FileNotFoundError(current_path)

    if not legacy_path.exists():
        raise FileNotFoundError(legacy_path)

    current = np.load(current_path)
    legacy = np.load(legacy_path)

    describe_array("CURRENT: Roman_positions.npy", current)
    describe_array("LEGACY: Roman_positionsv0.npy", legacy)

    print_header("DIRECT COMPARISON")

    print("same shape =", current.shape == legacy.shape)

    if current.shape != legacy.shape:
        print(
            "Arrays have different shapes; "
            "row-by-row comparison is not valid."
        )
        return

    if current.ndim != 2 or current.shape[1] < 1:
        print("Unexpected array structure.")
        return

    current64 = np.asarray(current, dtype=np.float64)
    legacy64 = np.asarray(legacy, dtype=np.float64)

    t_current = current64[:, 0]
    t_legacy = legacy64[:, 0]

    dt = t_current - t_legacy

    print("time offset current - legacy [d]:")
    print("  first  =", repr(float(dt[0])))
    print("  median =", repr(float(np.median(dt))))
    print("  mean   =", repr(float(np.mean(dt))))
    print("  min    =", repr(float(np.min(dt))))
    print("  max    =", repr(float(np.max(dt))))
    print("  std    =", repr(float(np.std(dt))))
    print(
        "  peak-to-peak =",
        repr(float(np.ptp(dt))),
    )

    offset = float(np.median(dt))

    residual = dt - offset

    print()
    print("after removing median time offset:")
    print(
        "  max |residual| [d] =",
        repr(float(np.max(np.abs(residual)))),
    )

    print(
        "  constant within 1e-10 d =",
        bool(np.all(np.abs(residual) <= 1e-10)),
    )

    print(
        "  constant within 1e-8 d  =",
        bool(np.all(np.abs(residual) <= 1e-8)),
    )

    if current.shape[1] > 1:
        current_pos = current64[:, 1:]
        legacy_pos = legacy64[:, 1:]

        pos_diff = current_pos - legacy_pos

        print()
        print("spatial/non-time columns:")

        print(
            "  array_equal =",
            bool(np.array_equal(current_pos, legacy_pos)),
        )

        print(
            "  allclose rtol=0 atol=0 =",
            bool(
                np.allclose(
                    current_pos,
                    legacy_pos,
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                )
            ),
        )

        print(
            "  max |current - legacy| =",
            repr(
                float(
                    np.nanmax(
                        np.abs(pos_diff)
                    )
                )
            ),
        )

        for j in range(pos_diff.shape[1]):
            col = pos_diff[:, j]

            print(
                f"  col {j + 1}: "
                f"max_abs_diff="
                f"{np.nanmax(np.abs(col)):.17g}"
            )

    print_header("INTERPRETATION FLAGS")

    constant_offset = bool(
        np.all(np.abs(residual) <= 1e-8)
    )

    same_positions = True

    if current.shape[1] > 1:
        same_positions = bool(
            np.allclose(
                current64[:, 1:],
                legacy64[:, 1:],
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
        )

    print(
        "constant_time_shift =",
        constant_offset,
    )

    print(
        "identical_non_time_columns =",
        same_positions,
    )

    if constant_offset and same_positions:
        print()
        print(
            "RESULT: current ephemerides are exactly the legacy "
            "non-time coordinates with a constant time translation."
        )
        print(
            "Measured translation [days] =",
            repr(offset),
        )
    else:
        print()
        print(
            "RESULT: the relation is more complicated than a "
            "pure constant time translation."
        )

    print()
    print(
        "NOTE: this script does NOT establish whether either "
        "ephemeris is physically correct for Roman/L2."
    )


if __name__ == "__main__":
    main()
