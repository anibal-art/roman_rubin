#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


GRID = Path(
    "noise_models/data/"
    "roman_f146_pandeia_2026p1.csv"
)


df = pd.read_csv(GRID)

required = {
    "detector",
    "mag_ab",
    "snr",
    "partial_saturated",
    "full_saturated",
    "sat_nresultants",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )


# ------------------------------------------------------------
# Normalize booleans robustly
# ------------------------------------------------------------

def as_bool(series):
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
    )


df["partial_saturated"] = as_bool(
    df["partial_saturated"]
)

df["full_saturated"] = as_bool(
    df["full_saturated"]
)


# ------------------------------------------------------------
# Basic integrity
# ------------------------------------------------------------

print()
print("=======================================")
print("ROMAN F146 NOISE GRID AUDIT")
print("=======================================")

print("Rows              =", len(df))
print(
    "Detectors         =",
    df["detector"].nunique(),
)
print(
    "Unique magnitudes =",
    df["mag_ab"].nunique(),
)

duplicates = df.duplicated(
    subset=[
        "detector",
        "background_level",
        "mag_ab",
    ],
    keep=False,
)

print(
    "Duplicate keys    =",
    int(duplicates.sum()),
)

if duplicates.any():
    print(
        df.loc[
            duplicates,
            [
                "detector",
                "background_level",
                "mag_ab",
            ],
        ].to_string(index=False)
    )
    raise RuntimeError(
        "Duplicate grid entries found."
    )


# ------------------------------------------------------------
# Saturation boundaries by detector
# ------------------------------------------------------------

summary = []

for detector, g in df.groupby(
    "detector",
    sort=True,
):

    g = g.sort_values("mag_ab")

    full = g[
        g["full_saturated"]
    ]

    partial_only = g[
        g["partial_saturated"]
        & ~g["full_saturated"]
    ]

    unsat = g[
        ~g["partial_saturated"]
        & ~g["full_saturated"]
    ]

    summary.append(
        {
            "detector":
                detector,

            "brightest_grid_ab":
                g["mag_ab"].min(),

            "faintest_full_sat_ab":
                (
                    full["mag_ab"].max()
                    if len(full)
                    else np.nan
                ),

            "brightest_partial_ab":
                (
                    partial_only["mag_ab"].min()
                    if len(partial_only)
                    else np.nan
                ),

            "faintest_partial_ab":
                (
                    partial_only["mag_ab"].max()
                    if len(partial_only)
                    else np.nan
                ),

            "brightest_unsaturated_ab":
                (
                    unsat["mag_ab"].min()
                    if len(unsat)
                    else np.nan
                ),
        }
    )


summary = pd.DataFrame(summary)

print()
print("=======================================")
print("SATURATION BOUNDARIES")
print("=======================================")
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


# ------------------------------------------------------------
# sat_nresultants structure
# ------------------------------------------------------------

print()
print("=======================================")
print("PARTIAL-SATURATION SEGMENTS")
print("=======================================")

for detector, g in df.groupby(
    "detector",
    sort=True,
):

    g = (
        g[
            g["partial_saturated"]
            & ~g["full_saturated"]
        ]
        .sort_values("mag_ab")
        .copy()
    )

    print()
    print(detector)

    if len(g) == 0:
        print("  no partial-saturation points")
        continue

    print(
        g[
            [
                "mag_ab",
                "snr",
                "sat_nresultants",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


# ------------------------------------------------------------
# Detect large adjacent S/N discontinuities
#
# Only in dense saturation-refinement region.
# ------------------------------------------------------------

print()
print("=======================================")
print("LARGE ADJACENT S/N JUMPS")
print("=======================================")

jump_rows = []

for detector, g in df.groupby(
    "detector",
    sort=True,
):

    g = (
        g[
            (g["mag_ab"] >= 15.5)
            & (g["mag_ab"] <= 18.75)
            & (g["snr"] > 0)
        ]
        .sort_values("mag_ab")
        .copy()
    )

    mags = g["mag_ab"].to_numpy()
    snr = g["snr"].to_numpy()

    for i in range(1, len(g)):

        dm = mags[i] - mags[i - 1]

        # Only compare actual neighboring fine-grid points.
        if dm > 0.051:
            continue

        ratio = snr[i] / snr[i - 1]

        if (
            ratio > 1.15
            or ratio < 1 / 1.15
        ):

            jump_rows.append(
                {
                    "detector":
                        detector,

                    "mag_left":
                        mags[i - 1],

                    "mag_right":
                        mags[i],

                    "snr_left":
                        snr[i - 1],

                    "snr_right":
                        snr[i],

                    "ratio":
                        ratio,

                    "sat_nresultants_left":
                        g.iloc[
                            i - 1
                        ][
                            "sat_nresultants"
                        ],

                    "sat_nresultants_right":
                        g.iloc[
                            i
                        ][
                            "sat_nresultants"
                        ],
                }
            )


jumps = pd.DataFrame(jump_rows)

if len(jumps):
    print(
        jumps.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )
else:
    print("No >15% adjacent jumps found.")


# ------------------------------------------------------------
# Unsaturated monotonicity
# ------------------------------------------------------------

print()
print("=======================================")
print("UNSATURATED MONOTONICITY")
print("=======================================")

bad = []

for detector, g in df.groupby(
    "detector",
    sort=True,
):

    g = (
        g[
            ~g["partial_saturated"]
            & ~g["full_saturated"]
        ]
        .sort_values("mag_ab")
    )

    dsnr = np.diff(
        g["snr"].to_numpy()
    )

    # S/N should decrease as magnitude becomes fainter.
    n_bad = int(
        np.sum(dsnr > 0)
    )

    print(
        f"{detector}: "
        f"violations = {n_bad}"
    )

    if n_bad:
        bad.append(detector)


# ------------------------------------------------------------
# Save compact audit products
# ------------------------------------------------------------

outdir = Path("diagnostics")
outdir.mkdir(
    parents=True,
    exist_ok=True,
)

summary.to_csv(
    outdir
    / "roman_f146_saturation_boundaries.csv",
    index=False,
)

jumps.to_csv(
    outdir
    / "roman_f146_snr_discontinuities.csv",
    index=False,
)


print()
print("=======================================")
print("AUDIT COMPLETE")
print("=======================================")

print(
    "Unsaturated monotonicity failures =",
    len(bad),
)

print()
print("Saved:")
print(
    " diagnostics/"
    "roman_f146_saturation_boundaries.csv"
)
print(
    " diagnostics/"
    "roman_f146_snr_discontinuities.csv"
)
