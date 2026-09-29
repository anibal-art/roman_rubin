#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


GRID = Path(
    "noise_models/data/"
    "roman_f146_pandeia_2026p1.csv"
)

METADATA = Path(
    "noise_models/data/"
    "roman_f146_pandeia_2026p1.json"
)

BUILDER = Path(
    "scripts/build_roman_f146_pandeia_grid.py"
)

MAG_MIN = 15.5
MAG_MAX = 18.75
MAG_STEP = 0.05

BACKGROUND_LEVEL = "medium"


# ============================================================
# Import the already validated Pandeia builder
# ============================================================

spec = importlib.util.spec_from_file_location(
    "roman_grid_builder",
    BUILDER,
)

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


# ============================================================
# Existing grid
# ============================================================

if not GRID.exists():
    raise FileNotFoundError(GRID)

df = pd.read_csv(GRID)

detectors = sorted(
    df["detector"].unique()
)

if len(detectors) != 18:
    raise RuntimeError(
        f"Expected 18 detectors, found {len(detectors)}"
    )


rows = df.to_dict(
    orient="records"
)

done = {
    (
        str(r["detector"]),
        str(r["background_level"]),
        round(float(r["mag_ab"]), 8),
    )
    for r in rows
}


mags = np.round(
    np.arange(
        MAG_MIN,
        MAG_MAX + MAG_STEP / 2,
        MAG_STEP,
    ),
    8,
)


n_requested = (
    len(detectors) * len(mags)
)

n_missing = sum(
    (
        detector,
        BACKGROUND_LEVEL,
        round(float(mag), 8),
    )
    not in done
    for detector in detectors
    for mag in mags
)


print()
print("=======================================")
print("Roman F146 saturation refinement")
print("=======================================")

print(
    f"Range      = {MAG_MIN:.2f} -- "
    f"{MAG_MAX:.2f} AB"
)

print(
    f"Step       = {MAG_STEP:.2f} mag"
)

print(
    f"Detectors  = {len(detectors)}"
)

print(
    f"Requested  = {n_requested}"
)

print(
    f"Already in coarse grid = "
    f"{n_requested - n_missing}"
)

print(
    f"New Pandeia calculations = "
    f"{n_missing}"
)


# ============================================================
# Add missing calculations
# ============================================================

counter = 0

for detector in detectors:

    print()
    print(
        f"Detector {detector}"
    )

    base = mod.build_base_calc(
        detector=detector,
        background_level=BACKGROUND_LEVEL,
    )

    for mag in mags:

        key = (
            detector,
            BACKGROUND_LEVEL,
            round(float(mag), 8),
        )

        if key in done:
            continue

        # RuntimeWarnings are expected for the fully saturated
        # regime. Saturation flags remain explicitly recorded.
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                RuntimeWarning,
            )

            result = mod.calculate_one(
                base_calc=base,
                mag_ab=float(mag),
            )

        row = {
            "detector":
                detector,

            "background":
                "gbtds_mid_5stripe",

            "background_level":
                BACKGROUND_LEVEL,

            "filter":
                "f146",

            "ma_table":
                "im_66_6_v2",

            "nexp":
                1,

            "nresultants":
                -1,

            "mag_ab":
                float(mag),

            **result,
        }

        rows.append(row)
        done.add(key)

        counter += 1

        print(
            f"[{counter:4d}/{n_missing}] "
            f"AB={mag:5.2f} "
            f"S/N={result['snr']:9.3f} "
            f"partial="
            f"{result['partial_saturated']} "
            f"full="
            f"{result['full_saturated']}"
        )

        # resumability
        out = (
            pd.DataFrame(rows)
            .drop_duplicates(
                subset=[
                    "detector",
                    "background_level",
                    "mag_ab",
                ],
                keep="last",
            )
            .sort_values(
                [
                    "background_level",
                    "detector",
                    "mag_ab",
                ]
            )
        )

        tmp = GRID.with_suffix(
            ".csv.tmp"
        )

        out.to_csv(
            tmp,
            index=False,
        )

        tmp.replace(GRID)


# ============================================================
# Final clean grid
# ============================================================

grid = (
    pd.DataFrame(rows)
    .drop_duplicates(
        subset=[
            "detector",
            "background_level",
            "mag_ab",
        ],
        keep="last",
    )
    .sort_values(
        [
            "background_level",
            "detector",
            "mag_ab",
        ]
    )
    .reset_index(drop=True)
)

grid.to_csv(
    GRID,
    index=False,
)


# ============================================================
# Update metadata without lying about a single uniform step
# ============================================================

with open(METADATA) as f:
    metadata = json.load(f)

metadata[
    "grid_mag_min_actual"
] = float(
    grid["mag_ab"].min()
)

metadata[
    "grid_mag_max_actual"
] = float(
    grid["mag_ab"].max()
)

metadata[
    "grid_n_rows"
] = int(
    len(grid)
)

metadata[
    "grid_n_detectors"
] = int(
    grid["detector"].nunique()
)

metadata[
    "grid_n_unique_magnitudes"
] = int(
    grid["mag_ab"].nunique()
)

metadata[
    "grid_sampling"
] = {
    "coarse": {
        "mag_min_ab": 12.0,
        "mag_max_ab": 30.0,
        "step_mag": 0.5,
    },

    "saturation_refinement": {
        "mag_min_ab": MAG_MIN,
        "mag_max_ab": MAG_MAX,
        "step_mag": MAG_STEP,
    },

    "benchmark_extra_ab":
        21.2,
}


benchmark = grid[
    np.isclose(
        grid["mag_ab"],
        21.2,
    )
    &
    (
        grid["background_level"]
        == "medium"
    )
]

metadata[
    "benchmark_snr_median"
] = float(
    benchmark["snr"].median()
)

metadata[
    "benchmark_snr_min"
] = float(
    benchmark["snr"].min()
)

metadata[
    "benchmark_snr_max"
] = float(
    benchmark["snr"].max()
)


with open(
    METADATA,
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )

    f.write("\n")


print()
print("=======================================")
print("REFINEMENT COMPLETE")
print("=======================================")

print(
    "Final rows =",
    len(grid),
)

print(
    "Unique magnitudes =",
    grid["mag_ab"].nunique(),
)

print()
print("Saved:")
print(" ", GRID)
print(" ", METADATA)
