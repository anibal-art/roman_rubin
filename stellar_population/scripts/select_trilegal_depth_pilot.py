#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "config/gbtds_trilegal_cells_production.csv"
)

OUTPUT = Path(
    "config/gbtds_trilegal_depth_pilot.csv"
)


# ============================================================
# Pilot definition
#
# Main mosaic:
#   10th, 50th, 90th percentile in Av
#
# Galactic Center:
#   50th, 90th percentile in Av
#
# Quantiles are computed from the COMPLETE production
# population, but representative cells are selected only from
# robust, well-sampled footprint cells.
# ============================================================

SPECS = [
    ("main_low",  "main", 0.10),
    ("main_mid",  "main", 0.50),
    ("main_high", "main", 0.90),

    ("gc_mid",    "gc",   0.50),
    ("gc_high",   "gc",   0.90),
]


df = pd.read_csv(INPUT)


required = {
    "field_id",
    "region",
    "l_deg",
    "b_deg",
    "av_inf",
    "ejk_median",
    "extinction_sigma",
    "extinction_sigma_was_capped",
    "extinction_sampling_mode",
    "fraction_union",
    "n_extinction_nodes_active",
    "sample_area_deg2",
    "target_area_deg2",
    "area_weight",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )


# ============================================================
# Robust candidate pool
# ============================================================

robust = df[
    (df["fraction_union"] >= 0.95)
    &
    (
        df["n_extinction_nodes_active"]
        >= 50
    )
    &
    (
        df["extinction_sampling_mode"]
        == "roman_active_geometry"
    )
    &
    (
        ~df[
            "extinction_sigma_was_capped"
        ].astype(bool)
    )
].copy()


print()
print("=======================================")
print("Production population")
print("=======================================")

print("All production cells =", len(df))
print("Robust pilot pool    =", len(robust))

print()
print("By region:")
print(
    df["region"]
    .value_counts()
    .to_string()
)


# ============================================================
# Deterministic representative selection
# ============================================================

selected = []
already_used = set()


for label, region, quantile in SPECS:

    region_all = df[
        df["region"] == region
    ]

    region_pool = robust[
        robust["region"] == region
    ].copy()

    region_pool = region_pool[
        ~region_pool["field_id"].isin(
            already_used
        )
    ]


    if len(region_pool) == 0:
        raise RuntimeError(
            f"No robust candidate for {label}"
        )


    target_av = float(
        region_all[
            "av_inf"
        ].quantile(
            quantile
        )
    )


    region_pool[
        "_delta_av"
    ] = np.abs(
        region_pool["av_inf"]
        - target_av
    )


    # Primary criterion:
    #   closest Av to requested population quantile.
    #
    # Tie-breaks:
    #   lower differential extinction,
    #   larger represented Roman area,
    #   more Surot nodes.
    region_pool = (
        region_pool
        .sort_values(
            [
                "_delta_av",
                "extinction_sigma",
                "target_area_deg2",
                "n_extinction_nodes_active",
                "field_id",
            ],
            ascending=[
                True,
                True,
                False,
                False,
                True,
            ],
        )
    )


    row = (
        region_pool
        .iloc[0]
        .drop(
            labels=["_delta_av"]
        )
        .copy()
    )


    row["pilot_label"] = label
    row["pilot_av_quantile"] = quantile
    row["pilot_target_av_inf"] = (
        target_av
    )

    row["pilot_delta_av_inf"] = (
        abs(
            float(row["av_inf"])
            - target_av
        )
    )


    selected.append(row)

    already_used.add(
        row["field_id"]
    )


pilot = pd.DataFrame(
    selected
)


# Put pilot metadata first.
first = [
    "pilot_label",
    "pilot_av_quantile",
    "pilot_target_av_inf",
    "pilot_delta_av_inf",
]

rest = [
    c for c in pilot.columns
    if c not in first
]

pilot = pilot[
    first + rest
]


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

pilot.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# Report
# ============================================================

cols = [
    "pilot_label",
    "field_id",
    "region",
    "refinement_level",
    "l_deg",
    "b_deg",
    "pilot_av_quantile",
    "pilot_target_av_inf",
    "av_inf",
    "ejk_median",
    "extinction_sigma",
    "target_area_deg2",
    "fraction_union",
    "season_coverage",
    "n_extinction_nodes_active",
]


print()
print("=======================================")
print("SELECTED DEPTH PILOT")
print("=======================================")
print()

print(
    pilot[
        cols
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print()
print("Saved:")
print(" ", OUTPUT)
