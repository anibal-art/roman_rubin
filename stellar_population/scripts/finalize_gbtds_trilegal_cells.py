#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd

from shapely import contains_xy
from shapely.geometry import (
    shape,
    box,
    mapping,
)
from shapely import wkt


# ============================================================
# Paths
# ============================================================

INPUT = Path(
    "extinction/processed/"
    "gbtds_real_adaptive_cells.parquet"
)

FOOTPRINT = Path(
    "config/gbtds_wfi_footprint_union.geojson"
)

DUST_MAIN = Path(
    "extinction/raw/"
    "surot2020_gbtds_real_main.parquet"
)

DUST_GC = Path(
    "extinction/raw/"
    "surot2020_gbtds_real_gc.parquet"
)

OUTPUT_PARQUET = Path(
    "extinction/processed/"
    "gbtds_real_adaptive_cells_final.parquet"
)

OUTPUT_ALL = Path(
    "config/"
    "gbtds_trilegal_cells_final.csv"
)

OUTPUT_PRODUCTION = Path(
    "config/"
    "gbtds_trilegal_cells_production.csv"
)

OUTPUT_GEOJSON = Path(
    "config/"
    "gbtds_trilegal_cells_final.geojson"
)

SUMMARY = Path(
    "diagnostics/"
    "gbtds_final_production_summary.csv"
)


# ============================================================
# Configuration
# ============================================================

SAMPLE_AREA_DEG2 = 1.0e-4

AV_PER_EJK = 6.0

MAX_EXTINCTION_SIGMA = 0.30

EJK_REFINEMENT_THRESHOLD = 0.10

# Do not launch TRILEGAL for pieces smaller than the
# statistical area simulated by one run.
MIN_PRODUCTION_AREA_DEG2 = 1.0e-4

# Active-geometry extinction requires at least this many
# actual Surot nodes.
MIN_ACTIVE_NODES = 10


# ============================================================
# Load footprints
# ============================================================

with open(FOOTPRINT) as f:
    gj = json.load(f)

geoms = {
    feat["properties"]["name"]:
        shape(feat["geometry"])
    for feat in gj["features"]
}

spring = geoms["spring"]
autumn = geoms["autumn"]
union = geoms["spring_or_autumn"]
overlap = geoms["spring_and_autumn"]


# ============================================================
# Load extinction maps
# ============================================================

dust = pd.concat(
    [
        pd.read_parquet(DUST_MAIN),
        pd.read_parquet(DUST_GC),
    ],
    ignore_index=True,
)

dust = (
    dust
    .drop_duplicates(
        subset=["GLON", "GLAT"]
    )
    .reset_index(drop=True)
)


# ============================================================
# Helpers
# ============================================================

def select_nodes(geom):
    minx, miny, maxx, maxy = geom.bounds

    candidate = dust[
        (dust["GLON"] >= minx)
        & (dust["GLON"] <= maxx)
        & (dust["GLAT"] >= miny)
        & (dust["GLAT"] <= maxy)
    ]

    if len(candidate) == 0:
        return candidate

    inside = contains_xy(
        geom,
        candidate["GLON"].to_numpy(),
        candidate["GLAT"].to_numpy(),
    )

    return candidate.loc[inside]


def extinction_stats(
    active_geom,
    full_geom,
):
    active = select_nodes(
        active_geom
    )

    n_active = len(active)

    if n_active >= MIN_ACTIVE_NODES:
        sample = active
        mode = "roman_active_geometry"

    else:
        sample = select_nodes(
            full_geom
        )

        if len(sample) == 0:
            raise RuntimeError(
                "No Surot nodes inside cell."
            )

        mode = "full_cell_fallback"

    ejk = (
        sample["E(J-Ks)"]
        .to_numpy(dtype=float)
    )

    p16, med, p84 = np.nanpercentile(
        ejk,
        [16, 50, 84],
    )

    sigma = 0.5 * (
        p84 - p16
    )

    raw_rel = (
        sigma / med
        if med > 0
        else 0.0
    )

    return {
        "n_extinction_nodes_active":
            n_active,

        "n_extinction_nodes_used":
            len(sample),

        "extinction_sampling_mode":
            mode,

        "ejk_median":
            med,

        "ejk_p16":
            p16,

        "ejk_p84":
            p84,

        "ejk_sigma_spatial":
            sigma,

        "ejk_measurement_error_median":
            sample[
                "e_E(J-Ks)"
            ].median(),

        "map_res_avg_arcmin":
            sample[
                "res-avg"
            ].median(),

        "map_res_max_arcmin":
            sample[
                "res-max"
            ].max(),

        "av_per_ejk":
            AV_PER_EJK,

        "av_inf":
            AV_PER_EJK * med,

        "av_p16":
            AV_PER_EJK * p16,

        "av_p84":
            AV_PER_EJK * p84,

        "av_sigma_spatial":
            AV_PER_EJK * sigma,

        "extinction_sigma_raw":
            raw_rel,

        "extinction_sigma":
            float(
                np.clip(
                    raw_rel,
                    0.0,
                    MAX_EXTINCTION_SIGMA,
                )
            ),

        "extinction_sigma_was_capped":
            bool(
                raw_rel
                > MAX_EXTINCTION_SIGMA
            ),

        "needs_refinement":
            bool(
                sigma
                > EJK_REFINEMENT_THRESHOLD
            ),
    }


def build_child(
    parent,
    full_geom,
    child_id,
):
    active = (
        full_geom
        .intersection(union)
    )

    if active.is_empty:
        return None

    if active.area <= 1e-12:
        return None

    spring_part = (
        full_geom
        .intersection(spring)
    )

    autumn_part = (
        full_geom
        .intersection(autumn)
    )

    overlap_part = (
        full_geom
        .intersection(overlap)
    )

    area_cell = float(
        full_geom.area
    )

    area_union = float(
        active.area
    )

    area_spring = float(
        spring_part.area
    )

    area_autumn = float(
        autumn_part.area
    )

    area_overlap = float(
        overlap_part.area
    )

    centroid = active.centroid

    l_deg = float(
        centroid.x
    )

    b_deg = float(
        centroid.y
    )

    row = parent.copy()

    row["field_id"] = child_id
    row["parent_cell_id"] = (
        parent["field_id"]
    )

    row["refinement_level"] = 2

    row["l_deg"] = l_deg
    row["b_deg"] = b_deg

    row["cell_center_l_deg"] = (
        full_geom.centroid.x
    )

    row["cell_center_b_deg"] = (
        full_geom.centroid.y
    )

    (
        row["l_min_deg"],
        row["b_min_deg"],
        row["l_max_deg"],
        row["b_max_deg"],
    ) = full_geom.bounds

    row["cell_geom_area_deg2"] = (
        area_cell
    )

    row["target_area_deg2"] = (
        area_union
    )

    row["area_union_deg2"] = (
        area_union
    )

    row["area_spring_deg2"] = (
        area_spring
    )

    row["area_autumn_deg2"] = (
        area_autumn
    )

    row["area_overlap_deg2"] = (
        area_overlap
    )

    row["fraction_union"] = (
        area_union / area_cell
    )

    row["fraction_spring"] = (
        area_spring / area_cell
    )

    row["fraction_autumn"] = (
        area_autumn / area_cell
    )

    row["fraction_overlap"] = (
        area_overlap / area_cell
    )

    row["fraction_spring_of_union"] = (
        area_spring / area_union
    )

    row["fraction_autumn_of_union"] = (
        area_autumn / area_union
    )

    row["fraction_overlap_of_union"] = (
        area_overlap / area_union
    )

    row["covered_spring"] = (
        area_spring > 1e-12
    )

    row["covered_autumn"] = (
        area_autumn > 1e-12
    )

    if (
        row["covered_spring"]
        and row["covered_autumn"]
    ):
        row["season_coverage"] = (
            "spring_and_autumn"
        )

    elif row["covered_spring"]:
        row["season_coverage"] = (
            "spring_only"
        )

    else:
        row["season_coverage"] = (
            "autumn_only"
        )

    row["sample_area_deg2"] = (
        SAMPLE_AREA_DEG2
    )

    row["area_weight"] = (
        area_union
        / SAMPLE_AREA_DEG2
    )

    stats = extinction_stats(
        active,
        full_geom,
    )

    row.update(stats)

    row["cell_geometry_wkt"] = (
        full_geom.wkt
    )

    row["union_geometry_wkt"] = (
        active.wkt
    )

    row["spring_geometry_wkt"] = (
        spring_part.wkt
    )

    row["autumn_geometry_wkt"] = (
        autumn_part.wkt
    )

    row["geometry_version"] = (
        "APT1420_PySIAF_adaptive_v2"
    )

    return row


# ============================================================
# Load current v1 adaptive result
# ============================================================

df = pd.read_parquet(
    INPUT
)

print()
print("=======================================")
print("Input adaptive grid")
print("=======================================")

print("Cells =", len(df))

print(
    "Area =",
    df["target_area_deg2"].sum(),
)

print(
    "Sigma capped =",
    int(
        df[
            "extinction_sigma_was_capped"
        ].sum()
    ),
)


# ============================================================
# Identify only physically meaningful capped cells
# ============================================================

refine_mask = (
    df[
        "extinction_sigma_was_capped"
    ]
    &
    (
        df[
            "extinction_sampling_mode"
        ]
        == "roman_active_geometry"
    )
)

parents = df[
    refine_mask
].copy()

print()
print(
    "Level-1 cells requiring "
    "one final refinement =",
    len(parents),
)

if len(parents):

    print(
        parents[
            [
                "field_id",
                "region",
                "target_area_deg2",
                "ejk_median",
                "ejk_sigma_spatial",
                "extinction_sigma_raw",
                "n_extinction_nodes_active",
            ]
        ]
        .to_string(index=False)
    )


# ============================================================
# Keep all other cells unchanged
# ============================================================

records = (
    df.loc[
        ~refine_mask
    ]
    .to_dict(
        orient="records"
    )
)


# ============================================================
# Split each selected 3' cell into 2x2 = 1.5' cells
# ============================================================

for _, parent_row in parents.iterrows():

    parent = (
        parent_row.to_dict()
    )

    parent_geom = wkt.loads(
        parent[
            "cell_geometry_wkt"
        ]
    )

    minx, miny, maxx, maxy = (
        parent_geom.bounds
    )

    xmid = 0.5 * (
        minx + maxx
    )

    ymid = 0.5 * (
        miny + maxy
    )

    x_edges = [
        minx,
        xmid,
        maxx,
    ]

    y_edges = [
        miny,
        ymid,
        maxy,
    ]

    for iy in range(2):

        for ix in range(2):

            child_geom = box(
                x_edges[ix],
                y_edges[iy],
                x_edges[ix + 1],
                y_edges[iy + 1],
            )

            child_id = (
                f"{parent['field_id']}"
                f"_r2_y{iy}_x{ix}"
            )

            child = build_child(
                parent,
                child_geom,
                child_id,
            )

            if child is not None:
                records.append(
                    child
                )


final = pd.DataFrame(
    records
)


# ============================================================
# Production selection
#
# We KEEP all cells in the master product.
#
# For TRILEGAL production we omit:
#   1) tiny fragments < 1e-4 deg2;
#   2) cells for which Surot had to fall back to the full
#      rectangular cell;
#   3) any cell still hitting TRILEGAL's sigma=0.30 ceiling.
# ============================================================

final[
    "production_enabled"
] = (
    (
        final[
            "target_area_deg2"
        ]
        >= MIN_PRODUCTION_AREA_DEG2
    )
    &
    (
        final[
            "extinction_sampling_mode"
        ]
        == "roman_active_geometry"
    )
    &
    (
        ~final[
            "extinction_sigma_was_capped"
        ]
    )
)


disabled_reason = []

for _, row in final.iterrows():

    reasons = []

    if (
        row[
            "target_area_deg2"
        ]
        < MIN_PRODUCTION_AREA_DEG2
    ):
        reasons.append(
            "tiny_fragment"
        )

    if (
        row[
            "extinction_sampling_mode"
        ]
        != "roman_active_geometry"
    ):
        reasons.append(
            "extinction_fallback"
        )

    if row[
        "extinction_sigma_was_capped"
    ]:
        reasons.append(
            "extinction_sigma_capped"
        )

    disabled_reason.append(
        ";".join(reasons)
    )

final[
    "production_disabled_reason"
] = disabled_reason


# ============================================================
# Integrity
# ============================================================

total_area = float(
    final[
        "target_area_deg2"
    ].sum()
)

expected_area = float(
    union.area
)

if not np.isclose(
    total_area,
    expected_area,
    atol=1e-7,
    rtol=0,
):
    raise RuntimeError(
        "Area not conserved: "
        f"{total_area} vs {expected_area}"
    )


enabled = final[
    final[
        "production_enabled"
    ]
].copy()

disabled = final[
    ~final[
        "production_enabled"
    ]
].copy()


# ============================================================
# Save master parquet
# ============================================================

OUTPUT_PARQUET.parent.mkdir(
    parents=True,
    exist_ok=True,
)

final.to_parquet(
    OUTPUT_PARQUET,
    index=False,
    compression="zstd",
)


# ============================================================
# CSV columns
# ============================================================

config_columns = [
    "field_id",
    "parent_cell_id",
    "refinement_level",
    "region",

    "l_deg",
    "b_deg",

    "l_min_deg",
    "l_max_deg",
    "b_min_deg",
    "b_max_deg",

    "target_area_deg2",
    "sample_area_deg2",
    "area_weight",

    "covered_spring",
    "covered_autumn",
    "season_coverage",

    "fraction_union",
    "fraction_spring",
    "fraction_autumn",
    "fraction_overlap",

    "fraction_spring_of_union",
    "fraction_autumn_of_union",
    "fraction_overlap_of_union",

    "area_spring_deg2",
    "area_autumn_deg2",
    "area_overlap_deg2",

    "ejk_median",
    "ejk_p16",
    "ejk_p84",
    "ejk_sigma_spatial",

    "av_inf",
    "extinction_sigma",

    "n_extinction_nodes_active",
    "n_extinction_nodes_used",
    "extinction_sampling_mode",

    "map_res_avg_arcmin",
    "map_res_max_arcmin",

    "needs_refinement",
    "extinction_sigma_raw",
    "extinction_sigma_was_capped",

    "production_enabled",
    "production_disabled_reason",

    "extinction_map",
    "extinction_catalog",
    "av_per_ejk",
    "geometry_version",
]


OUTPUT_ALL.parent.mkdir(
    parents=True,
    exist_ok=True,
)

final[
    config_columns
].to_csv(
    OUTPUT_ALL,
    index=False,
)

enabled[
    config_columns
].to_csv(
    OUTPUT_PRODUCTION,
    index=False,
)


# ============================================================
# GeoJSON: preserve every cell
# ============================================================

features = []

for _, row in final.iterrows():

    geom = wkt.loads(
        row[
            "union_geometry_wkt"
        ]
    )

    features.append(
        {
            "type": "Feature",

            "properties": {
                "field_id":
                    row["field_id"],

                "refinement_level":
                    int(
                        row[
                            "refinement_level"
                        ]
                    ),

                "region":
                    row["region"],

                "target_area_deg2":
                    row[
                        "target_area_deg2"
                    ],

                "production_enabled":
                    bool(
                        row[
                            "production_enabled"
                        ]
                    ),

                "disabled_reason":
                    row[
                        "production_disabled_reason"
                    ],

                "av_inf":
                    row["av_inf"],

                "extinction_sigma":
                    row[
                        "extinction_sigma"
                    ],
            },

            "geometry":
                mapping(geom),
        }
    )


OUTPUT_GEOJSON.write_text(
    json.dumps(
        {
            "type":
                "FeatureCollection",

            "features":
                features,
        },
        indent=2,
    )
)


# ============================================================
# Summary
# ============================================================

enabled_area = float(
    enabled[
        "target_area_deg2"
    ].sum()
)

disabled_area = float(
    disabled[
        "target_area_deg2"
    ].sum()
)


summary = pd.DataFrame(
    [
        {
            "metric":
                "roman_union_area_deg2",
            "value":
                expected_area,
        },
        {
            "metric":
                "final_cells",
            "value":
                len(final),
        },
        {
            "metric":
                "production_cells",
            "value":
                len(enabled),
        },
        {
            "metric":
                "disabled_cells",
            "value":
                len(disabled),
        },
        {
            "metric":
                "production_area_deg2",
            "value":
                enabled_area,
        },
        {
            "metric":
                "disabled_area_deg2",
            "value":
                disabled_area,
        },
        {
            "metric":
                "disabled_area_fraction",
            "value":
                disabled_area
                / expected_area,
        },
        {
            "metric":
                "remaining_sigma_caps",
            "value":
                int(
                    final[
                        "extinction_sigma_was_capped"
                    ].sum()
                ),
        },
        {
            "metric":
                "remaining_fallbacks",
            "value":
                int(
                    (
                        final[
                            "extinction_sampling_mode"
                        ]
                        != "roman_active_geometry"
                    ).sum()
                ),
        },
    ]
)

SUMMARY.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary.to_csv(
    SUMMARY,
    index=False,
)


# ============================================================
# Console output
# ============================================================

print()
print("=======================================")
print("FINAL PRODUCTION GRID")
print("=======================================")

print()
print(
    "Final cells          =",
    len(final),
)

print(
    "Level 0              =",
    int(
        (
            final[
                "refinement_level"
            ]
            == 0
        ).sum()
    ),
)

print(
    "Level 1              =",
    int(
        (
            final[
                "refinement_level"
            ]
            == 1
        ).sum()
    ),
)

print(
    "Level 2              =",
    int(
        (
            final[
                "refinement_level"
            ]
            == 2
        ).sum()
    ),
)


print()
print("Area:")

print(
    "Roman union          =",
    f"{expected_area:.9f}",
)

print(
    "Master-cell area     =",
    f"{total_area:.9f}",
)

print(
    "Production area      =",
    f"{enabled_area:.9f}",
)

print(
    "Disabled area        =",
    f"{disabled_area:.9f}",
)

print(
    "Disabled fraction    =",
    f"{disabled_area / expected_area:.8f}",
)


print()
print("Production:")

print(
    "Enabled cells        =",
    len(enabled),
)

print(
    "Disabled cells       =",
    len(disabled),
)


print()
print("Disabled reasons:")

if len(disabled):

    reasons = (
        disabled[
            "production_disabled_reason"
        ]
        .value_counts()
    )

    print(
        reasons.to_string()
    )


print()
print("Remaining sigma caps:")

remaining_caps = final[
    final[
        "extinction_sigma_was_capped"
    ]
]

print(
    len(remaining_caps)
)

if len(remaining_caps):

    print(
        remaining_caps[
            [
                "field_id",
                "region",
                "refinement_level",
                "target_area_deg2",
                "ejk_median",
                "ejk_sigma_spatial",
                "extinction_sigma_raw",
                "extinction_sampling_mode",
                "production_enabled",
            ]
        ]
        .to_string(index=False)
    )


print()
print("Saved:")
print(" ", OUTPUT_PARQUET)
print(" ", OUTPUT_ALL)
print(" ", OUTPUT_PRODUCTION)
print(" ", OUTPUT_GEOJSON)
print(" ", SUMMARY)

print()
print(
    "TRILEGAL has NOT been launched."
)
