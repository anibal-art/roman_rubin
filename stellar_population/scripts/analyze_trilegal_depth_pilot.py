#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


PILOT_CONFIG = Path(
    "config/gbtds_trilegal_depth_pilot.csv"
)

PRODUCTION_CONFIG = Path(
    "config/gbtds_trilegal_cells_production.csv"
)

OUTPUT_ROOT = Path(
    "pilot_f14626"
)

OUTPUT = Path(
    "diagnostics/"
    "trilegal_depth_pilot_summary.csv"
)

THRESHOLDS = [
    22.0,
    24.0,
    26.0,
]


pilot = pd.read_csv(
    PILOT_CONFIG
)

production = pd.read_csv(
    PRODUCTION_CONFIG
)


results = []


# ============================================================
# Read each pilot
# ============================================================

for _, row in pilot.iterrows():

    label = str(
        row["pilot_label"]
    )

    field_id = str(
        row["field_id"]
    )


    path = (
        OUTPUT_ROOT
        / label
        / "processed"
        / "trilegal_physical_catalog.parquet"
    )


    if not path.exists():
        raise FileNotFoundError(
            f"Missing pilot catalogue:\n  {path}"
        )


    df = pd.read_parquet(
        path
    )


    if df["F146mag"].isna().any():
        raise RuntimeError(
            f"{label}: NaN F146 magnitudes."
        )


    if df["F146mag"].max() > 26.05:
        raise RuntimeError(
            f"{label}: invalid F146 upper limit."
        )


    sample_area = float(
        row["sample_area_deg2"]
    )

    target_area = float(
        row["target_area_deg2"]
    )

    area_weight = (
        target_area
        / sample_area
    )


    parquet_bytes = (
        path.stat().st_size
    )

    bytes_per_row = (
        parquet_bytes / len(df)
        if len(df)
        else np.nan
    )


    for limit in THRESHOLDS:

        x = df[
            df["F146mag"]
            <= limit
        ]


        n = len(x)

        density = (
            n / sample_area
        )

        represented = (
            n * area_weight
        )


        if n:

            bulge_fraction = (
                (x["Gc"] == 4)
                .mean()
            )

            disk_fraction = (
                (x["Gc"] == 1)
                .mean()
            )

            f146_median = (
                x["F146mag"]
                .median()
            )

        else:

            bulge_fraction = np.nan
            disk_fraction = np.nan
            f146_median = np.nan


        results.append(
            {
                "pilot_label":
                    label,

                "field_id":
                    field_id,

                "region":
                    row["region"],

                "l_deg":
                    row["l_deg"],

                "b_deg":
                    row["b_deg"],

                "av_inf":
                    row["av_inf"],

                "extinction_sigma":
                    row[
                        "extinction_sigma"
                    ],

                "limit_f146":
                    limit,

                "n_raw":
                    n,

                "density_per_deg2":
                    density,

                "represented_stars":
                    represented,

                "bulge_fraction":
                    bulge_fraction,

                "thin_disk_fraction":
                    disk_fraction,

                "f146_median":
                    f146_median,

                "parquet_bytes_parent26":
                    parquet_bytes,

                "bytes_per_parent26_row":
                    bytes_per_row,
            }
        )


summary = pd.DataFrame(
    results
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# Main result table
# ============================================================

print()
print("=======================================")
print("TRILEGAL DEPTH PILOT")
print("=======================================")
print()

show = summary[
    [
        "pilot_label",
        "region",
        "av_inf",
        "limit_f146",
        "n_raw",
        "density_per_deg2",
        "bulge_fraction",
    ]
].copy()

print(
    show.to_string(
        index=False,
        float_format=lambda x: f"{x:.5g}",
    )
)


# ============================================================
# Growth of luminosity function
# ============================================================

print()
print("=======================================")
print("STAR-COUNT GROWTH")
print("=======================================")
print()

pivot = (
    summary
    .pivot(
        index=[
            "pilot_label",
            "region",
            "av_inf",
        ],
        columns="limit_f146",
        values="n_raw",
    )
    .reset_index()
)


pivot = pivot.rename(
    columns={
        22.0: "N22",
        24.0: "N24",
        26.0: "N26",
    }
)


pivot["N24/N22"] = (
    pivot["N24"]
    / pivot["N22"]
)

pivot["N26/N24"] = (
    pivot["N26"]
    / pivot["N24"]
)


print(
    pivot.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# Storage / runtime input
# ============================================================

print()
print("=======================================")
print("PARENT F146<26 FILE SIZES")
print("=======================================")
print()


parent26 = summary[
    summary[
        "limit_f146"
    ]
    == 26.0
].copy()


for _, row in parent26.iterrows():

    size_mb = (
        row[
            "parquet_bytes_parent26"
        ]
        / 1024**2
    )

    print(
        f"{row['pilot_label']:10s}: "
        f"{int(row['n_raw']):9,d} stars, "
        f"{size_mb:8.2f} MiB, "
        f"{row['bytes_per_parent26_row']:7.1f} "
        "bytes/star"
    )


# ============================================================
# VERY ROUGH extrapolation to all 510 production cells
#
# Every production cell is assigned the nearest pilot LOS in
# Av within its own region.
#
# This is a COMPUTATIONAL-SCALE estimate, not a Galactic-model
# prediction: stellar density also varies with (l,b).
# ============================================================

print()
print("=======================================")
print("ROUGH FULL-PRODUCTION SCALE")
print("=======================================")
print()

print(
    "Method: nearest pilot in Av within "
    "main/GC region."
)

print(
    "Use only as a storage/runtime scale estimate."
)


pilot26 = (
    parent26
    .set_index(
        "pilot_label"
    )
)


estimated_raw_rows = 0.0
estimated_physical_stars = 0.0
estimated_parquet_bytes = 0.0


for _, cell in production.iterrows():

    candidates = parent26[
        parent26["region"]
        == cell["region"]
    ].copy()


    j = (
        candidates["av_inf"]
        - cell["av_inf"]
    ).abs().idxmin()


    representative = (
        candidates.loc[j]
    )


    # Raw catalogue output:
    #
    # all production jobs simulate the same
    # 1e-4 deg^2 statistical sample.
    estimated_raw_rows += (
        representative["n_raw"]
    )


    # Approximate number of physical stars represented
    # by that cell.
    estimated_physical_stars += (
        representative[
            "density_per_deg2"
        ]
        * cell[
            "target_area_deg2"
        ]
    )


    estimated_parquet_bytes += (
        representative["n_raw"]
        * representative[
            "bytes_per_parent26_row"
        ]
    )


print()
print(
    "Production cells =",
    len(production),
)

print(
    "Estimated raw F146<26 rows =",
    f"{estimated_raw_rows:,.0f}",
)

print(
    "Approx represented stars over footprint =",
    f"{estimated_physical_stars:,.0f}",
)

print(
    "Approx per-cell parquet storage =",
    f"{estimated_parquet_bytes / 1024**3:.2f}",
    "GiB",
)

print()
print(
    "CAUTION: these are only order-of-magnitude "
    "computational estimates."
)


print()
print("Saved:")
print(" ", OUTPUT)
