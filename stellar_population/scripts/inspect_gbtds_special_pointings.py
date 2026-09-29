#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u


APT = Path(
    "config/apt_1420_pointings_normalized.parquet"
)

CENTERS = Path(
    "config/gbtds_apt_tile_centers.csv"
)

OUT = Path(
    "diagnostics/gbtds_special_pointings.csv"
)


df = pd.read_parquet(APT)
centers = pd.read_csv(CENTERS)


# ============================================================
# Season inferred from the explicit APT target name.
# ============================================================

def infer_season(name):
    name = str(name).lower()

    if name.startswith("spring"):
        return "spring"

    if name.startswith("autumn"):
        return "autumn"

    return "unknown"


df["season"] = (
    df["TARGET_NAME"]
    .map(infer_season)
)


# ============================================================
# Distance from the nearest nominal F146 tile center,
# within the same season.
# ============================================================

df["nearest_tile"] = ""
df["nearest_tile_sep_arcsec"] = np.nan


for season in ["spring", "autumn"]:

    mask = (
        df["season"] == season
    )

    x = df.loc[mask]

    nominal = centers[
        centers["season"] == season
    ]

    if len(x) == 0:
        continue

    sky = SkyCoord(
        ra=x["RA"].to_numpy() * u.deg,
        dec=x["DEC"].to_numpy() * u.deg,
        frame="icrs",
    )

    csky = SkyCoord(
        ra=nominal["ra_deg"].to_numpy() * u.deg,
        dec=nominal["dec_deg"].to_numpy() * u.deg,
        frame="icrs",
    )

    idx, sep, _ = (
        sky.match_to_catalog_sky(csky)
    )

    df.loc[
        mask,
        "nearest_tile",
    ] = nominal[
        "tile_name"
    ].to_numpy()[idx]

    df.loc[
        mask,
        "nearest_tile_sep_arcsec",
    ] = sep.arcsec


# ============================================================
# Full program summary
# ============================================================

print()
print("=======================================")
print("ALL APT POSITION ANGLES")
print("=======================================")
print()

print(
    df["PA"]
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print("=======================================")
print("BANDPASSES")
print("=======================================")
print()

print(
    df["BANDPASS"]
    .value_counts()
    .to_string()
)


print()
print("=======================================")
print("PA x BANDPASS")
print("=======================================")
print()

print(
    pd.crosstab(
        df["PA"],
        df["BANDPASS"],
    ).to_string()
)


print()
print("=======================================")
print("TARGET_NAME x PA")
print("=======================================")
print()

target_pa = pd.crosstab(
    df["TARGET_NAME"],
    df["PA"],
)

print(
    target_pa.to_string()
)


# ============================================================
# Explicitly inspect everything outside the two dominant PAs.
# ============================================================

dominant = df[
    df["PA"].isin(
        [90.6, 270.6]
    )
]

special = df[
    ~df["PA"].isin(
        [90.6, 270.6]
    )
].copy()


print()
print("=======================================")
print("DOMINANT CONFIGURATIONS")
print("=======================================")

print(
    "N dominant =",
    len(dominant),
    "/",
    len(df),
    "=",
    len(dominant) / len(df),
)


print()
print("=======================================")
print("SPECIAL CONFIGURATIONS")
print("=======================================")

print(
    "N special =",
    len(special),
)


# ============================================================
# Group special observations.
# ============================================================

summary = (
    special
    .groupby(
        [
            "season",
            "TARGET_NAME",
            "PA",
            "BANDPASS",
            "nearest_tile",
        ],
        dropna=False,
    )
    .agg(
        N=(
            "RA",
            "size",
        ),

        sep_med_arcsec=(
            "nearest_tile_sep_arcsec",
            "median",
        ),

        sep_p95_arcsec=(
            "nearest_tile_sep_arcsec",
            lambda x: np.nanpercentile(
                x,
                95,
            ),
        ),

        sep_max_arcsec=(
            "nearest_tile_sep_arcsec",
            "max",
        ),

        ra_med=(
            "RA",
            "median",
        ),

        dec_med=(
            "DEC",
            "median",
        ),

        observation_min=(
            "OBSERVATION",
            "min",
        ),

        observation_max=(
            "OBSERVATION",
            "max",
        ),

        visit_min=(
            "VISIT",
            "min",
        ),

        visit_max=(
            "VISIT",
            "max",
        ),

        exposure_time=(
            "EXPOSURE_TIME",
            "median",
        ),
    )
    .reset_index()
)


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

summary.to_csv(
    OUT,
    index=False,
)


print()
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# ============================================================
# How many special F146 exposures actually exist?
# ============================================================

special_f146 = special[
    special["BANDPASS"] == "F146"
]


print()
print("=======================================")
print("SPECIAL F146")
print("=======================================")

print(
    "N special F146 =",
    len(special_f146),
)


if len(special_f146):

    print()
    print(
        pd.crosstab(
            [
                special_f146["season"],
                special_f146["TARGET_NAME"],
            ],
            special_f146["PA"],
        ).to_string()
    )

    print()
    print(
        "Nearest-center separation [arcsec]:"
    )

    print(
        special_f146[
            "nearest_tile_sep_arcsec"
        ]
        .describe(
            percentiles=[
                0.50,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )


print()
print("Saved:")
print(" ", OUT)
