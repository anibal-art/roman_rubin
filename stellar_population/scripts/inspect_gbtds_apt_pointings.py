#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u


PATH = Path(
    "config/apt_1420_simulator_input.ecsv"
)


def find_column(columns, candidates):
    """
    Case-insensitive exact/fuzzy column finder.
    """
    lower = {
        str(c).lower(): c
        for c in columns
    }

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    # Fuzzy fallback
    for c in columns:
        low = str(c).lower()

        for candidate in candidates:
            if candidate.lower() in low:
                return c

    return None


if not PATH.exists():
    print()
    print("APT export not found:")
    print(" ", PATH)
    print()
    print(
        "Export Program 1420 with:"
    )
    print(
        "APT -> File -> Export -> Simulator Input"
    )
    sys.exit(1)


# ============================================================
# Read ECSV
# ============================================================

table = Table.read(
    PATH,
    format="ascii.ecsv",
)

df = table.to_pandas()

print()
print("=======================================")
print("APT 1420 Simulator Input")
print("=======================================")

print()
print("Rows:")
print(len(df))

print()
print("Columns:")

for i, c in enumerate(df.columns):
    print(
        f"{i:3d}: {c}"
    )


# ============================================================
# Identify RA / Dec
# ============================================================

ra_col = find_column(
    df.columns,
    [
        "ra",
        "ra_deg",
        "target_ra",
        "pointing_ra",
        "ra_v1",
    ],
)

dec_col = find_column(
    df.columns,
    [
        "dec",
        "dec_deg",
        "target_dec",
        "pointing_dec",
        "dec_v1",
    ],
)

if ra_col is None or dec_col is None:
    raise RuntimeError(
        "Could not identify RA/Dec columns.\n"
        f"RA candidate: {ra_col}\n"
        f"Dec candidate: {dec_col}\n"
        "Use the printed column list to update candidates."
    )

print()
print("Coordinate columns:")
print(" RA  =", ra_col)
print(" Dec =", dec_col)


# ============================================================
# Numeric conversion
# ============================================================

ra = pd.to_numeric(
    df[ra_col],
    errors="coerce",
)

dec = pd.to_numeric(
    df[dec_col],
    errors="coerce",
)

good = (
    ra.notna()
    & dec.notna()
)

work = df.loc[good].copy()

ra = ra.loc[good].to_numpy()
dec = dec.loc[good].to_numpy()

print()
print(
    "Rows with valid coordinates:",
    len(work),
)


# ============================================================
# Equatorial -> Galactic
# ============================================================

coord = SkyCoord(
    ra=ra * u.deg,
    dec=dec * u.deg,
    frame="icrs",
)

gal = coord.galactic

work["apt_ra_deg"] = ra
work["apt_dec_deg"] = dec
work["apt_l_deg"] = gal.l.deg
work["apt_b_deg"] = gal.b.deg


# Put Galactic longitude around zero rather than 359.x.
work["apt_l_signed_deg"] = (
    (
        work["apt_l_deg"]
        + 180.0
    )
    % 360.0
    - 180.0
)


# ============================================================
# Look for useful metadata columns
# ============================================================

possible = {
    "pa": [
        "pa",
        "position_angle",
        "position angle",
        "pa_v3",
        "orient",
        "orientation",
    ],
    "filter": [
        "filter",
        "optical_element",
        "element",
    ],
    "observation": [
        "observation",
        "observation_id",
        "obs_id",
        "obs",
    ],
    "visit": [
        "visit",
        "visit_id",
    ],
    "target": [
        "target",
        "target_name",
    ],
    "activity": [
        "activity",
        "activity_id",
    ],
}

identified = {}

for key, candidates in possible.items():
    identified[key] = find_column(
        work.columns,
        candidates,
    )

print()
print("Detected metadata columns:")

for key, value in identified.items():
    print(
        f"  {key:12s}: {value}"
    )


# ============================================================
# Coordinate diagnostics
# ============================================================

print()
print("=======================================")
print("Coordinate ranges")
print("=======================================")

print(
    "RA:",
    work["apt_ra_deg"].min(),
    work["apt_ra_deg"].max(),
)

print(
    "Dec:",
    work["apt_dec_deg"].min(),
    work["apt_dec_deg"].max(),
)

print(
    "l signed:",
    work["apt_l_signed_deg"].min(),
    work["apt_l_signed_deg"].max(),
)

print(
    "b:",
    work["apt_b_deg"].min(),
    work["apt_b_deg"].max(),
)


# ============================================================
# Print unique values for useful small-cardinality columns
# ============================================================

for key, col in identified.items():

    if col is None:
        continue

    nunique = work[col].nunique(
        dropna=True
    )

    print()
    print(
        f"{key}: column={col}, "
        f"unique={nunique}"
    )

    if nunique <= 50:
        print(
            work[col]
            .value_counts(
                dropna=False
            )
            .head(50)
            .to_string()
        )


# ============================================================
# Save normalized pointings
# ============================================================

out = Path(
    "config/"
    "apt_1420_pointings_normalized.parquet"
)

work.to_parquet(
    out,
    index=False,
    compression="zstd",
)

print()
print("Saved:")
print(" ", out)


# ============================================================
# Print a compact coordinate sample
# ============================================================

cols = [
    "apt_ra_deg",
    "apt_dec_deg",
    "apt_l_signed_deg",
    "apt_b_deg",
]

for key in [
    "pa",
    "filter",
    "observation",
    "visit",
    "target",
]:
    c = identified[key]

    if c is not None and c not in cols:
        cols.append(c)

print()
print("=======================================")
print("First pointings")
print("=======================================")

print(
    work[cols]
    .head(40)
    .to_string(index=False)
)
