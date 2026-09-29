#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u


INPUT = Path(
    "config/apt_1420_pointings_normalized.parquet"
)

OUTPUT = Path(
    "config/gbtds_apt_tile_centers.csv"
)

COMPARISON_OUTPUT = Path(
    "diagnostics/gbtds_spring_autumn_offsets.csv"
)

OUTLIER_OUTPUT = Path(
    "diagnostics/gbtds_apt_position_outliers.csv"
)


# ============================================================
# Configuration
# ============================================================

# Much smaller than the ~0.4 deg separation of the main tiles,
# but much larger than ordinary pointing jitter/dithers.
ANCHOR_MIN_SEPARATION_ARCMIN = 5.0

# Used only as a diagnostic after clustering.
OUTLIER_THRESHOLD_ARCSEC = 60.0


# ============================================================
# Helpers
# ============================================================

def signed_l(l_deg):
    return (
        (np.asarray(l_deg) + 180.0)
        % 360.0
        - 180.0
    )


def spherical_center(coords):
    """
    Unit-vector mean of a SkyCoord array, returned as a
    conventional spherical ICRS SkyCoord.
    """

    xyz = coords.cartesian.xyz.value

    vec = np.mean(
        xyz,
        axis=1,
    )

    vec /= np.linalg.norm(vec)

    cart = SkyCoord(
        x=vec[0],
        y=vec[1],
        z=vec[2],
        representation_type="cartesian",
        frame="icrs",
    )

    sph = cart.spherical

    return SkyCoord(
        ra=sph.lon,
        dec=sph.lat,
        frame="icrs",
    )


def find_initial_anchors(coords, n_clusters=6):
    """
    Walk through the APT exposure sequence and retain the first
    coordinate that is farther than ANCHOR_MIN_SEPARATION_ARCMIN
    from all previously retained coordinates.

    The APT sequence repeatedly visits six widely separated
    WFI_CEN positions, so this gives six empirical seeds without
    making any assumption about VISIT numbering.
    """

    anchors = []

    for c in coords:

        if not anchors:
            anchors.append(c)
            continue

        anchor_coords = SkyCoord(anchors)

        min_sep = np.min(
            c.separation(anchor_coords).arcmin
        )

        if min_sep > ANCHOR_MIN_SEPARATION_ARCMIN:
            anchors.append(c)

        if len(anchors) == n_clusters:
            break

    if len(anchors) != n_clusters:
        raise RuntimeError(
            f"Expected {n_clusters} spatial anchors, "
            f"found {len(anchors)}."
        )

    return SkyCoord(anchors)


def assign_to_centers(coords, centers):
    """
    Assign every coordinate to its nearest center.
    """

    index, separation, _ = (
        coords.match_to_catalog_sky(centers)
    )

    return (
        np.asarray(index, dtype=int),
        separation,
    )


def recompute_centers(coords, labels, n_clusters=6):
    centers = []

    for k in range(n_clusters):
        c = coords[labels == k]

        if len(c) == 0:
            raise RuntimeError(
                f"Spatial cluster {k} is empty."
            )

        centers.append(
            spherical_center(c)
        )

    return SkyCoord(centers)


def label_physical_tiles(centers):
    """
    Identify:
      - one GC field, closest to Galactic latitude zero;
      - five main fields near b ~ -1.4, sorted by decreasing l.

    Returns a mapping:
        cluster_index -> tile_name
    """

    gal = centers.galactic

    l = signed_l(gal.l.deg)
    b = gal.b.deg

    # The GC field is dramatically closer to b=0 than the
    # five main Bulge fields.
    gc_idx = int(
        np.argmax(b)
    )

    main_indices = [
        i
        for i in range(len(centers))
        if i != gc_idx
    ]

    main_sorted = sorted(
        main_indices,
        key=lambda i: l[i],
        reverse=True,
    )

    mapping = {
        gc_idx: "tileGC"
    }

    for number, idx in enumerate(
        main_sorted,
        start=1,
    ):
        mapping[idx] = f"tile{number}"

    return mapping


# ============================================================
# Load
# ============================================================

df = pd.read_parquet(INPUT)

required = [
    "RA",
    "DEC",
    "PA",
    "BANDPASS",
    "OBSERVATION",
    "VISIT",
    "TARGET_NAME",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# Main survey configurations
# ============================================================

configs = [
    {
        "season": "spring",
        "pa": 90.6,
        "target": "spring-tile3-00",
    },
    {
        "season": "autumn",
        "pa": 270.6,
        "target": "autumn-tile3-00",
    },
]


center_rows = []
outlier_rows = []


for cfg in configs:

    season = cfg["season"]

    sub = df[
        (df["TARGET_NAME"] == cfg["target"])
        & np.isclose(
            pd.to_numeric(
                df["PA"],
                errors="coerce",
            ),
            cfg["pa"],
            atol=1e-6,
        )
        & (df["BANDPASS"] == "F146")
    ].copy()

    if len(sub) == 0:
        raise RuntimeError(
            f"No F146 rows found for {season}"
        )

    coords = SkyCoord(
        ra=pd.to_numeric(
            sub["RA"],
            errors="raise",
        ).to_numpy() * u.deg,

        dec=pd.to_numeric(
            sub["DEC"],
            errors="raise",
        ).to_numpy() * u.deg,

        frame="icrs",
    )

    print()
    print("=======================================")
    print(season.upper())
    print("=======================================")
    print()

    print(
        "Selected F146 exposures:",
        len(sub),
    )

    # --------------------------------------------------------
    # 1. Empirical initial positions
    # --------------------------------------------------------

    anchors = find_initial_anchors(
        coords,
        n_clusters=6,
    )

    print()
    print("Initial empirical anchors:")

    anchor_gal = anchors.galactic

    for i in range(6):
        print(
            f"  anchor {i}: "
            f"RA={anchors[i].ra.deg:.6f} "
            f"Dec={anchors[i].dec.deg:.6f} "
            f"l={float(signed_l(anchor_gal[i].l.deg)):.6f} "
            f"b={anchor_gal[i].b.deg:.6f}"
        )

    # --------------------------------------------------------
    # 2. First nearest-center assignment
    # --------------------------------------------------------

    labels, sep = assign_to_centers(
        coords,
        anchors,
    )

    # --------------------------------------------------------
    # 3. Recalculate centers
    # --------------------------------------------------------

    centers = recompute_centers(
        coords,
        labels,
        n_clusters=6,
    )

    # --------------------------------------------------------
    # 4. Reassign to refined centers
    # --------------------------------------------------------

    labels, sep = assign_to_centers(
        coords,
        centers,
    )

    # One final recentering.
    centers = recompute_centers(
        coords,
        labels,
        n_clusters=6,
    )

    labels, sep = assign_to_centers(
        coords,
        centers,
    )

    # --------------------------------------------------------
    # Give physical tile names from Galactic location
    # --------------------------------------------------------

    tile_mapping = label_physical_tiles(
        centers
    )

    centers_gal = centers.galactic

    # --------------------------------------------------------
    # Diagnostics per spatial cluster
    # --------------------------------------------------------

    print()
    print("Spatial clusters:")

    for k in range(6):

        mask = labels == k

        x = sub.loc[mask].copy()

        this_sep = sep[mask].arcsec

        tile_name = tile_mapping[k]

        l = float(
            signed_l(
                centers_gal[k].l.deg
            )
        )

        b = float(
            centers_gal[k].b.deg
        )

        center_rows.append(
            {
                "season": season,
                "tile_name": tile_name,
                "cluster_id": k,

                "ra_deg": centers[k].ra.deg,
                "dec_deg": centers[k].dec.deg,

                "l_deg": l,
                "b_deg": b,

                "pa_deg": cfg["pa"],

                "n_f146_exposures": int(
                    mask.sum()
                ),

                "sep_median_arcsec": float(
                    np.median(this_sep)
                ),

                "sep_p95_arcsec": float(
                    np.percentile(
                        this_sep,
                        95,
                    )
                ),

                "sep_p99_arcsec": float(
                    np.percentile(
                        this_sep,
                        99,
                    )
                ),

                "sep_max_arcsec": float(
                    np.max(this_sep)
                ),

                "n_outside_60arcsec": int(
                    np.sum(
                        this_sep
                        > OUTLIER_THRESHOLD_ARCSEC
                    )
                ),

                "ra_min_deg": float(
                    x["RA"].min()
                ),

                "ra_max_deg": float(
                    x["RA"].max()
                ),

                "dec_min_deg": float(
                    x["DEC"].min()
                ),

                "dec_max_deg": float(
                    x["DEC"].max()
                ),
            }
        )

        print(
            f"  {tile_name:6s}: "
            f"N={mask.sum():6d} "
            f"l={l: .6f} "
            f"b={b: .6f} "
            f"median={np.median(this_sep):8.3f}\" "
            f"p95={np.percentile(this_sep,95):8.3f}\" "
            f"p99={np.percentile(this_sep,99):8.3f}\" "
            f"max={np.max(this_sep):8.3f}\""
        )

        # Save genuine geometric outliers for inspection.
        bad_local = (
            this_sep
            > OUTLIER_THRESHOLD_ARCSEC
        )

        if np.any(bad_local):

            bad = x.loc[
                bad_local
            ].copy()

            bad["season"] = season
            bad["assigned_tile"] = (
                tile_name
            )

            bad["separation_arcsec"] = (
                this_sep[bad_local]
            )

            outlier_rows.append(
                bad
            )


# ============================================================
# Build center table
# ============================================================

centers_df = pd.DataFrame(
    center_rows
)

tile_order = {
    "tile1": 1,
    "tile2": 2,
    "tile3": 3,
    "tile4": 4,
    "tile5": 5,
    "tileGC": 6,
}

centers_df["_order"] = (
    centers_df["tile_name"]
    .map(tile_order)
)

centers_df = (
    centers_df
    .sort_values(
        ["season", "_order"]
    )
    .drop(columns="_order")
    .reset_index(drop=True)
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

centers_df.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# Spring / Autumn comparison
# ============================================================

spring = (
    centers_df[
        centers_df["season"] == "spring"
    ]
    .set_index("tile_name")
)

autumn = (
    centers_df[
        centers_df["season"] == "autumn"
    ]
    .set_index("tile_name")
)

comparison_rows = []

for tile_name in tile_order:

    s = spring.loc[
        tile_name
    ]

    a = autumn.loc[
        tile_name
    ]

    cs = SkyCoord(
        ra=s["ra_deg"] * u.deg,
        dec=s["dec_deg"] * u.deg,
        frame="icrs",
    )

    ca = SkyCoord(
        ra=a["ra_deg"] * u.deg,
        dec=a["dec_deg"] * u.deg,
        frame="icrs",
    )

    comparison_rows.append(
        {
            "tile_name": tile_name,

            "spring_l_deg": (
                s["l_deg"]
            ),

            "spring_b_deg": (
                s["b_deg"]
            ),

            "autumn_l_deg": (
                a["l_deg"]
            ),

            "autumn_b_deg": (
                a["b_deg"]
            ),

            "delta_l_deg": (
                a["l_deg"]
                - s["l_deg"]
            ),

            "delta_b_deg": (
                a["b_deg"]
                - s["b_deg"]
            ),

            "center_separation_arcmin": (
                cs.separation(ca).arcmin
            ),
        }
    )

comparison = pd.DataFrame(
    comparison_rows
)

COMPARISON_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

comparison.to_csv(
    COMPARISON_OUTPUT,
    index=False,
)


# ============================================================
# Outliers
# ============================================================

if outlier_rows:

    outliers = pd.concat(
        outlier_rows,
        ignore_index=True,
    )

else:
    outliers = pd.DataFrame()

OUTLIER_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

outliers.to_csv(
    OUTLIER_OUTPUT,
    index=False,
)


# ============================================================
# Final reports
# ============================================================

print()
print("=======================================")
print("APT SPATIAL TILE CENTERS")
print("=======================================")
print()

cols = [
    "season",
    "tile_name",
    "ra_deg",
    "dec_deg",
    "l_deg",
    "b_deg",
    "pa_deg",
    "n_f146_exposures",
    "sep_median_arcsec",
    "sep_p95_arcsec",
    "sep_p99_arcsec",
    "sep_max_arcsec",
    "n_outside_60arcsec",
]

print(
    centers_df[cols]
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


print()
print("=======================================")
print("SPRING <-> AUTUMN OFFSETS")
print("=======================================")
print()

print(
    comparison
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


print()
print("=======================================")
print("INTEGRITY")
print("=======================================")

for season in [
    "spring",
    "autumn",
]:
    x = centers_df[
        centers_df["season"] == season
    ]

    total = int(
        x["n_f146_exposures"].sum()
    )

    print(
        f"{season:7s}: "
        f"{len(x)} clusters, "
        f"{total:,} assigned exposures"
    )

    if len(x) != 6:
        raise RuntimeError(
            f"{season}: expected 6 clusters."
        )

print()
print(
    "Exposures farther than "
    f"{OUTLIER_THRESHOLD_ARCSEC:.0f}\" "
    "from their nearest tile center:",
    len(outliers),
)

if len(outliers):
    print()
    print(
        outliers[
            [
                "season",
                "assigned_tile",
                "RA",
                "DEC",
                "PA",
                "OBSERVATION",
                "VISIT",
                "EXPOSURE",
                "TARGET_NAME",
                "separation_arcsec",
            ]
        ]
        .sort_values(
            "separation_arcsec",
            ascending=False,
        )
        .head(30)
        .to_string(index=False)
    )

print()
print("Saved:")
print(" ", OUTPUT)
print(" ", COMPARISON_OUTPUT)
print(" ", OUTLIER_OUTPUT)
