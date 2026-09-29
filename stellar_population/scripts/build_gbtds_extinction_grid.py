#!/usr/bin/env python3

from pathlib import Path
import time

import numpy as np
import pandas as pd

from astroquery.vizier import Vizier


# ============================================================
# Configuration
# ============================================================

CATALOG = "J/A+A/644/A140/ejkmap"

# Roman WFI instantaneous field area.
WFI_AREA_DEG2 = 0.281

# ----------------------------------------------------------------
# Population-sampling geometry
#
# Current STScI documentation places the main five-field mosaic
# around:
#
#   Spring reference:  (l,b) ~ (0.50, -1.40)
#   Autumn reference:  (l,b) ~ (0.35, -1.40)
#
# For the stellar-population sampling grid we use the midpoint
#
#   l = 0.425 deg
#
# rather than pretending that an older ROTAC center list is the
# exact current APT footprint.
#
# The center-to-center separation below comes from the five-field
# overguide layout (0.408974 deg). It provides a convenient
# equal-area surrogate for sampling extinction/population.
#
# THIS IS NOT A DETECTOR-LEVEL APT FOOTPRINT.
# ----------------------------------------------------------------

MAIN_CENTER_L = 0.425
MAIN_CENTER_B = -1.400
FIELD_SPACING_L = 0.408974

# GC reference from the ROTAC field layout; it places Sgr A* and
# the surrounding nuclear region inside the WFI footprint.
GC_L = 0.0
GC_B = -0.125

# Effective rectangle used only for population/extinction sampling.
#
# Width along l is the field spacing. Height is chosen so that
# width * height = 0.281 deg^2 exactly.
FIELD_WIDTH_L = FIELD_SPACING_L
FIELD_HEIGHT_B = WFI_AREA_DEG2 / FIELD_WIDTH_L

# 4 x 7 gives approximately 6 arcmin cells.
N_L = 4
N_B = 7

# Actual TRILEGAL area generated for each representative LOS.
SAMPLE_AREA_DEG2 = 1.0e-4

# Surot gives E(J-Ks), while TRILEGAL expects A_V.
#
# This is deliberately explicit because the conversion is an
# extinction-law assumption. 6.0 is approximately consistent
# with a CCM89 R_V=3.1 conversion in J/Ks.
AV_PER_EJK = 6.0

# TRILEGAL permits extinction_sigma <= 0.3.
MAX_TRILEGAL_EXTINCTION_SIGMA = 0.30

OUTDIR = Path("extinction")
RAW_DIR = OUTDIR / "raw"
PROCESSED_DIR = OUTDIR / "processed"

CONFIG_OUT = Path("config/gbtds_subfields.csv")


# ============================================================
# Geometry
# ============================================================

def build_field_table():
    rows = []

    offsets = np.arange(-2, 3) * FIELD_SPACING_L

    for i, offset in enumerate(offsets, start=1):
        rows.append(
            {
                "parent_field": f"gbtds_{i}",
                "field_number": i,
                "l_deg": MAIN_CENTER_L + offset,
                "b_deg": MAIN_CENTER_B,
                "field_area_deg2": WFI_AREA_DEG2,
                "geometry_role": "main",
            }
        )

    rows.append(
        {
            "parent_field": "gbtds_gc",
            "field_number": 6,
            "l_deg": GC_L,
            "b_deg": GC_B,
            "field_area_deg2": WFI_AREA_DEG2,
            "geometry_role": "galactic_center",
        }
    )

    return pd.DataFrame(rows)


def build_cells(fields):
    rows = []

    dl = FIELD_WIDTH_L / N_L
    db = FIELD_HEIGHT_B / N_B

    target_cell_area = WFI_AREA_DEG2 / (N_L * N_B)

    for _, field in fields.iterrows():

        l0 = field["l_deg"]
        b0 = field["b_deg"]

        l_min_field = l0 - FIELD_WIDTH_L / 2
        b_min_field = b0 - FIELD_HEIGHT_B / 2

        for ib in range(N_B):
            for il in range(N_L):

                lmin = l_min_field + il * dl
                lmax = lmin + dl

                bmin = b_min_field + ib * db
                bmax = bmin + db

                lc = 0.5 * (lmin + lmax)
                bc = 0.5 * (bmin + bmax)

                cell_id = (
                    f"{field['parent_field']}"
                    f"_b{ib:02d}_l{il:02d}"
                )

                rows.append(
                    {
                        "field_id": cell_id,
                        "parent_field": field["parent_field"],
                        "field_number": int(field["field_number"]),
                        "geometry_role": field["geometry_role"],

                        "l_deg": lc,
                        "b_deg": bc,

                        "l_min_deg": lmin,
                        "l_max_deg": lmax,
                        "b_min_deg": bmin,
                        "b_max_deg": bmax,

                        "target_area_deg2": target_cell_area,
                        "sample_area_deg2": SAMPLE_AREA_DEG2,

                        "area_weight": (
                            target_cell_area /
                            SAMPLE_AREA_DEG2
                        ),
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# VizieR / Surot map
# ============================================================

def query_surot_field(field):
    """
    Download the Surot+2020 map nodes covering one effective
    WFI sampling rectangle.
    """

    field_id = field["parent_field"]

    cache = RAW_DIR / f"surot2020_{field_id}.parquet"

    if cache.exists():
        print(f"  using cache: {cache}")
        return pd.read_parquet(cache)

    # Small margin around the surrogate footprint.
    margin = 0.02

    lmin = field["l_deg"] - FIELD_WIDTH_L / 2 - margin
    lmax = field["l_deg"] + FIELD_WIDTH_L / 2 + margin

    bmin = field["b_deg"] - FIELD_HEIGHT_B / 2 - margin
    bmax = field["b_deg"] + FIELD_HEIGHT_B / 2 + margin

    print(
        f"  querying VizieR: "
        f"l=[{lmin:.4f},{lmax:.4f}], "
        f"b=[{bmin:.4f},{bmax:.4f}]"
    )

    viz = Vizier(
        columns=[
            "GLON",
            "GLAT",
            "E(J-Ks)",
            "e_E(J-Ks)",
            "res-avg",
            "res-max",
            "Tile",
        ],
        row_limit=-1,
    )

    Vizier.TIMEOUT = 600

    result = viz.query_constraints(
        catalog=CATALOG,
        GLON=f"{lmin}..{lmax}",
        GLAT=f"{bmin}..{bmax}",
    )

    if len(result) == 0:
        raise RuntimeError(
            f"No Surot map values returned for {field_id}"
        )

    table = result[0]

    df = table.to_pandas()

    needed = [
        "GLON",
        "GLAT",
        "E(J-Ks)",
        "e_E(J-Ks)",
        "res-avg",
        "res-max",
    ]

    missing = [c for c in needed if c not in df.columns]

    if missing:
        print("Returned columns:")
        print(list(df.columns))

        raise RuntimeError(
            f"Missing expected VizieR columns: {missing}"
        )

    for c in needed:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df = df.dropna(
        subset=["GLON", "GLAT", "E(J-Ks)"]
    )

    cache.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        cache,
        index=False,
        compression="zstd",
    )

    print(
        f"  downloaded {len(df):,} map nodes"
    )

    return df


# ============================================================
# Extinction statistics per subfield
# ============================================================

def cell_extinction_statistics(cell, dust):
    mask = (
        (dust["GLON"] >= cell["l_min_deg"])
        & (dust["GLON"] < cell["l_max_deg"])
        & (dust["GLAT"] >= cell["b_min_deg"])
        & (dust["GLAT"] < cell["b_max_deg"])
    )

    sample = dust.loc[mask]

    if len(sample) == 0:
        return {
            "n_extinction_nodes": 0,
            "ejk_median": np.nan,
            "ejk_p16": np.nan,
            "ejk_p84": np.nan,
            "ejk_sigma_spatial": np.nan,
            "ejk_measurement_error_median": np.nan,
            "map_res_avg_arcmin": np.nan,
            "map_res_max_arcmin": np.nan,
        }

    ejk = sample["E(J-Ks)"].to_numpy()

    p16, med, p84 = np.nanpercentile(
        ejk,
        [16, 50, 84],
    )

    spatial_sigma = 0.5 * (p84 - p16)

    return {
        "n_extinction_nodes": len(sample),

        "ejk_median": med,
        "ejk_p16": p16,
        "ejk_p84": p84,
        "ejk_sigma_spatial": spatial_sigma,

        "ejk_measurement_error_median": (
            sample["e_E(J-Ks)"].median()
        ),

        "map_res_avg_arcmin": (
            sample["res-avg"].median()
        ),

        "map_res_max_arcmin": (
            sample["res-max"].max()
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONFIG_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = build_field_table()
    cells = build_cells(fields)

    print()
    print("=======================================")
    print("GBTDS spatial sampling geometry")
    print("=======================================")
    print()
    print(fields.to_string(index=False))

    print()
    print(f"WFI area             = {WFI_AREA_DEG2:.6f} deg^2")
    print(f"Effective width      = {FIELD_WIDTH_L:.6f} deg")
    print(f"Effective height     = {FIELD_HEIGHT_B:.6f} deg")
    print(f"Cells per WFI        = {N_L * N_B}")
    print(f"Total cells          = {len(cells)}")
    print(
        "Target area / cell   = "
        f"{cells.target_area_deg2.iloc[0]:.8f} deg^2"
    )
    print(
        "TRILEGAL sample area = "
        f"{SAMPLE_AREA_DEG2:.8f} deg^2"
    )
    print(
        "Area weight / cell   = "
        f"{cells.area_weight.iloc[0]:.3f}"
    )
    print(
        "Total represented area = "
        f"{cells.target_area_deg2.sum():.6f} deg^2"
    )

    all_results = []

    for _, field in fields.iterrows():

        print()
        print("=======================================")
        print(field["parent_field"])
        print("=======================================")

        dust = query_surot_field(field)

        field_cells = cells[
            cells["parent_field"]
            == field["parent_field"]
        ].copy()

        stats = []

        for _, cell in field_cells.iterrows():
            stats.append(
                cell_extinction_statistics(
                    cell,
                    dust,
                )
            )

        stats = pd.DataFrame(stats)

        field_cells = pd.concat(
            [
                field_cells.reset_index(drop=True),
                stats.reset_index(drop=True),
            ],
            axis=1,
        )

        all_results.append(field_cells)

        # Do not hammer VizieR.
        time.sleep(1)

    out = pd.concat(
        all_results,
        ignore_index=True,
    )

    if out["ejk_median"].isna().any():
        bad = out.loc[
            out["ejk_median"].isna(),
            [
                "field_id",
                "l_deg",
                "b_deg",
            ],
        ]

        raise RuntimeError(
            "Some cells have no Surot extinction nodes:\n"
            + bad.to_string(index=False)
        )

    # --------------------------------------------------------
    # Convert spatial reddening map into TRILEGAL parameters.
    # --------------------------------------------------------

    out["av_per_ejk"] = AV_PER_EJK

    out["av_inf"] = (
        AV_PER_EJK * out["ejk_median"]
    )

    out["av_p16"] = (
        AV_PER_EJK * out["ejk_p16"]
    )

    out["av_p84"] = (
        AV_PER_EJK * out["ejk_p84"]
    )

    out["av_sigma_spatial"] = (
        AV_PER_EJK *
        out["ejk_sigma_spatial"]
    )

    relative_sigma = (
        out["ejk_sigma_spatial"]
        / out["ejk_median"]
    )

    out["extinction_sigma_raw"] = relative_sigma

    out["extinction_sigma"] = (
        relative_sigma
        .fillna(0.0)
        .clip(
            lower=0.0,
            upper=MAX_TRILEGAL_EXTINCTION_SIGMA,
        )
    )

    out["extinction_sigma_was_capped"] = (
        out["extinction_sigma_raw"]
        > MAX_TRILEGAL_EXTINCTION_SIGMA
    )

    # Flag cells where a 6' cell may be too coarse.
    #
    # 0.10 mag in E(J-Ks) corresponds to roughly
    # 0.6 mag in Av with the current conversion.
    out["needs_refinement"] = (
        out["ejk_sigma_spatial"] > 0.10
    )

    out["extinction_map"] = "Surot2020_VVVEXMAP"
    out["extinction_catalog"] = CATALOG
    out["geometry_version"] = (
        "2026_effective_sampling_grid_v1"
    )

    # --------------------------------------------------------
    # Save full diagnostic table
    # --------------------------------------------------------

    full_path = (
        PROCESSED_DIR
        / "gbtds_subfields_extinction.parquet"
    )

    out.to_parquet(
        full_path,
        index=False,
        compression="zstd",
    )

    # --------------------------------------------------------
    # CSV consumed by TRILEGAL downloader
    # --------------------------------------------------------

    config_columns = [
        "field_id",
        "parent_field",
        "field_number",
        "geometry_role",

        "l_deg",
        "b_deg",

        "target_area_deg2",
        "sample_area_deg2",
        "area_weight",

        "av_inf",
        "extinction_sigma",

        "ejk_median",
        "ejk_sigma_spatial",
        "ejk_p16",
        "ejk_p84",

        "n_extinction_nodes",
        "map_res_avg_arcmin",
        "map_res_max_arcmin",

        "needs_refinement",
        "extinction_sigma_was_capped",

        "extinction_map",
        "extinction_catalog",
        "av_per_ejk",
        "geometry_version",
    ]

    out[
        config_columns
    ].to_csv(
        CONFIG_OUT,
        index=False,
    )

    print()
    print("=======================================")
    print("RESULT")
    print("=======================================")

    print()
    print("Per parent field:")

    summary = (
        out.groupby("parent_field")
        .agg(
            n_cells=("field_id", "size"),
            ejk_min=("ejk_median", "min"),
            ejk_med=("ejk_median", "median"),
            ejk_max=("ejk_median", "max"),
            av_min=("av_inf", "min"),
            av_med=("av_inf", "median"),
            av_max=("av_inf", "max"),
            n_refine=("needs_refinement", "sum"),
            n_sigma_capped=(
                "extinction_sigma_was_capped",
                "sum",
            ),
        )
    )

    print(summary.to_string())

    print()
    print("Global E(J-Ks):")
    print(
        out["ejk_median"]
        .describe()
        .to_string()
    )

    print()
    print("Global Av normalization:")
    print(
        out["av_inf"]
        .describe()
        .to_string()
    )

    print()
    print(
        "Cells flagged for spatial refinement:",
        int(out["needs_refinement"].sum()),
        "/",
        len(out),
    )

    print(
        "Cells hitting TRILEGAL sigma cap:",
        int(
            out[
                "extinction_sigma_was_capped"
            ].sum()
        ),
        "/",
        len(out),
    )

    print()
    print("Saved:")
    print(" ", full_path)
    print(" ", CONFIG_OUT)

    print()
    print("First 12 subfields:")
    print(
        out[
            [
                "field_id",
                "l_deg",
                "b_deg",
                "ejk_median",
                "ejk_sigma_spatial",
                "av_inf",
                "extinction_sigma",
                "needs_refinement",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
