#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT = Path(
    "extinction/processed/"
    "gbtds_subfields_extinction.parquet"
)

RAW_DIR = Path("extinction/raw")

OUTPUT_PARQUET = Path(
    "extinction/processed/"
    "gbtds_subfields_extinction_adaptive.parquet"
)

OUTPUT_CSV = Path(
    "config/gbtds_subfields_adaptive.csv"
)

DIAGNOSTIC_CSV = Path(
    "diagnostics/"
    "gbtds_extinction_refinement_summary.csv"
)

# Same definitions as the base grid.
SAMPLE_AREA_DEG2 = 1.0e-4
AV_PER_EJK = 6.0

# Refine a cell when its robust spatial scatter in E(J-Ks)
# exceeds this value.
EJK_SIGMA_THRESHOLD = 0.10

MAX_TRILEGAL_EXTINCTION_SIGMA = 0.30


# ============================================================
# Extinction statistics
# ============================================================

def extinction_statistics(cell, dust):
    mask = (
        (dust["GLON"] >= cell["l_min_deg"])
        & (dust["GLON"] < cell["l_max_deg"])
        & (dust["GLAT"] >= cell["b_min_deg"])
        & (dust["GLAT"] < cell["b_max_deg"])
    )

    sample = dust.loc[mask]

    if len(sample) == 0:
        raise RuntimeError(
            "No Surot nodes inside cell "
            f"{cell['field_id']}"
        )

    ejk = sample["E(J-Ks)"].to_numpy(dtype=float)

    p16, median, p84 = np.nanpercentile(
        ejk,
        [16, 50, 84],
    )

    sigma_spatial = 0.5 * (p84 - p16)

    result = {
        "n_extinction_nodes": len(sample),

        "ejk_median": median,
        "ejk_p16": p16,
        "ejk_p84": p84,
        "ejk_sigma_spatial": sigma_spatial,

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

    return result


# ============================================================
# Split one cell 2 x 2
# ============================================================

def split_cell(cell):
    lmid = 0.5 * (
        cell["l_min_deg"]
        + cell["l_max_deg"]
    )

    bmid = 0.5 * (
        cell["b_min_deg"]
        + cell["b_max_deg"]
    )

    l_edges = [
        cell["l_min_deg"],
        lmid,
        cell["l_max_deg"],
    ]

    b_edges = [
        cell["b_min_deg"],
        bmid,
        cell["b_max_deg"],
    ]

    children = []

    for ib in range(2):
        for il in range(2):
            lmin = l_edges[il]
            lmax = l_edges[il + 1]

            bmin = b_edges[ib]
            bmax = b_edges[ib + 1]

            child = cell.copy()

            child_id = (
                f"{cell['field_id']}"
                f"_r1_b{ib}_l{il}"
            )

            child["parent_cell_id"] = (
                cell["field_id"]
            )

            child["field_id"] = child_id
            child["refinement_level"] = 1

            child["l_min_deg"] = lmin
            child["l_max_deg"] = lmax
            child["b_min_deg"] = bmin
            child["b_max_deg"] = bmax

            child["l_deg"] = 0.5 * (
                lmin + lmax
            )

            child["b_deg"] = 0.5 * (
                bmin + bmax
            )

            # Equal-area subdivision.
            child["target_area_deg2"] = (
                cell["target_area_deg2"] / 4.0
            )

            child["sample_area_deg2"] = (
                SAMPLE_AREA_DEG2
            )

            child["area_weight"] = (
                child["target_area_deg2"]
                / SAMPLE_AREA_DEG2
            )

            children.append(child)

    return children


# ============================================================
# Add derived TRILEGAL extinction quantities
# ============================================================

def add_extinction_quantities(row):
    row = row.copy()

    row["av_per_ejk"] = AV_PER_EJK

    row["av_inf"] = (
        AV_PER_EJK
        * row["ejk_median"]
    )

    row["av_p16"] = (
        AV_PER_EJK
        * row["ejk_p16"]
    )

    row["av_p84"] = (
        AV_PER_EJK
        * row["ejk_p84"]
    )

    row["av_sigma_spatial"] = (
        AV_PER_EJK
        * row["ejk_sigma_spatial"]
    )

    if row["ejk_median"] > 0:
        rel_sigma = (
            row["ejk_sigma_spatial"]
            / row["ejk_median"]
        )
    else:
        rel_sigma = 0.0

    row["extinction_sigma_raw"] = rel_sigma

    row["extinction_sigma"] = np.clip(
        rel_sigma,
        0.0,
        MAX_TRILEGAL_EXTINCTION_SIGMA,
    )

    row["extinction_sigma_was_capped"] = (
        rel_sigma
        > MAX_TRILEGAL_EXTINCTION_SIGMA
    )

    row["needs_refinement"] = (
        row["ejk_sigma_spatial"]
        > EJK_SIGMA_THRESHOLD
    )

    return row


# ============================================================
# Main
# ============================================================

def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    base = pd.read_parquet(INPUT)

    print()
    print("=======================================")
    print("Adaptive GBTDS extinction grid")
    print("=======================================")
    print()

    print("Base cells:")
    print(" ", len(base))

    print(
        "Flagged base cells:",
        int(base["needs_refinement"].sum()),
    )

    print(
        "Base represented area:",
        base["target_area_deg2"].sum(),
        "deg^2",
    )

    # --------------------------------------------------------
    # Load each Surot cache once.
    # --------------------------------------------------------

    dust_cache = {}

    for parent_field in sorted(
        base["parent_field"].unique()
    ):
        p = (
            RAW_DIR
            / f"surot2020_{parent_field}.parquet"
        )

        if not p.exists():
            raise FileNotFoundError(p)

        dust_cache[parent_field] = (
            pd.read_parquet(p)
        )

        print(
            f"{parent_field}: "
            f"{len(dust_cache[parent_field]):,} "
            "Surot nodes"
        )

    # --------------------------------------------------------
    # Keep good base cells, refine bad cells.
    # --------------------------------------------------------

    output_rows = []

    for _, original in base.iterrows():
        cell = original.copy()

        if not bool(cell["needs_refinement"]):
            cell["parent_cell_id"] = ""
            cell["refinement_level"] = 0

            output_rows.append(
                add_extinction_quantities(cell)
            )

            continue

        # ----------------------------------------------------
        # Split flagged cell into four children.
        # ----------------------------------------------------

        children = split_cell(cell)

        dust = dust_cache[
            cell["parent_field"]
        ]

        for child in children:
            stats = extinction_statistics(
                child,
                dust,
            )

            for key, value in stats.items():
                child[key] = value

            output_rows.append(
                add_extinction_quantities(child)
            )

    out = pd.DataFrame(output_rows)

    # --------------------------------------------------------
    # Basic integrity tests
    # --------------------------------------------------------

    expected_n = (
        len(base)
        - int(base["needs_refinement"].sum())
        + 4 * int(base["needs_refinement"].sum())
    )

    if len(out) != expected_n:
        raise RuntimeError(
            f"Expected {expected_n} cells, "
            f"got {len(out)}"
        )

    area_before = base[
        "target_area_deg2"
    ].sum()

    area_after = out[
        "target_area_deg2"
    ].sum()

    if not np.isclose(
        area_before,
        area_after,
        rtol=0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Area is not conserved: "
            f"{area_before} -> {area_after}"
        )

    if out["ejk_median"].isna().any():
        raise RuntimeError(
            "NaN extinction values remain."
        )

    # Metadata
    out["extinction_map"] = (
        "Surot2020_VVVEXMAP"
    )

    out["extinction_catalog"] = (
        "J/A+A/644/A140/ejkmap"
    )

    out["geometry_version"] = (
        "2026_effective_sampling_grid_adaptive_v1"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PARQUET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DIAGNOSTIC_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_parquet(
        OUTPUT_PARQUET,
        index=False,
        compression="zstd",
    )

    config_columns = [
        "field_id",
        "parent_cell_id",
        "parent_field",
        "field_number",
        "geometry_role",
        "refinement_level",

        "l_deg",
        "b_deg",

        "l_min_deg",
        "l_max_deg",
        "b_min_deg",
        "b_max_deg",

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
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    summary = (
        out.groupby("parent_field")
        .agg(
            n_cells=(
                "field_id",
                "size",
            ),

            n_level0=(
                "refinement_level",
                lambda x: int((x == 0).sum()),
            ),

            n_level1=(
                "refinement_level",
                lambda x: int((x == 1).sum()),
            ),

            remaining_refine=(
                "needs_refinement",
                "sum",
            ),

            ejk_min=(
                "ejk_median",
                "min",
            ),

            ejk_med=(
                "ejk_median",
                "median",
            ),

            ejk_max=(
                "ejk_median",
                "max",
            ),

            sigma_med=(
                "ejk_sigma_spatial",
                "median",
            ),

            sigma_max=(
                "ejk_sigma_spatial",
                "max",
            ),

            av_med=(
                "av_inf",
                "median",
            ),

            av_max=(
                "av_inf",
                "max",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        DIAGNOSTIC_CSV,
        index=False,
    )

    print()
    print("=======================================")
    print("ADAPTIVE GRID RESULT")
    print("=======================================")
    print()

    print(
        "Cells before:",
        len(base),
    )

    print(
        "Cells after: ",
        len(out),
    )

    print(
        "Level-0 cells:",
        int(
            (out["refinement_level"] == 0)
            .sum()
        ),
    )

    print(
        "Level-1 cells:",
        int(
            (out["refinement_level"] == 1)
            .sum()
        ),
    )

    print()
    print(
        "Area before:",
        f"{area_before:.9f}",
        "deg^2",
    )

    print(
        "Area after: ",
        f"{area_after:.9f}",
        "deg^2",
    )

    print()
    print(
        "Cells still above "
        f"sigma_EJK>{EJK_SIGMA_THRESHOLD}:",
        int(out["needs_refinement"].sum()),
        "/",
        len(out),
    )

    print()
    print("Per-field summary:")
    print(
        summary.to_string(index=False)
    )

    remaining = out[
        out["needs_refinement"]
    ].copy()

    if len(remaining):
        print()
        print(
            "Worst remaining cells "
            "after 3-arcmin refinement:"
        )

        cols = [
            "field_id",
            "parent_field",
            "l_deg",
            "b_deg",
            "ejk_median",
            "ejk_sigma_spatial",
            "av_inf",
            "extinction_sigma",
            "n_extinction_nodes",
        ]

        print(
            remaining
            .sort_values(
                "ejk_sigma_spatial",
                ascending=False,
            )[cols]
            .head(30)
            .to_string(index=False)
        )

    print()
    print("Saved:")
    print(" ", OUTPUT_PARQUET)
    print(" ", OUTPUT_CSV)
    print(" ", DIAGNOSTIC_CSV)


if __name__ == "__main__":
    main()
