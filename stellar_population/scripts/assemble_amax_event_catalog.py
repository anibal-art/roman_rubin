#!/usr/bin/env python3

"""
Assemble the complete Roman-Rubin precomputed-event catalogue.

This script does NOT compute Amax and does NOT modify any scientific
parameter. It only assembles already-materialized per-field catalogues.

Required input invariant
------------------------
Every input event parquet must already contain:

    simulate_amax

which is the permissive necessary-condition flag produced by the
catalog Amax machinery.

For materialized event realizations we also require:

    blend_ratio_W149
    blend_ratio_u
    blend_ratio_g
    blend_ratio_r
    blend_ratio_i
    blend_ratio_z
    blend_ratio_y

and, for binary-lens/USBL events:

    caustic_origin

Outputs
-------
1. one unified catalogue containing FFP + BH + binary-lens events;
2. one catalogue per population;
3. a JSON manifest with provenance and counts.

No input parquet is modified.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_ROOT = (
    REPO_ROOT
    / "stellar_population"
    / "precomputed_events"
    / "production"
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "stellar_population"
    / "precomputed_events"
    / "assembled"
)


POPULATIONS = {
    "FFP": {
        "filename": "ffp_events.parquet",
        "system_key": "ffp",
    },
    "BH": {
        "filename": "bh_events.parquet",
        "system_key": "bh",
    },
    "Planets_systems": {
        "filename": "binary_lens_events.parquet",
        "system_key": "binary_lens",
    },
}


POPULATION_ORDER = {
    "FFP": 0,
    "BH": 1,
    "Planets_systems": 2,
}


BLEND_BANDS = (
    "W149",
    "u",
    "g",
    "r",
    "i",
    "z",
    "y",
)


REQUIRED_BLEND_COLUMNS = tuple(
    f"blend_ratio_{band}"
    for band in BLEND_BANDS
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the complete FFP/BH/binary-lens event catalogue "
            "with materialized Amax simulation flags."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root containing worker production directories.",
    )

    parser.add_argument(
        "--run-glob",
        default="gbtds_300_v1_w*",
        help=(
            "Glob selecting worker production directories. "
            "Default: gbtds_300_v1_w*"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where assembled catalogues are written.",
    )

    parser.add_argument(
        "--output-tag",
        default="gbtds_300_v1_amax",
        help="Name of the assembled catalogue directory.",
    )

    parser.add_argument(
        "--expected-fields",
        type=int,
        default=510,
        help="Expected number of unique sky fields.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit inputs but do not write output files.",
    )

    return parser.parse_args()


def worker_sort_key(path: Path):
    match = re.search(r"_w(\d+)$", path.name)

    if match is None:
        return (10**9, path.name)

    return (int(match.group(1)), path.name)


def discover_worker_roots(input_root: Path, run_glob: str):
    roots = sorted(
        (
            path
            for path in input_root.glob(run_glob)
            if path.is_dir()
        ),
        key=worker_sort_key,
    )

    if not roots:
        raise RuntimeError(
            f"No worker directories matched:\n"
            f"  root = {input_root}\n"
            f"  glob = {run_glob}"
        )

    return roots


def worker_index(worker_root: Path):
    match = re.search(r"_w(\d+)$", worker_root.name)

    if match is None:
        raise ValueError(
            f"Cannot infer worker number from {worker_root.name!r}"
        )

    return int(match.group(1))


def discover_event_files(worker_roots):
    """
    Discover actual event-catalogue shards.

    A directory is considered a candidate field only if it contains at
    least one of the expected population parquet files. This deliberately
    ignores empty/stale directories left by interrupted or previous runs.

    Once a directory is identified as a candidate field, all three
    population catalogues are mandatory.
    """

    records = []

    expected_filenames = {
        config["filename"]
        for config in POPULATIONS.values()
    }

    for worker_root in worker_roots:
        w = worker_index(worker_root)

        # Do NOT assume every directory below worker_root is a completed
        # field. Discover fields from actual event parquet files.
        candidate_field_dirs = set()

        for filename in expected_filenames:
            for path in worker_root.glob(
                f"*/{filename}"
            ):
                candidate_field_dirs.add(
                    path.parent
                )

        field_dirs = sorted(
            candidate_field_dirs,
            key=lambda path: path.name,
        )

        for field_dir in field_dirs:
            field_id = field_dir.name

            missing = []

            for population, config in POPULATIONS.items():
                path = (
                    field_dir
                    / config["filename"]
                )

                if not path.exists():
                    missing.append(
                        (
                            population,
                            path,
                        )
                    )

            if missing:
                details = "\n".join(
                    f"    {population}: {path}"
                    for population, path in missing
                )

                raise FileNotFoundError(
                    "Incomplete event-catalogue field:\n"
                    f"  worker = {worker_root.name}\n"
                    f"  field  = {field_id}\n"
                    f"  missing:\n{details}"
                )

            for population, config in POPULATIONS.items():
                path = (
                    field_dir
                    / config["filename"]
                )

                records.append(
                    {
                        "worker": w,
                        "worker_tag": worker_root.name,
                        "field_id_from_path": field_id,
                        "population": population,
                        "system_key_expected": config["system_key"],
                        "path": path,
                    }
                )

    if not records:
        raise RuntimeError(
            "No event-catalogue shards were discovered."
        )

    return pd.DataFrame(records)


def validate_inventory(inventory: pd.DataFrame, expected_fields: int):
    fields = sorted(
        inventory["field_id_from_path"].unique()
    )

    if len(fields) != expected_fields:
        raise RuntimeError(
            "Unexpected number of unique fields:\n"
            f"  found    = {len(fields)}\n"
            f"  expected = {expected_fields}"
        )

    expected_populations = set(POPULATIONS)

    for field_id, group in inventory.groupby(
        "field_id_from_path",
        sort=False,
    ):
        found = set(group["population"])

        if found != expected_populations:
            raise RuntimeError(
                f"Incomplete population set for field {field_id}:\n"
                f"  found    = {sorted(found)}\n"
                f"  expected = {sorted(expected_populations)}"
            )

    duplicates = inventory.duplicated(
        subset=[
            "field_id_from_path",
            "population",
        ],
        keep=False,
    )

    if duplicates.any():
        raise RuntimeError(
            "Duplicate field/population shards found:\n"
            + inventory.loc[
                duplicates,
                [
                    "worker_tag",
                    "field_id_from_path",
                    "population",
                    "path",
                ],
            ].to_string(index=False)
        )


def validate_materialized_columns(
    df: pd.DataFrame,
    population: str,
    path: Path,
):
    required = {
        "simulate_amax",
        "event_seed",
        *REQUIRED_BLEND_COLUMNS,
    }

    missing = sorted(
        required.difference(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{path} is not a complete materialized Amax catalogue.\n"
            f"Missing columns: {missing}"
        )

    if population == "Planets_systems":
        if "caustic_origin" not in df.columns:
            raise RuntimeError(
                f"{path} is a binary-lens catalogue but "
                "has no caustic_origin column."
            )

    if df["simulate_amax"].isna().any():
        raise RuntimeError(
            f"{path}: simulate_amax contains NaN."
        )

    values = set(
        df["simulate_amax"]
        .astype(bool)
        .unique()
        .tolist()
    )

    if not values.issubset({True, False}):
        raise RuntimeError(
            f"{path}: invalid simulate_amax values: {values}"
        )

    for column in REQUIRED_BLEND_COLUMNS:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"{path}: non-finite values in {column}"
            )

        if np.any(values < 0):
            raise RuntimeError(
                f"{path}: negative values in {column}"
            )

    if population == "Planets_systems":
        allowed = {
            "central_caustic",
            "second_caustic",
            "third_caustic",
        }

        actual = set(
            df["caustic_origin"]
            .astype(str)
            .unique()
        )

        invalid = actual.difference(allowed)

        if invalid:
            raise RuntimeError(
                f"{path}: invalid caustic_origin values: "
                f"{sorted(invalid)}"
            )


def load_one_shard(record):
    path = Path(record["path"])
    population = record["population"]

    df = pd.read_parquet(path)

    validate_materialized_columns(
        df,
        population,
        path,
    )

    field_id = record["field_id_from_path"]

    if "field_id" in df.columns:
        actual_fields = set(
            df["field_id"]
            .astype(str)
            .unique()
        )

        if actual_fields != {field_id}:
            raise RuntimeError(
                f"{path}: field_id mismatch.\n"
                f"  directory = {field_id}\n"
                f"  parquet   = {sorted(actual_fields)}"
            )
    else:
        df["field_id"] = field_id

    if "system_key" in df.columns:
        actual_system_keys = set(
            df["system_key"]
            .dropna()
            .astype(str)
            .unique()
        )

        expected = {
            record["system_key_expected"]
        }

        if actual_system_keys and actual_system_keys != expected:
            raise RuntimeError(
                f"{path}: unexpected system_key values.\n"
                f"  found    = {sorted(actual_system_keys)}\n"
                f"  expected = {sorted(expected)}"
            )

    # Explicit assembly provenance.
    df["catalog_population"] = population
    df["source_worker"] = int(record["worker"])
    df["source_worker_tag"] = record["worker_tag"]
    df["source_field_id"] = field_id
    df["source_filename"] = path.name

    # Preserve the original validated flag as the scientific authority.
    df["simulate_amax"] = df["simulate_amax"].astype(bool)

    # Local row number is useful for exact provenance back to the shard.
    df["source_row"] = np.arange(
        len(df),
        dtype=np.int64,
    )

    return df


def check_duplicate_events(df: pd.DataFrame):
    key = [
        "catalog_population",
        "source_field_id",
        "event_seed",
    ]

    duplicate = df.duplicated(
        subset=key,
        keep=False,
    )

    if duplicate.any():
        bad = df.loc[
            duplicate,
            key
            + [
                "source_worker_tag",
                "source_row",
            ],
        ]

        raise RuntimeError(
            "Duplicate materialized events found:\n"
            + bad.head(100).to_string(index=False)
        )


def make_summary(df: pd.DataFrame):
    rows = []

    for population, group in df.groupby(
        "catalog_population",
        sort=False,
    ):
        n = len(group)

        n_keep = int(
            group["simulate_amax"].sum()
        )

        n_reject = n - n_keep

        rows.append(
            {
                "population": population,
                "rows": int(n),
                "fields": int(
                    group["source_field_id"].nunique()
                ),
                "simulate_amax_true": n_keep,
                "simulate_amax_false": n_reject,
                "simulate_amax_fraction": (
                    float(n_keep / n)
                    if n
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    input_root = args.input_root.resolve()
    output_dir = (
        args.output_root.resolve()
        / args.output_tag
    )

    print("=" * 78)
    print("ASSEMBLE MATERIALIZED Amax EVENT CATALOGUE")
    print("=" * 78)
    print("input root :", input_root)
    print("run glob   :", args.run_glob)
    print("output dir :", output_dir)
    print("dry run    :", args.dry_run)

    worker_roots = discover_worker_roots(
        input_root,
        args.run_glob,
    )

    print()
    print("workers:")
    for path in worker_roots:
        print(" ", path.name)

    inventory = discover_event_files(
        worker_roots
    )

    validate_inventory(
        inventory,
        expected_fields=args.expected_fields,
    )

    print()
    print(
        "unique fields =",
        inventory["field_id_from_path"].nunique(),
    )
    print(
        "input shards  =",
        len(inventory),
    )

    frames = []

    inventory = inventory.sort_values(
        by=[
            "worker",
            "field_id_from_path",
            "population",
        ],
        key=lambda col: (
            col.map(POPULATION_ORDER)
            if col.name == "population"
            else col
        ),
        kind="stable",
    ).reset_index(drop=True)

    total = len(inventory)

    for i, record in inventory.iterrows():
        print(
            f"[{i + 1:4d}/{total:4d}]",
            f"w{record['worker']}",
            record["field_id_from_path"],
            record["population"],
        )

        df = load_one_shard(record)

        frames.append(df)

    catalog = pd.concat(
        frames,
        axis=0,
        ignore_index=True,
        sort=False,
    )

    del frames

    check_duplicate_events(catalog)

    # Deterministic global index of this assembled catalogue.
    catalog.insert(
        0,
        "catalog_event_index",
        np.arange(
            len(catalog),
            dtype=np.int64,
        ),
    )

    summary = make_summary(catalog)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(summary.to_string(index=False))

    print()
    print("total rows =", len(catalog))
    print(
        "total simulate_amax=True =",
        int(catalog["simulate_amax"].sum()),
    )
    print(
        "total simulate_amax=False =",
        int((~catalog["simulate_amax"]).sum()),
    )

    if args.dry_run:
        print()
        print("DRY RUN: no files written.")
        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # One single catalogue: all populations.
    # --------------------------------------------------------

    unified_path = (
        output_dir
        / "events_all_populations_amax.parquet"
    )

    catalog.to_parquet(
        unified_path,
        index=False,
        compression="zstd",
    )

    # --------------------------------------------------------
    # Population-specific views.
    # These contain exactly the same rows/columns as the unified file.
    # --------------------------------------------------------

    population_paths = {}

    filenames = {
        "FFP": "ffp_events_amax.parquet",
        "BH": "bh_events_amax.parquet",
        "Planets_systems": "binary_lens_events_amax.parquet",
    }

    for population, filename in filenames.items():
        subset = catalog[
            catalog["catalog_population"]
            == population
        ].copy()

        path = output_dir / filename

        subset.to_parquet(
            path,
            index=False,
            compression="zstd",
        )

        population_paths[population] = str(path)

    summary_path = (
        output_dir
        / "catalog_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    manifest = {
        "output_tag": args.output_tag,
        "input_root": str(input_root),
        "run_glob": args.run_glob,
        "worker_roots": [
            str(path)
            for path in worker_roots
        ],
        "expected_fields": int(
            args.expected_fields
        ),
        "actual_fields": int(
            catalog["source_field_id"].nunique()
        ),
        "input_shards": int(
            len(inventory)
        ),
        "total_rows": int(
            len(catalog)
        ),
        "flag_definition": {
            "column": "simulate_amax",
            "meaning": (
                "True means the event passes the permissive "
                "Amax photometric necessary-condition prefilter; "
                "False means it can be rejected before full "
                "Roman/Rubin simulation under that criterion."
            ),
        },
        "unified_catalog": str(
            unified_path
        ),
        "population_catalogs": (
            population_paths
        ),
        "summary": (
            summary
            .replace({np.nan: None})
            .to_dict(orient="records")
        ),
    }

    manifest_path = (
        output_dir
        / "catalog_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print("=" * 78)
    print("WRITTEN")
    print("=" * 78)
    print("unified :", unified_path)

    for population, path in population_paths.items():
        print(
            f"{population:16s}:",
            path,
        )

    print("summary :", summary_path)
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
