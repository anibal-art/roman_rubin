#!/usr/bin/env python3
"""
Materialize the complete catalog realization and Amax prefilter.

Input:
    existing immutable physical event catalogs

Adds:
    blend_ratio_<band>
    fsource_<band>
    ftotal_<band>
    caustic_origin               [binary lens only]
    Amax / Amax diagnostic columns from catalog.amax
    simulate_amax

The original physical catalogs are never modified.

Scientific authorities:
    catalog.blending.blending_columns
    catalog.caustic_origin.choose_catalog_caustic_origin
    catalog.amax.build_detection_limits
    catalog.amax.annotate_catalog
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from catalog.amax import (
    annotate_catalog,
    build_detection_limits,
)
from catalog.blending import (
    BAND_MAG,
    blending_columns,
)
from catalog.caustic_origin import (
    CAUSTIC_ORIGINS,
    choose_catalog_caustic_origin,
)


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_ROOT = (
    ROOT
    / "stellar_population"
    / "precomputed_events"
    / "production"
)


FILES = {
    "ffp_events.parquet": "FFP",
    "bh_events.parquet": "BH",
    "binary_lens_events.parquet": "Planets_systems",
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
    )

    p.add_argument(
        "--run-glob",
        default="gbtds_300_v1_w*",
    )

    p.add_argument(
        "--output-label",
        default="amax_v1",
        help=(
            "Inserted before _wN. Example: "
            "gbtds_300_v1_w0 -> "
            "gbtds_300_v1_amax_v1_w0"
        ),
    )

    p.add_argument(
        "--rubin-opsim-db",
        required=True,
        type=Path,
    )

    p.add_argument(
        "--margin-mag",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--expected-fields",
        type=int,
        default=510,
        help="0 disables the field-count assertion.",
    )

    p.add_argument(
        "--max-fields",
        type=int,
        default=0,
        help="0 processes every discovered field.",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and validate but do not write.",
    )

    return p.parse_args()


def worker_number(run_tag: str) -> int:
    m = re.search(r"_w(\d+)$", run_tag)

    if m is None:
        raise RuntimeError(
            f"Cannot infer worker number from {run_tag!r}"
        )

    return int(m.group(1))


def output_run_tag(
    input_run_tag: str,
    output_label: str,
) -> str:
    m = re.fullmatch(
        r"(.+)_w(\d+)",
        input_run_tag,
    )

    if m is None:
        raise RuntimeError(
            f"Unexpected run tag: {input_run_tag!r}"
        )

    prefix = m.group(1)
    worker = m.group(2)

    return (
        f"{prefix}_{output_label}_w{worker}"
    )


def discover_fields(
    input_root: Path,
    run_glob: str,
):
    workers = sorted(
        (
            p
            for p in input_root.glob(run_glob)
            if p.is_dir()
        ),
        key=lambda p: worker_number(p.name),
    )

    if not workers:
        raise RuntimeError(
            f"No workers matched {run_glob!r}"
        )

    fields = []

    for worker_root in workers:
        binary_paths = sorted(
            worker_root.glob(
                "*/binary_lens_events.parquet"
            )
        )

        for binary_path in binary_paths:
            field_dir = binary_path.parent

            missing = [
                filename
                for filename in FILES
                if not (
                    field_dir
                    / filename
                ).exists()
            ]

            if missing:
                raise RuntimeError(
                    f"Incomplete field {field_dir}:\n"
                    f"missing = {missing}"
                )

            fields.append(
                field_dir
            )

    # Exact duplicate field protection.
    field_ids = [
        p.name
        for p in fields
    ]

    if len(field_ids) != len(set(field_ids)):
        duplicates = sorted(
            {
                x
                for x in field_ids
                if field_ids.count(x) > 1
            }
        )

        raise RuntimeError(
            "Duplicate fields across workers: "
            f"{duplicates}"
        )

    return fields


def assert_original_columns_unchanged(
    original: pd.DataFrame,
    updated: pd.DataFrame,
    source: Path,
):
    cols = list(original.columns)

    if not updated.loc[:, cols].reset_index(
        drop=True
    ).equals(
        original.reset_index(drop=True)
    ):
        raise RuntimeError(
            "An original physical column changed:\n"
            f"{source}"
        )


def materialize_blending(
    df: pd.DataFrame,
    source: Path,
):
    forbidden = [
        c
        for c in df.columns
        if c.startswith(
            (
                "blend_ratio_",
                "fsource_",
                "ftotal_",
            )
        )
    ]

    if forbidden:
        raise RuntimeError(
            f"{source} already contains blending "
            f"columns: {forbidden}"
        )

    required = {
        "event_seed",
        *BAND_MAG.values(),
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise RuntimeError(
            f"{source}: missing blending inputs "
            f"{sorted(missing)}"
        )

    additions = blending_columns(df)

    expected = set()

    for band in BAND_MAG:
        expected.update(
            {
                f"blend_ratio_{band}",
                f"fsource_{band}",
                f"ftotal_{band}",
            }
        )

    missing_additions = (
        expected.difference(
            additions
        )
    )

    if missing_additions:
        raise RuntimeError(
            "blending_columns did not produce "
            f"{sorted(missing_additions)}"
        )

    out = df.copy()

    for name, values in additions.items():
        if name in out.columns:
            raise RuntimeError(
                f"Column collision: {name}"
            )

        out[name] = values

    # Freeze the current catalog realization contract.
    for band in BAND_MAG:
        g = np.asarray(
            out[f"blend_ratio_{band}"],
            dtype=float,
        )

        fs = np.asarray(
            out[f"fsource_{band}"],
            dtype=float,
        )

        ft = np.asarray(
            out[f"ftotal_{band}"],
            dtype=float,
        )

        if not np.all(
            np.isfinite(g)
        ):
            raise RuntimeError(
                f"{source}: non-finite g in {band}"
            )

        # Current catalog scheme:
        # independent_uniform_g_v1.
        if not np.all(
            (g >= 0.0)
            & (g <= 1.0)
        ):
            raise RuntimeError(
                f"{source}: invalid catalog g "
                f"in {band}"
            )

        if not np.allclose(
            ft,
            fs * (1.0 + g),
            rtol=1e-13,
            atol=0.0,
        ):
            raise RuntimeError(
                f"{source}: flux identity failed "
                f"in {band}"
            )

    return out


def materialize_origin(
    df: pd.DataFrame,
    population: str,
    source: Path,
):
    if population != "Planets_systems":
        return df

    if "event_seed" not in df.columns:
        raise RuntimeError(
            f"{source}: missing event_seed"
        )

    expected = np.asarray(
        [
            choose_catalog_caustic_origin(
                seed
            )
            for seed in df[
                "event_seed"
            ].to_numpy()
        ],
        dtype=object,
    )

    out = df.copy()

    if "caustic_origin" in out.columns:
        actual = (
            out["caustic_origin"]
            .astype(str)
            .to_numpy()
        )

        if not np.array_equal(
            actual,
            expected,
        ):
            raise RuntimeError(
                f"{source}: existing "
                "caustic_origin differs from "
                "catalog authority"
            )

        return out

    out["caustic_origin"] = (
        expected
    )

    actual_set = set(
        out["caustic_origin"]
        .astype(str)
        .unique()
    )

    invalid = (
        actual_set
        - set(CAUSTIC_ORIGINS)
    )

    if invalid:
        raise RuntimeError(
            f"{source}: invalid origins "
            f"{sorted(invalid)}"
        )

    return out


def atomic_write(
    df: pd.DataFrame,
    destination: Path,
):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        existing = pd.read_parquet(
            destination
        )

        if not existing.equals(
            df.reset_index(drop=True)
        ):
            raise RuntimeError(
                "Existing destination differs:\n"
                f"{destination}"
            )

        print(
            "  existing output identical"
        )
        return

    temporary = (
        destination.parent
        / (
            destination.stem
            + ".pending.parquet"
        )
    )

    temporary.unlink(
        missing_ok=True
    )

    df.to_parquet(
        temporary,
        index=False,
        compression="zstd",
    )

    verified = pd.read_parquet(
        temporary
    )

    if not verified.equals(
        df.reset_index(drop=True)
    ):
        temporary.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Parquet readback verification "
            f"failed:\n{destination}"
        )

    temporary.replace(
        destination
    )


def process_catalog(
    source: Path,
    destination: Path,
    population: str,
    limits,
    margin_mag: float,
    dry_run: bool,
):
    original = pd.read_parquet(
        source
    ).reset_index(drop=True)

    if "simulate_amax" in original.columns:
        raise RuntimeError(
            f"{source} already contains "
            "simulate_amax"
        )

    # --------------------------------------------------
    # 1. Materialized photometric realization
    # --------------------------------------------------

    materialized = (
        materialize_blending(
            original,
            source,
        )
    )

    # --------------------------------------------------
    # 2. Materialized CROIN origin for USBL
    # --------------------------------------------------

    materialized = (
        materialize_origin(
            materialized,
            population,
            source,
        )
    )

    assert_original_columns_unchanged(
        original,
        materialized,
        source,
    )

    pre_amax = (
        materialized.copy()
    )

    # --------------------------------------------------
    # 3. Amax authority
    # --------------------------------------------------

    out = annotate_catalog(
        pre_amax,
        population=population,
        limits=limits,
        margin_mag=margin_mag,
    ).reset_index(drop=True)

    # Amax is allowed to ADD columns only.
    assert_original_columns_unchanged(
        pre_amax,
        out,
        source,
    )

    if len(out) != len(original):
        raise RuntimeError(
            f"{source}: row count changed"
        )

    if "simulate_amax" not in out.columns:
        raise RuntimeError(
            "annotate_catalog did not produce "
            "simulate_amax"
        )

    if out["simulate_amax"].isna().any():
        raise RuntimeError(
            f"{source}: simulate_amax has NaN"
        )

    if not pd.api.types.is_bool_dtype(
        out["simulate_amax"]
    ):
        raise RuntimeError(
            f"{source}: simulate_amax is not "
            "boolean"
        )

    new_columns = [
        c
        for c in out.columns
        if c not in original.columns
    ]

    n_keep = int(
        out["simulate_amax"].sum()
    )

    n_reject = (
        len(out)
        - n_keep
    )

    print(
        f"  rows   = {len(out):,}"
    )
    print(
        f"  keep   = {n_keep:,} "
        f"({n_keep / len(out):.3%})"
    )
    print(
        f"  reject = {n_reject:,} "
        f"({n_reject / len(out):.3%})"
    )
    print(
        "  new columns =",
        new_columns,
    )

    if not dry_run:
        atomic_write(
            out,
            destination,
        )

        print(
            "  saved  =",
            destination,
        )

    return {
        "rows": len(out),
        "keep": n_keep,
        "reject": n_reject,
    }


def main():
    args = parse_args()

    input_root = (
        args.input_root
        .expanduser()
        .resolve()
    )

    opsim = (
        args.rubin_opsim_db
        .expanduser()
        .resolve()
    )

    if not opsim.exists():
        raise FileNotFoundError(
            opsim
        )

    fields = discover_fields(
        input_root,
        args.run_glob,
    )

    if (
        args.expected_fields
        and len(fields)
        != args.expected_fields
    ):
        raise RuntimeError(
            "Unexpected field count:\n"
            f"found    = {len(fields)}\n"
            f"expected = {args.expected_fields}"
        )

    if args.max_fields:
        fields = fields[
            :args.max_fields
        ]

    limits = build_detection_limits(
        opsim
    )

    print("=" * 78)
    print(
        "MATERIALIZE ROMAN-RUBIN "
        "AMAX EVENT CATALOGS"
    )
    print("=" * 78)
    print("input root :", input_root)
    print("run glob   :", args.run_glob)
    print("fields     :", len(fields))
    print("OpSim      :", opsim)
    print("margin mag :", args.margin_mag)
    print("dry run    :", args.dry_run)

    print()
    print("Detection limits:")

    for band, value in (
        limits.limits.items()
    ):
        print(
            f"  {band:5s} "
            f"{value:.6f}"
        )

    totals = {
        population: {
            "rows": 0,
            "keep": 0,
            "reject": 0,
        }
        for population in (
            FILES.values()
        )
    }

    for field_index, field_dir in enumerate(
        fields,
        start=1,
    ):
        input_run_tag = (
            field_dir.parent.name
        )

        output_tag = (
            output_run_tag(
                input_run_tag,
                args.output_label,
            )
        )

        destination_dir = (
            input_root
            / output_tag
            / field_dir.name
        )

        print()
        print("=" * 78)
        print(
            f"FIELD "
            f"{field_index}/{len(fields)}"
        )
        print(field_dir.name)
        print(
            f"{input_run_tag} "
            f"-> {output_tag}"
        )
        print("=" * 78)

        for filename, population in (
            FILES.items()
        ):
            source = (
                field_dir
                / filename
            )

            destination = (
                destination_dir
                / filename
            )

            print()
            print(
                population,
                ":",
                filename,
            )

            stats = process_catalog(
                source=source,
                destination=destination,
                population=population,
                limits=limits,
                margin_mag=args.margin_mag,
                dry_run=args.dry_run,
            )

            for key in (
                "rows",
                "keep",
                "reject",
            ):
                totals[
                    population
                ][key] += stats[key]

    print()
    print("=" * 78)
    print("TOTALS")
    print("=" * 78)

    for population, stats in (
        totals.items()
    ):
        rows = stats["rows"]
        keep = stats["keep"]

        frac = (
            keep / rows
            if rows
            else np.nan
        )

        print(
            f"{population:16s} "
            f"rows={rows:8,d} "
            f"keep={keep:8,d} "
            f"reject={stats['reject']:8,d} "
            f"keep_fraction={frac:.6f}"
        )

    if args.dry_run:
        print()
        print(
            "DRY RUN: no output files "
            "were written."
        )


if __name__ == "__main__":
    main()
