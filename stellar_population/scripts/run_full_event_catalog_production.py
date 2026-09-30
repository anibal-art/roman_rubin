#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_CELLS = (
    REPO_ROOT
    / "stellar_population/config/"
      "gbtds_trilegal_cells_production.csv"
)

DEFAULT_RUN_TAG = "gbtds_final_v1"

DEFAULT_F146_LIMIT = 28.0
DEFAULT_N_GENULENS = 5000
DEFAULT_SEED = 20260930
DEFAULT_MAX_DMU = 0.05

DEFAULT_T0_START = 2460413.013828608
DEFAULT_T0_END = (
    DEFAULT_T0_START
    + 365.25 * 8
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Resumable end-to-end production of the "
            "Roman+Rubin precomputed microlensing catalogues."
        )
    )

    p.add_argument(
        "--cells",
        type=Path,
        default=DEFAULT_CELLS,
    )

    p.add_argument(
        "--run-tag",
        default=DEFAULT_RUN_TAG,
    )

    p.add_argument(
        "--f146-limit",
        type=float,
        default=DEFAULT_F146_LIMIT,
    )

    p.add_argument(
        "--n-genulens",
        type=int,
        default=DEFAULT_N_GENULENS,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    p.add_argument(
        "--max-dmu",
        type=float,
        default=DEFAULT_MAX_DMU,
    )

    p.add_argument(
        "--t0-start",
        type=float,
        default=DEFAULT_T0_START,
    )

    p.add_argument(
        "--t0-end",
        type=float,
        default=DEFAULT_T0_END,
    )

    p.add_argument(
        "--field-id",
        default=None,
        help="Run exactly one production cell.",
    )

    p.add_argument(
        "--max-fields",
        type=int,
        default=None,
        help="Process at most N cells. Useful for smoke tests.",
    )

    p.add_argument(
        "--trilegal-insecure-ssl",
        action="store_true",
        help=(
            "Pass --insecure-ssl to the TRILEGAL downloader."
        ),
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
    )

    p.add_argument(
        "--assemble-only",
        action="store_true",
    )

    p.add_argument(
        "--no-assemble",
        action="store_true",
    )

    return p.parse_args()


def run_command(cmd, dry_run=False):
    printable = " ".join(
        str(x) for x in cmd
    )

    print()
    print("$", printable)

    if dry_run:
        return

    subprocess.run(
        [str(x) for x in cmd],
        cwd=REPO_ROOT,
        check=True,
    )


def parquet_rows(path: Path) -> int:
    if not path.exists():
        return 0

    return int(
        pq.ParquetFile(path)
        .metadata
        .num_rows
    )


def trilegal_paths(root: Path, field_id: str):
    return {
        "raw":
            root / "raw"
            / f"{field_id}.dat",

        "parquet":
            root / "processed"
            / f"{field_id}_physical.parquet",

        "metadata":
            root / "metadata"
            / f"{field_id}.json",
    }


def genulens_paths(root: Path, field_id: str):
    return {
        "parquet":
            root
            / f"{field_id}_genulens.parquet",

        "metadata":
            root
            / f"{field_id}_genulens.json",
    }


def event_paths(root: Path, field_id: str):
    d = root / field_id

    return {
        "dir": d,

        "matched":
            d / "matched_geometry_sources.parquet",

        "unmatched":
            d / "unmatched_genulens.parquet",

        "ffp":
            d / "ffp_events.parquet",

        "bh":
            d / "bh_events.parquet",

        "binary_lens":
            d / "binary_lens_events.parquet",

        "metadata":
            d / "precomputed_event_catalogs.json",
    }


def trilegal_complete(paths):
    return (
        paths["parquet"].exists()
        and paths["metadata"].exists()
    )


def genulens_complete(paths):
    return (
        paths["parquet"].exists()
        and paths["metadata"].exists()
    )


def events_complete(paths):
    required = [
        "matched",
        "unmatched",
        "ffp",
        "bh",
        "binary_lens",
        "metadata",
    ]

    return all(
        paths[k].exists()
        for k in required
    )


def git_commit():
    try:
        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=REPO_ROOT,
            text=True,
        ).strip()

    except Exception:
        return "unknown"


def write_status(records, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        records
    ).to_csv(
        path,
        index=False,
    )


def concatenate_parquets(
    files,
    output,
):
    """
    Stream-concatenate per-cell Parquet files without
    loading the full production catalogue into memory.
    """

    import pyarrow.parquet as pq

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = output.with_name(
        output.name + ".tmp"
    )

    if tmp.exists():
        tmp.unlink()

    writer = None
    schema = None
    total_rows = 0

    try:
        for i, path in enumerate(files, 1):

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing production file: {path}"
                )

            nrows = parquet_rows(path)

            if nrows == 0:
                continue

            table = pq.read_table(
                path
            ).replace_schema_metadata(
                None
            )

            if writer is None:

                schema = table.schema

                writer = pq.ParquetWriter(
                    tmp,
                    schema=schema,
                    compression="zstd",
                )

            elif table.schema != schema:

                raise RuntimeError(
                    "Parquet schema mismatch while "
                    f"assembling:\n{path}"
                )

            writer.write_table(
                table
            )

            total_rows += table.num_rows

            if (
                i == 1
                or i % 25 == 0
                or i == len(files)
            ):
                print(
                    f"  {i:4d}/{len(files)} "
                    f"files, "
                    f"{total_rows:,} rows"
                )

    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise RuntimeError(
            f"No non-empty files for {output}"
        )

    tmp.replace(
        output
    )

    return total_rows


def assemble_event_catalogues(
    cells,
    events_root,
    state_root,
    config,
):
    print()
    print("=" * 78)
    print("ASSEMBLING GLOBAL EVENT CATALOGUES")
    print("=" * 78)

    outputs = {}

    for key in [
        "ffp",
        "bh",
        "binary_lens",
    ]:

        files = [
            event_paths(
                events_root,
                str(row["field_id"]),
            )[key]
            for _, row in cells.iterrows()
        ]

        output = (
            events_root
            / f"{key}_events.parquet"
        )

        print()
        print(
            f"Assembling {key}:"
        )
        print(
            "  output:",
            output,
        )

        nrows = concatenate_parquets(
            files,
            output,
        )

        outputs[key] = {
            "path": str(output),
            "rows": int(nrows),
        }

    matched_files = [
        event_paths(
            events_root,
            str(row["field_id"]),
        )["matched"]
        for _, row in cells.iterrows()
    ]

    matched_output = (
        events_root
        / "matched_geometry_sources.parquet"
    )

    print()
    print(
        "Assembling matched geometry/source catalogue:"
    )

    nmatched = concatenate_parquets(
        matched_files,
        matched_output,
    )

    outputs["matched"] = {
        "path": str(matched_output),
        "rows": int(nmatched),
    }

    manifest = {
        "run_tag":
            config["run_tag"],

        "git_commit":
            git_commit(),

        "n_cells":
            int(len(cells)),

        "production_configuration":
            config,

        "outputs":
            outputs,
    }

    manifest_path = (
        events_root
        / "production_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
    )

    print()
    print(
        "Production manifest:",
        manifest_path,
    )

    print()
    print("=" * 78)
    print("GLOBAL ASSEMBLY COMPLETE")
    print("=" * 78)


def main():

    args = parse_args()

    cells_path = (
        args.cells.resolve()
    )

    cells = pd.read_csv(
        cells_path
    )

    if "production_enabled" in cells.columns:

        enabled = (
            cells["production_enabled"]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
        )

        cells = cells[
            enabled
        ].copy()

    cells = cells.reset_index(
        drop=True
    )

    if args.field_id is not None:

        cells = cells[
            cells["field_id"]
            .astype(str)
            == str(args.field_id)
        ].copy()

        if len(cells) != 1:
            raise ValueError(
                "field-id did not identify exactly "
                f"one production cell: {args.field_id}"
            )

    if args.max_fields is not None:
        cells = cells.iloc[
            :args.max_fields
        ].copy()

    run_tag = str(
        args.run_tag
    )

    tri_root = (
        REPO_ROOT
        / "stellar_population/trilegal/"
          "production"
        / run_tag
    )

    gen_root = (
        REPO_ROOT
        / "stellar_population/genulens/"
          "production"
        / run_tag
    )

    events_root = (
        REPO_ROOT
        / "stellar_population/"
          "precomputed_events/production"
        / run_tag
    )

    state_root = (
        REPO_ROOT
        / "stellar_population/production"
        / run_tag
    )

    cell_config_root = (
        state_root
        / "cell_configs"
    )

    for d in [
        tri_root,
        gen_root,
        events_root,
        state_root,
        cell_config_root,
    ]:
        d.mkdir(
            parents=True,
            exist_ok=True,
        )

    config = {
        "run_tag":
            run_tag,

        "git_commit":
            git_commit(),

        "cells_file":
            str(cells_path),

        "n_genulens_per_cell":
            int(args.n_genulens),

        "f146_limit_vega":
            float(args.f146_limit),

        "seed":
            int(args.seed),

        "max_delta_mu0_mag":
            float(args.max_dmu),

        "t0_range_jd": [
            float(args.t0_start),
            float(args.t0_end),
        ],

        "genulens_nsd":
            1,

        "genulens_small_gamma":
            1,

        "genulens_binary":
            0,

        "genulens_remnant":
            1,
    }

    (
        state_root
        / "production_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        )
    )

    print()
    print("=" * 78)
    print("ROMAN + RUBIN FULL EVENT-CATALOG PRODUCTION")
    print("=" * 78)
    print("run tag       :", run_tag)
    print("git commit    :", config["git_commit"])
    print("cells selected:", len(cells))
    print("TRILEGAL root :", tri_root)
    print("GENULENS root :", gen_root)
    print("events root   :", events_root)
    print("F146 limit    :", args.f146_limit)
    print("GEN/cell      :", args.n_genulens)
    print("max |dmu|     :", args.max_dmu)
    print(
        "t0 range      :",
        args.t0_start,
        args.t0_end,
    )

    if args.assemble_only:

        all_cells = pd.read_csv(
            cells_path
        )

        if "production_enabled" in all_cells.columns:
            enabled = (
                all_cells[
                    "production_enabled"
                ]
                .astype(str)
                .str.lower()
                .isin(
                    ["true", "1", "yes"]
                )
            )

            all_cells = all_cells[
                enabled
            ].reset_index(
                drop=True
            )

        assemble_event_catalogues(
            all_cells,
            events_root,
            state_root,
            config,
        )

        return

    records = []

    status_path = (
        state_root
        / "production_status.csv"
    )

    for index, row in cells.iterrows():

        field_id = str(
            row["field_id"]
        )

        print()
        print()
        print("#" * 78)
        print(
            f"FIELD {index + 1}/{len(cells)}: "
            f"{field_id}"
        )
        print("#" * 78)

        started = time.time()

        tri = trilegal_paths(
            tri_root,
            field_id,
        )

        gen = genulens_paths(
            gen_root,
            field_id,
        )

        evt = event_paths(
            events_root,
            field_id,
        )

        status = {
            "field_id":
                field_id,
            "trilegal":
                False,
            "genulens":
                False,
            "events":
                False,
            "n_trilegal":
                0,
            "n_genulens":
                0,
            "n_matched":
                0,
            "n_unmatched":
                0,
            "elapsed_s":
                0.0,
            "error":
                "",
        }

        try:

            # ====================================================
            # TRILEGAL
            # ====================================================

            if trilegal_complete(
                tri
            ):

                print(
                    "[SKIP] TRILEGAL:",
                    field_id,
                )

            else:

                one_cell_csv = (
                    cell_config_root
                    / f"{field_id}.csv"
                )

                pd.DataFrame(
                    [row]
                ).to_csv(
                    one_cell_csv,
                    index=False,
                )

                cmd = [
                    sys.executable,
                    (
                        REPO_ROOT
                        / "stellar_population/scripts/"
                          "download_trilegal.py"
                    ),
                    "--fields",
                    one_cell_csv,
                    "--outdir",
                    tri_root,
                    "--f146-limit",
                    str(args.f146_limit),
                ]

                if args.trilegal_insecure_ssl:
                    cmd.append(
                        "--insecure-ssl"
                    )

                run_command(
                    cmd,
                    dry_run=args.dry_run,
                )

                if not args.dry_run:

                    if not trilegal_complete(
                        tri
                    ):
                        raise RuntimeError(
                            "TRILEGAL stage did not "
                            "produce the expected files."
                        )

                    # download_trilegal.py creates a one-call
                    # master file. During per-cell production
                    # this is NOT the global master, so remove
                    # it to prevent accidental use.
                    scratch_master = (
                        tri_root
                        / "processed/"
                          "trilegal_physical_catalog.parquet"
                    )

                    if scratch_master.exists():
                        scratch_master.unlink()

            if not args.dry_run:
                status[
                    "n_trilegal"
                ] = parquet_rows(
                    tri["parquet"]
                )

            status["trilegal"] = (
                args.dry_run
                or trilegal_complete(tri)
            )


            # ====================================================
            # GENULENS
            # ====================================================

            if genulens_complete(
                gen
            ):

                print(
                    "[SKIP] GENULENS:",
                    field_id,
                )

            else:

                cmd = [
                    sys.executable,
                    (
                        REPO_ROOT
                        / "stellar_population/scripts/"
                          "build_genulens_trilegal_reservoir.py"
                    ),
                    "--cells",
                    cells_path,
                    "--output-dir",
                    gen_root,
                    "--n-simu",
                    str(args.n_genulens),
                    "--seed-base",
                    str(args.seed),
                    "--field-id",
                    field_id,
                    "--nsd",
                    "1",
                ]

                run_command(
                    cmd,
                    dry_run=args.dry_run,
                )

                if (
                    not args.dry_run
                    and not genulens_complete(gen)
                ):
                    raise RuntimeError(
                        "GENULENS stage did not "
                        "produce the expected files."
                    )

            if not args.dry_run:
                status[
                    "n_genulens"
                ] = parquet_rows(
                    gen["parquet"]
                )

            status["genulens"] = (
                args.dry_run
                or genulens_complete(gen)
            )


            # ====================================================
            # PRECOMPUTED EVENTS
            # ====================================================

            if events_complete(
                evt
            ):

                print(
                    "[SKIP] EVENTS:",
                    field_id,
                )

            else:

                evt["dir"].mkdir(
                    parents=True,
                    exist_ok=True,
                )

                cmd = [
                    sys.executable,
                    (
                        REPO_ROOT
                        / "stellar_population/scripts/"
                          "build_precomputed_event_catalogs.py"
                    ),
                    "--trilegal",
                    tri["parquet"],
                    "--genulens",
                    gen["parquet"],
                    "--output-dir",
                    evt["dir"],
                    "--max-dmu",
                    str(args.max_dmu),
                    "--seed",
                    str(args.seed),
                    "--t0-start",
                    str(args.t0_start),
                    "--t0-end",
                    str(args.t0_end),
                ]

                run_command(
                    cmd,
                    dry_run=args.dry_run,
                )

                if (
                    not args.dry_run
                    and not events_complete(evt)
                ):
                    raise RuntimeError(
                        "Event stage did not produce "
                        "the expected files."
                    )

            if not args.dry_run:

                meta = json.loads(
                    evt[
                        "metadata"
                    ].read_text()
                )

                status[
                    "n_matched"
                ] = int(
                    meta["n_matched"]
                )

                status[
                    "n_unmatched"
                ] = int(
                    meta["n_unmatched"]
                )

            status["events"] = (
                args.dry_run
                or events_complete(evt)
            )

        except Exception as exc:

            status["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

            status["elapsed_s"] = (
                time.time()
                - started
            )

            records.append(
                status
            )

            write_status(
                records,
                status_path,
            )

            print()
            print(
                "[FAILED]",
                field_id,
                status["error"],
            )

            raise

        status["elapsed_s"] = (
            time.time()
            - started
        )

        records.append(
            status
        )

        write_status(
            records,
            status_path,
        )

        print()
        print(
            "[COMPLETE]",
            field_id,
            f"{status['elapsed_s']:.1f} s",
        )

    print()
    print("=" * 78)
    print("SELECTED CELLS COMPLETE")
    print("=" * 78)
    print(
        "status:",
        status_path,
    )

    # ========================================================
    # Assemble only when this invocation represents the full
    # enabled production footprint.
    # ========================================================

    if args.no_assemble or args.dry_run:
        return

    all_cells = pd.read_csv(
        cells_path
    )

    if "production_enabled" in all_cells.columns:

        enabled = (
            all_cells[
                "production_enabled"
            ]
            .astype(str)
            .str.lower()
            .isin(
                ["true", "1", "yes"]
            )
        )

        all_cells = all_cells[
            enabled
        ].reset_index(
            drop=True
        )

    if len(cells) != len(all_cells):

        print()
        print(
            "Not assembling global catalogues because "
            "this was a subset run."
        )
        return

    missing = []

    for _, row in all_cells.iterrows():

        field_id = str(
            row["field_id"]
        )

        if not events_complete(
            event_paths(
                events_root,
                field_id,
            )
        ):
            missing.append(
                field_id
            )

    if missing:
        print()
        print(
            "Not assembling: incomplete fields =",
            len(missing),
        )
        return

    assemble_event_catalogues(
        all_cells,
        events_root,
        state_root,
        config,
    )


if __name__ == "__main__":
    main()
