#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import genulens


def deterministic_seed(base_seed: int, field_id: str) -> int:
    h = hashlib.sha256(field_id.encode("utf-8")).digest()
    offset = int.from_bytes(h[:4], byteorder="little", signed=False)
    return int((base_seed + offset) % (2**31 - 1))


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build GENULENS event reservoirs on the same "
            "GBTDS sightlines used by the TRILEGAL population."
        )
    )

    p.add_argument(
        "--cells",
        default=(
            "stellar_population/"
            "config/gbtds_trilegal_cells_production.csv"
        ),
    )

    p.add_argument(
        "--output-dir",
        default=(
            "stellar_population/"
            "genulens/production"
        ),
    )

    p.add_argument(
        "--n-simu",
        type=int,
        default=5000,
        help="Accepted GENULENS events per sightline.",
    )

    p.add_argument(
        "--seed-base",
        type=int,
        default=20260930,
    )

    p.add_argument(
        "--field-id",
        default=None,
        help="Run only one field_id.",
    )

    p.add_argument(
        "--max-fields",
        type=int,
        default=None,
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    p.add_argument(
        "--nsd",
        type=int,
        choices=[0, 1],
        default=1,
        help=(
            "Enable GENULENS nuclear stellar disk component. "
            "Recommended for the GBTDS footprint including GC."
        ),
    )

    return p.parse_args()


def make_config(
    l_deg: float,
    b_deg: float,
    n_simu: int,
    seed: int,
    nsd: int,
):
    cfg = genulens.Config(
        l=float(l_deg),
        b=float(b_deg),
        n_simu=int(n_simu),
        seed=int(seed),
    )

    # --------------------------------------------------------
    # We want an event/lens reservoir, not GENULENS to define
    # the final source catalogue.
    #
    # Keep low-Gamma events and their weights.
    # --------------------------------------------------------

    cfg.sampling.small_gamma = 1

    # Binary/planet structure is imposed downstream by our
    # microlensing population model, not here.
    cfg.sampling.binary = 0

    # Keep stellar remnants available in the Galactic model.
    cfg.sampling.remnant = 1

    cfg.sampling.verbosity = 3

    # Important for the central GBTDS field.
    cfg.model.nsd.enabled = int(nsd)

    return cfg


def run_field(
    row,
    output_dir: Path,
    n_simu: int,
    seed_base: int,
    overwrite: bool,
    nsd: int,
):
    field_id = str(row["field_id"])
    l_deg = float(row["l_deg"])
    b_deg = float(row["b_deg"])

    seed = deterministic_seed(
        seed_base,
        field_id,
    )

    out = (
        output_dir
        / f"{field_id}_genulens.parquet"
    )

    meta = (
        output_dir
        / f"{field_id}_genulens.json"
    )

    if out.exists() and not overwrite:
        print(
            f"[SKIP] {field_id}: "
            f"{out} already exists"
        )
        return None

    print()
    print("=" * 72)
    print("FIELD:", field_id)
    print("l,b  :", l_deg, b_deg)
    print("Nsimu:", n_simu)
    print("seed :", seed)
    print("NSD  :", nsd)
    print("=" * 72)

    cfg = make_config(
        l_deg=l_deg,
        b_deg=b_deg,
        n_simu=n_simu,
        seed=seed,
        nsd=nsd,
    )

    result = genulens.simulate(cfg)

    df = pd.DataFrame(
        result.to_numpy(),
        columns=list(result.columns),
    )

    required = {
        "wtj",
        "M_L",
        "D_L",
        "D_S",
        "t_E",
        "theta_E",
        "pi_E",
        "pi_EN",
        "pi_EE",
        "mu_rel",
        "mu_rel_N",
        "mu_rel_E",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{field_id}: missing columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Integrity
    # --------------------------------------------------------

    if len(df) == 0:
        raise RuntimeError(
            f"{field_id}: GENULENS returned no events."
        )

    if not np.all(
        np.isfinite(df["D_L"])
        &
        np.isfinite(df["D_S"])
    ):
        raise RuntimeError(
            f"{field_id}: non-finite distances."
        )

    if not np.all(
        df["D_L"] < df["D_S"]
    ):
        bad = int(
            np.sum(
                df["D_L"]
                >= df["D_S"]
            )
        )
        raise RuntimeError(
            f"{field_id}: {bad} events have DL >= DS."
        )

    if not np.all(
        np.isfinite(df["wtj"])
    ):
        raise RuntimeError(
            f"{field_id}: non-finite wtj."
        )

    # --------------------------------------------------------
    # Metadata needed for later TRILEGAL matching.
    # --------------------------------------------------------

    df.insert(
        0,
        "genulens_event_id",
        [
            f"{field_id}_G{i:08d}"
            for i in range(len(df))
        ],
    )

    df.insert(
        0,
        "field_id",
        field_id,
    )

    df["field_l_deg"] = l_deg
    df["field_b_deg"] = b_deg
    df["genulens_seed"] = seed

    for col in (
        "region",
        "target_area_deg2",
        "sample_area_deg2",
        "area_weight",
        "season_coverage",
        "refinement_level",
    ):
        if col in row.index:
            df[col] = row[col]

    # Keep explicit aliases for the future pairer.
    df["D_S_genulens"] = df["D_S"]
    df["D_L_genulens"] = df["D_L"]

    df.to_parquet(
        out,
        index=False,
        compression="zstd",
    )

    metadata = {
        "field_id": field_id,
        "l_deg": l_deg,
        "b_deg": b_deg,
        "n_simu_requested": int(n_simu),
        "n_events": int(len(df)),
        "seed": int(seed),
        "small_gamma": 1,
        "binary": 0,
        "remnant": 1,
        "nsd_enabled": int(nsd),

        "purpose": (
            "GENULENS geometry/kinematic reservoir for "
            "matching to external TRILEGAL sources."
        ),

        "geometry_authority": "GENULENS",
        "source_photometry_authority": "TRILEGAL",
        "mass_authority": "downstream_class_specific_priors",

        "source_match_variable": "D_S",

        "weight_column": "wtj",
        "weight_usage": "provenance_only",

        "sampling_interpretation": (
            "Unweighted GENULENS geometry/kinematic proposal; "
            "not an event-rate-weighted Galactic population."
        ),

        "output": str(out),
    }

    meta.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    summary = {
        "field_id": field_id,
        "l_deg": l_deg,
        "b_deg": b_deg,
        "n_events": len(df),
        "seed": seed,
        "D_S_min": float(df["D_S"].min()),
        "D_S_median": float(df["D_S"].median()),
        "D_S_max": float(df["D_S"].max()),
        "D_L_min": float(df["D_L"].min()),
        "D_L_median": float(df["D_L"].median()),
        "D_L_max": float(df["D_L"].max()),
        "mu_rel_median": float(df["mu_rel"].median()),
        "wtj_sum": float(df["wtj"].sum()),
        "output": str(out),
    }

    print(
        "D_S min/med/max =",
        summary["D_S_min"],
        summary["D_S_median"],
        summary["D_S_max"],
    )

    print(
        "D_L min/med/max =",
        summary["D_L_min"],
        summary["D_L_median"],
        summary["D_L_max"],
    )

    print(
        "mu_rel median    =",
        summary["mu_rel_median"],
    )

    print(
        "wtj sum          =",
        summary["wtj_sum"],
    )

    print("saved:", out)

    return summary


def main():
    args = parse_args()

    cells_path = Path(args.cells)
    output_dir = Path(args.output_dir)

    if not cells_path.is_file():
        raise FileNotFoundError(
            cells_path
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cells = pd.read_csv(
        cells_path
    )

    required = {
        "field_id",
        "l_deg",
        "b_deg",
    }

    missing = required - set(cells.columns)

    if missing:
        raise RuntimeError(
            "Cells config missing columns: "
            + ", ".join(sorted(missing))
        )

    # Production file should already be filtered, but honor
    # common enable-column names if they exist.
    for enable_col in (
        "production_enabled",
        "enabled",
        "use_for_production",
    ):
        if enable_col in cells.columns:
            vals = (
                cells[enable_col]
                .astype(str)
                .str.lower()
            )

            cells = cells[
                vals.isin(
                    {
                        "1",
                        "true",
                        "yes",
                        "y",
                    }
                )
            ].copy()

            break

    if args.field_id is not None:
        cells = cells[
            cells["field_id"].astype(str)
            == str(args.field_id)
        ].copy()

        if len(cells) != 1:
            raise RuntimeError(
                f"field-id {args.field_id!r}: "
                f"found {len(cells)} rows."
            )

    if args.max_fields is not None:
        cells = cells.iloc[
            :args.max_fields
        ].copy()

    print()
    print("=======================================")
    print("GENULENS -> TRILEGAL reservoir")
    print("=======================================")
    print("Cells file :", cells_path)
    print("N fields   :", len(cells))
    print("N/event LOS:", args.n_simu)
    print("Output dir :", output_dir)
    print("NSD        :", args.nsd)

    summaries = []

    for _, row in cells.iterrows():
        result = run_field(
            row=row,
            output_dir=output_dir,
            n_simu=args.n_simu,
            seed_base=args.seed_base,
            overwrite=args.overwrite,
            nsd=args.nsd,
        )

        if result is not None:
            summaries.append(result)

    if summaries:
        sdf = pd.DataFrame(
            summaries
        )

        summary_path = (
            output_dir
            / "genulens_run_summary.csv"
        )

        if summary_path.exists():
            old = pd.read_csv(
                summary_path
            )

            sdf = pd.concat(
                [old, sdf],
                ignore_index=True,
            )

            sdf = (
                sdf
                .drop_duplicates(
                    subset=["field_id"],
                    keep="last",
                )
                .sort_values("field_id")
            )

        sdf.to_csv(
            summary_path,
            index=False,
        )

        print()
        print("Summary:", summary_path)


if __name__ == "__main__":
    main()
