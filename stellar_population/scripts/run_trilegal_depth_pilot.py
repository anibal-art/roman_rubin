#!/usr/bin/env python3

from pathlib import Path
import json
import subprocess
import sys
import time

import pandas as pd


PILOT_CONFIG = Path(
    "config/gbtds_trilegal_depth_pilot.csv"
)

DOWNLOADER = Path(
    "scripts/download_trilegal.py"
)

OUTPUT_ROOT = Path(
    "pilot_f14626"
)

F146_LIMIT = 26.0

MANIFEST = (
    OUTPUT_ROOT
    / "pilot_run_manifest.csv"
)


# ============================================================
# Load
# ============================================================

pilot = pd.read_csv(
    PILOT_CONFIG
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

config_dir = (
    OUTPUT_ROOT
    / "configs"
)

config_dir.mkdir(
    parents=True,
    exist_ok=True,
)


manifest_rows = []


# ============================================================
# One independent TRILEGAL invocation per LOS
#
# This avoids accumulating all deep catalogues in RAM and also
# gives us field-level restartability.
# ============================================================

for i, row in pilot.iterrows():

    label = str(
        row["pilot_label"]
    )

    field_id = str(
        row["field_id"]
    )


    one_config = (
        config_dir
        / f"{label}.csv"
    )

    field_out = (
        OUTPUT_ROOT
        / label
    )


    row.to_frame().T.to_csv(
        one_config,
        index=False,
    )


    master = (
        field_out
        / "processed"
        / "trilegal_physical_catalog.parquet"
    )

    metadata = (
        field_out
        / "metadata"
        / f"{field_id}.json"
    )


    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    already_complete = False
    old_nstars = None


    if (
        master.exists()
        and metadata.exists()
    ):

        try:

            with open(metadata) as f:
                md = json.load(f)


            same_field = (
                md.get("field_id")
                == field_id
            )

            same_limit = abs(
                float(
                    md.get(
                        "f146_parent_limit",
                        -999,
                    )
                )
                - F146_LIMIT
            ) < 1e-9


            if same_field and same_limit:

                old_nstars = int(
                    md["n_stars"]
                )

                already_complete = True

        except Exception:
            already_complete = False


    if already_complete:

        print()
        print(
            "======================================="
        )

        print(
            f"SKIP {label}: already complete"
        )

        print(
            f"  field = {field_id}"
        )

        print(
            f"  stars = {old_nstars:,}"
        )


        manifest_rows.append(
            {
                "pilot_label":
                    label,

                "field_id":
                    field_id,

                "status":
                    "existing",

                "elapsed_seconds":
                    0.0,

                "n_stars":
                    old_nstars,

                "output_dir":
                    str(field_out),
            }
        )


        pd.DataFrame(
            manifest_rows
        ).to_csv(
            MANIFEST,
            index=False,
        )

        continue


    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    print()
    print()
    print(
        "======================================="
    )

    print(
        f"PILOT {i + 1}/{len(pilot)}: "
        f"{label}"
    )

    print(
        "======================================="
    )

    print(
        f"field_id = {field_id}"
    )

    print(
        f"region   = {row['region']}"
    )

    print(
        f"l,b      = "
        f"({float(row['l_deg']):.6f}, "
        f"{float(row['b_deg']):.6f})"
    )

    print(
        f"Av_inf   = "
        f"{float(row['av_inf']):.3f}"
    )

    print(
        f"sigma_ext= "
        f"{float(row['extinction_sigma']):.4f}"
    )


    cmd = [
        sys.executable,
        str(DOWNLOADER),

        "--fields",
        str(one_config),

        "--outdir",
        str(field_out),

        "--f146-limit",
        str(F146_LIMIT),

        "--insecure-ssl",
    ]


    t0 = time.monotonic()


    try:

        subprocess.run(
            cmd,
            check=True,
        )

        elapsed = (
            time.monotonic()
            - t0
        )


        if not master.exists():
            raise RuntimeError(
                "Downloader returned successfully "
                "but master parquet is missing."
            )


        result = pd.read_parquet(
            master,
            columns=[
                "F146mag",
            ],
        )


        nstars = len(result)


        if result["F146mag"].isna().any():
            raise RuntimeError(
                f"{label}: NaN F146 values "
                "in completed catalogue."
            )


        if (
            result["F146mag"].max()
            > F146_LIMIT + 0.05
        ):
            raise RuntimeError(
                f"{label}: unexpected "
                f"max(F146)="
                f"{result['F146mag'].max()}"
            )


        status = "complete"


    except Exception:

        elapsed = (
            time.monotonic()
            - t0
        )

        manifest_rows.append(
            {
                "pilot_label":
                    label,

                "field_id":
                    field_id,

                "status":
                    "failed",

                "elapsed_seconds":
                    elapsed,

                "n_stars":
                    None,

                "output_dir":
                    str(field_out),
            }
        )


        pd.DataFrame(
            manifest_rows
        ).to_csv(
            MANIFEST,
            index=False,
        )

        raise


    manifest_rows.append(
        {
            "pilot_label":
                label,

            "field_id":
                field_id,

            "status":
                status,

            "elapsed_seconds":
                elapsed,

            "n_stars":
                nstars,

            "output_dir":
                str(field_out),
        }
    )


    pd.DataFrame(
        manifest_rows
    ).to_csv(
        MANIFEST,
        index=False,
    )


    print()
    print(
        f"{label}: COMPLETE"
    )

    print(
        f"stars   = {nstars:,}"
    )

    print(
        f"elapsed = "
        f"{elapsed / 60:.2f} min"
    )


    # Small courtesy pause between independent submissions.
    if i < len(pilot) - 1:
        time.sleep(5)


print()
print("=======================================")
print("DEPTH PILOT COMPLETE")
print("=======================================")

print()
print(
    pd.DataFrame(
        manifest_rows
    ).to_string(
        index=False
    )
)

print()
print("Saved:")
print(" ", MANIFEST)
