#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path


SOURCE = Path(
    os.environ["pandeia_refdata"]
).resolve()

DESTINATION = (
    SOURCE.parent
    / (
        SOURCE.name
        + "-revh-gbtds"
    )
)


if (
    "revh-gbtds"
    in SOURCE.name.lower()
):
    raise RuntimeError(
        "pandeia_refdata already points "
        "to a patched RefData tree.\n"
        "Point it back to the original "
        "R2026.1 directory first."
    )


SOURCE_CONFIG = (
    SOURCE
    / "roman"
    / "wfi"
    / "config.json"
)

DEST_CONFIG = (
    DESTINATION
    / "roman"
    / "wfi"
    / "config.json"
)

MANIFEST = (
    DESTINATION
    / "LOCAL_REVH_GBTDS_PATCH.json"
)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


if not SOURCE_CONFIG.exists():
    raise FileNotFoundError(
        SOURCE_CONFIG
    )


print()
print("=======================================")
print("Roman Pandeia Rev-H local patch")
print("=======================================")

print()
print("Source:")
print(" ", SOURCE)

print("Destination:")
print(" ", DESTINATION)


# ============================================================
# Copy official R2026.1 RefData
# ============================================================

print()
print("Copying official RefData tree...")

shutil.copytree(
    SOURCE,
    DESTINATION,
    dirs_exist_ok=True,
)


original_sha = sha256(
    SOURCE_CONFIG
)


# ============================================================
# Read copied configuration
# ============================================================

with open(DEST_CONFIG) as f:
    cfg = json.load(f)


patterns = cfg[
    "readout_pattern_config"
]


if "im_66_6" not in patterns:
    raise RuntimeError(
        "Official R2026.1 im_66_6 "
        "template was not found."
    )


old = patterns[
    "im_66_6"
]


print()
print("Official Rev-G template:")
print(
    json.dumps(
        old,
        indent=2,
    )
)


# ============================================================
# IM_66_6_V2 — Revision H
#
# Roman WFI official Rev-H definition:
#
# Resultant   skips   reads
# --------------------------------
# R1          0       1
# R2          0       2
# R3          0       6
# R4          7       4
# R5          0       1
#
# Science read time = 3.16247 s
# Reset read time   = 3.16248 s
#
# MA table number = 1049
# MA table ID     = SCI1049
#
# We deliberately inherit all undocumented/internal
# Pandeia fields from the existing R2026.1 im_66_6
# entry and replace only quantities specified by the
# official Rev-H table.
# ============================================================

new = deepcopy(
    old
)


new.update(
    {
        "display_string":
            "IM_66_6_V2",

        "ma_table_description":
            (
                "Imaging, ~66 seconds, "
                "5 resultants"
            ),

        "ma_table_id":
            "SCI1049",

        "ma_table_number":
            1049,

        "frame_time":
            3.16247,

        "reset_frame_time":
            3.16248,

        "num_science_resultants":
            5,

        "min_science_resultants":
            5,

        # ------------------------------------
        # Exact Rev-H science read pattern.
        #
        # Reads 10..16 are skipped before R4.
        # ------------------------------------

        "science_read_pattern": [
            [1],
            [2, 3],
            [4, 5, 6, 7, 8, 9],
            [17, 18, 19, 20],
            [21],
        ],

        # ------------------------------------
        # Official Rev-H timing table
        # ------------------------------------

        "integration_duration": [
            6.32495,
            12.64989,
            31.62471,
            66.41189,
            69.57436,
        ],

        "accumulated_exposure_time": [
            3.16247,
            9.48741,
            28.46223,
            63.24941,
            66.41188,
        ],

        "effective_exposure_time": [
            3.16247,
            7.90618,
            20.55606,
            58.50570,
            66.41188,
        ],
    }
)


# ============================================================
# Internal consistency checks
# ============================================================

expected_pattern = [
    [1],
    [2, 3],
    [4, 5, 6, 7, 8, 9],
    [17, 18, 19, 20],
    [21],
]


if (
    new["science_read_pattern"]
    != expected_pattern
):
    raise RuntimeError(
        "Unexpected science-read pattern."
    )


used_reads = {
    r
    for resultant
    in new[
        "science_read_pattern"
    ]
    for r in resultant
}


expected_used = (
    set(range(1, 10))
    |
    set(range(17, 22))
)


if used_reads != expected_used:
    raise RuntimeError(
        "Rev-H used-read set is wrong."
    )


skipped_reads = (
    set(range(1, 22))
    - used_reads
)


if skipped_reads != set(
    range(10, 17)
):
    raise RuntimeError(
        "Expected skipped reads 10..16, "
        f"got {sorted(skipped_reads)}"
    )


if not abs(
    new[
        "effective_exposure_time"
    ][-1]
    - 66.41188
) < 1e-8:
    raise RuntimeError(
        "Unexpected final effective "
        "exposure time."
    )


# ============================================================
# Add Rev-H entry.
#
# Keep old im_66_6 as well. This lets us compare Rev G and
# Rev H using exactly the same detector/background model.
# ============================================================

patterns[
    "im_66_6_v2"
] = new


# ============================================================
# Write patched config
# ============================================================

with open(
    DEST_CONFIG,
    "w",
) as f:
    json.dump(
        cfg,
        f,
        indent=2,
    )

    f.write("\n")


patched_sha = sha256(
    DEST_CONFIG
)


# ============================================================
# Reproducibility manifest
# ============================================================

manifest = {
    "base_pandeia_refdata":
        str(SOURCE),

    "patched_pandeia_refdata":
        str(DESTINATION),

    "base_config_sha256":
        original_sha,

    "patched_config_sha256":
        patched_sha,

    "pandeia_release":
        "R2026.1",

    "base_ma_table_revision":
        "Revision G",

    "local_override_revision":
        "Revision H",

    "added_ma_table_name":
        "im_66_6_v2",

    "ma_table_number":
        1049,

    "ma_table_id":
        "SCI1049",

    "science_read_pattern":
        new[
            "science_read_pattern"
        ],

    "skipped_reads":
        sorted(
            skipped_reads
        ),

    "effective_exposure_time_s":
        new[
            "effective_exposure_time"
        ],

    "note": (
        "Local compatibility override. "
        "Pandeia Engine/PSF/detector/background "
        "model remains R2026.1; only the "
        "IM_66_6_V2 Rev-H MultiAccum definition "
        "is added to the copied reference tree."
    ),
}


with open(
    MANIFEST,
    "w",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
    )

    f.write("\n")


# ============================================================
# Report
# ============================================================

print()
print("=======================================")
print("Added IM_66_6_V2")
print("=======================================")

print(
    json.dumps(
        new,
        indent=2,
    )
)

print()
print(
    "Skipped science-read indices:",
    sorted(
        skipped_reads
    ),
)

print()
print(
    "Original config SHA256:",
    original_sha,
)

print(
    "Patched config SHA256: ",
    patched_sha,
)

print()
print("Patched RefData:")
print(" ", DESTINATION)

print()
print("Manifest:")
print(" ", MANIFEST)

print()
print("IMPORTANT:")
print(
    "The original R2026.1 RefData "
    "was not modified."
)
