#!/usr/bin/env python3

from pathlib import Path
import hashlib
import subprocess

import numpy as np


ROOT = Path(".").resolve()

FILES = [
    ROOT / "ephemerides/Roman_positions.npy",
    ROOT / "ephemerides/Roman_positionsv0.npy",
    ROOT / "ephemerides/Gaia.txt",
]


def header(title):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def run_git(args):
    cmd = ["git"] + list(args)

    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    print("$", " ".join(cmd))
    print(result.stdout.rstrip())

    return result.returncode


def print_text_edges(path, n=20):
    if not path.exists():
        print("MISSING:", path)
        return

    try:
        lines = path.read_text(
            errors="replace"
        ).splitlines()
    except Exception as error:
        print(
            "Could not read text:",
            repr(error),
        )
        return

    print(
        "number of text lines =",
        len(lines),
    )

    print()
    print("--- FIRST LINES ---")

    for line in lines[:n]:
        print(line)

    print()
    print("--- LAST LINES ---")

    for line in lines[-n:]:
        print(line)


def search_text_for_v0_values(
    text_path,
    v0,
):
    if not text_path.exists():
        return

    text = text_path.read_text(
        errors="replace"
    )

    header(
        "SEARCH Roman_positionsv0 VALUES "
        "INSIDE Gaia.txt"
    )

    indices = [
        0,
        len(v0) // 4,
        len(v0) // 2,
        3 * len(v0) // 4,
        len(v0) - 1,
    ]

    for i in indices:

        row = v0[i]

        print()
        print(
            f"v0 row {i}:",
            row,
        )

        for value_name, value in (
            ("RA", row[1]),
            ("Dec", row[2]),
            ("distance", row[3]),
        ):

            patterns = [
                f"{value:.8f}",
                f"{value:.7f}",
                f"{value:.6f}",
                f"{value:.5f}",
            ]

            matches = []

            for pattern in patterns:

                if pattern in text:
                    matches.append(
                        pattern
                    )

            print(
                f"  {value_name:8s} "
                f"value={value:.12g} "
                f"text_matches={matches}"
            )


def main():

    header("REPOSITORY")

    run_git(
        [
            "rev-parse",
            "HEAD",
        ]
    )

    run_git(
        [
            "status",
            "--short",
            "--",
            "ephemerides",
            "decision_tests",
        ]
    )

    header("FILES / HASHES")

    for path in FILES:

        print()
        print("path =", path)

        if not path.exists():
            print("exists = False")
            continue

        print("exists = True")
        print(
            "size   =",
            path.stat().st_size,
        )
        print(
            "sha256 =",
            sha256(path),
        )

        if path.suffix == ".npy":

            arr = np.load(path)

            print(
                "shape  =",
                arr.shape,
            )

            print(
                "first  =",
                arr[0],
            )

            print(
                "last   =",
                arr[-1],
            )

    header("Gaia.txt CONTENT")

    print_text_edges(
        ROOT / "ephemerides/Gaia.txt",
        n=20,
    )

    current = np.load(
        ROOT
        / "ephemerides/Roman_positions.npy"
    )

    legacy = np.load(
        ROOT
        / "ephemerides/Roman_positionsv0.npy"
    )

    search_text_for_v0_values(
        ROOT / "ephemerides/Gaia.txt",
        legacy,
    )

    header(
        "GIT HISTORY: Roman_positions.npy"
    )

    run_git(
        [
            "log",
            "--all",
            "--follow",
            "--date=iso",
            "--format=%h %ad %an %s",
            "--",
            "ephemerides/Roman_positions.npy",
        ]
    )

    header(
        "GIT HISTORY: Roman_positionsv0.npy"
    )

    run_git(
        [
            "log",
            "--all",
            "--follow",
            "--date=iso",
            "--format=%h %ad %an %s",
            "--",
            "ephemerides/Roman_positionsv0.npy",
        ]
    )

    header(
        "GIT HISTORY: Gaia.txt"
    )

    run_git(
        [
            "log",
            "--all",
            "--follow",
            "--date=iso",
            "--format=%h %ad %an %s",
            "--",
            "ephemerides/Gaia.txt",
        ]
    )

    header(
        "SEARCH TRACKED CODE FOR EPHEMERIS CREATION"
    )

    patterns = [
        "Roman_positionsv0",
        "Roman_positions.npy",
        "Gaia.txt",
        "3162.758",
        "3162.76",
        "horizons_API",
    ]

    for pattern in patterns:

        print()
        print(
            "-" * 88
        )
        print(
            "PATTERN:",
            pattern,
        )
        print(
            "-" * 88
        )

        run_git(
            [
                "grep",
                "-n",
                "-I",
                pattern,
                "--",
                "*.py",
                "*.ipynb",
                "*.md",
                "*.txt",
                "*.json",
            ]
        )

    header(
        "SEARCH GIT HISTORY FOR TIME SHIFT"
    )

    for pattern in (
        "3162.758",
        "3162.76",
        "Roman_positionsv0",
    ):

        print()
        print(
            "SEARCH HISTORY FOR:",
            pattern,
        )

        run_git(
            [
                "log",
                "--all",
                "--oneline",
                "-S",
                pattern,
            ]
        )

    header("CURRENT-v0 RELATION")

    if current.shape == legacy.shape:

        delta_t = (
            current[:, 0]
            - legacy[:, 0]
        )

        print(
            "median dt =",
            repr(
                float(
                    np.median(delta_t)
                )
            ),
        )

        print(
            "unique spatial arrays =",
            bool(
                np.array_equal(
                    current[:, 1:],
                    legacy[:, 1:],
                )
            ),
        )

    print()
    print(
        "This audit does not modify any ephemeris."
    )


if __name__ == "__main__":
    main()
