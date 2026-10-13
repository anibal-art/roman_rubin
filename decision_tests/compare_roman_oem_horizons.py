#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.time import Time
from astroquery.jplhorizons import Horizons


OEM_PATH = Path(
    "decision_tests/data/RST_103026.oem"
)

ROMAN_ID = "-211"
GEOCENTER = "500@399"

# Keep comfortably inside Horizons' known support.
COMPARE_START = "2027-02-11T00:00:00"
COMPARE_END = "2028-10-03T00:00:00"

N_COMPARE = 12

AU_KM = u.au.to(u.km)


def header(title):
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def parse_oem_time(token):
    """
    OEM:
        YYYY-DDDTHH:MM:SS.ssssss

    Astropy yday:
        YYYY:DDD:HH:MM:SS.ssssss
    """
    date_part, time_part = token.split("T")
    year, doy = date_part.split("-")

    value = f"{year}:{doy}:{time_part}"

    return Time(
        value,
        format="yday",
        scale="utc",
    )


def load_oem(path):

    records = []

    with path.open() as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if not (
                len(line) >= 5
                and line[:4].isdigit()
                and line[4] == "-"
            ):
                continue

            parts = line.split()

            if len(parts) != 7:
                continue

            t = parse_oem_time(
                parts[0]
            )

            x, y, z, vx, vy, vz = [
                float(x)
                for x in parts[1:]
            ]

            records.append(
                {
                    "time_oem": parts[0],
                    "jd_utc": float(t.jd),
                    "jd_tdb": float(t.tdb.jd),
                    "x_oem_km": x,
                    "y_oem_km": y,
                    "z_oem_km": z,
                    "vx_oem_km_s": vx,
                    "vy_oem_km_s": vy,
                    "vz_oem_km_s": vz,
                }
            )

    return pd.DataFrame(
        records
    )


def vector_angle_deg(a, b):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    denom = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denom == 0:
        return np.nan

    cosine = np.dot(a, b) / denom

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return float(
        np.rad2deg(
            np.arccos(cosine)
        )
    )


def main():

    header("LOAD OEM")

    if not OEM_PATH.exists():
        raise FileNotFoundError(
            OEM_PATH
        )

    oem = load_oem(
        OEM_PATH
    )

    print(
        "OEM rows =",
        len(oem),
    )

    print(
        "first =",
        oem.iloc[0]["time_oem"],
    )

    print(
        "last  =",
        oem.iloc[-1]["time_oem"],
    )

    print()
    print(
        "position units = km"
    )
    print(
        "velocity units = km/s"
    )

    # ------------------------------------------------------------
    # Restrict to overlap with JPL Horizons.
    # ------------------------------------------------------------

    start_jd = float(
        Time(
            COMPARE_START,
            format="isot",
            scale="utc",
        ).jd
    )

    end_jd = float(
        Time(
            COMPARE_END,
            format="isot",
            scale="utc",
        ).jd
    )

    overlap = oem[
        (oem["jd_utc"] >= start_jd)
        &
        (oem["jd_utc"] <= end_jd)
    ].reset_index(
        drop=True
    )

    header("OVERLAP")

    print(
        "rows in overlap =",
        len(overlap),
    )

    print(
        "first overlap =",
        overlap.iloc[0]["time_oem"],
    )

    print(
        "last overlap  =",
        overlap.iloc[-1]["time_oem"],
    )

    # ------------------------------------------------------------
    # Select evenly distributed OEM epochs.
    # ------------------------------------------------------------

    indices = np.linspace(
        0,
        len(overlap) - 1,
        N_COMPARE,
    )

    indices = np.unique(
        np.round(
            indices
        ).astype(int)
    )

    sample = overlap.iloc[
        indices
    ].copy().reset_index(
        drop=True
    )

    header("SELECTED EPOCHS")

    for i, row in sample.iterrows():

        print(
            f"{i:02d}  "
            f"{row['time_oem']}  "
            f"JD_UTC={row['jd_utc']:.9f}  "
            f"JD_TDB={row['jd_tdb']:.9f}"
        )

    # ------------------------------------------------------------
    # Query JPL vector table.
    #
    # Earth geocenter, geometric vectors, Earth equatorial
    # reference plane.
    # ------------------------------------------------------------

    header("QUERY JPL HORIZONS VECTORS")

    epochs_tdb = sample[
        "jd_tdb"
    ].to_numpy(
        dtype=float
    )

    try:

        obj = Horizons(
            id=ROMAN_ID,
            id_type=None,
            location=GEOCENTER,
            epochs=epochs_tdb,
        )

        vectors = obj.vectors(
            refplane="earth",
            aberrations="geometric",
        )

    except Exception as error:

        print(
            "QUERY FAILED"
        )

        print(
            "type =",
            type(error).__name__,
        )

        print(
            "error =",
            repr(error),
        )

        return

    print(
        "rows =",
        len(vectors),
    )

    print(
        "columns =",
        list(vectors.colnames),
    )

    print()

    for key in (
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
    ):

        if key in vectors.colnames:
            print(
                f"{key:3s} unit =",
                vectors[key].unit,
            )

    if len(vectors) != len(sample):

        raise RuntimeError(
            "Horizons returned a different "
            "number of epochs."
        )

    # ------------------------------------------------------------
    # Compare positions.
    # ------------------------------------------------------------

    header("OEM vs HORIZONS")

    records = []

    for i in range(
        len(sample)
    ):

        row = sample.iloc[i]
        hv = vectors[i]

        p_oem = np.array(
            [
                row["x_oem_km"],
                row["y_oem_km"],
                row["z_oem_km"],
            ],
            dtype=float,
        )

        # Astroquery Horizons vectors are returned in AU.
        p_horizons = np.array(
            [
                float(hv["x"]),
                float(hv["y"]),
                float(hv["z"]),
            ],
            dtype=float,
        ) * AU_KM

        delta = (
            p_oem
            - p_horizons
        )

        delta_signflip = (
            p_oem
            + p_horizons
        )

        norm_oem = float(
            np.linalg.norm(
                p_oem
            )
        )

        norm_horizons = float(
            np.linalg.norm(
                p_horizons
            )
        )

        difference_km = float(
            np.linalg.norm(
                delta
            )
        )

        difference_signflip_km = float(
            np.linalg.norm(
                delta_signflip
            )
        )

        angle_deg = vector_angle_deg(
            p_oem,
            p_horizons,
        )

        radial_difference_km = (
            norm_oem
            - norm_horizons
        )

        fractional_difference = (
            difference_km
            / norm_horizons
            if norm_horizons > 0
            else np.nan
        )

        print()
        print(
            row["time_oem"]
        )

        print(
            f"  |OEM|       = "
            f"{norm_oem:12.3f} km"
        )

        print(
            f"  |Horizons|  = "
            f"{norm_horizons:12.3f} km"
        )

        print(
            f"  |OEM-Horiz| = "
            f"{difference_km:12.3f} km"
        )

        print(
            f"  angle       = "
            f"{angle_deg:12.6f} deg"
        )

        print(
            f"  radial diff = "
            f"{radial_difference_km:12.3f} km"
        )

        print(
            f"  frac diff   = "
            f"{fractional_difference:12.6e}"
        )

        records.append(
            {
                "time_oem": row["time_oem"],
                "jd_utc": row["jd_utc"],
                "jd_tdb": row["jd_tdb"],
                "oem_norm_km": norm_oem,
                "horizons_norm_km": norm_horizons,
                "difference_km": difference_km,
                "difference_signflip_km":
                    difference_signflip_km,
                "angle_deg": angle_deg,
                "radial_difference_km":
                    radial_difference_km,
                "fractional_difference":
                    fractional_difference,
            }
        )

    result = pd.DataFrame(
        records
    )

    header("SUMMARY")

    for col in (
        "difference_km",
        "angle_deg",
        "radial_difference_km",
        "fractional_difference",
    ):

        x = result[
            col
        ].to_numpy(
            dtype=float
        )

        print(
            f"{col:24s} "
            f"min={np.nanmin(x):.8g}  "
            f"median={np.nanmedian(x):.8g}  "
            f"max={np.nanmax(x):.8g}"
        )

    direct = np.nanmedian(
        result[
            "difference_km"
        ]
    )

    flipped = np.nanmedian(
        result[
            "difference_signflip_km"
        ]
    )

    print()
    print(
        "median direct difference   =",
        direct,
        "km",
    )

    print(
        "median sign-flip difference =",
        flipped,
        "km",
    )

    print(
        "same sign convention =",
        bool(
            direct < flipped
        ),
    )

    output = Path(
        "decision_tests/"
        "roman_oem_vs_horizons.csv"
    )

    result.to_csv(
        output,
        index=False,
    )

    print()
    print(
        "saved =",
        output.resolve(),
    )

    header("INTERPRETATION")

    print(
        "This test compares the official Roman OEM "
        "trajectory against JPL Horizons at the same "
        "physical epochs."
    )

    print(
        "Do not convert the OEM to the production "
        "pyLIMA ephemeris until this comparison has "
        "been inspected."
    )


if __name__ == "__main__":
    main()
