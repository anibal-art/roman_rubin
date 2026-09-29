#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from pandeia.engine.calc_utils import build_default_calc
from pandeia.engine.perform_calculation import perform_calculation


DEFAULT_OUTPUT = Path(
    "noise_models/data/roman_f146_pandeia_2026p1.csv"
)

DEFAULT_METADATA = Path(
    "noise_models/data/roman_f146_pandeia_2026p1.json"
)

DETECTORS = [
    f"wfi{i:02d}"
    for i in range(1, 19)
]

F146_BENCHMARK_AB = 21.2

MAG_TO_FRAC = np.log(10.0) / 2.5
FRAC_TO_MAG = 2.5 / np.log(10.0)


def atomic_write_csv(df, path):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    tmp.replace(path)


def compute_ab_minus_vega():
    """
    Determine the current Roman F146 AB - Vega offset
    from STScI synphot reference data.

    This is deliberately not hard-coded.
    """

    try:
        import astropy.units as u
        import stsynphot as stsyn

        from synphot import (
            Observation,
            SourceSpectrum,
        )

        from synphot.units import VEGAMAG

        bp = stsyn.band(
            "roman, wfi, f146"
        )

        vega = (
            SourceSpectrum
            .from_vega()
        )

        obs = Observation(
            vega,
            bp,
            force="taper",
        )

        m_ab = float(
            obs.effstim(
                u.ABmag
            ).value
        )

        m_vega = float(
            obs.effstim(
                VEGAMAG,
                vegaspec=vega,
            ).value
        )

        return (
            m_ab - m_vega,
            None,
        )

    except Exception as exc:
        return (
            None,
            (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )


def build_base_calc(
    detector,
    background_level,
):
    """
    Current GBTDS F146 single-epoch setup.
    """

    calc = build_default_calc(
        "roman",
        "wfi",
        "imaging",
    )

    # ----------------------------------------
    # Instrument
    # ----------------------------------------

    calc[
        "configuration"
    ][
        "instrument"
    ][
        "filter"
    ] = "f146"

    calc[
        "configuration"
    ][
        "instrument"
    ][
        "detector"
    ] = detector

    # ----------------------------------------
    # GBTDS exposure
    #
    # Current survey:
    #   66 s
    #   IM_66_6_V2
    #   one exposure per epoch
    #   no truncation
    # ----------------------------------------

    calc[
        "configuration"
    ][
        "detector"
    ][
        "ma_table_name"
    ] = "im_66_6_v2"

    calc[
        "configuration"
    ][
        "detector"
    ][
        "nresultants"
    ] = -1

    calc[
        "configuration"
    ][
        "detector"
    ][
        "nexp"
    ] = 1

    # ----------------------------------------
    # GBTDS canned sky background
    # ----------------------------------------

    calc["background"] = (
        "gbtds_mid_5stripe"
    )

    calc["background_level"] = (
        background_level
    )

    # ----------------------------------------
    # Point source, flat f_nu.
    #
    # A flat f_nu source has constant AB
    # magnitude, so AB normalization at the
    # default reference wavelength is
    # consistent with its F146 AB magnitude.
    # ----------------------------------------

    spectrum = (
        calc["scene"][0]["spectrum"]
    )

    spectrum["sed"] = {
        "sed_type": "flat",
        "unit": "fnu",
        "z": 0.0,
    }

    spectrum[
        "normalization"
    ][
        "norm_fluxunit"
    ] = "abmag"

    return calc


def extract_saturation(warnings):
    warnings = (
        warnings
        if isinstance(warnings, dict)
        else {}
    )

    partial = (
        warnings.get(
            "partial_saturated"
        )
        is not None
    )

    full = (
        warnings.get(
            "full_saturated"
        )
        is not None
    )

    return partial, full


def calculate_one(
    base_calc,
    mag_ab,
):
    calc = deepcopy(
        base_calc
    )

    calc[
        "scene"
    ][0][
        "spectrum"
    ][
        "normalization"
    ][
        "norm_flux"
    ] = float(mag_ab)

    report = perform_calculation(
        calc
    )

    scalar = report[
        "scalar"
    ]

    snr = float(
        scalar["sn"]
    )

    warnings = report.get(
        "warnings",
        {},
    )

    partial, full = (
        extract_saturation(
            warnings
        )
    )

    if not np.isfinite(snr):
        snr = np.nan

    sigma_mag = (
        FRAC_TO_MAG / snr
        if np.isfinite(snr)
        and snr > 0
        else np.nan
    )

    frac_flux_err = (
        1.0 / snr
        if np.isfinite(snr)
        and snr > 0
        else np.nan
    )

    return {
        "snr": snr,

        "frac_flux_err":
            frac_flux_err,

        "sigma_mag":
            sigma_mag,

        "partial_saturated":
            partial,

        "full_saturated":
            full,

        "sat_nresultants":
            scalar.get(
                "sat_nresultants",
                np.nan,
            ),

        "total_exposure_time":
            scalar.get(
                "total_exposure_time",
                np.nan,
            ),

        "warnings_json":
            json.dumps(
                warnings,
                default=str,
                sort_keys=True,
            ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
    )

    parser.add_argument(
        "--mag-min",
        type=float,
        default=14.0,
    )

    parser.add_argument(
        "--mag-max",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--mag-step",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--background-levels",
        nargs="+",
        default=["medium"],
        choices=[
            "low",
            "medium",
            "high",
        ],
    )

    parser.add_argument(
        "--detectors",
        nargs="+",
        default=DETECTORS,
    )

    args = parser.parse_args()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.metadata.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------
    # Environment sanity check
    # ----------------------------------------

    required_env = [
        "pandeia_refdata",
        "PSF_DIR",
        "PYSYN_CDBS",
    ]

    missing_env = [
        k
        for k in required_env
        if not os.environ.get(k)
    ]

    if missing_env:
        raise RuntimeError(
            "Missing Pandeia environment "
            f"variables: {missing_env}"
        )

    engine_version = version(
        "pandeia.engine"
    )

    print()
    print(
        "======================================="
    )
    print(
        "Roman F146 Pandeia noise grid"
    )
    print(
        "======================================="
    )

    print(
        "Pandeia engine =",
        engine_version,
    )

    print(
        "RefData =",
        os.environ[
            "pandeia_refdata"
        ],
    )

    print(
        "PSFs    =",
        os.environ[
            "PSF_DIR"
        ],
    )

    print(
        "Synphot =",
        os.environ[
            "PYSYN_CDBS"
        ],
    )

    # Include exact survey benchmark.
    regular_grid = np.arange(
        args.mag_min,
        args.mag_max
        + args.mag_step * 0.5,
        args.mag_step,
    )

    mags = np.unique(
        np.round(
            np.concatenate(
                [
                    regular_grid,
                    [
                        F146_BENCHMARK_AB
                    ],
                ]
            ),
            8,
        )
    )

    # ----------------------------------------
    # Resume existing calculations
    # ----------------------------------------

    if args.output.exists():
        old = pd.read_csv(
            args.output
        )

        rows = old.to_dict(
            orient="records"
        )

    else:
        rows = []

    done = {
        (
            str(r["detector"]),
            str(
                r[
                    "background_level"
                ]
            ),
            round(
                float(r["mag_ab"]),
                8,
            ),
        )
        for r in rows
    }

    total = (
        len(args.detectors)
        * len(
            args.background_levels
        )
        * len(mags)
    )

    counter = len(done)

    for bg in (
        args.background_levels
    ):

        for detector in (
            args.detectors
        ):

            print()
            print(
                f"Detector {detector}, "
                f"background={bg}"
            )

            base = build_base_calc(
                detector=detector,
                background_level=bg,
            )

            for mag in mags:

                key = (
                    detector,
                    bg,
                    round(
                        float(mag),
                        8,
                    ),
                )

                if key in done:
                    continue

                result = calculate_one(
                    base_calc=base,
                    mag_ab=float(mag),
                )

                row = {
                    "detector":
                        detector,

                    "background":
                        "gbtds_mid_5stripe",

                    "background_level":
                        bg,

                    "filter":
                        "f146",

                    "ma_table":
                        "im_66_6_v2",

                    "nexp":
                        1,

                    "nresultants":
                        -1,

                    "mag_ab":
                        float(mag),

                    **result,
                }

                rows.append(
                    row
                )

                done.add(
                    key
                )

                counter += 1

                print(
                    f"[{counter:4d}/{total}] "
                    f"{detector} "
                    f"{bg:6s} "
                    f"AB={mag:5.2f} "
                    f"S/N="
                    f"{result['snr']:9.3f} "
                    f"partial_sat="
                    f"{result['partial_saturated']} "
                    f"full_sat="
                    f"{result['full_saturated']}"
                )

                atomic_write_csv(
                    pd.DataFrame(
                        rows
                    ).sort_values(
                        [
                            "background_level",
                            "detector",
                            "mag_ab",
                        ]
                    ),
                    args.output,
                )

    grid = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "background_level",
                "detector",
                "mag_ab",
            ]
        )
        .reset_index(drop=True)
    )

    atomic_write_csv(
        grid,
        args.output,
    )

    # ----------------------------------------
    # AB - Vega conversion
    # ----------------------------------------

    ab_minus_vega, conversion_error = (
        compute_ab_minus_vega()
    )

    # ----------------------------------------
    # Survey benchmark
    # ----------------------------------------

    benchmark = grid[
        np.isclose(
            grid["mag_ab"],
            F146_BENCHMARK_AB,
        )
        &
        (
            grid[
                "background_level"
            ]
            == "medium"
        )
    ].copy()

    metadata = {
        "model_name":
            "Roman_F146_Pandeia_R2026.1",

        "pandeia_engine_version":
            engine_version,

        "filter":
            "F146",

        "ma_table":
            "IM_66_6_V2",

        "exposure_time_nominal_s":
            66.0,

        "nexp":
            1,

        "background":
            "gbtds_mid_5stripe",

        "background_levels":
            list(
                args.background_levels
            ),

        "source_sed":
            "flat_fnu",

        "magnitude_grid_system":
            "AB",

        "mag_min":
            float(args.mag_min),

        "mag_max":
            float(args.mag_max),

        "mag_step":
            float(args.mag_step),

        "ab_minus_vega_f146":
            (
                float(
                    ab_minus_vega
                )
                if ab_minus_vega
                is not None
                else None
            ),

        "ab_minus_vega_error":
            conversion_error,

        "benchmark_mag_ab":
            F146_BENCHMARK_AB,

        "benchmark_snr_median":
            (
                float(
                    benchmark[
                        "snr"
                    ].median()
                )
                if len(benchmark)
                else None
            ),

        "benchmark_snr_min":
            (
                float(
                    benchmark[
                        "snr"
                    ].min()
                )
                if len(benchmark)
                else None
            ),

        "benchmark_snr_max":
            (
                float(
                    benchmark[
                        "snr"
                    ].max()
                )
                if len(benchmark)
                else None
            ),

        "notes": [
            (
                "Point-source aperture-photometry "
                "noise model derived from Pandeia."
            ),
            (
                "No additional empirical systematic "
                "floor has been applied."
            ),
            (
                "Crowding/confusion from unrelated "
                "Bulge stars is not included."
            ),
        ],
    }

    with open(
        args.metadata,
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(
        "======================================="
    )
    print(
        "BENCHMARK: F146_AB = 21.2"
    )
    print(
        "======================================="
    )

    if len(benchmark):

        print(
            benchmark[
                [
                    "detector",
                    "snr",
                    "partial_saturated",
                    "full_saturated",
                ]
            ]
            .to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.3f}",
            )
        )

        print()
        print(
            "median S/N =",
            f"{benchmark['snr'].median():.3f}",
        )

        print(
            "range S/N  =",
            (
                f"{benchmark['snr'].min():.3f}"
                " -- "
                f"{benchmark['snr'].max():.3f}"
            ),
        )

    print()
    print(
        "F146 AB - Vega =",
        ab_minus_vega,
    )

    if conversion_error:
        print(
            "WARNING: AB/Vega conversion "
            "could not be calculated:"
        )
        print(
            " ",
            conversion_error,
        )

    print()
    print("Saved:")
    print(" ", args.output)
    print(" ", args.metadata)


if __name__ == "__main__":
    main()
