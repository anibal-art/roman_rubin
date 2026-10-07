#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

from functions_roman_rubin import simulate_event_for_fit


MODEL = {
    "FFP": "FSPL",
    "BH": "FSPL",
    "Planets_systems": "USBL",
}

BANDS = ("W149", "u", "g", "r", "i", "z", "y")


def _pval(params, key):
    try:
        return params[key]
    except Exception:
        return getattr(params, key, None)


def _aliases(band):
    if band == "y":
        return ("y", "Y")
    return (band,)


def _expected_g(row, band):
    """
    Expected g = Fblend/Fsource from catalog.

    Single authority for the column convention: `catalog.blending.
    blending_columns`, the only function that writes these columns,
    always produces `fsource_{band}` and `ftotal_{band}`. No other
    naming convention exists in production.
    """
    for b in _aliases(band):
        fs_key = f"fsource_{b}"
        ft_key = f"ftotal_{b}"

        if fs_key not in row.index or ft_key not in row.index:
            continue

        if pd.notna(row[fs_key]) and pd.notna(row[ft_key]):
            fs = float(row[fs_key])
            ft = float(row[ft_key])

            if fs > 0:
                return ft / fs - 1.0

    return None


def _actual_g(pyparams, band):
    """
    g = Fblend/Fsource returned by the simulated pyLIMA model.
    """

    for b in _aliases(band):

        fs = _pval(
            pyparams,
            f"fsource_{b}",
        )

        ft = _pval(
            pyparams,
            f"ftotal_{b}",
        )

        if fs is None or ft is None:
            continue

        fs = float(fs)
        ft = float(ft)

        if fs > 0:
            return ft / fs - 1.0

    return None


def check_realization(row, model_obj, pyparams, population):
    """
    Verify that the full simulator used the SAME realization stored
    in the catalog.
    """

    origin_match = True
    expected_origin = None
    actual_origin = None

    if population == "Planets_systems":

        expected_origin = str(
            row["caustic_origin"]
        )

        origin = getattr(
            model_obj,
            "origin",
            None,
        )

        if origin is not None:
            actual_origin = str(
                origin[0]
            )

        origin_match = (
            actual_origin
            == expected_origin
        )

    blend_checks = []

    for band in BANDS:

        expected = _expected_g(
            row,
            band,
        )

        actual = _actual_g(
            pyparams,
            band,
        )

        if expected is None or actual is None:
            continue

        ok = np.isclose(
            expected,
            actual,
            rtol=1e-8,
            atol=1e-10,
        )

        blend_checks.append(
            (
                band,
                expected,
                actual,
                bool(ok),
            )
        )

    blend_available = (
        len(blend_checks) > 0
    )

    blend_match = (
        blend_available
        and all(x[3] for x in blend_checks)
    )

    realization_match = (
        origin_match
        and blend_match
    )

    return {
        "realization_match":
            realization_match,

        "origin_match":
            origin_match,

        "expected_origin":
            expected_origin,

        "actual_origin":
            actual_origin,

        "blend_match":
            blend_match,

        "blend_nbands_checked":
            len(blend_checks),

        "blend_details":
            blend_checks,
    }


def rejection_margin(df):
    """
    Distance in mag to the easiest faint limit.

    For rejected events this should be > 0.
    Small positive values are the dangerous boundary cases.
    """

    arrays = []

    for band in BANDS:

        m = f"m_peak_{band}"
        l = f"limit_mag_{band}"

        if m not in df or l not in df:
            continue

        arrays.append(
            df[m].to_numpy(float)
            - df[l].to_numpy(float)
        )

    if not arrays:
        return np.full(
            len(df),
            np.nan,
        )

    arr = np.column_stack(
        arrays
    )

    with np.errstate(
        all="ignore",
    ):
        return np.nanmin(
            arr,
            axis=1,
        )


def select_events(
    df,
    n_boundary,
    n_random,
    n_stress,
    n_keep,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    reject = df[
        ~df["simulate_amax"].astype(bool)
    ].copy()

    keep = df[
        df["simulate_amax"].astype(bool)
    ].copy()

    reject["_reject_margin_mag"] = (
        rejection_margin(reject)
    )

    selected = []

    # --------------------------------------------------------
    # 1. Most dangerous events:
    #    just barely rejected by Amax
    # --------------------------------------------------------

    boundary = (
        reject
        .sort_values(
            "_reject_margin_mag"
        )
        .head(n_boundary)
        .copy()
    )

    boundary["_test_group"] = (
        "reject_boundary"
    )

    selected.append(
        boundary
    )

    used = set(
        boundary.index
    )

    # --------------------------------------------------------
    # 2. Parallax-stress events
    # --------------------------------------------------------

    remaining = reject[
        ~reject.index.isin(used)
    ].copy()

    if "piE" in remaining:
        pie = remaining[
            "piE"
        ].to_numpy(float)
    else:
        pie = np.hypot(
            remaining[
                "piEN"
            ].to_numpy(float),
            remaining[
                "piEE"
            ].to_numpy(float),
        )

    remaining[
        "_parallax_score"
    ] = (
        np.abs(pie)
        * np.abs(
            remaining[
                "tE"
            ].to_numpy(float)
        )
    )

    stress = (
        remaining
        .sort_values(
            "_parallax_score",
            ascending=False,
        )
        .head(n_stress)
        .copy()
    )

    stress["_test_group"] = (
        "reject_parallax_stress"
    )

    selected.append(
        stress
    )

    used.update(
        stress.index
    )

    # --------------------------------------------------------
    # 3. Random rejected events
    # --------------------------------------------------------

    remaining = reject[
        ~reject.index.isin(used)
    ]

    n = min(
        n_random,
        len(remaining),
    )

    if n:
        random_idx = rng.choice(
            remaining.index.to_numpy(),
            size=n,
            replace=False,
        )

        random_reject = (
            remaining
            .loc[random_idx]
            .copy()
        )

        random_reject["_test_group"] = (
            "reject_random"
        )

        selected.append(
            random_reject
        )

    # --------------------------------------------------------
    # 4. Random kept events: sanity check only
    # --------------------------------------------------------

    n = min(
        n_keep,
        len(keep),
    )

    if n:
        keep_idx = rng.choice(
            keep.index.to_numpy(),
            size=n,
            replace=False,
        )

        keep_sample = (
            keep
            .loc[keep_idx]
            .copy()
        )

        keep_sample[
            "_reject_margin_mag"
        ] = np.nan

        keep_sample["_test_group"] = (
            "keep_random"
        )

        selected.append(
            keep_sample
        )

    out = pd.concat(
        selected,
        axis=0,
    )

    # Deduplicate in case groups overlap unexpectedly.
    out = out[
        ~out.index.duplicated(
            keep="first"
        )
    ]

    return out


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--catalog",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--population",
        required=True,
        choices=tuple(MODEL),
    )

    parser.add_argument(
        "--ephemerides",
        type=Path,
        default=Path(
            "ephemerides/Roman_positions.npy"
        ),
    )

    parser.add_argument(
        "--n-boundary",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--n-random",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--n-stress",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--n-keep",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261006,
    )

    args = parser.parse_args()

    df = pd.read_parquet(
        args.catalog
    )

    required = {
        "simulate_amax",
        "event_seed",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise KeyError(
            f"Missing catalog columns: {sorted(missing)}"
        )

    sample = select_events(
        df=df,
        n_boundary=args.n_boundary,
        n_random=args.n_random,
        n_stress=args.n_stress,
        n_keep=args.n_keep,
        seed=args.seed,
    )

    print()
    print("=" * 78)
    print("Amax vs FULL PIPELINE VALIDATION")
    print("=" * 78)
    print("population :", args.population)
    print("catalog    :", args.catalog)
    print("events     :", len(sample))
    print()

    print(
        sample[
            "_test_group"
        ].value_counts()
    )

    results = []

    model_name = MODEL[
        args.population
    ]

    for counter, (
        row_index,
        row,
    ) in enumerate(
        sample.iterrows(),
        start=1,
    ):

        event_seed = int(
            row["event_seed"]
        )

        event_params = row.to_dict()

        # pandas/numpy scalars -> normal Python scalars when possible
        for key, value in list(
            event_params.items()
        ):
            if hasattr(
                value,
                "item",
            ):
                try:
                    event_params[key] = (
                        value.item()
                    )
                except Exception:
                    pass

        log = io.StringIO()

        error = None
        model_obj = None
        pyparams = None
        decision = False

        try:

            with contextlib.redirect_stdout(
                log
            ):

                (
                    model_obj,
                    pyparams,
                    decision,
                ) = simulate_event_for_fit(
                    i=event_seed,
                    event_params=event_params,
                    path_ephemerides=str(
                        args.ephemerides
                    ),
                    model=model_name,
                    use_roman=True,
                    use_rubin=True,
                    truth_parallax=True,
                    apply_detection_criteria=True,
                    apply_photometric_filter=True,
                )

            decision = bool(
                decision
            )

        except Exception as exc:

            error = (
                f"{type(exc).__name__}: {exc}"
            )

        realization = {
            "realization_match": False,
            "origin_match": False,
            "expected_origin": None,
            "actual_origin": None,
            "blend_match": False,
            "blend_nbands_checked": 0,
            "blend_details": [],
        }

        if error is None:

            realization = (
                check_realization(
                    row=row,
                    model_obj=model_obj,
                    pyparams=pyparams,
                    population=args.population,
                )
            )

        prefilter = bool(
            row["simulate_amax"]
        )

        false_reject = (
            (not prefilter)
            and decision
        )

        result = {
            "row_index":
                int(row_index),

            "event_seed":
                event_seed,

            "test_group":
                row["_test_group"],

            "simulate_amax":
                prefilter,

            "pipeline_detected":
                decision,

            "false_reject":
                false_reject,

            "realization_match":
                realization[
                    "realization_match"
                ],

            "origin_match":
                realization[
                    "origin_match"
                ],

            "expected_origin":
                realization[
                    "expected_origin"
                ],

            "actual_origin":
                realization[
                    "actual_origin"
                ],

            "blend_match":
                realization[
                    "blend_match"
                ],

            "blend_nbands_checked":
                realization[
                    "blend_nbands_checked"
                ],

            "reject_margin_mag":
                row.get(
                    "_reject_margin_mag",
                    np.nan,
                ),

            "Amax_catalog":
                row.get(
                    "Amax_catalog",
                    np.nan,
                ),

            "tE":
                row.get(
                    "tE",
                    np.nan,
                ),

            "piE":
                row.get(
                    "piE",
                    np.nan,
                ),

            "error":
                error,
        }

        results.append(
            result
        )

        print(
            f"[{counter:3d}/{len(sample):3d}] "
            f"{result['test_group']:23s} "
            f"prefilter={prefilter!s:5s} "
            f"pipeline={decision!s:5s} "
            f"realization={result['realization_match']!s:5s} "
            f"FALSE_REJECT={false_reject!s:5s}"
        )

        # Fail immediately if we are not comparing the same event.
        if (
            error is None
            and not realization[
                "realization_match"
            ]
        ):
            print()
            print("REALIZATION MISMATCH")
            print(
                "origin:",
                realization[
                    "expected_origin"
                ],
                "->",
                realization[
                    "actual_origin"
                ],
            )

            print(
                "blend:",
                realization[
                    "blend_details"
                ],
            )

            raise SystemExit(3)

    result_df = pd.DataFrame(
        results
    )

    output = (
        args.catalog.parent
        / (
            args.catalog.stem
            + "_amax_validation.parquet"
        )
    )

    result_df.to_parquet(
        output,
        index=False,
    )

    print()
    print("=" * 78)
    print("CONFUSION MATRIX")
    print("=" * 78)

    valid = result_df[
        result_df["error"].isna()
    ]

    print(
        pd.crosstab(
            valid["simulate_amax"],
            valid["pipeline_detected"],
            rownames=[
                "simulate_amax"
            ],
            colnames=[
                "pipeline_detected"
            ],
            dropna=False,
        )
    )

    n_errors = int(
        result_df[
            "error"
        ].notna().sum()
    )

    n_mismatch = int(
        (
            ~valid[
                "realization_match"
            ]
        ).sum()
    )

    n_false_reject = int(
        valid[
            "false_reject"
        ].sum()
    )

    detected = int(
        valid[
            "pipeline_detected"
        ].sum()
    )

    captured = int(
        (
            valid[
                "pipeline_detected"
            ]
            &
            valid[
                "simulate_amax"
            ]
        ).sum()
    )

    recall = (
        captured / detected
        if detected
        else np.nan
    )

    print()
    print("errors               =", n_errors)
    print("realization mismatch =", n_mismatch)
    print("FALSE REJECTS        =", n_false_reject)
    print("detected by pipeline =", detected)
    print("captured by Amax     =", captured)
    print("Amax recall          =", recall)
    print("saved                =", output)

    if n_errors:
        print()
        print(
            result_df[
                result_df[
                    "error"
                ].notna()
            ][
                [
                    "event_seed",
                    "test_group",
                    "error",
                ]
            ].to_string(
                index=False
            )
        )

        raise SystemExit(4)

    if n_mismatch:
        raise SystemExit(3)

    if n_false_reject:
        print()
        print(
            "FAIL: the Amax catalog filter discarded "
            "events detected by the full pipeline."
        )

        print(
            result_df[
                result_df[
                    "false_reject"
                ]
            ].to_string(
                index=False
            )
        )

        raise SystemExit(2)

    print()
    print(
        "PASS: no Amax-rejected event was detected "
        "by the full noisy pipeline in this sample."
    )


if __name__ == "__main__":
    main()
