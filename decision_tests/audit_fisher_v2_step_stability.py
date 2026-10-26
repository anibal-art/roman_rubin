#!/usr/bin/env python3

import copy
import json
from pathlib import Path

import numpy as np

from launch_catalog_simfit_from_config import configure_environment


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "config_files"
    / "gbtds_300_v1_amax_v1.json"
)

config = json.loads(
    CONFIG.read_text()
)

resolved_paths = configure_environment(
    config.get("paths", {}),
    CONFIG.parent,
)


from catalog.event_reader import load_event_from_catalog

from functions_roman_rubin import sim_event

from analysis.fisher_information import (
    compute_fisher,
    rebuild_model,
    telescope_sets,
)


CATALOG = Path(
    config["input"]["event_catalog"]
).expanduser().resolve()

EPHEMERIDES = Path(
    resolved_paths["roman_ephemerides"]
).expanduser().resolve()


CASES = [
    ("FFP", 10, "FSPL"),
    ("BH", 308, "FSPL"),
]


STEP_FACTORS = [
    30.0,
    10.0,
    3.0,
    1.0,
    0.3,
    0.1,
]


def angle_degrees(v1, v2):
    """
    Angle between singular vectors.

    SVD vectors have arbitrary sign, so use |dot|.
    """

    v1 = np.asarray(
        v1,
        dtype=float,
    )

    v2 = np.asarray(
        v2,
        dtype=float,
    )

    cosine = abs(
        float(
            np.dot(
                v1,
                v2,
            )
        )
        / (
            np.linalg.norm(v1)
            * np.linalg.norm(v2)
        )
    )

    cosine = np.clip(
        cosine,
        0.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(
                cosine
            )
        )
    )


def relative_matrix_difference(A, B):
    A = np.asarray(
        A,
        dtype=float,
    )

    B = np.asarray(
        B,
        dtype=float,
    )

    denominator = max(
        np.linalg.norm(B),
        1e-300,
    )

    return float(
        np.linalg.norm(
            A - B
        )
        / denominator
    )


def run_configuration(
    label,
    model,
    params,
):
    print()
    print("=" * 110)
    print("CONFIGURATION:", label)
    print("=" * 110)

    results = {}

    for factor in STEP_FACTORS:

        result = compute_fisher(
            model,
            copy.deepcopy(
                params
            ),
            step_multipliers=(
                factor,
            ),
        )

        Jq = np.asarray(
            result[
                "jacobian_physical_q"
            ],
            dtype=float,
        )

        _, singular_values, Vh = (
            np.linalg.svd(
                Jq,
                full_matrices=False,
            )
        )

        weakest_vector = (
            Vh[-1].copy()
        )

        results[
            factor
        ] = {
            "result":
                result,

            "singular_values":
                singular_values,

            "weakest_vector":
                weakest_vector,
        }

        print()
        print(
            "factor =",
            factor,
        )

        print(
            "  rank =",
            result["rank"],
            "/",
            result["size"],
        )

        print(
            "  s_max =",
            singular_values[0],
        )

        print(
            "  s_min =",
            singular_values[-1],
        )

        print(
            "  s_min/s_max =",
            singular_values[-1]
            / singular_values[0],
        )

        print(
            "  condition(J) =",
            result[
                "raw_condition_number"
            ],
        )

        print(
            "  sigma_piE =",
            result[
                "sigma_piE"
            ],
        )

        print(
            "  frac_sigma_piE =",
            result[
                "fractional_sigma_piE"
            ],
        )

    # ============================================================
    # Use factor=1 as a reference only for comparison.
    # ============================================================

    reference = results[
        1.0
    ]

    print()
    print("-" * 110)
    print("CONVERGENCE RELATIVE TO factor=1")
    print("-" * 110)

    for factor in STEP_FACTORS:

        current = results[
            factor
        ]

        angle = angle_degrees(
            current[
                "weakest_vector"
            ],
            reference[
                "weakest_vector"
            ],
        )

        matrix_difference = (
            relative_matrix_difference(
                current[
                    "result"
                ][
                    "jacobian_physical_q"
                ],
                reference[
                    "result"
                ][
                    "jacobian_physical_q"
                ],
            )
        )

        sigma = current[
            "result"
        ][
            "sigma_piE"
        ]

        sigma_ref = reference[
            "result"
        ][
            "sigma_piE"
        ]

        if (
            np.isfinite(sigma)
            and np.isfinite(
                sigma_ref
            )
            and sigma_ref != 0
        ):
            sigma_ratio = (
                sigma
                / sigma_ref
            )
        else:
            sigma_ratio = np.nan

        print(
            "factor={:5.1f}  "
            "J_rel={:.6e}  "
            "weak_angle_deg={:.6e}  "
            "sigma_ratio={}".format(
                factor,
                matrix_difference,
                angle,
                sigma_ratio,
            )
        )

    # ============================================================
    # Plateau 3, 1, 0.3
    # ============================================================

    plateau_factors = [
        3.0,
        1.0,
        0.3,
    ]

    plateau_sigmas = np.asarray(
        [
            results[f][
                "result"
            ][
                "sigma_piE"
            ]
            for f in plateau_factors
        ],
        dtype=float,
    )

    plateau_smins = np.asarray(
        [
            results[f][
                "singular_values"
            ][-1]
            for f in plateau_factors
        ],
        dtype=float,
    )

    print()
    print("-" * 110)
    print("PLATEAU 3 -> 1 -> 0.3")
    print("-" * 110)

    print(
        "sigma_piE =",
        plateau_sigmas,
    )

    print(
        "s_min =",
        plateau_smins,
    )

    if np.all(
        np.isfinite(
            plateau_sigmas
        )
    ):
        print(
            "sigma max/min =",
            float(
                plateau_sigmas.max()
                / plateau_sigmas.min()
            ),
        )

    print(
        "s_min max/min =",
        float(
            plateau_smins.max()
            / plateau_smins.min()
        ),
    )

    # ============================================================
    # Adaptive calculation used by production v2.
    # ============================================================

    adaptive = compute_fisher(
        model,
        copy.deepcopy(
            params
        ),
    )

    adaptive_s = np.asarray(
        adaptive[
            "singular_values_q"
        ],
        dtype=float,
    )

    print()
    print("-" * 110)
    print("ADAPTIVE V2")
    print("-" * 110)

    print(
        "sigma_piE =",
        adaptive[
            "sigma_piE"
        ],
    )

    print(
        "condition(J) =",
        adaptive[
            "raw_condition_number"
        ],
    )

    print(
        "s_min/s_max =",
        adaptive_s[-1]
        / adaptive_s[0],
    )

    print(
        "derivative stability "
        "max rel L2 =",
        adaptive[
            "derivative_stability_max_rel_l2"
        ],
    )

    print(
        "derivative stability "
        "max rel max =",
        adaptive[
            "derivative_stability_max_rel_max"
        ],
    )

    print(
        "numerical_condition_warning =",
        adaptive[
            "numerical_condition_warning"
        ],
    )

    print()
    print(
        "SELECTED DERIVATIVE STEPS"
    )

    for telescope_name, diagnostics in (
        adaptive[
            "derivative_diagnostics"
        ].items()
    ):

        print(
            " telescope =",
            telescope_name,
        )

        for (
            parameter,
            diagnostic,
        ) in diagnostics.items():

            print(
                "   {:8s} "
                "h={:.6e} "
                "pair_relL2={:.6e} "
                "pair_relMax={:.6e}".format(
                    parameter,
                    diagnostic[
                        "selected_h"
                    ],
                    diagnostic[
                        "pair_relative_l2"
                    ],
                    diagnostic[
                        "pair_relative_max"
                    ],
                )
            )


def run_case(
    population,
    index,
    model_name,
):
    print()
    print("#" * 110)
    print(
        population,
        "catalog_event_index =",
        index,
    )
    print("#" * 110)

    event = load_event_from_catalog(
        CATALOG,
        index,
        expected_population=population,
    )

    seed = int(
        event[
            "simulation_seed"
        ]
    )

    (
        true_model,
        truth,
        decision,
    ) = sim_event(
        seed,
        event,
        str(EPHEMERIDES),
        model_name,
        time_window=None,
        use_roman=True,
        use_rubin=True,
        truth_parallax=True,
        rubin_pointing_mode="source",
        rubin_cache_cell_deg=None,
        apply_detection_criteria=True,
        apply_photometric_filter=True,
        rubin_saturation_mag=None,
        roman_saturation_mag=None,
    )

    assert decision is True
    assert true_model is not None

    (
        roman,
        rubin,
    ) = telescope_sets(
        true_model,
        min_rubin_points=5,
    )

    # ------------------------------------------------------------
    # Roman only
    # ------------------------------------------------------------

    R_model, R_params = rebuild_model(
        true_model,
        truth,
        roman,
        roman_at_earth=False,
    )

    run_configuration(
        f"{population} R",
        R_model,
        R_params,
    )

    # ------------------------------------------------------------
    # Roman + Rubin, only if it differs from R.
    # ------------------------------------------------------------

    if rubin:

        RR_model, RR_params = rebuild_model(
            true_model,
            truth,
            roman + rubin,
            roman_at_earth=False,
        )

        run_configuration(
            f"{population} RR",
            RR_model,
            RR_params,
        )

    else:

        print()
        print(
            population,
            "RR == R: no Rubin band "
            "has >=5 surviving points.",
        )

    # ------------------------------------------------------------
    # D_perp = 0.
    # Important for FFP because its weakest mode changed strongly.
    # ------------------------------------------------------------

    D_model, D_params = rebuild_model(
        true_model,
        truth,
        roman + rubin,
        roman_at_earth=True,
    )

    run_configuration(
        f"{population} RR_DPERP0",
        D_model,
        D_params,
    )


def main():
    print(
        "CONFIG =",
        CONFIG,
    )

    print(
        "CATALOG =",
        CATALOG,
    )

    print(
        "EPHEMERIDES =",
        EPHEMERIDES,
    )

    for case in CASES:
        run_case(
            *case
        )

    print()
    print(
        "=" * 110
    )
    print(
        "FISHER V2 STEP-STABILITY AUDIT FINISHED"
    )
    print(
        "=" * 110
    )


if __name__ == "__main__":
    main()
