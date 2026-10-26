#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np

# ============================================================
# Configure the EXACT active production environment BEFORE
# importing functions_roman_rubin / set_telescopes_pyLIMA.
# ============================================================

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
    config.get(
        "paths",
        {},
    ),
    CONFIG.parent,
)


# Only now import code that consumes the configured environment.
from catalog.event_reader import load_event_from_catalog
from functions_roman_rubin import sim_event
from analysis.fisher_information import (
    compute_e1_suite,
    rebuild_model,
)


CATALOG = Path(
    config["input"]["event_catalog"]
).expanduser().resolve()

EPHEMERIDES = Path(
    resolved_paths["roman_ephemerides"]
).expanduser().resolve()


# These were previously found to pass the real v5.3.5 simulation
# selection in the active materialized catalogue.
CASES = [
    {
        "population": "FFP",
        "catalog_event_index": 10,
        "model": "FSPL",
    },
    {
        "population": "BH",
        "catalog_event_index": 308,
        "model": "FSPL",
    },
]


def arr(x):
    if hasattr(x, "value"):
        x = x.value

    return np.asarray(
        x,
        dtype=float,
    )


def matrix_scale(F):
    F = np.asarray(
        F,
        dtype=float,
    )

    return max(
        1.0,
        float(
            np.max(
                np.abs(F)
            )
        ),
    )


def print_result(
    label,
    result,
):
    F = np.asarray(
        result["fisher_physical_q"],
        dtype=float,
    )

    eig = np.linalg.eigvalsh(
        0.5
        * (
            F
            + F.T
        )
    )

    scale = matrix_scale(F)

    print()
    print(
        "=" * 72
    )
    print(label)
    print(
        "=" * 72
    )

    print(
        "telescopes =",
        result["telescope_names"],
    )

    print(
        "N points =",
        result["telescope_n_points"],
    )

    print(
        "physical parameters =",
        result["physical_names"],
    )

    print(
        "nuisance parameters =",
        result["nuisance_names"],
    )

    print(
        "rank / size =",
        result["rank"],
        "/",
        result["size"],
    )

    print(
        "condition number =",
        result["condition_number"],
    )

    print(
        "nuisance rank / size =",
        result["nuisance_rank"],
        "/",
        result["nuisance_size"],
    )

    print(
        "symmetry max abs =",
        result["symmetry_max_abs"],
    )

    print(
        "additivity max abs =",
        result["additivity_max_abs"],
    )

    print(
        "min eigenvalue =",
        eig.min(),
    )

    print(
        "max eigenvalue =",
        eig.max(),
    )

    print(
        "min eigenvalue / scale =",
        eig.min() / scale,
    )

    print(
        "significant negative eigenvalue =",
        result[
            "has_significant_negative_eigenvalue"
        ],
    )

    print(
        "piE =",
        result["piE"],
    )

    print(
        "sigma_piE =",
        result["sigma_piE"],
    )

    print(
        "fractional sigma_piE =",
        result[
            "fractional_sigma_piE"
        ],
    )

    print(
        "sigmas:"
    )

    for name in result["physical_names"]:
        print(
            f"  {name:8s}",
            result["sigma"][name],
        )

    assert (
        result[
            "symmetry_max_abs"
        ]
        / scale
        < 1e-12
    )

    assert (
        result[
            "additivity_max_abs"
        ]
        / scale
        < 1e-10
    )

    assert not result[
        "has_significant_negative_eigenvalue"
    ]


def audit_dperp0(
    true_model,
    truth,
    telescope_names,
):
    model, _ = rebuild_model(
        true_model,
        truth,
        telescope_names,
        roman_at_earth=True,
    )

    roman = [
        tel
        for tel in model.event.telescopes
        if (
            str(tel.name).lower()
            in {
                "roman",
                "w149",
                "f146",
            }
            or
            str(tel.filter).lower()
            in {
                "w149",
                "f146",
            }
        )
    ]

    assert len(roman) == 1

    tel = roman[0]

    print()
    print(
        "D_PERP=0 CHECK"
    )
    print(
        "-" * 72
    )

    print(
        "Roman location =",
        tel.location,
    )

    print(
        "Roman altitude =",
        tel.altitude,
    )

    print(
        "spacecraft_positions size =",
        np.asarray(
            tel.spacecraft_positions[
                "photometry"
            ]
        ).size,
    )

    pos = arr(
        tel.telescope_positions[
            "photometry"
        ]
    )

    norms = np.linalg.norm(
        pos,
        axis=1,
    )

    print(
        "topocentric norm min/median/max =",
        norms.min(),
        np.median(norms),
        norms.max(),
    )

    assert tel.location == "Earth"

    assert (
        np.asarray(
            tel.spacecraft_positions[
                "photometry"
            ]
        ).size
        == 0
    )

    assert np.allclose(
        norms,
        0.0,
        rtol=0.0,
        atol=0.0,
    )


def run_case(case):
    population = case["population"]

    catalog_event_index = int(
        case[
            "catalog_event_index"
        ]
    )

    model_name = case["model"]

    print()
    print(
        "#" * 100
    )

    print(
        population,
        "catalog_event_index =",
        catalog_event_index,
    )

    print(
        "#" * 100
    )

    event = load_event_from_catalog(
        CATALOG,
        catalog_event_index,
        expected_population=population,
    )

    assert (
        int(
            event[
                "catalog_event_index"
            ]
        )
        == catalog_event_index
    )

    seed = int(
        event[
            "simulation_seed"
        ]
    )

    print(
        "catalog_event_index =",
        event[
            "catalog_event_index"
        ],
    )

    print(
        "simulation_seed =",
        seed,
    )

    print(
        "field_id =",
        event["field_id"],
    )

    print(
        "t0 =",
        event["t0"],
    )

    print(
        "u0 =",
        event["u0"],
    )

    print(
        "tE =",
        event["tE"],
    )

    print(
        "rho =",
        event["rho"],
    )

    print(
        "piEN =",
        event["piEN"],
    )

    print(
        "piEE =",
        event["piEE"],
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

    print()
    print(
        "decision =",
        decision,
    )

    assert decision is True

    assert true_model is not None

    print(
        "pyLIMA model type =",
        true_model.model_type(),
    )

    print(
        "Jacobian flag =",
        getattr(
            true_model,
            "Jacobian_flag",
            None,
        ),
    )

    print(
        "parallax model =",
        true_model.parallax_model,
    )

    print(
        "telescopes:"
    )

    for tel in true_model.event.telescopes:
        print(
            f"  {tel.name:8s}",
            f"filter={tel.filter}",
            f"location={tel.location}",
            f"N={len(tel.lightcurve)}",
        )

    suite = compute_e1_suite(
        true_model,
        truth,
        min_rubin_points=5,
        rcond=1e-12,
    )

    print()
    print(
        "Rubin bands used =",
        suite[
            "rubin_telescope_names_used"
        ],
    )

    R = suite[
        "configurations"
    ]["R"]

    RR = suite[
        "configurations"
    ]["RR"]

    E = suite[
        "configurations"
    ]["RR_DPERP0"]

    print_result(
        "R",
        R,
    )

    print_result(
        "RR",
        RR,
    )

    print_result(
        "RR_DPERP0",
        E,
    )

    # --------------------------------------------------------
    # Same dimensional physical problem in all three cases.
    # --------------------------------------------------------

    assert (
        R["physical_names"]
        ==
        RR["physical_names"]
        ==
        E["physical_names"]
    )

    assert (
        RR["n_data"]
        >=
        R["n_data"]
    )

    assert (
        RR["n_data"]
        ==
        E["n_data"]
    )

    # --------------------------------------------------------
    # Adding independent Rubin data cannot reduce Fisher
    # information, except for numerical roundoff.
    # --------------------------------------------------------

    delta = (
        RR[
            "fisher_physical_q"
        ]
        -
        R[
            "fisher_physical_q"
        ]
    )

    delta = (
        0.5
        * (
            delta
            + delta.T
        )
    )

    eig_delta = np.linalg.eigvalsh(
        delta
    )

    scale_delta = matrix_scale(
        delta
    )

    print()
    print(
        "F_RR - F_R"
    )
    print(
        "-" * 72
    )

    print(
        "min eigenvalue =",
        eig_delta.min(),
    )

    print(
        "max eigenvalue =",
        eig_delta.max(),
    )

    print(
        "min eigenvalue / scale =",
        eig_delta.min()
        / scale_delta,
    )

    assert (
        eig_delta.min()
        / scale_delta
        > -1e-8
    )

    print()
    print(
        "GAINS"
    )
    print(
        "-" * 72
    )

    for comparison, values in suite[
        "gains"
    ].items():

        print(
            comparison
        )

        for name, gain in values.items():
            print(
                f"  {name:8s}",
                gain,
            )

    audit_dperp0(
        true_model,
        truth,
        RR[
            "telescope_names"
        ],
    )

    print()
    print(
        "CASE PASS:",
        population,
        catalog_event_index,
    )

    return suite


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

    assert CONFIG.is_file()
    assert CATALOG.is_file()
    assert EPHEMERIDES.is_file()

    results = {}

    for case in CASES:
        results[
            case["population"]
        ] = run_case(
            case
        )

    print()
    print(
        "=" * 100
    )
    print(
        "FINAL SUMMARY"
    )
    print(
        "=" * 100
    )

    for population, suite in results.items():

        R = suite[
            "configurations"
        ]["R"]

        RR = suite[
            "configurations"
        ]["RR"]

        E = suite[
            "configurations"
        ]["RR_DPERP0"]

        print()
        print(
            population
        )

        print(
            "sigma_piE R         =",
            R["sigma_piE"],
        )

        print(
            "sigma_piE RR        =",
            RR["sigma_piE"],
        )

        print(
            "sigma_piE DPERP0    =",
            E["sigma_piE"],
        )

        print(
            "gain R -> RR        =",
            suite[
                "gains"
            ][
                "R_to_RR"
            ]["piE"],
        )

        print(
            "satellite gain      =",
            suite[
                "gains"
            ][
                "RR_DPERP0_to_RR"
            ]["piE"],
        )

    print()
    print(
        "ALL PRODUCTION-CONSISTENT "
        "REAL-EVENT FISHER AUDITS PASSED"
    )


if __name__ == "__main__":
    main()
