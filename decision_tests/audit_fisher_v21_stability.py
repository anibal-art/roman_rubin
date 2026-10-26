#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np

from launch_catalog_simfit_from_config import configure_environment


ROOT = Path(__file__).resolve().parents[1]

CONFIG = ROOT / "config_files" / "gbtds_300_v1_amax_v1.json"

config = json.loads(CONFIG.read_text())

resolved_paths = configure_environment(
    config.get("paths", {}),
    CONFIG.parent,
)


from catalog.event_reader import load_event_from_catalog
from functions_roman_rubin import sim_event

from analysis.fisher_information import compute_e1_suite


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


STABILITY_KEYS = [
    "fd_stability_sigma_piE_rel",
    "fd_stability_smin_rel",
    "fd_stability_weak_mode_angle_deg",
    "fd_stability_jacobian_rel",
]


def show_result(population, label, result):

    print()
    print("-" * 90)
    print(population, label)
    print("-" * 90)

    print("method =", result["method"])
    print("rank =", result["rank"], "/", result["size"])
    print("sigma_piE primary   =", result["sigma_piE"])
    print("sigma_piE alternate =", result["sigma_piE_alternate"])
    print("s_min/s_max =", result["singular_value_ratio"])
    print("condition(J) =", result["raw_condition_number"])
    print(
        "numerical_condition_warning =",
        result["numerical_condition_warning"],
    )

    assert (
        result["method"]
        ==
        "five_point_stable_pair"
        "+analytic_flux"
        "+nuisance_projection"
        "+svd"
    )

    assert result["rank"] == result["size"]

    assert np.isfinite(result["sigma_piE"])
    assert np.isfinite(result["sigma_piE_alternate"])

    for key in STABILITY_KEYS:

        value = float(result[key])

        print(key, "=", value)

        assert np.isfinite(value)
        assert value >= 0.0

    assert not result["has_significant_negative_eigenvalue"]


def run_case(population, index, model_name):

    print()
    print("=" * 100)
    print(population, "catalog_event_index =", index)
    print("=" * 100)

    event = load_event_from_catalog(
        CATALOG,
        index,
        expected_population=population,
    )

    seed = int(event["simulation_seed"])

    true_model, truth, decision = sim_event(
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

    assert bool(decision)
    assert true_model is not None

    suite = compute_e1_suite(
        true_model,
        truth,
        min_rubin_points=5,
    )

    configs = suite["configurations"]

    for label in ["R", "RR", "RR_DPERP0"]:
        show_result(
            population,
            label,
            configs[label],
        )

    if population == "FFP":

        print()
        print("FFP negative control")
        print(
            "Rubin bands used =",
            suite["rubin_telescope_names_used"],
        )

        assert suite["rubin_telescope_names_used"] == []

        assert np.array_equal(
            configs["R"]["jacobian_physical_q"],
            configs["RR"]["jacobian_physical_q"],
        )

        assert np.array_equal(
            configs["R"]["jacobian_physical_q_alternate"],
            configs["RR"]["jacobian_physical_q_alternate"],
        )

        assert (
            configs["R"]["sigma_piE"]
            ==
            configs["RR"]["sigma_piE"]
        )

        print("PASS: RR == R exactly")

    if population == "BH":

        print()
        print("BH Rubin control")
        print(
            "Rubin bands used =",
            suite["rubin_telescope_names_used"],
        )

        assert suite["rubin_telescope_names_used"] == ["z"]

        assert configs["R"]["fractional_sigma_piE"] > 1.0

        print(
            "PASS: piE finite but poorly characterized"
        )


def main():

    print("CONFIG =", CONFIG)
    print("CATALOG =", CATALOG)
    print("EPHEMERIDES =", EPHEMERIDES)

    for case in CASES:
        run_case(*case)

    print()
    print("=" * 100)
    print("ALL FISHER V2.1 STABILITY AUDITS PASSED")
    print("=" * 100)


if __name__ == "__main__":
    main()
