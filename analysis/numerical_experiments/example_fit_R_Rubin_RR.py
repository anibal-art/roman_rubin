#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np

from launch_catalog_simfit_from_config import (
    configure_environment,
)

from catalog.event_reader import (
    load_event_from_catalog,
)

from functions_roman_rubin import (
    simulate_event_for_fit,
    extract_lightcurves_for_fit,
    get_model_origin,
    get_event_coordinates,
)

from fit_lc import (
    fit_rubin_roman,
)


# ======================================================================
# Environment
# ======================================================================

ROOT = Path(
    "/share/storage3/rubin/microlensing/romanrubin/roman_rubin"
).resolve()

CONFIG = (
    ROOT
    / "config_files"
    / "gbtds_300_v1_amax_v1.json"
)

cfg = json.loads(
    CONFIG.read_text()
)

resolved = configure_environment(
    cfg.get("paths", {}),
    CONFIG.parent,
)

catalog_path = Path(
    cfg["input"]["event_catalog"]
).expanduser()

if not catalog_path.is_absolute():
    catalog_path = (
        CONFIG.parent
        / catalog_path
    )

catalog_path = catalog_path.resolve()

ephemerides = Path(
    resolved[
        "roman_ephemerides"
    ]
).resolve()


# ======================================================================
# Event
# ======================================================================

INDEX = 308
MODEL = "FSPL"

event = load_event_from_catalog(
    catalog_path=catalog_path,
    catalog_event_index=INDEX,
)

seed = int(
    event.get(
        "simulation_seed",
        INDEX,
    )
)


# ======================================================================
# Simulate exactly once
# ======================================================================

true_model, true_params, decision = (
    simulate_event_for_fit(
        i=seed,
        event_params=event,
        path_ephemerides=str(ephemerides),
        model=MODEL,

        use_roman=True,
        use_rubin=True,

        truth_parallax=True,

        apply_detection_criteria=False,
        apply_photometric_filter=True,

        add_photometric_noise=True,
        rubin_cache_cell_deg=None,
    )
)


# ======================================================================
# Extract exactly the simulated light curves
# ======================================================================

lc, _ = extract_lightcurves_for_fit(
    true_model
)

ra, dec = get_event_coordinates(
    true_model
)

origin = get_model_origin(
    true_model
)


# ======================================================================
# Explicit LOCAL fit setup
# ======================================================================

INITIAL_GUESS = {
    "t0":
        float(event["t0"]),

    "u0":
        float(event["u0"]),

    "tE":
        float(event["tE"]),

    "rho":
        float(event["rho"]),

    "piEN":
        float(event["piEN"]),

    "piEE":
        float(event["piEE"]),
}


BOUNDS = {
    "t0": [
        float(event["t0"]) - 50.0,
        float(event["t0"]) + 50.0,
    ],

    "u0": [
        -2.0,
        2.0,
    ],

    "tE": [
        0.1,
        300.0,
    ],

    "rho": [
        1.0e-6,
        1.0,
    ],

    "piEN": [
        -10.0,
        10.0,
    ],

    "piEE": [
        -10.0,
        10.0,
    ],
}


OPTIMIZER = {
    "xtol":
        1.0e-8,

    "ftol":
        1.0e-8,

    "gtol":
        1.0e-8,

    "max_nfev":
        50000,

    "x_scale":
        "jac",
}


EMPTY = np.empty(
    (0, 3),
    dtype=float,
)


OUT = Path(
    "/tmp/fits_event_308_R_Rubin_RR"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# Generic fit helper
# ======================================================================

def run_one(
    label,
    *,
    use_roman,
    use_rubin,
):

    output = (
        OUT
        / label
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    w149 = (
        lc["W149"]
        if use_roman
        else EMPTY
    )

    ground = {}

    for band in (
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
    ):

        ground[
            band
        ] = (
            lc.get(
                band,
                EMPTY,
            )
            if use_rubin
            else EMPTY
        )

    fit, fit_event, fit_model = (
        fit_rubin_roman(
            seed,
            true_params,

            str(output),
            str(ephemerides),

            MODEL,
            "TRF",

            origin,
            1,

            w149,
            ground["u"],
            ground["g"],
            ground["r"],
            ground["i"],
            ground["z"],
            ground["y"],

            fit_model=MODEL,
            fit_parallax=True,

            fit_bounds=BOUNDS,
            initial_guess=INITIAL_GUESS,
            optimizer_options=OPTIMIZER,

            random_state=seed,

            event_ra=ra,
            event_dec=dec,
        )
    )

    best_vector = np.asarray(
        fit.fit_results[
            "best_model"
        ],
        dtype=float,
    )

    best = (
        fit_model
        .compute_pyLIMA_parameters(
            best_vector
        )
    )

    def value(name):

        if hasattr(
            best,
            name,
        ):
            return float(
                getattr(
                    best,
                    name,
                )
            )

        return float(
            best[name]
        )

    result = {
        key:
            value(key)

        for key in (
            "t0",
            "u0",
            "tE",
            "rho",
            "piEN",
            "piEE",
        )
    }

    result[
        "piE"
    ] = float(
        np.hypot(
            result["piEN"],
            result["piEE"],
        )
    )

    result[
        "chi2"
    ] = float(
        fit.fit_results[
            "chi2"
        ]
    )

    print()
    print("=" * 80)
    print(label)
    print("=" * 80)

    for key, val in result.items():
        print(
            f"{key:10s} = {val}"
        )

    return {
        "fit":
            fit,

        "event":
            fit_event,

        "model":
            fit_model,

        "result":
            result,
    }


# ======================================================================
# Three fits, same noisy realization
# ======================================================================

results = {}

results[
    "R"
] = run_one(
    "R",
    use_roman=True,
    use_rubin=False,
)

results[
    "Rubin"
] = run_one(
    "Rubin",
    use_roman=False,
    use_rubin=True,
)

results[
    "RR"
] = run_one(
    "RR",
    use_roman=True,
    use_rubin=True,
)


print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

for label in (
    "R",
    "Rubin",
    "RR",
):

    r = results[
        label
    ][
        "result"
    ]

    print(
        f"{label:6s} "
        f"piE={r['piE']:.8g} "
        f"chi2={r['chi2']:.8f}"
    )
