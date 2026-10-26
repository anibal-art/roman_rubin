#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd

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
# Configuration
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

catalog_path = (
    catalog_path.resolve()
)

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
# Extract same noisy realization
# ======================================================================

lc_original, _ = (
    extract_lightcurves_for_fit(
        true_model
    )
)

ra, dec = (
    get_event_coordinates(
        true_model
    )
)

origin = (
    get_model_origin(
        true_model
    )
)


EMPTY = np.empty(
    (0, 3),
    dtype=float,
)


def copy_lc(
    source,
):

    out = {}

    for band in (
        "W149",
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
    ):

        arr = np.asarray(
            source.get(
                band,
                EMPTY,
            ),
            dtype=float,
        )

        if arr.size == 0:
            arr = (
                EMPTY.copy()
            )

        out[
            band
        ] = arr.copy()

    return out


lc_original = copy_lc(
    lc_original
)


# ======================================================================
# Controlled local fit
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


OUT = Path(
    "/tmp/loo_event_308"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================
# Fit helper
# ======================================================================

def fit_joint(
    lightcurves,
    output_dir,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fit, fit_event, model_fit = (
        fit_rubin_roman(
            seed,
            true_params,

            str(output_dir),
            str(ephemerides),

            MODEL,
            "TRF",

            origin,
            1,

            lightcurves[
                "W149"
            ],

            lightcurves["u"],
            lightcurves["g"],
            lightcurves["r"],
            lightcurves["i"],
            lightcurves["z"],
            lightcurves["y"],

            fit_model=MODEL,
            fit_parallax=True,

            fit_bounds=BOUNDS,
            initial_guess=INITIAL_GUESS,
            optimizer_options=OPTIMIZER,

            # Same numerical start policy for every ablation.
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

    pybest = (
        model_fit
        .compute_pyLIMA_parameters(
            best_vector
        )
    )

    def value(
        name,
    ):

        if hasattr(
            pybest,
            name,
        ):
            return float(
                getattr(
                    pybest,
                    name,
                )
            )

        return float(
            pybest[
                name
            ]
        )

    piEN = value(
        "piEN"
    )

    piEE = value(
        "piEE"
    )

    piE = float(
        np.hypot(
            piEN,
            piEE,
        )
    )

    # --------------------------------------------------------------
    # Propagate covariance to piE.
    # --------------------------------------------------------------

    sigma_piE = np.nan

    covariance = fit.fit_results.get(
        "covariance_matrix",
        None,
    )

    model_dict = getattr(
        model_fit,
        "model_dictionnary",
        {},
    )

    if (
        covariance is not None
        and "piEN" in model_dict
        and "piEE" in model_dict
        and piE > 0
    ):

        covariance = np.asarray(
            covariance,
            dtype=float,
        )

        iN = int(
            model_dict[
                "piEN"
            ]
        )

        iE = int(
            model_dict[
                "piEE"
            ]
        )

        C = covariance[
            np.ix_(
                [
                    iN,
                    iE,
                ],
                [
                    iN,
                    iE,
                ],
            )
        ]

        grad = np.array(
            [
                piEN / piE,
                piEE / piE,
            ],
            dtype=float,
        )

        variance = float(
            grad
            @ C
            @ grad
        )

        if variance >= 0:
            sigma_piE = float(
                np.sqrt(
                    variance
                )
            )

    return {
        "fit":
            fit,

        "model":
            model_fit,

        "piEN":
            piEN,

        "piEE":
            piEE,

        "piE":
            piE,

        "sigma_piE":
            sigma_piE,

        "chi2":
            float(
                fit.fit_results[
                    "chi2"
                ]
            ),
    }


# ======================================================================
# Baseline full RR fit
# ======================================================================

print()
print("=" * 100)
print("FULL RR FIT")
print("=" * 100)

baseline = fit_joint(
    lc_original,
    OUT
    / "full",
)

print(
    "piE        =",
    baseline[
        "piE"
    ],
)

print(
    "sigma_piE  =",
    baseline[
        "sigma_piE"
    ],
)

print(
    "chi2       =",
    baseline[
        "chi2"
    ],
)


# ======================================================================
# Inventory Rubin points
# ======================================================================

rubin_points = []

for band in (
    "u",
    "g",
    "r",
    "i",
    "z",
    "y",
):

    arr = lc_original[
        band
    ]

    for row_index in range(
        len(arr)
    ):

        rubin_points.append(
            (
                band,
                row_index,
                float(
                    arr[
                        row_index,
                        0,
                    ]
                ),
            )
        )


print()
print("=" * 100)
print("RUBIN POINTS")
print("=" * 100)

print(
    "N =",
    len(
        rubin_points
    ),
)

for item in rubin_points:
    print(item)


# ======================================================================
# Leave one Rubin point out
# ======================================================================

rows = []

for global_index, (
    band,
    row_index,
    time_jd,
) in enumerate(
    rubin_points
):

    print()
    print("=" * 100)

    print(
        f"LOO {global_index + 1}/"
        f"{len(rubin_points)}"
    )

    print(
        "remove:",
        band,
        row_index,
        time_jd,
    )

    print("=" * 100)

    lc_loo = copy_lc(
        lc_original
    )

    lc_loo[
        band
    ] = np.delete(
        lc_loo[
            band
        ],
        row_index,
        axis=0,
    )

    result = fit_joint(
        lc_loo,
        (
            OUT
            / (
                f"loo_{global_index:04d}_"
                f"{band}_{row_index:04d}"
            )
        ),
    )

    delta_piE = (
        result["piE"]
        - baseline["piE"]
    )

    if (
        np.isfinite(
            baseline[
                "sigma_piE"
            ]
        )
        and baseline[
            "sigma_piE"
        ] > 0
    ):

        delta_piE_sigma = (
            abs(
                delta_piE
            )
            / baseline[
                "sigma_piE"
            ]
        )

    else:

        delta_piE_sigma = np.nan

    if (
        np.isfinite(
            result[
                "sigma_piE"
            ]
        )
        and np.isfinite(
            baseline[
                "sigma_piE"
            ]
        )
        and baseline[
            "sigma_piE"
        ] > 0
    ):

        sigma_ratio = (
            result[
                "sigma_piE"
            ]
            / baseline[
                "sigma_piE"
            ]
        )

    else:

        sigma_ratio = np.nan

    rows.append(
        {
            "removed_global_index":
                global_index,

            "band":
                band,

            "band_row_index":
                row_index,

            "time_jd":
                time_jd,

            "n_band_before":
                int(
                    len(
                        lc_original[
                            band
                        ]
                    )
                ),

            "n_band_after":
                int(
                    len(
                        lc_loo[
                            band
                        ]
                    )
                ),

            "piE_full":
                baseline[
                    "piE"
                ],

            "piE_minus_i":
                result[
                    "piE"
                ],

            "delta_piE":
                delta_piE,

            "abs_delta_piE_over_sigma_full":
                delta_piE_sigma,

            "sigma_piE_full":
                baseline[
                    "sigma_piE"
                ],

            "sigma_piE_minus_i":
                result[
                    "sigma_piE"
                ],

            "sigma_ratio_minus_i_over_full":
                sigma_ratio,

            "chi2_full":
                baseline[
                    "chi2"
                ],

            "chi2_minus_i":
                result[
                    "chi2"
                ],

            "delta_chi2":
                (
                    result[
                        "chi2"
                    ]
                    - baseline[
                        "chi2"
                    ]
                ),
        }
    )


# ======================================================================
# Save / rank points by impact
# ======================================================================

df = pd.DataFrame(
    rows
)

output_csv = (
    OUT
    / "leave_one_out.csv"
)

df.to_csv(
    output_csv,
    index=False,
)


print()
print("=" * 100)
print("MOST INFLUENTIAL POINTS")
print("=" * 100)

print(
    df.sort_values(
        "abs_delta_piE_over_sigma_full",
        ascending=False,
    ).to_string(
        index=False
    )
)

print()
print(
    "saved:",
    output_csv,
)
