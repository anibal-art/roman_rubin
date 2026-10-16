#!/usr/bin/env python3

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "/share/storage3/rubin/microlensing/romanrubin/roman_rubin"
).resolve()

CONFIG = ROOT / "config_files/gbtds_300_v1_amax_v1.json"

OUT = Path(
    "/tmp/exp1_cumulative_two_events.parquet"
)


# ============================================================
# Production environment
# ============================================================

from launch_catalog_simfit_from_config import configure_environment

cfg = json.loads(CONFIG.read_text())

resolved_paths = configure_environment(
    cfg.get("paths", {}),
    CONFIG.parent,
)

catalog_path = Path(
    cfg["input"]["event_catalog"]
).expanduser().resolve()

path_ephemerides = Path(
    resolved_paths["roman_ephemerides"]
).expanduser().resolve()


from catalog.event_reader import load_event_from_catalog

from functions_roman_rubin import (
    build_event_params_from_custom_system,
    simulate_event_for_fit,
)

from analysis.fisher_information import (
    rebuild_model,
    compute_fisher,
)


CASES = [
    ("FFP", 10, "FSPL"),
    ("BH", 308, "FSPL"),
]

RUBIN_BANDS = (
    "u",
    "g",
    "r",
    "i",
    "z",
    "y",
)


def arr(x):
    return np.asarray(
        getattr(x, "value", x),
        dtype=float,
    )


def get_param(params, name):
    try:
        return float(params[name])
    except Exception:
        return float(getattr(params, name))


def chronological_points(
    model,
    t0,
    tE,
):

    points = []

    for tel in model.event.telescopes:

        if (
            tel.name == "W149"
            or tel.lightcurve is None
            or len(tel.lightcurve) == 0
        ):
            continue

        times = arr(
            tel.lightcurve["time"]
        )

        for j, time_jd in enumerate(times):

            points.append(
                {
                    "band": tel.name,
                    "local_index": int(j),
                    "time_jd": float(time_jd),
                    "tau": float(
                        (
                            float(time_jd)
                            - t0
                        )
                        / tE
                    ),
                }
            )

    points.sort(
        key=lambda p: (
            p["time_jd"],
            p["band"],
            p["local_index"],
        )
    )

    return points


def subset_model(
    source_model,
    points,
):

    model = copy.deepcopy(
        source_model
    )

    selected = {}

    for point in points:

        selected.setdefault(
            point["band"],
            [],
        ).append(
            point["local_index"]
        )

    kept = []

    for tel in model.event.telescopes:

        if (
            tel.lightcurve is None
            or len(tel.lightcurve) == 0
        ):
            continue

        if tel.name == "W149":

            kept.append(tel)
            continue

        indices = selected.get(
            tel.name,
            [],
        )

        if not indices:
            continue

        tel.lightcurve = tel.lightcurve[
            np.asarray(
                indices,
                dtype=int,
            )
        ]

        kept.append(tel)

    model.event.telescopes = kept

    return model


def fisher_for_points(
    source_model,
    true_params,
    points,
):

    source = subset_model(
        source_model,
        points,
    )

    names = [
        tel.name
        for tel in source.event.telescopes
    ]

    model, params = rebuild_model(
        source,
        true_params,
        names,
        roman_at_earth=False,
    )

    result = compute_fisher(
        model,
        params,
        physical_derivative_method="auto",
    )

    return result


def safe_gain(a, b):

    a = float(a)
    b = float(b)

    if (
        np.isfinite(a)
        and np.isfinite(b)
        and b > 0
    ):
        return a / b

    return np.nan


rows = []


for (
    population,
    index,
    model_name,
) in CASES:

    print()
    print("=" * 100)
    print(
        population,
        "catalog_event_index=",
        index,
    )
    print("=" * 100)

    catalog_row = load_event_from_catalog(
        catalog_path,
        index,
        expected_population=population,
    )

    event_params = (
        build_event_params_from_custom_system(
            custom_system=catalog_row,
            model=model_name,
            use_roman=True,
            use_rubin=True,
            rubin_pointing_mode="source",
        )
    )

    seed = int(
        catalog_row["simulation_seed"]
    )

    (
        true_model,
        true_params,
        detected,
    ) = simulate_event_for_fit(
        seed,
        event_params,
        path_ephemerides,
        model_name,
        time_window=None,
        use_roman=True,
        use_rubin=True,
        truth_parallax=True,
        rubin_pointing_mode="source",
        rubin_cache_cell_deg=None,
        apply_detection_criteria=True,
        apply_photometric_filter=True,
    )

    if not detected:
        raise RuntimeError(
            f"{population} {index} "
            "is no longer Roman-detected."
        )

    t0 = get_param(
        true_params,
        "t0",
    )

    tE = get_param(
        true_params,
        "tE",
    )

    piEN = get_param(
        true_params,
        "piEN",
    )

    piEE = get_param(
        true_params,
        "piEE",
    )

    piE_true = float(
        np.hypot(
            piEN,
            piEE,
        )
    )

    points = chronological_points(
        true_model,
        t0,
        tE,
    )

    print(
        "Rubin points =",
        len(points),
    )

    roman = fisher_for_points(
        true_model,
        true_params,
        [],
    )

    sigma_R = float(
        roman["sigma_piE"]
    )

    sigma_R_alt = float(
        roman[
            "sigma_piE_alternate"
        ]
    )

    previous_F = None

    counts_band = {
        band: 0
        for band in RUBIN_BANDS
    }

    n_peak = 0
    n_wings = 0
    n_outer = 0

    for N in range(
        len(points) + 1
    ):

        if N == 0:

            result = roman

            added_band = None
            added_time = np.nan
            added_tau = np.nan

        else:

            point = points[
                N - 1
            ]

            added_band = point["band"]
            added_time = point["time_jd"]
            added_tau = point["tau"]

            counts_band[
                added_band
            ] += 1

            abs_tau = abs(
                added_tau
            )

            if abs_tau <= 1.0:
                n_peak += 1

            elif abs_tau <= 2.0:
                n_wings += 1

            else:
                n_outer += 1

            result = fisher_for_points(
                true_model,
                true_params,
                points[:N],
            )

        sigma = float(
            result["sigma_piE"]
        )

        sigma_alt = float(
            result[
                "sigma_piE_alternate"
            ]
        )

        gain = safe_gain(
            sigma_R,
            sigma,
        )

        gain_alt = safe_gain(
            sigma_R_alt,
            sigma_alt,
        )

        F = np.asarray(
            result[
                "fisher_physical_q"
            ],
            dtype=float,
        )

        if previous_F is None:

            min_eig_dF = np.nan
            min_eig_dF_rel = np.nan

        else:

            delta = (
                F
                - previous_F
            )

            delta = (
                delta
                + delta.T
            ) / 2.0

            eig = np.linalg.eigvalsh(
                delta
            )

            min_eig_dF = float(
                np.min(eig)
            )

            scale = max(
                1.0,
                float(
                    np.max(
                        np.abs(F)
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            previous_F
                        )
                    )
                ),
            )

            min_eig_dF_rel = (
                min_eig_dF
                / scale
            )

        previous_F = F

        row = {
            "population":
                population,

            "catalog_event_index":
                index,

            "field_id":
                str(
                    catalog_row.get(
                        "field_id",
                        "",
                    )
                ),

            "N":
                N,

            "N_rubin_total":
                len(points),

            "added_band":
                added_band,

            "added_time_jd":
                added_time,

            "added_tau":
                added_tau,

            "n_peak":
                n_peak,

            "n_wings":
                n_wings,

            "n_outer":
                n_outer,

            "t0":
                t0,

            "tE":
                tE,

            "piE_true":
                piE_true,

            "sigma_piE":
                sigma,

            "sigma_piE_alternate":
                sigma_alt,

            "gain_piE":
                gain,

            "gain_piE_alternate":
                gain_alt,

            "gain_primary_minus_alternate":
                (
                    gain
                    - gain_alt
                ),

            "fractional_sigma_piE":
                float(
                    result[
                        "fractional_sigma_piE"
                    ]
                ),

            "rank":
                int(
                    result["rank"]
                ),

            "size":
                int(
                    result["size"]
                ),

            "full_rank":
                bool(
                    result[
                        "full_rank"
                    ]
                ),

            "condition_number":
                float(
                    result[
                        "condition_number"
                    ]
                ),

            "singular_value_ratio":
                float(
                    result[
                        "singular_value_ratio"
                    ]
                ),

            "numerical_condition_warning":
                bool(
                    result[
                        "numerical_condition_warning"
                    ]
                ),

            "derivative_stability_max_rel_l2":
                float(
                    result[
                        "derivative_stability_max_rel_l2"
                    ]
                ),

            "derivative_stability_max_rel_max":
                float(
                    result[
                        "derivative_stability_max_rel_max"
                    ]
                ),

            "fd_stability_sigma_piE_rel":
                float(
                    result[
                        "fd_stability_sigma_piE_rel"
                    ]
                ),

            "fd_stability_smin_rel":
                float(
                    result[
                        "fd_stability_smin_rel"
                    ]
                ),

            "fd_stability_weak_mode_angle_deg":
                float(
                    result[
                        "fd_stability_weak_mode_angle_deg"
                    ]
                ),

            "fd_stability_jacobian_rel":
                float(
                    result[
                        "fd_stability_jacobian_rel"
                    ]
                ),

            "min_eig_dF":
                min_eig_dF,

            "min_eig_dF_rel":
                min_eig_dF_rel,

            "nuisance_rank":
                result.get(
                    "nuisance_rank",
                    np.nan,
                ),

            "nuisance_size":
                result.get(
                    "nuisance_size",
                    np.nan,
                ),

            "nuisance_full_rank":
                result.get(
                    "nuisance_full_rank",
                    np.nan,
                ),
        }

        for band in RUBIN_BANDS:

            row[
                f"N_{band}"
            ] = (
                counts_band[
                    band
                ]
            )

        rows.append(row)

        if (
            N == 0
            or N == len(points)
            or N % 20 == 0
        ):

            print(
                f"N={N:4d} "
                f"sigma={sigma:.9g} "
                f"gain={gain:.9g} "
                f"fd_rel="
                f"{row['fd_stability_sigma_piE_rel']:.3e} "
                f"minEigRel="
                f"{min_eig_dF_rel:.3e}"
            )


df = pd.DataFrame(
    rows
)

df.to_parquet(
    OUT,
    index=False,
)


print()
print("=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

for population, group in df.groupby(
    "population",
    sort=False,
):

    first = group.iloc[0]
    last = group.iloc[-1]

    print()
    print(population)
    print(
        "N Rubin =",
        int(
            last["N"]
        ),
    )

    print(
        "sigma_piE Roman =",
        first[
            "sigma_piE"
        ],
    )

    print(
        "sigma_piE full  =",
        last[
            "sigma_piE"
        ],
    )

    print(
        "gain primary    =",
        last[
            "gain_piE"
        ],
    )

    print(
        "gain alternate  =",
        last[
            "gain_piE_alternate"
        ],
    )

    print(
        "max fd sigma rel =",
        group[
            "fd_stability_sigma_piE_rel"
        ].max(),
    )

    print(
        "max |negative dF rel| =",
        abs(
            min(
                0.0,
                group[
                    "min_eig_dF_rel"
                ].min(
                    skipna=True
                ),
            )
        ),
    )

    print(
        "numerical warnings =",
        int(
            group[
                "numerical_condition_warning"
            ].sum()
        ),
        "/",
        len(group),
    )


print()
print(
    "OUTPUT =",
    OUT,
)

print(
    "rows =",
    len(df),
)
