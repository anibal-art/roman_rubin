#!/usr/bin/env python3

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "/share/storage3/rubin/microlensing/romanrubin/roman_rubin"
).resolve()

CONFIG = (
    ROOT
    / "config_files"
    / "gbtds_300_v1_amax_v1.json"
)

OUT = Path(
    "/tmp/exp3_tau_windows.parquet"
)

WINDOWS = [
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
]

RUBIN_BANDS = (
    "u", "g", "r", "i", "z", "y",
)


# ============================================================
# Production environment
# ============================================================

from launch_catalog_simfit_from_config import (
    configure_environment,
)

cfg = json.loads(
    CONFIG.read_text()
)

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


from catalog.event_reader import (
    load_event_from_catalog,
)

from functions_roman_rubin import (
    build_event_params_from_custom_system,
    sim_event,
)

from analysis.fisher_information import (
    rebuild_model,
    compute_fisher,
)


CASES = [
    ("FFP", 10, "FSPL"),
    ("BH", 308, "FSPL"),
]


def arr(x):
    return np.asarray(
        getattr(x, "value", x),
        dtype=float,
    )


def get_param(params, name):
    try:
        return float(params[name])
    except Exception:
        return float(
            getattr(params, name)
        )


def rubin_tau_by_band(
    model,
    t0,
    tE,
):
    out = {}

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

        tau = (
            times - t0
        ) / tE

        out[tel.name] = {
            "times": times,
            "tau": tau,
        }

    return out


def subset_model_window(
    source_model,
    tau_by_band,
    window,
):
    """
    Keep ALL Roman data.

    For every Rubin band, keep only observations satisfying
        |tau| <= window.

    Rubin points retain their original chronological ordering.
    """

    model = copy.deepcopy(
        source_model
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

        if tel.name not in tau_by_band:
            continue

        tau = tau_by_band[
            tel.name
        ]["tau"]

        indices = np.where(
            np.abs(tau) <= window
        )[0]

        if len(indices) == 0:
            continue

        tel.lightcurve = (
            tel.lightcurve[
                indices
            ]
        )

        kept.append(tel)

    model.event.telescopes = kept

    return model


def subset_model_roman_only(
    source_model,
):
    model = copy.deepcopy(
        source_model
    )

    model.event.telescopes = [
        tel
        for tel in model.event.telescopes
        if (
            tel.name == "W149"
            and tel.lightcurve is not None
            and len(tel.lightcurve) > 0
        )
    ]

    return model


def rebuild_and_fisher(
    source,
    true_params,
):
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

    return compute_fisher(
        model,
        params,
        physical_derivative_method="auto",
    )


def safe_gain(a, b):
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
    event_index,
    model_name,
) in CASES:

    print()
    print("=" * 110)
    print(
        population,
        "catalog_event_index=",
        event_index,
    )
    print("=" * 110)

    catalog_row = (
        load_event_from_catalog(
            catalog_path,
            event_index,
            expected_population=
                population,
        )
    )

    event_params = (
        build_event_params_from_custom_system(
            custom_system=
                catalog_row,
            model=model_name,
            use_roman=True,
            use_rubin=True,
            rubin_pointing_mode=
                "source",
        )
    )

    simulation_seed = int(
        catalog_row[
            "simulation_seed"
        ]
    )

    (
        true_model,
        true_params,
        detected,
    ) = sim_event(
        simulation_seed,
        event_params,
        path_ephemerides,
        model_name,
        time_window=None,
        use_roman=True,
        use_rubin=True,
        truth_parallax=True,
        rubin_pointing_mode=
            "source",
        rubin_cache_cell_deg=None,
        apply_detection_criteria=True,
        apply_photometric_filter=True,
    )

    if not detected:
        raise RuntimeError(
            f"{population} {event_index} "
            "is not Roman-detected."
        )

    t0 = get_param(
        true_params,
        "t0",
    )

    tE = get_param(
        true_params,
        "tE",
    )

    tau_by_band = rubin_tau_by_band(
        true_model,
        t0,
        tE,
    )

    # --------------------------------------------------------
    # Roman reference
    # --------------------------------------------------------

    roman_source = (
        subset_model_roman_only(
            true_model
        )
    )

    roman = rebuild_and_fisher(
        roman_source,
        true_params,
    )

    sigma_R = float(
        roman[
            "sigma_piE"
        ]
    )

    F_previous = np.asarray(
        roman[
            "fisher_physical_q"
        ],
        dtype=float,
    )

    print(
        "sigma_R =",
        sigma_R,
    )

    print(
        "active Rubin bands =",
        list(
            tau_by_band.keys()
        ),
    )

    # --------------------------------------------------------
    # Full Rubin reference
    # --------------------------------------------------------

    full = rebuild_and_fisher(
        copy.deepcopy(
            true_model
        ),
        true_params,
    )

    sigma_full = float(
        full[
            "sigma_piE"
        ]
    )

    gain_full = safe_gain(
        sigma_R,
        sigma_full,
    )

    total_reduction = (
        sigma_R
        - sigma_full
    )

    print(
        "sigma_full =",
        sigma_full,
    )

    print(
        "gain_full =",
        gain_full,
    )

    # --------------------------------------------------------
    # Nested tau windows
    # --------------------------------------------------------

    for window in WINDOWS:

        source = (
            subset_model_window(
                true_model,
                tau_by_band,
                window,
            )
        )

        result = rebuild_and_fisher(
            source,
            true_params,
        )

        sigma = float(
            result[
                "sigma_piE"
            ]
        )

        gain = safe_gain(
            sigma_R,
            sigma,
        )

        F = np.asarray(
            result[
                "fisher_physical_q"
            ],
            dtype=float,
        )

        dF = (
            F - F_previous
        )

        dF = (
            dF + dF.T
        ) / 2.0

        eig = np.linalg.eigvalsh(
            dF
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
                        F_previous
                    )
                )
            ),
        )

        min_eig_dF_rel = (
            float(
                np.min(eig)
            )
            / scale
        )

        F_previous = F

        counts = {
            band: 0
            for band in RUBIN_BANDS
        }

        all_selected_tau = []

        for band, values in (
            tau_by_band.items()
        ):

            mask = (
                np.abs(
                    values["tau"]
                )
                <= window
            )

            counts[band] = int(
                np.sum(mask)
            )

            all_selected_tau.extend(
                values["tau"][
                    mask
                ].tolist()
            )

        n_rubin = sum(
            counts.values()
        )

        if (
            np.isfinite(
                total_reduction
            )
            and total_reduction > 0
        ):
            fraction_full_improvement = (
                (
                    sigma_R
                    - sigma
                )
                / total_reduction
            )
        else:
            fraction_full_improvement = (
                np.nan
            )

        row = {
            "population":
                population,

            "catalog_event_index":
                event_index,

            "window_abs_tau":
                window,

            "N_rubin":
                n_rubin,

            "sigma_R_piE":
                sigma_R,

            "sigma_full_piE":
                sigma_full,

            "sigma_piE":
                sigma,

            "sigma_piE_alternate":
                float(
                    result[
                        "sigma_piE_alternate"
                    ]
                ),

            "gain_piE":
                gain,

            "gain_minus_1":
                gain - 1.0,

            "gain_full_piE":
                gain_full,

            "fraction_full_improvement":
                fraction_full_improvement,

            "rank":
                int(
                    result["rank"]
                ),

            "size":
                int(
                    result["size"]
                ),

            "nuisance_rank":
                int(
                    result[
                        "nuisance_rank"
                    ]
                ),

            "nuisance_size":
                int(
                    result[
                        "nuisance_size"
                    ]
                ),

            "condition_number":
                float(
                    result[
                        "condition_number"
                    ]
                ),

            "fd_stability_sigma_piE_rel":
                float(
                    result[
                        "fd_stability_sigma_piE_rel"
                    ]
                ),

            "min_eig_dF_rel":
                min_eig_dF_rel,

            "min_abs_tau_selected":
                (
                    float(
                        np.min(
                            np.abs(
                                all_selected_tau
                            )
                        )
                    )
                    if all_selected_tau
                    else np.nan
                ),
        }

        for band in RUBIN_BANDS:
            row[
                f"N_{band}"
            ] = counts[
                band
            ]

        rows.append(
            row
        )

        print(
            f"w={window:6.2f} "
            f"N_Rubin={n_rubin:3d} "
            f"sigma={sigma:.9g} "
            f"gain={gain:.9g} "
            f"frac_full="
            f"{fraction_full_improvement:.6g} "
            f"nuis="
            f"{row['nuisance_rank']}/"
            f"{row['nuisance_size']} "
            f"dF_rel="
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
print("=" * 120)
print("SUMMARY")
print("=" * 120)

cols = [
    "population",
    "catalog_event_index",
    "window_abs_tau",
    "N_rubin",
    "N_z",
    "N_y",
    "sigma_piE",
    "gain_piE",
    "fraction_full_improvement",
    "nuisance_rank",
    "nuisance_size",
    "fd_stability_sigma_piE_rel",
    "min_eig_dF_rel",
]

with pd.option_context(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
    "display.width",
    240,
    "display.float_format",
    lambda x:
        f"{x:.6e}",
):
    print(
        df[cols].to_string(
            index=False
        )
    )


print()
print(
    "OUTPUT =",
    OUT,
)
