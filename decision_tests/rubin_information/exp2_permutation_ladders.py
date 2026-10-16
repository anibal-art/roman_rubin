#!/usr/bin/env python3

import copy
import hashlib
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
    "/tmp/exp2_perm100_n10.parquet"
)

ANALYSIS_SEED = 20261015
N_PERMUTATIONS = 100
MAX_N = 10


# ============================================================
# Exact production environment
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
    simulate_event_for_fit,
)

from analysis.fisher_information import (
    rebuild_model,
    compute_fisher,
)


CASES = [
    ("FFP", 10, "FSPL", "z"),
    ("BH", 308, "FSPL", "z"),
]


# ============================================================
# Helpers
# ============================================================

def arr(x):
    return np.asarray(
        getattr(x, "value", x),
        dtype=float,
    )


def get_param(
    params,
    name,
):
    try:
        return float(
            params[name]
        )
    except Exception:
        return float(
            getattr(
                params,
                name,
            )
        )


def analysis_seed(
    event_index,
    band,
    permutation,
):
    """
    Deterministic ANALYSIS-ONLY RNG.

    Never passed to simulation or fitting.
    """

    payload = (
        f"{ANALYSIS_SEED}:"
        f"{event_index}:"
        f"{band}:"
        f"{permutation}"
    ).encode()

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def find_telescope(
    model,
    name,
):
    for tel in model.event.telescopes:
        if (
            tel.name == name
            and tel.lightcurve is not None
            and len(tel.lightcurve) > 0
        ):
            return tel

    raise RuntimeError(
        f"Missing active telescope {name}"
    )


def subset_model(
    source_model,
    band,
    indices,
):
    """
    Keep:
      - all Roman W149 points
      - selected Rubin points from exactly one band

    Important:
    selected indices are sorted chronologically before being
    inserted into the model.  The random permutation controls
    selection, NOT the temporal ordering of the light curve.
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

        if tel.name != band:
            continue

        if len(indices) == 0:
            continue

        idx = np.sort(
            np.asarray(
                indices,
                dtype=int,
            )
        )

        tel.lightcurve = (
            tel.lightcurve[idx]
        )

        kept.append(tel)

    model.event.telescopes = kept

    return model


def fisher_subset(
    source_model,
    true_params,
    band,
    indices,
):
    source = subset_model(
        source_model,
        band,
        indices,
    )

    names = [
        tel.name
        for tel
        in source.event.telescopes
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


def safe_gain(
    sigma_R,
    sigma,
):
    if (
        np.isfinite(sigma_R)
        and np.isfinite(sigma)
        and sigma > 0
    ):
        return (
            sigma_R
            / sigma
        )

    return np.nan


def completed_permutations(
    df,
    population,
    event_index,
    band,
):
    if df.empty:
        return set()

    sub = df[
        (df["population"] == population)
        & (
            df["catalog_event_index"]
            == event_index
        )
        & (df["band"] == band)
    ]

    completed = set()

    for permutation, g in sub.groupby(
        "permutation"
    ):
        present = set(
            g["N"]
            .astype(int)
            .tolist()
        )

        expected = set(
            range(
                1,
                MAX_N + 1,
            )
        )

        if expected.issubset(
            present
        ):
            completed.add(
                int(permutation)
            )

    return completed


def save_checkpoint(
    df,
):
    df = (
        df.drop_duplicates(
            subset=[
                "population",
                "catalog_event_index",
                "band",
                "permutation",
                "N",
            ],
            keep="last",
        )
        .sort_values(
            [
                "population",
                "catalog_event_index",
                "band",
                "permutation",
                "N",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    tmp = OUT.with_suffix(
        ".tmp.parquet"
    )

    df.to_parquet(
        tmp,
        index=False,
    )

    tmp.replace(
        OUT
    )

    return df


# ============================================================
# Resume existing checkpoint if present
# ============================================================

if OUT.exists():
    output_df = pd.read_parquet(
        OUT
    )

    print(
        "Resuming existing checkpoint:",
        OUT,
    )

    print(
        "Existing rows =",
        len(output_df),
    )

else:
    output_df = pd.DataFrame()


# ============================================================
# Main experiment
# ============================================================

for (
    population,
    event_index,
    model_name,
    band,
) in CASES:

    print()
    print("=" * 110)
    print(
        f"{population} "
        f"catalog_event_index={event_index} "
        f"band={band}"
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
    ) = simulate_event_for_fit(
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
            f"{population} "
            f"{event_index} "
            "is not Roman-detected."
        )

    tel = find_telescope(
        true_model,
        band,
    )

    times = arr(
        tel.lightcurve[
            "time"
        ]
    )

    t0 = get_param(
        true_params,
        "t0",
    )

    tE = get_param(
        true_params,
        "tE",
    )

    tau = (
        times
        - t0
    ) / tE

    n_available = len(
        times
    )

    max_n = min(
        MAX_N,
        n_available,
    )

    if max_n < MAX_N:
        raise RuntimeError(
            f"{population} {event_index} "
            f"has only {n_available} "
            f"{band} points."
        )

    # --------------------------------------------------------
    # Roman-only reference computed once per event
    # --------------------------------------------------------

    roman = fisher_subset(
        true_model,
        true_params,
        band,
        [],
    )

    sigma_R = float(
        roman[
            "sigma_piE"
        ]
    )

    sigma_R_alt = float(
        roman[
            "sigma_piE_alternate"
        ]
    )

    roman_nuisance_rank = int(
        roman[
            "nuisance_rank"
        ]
    )

    roman_nuisance_size = int(
        roman[
            "nuisance_size"
        ]
    )

    F_R = np.asarray(
        roman[
            "fisher_physical_q"
        ],
        dtype=float,
    )

    print(
        "N available =",
        n_available,
    )

    print(
        "sigma_R =",
        sigma_R,
    )

    print(
        "Roman nuisance rank/size =",
        f"{roman_nuisance_rank}/"
        f"{roman_nuisance_size}",
    )

    completed = (
        completed_permutations(
            output_df,
            population,
            event_index,
            band,
        )
    )

    if completed:
        print(
            "Already completed permutations =",
            len(completed),
        )

    for permutation in range(
        N_PERMUTATIONS
    ):

        if permutation in completed:
            continue

        seed = analysis_seed(
            event_index,
            band,
            permutation,
        )

        rng = np.random.default_rng(
            seed
        )

        # Uniform random ordering of the available real points.
        order = rng.permutation(
            n_available
        )

        permutation_rows = []

        previous_F = (
            F_R.copy()
        )

        for N in range(
            1,
            max_n + 1,
        ):

            selected_in_order = (
                order[:N]
                .astype(int)
            )

            # For actual model construction, restore
            # chronological row order.
            selected_for_model = (
                np.sort(
                    selected_in_order
                )
                .astype(int)
                .tolist()
            )

            added_index = int(
                order[
                    N - 1
                ]
            )

            selected_tau = (
                tau[
                    selected_in_order
                ]
            )

            result = fisher_subset(
                true_model,
                true_params,
                band,
                selected_for_model,
            )

            sigma = float(
                result[
                    "sigma_piE"
                ]
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

            nuisance_rank = int(
                result[
                    "nuisance_rank"
                ]
            )

            nuisance_size = int(
                result[
                    "nuisance_size"
                ]
            )

            band_rank = (
                nuisance_rank
                - roman_nuisance_rank
            )

            band_nuisance_size = (
                nuisance_size
                - roman_nuisance_size
            )

            residual_dof = (
                N
                - band_rank
            )

            F = np.asarray(
                result[
                    "fisher_physical_q"
                ],
                dtype=float,
            )

            dF = (
                F
                - previous_F
            )

            dF = (
                dF
                + dF.T
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
                            previous_F
                        )
                    )
                ),
            )

            min_eig_dF = float(
                np.min(
                    eig
                )
            )

            min_eig_dF_rel = (
                min_eig_dF
                / scale
            )

            previous_F = F

            abs_tau = np.abs(
                selected_tau
            )

            n_peak = int(
                np.sum(
                    abs_tau <= 1.0
                )
            )

            n_le2 = int(
                np.sum(
                    abs_tau <= 2.0
                )
            )

            n_wings = (
                n_le2
                - n_peak
            )

            n_outer = (
                N
                - n_le2
            )

            if n_peak > 0:
                coverage_class = (
                    "contains_peak"
                )

            elif n_wings > 0:
                coverage_class = (
                    "contains_wing_no_peak"
                )

            else:
                coverage_class = (
                    "outer_only"
                )

            row = {
                "population":
                    population,

                "catalog_event_index":
                    event_index,

                "field_id":
                    str(
                        catalog_row.get(
                            "field_id",
                            "",
                        )
                    ),

                "band":
                    band,

                "permutation":
                    permutation,

                "analysis_seed":
                    seed,

                "simulation_seed":
                    simulation_seed,

                "N":
                    N,

                "N_available":
                    n_available,

                "added_index":
                    added_index,

                "added_time_jd":
                    float(
                        times[
                            added_index
                        ]
                    ),

                "added_tau":
                    float(
                        tau[
                            added_index
                        ]
                    ),

                "selected_indices_json":
                    json.dumps(
                        selected_for_model
                    ),

                "selected_tau_json":
                    json.dumps(
                        [
                            float(x)
                            for x
                            in selected_tau
                        ]
                    ),

                "tau_min":
                    float(
                        np.min(
                            selected_tau
                        )
                    ),

                "tau_max":
                    float(
                        np.max(
                            selected_tau
                        )
                    ),

                "tau_span":
                    float(
                        np.max(
                            selected_tau
                        )
                        - np.min(
                            selected_tau
                        )
                    ),

                "min_abs_tau":
                    float(
                        np.min(
                            abs_tau
                        )
                    ),

                "max_abs_tau":
                    float(
                        np.max(
                            abs_tau
                        )
                    ),

                "n_peak":
                    n_peak,

                "n_wings":
                    n_wings,

                "n_outer":
                    n_outer,

                "coverage_class":
                    coverage_class,

                "roman_nuisance_rank":
                    roman_nuisance_rank,

                "roman_nuisance_size":
                    roman_nuisance_size,

                "nuisance_rank":
                    nuisance_rank,

                "nuisance_size":
                    nuisance_size,

                "band_nuisance_rank":
                    band_rank,

                "band_nuisance_size":
                    band_nuisance_size,

                "band_flux_full_rank":
                    bool(
                        band_rank
                        == band_nuisance_size
                    ),

                "band_residual_dof":
                    residual_dof,

                "sigma_R_piE":
                    sigma_R,

                "sigma_piE":
                    sigma,

                "sigma_piE_alternate":
                    sigma_alt,

                "gain_piE":
                    gain,

                "gain_minus_1":
                    (
                        gain
                        - 1.0
                    ),

                "gain_piE_alternate":
                    gain_alt,

                "gain_alt_minus_1":
                    (
                        gain_alt
                        - 1.0
                    ),

                "rank":
                    int(
                        result[
                            "rank"
                        ]
                    ),

                "size":
                    int(
                        result[
                            "size"
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
            }

            permutation_rows.append(
                row
            )

        # ----------------------------------------------------
        # Commit one complete permutation at a time.
        # ----------------------------------------------------

        new_df = pd.DataFrame(
            permutation_rows
        )

        if output_df.empty:
            output_df = new_df

        else:
            output_df = pd.concat(
                [
                    output_df,
                    new_df,
                ],
                ignore_index=True,
            )

        output_df = save_checkpoint(
            output_df
        )

        if (
            permutation == 0
            or (
                permutation + 1
            ) % 10 == 0
        ):
            print(
                f"{population} {event_index} "
                f"{band}: completed "
                f"{permutation + 1}/"
                f"{N_PERMUTATIONS}"
            )


# ============================================================
# Final summaries
# ============================================================

df = pd.read_parquet(
    OUT
)

df = (
    df[
        df["permutation"]
        < N_PERMUTATIONS
    ]
    .copy()
)


print()
print("=" * 120)
print("SUMMARY BY N")
print("=" * 120)

summary = (
    df.groupby(
        [
            "population",
            "catalog_event_index",
            "band",
            "N",
        ],
        as_index=False,
    )
    .agg(
        n_permutations=(
            "permutation",
            "nunique",
        ),

        full_rank_fraction=(
            "band_flux_full_rank",
            "mean",
        ),

        residual_dof_min=(
            "band_residual_dof",
            "min",
        ),

        residual_dof_median=(
            "band_residual_dof",
            "median",
        ),

        residual_dof_max=(
            "band_residual_dof",
            "max",
        ),

        gain_minus_1_min=(
            "gain_minus_1",
            "min",
        ),

        gain_minus_1_p05=(
            "gain_minus_1",
            lambda x:
                x.quantile(
                    0.05
                ),
        ),

        gain_minus_1_median=(
            "gain_minus_1",
            "median",
        ),

        gain_minus_1_p95=(
            "gain_minus_1",
            lambda x:
                x.quantile(
                    0.95
                ),
        ),

        gain_minus_1_max=(
            "gain_minus_1",
            "max",
        ),

        fraction_contains_peak=(
            "n_peak",
            lambda x:
                np.mean(
                    np.asarray(x)
                    > 0
                ),
        ),

        fraction_contains_tau_le2=(
            "n_wings",
            lambda x:
                np.mean(
                    (
                        np.asarray(x)
                        > 0
                    )
                    |
                    (
                        np.asarray(
                            df.loc[
                                x.index,
                                "n_peak",
                            ]
                        )
                        > 0
                    )
                ),
        ),

        fd_rel_max=(
            "fd_stability_sigma_piE_rel",
            "max",
        ),

        min_eig_dF_rel_min=(
            "min_eig_dF_rel",
            "min",
        ),
    )
)

with pd.option_context(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
    "display.width",
    260,
    "display.float_format",
    lambda x:
        f"{x:.6e}",
):
    print(
        summary.to_string(
            index=False
        )
    )


print()
print("=" * 120)
print("TEMPORAL COVERAGE CHECK: N = 3, 5, 10")
print("=" * 120)

coverage = (
    df[
        df["N"].isin(
            [
                3,
                5,
                10,
            ]
        )
    ]
    .groupby(
        [
            "population",
            "catalog_event_index",
            "band",
            "N",
            "coverage_class",
        ],
        as_index=False,
    )
    .agg(
        n_subsets=(
            "permutation",
            "count",
        ),

        full_rank_fraction=(
            "band_flux_full_rank",
            "mean",
        ),

        gain_minus_1_median=(
            "gain_minus_1",
            "median",
        ),

        gain_minus_1_max=(
            "gain_minus_1",
            "max",
        ),

        min_abs_tau_median=(
            "min_abs_tau",
            "median",
        ),

        tau_span_median=(
            "tau_span",
            "median",
        ),
    )
)

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
        coverage.to_string(
            index=False
        )
    )


print()
print("=" * 120)
print("FINAL")
print("=" * 120)

print(
    "OUTPUT =",
    OUT,
)

print(
    "rows =",
    len(df),
)

for (
    population,
    event_index,
    band,
), g in df.groupby(
    [
        "population",
        "catalog_event_index",
        "band",
    ]
):
    print(
        population,
        event_index,
        band,
        "permutations =",
        g[
            "permutation"
        ].nunique(),
        "rows =",
        len(g),
    )
