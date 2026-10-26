#!/usr/bin/env python3

import copy
import json
from pathlib import Path

import numpy as np


ROOT = Path(
    "/share/storage3/rubin/microlensing/romanrubin/roman_rubin"
).resolve()

CONFIG = (
    ROOT
    / "config_files"
    / "gbtds_300_v1_amax_v1.json"
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
    ("FFP", 10, "FSPL", "z"),
    ("BH", 308, "FSPL", "z"),
]


def arr(x):

    return np.asarray(
        getattr(
            x,
            "value",
            x,
        ),
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


def find_tel(
    model,
    name,
):

    for tel in (
        model.event.telescopes
    ):

        if (
            tel.name == name
            and tel.lightcurve is not None
            and len(
                tel.lightcurve
            )
            > 0
        ):
            return tel

    raise RuntimeError(
        f"Missing telescope {name}"
    )


def fisher_subset(
    source_model,
    true_params,
    band,
    indices,
):

    source = copy.deepcopy(
        source_model
    )

    kept = []

    for tel in (
        source.event.telescopes
    ):

        if (
            tel.lightcurve is None
            or len(
                tel.lightcurve
            )
            == 0
        ):
            continue

        if tel.name == "W149":

            kept.append(
                tel
            )

            continue

        if tel.name != band:
            continue

        if len(indices) == 0:
            continue

        idx = np.asarray(
            indices,
            dtype=int,
        )

        tel.lightcurve = (
            tel.lightcurve[
                idx
            ]
        )

        kept.append(
            tel
        )

    source.event.telescopes = (
        kept
    )

    names = [
        tel.name
        for tel in (
            source.event.telescopes
        )
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
        np.isfinite(
            sigma_R
        )
        and np.isfinite(
            sigma
        )
        and sigma > 0
    ):
        return (
            sigma_R
            / sigma
        )

    return np.nan


def closest_indices(
    tau,
    n=3,
):

    return (
        np.argsort(
            np.abs(tau)
        )[:n]
        .astype(int)
        .tolist()
    )


def far_indices(
    tau,
    n=3,
):

    return (
        np.argsort(
            np.abs(tau)
        )[-n:]
        .astype(int)
        .tolist()
    )


for (
    population,
    event_index,
    model_name,
    band,
) in CASES:

    print()
    print("=" * 110)
    print(
        population,
        event_index,
        band,
    )
    print("=" * 110)

    row = load_event_from_catalog(
        catalog_path,
        event_index,
        expected_population=
            population,
    )

    event_params = (
        build_event_params_from_custom_system(
            custom_system=row,
            model=model_name,
            use_roman=True,
            use_rubin=True,
            rubin_pointing_mode=
                "source",
        )
    )

    seed = int(
        row[
            "simulation_seed"
        ]
    )

    (
        true_model,
        true_params,
        detected,
    ) = sim_event(
        seed,
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
            "Event no longer Roman-detected."
        )

    tel = find_tel(
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

    subsets = {
        "closest_3":
            closest_indices(
                tau,
                3,
            ),

        "farthest_3":
            far_indices(
                tau,
                3,
            ),
    }


    # --------------------------------------------------------
    # BH-specific physically defined subsets
    # --------------------------------------------------------

    peak = np.where(
        np.abs(tau)
        <= 1.0
    )[0]

    wings = np.where(
        (
            np.abs(tau)
            > 1.0
        )
        & (
            np.abs(tau)
            <= 2.0
        )
    )[0]

    outer10 = np.where(
        np.abs(tau)
        > 10.0
    )[0]


    if len(peak) >= 3:

        subsets[
            "peak_3"
        ] = (
            peak[:3]
            .astype(int)
            .tolist()
        )


    if len(wings) >= 3:

        wing_order = wings[
            np.argsort(
                np.abs(
                    tau[
                        wings
                    ]
                )
            )
        ]

        subsets[
            "wings_3"
        ] = (
            wing_order[:3]
            .astype(int)
            .tolist()
        )


    if len(outer10) >= 3:

        subsets[
            "outer_gt10_3"
        ] = (
            outer10[:3]
            .astype(int)
            .tolist()
        )


    if (
        len(peak) >= 1
        and len(wings) >= 1
        and len(outer10) >= 1
    ):

        subsets[
            "mixed_peak_wing_outer"
        ] = [
            int(
                peak[
                    np.argmin(
                        np.abs(
                            tau[
                                peak
                            ]
                        )
                    )
                ]
            ),
            int(
                wings[
                    np.argmin(
                        np.abs(
                            tau[
                                wings
                            ]
                        )
                    )
                ]
            ),
            int(
                outer10[
                    np.argmin(
                        np.abs(
                            tau[
                                outer10
                            ]
                        )
                    )
                ]
            ),
        ]


    print(
        "N available =",
        len(tau),
    )

    print(
        "sigma_R =",
        sigma_R,
    )

    print()

    print(
        f"{'subset':>24} "
        f"{'tau values':>42} "
        f"{'band_rank':>10} "
        f"{'resDOF':>7} "
        f"{'sigma_piE':>14} "
        f"{'gain-1':>14}"
    )

    print(
        "-" * 120
    )


    for (
        label,
        indices,
    ) in subsets.items():

        result = fisher_subset(
            true_model,
            true_params,
            band,
            indices,
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

        nuisance_rank = int(
            result[
                "nuisance_rank"
            ]
        )

        # Roman contributes rank 2.
        band_rank = (
            nuisance_rank
            - 2
        )

        residual_dof = (
            len(indices)
            - band_rank
        )

        tau_values = [
            float(
                tau[i]
            )
            for i in indices
        ]

        tau_text = (
            "["
            + ", ".join(
                f"{x:.3f}"
                for x in tau_values
            )
            + "]"
        )

        print(
            f"{label:>24} "
            f"{tau_text:>42} "
            f"{band_rank:10d} "
            f"{residual_dof:7d} "
            f"{sigma:14.8g} "
            f"{gain - 1.0:14.6e}"
        )


print()
print("=" * 110)
print("DONE")
print("=" * 110)
