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
)

from analysis.fisher_information import (
    telescope_sets,
    rebuild_model,
    compute_fisher,
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

catalog_path = catalog_path.resolve()

ephemerides = Path(
    resolved["roman_ephemerides"]
).resolve()


# ======================================================================
# Event
# ======================================================================

CATALOG_EVENT_INDEX = 308
MODEL = "FSPL"

event = load_event_from_catalog(
    catalog_path=catalog_path,
    catalog_event_index=CATALOG_EVENT_INDEX,
)

seed = int(
    event.get(
        "simulation_seed",
        CATALOG_EVENT_INDEX,
    )
)

print("=" * 100)
print("CATALOG EVENT")
print("=" * 100)

for key in (
    "system_type",
    "t0",
    "u0",
    "tE",
    "rho",
    "piEN",
    "piEE",
    "ra",
    "dec",
):
    print(
        f"{key:15s} =",
        event.get(key),
    )


# ======================================================================
# Simulate ONCE with Roman + Rubin
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

        # We want to inspect the information even if a selection
        # criterion would reject the event.
        apply_detection_criteria=False,

        apply_photometric_filter=True,

        add_photometric_noise=True,

        rubin_cache_cell_deg=None,
    )
)

print()
print("=" * 100)
print("SIMULATION")
print("=" * 100)

print(
    "decision =",
    decision,
)

print(
    "telescopes =",
    [
        tel.name
        for tel in true_model.event.telescopes
    ],
)


# ======================================================================
# Identify Roman / Rubin telescopes
# ======================================================================

roman_names, rubin_names = telescope_sets(
    true_model,
    # Keep every Rubin band containing at least one surviving point.
    min_rubin_points=1,
)

roman_names = list(
    roman_names
)

rubin_names = list(
    rubin_names
)

rr_names = (
    roman_names
    + rubin_names
)

print()
print(
    "Roman =",
    roman_names,
)

print(
    "Rubin =",
    rubin_names,
)

print(
    "RR    =",
    rr_names,
)

if not roman_names:
    raise RuntimeError(
        "No Roman data survived."
    )

if not rubin_names:
    raise RuntimeError(
        "No Rubin data survived."
    )


# ======================================================================
# Fisher configurations
# ======================================================================

CONFIGURATIONS = {
    "R":
        roman_names,

    "Rubin":
        rubin_names,

    "RR":
        rr_names,
}

OUT = Path(
    "/tmp/fisher_event_308"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

results = {}


for label, telescope_names in CONFIGURATIONS.items():

    print()
    print("=" * 100)
    print(label)
    print("=" * 100)

    # --------------------------------------------------------------
    # Fresh pyLIMA model containing only the requested telescopes.
    #
    # source_model is NOT mutated.
    # --------------------------------------------------------------

    model_cfg, params_cfg = rebuild_model(
        true_model,
        true_params,
        telescope_names,
        roman_at_earth=False,
    )

    # --------------------------------------------------------------
    # Fisher evaluated at truth.
    #
    # fisher_physical_native is already the physical Fisher after
    # projecting out the photometric nuisance parameters.
    # --------------------------------------------------------------

    fisher = compute_fisher(
        model_cfg,
        params_cfg,
    )

    results[
        label
    ] = fisher

    print(
        "telescopes =",
        telescope_names,
    )

    print(
        "physical parameters =",
        fisher[
            "physical_names"
        ],
    )

    print(
        "nuisance parameters =",
        fisher[
            "nuisance_names"
        ],
    )

    print(
        "N data =",
        fisher[
            "n_data"
        ],
    )

    print(
        "rank =",
        fisher["rank"],
        "/",
        fisher["size"],
    )

    print(
        "condition number =",
        fisher[
            "condition_number"
        ],
    )

    print(
        "singular values =",
        fisher[
            "singular_values_q"
        ],
    )

    print(
        "sigma(piE) =",
        fisher[
            "sigma_piE"
        ],
    )

    print(
        "fractional sigma(piE) =",
        fisher[
            "fractional_sigma_piE"
        ],
    )

    print(
        "sigmas:"
    )

    for name, sigma in (
        fisher["sigma"].items()
    ):
        print(
            f"  {name:10s} {sigma}"
        )

    # --------------------------------------------------------------
    # Save matrices
    # --------------------------------------------------------------

    np.savez_compressed(
        OUT
        / f"fisher_{label}.npz",

        physical_names=np.asarray(
            fisher[
                "physical_names"
            ],
            dtype="U",
        ),

        nuisance_names=np.asarray(
            fisher[
                "nuisance_names"
            ],
            dtype="U",
        ),

        # Full physical+nuisance Fisher
        fisher_full=
            fisher[
                "fisher_full"
            ],

        # Physical Fisher after nuisance projection
        fisher_physical_native=
            fisher[
                "fisher_physical_native"
            ],

        # Same Fisher in scaled q-coordinates
        fisher_physical_q=
            fisher[
                "fisher_physical_q"
            ],

        covariance_native_pinv=
            fisher[
                "covariance_native_pinv"
            ],

        covariance_q_pinv=
            fisher[
                "covariance_q_pinv"
            ],

        singular_values_q=
            fisher[
                "singular_values_q"
            ],

        eigenvalues_q=
            fisher[
                "eigenvalues_q"
            ],
    )


# ======================================================================
# Direct gains
# ======================================================================

print()
print("=" * 100)
print("INFORMATION GAINS")
print("=" * 100)

for parameter in (
    "t0",
    "u0",
    "tE",
    "rho",
    "piEN",
    "piEE",
):

    sR = results[
        "R"
    ][
        "sigma"
    ].get(
        parameter,
        np.nan,
    )

    sL = results[
        "Rubin"
    ][
        "sigma"
    ].get(
        parameter,
        np.nan,
    )

    sRR = results[
        "RR"
    ][
        "sigma"
    ].get(
        parameter,
        np.nan,
    )

    print()
    print(parameter)

    print(
        "  sigma R     =",
        sR,
    )

    print(
        "  sigma Rubin =",
        sL,
    )

    print(
        "  sigma RR    =",
        sRR,
    )

    print(
        "  gain R -> RR =",
        (
            sR / sRR
            if (
                np.isfinite(sR)
                and np.isfinite(sRR)
                and sRR > 0
            )
            else np.nan
        ),
    )


print()
print("piE")

for label in (
    "R",
    "Rubin",
    "RR",
):
    print(
        f"  sigma {label:5s} =",
        results[
            label
        ][
            "sigma_piE"
        ],
    )


print()
print(
    "output =",
    OUT,
)
