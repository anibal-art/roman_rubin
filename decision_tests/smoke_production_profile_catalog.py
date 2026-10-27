#!/usr/bin/env python3

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from launch_catalog_simfit_from_config import (
    configure_environment,
)

from catalog.event_reader import (
    load_event_from_catalog,
)

from functions_roman_rubin import (
    sim_fit,
)


ROOT = Path(
    "/share/storage3/rubin/microlensing/romanrubin/roman_rubin"
).resolve()

CONFIG = (
    ROOT
    / "config_files"
    / "gbtds_300_v1_amax_v1.json"
)


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--population",
        required=True,
        choices=[
            "FFP",
            "BH",
            "Planets_systems",
        ],
    )

    p.add_argument(
        "--event-index",
        required=True,
        type=int,
    )

    p.add_argument(
        "--output-root",
        required=True,
    )

    return p.parse_args()


def resolve_input_path(value, base):

    p = Path(
        value
    ).expanduser()

    if not p.is_absolute():
        p = (
            base
            / p
        )

    return p.resolve()


def collect_fit_timings(
    result,
):

    out = {}

    if not isinstance(
        result,
        dict,
    ):
        return out

    pipeline = result.get(
        "pipeline_timings",
        {},
    )

    if isinstance(
        pipeline,
        dict,
    ):
        for key, value in pipeline.items():
            try:
                out[str(key)] = float(value)
            except Exception:
                pass

    true_model = result.get(
        "pyLIMAmodel_true",
        None,
    )

    simulation_timings = getattr(
        true_model,
        "pipeline_timings",
        {},
    )

    if isinstance(
        simulation_timings,
        dict,
    ):
        for key, value in simulation_timings.items():
            try:
                out[str(key)] = float(value)
            except Exception:
                pass

    for fit_key in (
        "fit_rr",
        "fit_roman",
    ):

        fit_obj = result.get(
            fit_key,
            None,
        )

        fit_results = getattr(
            fit_obj,
            "fit_results",
            None,
        )

        if not isinstance(
            fit_results,
            dict,
        ):
            continue

        for key, value in fit_results.items():

            if not str(key).startswith(
                "timing_"
            ):
                continue

            try:
                out[
                    f"{fit_key}_{key}"
                ] = float(value)
            except Exception:
                pass

    return out


args = parse_args()


# ======================================================================
# Configuration
# ======================================================================

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

catalog_path = resolve_input_path(
    config[
        "input"
    ][
        "event_catalog"
    ],
    CONFIG.parent,
)

ephemerides = Path(
    resolved_paths[
        "roman_ephemerides"
    ]
).expanduser().resolve()


population = args.population

model = str(
    config[
        "models"
    ][
        population
    ]
)

observing = config.get(
    "observing",
    {},
)

simulation = config.get(
    "simulation",
    {},
)

fit_cfg = config.get(
    "fit",
    {},
)


# ======================================================================
# Event
# ======================================================================

event = load_event_from_catalog(
    catalog_path,
    args.event_index,
    expected_population=population,
)

simulation_seed = int(
    event.get(
        "simulation_seed",
        args.event_index,
    )
)

catalog_ra = float(
    event["ra"]
)

catalog_dec = float(
    event["dec"]
)


# ======================================================================
# Output
# ======================================================================

event_root = (
    Path(
        args.output_root
    ).expanduser().resolve()
    / f"{population}_{args.event_index}"
)

model_dir = (
    event_root
    / "models"
)

fit_dir = (
    event_root
    / "fits"
)

results_dir = (
    event_root
    / "results"
)

for p in (
    model_dir,
    fit_dir,
    results_dir,
):
    p.mkdir(
        parents=True,
        exist_ok=True,
    )


print("=" * 100)
print("PRODUCTION SMOKE + TIMING")
print("=" * 100)

print(
    "population          =",
    population,
)

print(
    "catalog_event_index =",
    args.event_index,
)

print(
    "simulation_seed     =",
    simulation_seed,
)

print(
    "model               =",
    model,
)

print(
    "field_id            =",
    event.get(
        "field_id",
    ),
)

print(
    "RA, Dec             =",
    catalog_ra,
    catalog_dec,
)

print(
    "initial_guess       =",
    fit_cfg.get(
        "initial_guess",
        None,
    ),
)

print(
    "fit_bounds          =",
    fit_cfg.get(
        "bounds",
        None,
    ),
)

print(
    "optimizer_options   =",
    fit_cfg.get(
        "optimizer_options",
        None,
    ),
)

print(
    "ephemerides         =",
    ephemerides,
)

print(
    "catalog             =",
    catalog_path,
)


# ======================================================================
# Pre-run catalogue contract checks
# ======================================================================

required_common = [
    "t0",
    "u0",
    "tE",
    "piEN",
    "piEE",
    "ra",
    "dec",
    "W149",
]

if model.upper() == "FSPL":
    required_common.append(
        "rho"
    )

if model.upper() == "USBL":
    required_common.extend(
        [
            "rho",
            "s",
            "q",
            "alpha",
            "caustic_origin",
        ]
    )

missing = [
    key
    for key in required_common
    if key not in event
]

if missing:
    raise RuntimeError(
        "Missing materialized catalogue inputs: "
        f"{missing}"
    )


# Every active photometric band must have materialized fluxes.
band_candidates = [
    "W149",
    "u",
    "g",
    "r",
    "i",
    "z",
    "Y",
]

materialized_flux_status = {}

for band in band_candidates:

    mag_key = band

    if mag_key not in event:
        continue

    fsource_key = (
        f"fsource_{band}"
    )

    ftotal_key = (
        f"ftotal_{band}"
    )

    materialized_flux_status[
        band
    ] = (
        fsource_key in event
        and ftotal_key in event
    )


print()
print(
    "materialized flux columns =",
    materialized_flux_status,
)


# ======================================================================
# Timed production-equivalent sim_fit
# ======================================================================

wall0 = time.perf_counter()
cpu0 = time.process_time()

result = sim_fit(
    i=simulation_seed,

    system_type=population,

    model=model,

    algo=str(
        fit_cfg.get(
            "algorithm",
            "TRF",
        )
    ),

    # Current API placeholders.
    # Scientific truth comes from custom_system.
    path_TRILEGAL_set="custom_system",
    path_GENULENS_set="custom_system",

    path_to_save_model=str(
        model_dir
    ),

    path_to_save_fit=str(
        fit_dir
    ),

    path_ephemerides=str(
        ephemerides
    ),

    path_to_save_results=str(
        results_dir
    ),

    custom_system=event,
    catalog_mode="custom_system",

    use_roman=bool(
        observing.get(
            "use_roman",
            True,
        )
    ),

    use_rubin=bool(
        observing.get(
            "use_rubin",
            True,
        )
    ),

    truth_parallax=bool(
        simulation.get(
            "truth_parallax",
            True,
        )
    ),

    fit_model=model,

    fit_parallax=fit_cfg.get(
        "parallax",
        True,
    ),

    # IMPORTANT:
    # production-consistent initialization.
    initial_guess=fit_cfg.get(
        "initial_guess",
        None,
    ),

    fit_bounds=fit_cfg.get(
        "bounds",
        None,
    ),

    optimizer_options=fit_cfg.get(
        "optimizer_options",
        None,
    ),

    rubin_cache_cell_deg=simulation.get(
        "rubin_cache_cell_deg",
        None,
    ),

    apply_detection_criteria=bool(
        simulation.get(
            "apply_detection_criteria",
            True,
        )
    ),

    apply_photometric_filter=bool(
        simulation.get(
            "apply_photometric_filter",
            True,
        )
    ),

    add_photometric_noise=True,

    time_window=simulation.get(
        "time_window",
        None,
    ),

    fit_time_window=fit_cfg.get(
        "time_window",
        None,
    ),

    return_data=True,
)

cpu_s = float(
    time.process_time()
    - cpu0
)

wall_s = float(
    time.perf_counter()
    - wall0
)


# ======================================================================
# Memory
# Linux ru_maxrss is KiB.
# ======================================================================

usage = resource.getrusage(
    resource.RUSAGE_SELF
)

max_rss_gib = (
    float(
        usage.ru_maxrss
    )
    / 1024.0
    / 1024.0
)


# ======================================================================
# Result checks
# ======================================================================

status = None

if isinstance(
    result,
    dict,
):
    status = result.get(
        "status",
        None,
    )

true_model = (
    result.get(
        "pyLIMAmodel_true",
        None,
    )
    if isinstance(
        result,
        dict,
    )
    else None
)

geometry_ok = False

event_ra_used = np.nan
event_dec_used = np.nan

if (
    true_model is not None
    and getattr(
        true_model,
        "event",
        None,
    ) is not None
):

    event_ra_used = float(
        true_model.event.ra
    )

    event_dec_used = float(
        true_model.event.dec
    )

    geometry_ok = (
        np.isclose(
            event_ra_used,
            catalog_ra,
            rtol=0.0,
            atol=1e-10,
        )
        and
        np.isclose(
            event_dec_used,
            catalog_dec,
            rtol=0.0,
            atol=1e-10,
        )
    )


timings = collect_fit_timings(
    result
)


# ======================================================================
# Summary
# ======================================================================

summary = {
    "population":
        population,

    "catalog_event_index":
        int(
            args.event_index
        ),

    "simulation_seed":
        simulation_seed,

    "model":
        model,

    "field_id":
        str(
            event.get(
                "field_id",
                "",
            )
        ),

    "status":
        str(status),

    "wall_s":
        wall_s,

    "cpu_s":
        cpu_s,

    "cpu_over_wall":
        (
            cpu_s
            / wall_s
            if wall_s > 0
            else np.nan
        ),

    "max_rss_gib":
        max_rss_gib,

    "catalog_ra":
        catalog_ra,

    "catalog_dec":
        catalog_dec,

    "event_ra_used":
        event_ra_used,

    "event_dec_used":
        event_dec_used,

    "geometry_ok":
        bool(
            geometry_ok
        ),

    "materialized_flux_status":
        materialized_flux_status,

    "initial_guess":
        fit_cfg.get(
            "initial_guess",
            None,
        ),

    "fit_bounds":
        fit_cfg.get(
            "bounds",
            None,
        ),

    "optimizer_options":
        fit_cfg.get(
            "optimizer_options",
            None,
        ),

    "pipeline_timings":
        timings,
}


summary_path = (
    event_root
    / "smoke_timing_summary.json"
)

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
        allow_nan=True,
    )
    + "\n"
)


print()
print("=" * 100)
print("SMOKE SUMMARY")
print("=" * 100)

print(
    json.dumps(
        summary,
        indent=2,
        allow_nan=True,
    )
)

print()
print(
    "summary =",
    summary_path,
)

print()
print(
    "FULL FIT =",
    status == "fitted",
)

print(
    "GEOMETRY =",
    geometry_ok,
)
