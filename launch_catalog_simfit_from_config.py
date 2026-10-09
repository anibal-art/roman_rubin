#!/usr/bin/env python3

"""
Run materialized Roman-Rubin events from one JSON configuration file.

This is the active launcher for materialized event catalogues.

The superseded chunk-based simulation/fitting launcher is retained at
legacy/chunk_based_simfit_pipeline/launch_simfit_from_config.py for
historical reproducibility. Its input contract used
TRILEGAL_chunk_j.csv + Genulens_chunk_j.csv.

Current execution is serial by design. The configuration schema is
kept independent of execution strategy so the same config can later
drive multiprocessing/Slurm without changing the scientific setup.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


POPULATIONS = (
    "FFP",
    "BH",
    "Planets_systems",
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
    )

    parser.add_argument(
        "--population",
        choices=POPULATIONS,
        default=None,
    )

    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
    )

    return parser.parse_args()


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def resolve_path(value, config_dir):
    if value is None:
        return None

    text = os.path.expandvars(
        os.path.expanduser(
            str(value)
        )
    )

    path = Path(text)

    if not path.is_absolute():
        path = config_dir / path

    return path.resolve()


def require_file(path, label):
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


def configure_environment(paths, config_dir):
    mapping = {
        "rubin_sim_data_dir":
            "RUBIN_SIM_DATA_DIR",

        "rubin_throughputs_dir":
            "RUBIN_THROUGHPUTS_DIR",

        "rubin_opsim_db_path":
            "RUBIN_OPSIM_DB_PATH",

        "roman_ephemerides":
            "ROMAN_EPHEMERIDES",
    }

    resolved = {}

    for key, env_key in mapping.items():
        value = paths.get(key)

        if value in (None, ""):
            continue

        path = resolve_path(
            value,
            config_dir,
        )

        resolved[key] = path
        os.environ[env_key] = str(path)

    if "rubin_sim_data_dir" in resolved:
        os.environ["SIMS_DATA_DIR"] = str(
            resolved["rubin_sim_data_dir"]
        )

    return resolved


def select_catalog_events(
    catalog_path,
    population,
    selection,
    cli_max_events=None,
):
    columns = [
        "catalog_event_index",
        "system_type",
        "simulate_amax",
        "field_id",
        "ra",
        "dec",
    ]

    table = pd.read_parquet(
        catalog_path,
        columns=columns,
    )

    table = table[
        table["system_type"].astype(str)
        == population
    ].copy()

    require_amax = selection.get(
        "simulate_amax",
        None,
    )

    if require_amax is not None:
        table = table[
            table["simulate_amax"].astype(bool)
            == bool(require_amax)
        ]

    table = table.sort_values(
        "catalog_event_index",
        kind="stable",
    ).reset_index(drop=True)

    start = int(
        selection.get(
            "population_row_start",
            0,
        )
    )

    stop = selection.get(
        "population_row_stop",
        None,
    )

    if stop is not None:
        stop = int(stop)

    table = table.iloc[
        start:stop
    ].reset_index(drop=True)

    max_events = (
        cli_max_events
        if cli_max_events is not None
        else selection.get(
            "max_events",
            None,
        )
    )

    if max_events is not None:
        max_events = int(max_events)

        if max_events <= 0:
            raise ValueError(
                "max_events must be > 0"
            )

        table = table.iloc[
            :max_events
        ].reset_index(drop=True)

    if len(table) == 0:
        raise RuntimeError(
            "Catalogue selection is empty."
        )

    return table


def build_simfit_kwargs(
    config,
    simulation_model,
):
    observing = config.get(
        "observing",
        {},
    )

    simulation = config.get(
        "simulation",
        {},
    )

    fit = config.get(
        "fit",
        {},
    )

    truth_parallax = bool(
        simulation.get(
            "truth_parallax",
            True,
        )
    )

    fit_model = fit.get(
        "model",
        "same_as_simulation",
    )

    if fit_model in (
        None,
        "",
        "same_as_simulation",
    ):
        fit_model = simulation_model

    kwargs = {
        "catalog_mode":
            "custom_system",

        "use_roman":
            bool(
                observing.get(
                    "use_roman",
                    True,
                )
            ),

        "use_rubin":
            bool(
                observing.get(
                    "use_rubin",
                    True,
                )
            ),

        "truth_parallax":
            truth_parallax,

        "fit_model":
            str(fit_model),

        "fit_parallax":
            fit.get(
                "parallax",
                truth_parallax,
            ),

        "rubin_pointing_mode":
            str(
                simulation.get(
                    "rubin_pointing_mode",
                    "source",
                )
            ),

        "rubin_cache_cell_deg":
            simulation.get(
                "rubin_cache_cell_deg",
                None,
            ),

        "apply_detection_criteria":
            bool(
                simulation.get(
                    "apply_detection_criteria",
                    True,
                )
            ),

        "apply_photometric_filter":
            bool(
                simulation.get(
                    "apply_photometric_filter",
                    True,
                )
            ),
    }

    if simulation.get(
        "time_window",
        None,
    ) is not None:

        value = simulation[
            "time_window"
        ]

        if (
            not isinstance(
                value,
                (list, tuple),
            )
            or len(value) != 2
        ):
            raise ValueError(
                "simulation.time_window must be "
                "null or [JD_min, JD_max]."
            )

        kwargs["time_window"] = (
            float(value[0]),
            float(value[1]),
        )

    optional = {
        "fit_time_window":
            fit.get(
                "time_window",
                None,
            ),

        "fit_defaults":
            fit.get(
                "defaults",
                None,
            ),

        "fit_bounds":
            fit.get(
                "bounds",
                None,
            ),

        "initial_guess":
            fit.get(
                "initial_guess",
                None,
            ),

        "optimizer_options":
            fit.get(
                "optimizer_options",
                None,
            ),

        "rubin_saturation_mag":
            simulation.get(
                "rubin_saturation_mag",
                None,
            ),

        "roman_saturation_mag":
            simulation.get(
                "roman_saturation_mag",
                None,
            ),
    }

    for key, value in optional.items():
        if value is not None:
            kwargs[key] = value

    if (
        not kwargs["use_roman"]
        and not kwargs["use_rubin"]
    ):
        raise ValueError(
            "At least one observatory must be enabled."
        )

    if (
        kwargs["use_rubin"]
        and kwargs["rubin_pointing_mode"]
        != "source"
    ):
        raise ValueError(
            "This materialized GBTDS catalogue requires "
            "simulation.rubin_pointing_mode='source'."
        )

    return kwargs


def main():
    args = parse_args()

    config_path = Path(
        args.config
    ).expanduser().resolve()

    require_file(
        config_path,
        "config",
    )

    config_dir = (
        config_path.parent
    )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    input_cfg = config.get(
        "input",
        {},
    )

    selection_cfg = config.get(
        "selection",
        {},
    )

    paths_cfg = config.get(
        "paths",
        {},
    )

    output_cfg = config.get(
        "output",
        {},
    )

    models = config.get(
        "models",
        {},
    )

    population = (
        args.population
        or selection_cfg.get(
            "population",
            None,
        )
    )

    if population not in POPULATIONS:
        raise ValueError(
            f"Invalid population: {population!r}"
        )

    if population not in models:
        raise KeyError(
            f"No simulation model configured for "
            f"{population!r}"
        )

    simulation_model = str(
        models[population]
    )

    catalog_path = resolve_path(
        input_cfg.get(
            "event_catalog",
            None,
        ),
        config_dir,
    )

    require_file(
        catalog_path,
        "materialized event catalogue",
    )

    resolved_paths = (
        configure_environment(
            paths_cfg,
            config_dir,
        )
    )

    path_ephemerides = (
        resolved_paths.get(
            "roman_ephemerides",
            None,
        )
    )

    require_file(
        path_ephemerides,
        "Roman ephemerides",
    )

    if bool(
        config.get(
            "observing",
            {},
        ).get(
            "use_rubin",
            True,
        )
    ):
        require_file(
            resolved_paths.get(
                "rubin_opsim_db_path",
                None,
            ),
            "Rubin OpSim database",
        )

    selected = select_catalog_events(
        catalog_path,
        population,
        selection_cfg,
        cli_max_events=args.max_events,
    )

    output_root = resolve_path(
        output_cfg.get(
            "root_dir",
            "./runs",
        ),
        config_dir,
    )

    run_name = str(
        output_cfg.get(
            "run_name",
            "gbtds_catalog_run",
        )
    )

    run_dir = (
        output_root
        / run_name
        / population
    )

    model_dir = (
        run_dir
        / "models"
    )

    fit_dir = (
        run_dir
        / "fits"
    )

    results_dir = (
        run_dir
        / "results"
    )

    config_out = (
        run_dir
        / "config"
    )

    for path in (
        model_dir,
        fit_dir,
        results_dir,
        config_out,
    ):
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

    shutil.copy2(
        config_path,
        config_out
        / "config_used.json",
    )

    (
        config_out
        / "CONFIG.SHA256"
    ).write_text(
        f"{sha256(config_path)}  config_used.json\n"
    )

    simfit_kwargs = (
        build_simfit_kwargs(
            config,
            simulation_model,
        )
    )

    print("=" * 80)
    print("MATERIALIZED CATALOG RUN")
    print("=" * 80)

    print(
        "config             =",
        config_path,
    )

    print(
        "catalog            =",
        catalog_path,
    )

    print(
        "population         =",
        population,
    )

    print(
        "simulation model   =",
        simulation_model,
    )

    print(
        "selected events    =",
        len(selected),
    )

    print(
        "first catalog idx  =",
        int(
            selected.iloc[0][
                "catalog_event_index"
            ]
        ),
    )

    print(
        "last catalog idx   =",
        int(
            selected.iloc[-1][
                "catalog_event_index"
            ]
        ),
    )

    print(
        "unique fields      =",
        selected[
            "field_id"
        ].nunique(),
    )

    print(
        "Roman ephemerides  =",
        path_ephemerides,
    )

    print(
        "Rubin OpSim        =",
        resolved_paths.get(
            "rubin_opsim_db_path"
        ),
    )

    print(
        "Rubin throughputs  =",
        resolved_paths.get(
            "rubin_throughputs_dir"
        ),
    )

    print(
        "run directory      =",
        run_dir,
    )

    print(
        "sim_fit kwargs     =",
        simfit_kwargs,
    )

    print()
    print(
        selected.head(
            min(10, len(selected))
        ).to_string(
            index=False
        )
    )

    if args.prepare_only:
        print()
        print("=" * 80)
        print("PREPARE-ONLY VALIDATION PASSED")
        print("=" * 80)
        return

    # Import only after the config has established Rubin/Roman paths.
    from catalog.event_reader import (
        load_event_from_catalog,
    )

    from functions_roman_rubin import (
        sim_fit,
    )

    algo = str(
        config.get(
            "fit",
            {},
        ).get(
            "algorithm",
            "TRF",
        )
    )

    for k, record in selected.iterrows():

        catalog_event_index = int(
            record[
                "catalog_event_index"
            ]
        )

        event = load_event_from_catalog(
            catalog_path,
            catalog_event_index,
            expected_population=population,
        )

        # `i` is the collision-free simulation identity.
        seed = int(
            event[
                "simulation_seed"
            ]
        )

        kwargs = dict(
            simfit_kwargs
        )

        kwargs[
            "custom_system"
        ] = event

        print()
        print("=" * 80)

        print(
            f"EVENT {k + 1}/{len(selected)}"
        )

        print(
            "catalog_event_index =",
            catalog_event_index,
        )

        print(
            "field_id            =",
            event["field_id"],
        )

        print(
            "RA, Dec             =",
            event["ra"],
            event["dec"],
        )

        print("=" * 80)

        sim_fit(
            i=seed,
            system_type=population,
            model=simulation_model,
            algo=algo,
            path_TRILEGAL_set=None,
            path_GENULENS_set=None,
            path_to_save_model=str(
                model_dir
            ),
            path_to_save_fit=str(
                fit_dir
            ),
            path_ephemerides=str(
                path_ephemerides
            ),
            path_to_save_results=str(
                results_dir
            ),
            **kwargs,
        )


if __name__ == "__main__":
    main()
