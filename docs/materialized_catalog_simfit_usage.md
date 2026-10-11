# Materialized-catalog simulation and fitting

## Active entry point

The active launcher for simulation and fitting from the materialized GBTDS
event catalog is:

    python -u launch_catalog_simfit_from_config.py \
        --config config_files/gbtds_300_v1_amax_v1.json

The JSON configuration defines the materialized event catalog, Roman and
Rubin inputs, event selection, simulation settings, fit settings, and output
location.

The command line can additionally restrict the configured run to one
population or to a small number of selected events.

## Safe test example

For exploratory runs, do not write into the production output directory.
Start from the production configuration and change only the output location:

    python - <<'PY'
    import json
    from pathlib import Path

    src = Path("config_files/gbtds_300_v1_amax_v1.json")
    dst = Path("/tmp/gbtds_300_v1_example.json")

    cfg = json.loads(src.read_text())

    cfg.setdefault("output", {})
    cfg["output"]["root_dir"] = "/tmp/roman_rubin_example"
    cfg["output"]["run_name"] = "planets_first_3"

    dst.write_text(json.dumps(cfg, indent=2) + "\n")

    print("Wrote:", dst)
    PY

First inspect which catalog events would be selected:

    python -u launch_catalog_simfit_from_config.py \
        --config /tmp/gbtds_300_v1_example.json \
        --population Planets_systems \
        --max-events 3 \
        --prepare-only

`--prepare-only` resolves the configuration, catalog, population, event
selection, observing inputs, simulation model, fit options, and output
location without simulating or fitting the events.

After verifying the printed selection, execute the same sample by removing
`--prepare-only`:

    python -u launch_catalog_simfit_from_config.py \
        --config /tmp/gbtds_300_v1_example.json \
        --population Planets_systems \
        --max-events 3

## Output layout

For each population the launcher creates:

    <output.root_dir>/
        <output.run_name>/
            <population>/
                models/
                fits/
                results/
                    true/
                    fit_rr/
                    fit_roman/
                config/
                    config_used.json
                    CONFIG.SHA256

`config_used.json` records the exact JSON configuration used by the run and
`CONFIG.SHA256` records its checksum.

## Deterministic population slicing

A large population can be divided into deterministic execution chunks using:

    {
      "selection": {
        "population_row_start": 0,
        "population_row_stop": 125
      }
    }

The following non-overlapping chunk can use:

    {
      "selection": {
        "population_row_start": 125,
        "population_row_stop": 250
      }
    }

The slice is applied after selecting the population, applying the configured
`simulate_amax` condition, and stable sorting by `catalog_event_index`.

Therefore:

- `catalog_event_index` is the permanent event identity and simulation seed.
- `population_row_start` and `population_row_stop` are execution-partition
  coordinates inside the selected population.

Population-row limits must not be interpreted as ranges of
`catalog_event_index`.

## Fit configuration

A typical fit section is:

    {
      "fit": {
        "algorithm": "TRF",
        "model": "same_as_simulation",
        "parallax": true,
        "initial_guess": null,
        "bounds": null,
        "optimizer_options": null,
        "time_window": null
      }
    }

For the current materialized catalog:

- FFP uses FSPL.
- BH uses FSPL.
- Planets_systems uses USBL.

`model: "same_as_simulation"` resolves to the model associated with the
selected population.

`time_window: null` means that the full available light curve is fitted.

Fit initialization, bounds, optimizer settings, and execution partitioning
belong to the simulation/fitting configuration. They must not be implemented
by modifying the materialized physical event catalog.

## USBL trajectory coordinates

For USBL, the independent trajectory coordinates depend on the realized
pyLIMA origin:

- `center_of_mass` uses `t0`, `u0`.
- shifted origins, including caustic-centered origins, use
  `t_center`, `u_center`.

The fit parameter order, initial guess, bounds, and parallax reference epoch
use the active coordinates associated with the realized origin.

## Dataset-specific true chi-square

The Roman+Rubin fit is compared against the true-model chi-square evaluated
on the complete active Roman+Rubin dataset:

    chi2_true
    n_data_true
    delta_chi2_true = chi2_fit - chi2_true

The Roman-only fit is compared against an independently evaluated true-model
chi-square using exactly the W149 dataset used by that fit.

The true-event table additionally stores:

    chi2_true_roman
    n_data_true_roman

Degrees of freedom are computed from the dataset actually used by each fit:

    dof = N_data - N_fit_parameters

where `N_fit_parameters` is the length of pyLIMA's fitted `best_model`
vector, including the active photometric flux parameters.

## Validated planetary example

Catalog event `616` was used for the controlled end-to-end validation of the
USBL-origin correction and dataset-specific diagnostics.

Validated Roman+Rubin result:

    N_data          = 52786
    dof             = 52767
    chi2_true       = 52526.60129742518
    chi2_fit        = 52494.36787030658
    delta_chi2_true = -32.23342711859732

Validated Roman-only result:

    N_data          = 51606
    dof             = 51595
    chi2_true       = 51358.153107152466
    chi2_fit        = 51341.460786895994
    delta_chi2_true = -16.692320256472158

The detailed scientific and technical validation is recorded in:

    docs/materialized_simfit_validation_2026-10-11.md

## Before production

Before submitting the full catalog:

1. Freeze the exact JSON configuration.
2. Validate every intended population/chunk with `--prepare-only`.
3. Verify complete and non-overlapping population-row ranges.
4. Record the frozen source commit and configuration SHA256.
5. Run a small real-fit smoke test using the frozen configuration.
6. Submit the production arrays only after those checks pass.
