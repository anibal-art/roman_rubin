# Legacy chunk-based simulation/fitting pipeline

This directory contains the superseded Roman–Rubin simulation/fitting
orchestration that consumed source/lens catalogue chunks directly.

Its historical execution model used inputs such as:

- `TRILEGAL_chunk_<j>.csv`
- `Genulens_chunk_<j>.csv`
- `n_db_file`
- `launch_simfit_from_config.py`
- `simfit_parallel_runner.py`
- the associated Slurm launcher and JSON configs

## Active pipeline

The active simulation/fitting entry point is:

    launch_catalog_simfit_from_config.py

with the current materialized-catalog configuration:

    config_files/gbtds_300_v1_amax_v1.json

The active pipeline consumes a materialized event catalogue whose rows
already carry the physical microlensing realization, materialized
blending, caustic origin where applicable, field identity, and sky
coordinates.

## Important distinction

The catalogue-production pipeline is NOT legacy.

In particular, the following remain active and scientifically essential:

- `stellar_population/`
- `catalog/`
- `ulens_params.py`
- catalogue generation, validation, Amax materialization and assembly code

Those components define how the physical event catalogue is generated
and remain part of the reproducible Roman–Rubin workflow.

This directory is retained only for historical reproducibility of the
old direct-from-chunks simulation/fitting orchestration.

Active code must not import modules from this directory.
