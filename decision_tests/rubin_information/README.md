# Rubin information experiments

This directory freezes the first validated experiments used to study how
Rubin observations contribute information to Roman microlensing events.

The scripts in this commit are an exact snapshot of the versions validated
interactively on CHE. Scientific logic has not been refactored in this commit.

## Scope

Two production-catalog events are used as controlled examples:

- FFP: `catalog_event_index=10`, FSPL, `simulation_seed=10`
- BH: `catalog_event_index=308`, FSPL, `simulation_seed=308`

Both events are loaded from the materialized production catalog and simulated
through the current production-consistent Roman+Rubin path.

Roman-only event selection uses the validated forecast criteria:

- Roman W149 only
- `Delta chi2 >= 300`
- `n3sigma >= 6`

## Experiment 1: chronological information accumulation

`exp1_cumulative_information.py`

Roman data are held fixed.

The real Rubin observations are added one by one in their true chronological
order. Each point retains its original time, uncertainty, geometry, and band.

The main diagnostic is

`G_piE(N) = sigma_R(piE) / sigma_R+Rubin_N(piE)`.

This experiment asks when the Rubin information enters as the survey proceeds.

Validated reference values include:

- FFP 10:
  - `sigma_R(piE) = 210315.43384561857`
  - `sigma_full(piE) = 209573.41390160637`
  - `G_piE = 1.003540620588261`

- BH 308:
  - `sigma_R(piE) = 3.841050707699907`
  - `sigma_full(piE) = 3.5322032958487863`
  - `G_piE = 1.0874376093284588`

## Experiment 2: permutation ladders

`exp2_permutation_ladders.py`

The real Rubin observations are not moved in time.

A deterministic analysis-only permutation changes only the order in which
existing observations are selected. Once a subset is chosen, its lightcurve
rows are restored to chronological order before Fisher evaluation.

Each permutation therefore defines nested subsets

`S_1 subset S_2 subset ... subset S_10`.

The experiment separates:

- number of Rubin observations `N`;
- local photometric nuisance rank;
- residual degrees of freedom;
- temporal coverage;
- information on `piE`.

The analysis-only RNG is deterministic and is never passed to simulation or
fitting.

For the validated experiment:

- 100 permutations per event;
- `N=1,...,10`;
- 2000 Fisher evaluations;
- FFP 10, band z;
- BH 308, band z.

Important result:

A fixed number of observations does not determine the information on parallax.
The temporal placement of the selected observations matters strongly.

The historical `min_rubin_points=5` does not receive a special justification
from this experiment.

No final `N_min` is selected here.

## Experiment 3: nested tau windows

`exp3_tau_windows.py`

Define

`tau = (t - t0) / tE`.

Roman is held fixed and Rubin observations are added using nested windows

`|tau| <= w`.

Validated windows:

`w = 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64`.

The fractional recovery of the full Rubin improvement is

`(sigma_R - sigma_window) / (sigma_R - sigma_full)`.

For BH 308:

- `|tau| <= 2`: about 24.4% of the full improvement
- `|tau| <= 4`: about 32.1%
- `|tau| <= 8`: about 66.3%
- `|tau| <= 16`: about 92.3%
- `|tau| <= 32`: about 98.5%
- `|tau| <= 64`: about 99.95%

For FFP 10 the Rubin information remains extremely weak. The small improvement
must not be interpreted as successful parallax characterization.

## Diagnostic: equal N, different temporal placement

`diagnose_sameN_location.py`

This compares fixed-size subsets with different temporal placements.

It demonstrates directly that the same `N`, nuisance rank, and residual degree
of freedom can yield very different parallax information.

## Numerical checks

The experiments retain Fisher diagnostics including:

- primary/alternate finite-difference solutions;
- derivative stability;
- SVD rank and condition number;
- nuisance rank;
- nested Fisher monotonicity.

Small negative eigenvalues of incremental Fisher matrices observed at relative
levels around `1e-16` are consistent with floating-point roundoff.

## Reference results

`reference_results.json` is generated directly from the validated parquet
outputs produced on CHE.

These values are intended for reproducibility checks, not as population-level
scientific conclusions.

## Current limitations

This snapshot still contains CHE-specific path assumptions inherited from the
validated scripts.

A later, separate commit should make the experiment drivers portable across
machines without changing their scientific logic.

Population-level validation, TRF boundary tests, and the final Rubin-band
minimum-point criterion remain open.
