# Materialized catalog simulation-fit validation — 2026-10-11

## Scope

This note records the scientific and technical validation performed before
large-scale simulation and fitting of the materialized Roman–Rubin event
catalog.

The validation concerns:

- USBL trajectory coordinates and pyLIMA origin handling;
- initialization and local fit bounds;
- true-model chi-square diagnostics for Roman+Rubin and Roman-only fits;
- degrees of freedom for each fitted dataset;
- runtime measurements for a controlled planetary event.

No simulation RNG ordering was intentionally changed by these corrections.

## USBL trajectory-coordinate correction

For a USBL model pyLIMA uses different independent trajectory coordinates
depending on the adopted origin.

- `center_of_mass` uses `t0`, `u0`.
- shifted origins such as `central_caustic`, `second_caustic`, and
  `third_caustic` use `t_center`, `u_center`.

The previous fit preparation always extracted transformed `t0`, `u0` values
and could then label them as `t_center`, `u_center`.  For caustic-centered
catalog events this could initialize the optimizer around the wrong
trajectory.

`fit_lc.py` was made origin-aware for:

- initial physical parameters;
- fit parameter order;
- initial-guess construction;
- narrow, wide, and custom bounds;
- the parallax reference epoch used by the fit.

The production simulation-fit path passes the realized USBL origin explicitly.

## Controlled validation event

Catalog event:

- population: `Planets_systems`
- `catalog_event_index = 616`
- field: `gbtds_base_y000_x000`

True independent USBL parameters used in the controlled fit:

- `t_center = 2461680.570076437`
- `u_center = 0.0005644745982075469`
- `tE = 44.302386694968845`
- `rho = 0.000724605574603695`
- `separation = 4.031675499757275`
- `mass_ratio = 8.202960644306895e-08`
- `alpha = 5.159680553729883`
- `piEN = -0.006612866353513408`
- `piEE = 0.011374786735280003`

The controlled validation config used:

- `initial_guess = "truth"`
- local custom bounds centered on truth
- the same USBL model for simulation and fit
- parallax enabled in simulation and fit
- the full simulated light curve.

Both the Roman+Rubin and Roman-only fits received exactly the true
caustic-centered coordinates as their initial physical parameters.

## True chi-square diagnostics

The existing `chi2_true` remains the true-model chi-square evaluated on all
active Roman+Rubin simulated data.

For event 616:

- `N_RR = 52786`
- `chi2_true_RR = 52526.60129742518`
- `chi2_fit_RR = 52494.36787030658`
- `delta_chi2_true_RR = -32.23342711859732`

A separate Roman-only true chi-square is now evaluated immediately before the
Roman-only fit, using the same true physical model and exactly the W149
light curve used by that fit.

For event 616:

- `N_Roman = 51606`
- `chi2_true_Roman = 51358.153107152466`
- `chi2_fit_Roman = 51341.460786895994`
- `delta_chi2_true_Roman = -16.692320256472158`

This replaces the previous invalid comparison of the Roman-only fit chi-square
against the Roman+Rubin true chi-square.

## Degrees of freedom

For saved fit summaries the degrees of freedom are defined from the exact
fitted dataset:

`dof = N_data - N_fit_parameters`

where `N_fit_parameters` is the length of pyLIMA's `best_model` vector,
including physical and active-telescope photometric flux parameters.

For event 616:

- Roman+Rubin: `52786 - 19 = 52767`
- Roman-only: `51606 - 11 = 51595`

This avoids inferring the fitted dataset from the structure of a saved event.

## Runtime

Controlled event 616 timings:

- validation before adding Roman-only true chi-square:
  - internal simulation-fit time: about 22.30 s
  - Slurm elapsed time: 31 s
- validation including Roman-only true chi-square:
  - internal simulation-fit time: about 17.16 s
  - Slurm elapsed time: 24 s

The difference is consistent with ordinary runtime variability.  The
additional Roman-only true-model evaluation did not introduce a concerning
runtime cost.

These timings were obtained with the controlled truth initial guess and local
bounds and must not be interpreted as representative production USBL timing
until the production initialization/bound configuration is frozen and a
broader timing sample is measured.

## Test coverage

Dedicated regression tests cover:

- shifted-origin USBL trajectory parameter names;
- center-of-mass USBL trajectory parameter names;
- truth initial guesses and bounds under both origin conventions;
- parallax reference epoch consistency;
- Roman-only true chi-square evaluation on W149 only;
- ordering of the Roman true-chi-square evaluation before the Roman-only fit;
- dataset-specific degrees of freedom.

The full test suite must pass before this change is committed.

## Production note

The controlled validation configuration used `initial_guess="truth"` and
explicit local bounds.  Pilot configuration files under the run directory are
not part of this source-code commit.  The final production configuration must
be frozen separately before the full catalog run.

## Configuration usage example

The active materialized-catalog launcher, safe test procedure, deterministic
population slicing, output structure, and fit-configuration example are
documented in:

`docs/materialized_catalog_simfit_usage.md`
