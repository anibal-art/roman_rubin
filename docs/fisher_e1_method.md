# E1 Fisher information - Roman-Rubin

## Purpose

E1 measures the local information available around the simulated truth and
quantifies the gain obtained by adding Rubin observations to Roman.

For every event, three paired configurations are evaluated:

- `R`: Roman only.
- `RR`: Roman plus usable Rubin bands.
- `RR_DPERP0`: same RR data definition, but Roman is placed at the Earth
  centre, removing the Roman-Earth satellite baseline.

E1 is a local-information experiment. It does not replace the global
parallax-identification experiment E2 or the null calibration E3.

## Physical parameters

For the FSPL validation cases the physical parameter vector is

    (t0, u0, tE, rho, piEN, piEE).

Each telescope additionally has local photometric nuisance parameters.

For `blend_flux_parameter="ftotal"` these are

    (fsource, ftotal).

## Weighted Jacobian and Fisher matrix

For each photometric datum,

    J_ki = (1 / sigma_k) dF_k / dtheta_i.

The Fisher matrix is

    F = J^T J.

Fisher matrices are retained for diagnostics, but the production covariance
is obtained from the SVD of the marginalized Jacobian rather than by directly
inverting J^T J.

## Why the pyLIMA numerical Jacobian is not used for physical columns

pyLIMA 1.9.8 uses

    scipy.optimize._numdiff.approx_derivative(
        ...,
        method="2-point",
    )

for models whose `Jacobian_flag == "Numerical"`.

The parameter vector contains the absolute Julian date t0, of order 2.46e6.
With the default SciPy relative scaling, the numerical perturbation in t0 is
too large for short-timescale FFP events.

An independent finite-difference audit found:

- excellent agreement for u0, tE, rho, piEN and piEE;
- a persistent several-percent discrepancy in the pyLIMA t0 derivative for a
  short FFP;
- a much smaller discrepancy for a long BH event.

Therefore E1 computes its physical derivatives independently.

No installed pyLIMA source file is modified.

## Physical finite differences

Physical derivatives use the five-point centred stencil

    f'(x) =
      [-f(x+2h) + 8 f(x+h) - 8 f(x-h) + f(x-2h)] / (12 h).

The truncation error is O(h^4).

The reference step is tied to the physical parameter scale rather than its
absolute numerical value.

In particular:

- the t0 step scales with tE, not absolute JD;
- the tE step scales with tE;
- the rho step scales with rho;
- parallax steps scale with their respective piE components, with a finite
  floor.

Several neighbouring step sizes are evaluated.

The algorithm selects the smaller member of the adjacent pair with the
smallest relative derivative disagreement.

## Version 2.1 stable-pair diagnostic

Version 2.1 retains both members of the selected stable finite-difference
pair:

- the primary derivative;
- the alternate derivative.

The full Fisher/SVD calculation is evaluated for both solutions without
additional light-curve evaluations.

The following quantities are stored:

- `fd_stability_sigma_piE_rel`
- `fd_stability_smin_rel`
- `fd_stability_weak_mode_angle_deg`
- `fd_stability_jacobian_rel`

These diagnostics answer the direct numerical question:

Does the inferred scientific solution change when the other equally plausible
member of the finite-difference stability pair is used?

This avoids defining physical identifiability through an arbitrary fixed
Fisher `rcond`.

## Analytical flux derivatives

For `ftotal` the model can be written locally as

    F = fsource * (A - 1) + ftotal.

Therefore

    dF / dfsource = A - 1
    dF / dftotal  = 1.

These columns were independently checked against finite differences and agree
essentially to machine precision.

## Flux nuisance marginalization

Write the weighted Jacobian as

    J = [J_phys  J_flux].

The column space of the nuisance Jacobian is obtained by SVD and projected
out:

    J_phys,marg = (I - Q_f Q_f^T) J_phys.

For Gaussian least squares this is equivalent to Schur-complement
marginalization.

A unit test explicitly verifies the equivalence between the projection and
the Schur complement.

## Local coordinate scaling

Before the physical SVD, local scaled coordinates are used.

For the current FSPL case:

    q_t0  = delta(t0) / tE
    q_tE  = delta(log tE)
    q_rho = delta(log rho)

while u0, piEN and piEE remain dimensionless.

Locally,

    J_q = J_phys,marg D,

where D = dx/dq.

This reduces dependence of the conditioning diagnostics on arbitrary
parameter units.

## SVD and covariance

The marginalized scaled Jacobian is decomposed as

    J_q = U Sigma V^T.

The singular values quantify the local information in orthogonal parameter
combinations.

For resolved numerical directions,

    C_q = V Sigma^(-2) V^T.

Working directly with J is numerically preferable to obtaining the covariance
by first forming and inverting J^T J because

    kappa(J^T J) = kappa(J)^2.

The Fisher matrices are still returned for diagnostics and output.

## Parallax magnitude

The parallax magnitude is

    piE = sqrt(piEN^2 + piEE^2).

For non-zero piE, the local uncertainty is propagated from the SVD covariance
using the gradient of piE.

E1 is not used to calibrate the null distribution at piE = 0. That belongs
to E3.

## Rubin-band rule

A Rubin band enters E1 only when at least five surviving photometric points
are available.

This does not define the event sample.

A Roman-detected event remains in the sample even when Rubin contributes no
usable band.

Therefore `RR == R` is a valid natural control.

## Validation event: FFP 10

Reproducible identifiers:

- population: `FFP`
- `catalog_event_index = 10`
- `simulation_seed = 10`

Only four Rubin z points survive the photometric filtering.

Because E1 requires at least five points per Rubin band, no Rubin band enters.

The validation therefore confirms exactly

    RR == R.

For Roman-only,

    sigma(piE) approximately 2.69e5.

The primary and alternate stable-pair solutions differ fractionally by

    4.25e-4

in sigma(piE), approximately 0.042 percent.

The scientific conclusion is not that this enormous uncertainty is measured
precisely. The conclusion is that this event is locally extremely poorly
constrained in parallax.

For `RR_DPERP0`, the primary/alternate difference in sigma(piE) is about

    1.51e-3

or approximately 0.15 percent.

This configuration is also flagged as numerically ill-conditioned.

## Validation event: BH 308

Reproducible identifiers:

- population: `BH`
- `catalog_event_index = 308`
- `simulation_seed = 308`

Five Rubin z points survive and therefore enter RR.

For Roman-only,

    sigma(piE) approximately 4.662

while the true parallax magnitude is approximately

    piE approximately 0.0123.

Thus the event is locally very poorly characterized in parallax.

The numerical Fisher result itself is highly stable.

For Roman-only the primary/alternate relative difference in sigma(piE) is

    8.89e-7.

RR and RR_DPERP0 show stability of the same order.

## Roman-at-Earth control

`RR_DPERP0` rebuilds Roman as an Earth-centred telescope.

Validation explicitly checks:

- Roman `location == "Earth"`;
- no spacecraft ephemeris remains;
- the topocentric position norm is exactly zero.

This removes the Roman-Earth satellite baseline while retaining the same
local data definition.

There is no requirement that RR contain more information than RR_DPERP0,
because the observing geometry itself is changed.

## Validation commands

Unit tests:

    python -m pytest -q \
        tests/test_fisher_information.py \
        tests/test_fisher_information_v2.py

Production-consistent real-event smoke:

    PYTHONPATH=. python -u \
        decision_tests/audit_fisher_production_events.py

Detailed finite-difference step scan:

    PYTHONPATH=. python -u \
        decision_tests/audit_fisher_v2_step_stability.py

Final v2.1 primary-versus-alternate stability audit:

    PYTHONPATH=. python -u \
        decision_tests/audit_fisher_v21_stability.py

## Independent-agent validation checklist

An independent validator should inspect and attempt to falsify the
implementation rather than accepting the stored smoke-test numbers.

Required checks:

1. Confirm that no installed pyLIMA source file is modified.
2. Confirm that physical derivatives use the five-point centred stencil.
3. Confirm that the t0 step is tied to tE rather than absolute Julian date.
4. Confirm that flux derivatives are analytical and correct for `ftotal`.
5. Verify that nuisance projection agrees with the Schur complement.
6. Confirm that the production covariance is obtained from SVD of the
   marginalized scaled Jacobian rather than direct inversion of the Fisher
   matrix.
7. Confirm the local coordinate transformation and covariance back-transform.
8. Verify the propagation of sigma(piE).
9. Confirm that primary and alternate derivatives are the two members of the
   same selected stable finite-difference pair.
10. Confirm that the `fd_stability_*` quantities compare complete scientific
    solutions and not only individual derivative columns.
11. Re-run FFP 10 and verify exactly `RR == R`.
12. Re-run BH 308 and verify that RR contains only the surviving z band.
13. Verify that `RR_DPERP0` places Roman exactly at Earth.
14. Confirm that Fisher output is not used as an event-selection condition.
15. Confirm that E1 remains a local-information experiment and is not
    interpreted as the global parallax-detection statistic.

## Known limitations

Fisher information is local and does not describe multimodality or discrete
u0-sign branches.

Extremely large uncertainties should be interpreted as lack of local
characterization rather than precise measurements of an enormous error bar.

E2 and E3 remain necessary for global parallax identification and
significance calibration.

FFP 10 is a useful negative control but does not demonstrate a Rubin
information gain.

Before population-scale scientific interpretation, an additional FFP with
substantial useful Rubin coverage should be validated.
