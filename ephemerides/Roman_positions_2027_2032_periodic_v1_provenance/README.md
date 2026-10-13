# Roman production ephemeris: periodic v1

## Purpose

This directory contains the complete provenance record for

`Roman_positions_2027_2032_periodic_v1.npy`.

The `.npy` file is the canonical Roman geocentric ephemeris used for
Roman-Rubin microlensing simulations after the legacy time-shifted
ephemeris was rejected.

## Data format

The production array has four float64 columns:

1. `JD_UTC`
2. `RA_deg_geocentric`
3. `Dec_deg_geocentric`
4. `Earth_Roman_distance_AU`

The grid runs daily from 2027-01-01 through 2032-01-01 inclusive.

The encoded vector is Earth -> Roman. pyLIMA 1.9.8 internally stores
the opposite Cartesian sign when it evaluates spacecraft positions.
That behavior was independently reproduced and is documented in
`validation/audit_pylima_periodic_candidate.log`.

## Construction model

The available Roman/Horizons geocentric positions are transformed to
a Sun-Earth rotating Cartesian frame.

The basis is defined by:

- +X: anti-solar direction
- +Y: normalized `(ecliptic north) x X`
- +Z: normalized `X x Y`
- obliquity: 23.439291111 deg

In each rotating-frame coordinate the model is

    r_i(t) =
        C_i
        + A_i cos(2 pi (JD - t_ref) / P)
        + B_i sin(2 pi (JD - t_ref) / P)

with a common period P.

The period was searched over 175--190 days with 0.01-day spacing.
The adopted value is approximately 181.21 days.

The coefficients, exact period, reference epoch, fit interval, and
all hashes are stored in `manifest.json`.

## Model-selection protocol

The first-harmonic periodic model class was selected using a blind
validation strategy: the model was first fitted using 2027 information
and tested against independent 2028 support.

Only after the model class passed that validation was the final
production candidate refitted using all available Horizons support.

No Horizons trajectory was extrapolated into the 2030--2031 Roman
seasons. Instead, the validated periodic model is evaluated there.

## Reproducibility

Direct construction dependencies are frozen under `construction/`.

The final product was also checked independently through pyLIMA.
The final audit reproduced pyLIMA's spacecraft sign convention and
distance to numerical precision (~1e-17 AU).

`SHA256SUMS` identifies every frozen artifact.

Important historical detail: `decision_tests/` was still untracked in
Git at the moment this product was promoted. Therefore the repository
commit at promotion is recorded for context, but the authoritative
construction record is the frozen generator, input, metadata,
validation files, software environment, and cryptographic hashes stored
here.

## Activation

Creation of this product does NOT by itself replace
`ephemerides/Roman_positions.npy` and does not modify production
configuration. Activation must be done separately and followed by a
simulation regression test.
