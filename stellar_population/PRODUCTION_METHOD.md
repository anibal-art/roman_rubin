# Roman + Rubin microlensing production methodology

This document defines the production configuration used to construct
the precomputed microlensing event catalogues for the Roman + Rubin
characterization experiment.

The generated catalogues are controlled parameter-space samples.
They are not intended to represent an absolute Galactic event-rate
prediction.

## 1. GBTDS spatial footprint

The production footprint is represented by

`stellar_population/config/gbtds_trilegal_cells_production.csv`

and contains 510 enabled adaptive spatial cells.

Each cell stores its physical target area, the TRILEGAL sampling area,
the area ratio, seasonal Roman coverage, and extinction information.

The TRILEGAL sampling area is 1e-4 deg^2 per cell. The physical target
areas differ because the spatial tessellation is adaptive.

The 38 rejected/tiny/fallback cells remain documented in

`stellar_population/config/gbtds_trilegal_cells_final.csv`

but are not part of the production set.

## 2. Extinction

Spatial extinction is based on the Surot et al. (2020) VVV extinction
map.

Per-cell quantities include E(J-Ks), Av(infinity), and a differential
extinction width.

The conversion currently adopted in the footprint construction is

Av / E(J-Ks) = 6.

TRILEGAL receives

`extinction_kind = 2`

so that the supplied normalization corresponds to extinction at
infinity. TRILEGAL then applies its internal distance dependence.

`extinction_sigma` is supplied per cell as the fractional differential
extinction.

## 3. TRILEGAL source catalogue

TRILEGAL version/interface:

`https://stev.oapd.inaf.it/cgi-bin/trilegal_1.7`

The photometric system is the combined LSST + Gaia + Euclid +
Roman2024 table

`tab_mag_odfnew/tab_mag_lsstR1.9_gaiaEDR3_euclid_Roman2024.dat`

Roman magnitudes are Vega magnitudes in this table.

The catalogue selection band is Roman F146 (filter index 22).

The production faint limit is F146 = 28 Vega mag.

TRILEGAL provides the source analogue properties, including:

- ugrizy photometry
- Roman F146 and other Roman bands
- luminosity
- effective temperature
- stellar radius information through L and Teff
- Galactic component
- source extinction
- source distance modulus

TRILEGAL source distances are used only for analogue matching.
After matching, the authoritative source distance is the GENULENS D_S.

## 4. GENULENS geometry and kinematics

GENULENS provides the Galactic lens/source geometry and relative
proper motion.

Production configuration:

- `small_gamma = 1`
- `binary = 0`
- `remnant = 1`
- nuclear stellar disk enabled

The quantities used downstream are primarily

- D_L
- D_S
- mu_rel
- mu_rel_N
- mu_rel_E
- source/lens Galactic component information

GENULENS binary structure and GENULENS lens mass are not used to define
the final simulated event classes.

GENULENS quantities such as

- wtj
- M_L
- t_E
- theta_E
- pi_E
- pi_EN
- pi_EE

are retained in the matched catalogues only as provenance/diagnostic
columns.

`wtj` is not applied as a science weight in the controlled
characterization experiment.

Consequently the catalogues should not be interpreted as an absolute
Galactic microlensing event-rate prediction.

## 5. TRILEGAL--GENULENS source matching

GENULENS is authoritative for D_L, D_S and the relative proper-motion
vector.

TRILEGAL supplies a photometric/stellar analogue for the source.

The matching procedure requires:

1. compatible Galactic source component;
2. |delta mu0| <= 0.05 mag.

Within the accepted distance-modulus window a TRILEGAL source analogue
is selected deterministically from a seeded random draw.

GENULENS component mapping:

- iS = 0--6 -> TRILEGAL Gc = 1, thin disk
- iS = 7    -> TRILEGAL Gc = 2, thick disk
- iS = 8    -> TRILEGAL Gc = 4, bulge/bar
- iS = 10   -> TRILEGAL Gc = 3, stellar halo

GENULENS iS = 9 (nuclear stellar disk) currently has no direct
TRILEGAL analogue and is written to the unmatched catalogue.

After matching, TRILEGAL apparent magnitudes are shifted by

m_final = m_TRILEGAL + delta_mu0

to place the analogue at the GENULENS source distance.

The cell extinction is held fixed over this <= 0.05 mag
distance-modulus displacement.

## 6. Microlensing physical parameters

The event-parameter authority is

`ulens_params.event_param`.

The Einstein radius, event timescale, microlensing parallax, source
size and related quantities are recomputed from the imposed
class-specific lens parameters together with the GENULENS geometry.

The microlensing-parallax vector is aligned with the GENULENS
relative-proper-motion vector:

pi_E_vector parallel to mu_rel_vector.

The GENULENS pi_E amplitude is not copied into the final event.

## 7. Free-floating planets

The primary lens mass is sampled log-uniformly over

0.01 M_earth <= M_FFP <= 13 M_Jup.

The source properties and Galactic geometry are inherited from the
matched TRILEGAL--GENULENS row.

## 8. Compact/BH lens scan

The primary compact-lens mass is sampled log-uniformly over

1 <= M_L / M_sun <= 100.

This is a controlled lens-mass scan, not an assumed astrophysical
black-hole mass function.

## 9. Binary-lens parameter scan

Binary lenses use three independent primary parameters:

M_host, q, s.

They are sampled independently and uniformly in logarithmic space:

0.08 <= M_host / M_sun <= 10

1e-8 <= q <= 1e-1

0.1 <= s <= 10

The companion mass is derived from

M_companion = q M_host.

No independent companion-mass cut is imposed.

Therefore the complete binary-lens scan includes planetary,
brown-dwarf and stellar-mass companions. Scientific sub-samples may be
defined afterward from the derived companion mass.

The projected separation is primary in Einstein-radius units.
The physical projected separation is derived as

a_perp [AU] = s theta_E [mas] D_L [kpc].

An independent orbital semi-major axis is not sampled for
`Planets_systems`.

## 10. Roman F146 photometric noise

The Roman single-epoch photometric-noise model is implemented in

`stellar_population/noise_models/roman_f146.py`

and integrated into the simulation pipeline through

`stellar_population/noise_models/roman_photometry.py`.

The model is based on Pandeia Roman R2026.1 using the GBTDS
`gbtds_mid_5stripe` background configuration and the local
IM_66_6_V2 Revision-H MultiAccum compatibility definition.

The underlying detector/background/PSF model remains Pandeia R2026.1;
the local reference-data patch adds the Revision-H GBTDS read pattern.

The production interpolation retains the individual WFI detector
responses.

Full saturation produces an invalid photometric point.

Partial saturation is retained when Pandeia indicates that a usable
partial ramp remains.

Unsaturated/usable points use the Pandeia S/N interpolation.

No arbitrary systematic photometric-error floor is added.

For a microlensing light curve, the noise model is evaluated on the
instantaneous total Roman flux, not only on the unmagnified source.

The simulation calls pyLIMA with `add_noise=False`. Instrument-specific
Roman and Rubin noise is applied afterward, and magnitude/flux/error
representations are synchronized before detection criteria and
likelihood evaluation.

## 11. Rubin photometry

Rubin cadence and visits come from the Rubin simulation/OpSim
infrastructure used by the main simulation pipeline.

Rubin photometric uncertainties remain handled by the Rubin-specific
photometric model in `functions_roman_rubin.py`.

Roman Pandeia noise and Rubin photometric noise are therefore applied
as separate instrument-specific models.

## 12. Reproducibility

Source matching and class-specific event generation use deterministic
seeds derived from the base seed, field identifier, GENULENS event
identifier and event class.

The precomputed-event builder writes:

- matched geometry/source catalogue
- unmatched GENULENS catalogue
- FFP events
- compact/BH events
- binary-lens events
- JSON production metadata

The JSON metadata records the source and geometry authorities, matching
rule, seeds, class-specific priors and output paths.

Generated Parquet production products are not committed to Git.
The code, configurations and methodology required to regenerate them
are version controlled.

## 13. Amax detectability prefilter (experimental, not active)

`catalog/amax.py` implements a closed-form/USBL peak-magnification
(Amax) estimate intended as a future catalog-level detectability
prefilter, with the per-band blending and caustic-origin realization
materialized by `catalog/blending.py` and `catalog/caustic_origin.py`.

This is **experimental and not wired into production** as of this
writing:

- it is not applied as a filter when building the precomputed event
  catalogues (no rows are dropped based on it);
- a validation run (`scripts/validation/validate_catalog_amax.py`)
  found that the catalog's precomputed blending realization and the
  live simulation pipeline were not using the same realization — this
  has since been addressed architecturally (the simulation core can
  now consume a catalog's precomputed `blend_ratio_<band>` columns
  instead of always resampling), but Amax itself has not been
  re-validated end-to-end after that change, and no production
  catalogue has been regenerated or filtered with it;
- the two independent implementations of the Roman F146 S/N=5
  boundary (`catalog/amax.py::roman_f146_5sigma_vega` and
  `stellar_population/scripts/audit_f146_preproduction.py::roman_limit`)
  have not been reconciled; it is not yet established which, if
  either, should be the production authority.

Do not treat Amax as a certified detection criterion until this
section is updated to say otherwise.
