"""Minimal, generic simulation core.

Receives an already-built pyLIMA model and an already-decided
parameter vector (physical + flux parameters, in that order, as
`functions_roman_rubin.sim_event` assembles them) and does only the
two deterministic pyLIMA calls common to any model/survey:
`pyLIMA_model.compute_pyLIMA_parameters` and `pyLIMA.simulations
.simulator.simulate_lightcurve(..., add_noise=False)`.

Nothing here samples anything: noise is added by a later,
instrument-specific stage (`apply_roman_rubin_photometry`), which
stays in `functions_roman_rubin.py` and is not called from here.

This module does not know about Roman, W149, "ground" telescopes, or
any other survey-specific concept -- reinforcing the theoretical flux
for non-Roman telescopes (needed because `add_noise=False` only fills
Roman's lightcurve table by itself) is survey-specific and stays where
it already lived: `functions_roman_rubin.inject_model_flux_for_ground_
telescopes`, called from the adapter (`sim_event`) right after this
module returns. Not duplicated here.

Deliberately excluded from this module (and why it must stay out):

- reading data/catalog rows, deciding blending or caustic_origin, or
  generating parameters -- that is `simulation/realization.py`'s job,
  finished before this module is ever called;
- any knowledge of `event_seed`, `system_type`, `catalog_mode`, or
  Amax -- this module never inspects a catalog row;
- the monkey-patching points LRT relies on
  (`functions_roman_rubin.{model_choice, flux_parameters_model,
  extract_lightcurves_for_fit, fit_rubin_roman}`): those calls stay in
  `functions_roman_rubin.sim_event` itself, called unqualified, so
  LRT's runtime replacement of those module attributes keeps working.
  This module is called *after* all of them have already run.

Only code whose extraction does not change WHEN anything is drawn from
the RNG lives here. Nothing below touches `np.random` or any `rng`.
"""

from pyLIMA.simulations import simulator


def simulate_light_curve(
    pyLIMA_model,
    parameter_vector,
    sim_timer=None,
):
    """Compute pyLIMA_parameters and simulate the noiseless light
    curve for an already-built model and an already-decided parameter
    vector.

    `pyLIMA_model.event.telescopes` lightcurves are populated in place
    (pyLIMA's own convention). `sim_timer`, if given, is any object
    exposing `.start(name)`/`.stop(name)` (duck-typed, e.g.
    `timing_utils.StageTimer`) -- used only to keep the same profiling
    granularity the caller had before this code lived here; passing
    `None` skips timing entirely.
    """

    pyLIMA_parameters = pyLIMA_model.compute_pyLIMA_parameters(
        parameter_vector
    )

    if sim_timer is not None:
        sim_timer.start("simulation_pylima_lightcurve")

    # pyLIMA computes the Full-parallax geometry when the model is
    # constructed. simulator.simulate_lightcurve() calls
    # define_pyLIMA_standard_parameters() again, which otherwise
    # recomputes exactly the same geometry at the same observation
    # times.
    #
    # Reuse is allowed only when the already-computed photometric
    # shifts are present and aligned with every current lightcurve.
    parallax_model = getattr(
        pyLIMA_model,
        "parallax_model",
        ["None", 0.0],
    )

    reuse_existing_parallax = (
        parallax_model[0] != "None"
    )

    if reuse_existing_parallax:

        for tel in pyLIMA_model.event.telescopes:

            if tel.lightcurve is None:
                continue

            shifts = getattr(
                tel,
                "deltas_positions",
                {},
            ).get(
                "photometry",
                None,
            )

            expected = (
                2,
                len(tel.lightcurve),
            )

            if (
                shifts is None
                or getattr(shifts, "shape", None) != expected
            ):
                raise RuntimeError(
                    "Cannot reuse precomputed parallax geometry: "
                    f"{tel.name}: "
                    f"shape={getattr(shifts, 'shape', None)}, "
                    f"expected={expected}"
                )

        event = pyLIMA_model.event

        original_compute_parallax = (
            event.compute_parallax_all_telescopes
        )

        skipped_calls = 0

        def _reuse_precomputed_parallax(*args, **kwargs):
            nonlocal skipped_calls
            skipped_calls += 1
            return None

        event.compute_parallax_all_telescopes = (
            _reuse_precomputed_parallax
        )

        try:
            simulator.simulate_lightcurve(
                pyLIMA_model,
                pyLIMA_parameters,
                add_noise=False,
            )
        finally:
            event.compute_parallax_all_telescopes = (
                original_compute_parallax
            )

        if skipped_calls != 1:
            raise RuntimeError(
                "Unexpected pyLIMA parallax call count inside "
                "simulate_lightcurve: "
                f"{skipped_calls}; expected exactly 1"
            )

    else:

        simulator.simulate_lightcurve(
            pyLIMA_model,
            pyLIMA_parameters,
            add_noise=False,
        )

    if sim_timer is not None:
        sim_timer.stop("simulation_pylima_lightcurve")

    return pyLIMA_parameters
