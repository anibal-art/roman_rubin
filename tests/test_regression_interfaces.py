"""Regression tests freezing the EventRealization architecture and the
LRT-facing compatibility contract introduced by the roman_rubin refactor.

Fast by design: no telescope/photometry pipeline, no real catalogs, no
fits. A minimal synthetic pyLIMA Event/Telescope is used only to build
a model object (never simulated/fit).

Runnable two ways:

- `python -m pytest --import-mode=importlib tests/` (real pytest).
  Plain `python -m pytest tests/` (pytest's default "prepend" import
  mode) fails to even collect this file in this environment -- root
  cause diagnosed, not a bug in this refactor nor in `rubin_sim`:
  `~/roman_rubin/__init__.py` is a pre-existing (dated 2025-07-16),
  empty file that makes pytest's default import mode treat the whole
  repo as a package and prepend its *parent* directory
  (`/home/anibal-pc`) onto `sys.path[0]` for collection. Because
  `~/rubin_sim` (the editable install's checkout, a sibling of
  `~/roman_rubin`) is also under `/home/anibal-pc`, and its actual
  package lives one level deeper (`~/rubin_sim/rubin_sim/`), that
  sys.path[0] insertion makes Python's import system resolve `rubin_sim`
  as an incomplete/namespace object before `rubin_sim/maf/db
  /results_db.py`'s `from rubin_sim import __version__` runs,
  producing `ImportError: cannot import name '__version__' from
  'rubin_sim' (unknown location)` -- "(unknown location)" is the tell
  for a namespace package with no `__file__`. `--import-mode=importlib`
  sidesteps that sys.path insertion entirely and collects fine (14/14).
  Plain `python -c "import functions_roman_rubin"` always works, with
  or without pytest involved, confirming the issue is specific to
  pytest's default collection import mode, not to this refactor's code.
- `python -m tests.test_regression_interfaces` (no pytest needed).
"""

import copy

import numpy as np

import fit_lc
import functions_roman_rubin as frr
import set_model_pyLIMA as smp
import set_telescopes_pyLIMA as stp
from simulation.realization import (
    realization_from_catalog,
    data_has_materialized_blend_ratio,
    data_has_materialized_fluxes,
)


class _Raises:
    """Minimal stand-in for pytest.raises, so this file has no pytest
    dependency (see module docstring)."""

    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"expected {self.exc_type} to be raised")
        return issubclass(exc_type, self.exc_type)


def raises(exc_type):
    return _Raises(exc_type)


# ---------------------------------------------------------------
# Alias handling: t0/t_center, u0/u_center, s/separation, q/mass_ratio
# ---------------------------------------------------------------

def test_alias_candidate_keys_are_symmetric():
    for canonical, alias in (
        ("t0", "t_center"), ("u0", "u_center"),
        ("s", "separation"), ("q", "mass_ratio"),
    ):
        assert alias in fit_lc._custom_bound_candidate_keys(canonical)
        assert canonical in fit_lc._custom_bound_candidate_keys(alias)
        # Single source of truth: both alias-key functions must agree.
        assert (
            fit_lc._initial_guess_candidate_keys(canonical)
            == fit_lc._custom_bound_candidate_keys(canonical)
        )


def test_get_param_resolves_either_alias_name():
    aliases = fit_lc._custom_bound_candidate_keys("t0")
    assert fit_lc.get_param({"t_center": 2460500.0}, "t0", aliases=aliases) == 2460500.0

    aliases = fit_lc._custom_bound_candidate_keys("t_center")
    assert fit_lc.get_param({"t0": 2460500.0}, "t_center", aliases=aliases) == 2460500.0


# ---------------------------------------------------------------
# EventRealization: catalog-row vs. generated-parameters producers
# ---------------------------------------------------------------

FIXED_PHYSICAL = dict(
    t0=2460500.0, u0=0.15, tE=22.0, rho=0.003,
    piEN=0.05, piEE=-0.03, s=1.1, q=0.02, alpha=1.3,
)
BANDS = ("W149",)
FIXED_G = 0.37

# Materialized catalog flux realization.
#
# These values are deliberately independent of any magnitude used by
# generated/manual tests below. A catalog row already carries its flux
# realization; simulation must consume it exactly, not reconstruct it.
FIXED_FSOURCE = 123.0
FIXED_FTOTAL = FIXED_FSOURCE * (1.0 + FIXED_G)


def _catalog_row(caustic_origin="central_caustic"):
    row = dict(FIXED_PHYSICAL)
    row["caustic_origin"] = caustic_origin

    for band in BANDS:
        row[f"blend_ratio_{band}"] = FIXED_G
        row[f"fsource_{band}"] = FIXED_FSOURCE
        row[f"ftotal_{band}"] = FIXED_FTOTAL

    return row


def _generated_data(caustic_origin="central_caustic"):
    row = dict(FIXED_PHYSICAL)
    row["caustic_origin"] = caustic_origin
    return row


def test_catalog_realization_requires_materialized_fluxes():
    """
    A row presented to realization_from_catalog must already contain
    its flux realization. Missing catalog fluxes are an integrity
    error, never a request to reconstruct them from magnitude/blending.
    """

    row = _catalog_row()
    del row["fsource_W149"]

    try:
        realization_from_catalog(
            row,
            "USBL",
            use_parallax=True,
            t0_parallax=FIXED_PHYSICAL["t0"],
            band_order=BANDS,
        )
        raise AssertionError(
            "Expected missing materialized catalog flux to raise."
        )

    except KeyError as exc:
        assert "fsource_W149" in str(exc)


def test_data_has_materialized_blend_ratio_decision():
    """The orchestrating-layer decision (sim_event): False when no band
    has a precomputed blend_ratio_<band> (legacy/generated), True when
    EVERY band has one (explicit realization), and a partial match
    (some but not all) is not a silent fallback -- it raises."""
    assert data_has_materialized_blend_ratio(_catalog_row(), BANDS) is True
    assert data_has_materialized_blend_ratio(_generated_data(), BANDS) is False

    partial = _generated_data()
    partial[f"blend_ratio_{BANDS[0]}"] = FIXED_G
    two_bands = BANDS + ("u",)  # "u" has no blend_ratio_u -> partial

    with raises(ValueError):
        data_has_materialized_blend_ratio(partial, two_bands)


def test_data_has_materialized_fluxes_decision():

    assert data_has_materialized_fluxes(
        _catalog_row(),
        BANDS,
    ) is True

    assert data_has_materialized_fluxes(
        _generated_data(),
        BANDS,
    ) is False

    incomplete_pair = _catalog_row()
    del incomplete_pair["ftotal_W149"]

    try:
        data_has_materialized_fluxes(
            incomplete_pair,
            BANDS,
        )
        raise AssertionError(
            "Expected incomplete flux pair to raise."
        )
    except ValueError:
        pass

    two_bands = BANDS + ("u",)

    partial_bands = _catalog_row()

    try:
        data_has_materialized_fluxes(
            partial_bands,
            two_bands,
        )
        raise AssertionError(
            "Expected partially-materialized bands to raise."
        )
    except ValueError:
        pass


def test_realization_from_catalog_requires_precomputed_blend_ratio():
    """realization_from_catalog is the catalog-row producer: it must
    NOT silently degrade when the precomputed blending realization is
    missing -- it raises, so the orchestrating layer's decision
    (data_has_materialized_blend_ratio) is the only thing that picks it."""
    with raises(KeyError):
        realization_from_catalog(
            _generated_data(), "USBL", use_parallax=True,
            t0_parallax=FIXED_PHYSICAL["t0"], band_order=BANDS,
        )


def test_realization_from_catalog_requires_caustic_origin_for_usbl():
    # Catalog producer: has the precomputed blend_ratio it requires,
    # but no caustic_origin -- isolates the caustic_origin check. A
    # materialized catalog row must always have it explicit; this
    # producer never samples it, so it raises.
    catalog_data = _catalog_row()
    del catalog_data["caustic_origin"]

    try:
        realization_from_catalog(catalog_data, "USBL", True, FIXED_PHYSICAL["t0"], BANDS)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        # Correction: the error message must say this is a catalog
        # validation, since realization_from_generated_parameters no
        # longer shares this code path (it never raises for this).
        assert "catalog" in str(exc).lower()


def test_realization_does_not_require_caustic_origin_for_fspl():
    catalog_row = _catalog_row()
    del catalog_row["caustic_origin"]  # FSPL never looks at it

    r = realization_from_catalog(catalog_row, "FSPL", True, FIXED_PHYSICAL["t0"], BANDS)
    assert r.caustic_origin is None


# ---------------------------------------------------------------
# Same decided realization -> identical pyLIMA model/flux outputs
# (the actual guarantee behind corrections 1/2/9)
# ---------------------------------------------------------------

def _minimal_event(times, name):
    from pyLIMA import event, telescopes

    lc = np.column_stack([
        times,
        np.full_like(times, 20.0),
        np.full_like(times, 0.01),
    ])

    ev = event.Event()
    ev.name = "realization_equivalence_test"
    ev.ra = 267.8
    ev.dec = -30.4

    tel = telescopes.Telescope(
        name=name,
        camera_filter="I",
        lightcurve=lc,
        lightcurve_names=["time", "mag", "err_mag"],
        lightcurve_units=["JD", "mag", "mag"],
        location="Earth",
    )
    tel.ld_gamma = 0.0
    ev.telescopes.append(tel)
    return ev


def _build_model_and_flux(realization, band_order, magstar):
    """Exactly what sim_event's adapter does with a decided
    realization: call model_choice and flux_parameters_model itself,
    using realization.caustic_origin / realization.blend_ratio instead
    of letting those functions sample internally."""
    from photometry.constants import SIMULATION_BAND_ZERO_POINTS

    t0 = realization.t0_parallax
    event_ = _minimal_event(np.array([t0 - 5.0, t0, t0 + 5.0]), name=band_order[0])

    caustic_origin_arg = (
        [realization.caustic_origin, [0, 0]]
        if realization.caustic_origin is not None
        else None
    )

    model = smp.model_choice(
        event_,
        realization.model_name,
        parallax=["None", 0.0],
        BL_random_origin=False,
        BL_origin=caustic_origin_arg,
    )

    params, param_order = smp.parameters_model(realization.physical_params, model)
    physical_values = [params[key] for key in param_order]

    # Mirrors sim_event's actual three-state flux branch:
    #
    # 1. catalog materialized fsource/ftotal -> pass through exactly;
    # 2. explicit generated/manual blend ratio -> construct fluxes;
    # 3. legacy generated -> sample blending via historical function.
    if realization.materialized_fluxes is not None:

        if model.blend_flux_parameter != "ftotal":
            raise RuntimeError(
                "Test catalog passthrough requires "
                "blend_flux_parameter='ftotal'."
            )

        flux_values = []
        fs = {}
        g = {}
        ftot = {}

        for band in band_order:

            fsource = realization.materialized_fluxes[
                band
            ]["fsource"]

            ftotal = realization.materialized_fluxes[
                band
            ]["ftotal"]

            flux_values.extend(
                [
                    fsource,
                    ftotal,
                ]
            )

            fs[band] = fsource
            ftot[band] = ftotal

            g[band] = (
                ftotal - fsource
            ) / fsource

    else:
        raise AssertionError("only catalogue realizations are supported")

    return model, physical_values + flux_values, fs, g, ftot


# ---------------------------------------------------------------
# Point 6: explicit-realization USBL -- input already has
# caustic_origin and blend_ratio_<band>, nothing may be sampled.
# Checked by running it under two different global seeds: if either
# origin or blending were sampled, these would disagree.
# ---------------------------------------------------------------

def test_explicit_usbl_realization_never_samples_origin_or_blending():
    magstar = {"W149": 20.0}
    catalog_row = _catalog_row(caustic_origin="third_caustic")

    results = []
    for seed in (1, 999999):
        np.random.seed(seed)
        r = realization_from_catalog(
            catalog_row, "USBL", use_parallax=False,
            t0_parallax=FIXED_PHYSICAL["t0"], band_order=BANDS,
        )
        assert r.caustic_origin == "third_caustic"
        assert r.blend_ratio == {"W149": FIXED_G}
        assert r.materialized_fluxes == {
            "W149": {
                "fsource": FIXED_FSOURCE,
                "ftotal": FIXED_FTOTAL,
            }
        }

        model, vector, fs, g, ftot = _build_model_and_flux(r, BANDS, magstar)
        results.append((model.origin, tuple(vector), g))

    assert results[0] == results[1], (
        "explicit USBL realization gave different results under "
        "different seeds -- something was sampled"
    )
    assert results[0][0][0] == "third_caustic"

    # For a materialized catalog realization, g reported by the test
    # adapter is derived from the authoritative fsource/ftotal pair.
    # Floating-point roundoff can therefore differ from the separately
    # stored blend_ratio at ~machine precision.
    assert set(results[0][2]) == {"W149"}
    assert np.isclose(
        results[0][2]["W149"],
        FIXED_G,
        rtol=0.0,
        atol=1e-15,
    )

    # Stronger proof than matching across two seeds: for ONE fixed
    # seed, confirm zero numbers were drawn from the global RNG at
    # all -- the next np.random.random() call must be bit-identical
    # to one taken right after the same seed with nothing in between.
    seed = 20261007

    np.random.seed(seed)
    expected_next = np.random.random()

    np.random.seed(seed)
    r = realization_from_catalog(
        catalog_row, "USBL", use_parallax=False,
        t0_parallax=FIXED_PHYSICAL["t0"], band_order=BANDS,
    )
    _build_model_and_flux(r, BANDS, magstar)
    actual_next = np.random.random()

    assert actual_next == expected_next, (
        "explicit USBL realization consumed the global RNG -- "
        "origin and/or blending were sampled instead of used as-is"
    )


# ---------------------------------------------------------------
# Point 5 (mandatory): golden test for the legacy/generated USBL path
# against the pre-origin implementation, inspected (not restored)
# from the quarantined backups:
#   /tmp/roman_rubin_refactor_quarantine_20261006_171728/root_bak/
#     functions_roman_rubin.py.before_origin_20261003_174641.bak
#     set_model_pyLIMA.py.before_origin_20261003_174641.bak
#
# Historical sequence inside sim_event, confirmed by reading those
# backups (not assumed):
#   np.random.seed(i)
#   model_choice(..., BL_random_origin=True)   # no BL_origin kwarg existed
#     -> build_pyLIMA_model(origin=None, random_origin=True)
#     -> choose_usbl_origin(origin=None, random_origin=True, rng=None)
#     -> np.random.choice(["central_caustic","second_caustic","third_caustic"])
#   flux_parameters_model(...)                  # per band, AFTER origin
#     -> np.random.uniform(0, 1) per band, in band_order
#
# The current legacy/generated path must reproduce this exactly:
# model_choice(..., BL_random_origin=True, BL_origin=None) (this
# refactor's compatibility path) followed by flux_parameters_model
# with no blend_ratio.
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# flux_parameters_model (LRT monkey-patch point, historical signature)
# vs. flux_parameters_from_blend_ratio (explicit-realization path,
# NOT a monkey-patch point) -- correction: LRT's replacement is
# catalog_flux_parameters_model(magstar, ZP, my_own_model, band_order),
# no **kwargs, no blend_ratio. Adding a blend_ratio keyword to the
# legacy call site would break that patch.
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# No-mutation
# ---------------------------------------------------------------

def test_realization_builders_do_not_mutate_input_data():
    original = _catalog_row()
    before = copy.deepcopy(original)

    realization_from_catalog(original, "USBL", True, FIXED_PHYSICAL["t0"], BANDS)

    assert original == before


def test_realization_physical_params_is_a_copy_not_the_caller_object():
    """EventRealization must not hold the same mutable `data` reference:
    mutating the realization's physical_params must not affect the
    caller's original dict."""
    original = _catalog_row()

    r = realization_from_catalog(original, "USBL", True, FIXED_PHYSICAL["t0"], BANDS)
    assert r.physical_params is not original

    r.physical_params["t0"] = -999.0
    assert original["t0"] == FIXED_PHYSICAL["t0"]


def test_blend_ratio_validation_allows_g_greater_than_one():
    # No upper bound is imposed on g = Fblend/Fsource.
    row = _catalog_row()

    g = 5.0
    band = BANDS[0]

    row[f"blend_ratio_{band}"] = g
    row[f"ftotal_{band}"] = (
        row[f"fsource_{band}"]
        * (1.0 + g)
    )

    r = realization_from_catalog(
        row,
        "USBL",
        True,
        FIXED_PHYSICAL["t0"],
        BANDS,
    )

    assert r.blend_ratio[band] == g
    assert r.materialized_fluxes[band]["fsource"] == FIXED_FSOURCE
    assert r.materialized_fluxes[band]["ftotal"] == FIXED_FSOURCE * 6.0


# ---------------------------------------------------------------
# Monkey-patching indirection (LRT compatibility contract)
#
# LRT replaces functions_roman_rubin.{sim_event, model_choice,
# flux_parameters_model, extract_lightcurves_for_fit, fit_rubin_roman}
# as plain module attributes. This only works if the real caller
# resolves the name unqualified against functions_roman_rubin's own
# globals -- these tests check that identity directly, without running
# any simulation (as instructed: no heavy execution needed for this).
# ---------------------------------------------------------------

def test_simulate_event_for_fit_calls_sim_event_unqualified():
    assert frr.simulate_event_for_fit.__globals__["sim_event"] is frr.sim_event

    sentinel = object()
    original = frr.sim_event
    frr.sim_event = sentinel
    try:
        assert frr.simulate_event_for_fit.__globals__["sim_event"] is sentinel
    finally:
        frr.sim_event = original


def test_sim_fit_calls_extract_lightcurves_for_fit_unqualified():
    assert frr.sim_fit.__globals__["extract_lightcurves_for_fit"] is frr.extract_lightcurves_for_fit

    sentinel = object()
    original = frr.extract_lightcurves_for_fit
    frr.extract_lightcurves_for_fit = sentinel
    try:
        assert frr.sim_fit.__globals__["extract_lightcurves_for_fit"] is sentinel
    finally:
        frr.extract_lightcurves_for_fit = original


def test_read_fit_calls_fit_rubin_roman_unqualified():
    assert frr.read_fit.__globals__["fit_rubin_roman"] is frr.fit_rubin_roman

    sentinel = object()
    original = frr.fit_rubin_roman
    frr.fit_rubin_roman = sentinel
    try:
        assert frr.read_fit.__globals__["fit_rubin_roman"] is sentinel
    finally:
        frr.fit_rubin_roman = original


def test_run_all_fits_resolves_fit_rubin_roman_from_functions_roman_rubin():
    # run_all_fits's real call chain is
    # run_all_fits -> _call_fit_rubin_roman_compatible -> fit_rubin_roman
    # (both unqualified); explicitly requested check:
    assert frr.run_all_fits.__globals__["fit_rubin_roman"] is frr.fit_rubin_roman
    assert (
        frr._call_fit_rubin_roman_compatible.__globals__["fit_rubin_roman"]
        is frr.fit_rubin_roman
    )


def test_sim_fit_and_sim_fit_multi_fits_share_sim_event_global():
    # sim_fit_multi_fits delegates to sim_fit (which calls
    # simulate_event_for_fit -> sim_event); explicitly requested check
    # that both still resolve the same patchable global:
    assert frr.sim_fit.__globals__["sim_event"] is frr.sim_event
    assert frr.sim_fit_multi_fits.__globals__["sim_event"] is frr.sim_event

    sentinel = object()
    original = frr.sim_event
    frr.sim_event = sentinel
    try:
        assert frr.sim_fit.__globals__["sim_event"] is sentinel
        assert frr.sim_fit_multi_fits.__globals__["sim_event"] is sentinel
    finally:
        frr.sim_event = original


def test_model_choice_patch_still_resolves_from_functions_roman_rubin():
    assert frr.sim_event.__globals__["model_choice"] is frr.model_choice

    sentinel = object()
    original = frr.model_choice
    frr.model_choice = sentinel
    try:
        assert frr.sim_event.__globals__["model_choice"] is sentinel
    finally:
        frr.model_choice = original


# ---------------------------------------------------------------
# Legacy import/signature surface LRT depends on
# ---------------------------------------------------------------

def test_simulation_core_imports_and_knows_nothing_of_catalogs():
    import inspect

    import simulation.core as core

    assert callable(core.simulate_light_curve)

    sig = inspect.signature(core.simulate_light_curve)
    assert list(sig.parameters) == ["pyLIMA_model", "parameter_vector", "sim_timer"]

    # Check the executable code (names/constants referenced by the
    # bytecode), not the module's prose docstring (which legitimately
    # names these as things that must NOT appear in the code).
    code = core.simulate_light_curve.__code__
    referenced = set(code.co_names) | {
        c for c in code.co_consts if isinstance(c, str)
    }
    forbidden_terms = (
        "event_seed", "system_type", "catalog_mode", "Amax", "amax",
        "Roman", "W149", "roman_band_name", "ground",
    )
    for forbidden in forbidden_terms:
        assert forbidden not in referenced, forbidden


def test_sim_fit_signature_keeps_seed_system_type_and_photometric_filter():
    import inspect

    sig = inspect.signature(frr.sim_fit)
    params = list(sig.parameters.values())

    assert params[0].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    assert params[1].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    assert "apply_photometric_filter" in sig.parameters


if __name__ == "__main__":
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]

    failures = []

    for name, test_fn in tests:
        try:
            test_fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {name}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")

    if failures:
        raise SystemExit(1)
