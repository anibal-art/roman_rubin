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
    realization_from_generated_parameters,
    data_has_materialized_blend_ratio,
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


def _catalog_row(caustic_origin="central_caustic"):
    row = dict(FIXED_PHYSICAL)
    row["caustic_origin"] = caustic_origin
    for band in BANDS:
        row[f"blend_ratio_{band}"] = FIXED_G
    return row


def _generated_data(caustic_origin="central_caustic"):
    row = dict(FIXED_PHYSICAL)
    row["caustic_origin"] = caustic_origin
    return row


def test_realization_from_catalog_and_generated_agree_on_shared_fields():
    r_catalog = realization_from_catalog(
        _catalog_row(), "USBL", use_parallax=True,
        t0_parallax=FIXED_PHYSICAL["t0"], band_order=BANDS,
    )
    r_generated = realization_from_generated_parameters(
        _generated_data(), "USBL", use_parallax=True,
        t0_parallax=FIXED_PHYSICAL["t0"], band_order=BANDS,
        blend_ratio={b: FIXED_G for b in BANDS},
    )

    assert r_catalog.model_name == r_generated.model_name
    assert r_catalog.use_parallax == r_generated.use_parallax
    assert r_catalog.t0_parallax == r_generated.t0_parallax
    assert r_catalog.caustic_origin == r_generated.caustic_origin == "central_caustic"
    assert r_catalog.blend_ratio == r_generated.blend_ratio == {"W149": FIXED_G}

    for key, value in FIXED_PHYSICAL.items():
        assert r_catalog.physical_params[key] == value
        assert r_generated.physical_params[key] == value


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


def test_realization_from_generated_parameters_defaults_to_sampled_blending():
    """The on-the-fly/manual producer: blend_ratio stays None (sampled
    later by flux_parameters_model, unchanged default behavior) unless
    explicitly overridden."""
    r = realization_from_generated_parameters(
        _generated_data(), "USBL", use_parallax=True,
        t0_parallax=FIXED_PHYSICAL["t0"], band_order=BANDS,
    )
    assert r.blend_ratio is None


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


def test_realization_from_generated_parameters_defers_missing_caustic_origin():
    """The legacy/on-the-fly producer must NOT raise when `data` has
    no explicit caustic_origin for USBL (this was the real bug the
    smoke test found: catalog_mode='custom_system' data commonly has
    neither materialized blending nor an explicit caustic_origin).
    It must return caustic_origin=None, meaning "not yet decided" --
    the caller then falls back to the historical random-origin
    mechanism; this function itself never samples anything."""
    generated_data = dict(FIXED_PHYSICAL)  # no caustic_origin key

    r = realization_from_generated_parameters(
        generated_data, "USBL", True, FIXED_PHYSICAL["t0"], BANDS,
    )
    assert r.caustic_origin is None


def test_realization_from_generated_parameters_uses_explicit_caustic_origin_when_present():
    """If generated/manual data DOES carry an explicit caustic_origin,
    it is used (validated), not sampled."""
    generated_data = _generated_data(caustic_origin="second_caustic")

    r = realization_from_generated_parameters(
        generated_data, "USBL", True, FIXED_PHYSICAL["t0"], BANDS,
    )
    assert r.caustic_origin == "second_caustic"


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


def _minimal_multiband_event(times, names):
    from pyLIMA import event, telescopes

    lc = np.column_stack([
        times,
        np.full_like(times, 20.0),
        np.full_like(times, 0.01),
    ])

    ev = event.Event()
    ev.name = "golden_usbl_legacy_test"
    ev.ra = 267.8
    ev.dec = -30.4

    for name in names:
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

    # Mirrors sim_event's actual branch: explicit path (blend_ratio
    # decided) uses the non-monkey-patched flux_parameters_from_blend_ratio;
    # legacy path uses flux_parameters_model with its historical,
    # patchable signature -- never a blend_ratio keyword.
    if realization.blend_ratio is not None:
        flux_values, fs, g, ftot = smp.flux_parameters_from_blend_ratio(
            magstar, SIMULATION_BAND_ZERO_POINTS, model,
            band_order, realization.blend_ratio,
        )
    else:
        flux_values, fs, g, ftot = smp.flux_parameters_model(
            magstar, SIMULATION_BAND_ZERO_POINTS, model,
            band_order=band_order,
        )

    return model, physical_values + flux_values, fs, g, ftot


def test_catalog_and_generated_realizations_build_identical_model_and_flux():
    magstar = {"W149": 20.0}

    r_catalog = realization_from_catalog(
        _catalog_row(), "USBL", use_parallax=False,
        t0_parallax=FIXED_PHYSICAL["t0"], band_order=BANDS,
    )
    r_generated = realization_from_generated_parameters(
        _generated_data(), "USBL", use_parallax=False,
        t0_parallax=FIXED_PHYSICAL["t0"], band_order=BANDS,
        blend_ratio={"W149": FIXED_G},
    )

    model_a, vector_a, fs_a, g_a, ft_a = _build_model_and_flux(r_catalog, BANDS, magstar)
    model_b, vector_b, fs_b, g_b, ft_b = _build_model_and_flux(r_generated, BANDS, magstar)

    assert model_a.origin == model_b.origin
    assert np.array_equal(np.asarray(vector_a, dtype=float), np.asarray(vector_b, dtype=float))
    assert fs_a == fs_b
    assert g_a == g_b == {"W149": FIXED_G}
    assert ft_a == ft_b


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

        model, vector, fs, g, ftot = _build_model_and_flux(r, BANDS, magstar)
        results.append((model.origin, tuple(vector), g))

    assert results[0] == results[1], (
        "explicit USBL realization gave different results under "
        "different seeds -- something was sampled"
    )
    assert results[0][0][0] == "third_caustic"
    assert results[0][2] == {"W149": FIXED_G}

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

def test_golden_usbl_legacy_random_origin_matches_pre_origin_backup():
    seed = 20261007
    band_order = ["W149", "u"]
    magstar = {"W149": 20.0, "u": 21.0}
    zp = {"W149": 27.615, "u": 27.03}
    data = dict(FIXED_PHYSICAL)  # no caustic_origin key

    # Expected: the historical algorithm, verbatim, same seed, same
    # RNG call sequence (origin, THEN blending per band in order), and
    # the explicit parameter vector built by hand from that sequence
    # (not inferred from origin/g alone -- the formula itself, same as
    # pre-origin flux_parameters_model: f_source = baseline/(1+g),
    # f_total = f_source*(1+g), appended in that order per band since
    # blend_flux_parameter="ftotal").
    np.random.seed(seed)
    expected_origin = str(
        np.random.choice(["central_caustic", "second_caustic", "third_caustic"])
    )
    expected_g = {band: np.random.uniform(0, 1) for band in band_order}
    expected_next_random = np.random.random()

    expected_physical_order = smp.physical_parameter_order("USBL", use_parallax=False)
    expected_physical_values = [data[key] for key in expected_physical_order]

    expected_flux_values = []
    for band in band_order:
        flux_baseline = 10 ** ((zp[band] - magstar[band]) / 2.5)
        g = expected_g[band]
        f_source = flux_baseline / (1 + g)
        f_total = f_source + g * f_source
        expected_flux_values += [f_source, f_total]

    expected_vector = expected_physical_values + expected_flux_values

    # Actual: current code, same seed, same call sequence sim_event
    # uses for the legacy/generated USBL path (realization.caustic_origin
    # is None -> BL_random_origin=True, BL_origin=None; blend_ratio is
    # None -> flux_parameters_model, no override).
    np.random.seed(seed)

    realization = realization_from_generated_parameters(
        data, "USBL", use_parallax=False,
        t0_parallax=FIXED_PHYSICAL["t0"], band_order=band_order,
    )
    assert realization.caustic_origin is None  # "not yet decided"
    assert realization.blend_ratio is None

    event_ = _minimal_multiband_event(
        np.array([2460495.0, 2460500.0, 2460505.0]), band_order,
    )

    model = smp.model_choice(
        event_, "USBL",
        parallax=["None", 0.0],
        BL_random_origin=True,
        BL_origin=None,
    )
    actual_origin = str(model.origin[0])

    params, param_order = smp.parameters_model(realization.physical_params, model)
    actual_physical_values = [params[key] for key in param_order]

    actual_flux_values, fs, actual_g, ftot = smp.flux_parameters_model(
        magstar, zp, model, band_order=band_order,
    )
    actual_next_random = np.random.random()

    actual_vector = actual_physical_values + actual_flux_values

    assert actual_origin == expected_origin
    assert actual_g == expected_g
    assert actual_physical_values == expected_physical_values
    assert actual_flux_values == expected_flux_values
    assert actual_vector == expected_vector

    # Freezes the RNG sequence through the point immediately after
    # blending (photometric noise is drawn after this): if moving the
    # origin draw, reordering it relative to blending, or drawing it
    # with a different mechanism ever changed how many numbers are
    # consumed from the global RNG, this would catch it even if origin
    # and g happened to still match by coincidence.
    assert actual_next_random == expected_next_random


# ---------------------------------------------------------------
# flux_parameters_model (LRT monkey-patch point, historical signature)
# vs. flux_parameters_from_blend_ratio (explicit-realization path,
# NOT a monkey-patch point) -- correction: LRT's replacement is
# catalog_flux_parameters_model(magstar, ZP, my_own_model, band_order),
# no **kwargs, no blend_ratio. Adding a blend_ratio keyword to the
# legacy call site would break that patch.
# ---------------------------------------------------------------

def test_flux_parameters_model_signature_has_no_blend_ratio_kwarg():
    import inspect

    sig = inspect.signature(smp.flux_parameters_model)
    assert list(sig.parameters) == ["magstar", "ZP", "pyLIMA_model", "band_order", "rng"]
    assert "blend_ratio" not in sig.parameters


def test_flux_parameters_from_blend_ratio_matches_shared_formula():
    """Direct unit test of the explicit-realization function: same
    flux formula as flux_parameters_model, with g fixed instead of
    sampled -- no duplicated math (both call
    set_model_pyLIMA._flux_parameters_for_bands)."""
    class _FakeModel:
        blend_flux_parameter = "ftotal"

    magstar = {"W149": 20.0}
    zp = {"W149": 27.615}
    g = 0.42

    flux_values, fs, g_out, ftot = smp.flux_parameters_from_blend_ratio(
        magstar, zp, _FakeModel(), ["W149"], {"W149": g},
    )

    flux_baseline = 10 ** ((zp["W149"] - magstar["W149"]) / 2.5)
    expected_fsource = flux_baseline / (1 + g)
    expected_ftotal = expected_fsource * (1 + g)

    assert g_out == {"W149": g}
    assert abs(fs["W149"] - expected_fsource) < 1e-12
    assert abs(ftot["W149"] - expected_ftotal) < 1e-12
    assert flux_values == [fs["W149"], ftot["W149"]]


def test_lrt_flux_parameters_model_patch_is_still_callable_via_legacy_path():
    """LRT replaces functions_roman_rubin.flux_parameters_model with
    catalog_flux_parameters_model(magstar, ZP, my_own_model, band_order)
    -- positional, no **kwargs, no blend_ratio. The legacy call site in
    sim_event must keep calling the exact historical signature, via the
    exact historical (patchable) global, or this patch breaks."""
    captured = {}

    def catalog_flux_parameters_model(magstar, ZP, my_own_model, band_order):
        captured["args"] = (magstar, ZP, my_own_model, band_order)
        return (["ok"], {}, {}, {})

    assert frr.sim_event.__globals__["flux_parameters_model"] is frr.flux_parameters_model

    original = frr.flux_parameters_model
    frr.flux_parameters_model = catalog_flux_parameters_model
    try:
        assert frr.sim_event.__globals__["flux_parameters_model"] is catalog_flux_parameters_model

        # Call it exactly the way sim_event's legacy call site does
        # (band_order as keyword; no other keyword).
        patched = frr.sim_event.__globals__["flux_parameters_model"]
        result = patched(
            {"W149": 20.0}, {"W149": 27.615}, object(), band_order=["W149"],
        )

        assert result == (["ok"], {}, {}, {})
        assert captured["args"][3] == ["W149"]
    finally:
        frr.flux_parameters_model = original


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


def test_blend_ratio_validation_rejects_negative_and_non_finite():
    for bad_g in (-0.1, float("nan"), float("inf")):
        bad_row = _catalog_row()
        bad_row[f"blend_ratio_{BANDS[0]}"] = bad_g

        with raises(ValueError):
            realization_from_catalog(bad_row, "USBL", True, FIXED_PHYSICAL["t0"], BANDS)

        with raises(ValueError):
            realization_from_generated_parameters(
                _generated_data(), "USBL", True, FIXED_PHYSICAL["t0"], BANDS,
                blend_ratio={BANDS[0]: bad_g},
            )


def test_blend_ratio_validation_allows_g_greater_than_one():
    # No upper bound is imposed on g = Fblend/Fsource.
    row = _catalog_row()
    row[f"blend_ratio_{BANDS[0]}"] = 5.0

    r = realization_from_catalog(row, "USBL", True, FIXED_PHYSICAL["t0"], BANDS)
    assert r.blend_ratio[BANDS[0]] == 5.0


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

def test_sim_event_calls_model_choice_and_flux_parameters_model_unqualified():
    assert frr.sim_event.__globals__["model_choice"] is frr.model_choice
    assert frr.sim_event.__globals__["flux_parameters_model"] is frr.flux_parameters_model

    sentinel = object()
    original = frr.model_choice
    frr.model_choice = sentinel
    try:
        assert frr.sim_event.__globals__["model_choice"] is sentinel
    finally:
        frr.model_choice = original


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


def test_lrt_facing_names_are_present_and_callable():
    for name in (
        "sim_event", "sim_fit", "sim_fit_multi_fits", "fit_rubin_roman",
        "model_choice", "flux_parameters_model", "extract_lightcurves_for_fit",
    ):
        assert callable(getattr(frr, name)), name

    assert callable(stp.tel_roman_rubin)
    assert callable(stp.configure_rubin_paths)
    assert callable(smp.build_pyLIMA_model)


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
