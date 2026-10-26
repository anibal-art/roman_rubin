"""LEGACY - not used by production. Kept for reference only.

Tests of the removed generation paths, formerly in
tests/test_regression_interfaces.py. Not collected by pytest; they use
helpers from that file and will not run as is.
"""


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

    assert r_catalog.materialized_fluxes == {
        "W149": {
            "fsource": FIXED_FSOURCE,
            "ftotal": FIXED_FTOTAL,
        }
    }

    assert r_generated.materialized_fluxes is None

    for key, value in FIXED_PHYSICAL.items():
        assert r_catalog.physical_params[key] == value
        assert r_generated.physical_params[key] == value


def test_explicit_blend_without_fluxes_remains_generated():

    data = _generated_data()

    for band in BANDS:
        data[f"blend_ratio_{band}"] = FIXED_G

    assert data_has_materialized_blend_ratio(
        data,
        BANDS,
    ) is True

    assert data_has_materialized_fluxes(
        data,
        BANDS,
    ) is False

    explicit_blend_ratio = {
        band: float(
            data[f"blend_ratio_{band}"]
        )
        for band in BANDS
    }

    r = realization_from_generated_parameters(
        data,
        "USBL",
        use_parallax=True,
        t0_parallax=FIXED_PHYSICAL["t0"],
        band_order=BANDS,
        blend_ratio=explicit_blend_ratio,
    )

    assert r.blend_ratio == {
        "W149": FIXED_G,
    }

    assert r.materialized_fluxes is None


def test_realization_from_generated_parameters_defaults_to_sampled_blending():
    """The on-the-fly/manual producer: blend_ratio stays None (sampled
    later by flux_parameters_model, unchanged default behavior) unless
    explicitly overridden."""
    r = realization_from_generated_parameters(
        _generated_data(), "USBL", use_parallax=True,
        t0_parallax=FIXED_PHYSICAL["t0"], band_order=BANDS,
    )
    assert r.blend_ratio is None


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


def test_catalog_and_generated_realizations_have_distinct_flux_authority():
    """
    Catalog and generated realizations may share the same physical
    parameters and blend ratio, but their flux authority is different.

    Catalog:
        fsource/ftotal are already materialized inputs.

    Generated/manual:
        fluxes are constructed later from its own magnitude/blending
        mechanism.
    """

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

    # Same event physics/origin/blending realization.
    assert r_catalog.caustic_origin == r_generated.caustic_origin
    assert r_catalog.blend_ratio == r_generated.blend_ratio

    for key in FIXED_PHYSICAL:
        assert (
            r_catalog.physical_params[key]
            == r_generated.physical_params[key]
        )

    # But only the catalog owns an already-materialized flux
    # realization.
    assert r_catalog.materialized_fluxes == {
        "W149": {
            "fsource": FIXED_FSOURCE,
            "ftotal": FIXED_FTOTAL,
        }
    }

    assert r_generated.materialized_fluxes is None

    # The generated/manual path remains independently functional and
    # keeps its historical explicit-blend-ratio construction.
    model_generated, _, fs_generated, g_generated, ft_generated = (
        _build_model_and_flux(
            r_generated,
            BANDS,
            magstar,
        )
    )

    assert model_generated.origin[0] == "central_caustic"
    assert g_generated == {"W149": FIXED_G}

    # Deliberately prove that the catalog flux realization is NOT
    # required to equal a reconstruction from this unrelated magstar.
    assert fs_generated["W149"] != FIXED_FSOURCE
    assert ft_generated["W149"] != FIXED_FTOTAL


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


def test_lrt_facing_names_are_present_and_callable():
    for name in (
        "sim_event", "sim_fit", "sim_fit_multi_fits", "fit_rubin_roman",
        "model_choice", "flux_parameters_model", "extract_lightcurves_for_fit",
    ):
        assert callable(getattr(frr, name)), name

    assert callable(stp.tel_roman_rubin)
    assert callable(stp.configure_rubin_paths)
    assert callable(smp.build_pyLIMA_model)


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
