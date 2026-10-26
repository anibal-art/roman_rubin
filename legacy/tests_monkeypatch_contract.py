"""Tests of the removed monkey-patch contract of functions_roman_rubin.

They checked that replacing functions_roman_rubin.<name> at runtime changed
what sim_event / sim_fit / run_all_fits called. The orchestrators moved to
their own modules (step 6.1) and the contract was dropped. Not imported.
"""


# from tests/test_regression_interfaces.py
def test_simulate_event_for_fit_calls_sim_event_unqualified():
    assert frr.simulate_event_for_fit.__globals__["sim_event"] is frr.sim_event

    sentinel = object()
    original = frr.sim_event
    frr.sim_event = sentinel
    try:
        assert frr.simulate_event_for_fit.__globals__["sim_event"] is sentinel
    finally:
        frr.sim_event = original


# from tests/test_regression_interfaces.py
def test_sim_fit_calls_extract_lightcurves_for_fit_unqualified():
    assert frr.sim_fit.__globals__["extract_lightcurves_for_fit"] is frr.extract_lightcurves_for_fit

    sentinel = object()
    original = frr.extract_lightcurves_for_fit
    frr.extract_lightcurves_for_fit = sentinel
    try:
        assert frr.sim_fit.__globals__["extract_lightcurves_for_fit"] is sentinel
    finally:
        frr.extract_lightcurves_for_fit = original


# from tests/test_regression_interfaces.py
def test_run_all_fits_resolves_fit_rubin_roman_from_functions_roman_rubin():
    # run_all_fits's real call chain is
    # run_all_fits -> _call_fit_rubin_roman_compatible -> fit_rubin_roman
    # (both unqualified); explicitly requested check:
    assert frr.run_all_fits.__globals__["fit_rubin_roman"] is frr.fit_rubin_roman
    assert (
        frr._call_fit_rubin_roman_compatible.__globals__["fit_rubin_roman"]
        is frr.fit_rubin_roman
    )


# from tests/test_regression_interfaces.py
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


# from tests/test_regression_interfaces.py
def test_model_choice_patch_still_resolves_from_functions_roman_rubin():
    assert frr.sim_event.__globals__["model_choice"] is frr.model_choice

    sentinel = object()
    original = frr.model_choice
    frr.model_choice = sentinel
    try:
        assert frr.sim_event.__globals__["model_choice"] is sentinel
    finally:
        frr.model_choice = original


# from tests/test_roman_true_chi2.py
def test_run_all_fits_computes_roman_true_chi2_before_roman_fit(
    monkeypatch,
):
    sequence = []
    call_count = {"n": 0}

    class FakeFit:
        def __init__(self):
            self.fit_results = {}

    def fake_fit_call(*args, **kwargs):
        call_count["n"] += 1

        if call_count["n"] == 1:
            sequence.append("fit_rr")
        else:
            sequence.append("fit_roman")

        return FakeFit(), object(), object()

    def fake_roman_true_chi2(
        true_model,
        pyLIMA_parameters,
        telescope_name,
    ):
        sequence.append("chi2_true_roman")

        assert telescope_name == "W149"

        return {
            "chi2_true": 50.0,
            "n_data_true": 3,
        }

    monkeypatch.setattr(
        frr,
        "_call_fit_rubin_roman_compatible",
        fake_fit_call,
    )

    monkeypatch.setattr(
        frr,
        "true_photometric_chi2_for_telescope",
        fake_roman_true_chi2,
    )

    lc_to_fit = {
        "W149": [1, 2, 3],
        "u": [],
        "g": [],
        "r": [],
        "i": [],
        "z": [],
        "y": [],
    }

    (
        fit_rr,
        event_fit_rr,
        model_rr,
        fit_roman,
        event_fit_roman,
        model_roman,
    ) = frr.run_all_fits(
        Source=616,
        pyLIMA_parameters=object(),
        path_to_save_fit="/tmp",
        path_ephemerides="/tmp",
        model="USBL",
        algo="TRF",
        origin=["central_caustic", [0, 0]],
        rango=1,
        lc_to_fit=lc_to_fit,
        use_roman=True,
        use_rubin=False,
        fit_model="USBL",
        fit_parallax=True,
        true_model=object(),
    )

    assert sequence == [
        "fit_rr",
        "chi2_true_roman",
        "fit_roman",
    ]

    assert fit_roman.fit_results["chi2_true_roman"] == 50.0
    assert fit_roman.fit_results["n_data_true_roman"] == 3
