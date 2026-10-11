from types import SimpleNamespace

import functions_roman_rubin as frr


def test_true_photometric_chi2_for_telescope_uses_only_w149(
    monkeypatch,
):
    roman = SimpleNamespace(
        name="W149",
        lightcurve=[1, 2, 3, 4],
    )
    rubin = SimpleNamespace(
        name="r",
        lightcurve=[1, 2],
    )

    model = SimpleNamespace(
        event=SimpleNamespace(
            telescopes=[roman, rubin],
        )
    )

    params = object()
    captured = {}

    def fake_chi2(model_view, parameters):
        captured["names"] = [
            telescope.name
            for telescope in model_view.event.telescopes
        ]
        captured["parameters"] = parameters
        return 123.5

    from pyLIMA.fits import objective_functions

    monkeypatch.setattr(
        objective_functions,
        "all_telescope_photometric_chi2",
        fake_chi2,
    )

    result = frr.true_photometric_chi2_for_telescope(
        model,
        params,
        telescope_name="W149",
    )

    assert result == {
        "chi2_true": 123.5,
        "n_data_true": 4,
    }

    assert captured["names"] == ["W149"]
    assert captured["parameters"] is params

    # Original simulation model remains unchanged.
    assert model.event.telescopes == [roman, rubin]


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
