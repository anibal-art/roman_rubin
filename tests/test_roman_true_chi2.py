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

