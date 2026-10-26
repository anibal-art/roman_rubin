import pytest

import fit_lc
import fitting.fit_parameters
from fit_lc import (
    apply_narrow_bounds,
    build_fit_pyLIMA_model,
    fit_guess_values,
    fit_parameter_order,
    fit_trajectory_parameter_names,
    initial_params_for_fit_model,
)


EVENT_PARAMS = {
    "t_center": 100.0,
    "u_center": 0.01,
    "t0": 172.5,
    "u0": -3.4,
    "tE": 40.0,
    "rho": 1.0e-3,
    "separation": 1.2,
    "mass_ratio": 2.0e-4,
    "alpha": 0.7,
    "piEN": 0.1,
    "piEE": -0.2,
}


class FakeFit:
    def __init__(self, names):
        self.fit_parameters = {
            name: [None, [-999.0, 999.0]]
            for name in names
        }


@pytest.mark.parametrize(
    "origin_name",
    [
        "central_caustic",
        "second_caustic",
        "third_caustic",
    ],
)
def test_usbl_caustic_origin_uses_simulated_t_center_and_u_center(
    origin_name,
):
    origin = [origin_name, [0, 0]]

    assert fit_trajectory_parameter_names(
        "USBL",
        Origin=origin,
    ) == ("t_center", "u_center")

    fit_params = initial_params_for_fit_model(
        EVENT_PARAMS,
        "USBL",
        fit_parallax=True,
        Origin=origin,
    )

    assert fit_params["t0"] == EVENT_PARAMS["t_center"]
    assert fit_params["u0"] == EVENT_PARAMS["u_center"]

    order = fit_parameter_order(
        "USBL",
        fit_parallax=True,
        Origin=origin,
    )

    assert order[:2] == ["t_center", "u_center"]

    guess = fit_guess_values(
        fit_params,
        "USBL",
        fit_parallax=True,
        Origin=origin,
    )

    assert guess[0] == EVENT_PARAMS["t_center"]
    assert guess[1] == EVENT_PARAMS["u_center"]


def test_usbl_center_of_mass_uses_t0_and_u0():
    origin = ["center_of_mass", [0, 0]]

    assert fit_trajectory_parameter_names(
        "USBL",
        Origin=origin,
    ) == ("t0", "u0")

    fit_params = initial_params_for_fit_model(
        EVENT_PARAMS,
        "USBL",
        fit_parallax=True,
        Origin=origin,
    )

    assert fit_params["t0"] == EVENT_PARAMS["t0"]
    assert fit_params["u0"] == EVENT_PARAMS["u0"]

    order = fit_parameter_order(
        "USBL",
        fit_parallax=True,
        Origin=origin,
    )

    assert order[:2] == ["t0", "u0"]

    guess = fit_guess_values(
        fit_params,
        "USBL",
        fit_parallax=True,
        Origin=origin,
    )

    assert guess[0] == EVENT_PARAMS["t0"]
    assert guess[1] == EVENT_PARAMS["u0"]


@pytest.mark.parametrize(
    "origin, expected_time, expected_impact, parameter_names",
    [
        (
            ["central_caustic", [0, 0]],
            EVENT_PARAMS["t_center"],
            EVENT_PARAMS["u_center"],
            ["t_center", "u_center"],
        ),
        (
            ["center_of_mass", [0, 0]],
            EVENT_PARAMS["t0"],
            EVENT_PARAMS["u0"],
            ["t0", "u0"],
        ),
    ],
)
def test_usbl_narrow_bounds_are_centered_on_active_trajectory_coordinates(
    origin,
    expected_time,
    expected_impact,
    parameter_names,
):
    fit_params = initial_params_for_fit_model(
        EVENT_PARAMS,
        "USBL",
        fit_parallax=True,
        Origin=origin,
    )

    names = fit_parameter_order(
        "USBL",
        fit_parallax=True,
        Origin=origin,
    )

    fit = FakeFit(names)

    apply_narrow_bounds(
        fit,
        fit_params,
        "USBL",
        fit_parallax=True,
        rango=1,
        Origin=origin,
    )

    time_name, impact_name = parameter_names

    assert fit.fit_parameters[time_name][1] == pytest.approx(
        [expected_time - 10.0, expected_time + 10.0]
    )

    impact_bounds = fit.fit_parameters[impact_name][1]

    assert impact_bounds[0] <= expected_impact <= impact_bounds[1]


@pytest.mark.parametrize(
    "origin, expected_tref",
    [
        (
            ["central_caustic", [0, 0]],
            EVENT_PARAMS["t_center"],
        ),
        (
            ["center_of_mass", [0, 0]],
            EVENT_PARAMS["t0"],
        ),
    ],
)
def test_usbl_parallax_reference_epoch_uses_active_trajectory_time(
    monkeypatch,
    origin,
    expected_tref,
):
    fit_params = initial_params_for_fit_model(
        EVENT_PARAMS,
        "USBL",
        fit_parallax=True,
        Origin=origin,
    )

    captured = {}

    def fake_build_pyLIMA_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        fitting.fit_parameters,
        "build_pyLIMA_model",
        fake_build_pyLIMA_model,
    )

    result = build_fit_pyLIMA_model(
        e=object(),
        fit_model="USBL",
        fit_params=fit_params,
        Origin=origin,
        fit_parallax=True,
    )

    assert result is not None
    assert captured["t0_parallax"] == expected_tref
    assert captured["origin"] == origin


def test_non_usbl_models_keep_t0_u0_parameterization():
    assert fit_trajectory_parameter_names(
        "FSPL",
        Origin=["third_caustic", [0, 0]],
    ) == ("t0", "u0")

    assert fit_trajectory_parameter_names(
        "PSPL",
        Origin=["third_caustic", [0, 0]],
    ) == ("t0", "u0")
