import copy

import numpy as np
from astropy.table import QTable

from functions_roman_rubin import (
    roman_detection_chi2_statistics,
)


class _FakeTelescope:
    def __init__(
        self,
        name,
        flux,
        err_flux,
    ):
        self.name = name

        self.lightcurve = QTable(
            [
                np.asarray(flux, dtype=float),
                np.asarray(err_flux, dtype=float),
            ],
            names=[
                "flux",
                "err_flux",
            ],
        )


class _FakeEvent:
    def __init__(self, telescopes):
        self.telescopes = list(telescopes)


class _FakeModel:
    def __init__(
        self,
        telescopes,
        true_flux_by_telescope,
    ):
        self.event = _FakeEvent(telescopes)

        self.true_flux_by_telescope = {
            str(key): np.asarray(
                value,
                dtype=float,
            )
            for key, value
            in true_flux_by_telescope.items()
        }

        self.compute_calls = []

    def compute_the_microlensing_model(
        self,
        telescope,
        pyLIMA_parameters,
    ):
        self.compute_calls.append(
            str(telescope.name)
        )

        return {
            "photometry":
                self.true_flux_by_telescope[
                    str(telescope.name)
                ].copy()
        }


def test_roman_detection_chi2_matches_explicit_formula():

    roman = _FakeTelescope(
        "W149",
        flux=[10.0, 13.0, 12.0],
        err_flux=[1.0, 2.0, 1.0],
    )

    model = _FakeModel(
        [roman],
        {
            "W149": [10.0, 11.0, 12.0],
        },
    )

    params = {
        "ftotal_W149": 9.0,
    }

    result = roman_detection_chi2_statistics(
        model,
        params,
    )

    expected_chi2_true = (
        ((10.0 - 10.0) / 1.0) ** 2
        + ((13.0 - 11.0) / 2.0) ** 2
        + ((12.0 - 12.0) / 1.0) ** 2
    )

    expected_chi2_flat = (
        ((10.0 - 9.0) / 1.0) ** 2
        + ((13.0 - 9.0) / 2.0) ** 2
        + ((12.0 - 9.0) / 1.0) ** 2
    )

    expected_delta = (
        expected_chi2_flat
        - expected_chi2_true
    )

    assert result["n_data"] == 3
    assert result["baseline_flux"] == 9.0
    assert result["chi2_true"] == expected_chi2_true
    assert result["chi2_flat"] == expected_chi2_flat
    assert result["delta_chi2"] == expected_delta

    assert model.compute_calls == [
        "W149",
    ]


def test_roman_detection_chi2_ignores_rubin_telescopes():

    roman = _FakeTelescope(
        "W149",
        flux=[10.0, 11.0],
        err_flux=[1.0, 1.0],
    )

    rubin = _FakeTelescope(
        "z",
        flux=[1.0e9, -1.0e9],
        err_flux=[1.0e-9, 1.0e-9],
    )

    model = _FakeModel(
        [
            roman,
            rubin,
        ],
        {
            "W149": [10.0, 11.0],
            "z": [0.0, 0.0],
        },
    )

    params = {
        "ftotal_W149": 9.0,
        "ftotal_z": 123.0,
    }

    result = roman_detection_chi2_statistics(
        model,
        params,
    )

    expected_chi2_true = 0.0

    expected_chi2_flat = (
        (10.0 - 9.0) ** 2
        + (11.0 - 9.0) ** 2
    )

    assert result["chi2_true"] == expected_chi2_true
    assert result["chi2_flat"] == expected_chi2_flat

    # Critical Roman-only guarantee:
    # Rubin was never evaluated at all.
    assert model.compute_calls == [
        "W149",
    ]


def test_roman_detection_chi2_does_not_mutate_inputs():

    roman = _FakeTelescope(
        "W149",
        flux=[10.0, 13.0, 12.0],
        err_flux=[1.0, 2.0, 1.0],
    )

    model = _FakeModel(
        [roman],
        {
            "W149": [10.0, 11.0, 12.0],
        },
    )

    params = {
        "ftotal_W149": 9.0,
        "t0": 123.0,
    }

    flux_before = np.asarray(
        roman.lightcurve["flux"],
        dtype=float,
    ).copy()

    err_before = np.asarray(
        roman.lightcurve["err_flux"],
        dtype=float,
    ).copy()

    params_before = copy.deepcopy(params)

    roman_detection_chi2_statistics(
        model,
        params,
    )

    assert np.array_equal(
        np.asarray(
            roman.lightcurve["flux"],
            dtype=float,
        ),
        flux_before,
    )

    assert np.array_equal(
        np.asarray(
            roman.lightcurve["err_flux"],
            dtype=float,
        ),
        err_before,
    )

    assert params == params_before


def test_roman_detection_chi2_rejects_invalid_uncertainty():

    roman = _FakeTelescope(
        "W149",
        flux=[10.0, 11.0],
        err_flux=[1.0, 0.0],
    )

    model = _FakeModel(
        [roman],
        {
            "W149": [10.0, 11.0],
        },
    )

    params = {
        "ftotal_W149": 9.0,
    }

    try:
        roman_detection_chi2_statistics(
            model,
            params,
        )
        raise AssertionError(
            "Expected invalid err_flux to raise."
        )

    except RuntimeError as exc:
        assert "strictly positive" in str(exc)


def test_roman_detection_chi2_rejects_nonfinite_data():

    roman = _FakeTelescope(
        "W149",
        flux=[10.0, np.nan],
        err_flux=[1.0, 1.0],
    )

    model = _FakeModel(
        [roman],
        {
            "W149": [10.0, 11.0],
        },
    )

    params = {
        "ftotal_W149": 9.0,
    }

    try:
        roman_detection_chi2_statistics(
            model,
            params,
        )
        raise AssertionError(
            "Expected non-finite Roman flux to raise."
        )

    except RuntimeError as exc:
        assert "non-finite" in str(exc)


def test_roman_detection_chi2_requires_exactly_one_roman_telescope():

    roman_a = _FakeTelescope(
        "W149",
        flux=[10.0],
        err_flux=[1.0],
    )

    roman_b = _FakeTelescope(
        "W149",
        flux=[10.0],
        err_flux=[1.0],
    )

    model = _FakeModel(
        [
            roman_a,
            roman_b,
        ],
        {
            "W149": [10.0],
        },
    )

    params = {
        "ftotal_W149": 9.0,
    }

    try:
        roman_detection_chi2_statistics(
            model,
            params,
        )
        raise AssertionError(
            "Expected duplicate W149 telescope to raise."
        )

    except RuntimeError as exc:
        assert "exactly one telescope" in str(exc)
