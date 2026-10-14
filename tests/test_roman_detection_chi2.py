import copy

import numpy as np
from astropy.table import QTable

from functions_roman_rubin import (
    roman_detection_chi2_statistics,
    roman_detection_n3sigma_statistics,
    combine_roman_detection_statistics,
)


class _FakeTelescope:
    def __init__(
        self,
        name,
        flux,
        err_flux,
        time=None,
    ):
        self.name = name

        flux = np.asarray(
            flux,
            dtype=float,
        )

        err_flux = np.asarray(
            err_flux,
            dtype=float,
        )

        if time is None:
            time = np.arange(
                len(flux),
                dtype=float,
            )

        self.lightcurve = QTable(
            [
                np.asarray(time, dtype=float),
                flux,
                err_flux,
            ],
            names=[
                "time",
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


def test_roman_n3sigma_uses_flux_and_chronological_order():

    # Deliberately unsorted times.
    roman = _FakeTelescope(
        "W149",
        time=[3.0, 1.0, 2.0, 4.0],
        flux=[
            13.0,
            13.0,
            13.0,
            9.0,
        ],
        err_flux=[
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    )

    model = _FakeModel(
        [roman],
        {
            "W149": [
                13.0,
                13.0,
                13.0,
                9.0,
            ],
        },
    )

    params = {
        "ftotal_W149": 10.0,
    }

    result = roman_detection_n3sigma_statistics(
        model,
        params,
        nsigma=3.0,
    )

    # Exact equality with baseline + 3 sigma MUST count:
    # "at least 3 sigma".
    assert result["n3sigma_n_above"] == 3
    assert result["n3sigma_max_run"] == 3


def test_roman_n3sigma_ignores_rubin():

    roman = _FakeTelescope(
        "W149",
        flux=[13.0] * 6,
        err_flux=[1.0] * 6,
    )

    rubin = _FakeTelescope(
        "z",
        flux=[1.0e20] * 100,
        err_flux=[1.0e-20] * 100,
    )

    model = _FakeModel(
        [
            roman,
            rubin,
        ],
        {
            "W149": [13.0] * 6,
            "z": [0.0] * 100,
        },
    )

    params = {
        "ftotal_W149": 10.0,
    }

    result = roman_detection_n3sigma_statistics(
        model,
        params,
    )

    assert result["n_data"] == 6
    assert result["n3sigma_max_run"] == 6


def test_roman_detection_threshold_boundaries_are_inclusive():

    chi2_stats = {
        "roman_band_name": "W149",
        "n_data": 10,
        "baseline_flux": 5.0,
        "chi2_true": 10.0,
        "chi2_flat": 310.0,
        "delta_chi2": 300.0,
    }

    n3sigma_stats = {
        "roman_band_name": "W149",
        "n_data": 10,
        "baseline_flux": 5.0,
        "n3sigma_nsigma": 3.0,
        "n3sigma_n_above": 6,
        "n3sigma_max_run": 6,
    }

    result = combine_roman_detection_statistics(
        chi2_stats,
        n3sigma_stats,
        delta_chi2_min=300.0,
        n3sigma_min=6,
    )

    assert result["pass_delta_chi2"] is True
    assert result["pass_n3sigma"] is True
    assert result["roman_detected"] is True


def test_roman_detection_requires_both_criteria():

    base_chi2 = {
        "roman_band_name": "W149",
        "n_data": 10,
        "baseline_flux": 5.0,
        "chi2_true": 10.0,
        "chi2_flat": 310.0,
        "delta_chi2": 300.0,
    }

    base_n3 = {
        "roman_band_name": "W149",
        "n_data": 10,
        "baseline_flux": 5.0,
        "n3sigma_nsigma": 3.0,
        "n3sigma_n_above": 6,
        "n3sigma_max_run": 6,
    }

    fail_chi2 = dict(
        base_chi2,
        delta_chi2=np.nextafter(
            300.0,
            -np.inf,
        ),
    )

    result = combine_roman_detection_statistics(
        fail_chi2,
        base_n3,
    )

    assert result["pass_delta_chi2"] is False
    assert result["pass_n3sigma"] is True
    assert result["roman_detected"] is False

    fail_n3 = dict(
        base_n3,
        n3sigma_max_run=5,
    )

    result = combine_roman_detection_statistics(
        base_chi2,
        fail_n3,
    )

    assert result["pass_delta_chi2"] is True
    assert result["pass_n3sigma"] is False
    assert result["roman_detected"] is False
