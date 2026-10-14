import numpy as np
from astropy.table import QTable
import astropy.units as u

from analysis.fisher_information import (
    expected_fisher_flux_error,
)


class DummyTelescope:

    def __init__(
        self,
        model_flux,
        observed_flux,
        err_mag,
        observed_err_flux,
    ):

        self.name = "TEST"

        self.lightcurve = QTable(
            [
                np.arange(
                    len(model_flux),
                    dtype=float,
                ),
                np.asarray(
                    observed_flux,
                    dtype=float,
                ),
                np.asarray(
                    observed_err_flux,
                    dtype=float,
                ),
                np.asarray(
                    err_mag,
                    dtype=float,
                ),
            ],
            names=[
                "time",
                "flux",
                "err_flux",
                "err_mag",
            ],
        )

        self.lightcurve["time"].unit = u.day


class DummyModel:

    def __init__(
        self,
        model_flux,
    ):
        self.model_flux = np.asarray(
            model_flux,
            dtype=float,
        )

    def compute_the_microlensing_model(
        self,
        telescope,
        pyparams,
    ):
        return {
            "photometry":
                self.model_flux.copy(),
            "astrometry":
                None,
        }


def test_expected_fisher_flux_error_formula():

    model_flux = np.array(
        [
            100.0,
            250.0,
            1000.0,
        ]
    )

    observed_flux = np.array(
        [
            80.0,
            300.0,
            900.0,
        ]
    )

    err_mag = np.array(
        [
            0.01,
            0.05,
            0.20,
        ]
    )

    observed_err_flux = (
        observed_flux
        * np.log(10.0)
        / 2.5
        * err_mag
    )

    telescope = DummyTelescope(
        model_flux=model_flux,
        observed_flux=observed_flux,
        err_mag=err_mag,
        observed_err_flux=observed_err_flux,
    )

    model = DummyModel(
        model_flux
    )

    got = expected_fisher_flux_error(
        model,
        telescope,
        pyparams={},
    )

    expected = (
        model_flux
        * np.log(10.0)
        / 2.5
        * err_mag
    )

    assert np.allclose(
        got,
        expected,
        rtol=1e-14,
        atol=0.0,
    )


def test_expected_fisher_flux_error_ignores_noise_realization():

    model_flux = np.array(
        [
            100.0,
            250.0,
            1000.0,
        ]
    )

    err_mag = np.array(
        [
            0.01,
            0.05,
            0.20,
        ]
    )

    observed_flux_a = np.array(
        [
            80.0,
            300.0,
            900.0,
        ]
    )

    observed_flux_b = np.array(
        [
            120.0,
            180.0,
            1300.0,
        ]
    )

    observed_err_a = (
        observed_flux_a
        * np.log(10.0)
        / 2.5
        * err_mag
    )

    observed_err_b = (
        observed_flux_b
        * np.log(10.0)
        / 2.5
        * err_mag
    )

    telescope_a = DummyTelescope(
        model_flux=model_flux,
        observed_flux=observed_flux_a,
        err_mag=err_mag,
        observed_err_flux=observed_err_a,
    )

    telescope_b = DummyTelescope(
        model_flux=model_flux,
        observed_flux=observed_flux_b,
        err_mag=err_mag,
        observed_err_flux=observed_err_b,
    )

    model = DummyModel(
        model_flux
    )

    sigma_a = expected_fisher_flux_error(
        model,
        telescope_a,
        pyparams={},
    )

    sigma_b = expected_fisher_flux_error(
        model,
        telescope_b,
        pyparams={},
    )

    assert not np.allclose(
        observed_err_a,
        observed_err_b,
    )

    assert np.array_equal(
        sigma_a,
        sigma_b,
    )
