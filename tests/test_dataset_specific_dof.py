import numpy as np
import pytest

from functions_roman_rubin import (
    fit_chi2_dof_from_n_data,
)


def test_rr_dof_matches_exact_fitted_dataset():
    data = {
        "chi2": 52494.36787030658,
        "best_model": np.zeros(19),
    }

    result = fit_chi2_dof_from_n_data(
        data,
        52786,
    )

    assert result["chi2"] == pytest.approx(
        52494.36787030658
    )
    assert result["n_data"] == 52786
    assert result["n_fit_parameters"] == 19
    assert result["dof"] == 52767


def test_roman_dof_matches_exact_fitted_dataset():
    data = {
        "chi2": 51341.460786895994,
        "best_model": np.zeros(11),
    }

    result = fit_chi2_dof_from_n_data(
        data,
        51606,
    )

    assert result["chi2"] == pytest.approx(
        51341.460786895994
    )
    assert result["n_data"] == 51606
    assert result["n_fit_parameters"] == 11
    assert result["dof"] == 51595


def test_dataset_specific_dof_rejects_nonfinite_n_data():
    data = {
        "chi2": 1.0,
        "best_model": np.zeros(2),
    }

    with pytest.raises(ValueError):
        fit_chi2_dof_from_n_data(
            data,
            np.nan,
        )
