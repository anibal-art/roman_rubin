import numpy as np

from analysis.fisher_information import (
    decompose_jacobian_svd,
    functional_sigma,
    marginalize_fluxes,
    project_out_nuisance,
)


def test_projection_matches_schur_complement():

    rng = np.random.default_rng(
        12345
    )

    Jphysical = rng.normal(
        size=(
            100,
            3,
        )
    )

    Jnuisance = rng.normal(
        size=(
            100,
            2,
        )
    )

    J = np.column_stack(
        [
            Jphysical,
            Jnuisance,
        ]
    )

    F = (
        J.T
        @ J
    )

    (
        projected,
        diagnostic,
    ) = project_out_nuisance(
        Jphysical,
        Jnuisance,
    )

    assert (
        diagnostic[
            "nuisance_full_rank"
        ]
    )

    Fprojection = (
        projected.T
        @ projected
    )

    (
        Fschur,
        _,
    ) = marginalize_fluxes(
        F,
        3,
        rcond=1e-14,
    )

    assert np.allclose(
        Fprojection,
        Fschur,
        rtol=1e-10,
        atol=1e-10,
    )


def test_svd_covariance_known_diagonal():

    J = np.diag(
        [
            2.0,
            3.0,
            4.0,
        ]
    )

    dec = decompose_jacobian_svd(
        J
    )

    assert dec[
        "full_rank"
    ]

    assert dec[
        "rank"
    ] == 3

    expected = np.diag(
        [
            1 / 4,
            1 / 9,
            1 / 16,
        ]
    )

    assert np.allclose(
        dec[
            "covariance_pinv"
        ],
        expected,
        rtol=1e-14,
        atol=1e-14,
    )


def test_true_null_direction_is_infinite():

    J = np.diag(
        [
            2.0,
            0.0,
        ]
    )

    dec = decompose_jacobian_svd(
        J
    )

    assert dec[
        "rank"
    ] == 1

    sigma_x = functional_sigma(
        np.array(
            [
                1.0,
                0.0,
            ]
        ),
        dec,
    )

    sigma_null = functional_sigma(
        np.array(
            [
                0.0,
                1.0,
            ]
        ),
        dec,
    )

    assert np.isclose(
        sigma_x,
        0.5,
    )

    assert np.isinf(
        sigma_null
    )


def test_projection_removes_exact_nuisance_component():

    x = np.linspace(
        -1,
        1,
        100,
    )

    nuisance = np.column_stack(
        [
            np.ones_like(
                x
            ),
            x,
        ]
    )

    physical = np.column_stack(
        [
            x ** 2,
            3.0
            + 2.0 * x,
        ]
    )

    projected, diagnostic = (
        project_out_nuisance(
            physical,
            nuisance,
        )
    )

    assert diagnostic[
        "nuisance_rank"
    ] == 2

    # Second physical column lies entirely in nuisance span.
    assert np.linalg.norm(
        projected[
            :,
            1,
        ]
    ) < 1e-12

    # x^2 retains information.
    assert np.linalg.norm(
        projected[
            :,
            0,
        ]
    ) > 1.0


def test_svd_solution_stability_identical():

    from analysis.fisher_information import (
        svd_solution_stability,
    )

    J = np.diag(
        [
            4.0,
            2.0,
            1.0,
        ]
    )

    dec = decompose_jacobian_svd(
        J
    )

    out = svd_solution_stability(
        J,
        dec,
        J.copy(),
        dec,
        2.0,
        2.0,
    )

    assert np.isclose(
        out["fd_stability_jacobian_rel"],
        0.0,
    )

    assert np.isclose(
        out["fd_stability_smin_rel"],
        0.0,
    )

    assert np.isclose(
        out["fd_stability_weak_mode_angle_deg"],
        0.0,
    )

    assert np.isclose(
        out["fd_stability_sigma_piE_rel"],
        0.0,
    )


def test_svd_solution_stability_detects_change():

    from analysis.fisher_information import (
        svd_solution_stability,
    )

    J1 = np.diag(
        [
            4.0,
            2.0,
            1.0,
        ]
    )

    J2 = np.diag(
        [
            4.0,
            2.0,
            0.8,
        ]
    )

    dec1 = decompose_jacobian_svd(
        J1
    )

    dec2 = decompose_jacobian_svd(
        J2
    )

    out = svd_solution_stability(
        J1,
        dec1,
        J2,
        dec2,
        1.0,
        1.25,
    )

    assert out[
        "fd_stability_jacobian_rel"
    ] > 0

    assert np.isclose(
        out["fd_stability_smin_rel"],
        0.2,
    )

    assert np.isclose(
        out["fd_stability_weak_mode_angle_deg"],
        0.0,
    )

    assert np.isclose(
        out["fd_stability_sigma_piE_rel"],
        0.25,
    )
