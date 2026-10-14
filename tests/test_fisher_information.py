import numpy as np

from analysis.fisher_information import (
    decompose_fisher,
    functional_sigma,
    marginalize_fluxes,
)


def test_schur_complement_matches_full_covariance():

    M = np.array(
        [
            [4.0, 0.2, 0.5, 0.1],
            [0.2, 3.0, 0.3, 0.4],
            [0.5, 0.3, 2.5, 0.1],
            [0.1, 0.4, 0.1, 2.0],
        ]
    )

    F = (
        M.T
        @ M
        + np.eye(
            4
        )
    )

    (
        Fphys,
        diag,
    ) = marginalize_fluxes(
        F,
        2,
    )

    assert (
        diag[
            "nuisance_full_rank"
        ]
    )

    assert np.allclose(
        np.linalg.inv(
            Fphys
        ),
        np.linalg.inv(
            F
        )[
            :2,
            :2,
        ],
        rtol=1e-11,
        atol=1e-12,
    )


def test_null_direction_has_infinite_sigma():

    F = np.diag(
        [
            4.0,
            0.0,
        ]
    )

    dec = decompose_fisher(
        F
    )

    assert np.isclose(
        functional_sigma(
            np.array(
                [
                    1.0,
                    0.0,
                ]
            ),
            dec,
        ),
        0.5,
    )

    assert np.isinf(
        functional_sigma(
            np.array(
                [
                    0.0,
                    1.0,
                ]
            ),
            dec,
        )
    )


def test_full_rank_covariance():

    F = np.diag(
        [
            4.0,
            9.0,
            16.0,
        ]
    )

    dec = decompose_fisher(
        F
    )

    assert dec[
        "full_rank"
    ]

    assert dec[
        "rank"
    ] == 3

    assert np.allclose(
        dec[
            "covariance_pinv"
        ],
        np.diag(
            [
                1 / 4,
                1 / 9,
                1 / 16,
            ]
        ),
    )
