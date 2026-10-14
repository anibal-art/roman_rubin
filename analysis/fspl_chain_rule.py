"""
FSPL/FSPLarge physical derivatives for Roman--Rubin E1.

The microlensing trajectory is differentiated analytically using
the exact pyLIMA parallax convention.

Only the local finite-source magnification A(u, rho) is
differentiated numerically.

This module does not modify pyLIMA or telescope/model state.
"""

import numpy as np

from pyLIMA.magnification import magnification_VBB


MACHINE_EPS = np.finfo(float).eps

# For a five-point centred first derivative:
#
# truncation ~ h^4
# roundoff   ~ eps / h
#
# so the natural floating-point scale is eps^(1/5).
LOCAL_REL_STEP_LARGE = MACHINE_EPS ** (1.0 / 5.0)
LOCAL_REL_STEP_SMALL = 0.5 * LOCAL_REL_STEP_LARGE


def _array(value):
    if hasattr(value, "value"):
        value = value.value

    return np.asarray(
        value,
        dtype=float,
    )


def _five_point_derivative(
    evaluate,
    x,
    h,
):
    """
    Five-point centred derivative:

      [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / (12h)
    """

    return (
        -evaluate(x + 2.0 * h)
        + 8.0 * evaluate(x + h)
        - 8.0 * evaluate(x - h)
        + evaluate(x - 2.0 * h)
    ) / (
        12.0 * h
    )


def _pair_diagnostics(
    primary,
    alternate,
):
    primary = np.asarray(
        primary,
        dtype=float,
    )

    alternate = np.asarray(
        alternate,
        dtype=float,
    )

    difference = (
        primary
        - alternate
    )

    tiny = np.finfo(float).tiny

    l2_scale = max(
        float(
            np.linalg.norm(
                primary
            )
        ),
        float(
            np.linalg.norm(
                alternate
            )
        ),
        tiny,
    )

    max_scale = max(
        float(
            np.max(
                np.abs(
                    primary
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    alternate
                )
            )
        ),
        tiny,
    )

    return {
        "pair_relative_l2":
            float(
                np.linalg.norm(
                    difference
                )
                / l2_scale
            ),

        "pair_relative_max":
            float(
                np.max(
                    np.abs(
                        difference
                    )
                )
                / max_scale
            ),
    }


def _fspl_magnification_from_u(
    telescope,
    u,
    rho,
):
    """
    Evaluate the exact same VBM FSPL magnification used by
    pyLIMA FSPLarge, but parameterized directly by u.

    magnification_FSPL computes sqrt(tau^2 + beta^2), so
    passing tau=u, beta=0 evaluates A(u, rho).
    """

    u = np.asarray(
        u,
        dtype=float,
    )

    beta = np.zeros_like(
        u
    )

    linear_ld = getattr(
        telescope,
        "ld_a1",
        0.0,
    )

    if linear_ld is None:
        linear_ld = 0.0

    sqrt_ld = getattr(
        telescope,
        "ld_a2",
        None,
    )

    if (
        sqrt_ld is not None
        and float(sqrt_ld) > 0.0
    ):
        return (
            magnification_VBB.magnification_FSPL(
                u,
                beta,
                float(rho),
                float(linear_ld),
                float(sqrt_ld),
            )
        )

    return (
        magnification_VBB.magnification_FSPL(
            u,
            beta,
            float(rho),
            float(linear_ld),
        )
    )


def fspl_chain_rule_flux_derivatives(
    model,
    telescope,
    pyparams,
    physical_names,
):
    """
    Return FSPL physical flux derivatives using

        dF/dtheta = Fs * dA/du * du/dtheta

    and

        dF/drho = Fs * dA/drho.

    The trajectory derivatives are analytic.

    dA/du and dA/drho are evaluated by local five-point
    differences in (u, rho), not by perturbing t0/u0/tE/piE.

    Returns
    -------
    primary_rows : ndarray, shape (Nphysical, Ndata)
    alternate_rows : ndarray, shape (Nphysical, Ndata)
    diagnostics : dict keyed by physical parameter
    """

    model_type = str(
        model.model_type()
    ).upper()

    if "FSPL" not in model_type:
        raise ValueError(
            "fspl_chain_rule_flux_derivatives "
            "requires an FSPL/FSPLarge model."
        )

    double_source = getattr(
        model,
        "double_source_model",
        ["None"],
    )

    if str(double_source[0]).lower() != "none":
        raise NotImplementedError(
            "FSPL chain-rule Fisher currently supports "
            "one source only."
        )

    orbital_motion = getattr(
        model,
        "orbital_motion_model",
        ["None"],
    )

    if str(orbital_motion[0]).lower() != "none":
        raise NotImplementedError(
            "FSPL chain-rule Fisher currently does not "
            "support orbital motion."
        )

    lightcurve = telescope.lightcurve

    if (
        lightcurve is None
        or len(lightcurve) == 0
    ):
        raise ValueError(
            f"{telescope.name}: empty light curve."
        )

    time = _array(
        lightcurve["time"]
    )

    n = len(time)

    t0 = float(
        pyparams["t0"]
    )

    u0 = float(
        pyparams["u0"]
    )

    tE = float(
        pyparams["tE"]
    )

    rho = float(
        pyparams["rho"]
    )

    if (
        not np.isfinite(tE)
        or tE <= 0.0
    ):
        raise ValueError(
            f"Invalid tE={tE}"
        )

    if (
        not np.isfinite(rho)
        or rho <= 0.0
    ):
        raise ValueError(
            f"Invalid rho={rho}"
        )

    # ========================================================
    # Exact pyLIMA trajectory convention
    # ========================================================

    base_tau = (
        time - t0
    ) / tE

    delta_tau = np.zeros(
        n,
        dtype=float,
    )

    delta_beta = np.zeros(
        n,
        dtype=float,
    )

    d_N = np.zeros(
        n,
        dtype=float,
    )

    d_E = np.zeros(
        n,
        dtype=float,
    )

    has_parallax = (
        "piEN" in pyparams.keys()
        and "piEE" in pyparams.keys()
        and str(
            getattr(
                model,
                "parallax_model",
                ["None"],
            )[0]
        ).lower()
        != "none"
    )

    if has_parallax:

        positions = np.asarray(
            telescope.deltas_positions[
                "photometry"
            ],
            dtype=float,
        )

        if positions.shape != (
            2,
            n,
        ):
            raise RuntimeError(
                f"{telescope.name}: "
                "unexpected parallax position shape "
                f"{positions.shape}, expected {(2, n)}"
            )

        d_N = positions[0]
        d_E = positions[1]

        (
            delta_tau,
            delta_beta,
        ) = (
            model.parallax_trajectory_shifts(
                positions,
                pyparams,
            )
        )

        delta_tau = np.asarray(
            delta_tau,
            dtype=float,
        )

        delta_beta = np.asarray(
            delta_beta,
            dtype=float,
        )

    tau = (
        base_tau
        + delta_tau
    )

    beta = (
        u0
        + delta_beta
    )

    u = np.hypot(
        tau,
        beta,
    )

    # ========================================================
    # Independent consistency check against sources_trajectory
    # ========================================================

    (
        source_x,
        source_y,
        source2_x,
        source2_y,
        _,
        _,
    ) = model.sources_trajectory(
        telescope,
        pyparams,
        data_type="photometry",
    )

    if source2_x is not None:
        raise RuntimeError(
            "Unexpected second source in FSPL chain rule."
        )

    source_x = np.asarray(
        source_x,
        dtype=float,
    )

    source_y = np.asarray(
        source_y,
        dtype=float,
    )

    trajectory_x_max_abs = float(
        np.max(
            np.abs(
                source_x + tau
            )
        )
    )

    trajectory_y_max_abs = float(
        np.max(
            np.abs(
                source_y + beta
            )
        )
    )

    # ========================================================
    # Local magnification derivatives
    # ========================================================

    scale_u = np.maximum(
        np.abs(u),
        rho,
    )

    h_u_small = (
        LOCAL_REL_STEP_SMALL
        * scale_u
    )

    h_u_large = (
        LOCAL_REL_STEP_LARGE
        * scale_u
    )

    def evaluate_u(value):
        return _fspl_magnification_from_u(
            telescope,
            value,
            rho,
        )

    dA_du_primary = (
        _five_point_derivative(
            evaluate_u,
            u,
            h_u_small,
        )
    )

    dA_du_alternate = (
        _five_point_derivative(
            evaluate_u,
            u,
            h_u_large,
        )
    )

    h_rho_small = (
        LOCAL_REL_STEP_SMALL
        * rho
    )

    h_rho_large = (
        LOCAL_REL_STEP_LARGE
        * rho
    )

    def evaluate_rho(value):
        return _fspl_magnification_from_u(
            telescope,
            u,
            value,
        )

    dA_drho_primary = (
        _five_point_derivative(
            evaluate_rho,
            rho,
            h_rho_small,
        )
    )

    dA_drho_alternate = (
        _five_point_derivative(
            evaluate_rho,
            rho,
            h_rho_large,
        )
    )

    if (
        not np.all(
            np.isfinite(
                dA_du_primary
            )
        )
        or not np.all(
            np.isfinite(
                dA_drho_primary
            )
        )
    ):
        raise RuntimeError(
            f"{telescope.name}: "
            "non-finite local FSPL derivatives."
        )

    # Check that A(u,rho) reproduces model_magnification.
    A_local = _fspl_magnification_from_u(
        telescope,
        u,
        rho,
    )

    A_model = np.asarray(
        model.model_magnification(
            telescope,
            pyparams,
        ),
        dtype=float,
    )

    magnification_max_abs = float(
        np.max(
            np.abs(
                A_local
                - A_model
            )
        )
    )

    magnification_relative_l2 = float(
        np.linalg.norm(
            A_local
            - A_model
        )
        / max(
            np.linalg.norm(
                A_model
            ),
            np.finfo(float).tiny,
        )
    )

    # ========================================================
    # Analytic du / dtheta
    # ========================================================

    inverse_u = np.divide(
        1.0,
        u,
        out=np.zeros_like(
            u
        ),
        where=(
            u > 0.0
        ),
    )

    radial_tau = (
        tau
        * inverse_u
    )

    radial_beta = (
        beta
        * inverse_u
    )

    du = {
        "t0":
            radial_tau
            * (
                -1.0
                / tE
            ),

        "u0":
            radial_beta,

        "tE":
            radial_tau
            * (
                -base_tau
                / tE
            ),
    }

    if has_parallax:

        du["piEN"] = (
            radial_tau
            * d_N
            + radial_beta
            * d_E
        )

        du["piEE"] = (
            radial_tau
            * d_E
            - radial_beta
            * d_N
        )

    # ========================================================
    # Flux derivatives
    # ========================================================

    fsource_name = (
        "fsource_"
        + telescope.name
    )

    fsource = float(
        pyparams[
            fsource_name
        ]
    )

    primary_rows = []

    alternate_rows = []

    diagnostics = {}

    limb_ratio = (
        u / rho
    )

    n_near_limb = int(
        np.sum(
            np.abs(
                limb_ratio - 1.0
            )
            < 0.05
        )
    )

    for name in physical_names:

        if name == "rho":

            primary = (
                fsource
                * dA_drho_primary
            )

            alternate = (
                fsource
                * dA_drho_alternate
            )

        elif name in du:

            primary = (
                fsource
                * dA_du_primary
                * du[name]
            )

            alternate = (
                fsource
                * dA_du_alternate
                * du[name]
            )

        else:
            raise NotImplementedError(
                "FSPL chain-rule derivative "
                f"not implemented for {name!r}."
            )

        primary_rows.append(
            primary
        )

        alternate_rows.append(
            alternate
        )

        pair = _pair_diagnostics(
            primary,
            alternate,
        )

        diagnostics[name] = {
            "method":
                "fspl_chain_rule_local_five_point",

            # Kept for compatibility with existing audit
            # printers. There is no physical-parameter FD
            # step in this method.
            "selected_step":
                np.nan,

            **pair,

            "local_u_relative_step_primary":
                float(
                    LOCAL_REL_STEP_SMALL
                ),

            "local_u_relative_step_alternate":
                float(
                    LOCAL_REL_STEP_LARGE
                ),

            "local_rho_step_primary":
                float(
                    h_rho_small
                ),

            "local_rho_step_alternate":
                float(
                    h_rho_large
                ),

            "trajectory_x_max_abs":
                trajectory_x_max_abs,

            "trajectory_y_max_abs":
                trajectory_y_max_abs,

            "magnification_max_abs":
                magnification_max_abs,

            "magnification_relative_l2":
                magnification_relative_l2,

            "n_near_limb_005":
                n_near_limb,
        }

    return (
        np.asarray(
            primary_rows,
            dtype=float,
        ),
        np.asarray(
            alternate_rows,
            dtype=float,
        ),
        diagnostics,
    )
