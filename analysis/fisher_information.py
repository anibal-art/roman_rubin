"""
Local Fisher information for Roman--Rubin E1.

Version 2.1
-----------

Scientific/numerical design:

1. Physical derivatives are evaluated here, independently of pyLIMA's
   Numerical Jacobian, using a five-point centred finite difference.

2. The finite-difference scale is tied to the physical parameter scale,
   not to the absolute numerical value of e.g. t0 ~ 2.46e6 JD.

3. Source/blend flux derivatives are analytical.

4. Flux nuisance parameters are marginalized by projecting the physical
   Jacobian out of the nuisance-column subspace.

5. The physical Jacobian is expressed in locally scaled coordinates:

       q_t0  = delta(t0) / tE
       q_u0  = u0
       q_tE  = delta(log tE)
       q_rho = delta(log rho)
       q_pi  = piE components

   locally, so

       dx/dq = diag(tE, 1, tE, rho, 1, 1)

   for an FSPL+parallax model.

6. Covariances and rank diagnostics are obtained directly from the SVD
   of the marginalized Jacobian. We do not form/invert J.T @ J to obtain
   the covariance.

Fisher matrices are still returned for diagnostics and output, because
E1 explicitly stores them.
"""

from __future__ import annotations

import copy
from typing import Mapping

import numpy as np

from analysis.fspl_chain_rule import (
    fspl_chain_rule_flux_derivatives,
)

from set_model_pyLIMA import (
    build_pyLIMA_model,
    normalize_model_name,
)


FLUX_PREFIXES = (
    "fsource_",
    "ftotal_",
    "fblend_",
    "gblend_",
)

DEFAULT_STEP_MULTIPLIERS = (
    10.0,
    3.0,
    1.0,
    0.3,
)

MACHINE_EPS = np.finfo(float).eps


# ======================================================================
# Basic helpers
# ======================================================================


def _array(x):
    return np.asarray(
        x.value
        if hasattr(x, "value")
        else x,
        dtype=float,
    )


def _get(params, name):
    if isinstance(params, Mapping):
        value = params[name]
    else:
        try:
            value = params[name]
        except Exception:
            value = getattr(
                params,
                name,
            )

    if value is None:
        raise ValueError(
            f"{name} is None"
        )

    return float(value)


def parameter_names(model):
    return [
        name
        for name, _ in sorted(
            model.model_dictionnary.items(),
            key=lambda item: item[1],
        )
    ]


def physical_names(model):
    return [
        name
        for name in parameter_names(model)
        if not name.startswith(
            FLUX_PREFIXES
        )
    ]


def local_flux_names(
    model,
    telescope,
):
    blend = getattr(
        model,
        "blend_flux_parameter",
        "ftotal",
    )

    if blend == "ftotal":
        return [
            f"fsource_{telescope.name}",
            f"ftotal_{telescope.name}",
        ]

    if blend == "fblend":
        return [
            f"fsource_{telescope.name}",
            f"fblend_{telescope.name}",
        ]

    if blend == "gblend":
        return [
            f"fsource_{telescope.name}",
            f"gblend_{telescope.name}",
        ]

    if blend == "noblend":
        return [
            f"fsource_{telescope.name}",
        ]

    raise ValueError(
        "Unsupported blend_flux_parameter="
        f"{blend!r}"
    )


def parameter_vector(
    model,
    params,
):
    return np.asarray(
        [
            _get(
                params,
                name,
            )
            for name in parameter_names(
                model
            )
        ],
        dtype=float,
    )


def is_roman(telescope):
    name = str(
        getattr(
            telescope,
            "name",
            "",
        )
    ).lower()

    filt = str(
        getattr(
            telescope,
            "filter",
            "",
        )
    ).lower()

    return (
        name
        in {
            "roman",
            "w149",
            "f146",
        }
        or
        filt
        in {
            "w149",
            "f146",
        }
    )


# ======================================================================
# Model reconstruction for R / RR / RR(D_perp=0)
# ======================================================================


def clear_geometry(telescope):
    for attr in (
        "deltas_positions",
        "Earth_positions",
        "Earth_speeds",
        "sidereal_times",
        "telescope_positions",
        "Earth_positions_projected",
        "Earth_speeds_projected",
    ):
        if hasattr(
            telescope,
            attr,
        ):
            setattr(
                telescope,
                attr,
                {},
            )


def rebuild_model(
    source_model,
    true_params,
    telescope_names,
    *,
    roman_at_earth=False,
):
    """
    Build a fresh pyLIMA model for one E1 configuration.

    source_model is never mutated.
    """

    event = copy.deepcopy(
        source_model.event
    )

    keep = set(
        telescope_names
    )

    event.telescopes = [
        telescope
        for telescope
        in event.telescopes
        if (
            telescope.name in keep
            and telescope.lightcurve
            is not None
            and len(
                telescope.lightcurve
            )
            > 0
        )
    ]

    if not event.telescopes:
        raise ValueError(
            "No telescopes left in "
            "Fisher configuration."
        )

    for telescope in event.telescopes:

        clear_geometry(
            telescope
        )

        if (
            roman_at_earth
            and is_roman(
                telescope
            )
        ):
            telescope.location = "Earth"

            # pyLIMA represents the Earth centre by using
            # an altitude equal to minus its Earth radius.
            # Set it explicitly rather than relying on the
            # Telescope default.
            telescope.altitude = -6378100.0

            telescope.spacecraft_name = None

            telescope.spacecraft_positions = {
                "astrometry": [],
                "photometry": [],
            }

    if roman_at_earth:
        # Earth-centred Roman requires sidereal-time machinery.
        # Configure Astropy locally so this module can be run
        # independently of functions_roman_rubin imports.
        from astropy.utils import iers

        iers.conf.auto_max_age = None
        iers.conf.auto_download = False
        iers.conf.iers_degraded_accuracy = "warn"

    event.check_event()

    parallax = getattr(
        source_model,
        "parallax_model",
        [
            "None",
            0.0,
        ],
    )

    use_parallax = (
        str(
            parallax[0]
        ).lower()
        != "none"
    )

    kwargs = dict(
        pyLIMA_event=event,
        model=normalize_model_name(
            source_model.model_type()
        ),
        use_parallax=use_parallax,
        t0_parallax=(
            float(
                parallax[1]
            )
            if use_parallax
            else None
        ),
        blend_flux_parameter=getattr(
            source_model,
            "blend_flux_parameter",
            "ftotal",
        ),
    )

    origin = getattr(
        source_model,
        "origin",
        None,
    )

    if origin is not None:
        kwargs["origin"] = copy.deepcopy(
            origin
        )

    model = build_pyLIMA_model(
        **kwargs
    )

    vector = parameter_vector(
        model,
        true_params,
    )

    pyparams = (
        model.compute_pyLIMA_parameters(
            vector
        )
    )

    return (
        model,
        pyparams,
    )


# ======================================================================
# Five-point physical derivatives
# ======================================================================


def _model_flux(
    model,
    telescope,
    vector,
):
    params = (
        model.compute_pyLIMA_parameters(
            np.asarray(
                vector,
                dtype=float,
            )
        )
    )

    flux = (
        model.compute_the_microlensing_model(
            telescope,
            params,
        )["photometry"]
    )

    flux = _array(
        flux
    )

    if not np.all(
        np.isfinite(
            flux
        )
    ):
        raise RuntimeError(
            f"{telescope.name}: "
            "non-finite model flux"
        )

    return flux


def _base_step(
    name,
    value,
    params,
):
    """
    Physically scaled reference step.

    The final derivative is not tied to this single value:
    several multiples are evaluated and a stable adjacent
    pair is selected.
    """

    value = float(
        value
    )

    if name == "t0":
        tE = abs(
            _get(
                params,
                "tE",
            )
        )

        return (
            1e-5
            * max(
                tE,
                0.1,
            )
        )

    if name == "tE":
        return (
            1e-5
            * max(
                abs(
                    value
                ),
                0.1,
            )
        )

    if name == "u0":
        return (
            1e-5
            * max(
                abs(
                    value
                ),
                0.01,
            )
        )

    if name == "rho":
        return (
            1e-5
            * max(
                abs(
                    value
                ),
                1e-5,
            )
        )

    if name in (
        "piEN",
        "piEE",
    ):
        return (
            1e-5
            * max(
                abs(
                    value
                ),
                1e-3,
            )
        )

    return (
        1e-5
        * max(
            abs(
                value
            ),
            1.0,
        )
    )


def _safe_step(
    name,
    value,
    h,
):
    """
    Keep all five stencil positions physical for positive
    parameters such as tE and rho.
    """

    h = float(
        abs(
            h
        )
    )

    if name in (
        "tE",
        "rho",
    ):
        value = float(
            value
        )

        if value <= 0:
            raise ValueError(
                f"{name} must be >0, got {value}"
            )

        # Need x - 2h > 0.
        h = min(
            h,
            0.2 * value,
        )

    if (
        not np.isfinite(
            h
        )
        or h <= 0
    ):
        raise ValueError(
            f"Invalid finite-difference "
            f"step for {name}: {h}"
        )

    return h


def five_point_derivative(
    model,
    telescope,
    vector,
    index,
    h,
):
    """
    Five-point centred derivative:

      f'(x) =
        [-f(x+2h) + 8 f(x+h)
         -8 f(x-h) + f(x-2h)] / (12h)

    Truncation error: O(h^4).
    """

    vector = np.asarray(
        vector,
        dtype=float,
    )

    evaluations = {}

    for multiplier in (
        2.0,
        1.0,
        -1.0,
        -2.0,
    ):
        trial = vector.copy()

        trial[
            index
        ] += (
            multiplier
            * h
        )

        evaluations[
            multiplier
        ] = _model_flux(
            model,
            telescope,
            trial,
        )

    derivative = (
        -evaluations[2.0]
        + 8.0
        * evaluations[1.0]
        - 8.0
        * evaluations[-1.0]
        + evaluations[-2.0]
    ) / (
        12.0
        * h
    )

    if not np.all(
        np.isfinite(
            derivative
        )
    ):
        raise RuntimeError(
            f"{telescope.name}: "
            "non-finite five-point derivative"
        )

    return derivative


def _derivative_distance(
    first,
    second,
):
    first = np.asarray(
        first,
        dtype=float,
    )

    second = np.asarray(
        second,
        dtype=float,
    )

    difference = (
        first
        - second
    )

    scale_l2 = max(
        np.linalg.norm(
            first
        ),
        np.linalg.norm(
            second
        ),
        1e-300,
    )

    scale_max = max(
        float(
            np.max(
                np.abs(
                    first
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    second
                )
            )
        ),
        1e-300,
    )

    return {
        "relative_l2":
            float(
                np.linalg.norm(
                    difference
                )
                / scale_l2
            ),

        "relative_max":
            float(
                np.max(
                    np.abs(
                        difference
                    )
                )
                / scale_max
            ),
    }


def stable_five_point_derivative(
    model,
    telescope,
    vector,
    index,
    name,
    params,
    *,
    step_multipliers=
        DEFAULT_STEP_MULTIPLIERS,
):
    """
    Evaluate several five-point steps and select the smaller
    step of the most mutually consistent adjacent pair.

    This detects the plateau between large-h truncation error
    and small-h roundoff/cancellation.
    """

    value = float(
        vector[
            index
        ]
    )

    reference_step = _base_step(
        name,
        value,
        params,
    )

    candidates = []

    seen_steps = set()

    for multiplier in step_multipliers:

        h = _safe_step(
            name,
            value,
            reference_step
            * float(
                multiplier
            ),
        )

        # _safe_step may cap multiple candidates to the same h.
        key = float(
            h
        )

        if key in seen_steps:
            continue

        seen_steps.add(
            key
        )

        derivative = five_point_derivative(
            model,
            telescope,
            vector,
            index,
            h,
        )

        candidates.append(
            {
                "multiplier":
                    float(
                        multiplier
                    ),

                "h":
                    float(
                        h
                    ),

                "derivative":
                    derivative,
            }
        )

    if not candidates:
        raise RuntimeError(
            f"No derivative candidates for {name}"
        )

    # Largest h -> smallest h.
    candidates.sort(
        key=lambda item:
            item["h"],
        reverse=True,
    )

    if len(
        candidates
    ) == 1:
        selected = candidates[0]

        diagnostic = {
            "parameter":
                name,

            "selected_h":
                selected["h"],

            "selected_multiplier":
                selected[
                    "multiplier"
                ],

            "alternate_h":
                selected["h"],

            "alternate_multiplier":
                selected[
                    "multiplier"
                ],

            "pair_relative_l2":
                np.nan,

            "pair_relative_max":
                np.nan,

            "n_candidates":
                1,
        }

        return (
            selected[
                "derivative"
            ],
            selected[
                "derivative"
            ].copy(),
            diagnostic,
        )

    comparisons = []

    for i in range(
        len(
            candidates
        )
        - 1
    ):
        large = candidates[
            i
        ]

        small = candidates[
            i + 1
        ]

        distance = (
            _derivative_distance(
                large[
                    "derivative"
                ],
                small[
                    "derivative"
                ],
            )
        )

        comparisons.append(
            {
                "large":
                    large,

                "small":
                    small,

                **distance,
            }
        )

    best = min(
        comparisons,
        key=lambda item:
            (
                item[
                    "relative_l2"
                ],
                item[
                    "relative_max"
                ],
            ),
    )

    # Prefer the smaller h within the stable pair for the
    # primary derivative, but retain the larger-h member as an
    # equally plausible numerical alternative.
    selected = best[
        "small"
    ]

    alternate = best[
        "large"
    ]

    diagnostic = {
        "parameter":
            name,

        "selected_h":
            selected[
                "h"
            ],

        "selected_multiplier":
            selected[
                "multiplier"
            ],

        "alternate_h":
            alternate[
                "h"
            ],

        "alternate_multiplier":
            alternate[
                "multiplier"
            ],

        "pair_relative_l2":
            best[
                "relative_l2"
            ],

        "pair_relative_max":
            best[
                "relative_max"
            ],

        "n_candidates":
            len(
                candidates
            ),
    }

    return (
        selected[
            "derivative"
        ],
        alternate[
            "derivative"
        ],
        diagnostic,
    )


# ======================================================================
# Analytical flux nuisance derivatives
# ======================================================================


def _flux_derivatives(
    model,
    telescope,
    params,
):
    amplification = _array(
        model.model_magnification(
            telescope,
            params,
        )
    )

    blend = getattr(
        model,
        "blend_flux_parameter",
        "ftotal",
    )

    if blend == "ftotal":
        return np.vstack(
            [
                amplification
                - 1.0,

                np.ones_like(
                    amplification
                ),
            ]
        )

    if blend == "fblend":
        return np.vstack(
            [
                amplification,

                np.ones_like(
                    amplification
                ),
            ]
        )

    if blend == "gblend":
        fsource = _get(
            params,
            f"fsource_{telescope.name}",
        )

        gblend = _get(
            params,
            f"gblend_{telescope.name}",
        )

        return np.vstack(
            [
                amplification
                + gblend,

                np.full_like(
                    amplification,
                    fsource,
                ),
            ]
        )

    if blend == "noblend":
        return np.vstack(
            [
                amplification,
            ]
        )

    raise ValueError(
        "Unsupported blend_flux_parameter="
        f"{blend!r}"
    )


# ======================================================================
# Global weighted Jacobian
# ======================================================================



def expected_fisher_flux_error(
    model,
    telescope,
    pyparams,
):
    """
    Expected local flux uncertainty used by E1 Fisher.

    The simulated light curve stores err_mag from the
    instrumental noise model evaluated before the random
    magnitude draw.

    The observed err_flux, however, is reconstructed using
    the noisy realized flux. That makes the Fisher weights
    realization-dependent.

    E1 instead uses

        sigma_F =
            F_model
            * ln(10) / 2.5
            * sigma_mag

    where F_model is evaluated at the truth for the current
    Fisher configuration.

    This changes only the Fisher weights. It does not mutate
    the simulated photometry or the uncertainties used by
    the fitting pipeline.
    """

    lightcurve = telescope.lightcurve

    if lightcurve is None or len(lightcurve) == 0:
        raise ValueError(
            f"{telescope.name}: empty light curve."
        )

    if "err_mag" not in lightcurve.colnames:
        raise KeyError(
            f"{telescope.name}: missing err_mag."
        )

    err_mag = _array(
        lightcurve["err_mag"]
    )

    model_result = (
        model.compute_the_microlensing_model(
            telescope,
            pyparams,
        )
    )

    model_flux = _array(
        model_result["photometry"]
    )

    n = len(lightcurve)

    if model_flux.shape != (n,):
        raise RuntimeError(
            f"{telescope.name}: "
            f"model flux shape={model_flux.shape}, "
            f"expected {(n,)}."
        )

    if err_mag.shape != (n,):
        raise RuntimeError(
            f"{telescope.name}: "
            f"err_mag shape={err_mag.shape}, "
            f"expected {(n,)}."
        )

    if (
        not np.all(np.isfinite(model_flux))
        or np.any(model_flux <= 0.0)
    ):
        raise RuntimeError(
            f"{telescope.name}: "
            "invalid model flux for Fisher weights."
        )

    if (
        not np.all(np.isfinite(err_mag))
        or np.any(err_mag <= 0.0)
    ):
        raise RuntimeError(
            f"{telescope.name}: "
            "invalid err_mag for Fisher weights."
        )

    expected_err_flux = (
        model_flux
        * np.log(10.0)
        / 2.5
        * err_mag
    )

    if (
        not np.all(
            np.isfinite(
                expected_err_flux
            )
        )
        or np.any(
            expected_err_flux <= 0.0
        )
    ):
        raise RuntimeError(
            f"{telescope.name}: "
            "invalid expected Fisher flux error."
        )

    return expected_err_flux


def _resolve_physical_derivative_method(
    model,
    requested_method,
):
    """
    Resolve the physical-derivative engine.

    The VBM chain-rule implementation is valid specifically for
    pyLIMA FSPLarge. Other model families, including the Yoo FSPL
    implementation, retain the validated five-point fallback.
    """

    if requested_method != "auto":
        return requested_method

    model_type = str(
        model.model_type()
    ).strip()

    if model_type.lower() == "fsplarge":
        return "fspl_chain_rule"

    return "five_point_stable_pair"


def weighted_jacobian(
    model,
    pyparams,
    *,
    step_multipliers=
        DEFAULT_STEP_MULTIPLIERS,
    physical_derivative_method=
        "auto",
):
    """
    Build the global normalized Jacobian.

    Physical columns:
        our adaptive five-point derivatives.

    Flux columns:
        exact analytical derivatives.

    Every row is divided by the expected local flux error

        F_model * ln(10) / 2.5 * err_mag,

    so Fisher information is independent of the random
    photometric realization.

    The simulated light curve itself is never mutated.

    Therefore F = J.T @ J.
    """

    phys = physical_names(
        model
    )

    all_names = parameter_names(
        model
    )

    parameter_index = {
        name: i
        for i, name in enumerate(
            all_names
        )
    }

    vector = parameter_vector(
        model,
        pyparams,
    )

    telescopes = [
        telescope
        for telescope
        in model.event.telescopes
        if (
            telescope.lightcurve
            is not None
            and len(
                telescope.lightcurve
            )
            > 0
        )
    ]

    if not telescopes:
        raise ValueError(
            "No non-empty telescopes."
        )

    nuisance = []

    for telescope in telescopes:
        nuisance.extend(
            local_flux_names(
                model,
                telescope,
            )
        )

    names = (
        phys
        + nuisance
    )

    global_column = {
        name: i
        for i, name in enumerate(
            names
        )
    }

    nphys = len(
        phys
    )

    resolved_derivative_method = (
        _resolve_physical_derivative_method(
            model,
            physical_derivative_method,
        )
    )

    blocks = []

    alternate_blocks = []

    telescope_contributions = {}

    telescope_n_points = {}

    telescope_weight_diagnostics = {}

    derivative_diagnostics = {}

    for telescope in telescopes:

        n = len(
            telescope.lightcurve
        )

        observed_err = _array(
            telescope.lightcurve[
                "err_flux"
            ]
        )

        err = expected_fisher_flux_error(
            model,
            telescope,
            pyparams,
        )

        if err.shape != (
            n,
        ):
            raise RuntimeError(
                f"{telescope.name}: "
                f"err_flux shape={err.shape}, "
                f"expected {(n,)}"
            )

        if (
            np.any(
                err <= 0
            )
            or not np.all(
                np.isfinite(
                    err
                )
            )
        ):
            raise RuntimeError(
                f"{telescope.name}: "
                "invalid err_flux."
            )

        observed_valid = (
            np.isfinite(observed_err)
            & (observed_err > 0.0)
        )

        if np.any(observed_valid):
            ratio = (
                err[observed_valid]
                / observed_err[observed_valid]
            )

            fractional_difference = (
                ratio - 1.0
            )

            telescope_weight_diagnostics[
                telescope.name
            ] = {
                "weight_mode":
                    "model_flux_x_stored_err_mag",

                "median_expected_over_observed":
                    float(
                        np.median(
                            ratio
                        )
                    ),

                "max_abs_fractional_difference":
                    float(
                        np.max(
                            np.abs(
                                fractional_difference
                            )
                        )
                    ),

                "n_compared":
                    int(
                        np.sum(
                            observed_valid
                        )
                    ),
            }

        else:
            telescope_weight_diagnostics[
                telescope.name
            ] = {
                "weight_mode":
                    "model_flux_x_stored_err_mag",

                "median_expected_over_observed":
                    np.nan,

                "max_abs_fractional_difference":
                    np.nan,

                "n_compared":
                    0,
            }

        if (
            resolved_derivative_method
            == "five_point_stable_pair"
        ):

            physical_rows = []

            alternate_physical_rows = []

            telescope_diag = {}

            for name in phys:

                index = parameter_index[
                    name
                ]

                (
                    derivative,
                    alternate_derivative,
                    diagnostic,
                ) = (
                    stable_five_point_derivative(
                        model,
                        telescope,
                        vector,
                        index,
                        name,
                        pyparams,
                        step_multipliers=
                            step_multipliers,
                    )
                )

                physical_rows.append(
                    derivative
                )

                alternate_physical_rows.append(
                    alternate_derivative
                )

                telescope_diag[
                    name
                ] = diagnostic

            physical_rows = np.asarray(
                physical_rows,
                dtype=float,
            )

            alternate_physical_rows = np.asarray(
                alternate_physical_rows,
                dtype=float,
            )

        elif (
            resolved_derivative_method
            == "fspl_chain_rule"
        ):

            (
                physical_rows,
                alternate_physical_rows,
                telescope_diag,
            ) = (
                fspl_chain_rule_flux_derivatives(
                    model,
                    telescope,
                    pyparams,
                    phys,
                )
            )

        else:
            raise ValueError(
                "Unknown physical_derivative_method: "
                f"{resolved_derivative_method!r}"
            )

        flux_names = local_flux_names(
            model,
            telescope,
        )

        nuisance_rows = (
            _flux_derivatives(
                model,
                telescope,
                pyparams,
            )
        )

        if nuisance_rows.shape != (
            len(
                flux_names
            ),
            n,
        ):
            raise RuntimeError(
                f"{telescope.name}: "
                "unexpected nuisance "
                "Jacobian shape "
                f"{nuisance_rows.shape}"
            )

        physical_rows = (
            physical_rows
            / err[
                None,
                :,
            ]
        )

        alternate_physical_rows = (
            alternate_physical_rows
            / err[
                None,
                :,
            ]
        )

        nuisance_rows = (
            nuisance_rows
            / err[
                None,
                :,
            ]
        )

        block = np.zeros(
            (
                n,
                len(
                    names
                ),
            ),
            dtype=float,
        )

        block[
            :,
            :nphys,
        ] = (
            physical_rows.T
        )

        for local_row, name in enumerate(
            flux_names
        ):
            block[
                :,
                global_column[
                    name
                ],
            ] = nuisance_rows[
                local_row
            ]

        if not np.all(
            np.isfinite(
                block
            )
        ):
            raise RuntimeError(
                f"{telescope.name}: "
                "non-finite weighted Jacobian."
            )

        alternate_block = (
            block.copy()
        )

        alternate_block[
            :,
            :nphys,
        ] = (
            alternate_physical_rows.T
        )

        blocks.append(
            block
        )

        alternate_blocks.append(
            alternate_block
        )

        telescope_contributions[
            telescope.name
        ] = (
            block.T
            @ block
        )

        telescope_n_points[
            telescope.name
        ] = n

        derivative_diagnostics[
            telescope.name
        ] = telescope_diag

    J = np.vstack(
        blocks
    )

    J_alternate = np.vstack(
        alternate_blocks
    )

    F = (
        J.T
        @ J
    )

    Fsum = sum(
        telescope_contributions.values(),
        np.zeros_like(
            F
        ),
    )

    max_stability_l2 = 0.0

    max_stability_max = 0.0

    for telescope_diag in (
        derivative_diagnostics.values()
    ):
        for diagnostic in (
            telescope_diag.values()
        ):
            value = diagnostic[
                "pair_relative_l2"
            ]

            if np.isfinite(
                value
            ):
                max_stability_l2 = max(
                    max_stability_l2,
                    float(
                        value
                    ),
                )

            value = diagnostic[
                "pair_relative_max"
            ]

            if np.isfinite(
                value
            ):
                max_stability_max = max(
                    max_stability_max,
                    float(
                        value
                    ),
                )

    return {
        "physical_derivative_method":
            resolved_derivative_method,

        "jacobian":
            J,

        "jacobian_alternate":
            J_alternate,

        "fisher_full":
            F,

        "parameter_names":
            names,

        "physical_names":
            phys,

        "nuisance_names":
            nuisance,

        "telescope_contributions":
            telescope_contributions,

        "telescope_n_points":
            telescope_n_points,

        "telescope_weight_diagnostics":
            telescope_weight_diagnostics,

        "weight_mode":
            "model_flux_x_stored_err_mag",

        "derivative_diagnostics":
            derivative_diagnostics,

        "derivative_stability_max_rel_l2":
            max_stability_l2,

        "derivative_stability_max_rel_max":
            max_stability_max,

        "n_data":
            J.shape[0],

        "symmetry_max_abs":
            float(
                np.max(
                    np.abs(
                        F
                        - F.T
                    )
                )
            ),

        "additivity_max_abs":
            float(
                np.max(
                    np.abs(
                        F
                        - Fsum
                    )
                )
            ),
    }


# ======================================================================
# Flux marginalization by subspace projection
# ======================================================================


def _default_svd_tolerance(
    singular_values,
    shape,
):
    singular_values = np.asarray(
        singular_values,
        dtype=float,
    )

    if singular_values.size == 0:
        return 0.0

    return float(
        MACHINE_EPS
        * max(
            shape
        )
        * singular_values[0]
    )


def project_out_nuisance(
    Jphysical,
    Jnuisance,
):
    """
    Project Jphysical orthogonally to the column space
    of Jnuisance.

    For Gaussian least squares this is equivalent to
    marginalizing/profile-eliminating the linear nuisance
    parameters.
    """

    Jphysical = np.asarray(
        Jphysical,
        dtype=float,
    )

    Jnuisance = np.asarray(
        Jnuisance,
        dtype=float,
    )

    if Jnuisance.size == 0:
        return (
            Jphysical.copy(),
            {
                "nuisance_rank":
                    0,

                "nuisance_size":
                    0,

                "nuisance_full_rank":
                    True,

                "nuisance_singular_values":
                    np.asarray(
                        [],
                        dtype=float,
                    ),

                "nuisance_tolerance":
                    0.0,
            },
        )

    U, singular_values, _ = (
        np.linalg.svd(
            Jnuisance,
            full_matrices=False,
        )
    )

    tolerance = (
        _default_svd_tolerance(
            singular_values,
            Jnuisance.shape,
        )
    )

    rank = int(
        np.count_nonzero(
            singular_values
            > tolerance
        )
    )

    if rank:
        Q = U[
            :,
            :rank,
        ]

        projected = (
            Jphysical
            - Q
            @ (
                Q.T
                @ Jphysical
            )
        )

    else:
        projected = (
            Jphysical.copy()
        )

    return (
        projected,
        {
            "nuisance_rank":
                rank,

            "nuisance_size":
                Jnuisance.shape[1],

            "nuisance_full_rank":
                rank
                == Jnuisance.shape[1],

            "nuisance_singular_values":
                singular_values,

            "nuisance_tolerance":
                tolerance,
        },
    )


# ======================================================================
# Compatibility Schur complement helper
# ======================================================================


def marginalize_fluxes(
    F,
    nphys,
    rcond=1e-12,
):
    """
    Compatibility/testing implementation of the Schur complement.

    Production compute_fisher() does NOT use this to obtain
    the covariance.
    """

    F = np.asarray(
        F,
        dtype=float,
    )

    A = F[
        :nphys,
        :nphys,
    ]

    B = F[
        :nphys,
        nphys:,
    ]

    C = F[
        nphys:,
        nphys:,
    ]

    if C.size == 0:
        return (
            A.copy(),
            {
                "nuisance_rank":
                    0,

                "nuisance_size":
                    0,

                "nuisance_full_rank":
                    True,
            },
        )

    singular_values = (
        np.linalg.svd(
            C,
            compute_uv=False,
        )
    )

    tolerance = (
        rcond
        * singular_values[0]
        if singular_values.size
        else 0.0
    )

    rank = int(
        np.count_nonzero(
            singular_values
            > tolerance
        )
    )

    Cinv = np.linalg.pinv(
        C,
        rcond=rcond,
        hermitian=True,
    )

    Fphys = (
        A
        - B
        @ Cinv
        @ B.T
    )

    Fphys = (
        0.5
        * (
            Fphys
            + Fphys.T
        )
    )

    return (
        Fphys,
        {
            "nuisance_rank":
                rank,

            "nuisance_size":
                C.shape[0],

            "nuisance_full_rank":
                rank
                == C.shape[0],
        },
    )


# ======================================================================
# Local physical coordinate scaling
# ======================================================================


def scaled_coordinates(
    names,
    pyparams,
    log_parameters=(
        "tE",
        "rho",
    ),
    scale_t0_by_tE=True,
):
    """
    Return local coordinate names and D = dx/dq.

    FSPL example:

        q_t0  = delta(t0)/tE
        q_u0  = u0
        q_tE  = delta(log tE)
        q_rho = delta(log rho)
        q_piN = piEN
        q_piE = piEE

    so locally

        x = x0 + D q.
    """

    q_names = list(
        names
    )

    scales = np.ones(
        len(
            names
        ),
        dtype=float,
    )

    if (
        scale_t0_by_tE
        and "t0" in names
        and "tE" in names
    ):
        tE = _get(
            pyparams,
            "tE",
        )

        if tE <= 0:
            raise ValueError(
                f"tE must be >0, got {tE}"
            )

        i = names.index(
            "t0"
        )

        scales[
            i
        ] = tE

        q_names[
            i
        ] = "dt0_over_tE"

    for name in log_parameters:

        if name not in names:
            continue

        value = _get(
            pyparams,
            name,
        )

        if value <= 0:
            raise ValueError(
                "Cannot log-scale "
                f"{name}={value}"
            )

        i = names.index(
            name
        )

        scales[
            i
        ] = value

        q_names[
            i
        ] = (
            f"log_{name}"
        )

    return (
        q_names,
        np.diag(
            scales
        ),
    )


# ======================================================================
# SVD of marginalized physical Jacobian
# ======================================================================


def decompose_jacobian_svd(
    J,
):
    J = np.asarray(
        J,
        dtype=float,
    )

    U, singular_values, Vh = (
        np.linalg.svd(
            J,
            full_matrices=False,
        )
    )

    tolerance = (
        _default_svd_tolerance(
            singular_values,
            J.shape,
        )
    )

    identified = (
        singular_values
        > tolerance
    )

    rank = int(
        np.count_nonzero(
            identified
        )
    )

    nparams = J.shape[1]

    covariance = np.zeros(
        (
            nparams,
            nparams,
        ),
        dtype=float,
    )

    if rank:
        Vr = (
            Vh[
                identified,
                :
            ].T
        )

        sr = singular_values[
            identified
        ]

        covariance = (
            Vr
            @ np.diag(
                1.0
                / (
                    sr
                    * sr
                )
            )
            @ Vr.T
        )

        condition_number = float(
            sr[0]
            / sr[-1]
        )

    else:
        condition_number = (
            np.inf
        )

    null_basis = (
        Vh[
            ~identified,
            :
        ].T
    )

    if (
        singular_values.size
        and singular_values[-1] > 0
    ):
        raw_condition_number = float(
            singular_values[0]
            / singular_values[-1]
        )

        singular_value_ratio = float(
            singular_values[-1]
            / singular_values[0]
        )

    else:
        raw_condition_number = (
            np.inf
        )

        singular_value_ratio = (
            0.0
        )

    # Purely numerical warning:
    # beyond this scale, squaring the condition number
    # to form Fisher approaches 1/eps.
    numerical_condition_warning = bool(
        raw_condition_number
        >
        1.0
        / np.sqrt(
            MACHINE_EPS
        )
    )

    return {
        "U":
            U,

        "singular_values":
            singular_values,

        "Vh":
            Vh,

        "rank":
            rank,

        "size":
            nparams,

        "full_rank":
            rank
            == nparams,

        "tolerance":
            tolerance,

        "condition_number":
            condition_number,

        "raw_condition_number":
            raw_condition_number,

        "singular_value_ratio":
            singular_value_ratio,

        "numerical_condition_warning":
            numerical_condition_warning,

        "covariance_pinv":
            covariance,

        "null_basis":
            null_basis,
    }


def svd_solution_stability(
    J_primary,
    decomposition_primary,
    J_alternate,
    decomposition_alternate,
    sigma_primary,
    sigma_alternate,
):
    """
    Compare two Fisher/SVD solutions obtained from the two members
    of the locally most stable finite-difference pair.

    No new model evaluations are required.

    The returned quantities diagnose sensitivity of the actual
    inferred weak direction, rather than comparing a generic
    derivative error with an arbitrary rcond.
    """

    J_primary = np.asarray(
        J_primary,
        dtype=float,
    )

    J_alternate = np.asarray(
        J_alternate,
        dtype=float,
    )

    denominator = max(
        np.linalg.norm(
            J_primary
        ),
        1e-300,
    )

    jacobian_rel = float(
        np.linalg.norm(
            J_alternate
            - J_primary
        )
        / denominator
    )

    s_primary = np.asarray(
        decomposition_primary[
            "singular_values"
        ],
        dtype=float,
    )

    s_alternate = np.asarray(
        decomposition_alternate[
            "singular_values"
        ],
        dtype=float,
    )

    if (
        s_primary.size
        and s_alternate.size
    ):
        smin_primary = float(
            s_primary[-1]
        )

        smin_alternate = float(
            s_alternate[-1]
        )

        smin_rel = float(
            abs(
                smin_alternate
                - smin_primary
            )
            / max(
                abs(
                    smin_primary
                ),
                1e-300,
            )
        )

        v_primary = np.asarray(
            decomposition_primary[
                "Vh"
            ][-1],
            dtype=float,
        )

        v_alternate = np.asarray(
            decomposition_alternate[
                "Vh"
            ][-1],
            dtype=float,
        )

        norm_product = (
            np.linalg.norm(
                v_primary
            )
            * np.linalg.norm(
                v_alternate
            )
        )

        if norm_product > 0:
            cosine = abs(
                float(
                    np.dot(
                        v_primary,
                        v_alternate,
                    )
                    / norm_product
                )
            )

            cosine = float(
                np.clip(
                    cosine,
                    0.0,
                    1.0,
                )
            )

            weak_mode_angle_deg = float(
                np.degrees(
                    np.arccos(
                        cosine
                    )
                )
            )

        else:
            weak_mode_angle_deg = np.nan

    else:
        smin_primary = np.nan
        smin_alternate = np.nan
        smin_rel = np.nan
        weak_mode_angle_deg = np.nan

    if (
        np.isfinite(
            sigma_primary
        )
        and np.isfinite(
            sigma_alternate
        )
    ):
        sigma_rel = float(
            abs(
                sigma_alternate
                - sigma_primary
            )
            / max(
                abs(
                    sigma_primary
                ),
                1e-300,
            )
        )

    elif (
        np.isinf(
            sigma_primary
        )
        and np.isinf(
            sigma_alternate
        )
    ):
        sigma_rel = 0.0

    else:
        sigma_rel = np.inf

    return {
        "fd_stability_jacobian_rel":
            jacobian_rel,

        "fd_stability_smin_primary":
            smin_primary,

        "fd_stability_smin_alternate":
            smin_alternate,

        "fd_stability_smin_rel":
            smin_rel,

        "fd_stability_weak_mode_angle_deg":
            weak_mode_angle_deg,

        "fd_stability_sigma_piE_rel":
            sigma_rel,
    }



# ======================================================================
# Compatibility eigendecomposition helper used by old tests/audits
# ======================================================================


def decompose_fisher(
    F,
    rcond=1e-12,
):
    """
    Legacy diagnostic helper.

    Production compute_fisher() uses decompose_jacobian_svd()
    instead.
    """

    F = np.asarray(
        F,
        dtype=float,
    )

    F = (
        0.5
        * (
            F
            + F.T
        )
    )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            F
        )
    )

    threshold = (
        rcond
        * np.max(
            np.abs(
                eigenvalues
            )
        )
        if eigenvalues.size
        else 0.0
    )

    identified = (
        eigenvalues
        > threshold
    )

    rank = int(
        np.count_nonzero(
            identified
        )
    )

    covariance = np.zeros_like(
        F
    )

    if rank:
        V = eigenvectors[
            :,
            identified,
        ]

        w = eigenvalues[
            identified
        ]

        covariance = (
            V
            @ np.diag(
                1.0
                / w
            )
            @ V.T
        )

        condition_number = float(
            w.max()
            / w.min()
        )

    else:
        condition_number = (
            np.inf
        )

    return {
        "eigenvalues":
            eigenvalues,

        "rank":
            rank,

        "size":
            F.shape[0],

        "full_rank":
            rank
            == F.shape[0],

        "condition_number":
            condition_number,

        "covariance_pinv":
            covariance,

        "null_basis":
            eigenvectors[
                :,
                ~identified,
            ],

        "threshold":
            threshold,

        "has_significant_negative_eigenvalue":
            bool(
                np.any(
                    eigenvalues
                    < -threshold
                )
            ),
    }


def functional_sigma(
    gradient_q,
    decomposition,
    null_projection_tolerance=1e-8,
):
    """
    Standard deviation of a locally linear scalar functional.

    If its gradient has a component in a true numerical
    null-space direction, return infinity.
    """

    gradient = np.asarray(
        gradient_q,
        dtype=float,
    )

    null_basis = decomposition[
        "null_basis"
    ]

    if null_basis.size:

        projection = (
            null_basis.T
            @ gradient
        )

        if (
            np.linalg.norm(
                projection
            )
            >
            null_projection_tolerance
            * max(
                1.0,
                np.linalg.norm(
                    gradient
                ),
            )
        ):
            return np.inf

    covariance = decomposition[
        "covariance_pinv"
    ]

    variance = float(
        gradient
        @ covariance
        @ gradient
    )

    scale = max(
        1.0,
        np.linalg.norm(
            covariance
        ),
    )

    if (
        variance < 0
        and abs(
            variance
        )
        <
        1e-12
        * scale
    ):
        variance = 0.0

    if variance < 0:
        return np.nan

    return float(
        np.sqrt(
            variance
        )
    )


# ======================================================================
# Main Fisher calculation
# ======================================================================


def compute_fisher(
    model,
    pyparams,
    *,
    rcond=None,
    step_multipliers=
        DEFAULT_STEP_MULTIPLIERS,
    physical_derivative_method=
        "auto",
):
    """
    Compute local Fisher information at the supplied truth point.

    `rcond` is accepted only for backwards API compatibility.
    It is deliberately NOT used to decide the physical rank.

    Rank/covariance come from the SVD of the marginalized,
    scaled physical Jacobian using a machine-precision rank
    tolerance.
    """

    out = weighted_jacobian(
        model,
        pyparams,
        step_multipliers=
            step_multipliers,
        physical_derivative_method=
            physical_derivative_method,
    )

    names = out[
        "physical_names"
    ]

    nphys = len(
        names
    )

    Jfull = out[
        "jacobian"
    ]

    Jfull_alternate = out[
        "jacobian_alternate"
    ]

    Jphysical = (
        Jfull[
            :,
            :nphys,
        ]
    )

    Jphysical_alternate = (
        Jfull_alternate[
            :,
            :nphys,
        ]
    )

    Jnuisance = (
        Jfull[
            :,
            nphys:,
        ]
    )

    (
        Jmarg,
        nuisance_diag,
    ) = project_out_nuisance(
        Jphysical,
        Jnuisance,
    )

    (
        Jmarg_alternate,
        nuisance_diag_alternate,
    ) = project_out_nuisance(
        Jphysical_alternate,
        Jnuisance,
    )

    if (
        nuisance_diag_alternate[
            "nuisance_rank"
        ]
        != nuisance_diag[
            "nuisance_rank"
        ]
    ):
        raise RuntimeError(
            "Primary and alternate nuisance projection "
            "produced different ranks."
        )

    (
        q_names,
        D,
    ) = scaled_coordinates(
        names,
        pyparams,
    )

    Jq = (
        Jmarg
        @ D
    )

    Jq_alternate = (
        Jmarg_alternate
        @ D
    )

    decomposition = (
        decompose_jacobian_svd(
            Jq
        )
    )

    decomposition_alternate = (
        decompose_jacobian_svd(
            Jq_alternate
        )
    )

    # Returned only for diagnostics/output.
    Fnative = (
        Jmarg.T
        @ Jmarg
    )

    Fq = (
        Jq.T
        @ Jq
    )

    Fnative = (
        0.5
        * (
            Fnative
            + Fnative.T
        )
    )

    Fq = (
        0.5
        * (
            Fq
            + Fq.T
        )
    )

    covariance_q = decomposition[
        "covariance_pinv"
    ]

    covariance_native = (
        D
        @ covariance_q
        @ D.T
    )

    sigma = {}

    fractional_sigma = {}

    for i, name in enumerate(
        names
    ):
        gradient_q = np.zeros(
            nphys,
            dtype=float,
        )

        gradient_q[
            i
        ] = D[
            i,
            i,
        ]

        value_sigma = (
            functional_sigma(
                gradient_q,
                decomposition,
            )
        )

        sigma[
            name
        ] = value_sigma

        value = _get(
            pyparams,
            name,
        )

        if np.isinf(
            value_sigma
        ):
            fractional_sigma[
                name
            ] = np.inf

        elif (
            np.isfinite(
                value_sigma
            )
            and value != 0
        ):
            fractional_sigma[
                name
            ] = (
                value_sigma
                / abs(
                    value
                )
            )

        else:
            fractional_sigma[
                name
            ] = np.nan

    piE = np.nan

    sigma_piE = np.nan

    fractional_sigma_piE = np.nan

    sigma_piE_alternate = np.nan

    fractional_sigma_piE_alternate = np.nan

    piE_gradient_q = None

    if (
        "piEN" in names
        and "piEE" in names
    ):
        piEN = _get(
            pyparams,
            "piEN",
        )

        piEE = _get(
            pyparams,
            "piEE",
        )

        piE = float(
            np.hypot(
                piEN,
                piEE,
            )
        )

        if piE > 0:

            gradient_native = np.zeros(
                nphys,
                dtype=float,
            )

            gradient_native[
                names.index(
                    "piEN"
                )
            ] = (
                piEN
                / piE
            )

            gradient_native[
                names.index(
                    "piEE"
                )
            ] = (
                piEE
                / piE
            )

            gradient_q = (
                D.T
                @ gradient_native
            )

            piE_gradient_q = (
                gradient_q.copy()
            )

            sigma_piE = (
                functional_sigma(
                    gradient_q,
                    decomposition,
                )
            )

            sigma_piE_alternate = (
                functional_sigma(
                    gradient_q,
                    decomposition_alternate,
                )
            )

            fractional_sigma_piE = (
                sigma_piE
                / piE
                if np.isfinite(
                    sigma_piE
                )
                else sigma_piE
            )

            fractional_sigma_piE_alternate = (
                sigma_piE_alternate
                / piE
                if np.isfinite(
                    sigma_piE_alternate
                )
                else sigma_piE_alternate
            )

    stability = svd_solution_stability(
        Jq,
        decomposition,
        Jq_alternate,
        decomposition_alternate,
        sigma_piE,
        sigma_piE_alternate,
    )

    eigenvalues_q = np.linalg.eigvalsh(
        Fq
    )

    eigen_scale = max(
        1.0,
        float(
            np.max(
                np.abs(
                    eigenvalues_q
                )
            )
        ),
    )

    negative_tolerance = (
        100.0
        * MACHINE_EPS
        * eigen_scale
        * max(
            Fq.shape
        )
    )

    return {
        "method":
            (
                "five_point_stable_pair"
                if out[
                    "physical_derivative_method"
                ]
                == "five_point_stable_pair"
                else
                "fspl_chain_rule_local_five_point"
            )
            + "+analytic_flux"
            + "+nuisance_projection"
            + "+svd",

        "physical_derivative_method":
            out[
                "physical_derivative_method"
            ],

        "physical_names":
            names,

        "q_names":
            q_names,

        "nuisance_names":
            out[
                "nuisance_names"
            ],

        "jacobian_full":
            Jfull,

        "jacobian_physical_marginalized":
            Jmarg,

        "jacobian_physical_q":
            Jq,

        "jacobian_physical_q_alternate":
            Jq_alternate,

        "fisher_full":
            out[
                "fisher_full"
            ],

        "fisher_physical_native":
            Fnative,

        "fisher_physical_q":
            Fq,

        "covariance_q_pinv":
            covariance_q,

        "covariance_native_pinv":
            covariance_native,

        "singular_values_q":
            decomposition[
                "singular_values"
            ],

        "singular_values_q_alternate":
            decomposition_alternate[
                "singular_values"
            ],

        "svd_tolerance":
            decomposition[
                "tolerance"
            ],

        "rank":
            decomposition[
                "rank"
            ],

        "size":
            decomposition[
                "size"
            ],

        "full_rank":
            decomposition[
                "full_rank"
            ],

        "condition_number":
            decomposition[
                "condition_number"
            ],

        "raw_condition_number":
            decomposition[
                "raw_condition_number"
            ],

        "singular_value_ratio":
            decomposition[
                "singular_value_ratio"
            ],

        "numerical_condition_warning":
            decomposition[
                "numerical_condition_warning"
            ],

        "eigenvalues_q":
            eigenvalues_q,

        "has_significant_negative_eigenvalue":
            bool(
                np.any(
                    eigenvalues_q
                    < -negative_tolerance
                )
            ),

        "sigma":
            sigma,

        "fractional_sigma":
            fractional_sigma,

        "piE":
            piE,

        "sigma_piE":
            sigma_piE,

        "fractional_sigma_piE":
            fractional_sigma_piE,

        "sigma_piE_alternate":
            sigma_piE_alternate,

        "fractional_sigma_piE_alternate":
            fractional_sigma_piE_alternate,

        **stability,

        "n_data":
            out[
                "n_data"
            ],

        "telescope_n_points":
            out[
                "telescope_n_points"
            ],

        "weight_mode":
            out[
                "weight_mode"
            ],

        "telescope_weight_diagnostics":
            out[
                "telescope_weight_diagnostics"
            ],

        "telescope_contributions":
            out[
                "telescope_contributions"
            ],

        "symmetry_max_abs":
            out[
                "symmetry_max_abs"
            ],

        "additivity_max_abs":
            out[
                "additivity_max_abs"
            ],

        "derivative_diagnostics":
            out[
                "derivative_diagnostics"
            ],

        "derivative_stability_max_rel_l2":
            out[
                "derivative_stability_max_rel_l2"
            ],

        "derivative_stability_max_rel_max":
            out[
                "derivative_stability_max_rel_max"
            ],

        **nuisance_diag,
    }


# ======================================================================
# E1 three-configuration suite
# ======================================================================


def _gain(
    before,
    after,
):
    if (
        np.isnan(
            before
        )
        or np.isnan(
            after
        )
    ):
        return np.nan

    if (
        np.isinf(
            before
        )
        and np.isinf(
            after
        )
    ):
        return np.nan

    if after == 0:
        return np.inf

    return float(
        before
        / after
    )


def telescope_sets(
    model,
    min_rubin_points=5,
):
    roman = []

    rubin = []

    for telescope in (
        model.event.telescopes
    ):

        if (
            telescope.lightcurve
            is None
            or len(
                telescope.lightcurve
            )
            == 0
        ):
            continue

        if is_roman(
            telescope
        ):
            roman.append(
                telescope.name
            )

        elif (
            len(
                telescope.lightcurve
            )
            >= min_rubin_points
        ):
            rubin.append(
                telescope.name
            )

    if len(
        roman
    ) != 1:
        raise RuntimeError(
            "E1 expects exactly one "
            "Roman telescope; "
            f"found {roman}"
        )

    return (
        roman,
        rubin,
    )


def compute_e1_suite(
    true_model,
    true_params,
    *,
    min_rubin_points=5,
    rcond=None,
    step_multipliers=
        DEFAULT_STEP_MULTIPLIERS,
    physical_derivative_method=
        "auto",
):
    """
    Compute E1 for:

      R
      RR
      RR_DPERP0
    """

    (
        roman,
        rubin,
    ) = telescope_sets(
        true_model,
        min_rubin_points=
            min_rubin_points,
    )

    rr = (
        roman
        + rubin
    )

    specs = {
        "R":
            (
                roman,
                False,
            ),

        "RR":
            (
                rr,
                False,
            ),

        "RR_DPERP0":
            (
                rr,
                True,
            ),
    }

    results = {}

    for (
        label,
        (
            names,
            roman_at_earth,
        ),
    ) in specs.items():

        (
            model,
            params,
        ) = rebuild_model(
            true_model,
            true_params,
            names,
            roman_at_earth=
                roman_at_earth,
        )

        result = compute_fisher(
            model,
            params,
            rcond=rcond,
            step_multipliers=
                step_multipliers,
            physical_derivative_method=
                physical_derivative_method,
        )

        result[
            "telescope_names"
        ] = list(
            names
        )

        result[
            "roman_at_earth"
        ] = (
            roman_at_earth
        )

        results[
            label
        ] = result

    gains = {
        "R_to_RR":
            {},

        "R_to_RR_DPERP0":
            {},

        "RR_DPERP0_to_RR":
            {},
    }

    for name in results[
        "R"
    ][
        "physical_names"
    ]:

        sR = results[
            "R"
        ][
            "sigma"
        ].get(
            name,
            np.nan,
        )

        sRR = results[
            "RR"
        ][
            "sigma"
        ].get(
            name,
            np.nan,
        )

        sE = results[
            "RR_DPERP0"
        ][
            "sigma"
        ].get(
            name,
            np.nan,
        )

        gains[
            "R_to_RR"
        ][
            name
        ] = _gain(
            sR,
            sRR,
        )

        gains[
            "R_to_RR_DPERP0"
        ][
            name
        ] = _gain(
            sR,
            sE,
        )

        gains[
            "RR_DPERP0_to_RR"
        ][
            name
        ] = _gain(
            sE,
            sRR,
        )

    gains[
        "R_to_RR"
    ][
        "piE"
    ] = _gain(
        results[
            "R"
        ][
            "sigma_piE"
        ],
        results[
            "RR"
        ][
            "sigma_piE"
        ],
    )

    gains[
        "R_to_RR_DPERP0"
    ][
        "piE"
    ] = _gain(
        results[
            "R"
        ][
            "sigma_piE"
        ],
        results[
            "RR_DPERP0"
        ][
            "sigma_piE"
        ],
    )

    gains[
        "RR_DPERP0_to_RR"
    ][
        "piE"
    ] = _gain(
        results[
            "RR_DPERP0"
        ][
            "sigma_piE"
        ],
        results[
            "RR"
        ][
            "sigma_piE"
        ],
    )

    return {
        "configurations":
            results,

        "gains":
            gains,

        "min_rubin_points":
            min_rubin_points,

        "roman_telescope_names":
            roman,

        "rubin_telescope_names_used":
            rubin,
    }


# ======================================================================
# Compact production row
# ======================================================================


def compact_summary(
    suite,
):
    row = {
        "fisher_method":
            suite[
                "configurations"
            ][
                "R"
            ][
                "method"
            ],

        "min_rubin_points":
            suite[
                "min_rubin_points"
            ],

        "roman_telescope_names":
            suite[
                "roman_telescope_names"
            ],

        "rubin_telescope_names_used":
            suite[
                "rubin_telescope_names_used"
            ],
    }

    for (
        label,
        result,
    ) in suite[
        "configurations"
    ].items():

        prefix = (
            label.lower()
        )

        row[
            f"{prefix}_n_data"
        ] = result[
            "n_data"
        ]

        row[
            f"{prefix}_rank"
        ] = result[
            "rank"
        ]

        row[
            f"{prefix}_size"
        ] = result[
            "size"
        ]

        row[
            f"{prefix}_full_rank"
        ] = result[
            "full_rank"
        ]

        row[
            f"{prefix}_weight_mode"
        ] = result[
            "weight_mode"
        ]

        for (
            telescope_name,
            weight_diag,
        ) in result[
            "telescope_weight_diagnostics"
        ].items():

            safe_telescope_name = "".join(
                character
                if character.isalnum()
                else "_"
                for character in str(
                    telescope_name
                )
            )

            for diagnostic_name in (
                "median_expected_over_observed",
                "max_abs_fractional_difference",
                "n_compared",
            ):
                row[
                    (
                        f"{prefix}_weight_"
                        f"{safe_telescope_name}_"
                        f"{diagnostic_name}"
                    )
                ] = weight_diag[
                    diagnostic_name
                ]

        row[
            f"{prefix}_condition_number"
        ] = result[
            "condition_number"
        ]

        row[
            f"{prefix}_raw_condition_number"
        ] = result[
            "raw_condition_number"
        ]

        row[
            f"{prefix}_singular_value_ratio"
        ] = result[
            "singular_value_ratio"
        ]

        row[
            f"{prefix}_numerical_condition_warning"
        ] = result[
            "numerical_condition_warning"
        ]

        row[
            f"{prefix}_derivative_stability_max_rel_l2"
        ] = result[
            "derivative_stability_max_rel_l2"
        ]

        row[
            f"{prefix}_derivative_stability_max_rel_max"
        ] = result[
            "derivative_stability_max_rel_max"
        ]

        row[
            f"{prefix}_sigma_piE"
        ] = result[
            "sigma_piE"
        ]

        row[
            f"{prefix}_fractional_sigma_piE"
        ] = result[
            "fractional_sigma_piE"
        ]

        row[
            f"{prefix}_sigma_piE_alternate"
        ] = result[
            "sigma_piE_alternate"
        ]

        for stability_name in (
            "fd_stability_sigma_piE_rel",
            "fd_stability_smin_rel",
            "fd_stability_weak_mode_angle_deg",
            "fd_stability_jacobian_rel",
        ):
            row[
                f"{prefix}_{stability_name}"
            ] = result[
                stability_name
            ]

        for (
            name,
            value,
        ) in result[
            "sigma"
        ].items():

            row[
                f"{prefix}_sigma_{name}"
            ] = value

        for (
            name,
            value,
        ) in result[
            "fractional_sigma"
        ].items():

            row[
                f"{prefix}_fractional_sigma_{name}"
            ] = value

    for (
        comparison,
        values,
    ) in suite[
        "gains"
    ].items():

        for (
            name,
            value,
        ) in values.items():

            row[
                f"gain_{comparison}_{name}"
            ] = value

    return row
