#!/usr/bin/env python3

import copy
import json
from pathlib import Path

import numpy as np

from launch_catalog_simfit_from_config import configure_environment


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "config_files"
    / "gbtds_300_v1_amax_v1.json"
)

config = json.loads(
    CONFIG.read_text()
)

resolved_paths = configure_environment(
    config.get("paths", {}),
    CONFIG.parent,
)


from catalog.event_reader import load_event_from_catalog

from functions_roman_rubin import simulate_event_for_fit

from analysis.fisher_information import (
    rebuild_model,
    parameter_names,
    physical_names,
    local_flux_names,
    weighted_jacobian,
    marginalize_fluxes,
    scaled_coordinates,
    decompose_fisher,
    functional_sigma,
)


CATALOG = Path(
    config["input"]["event_catalog"]
).expanduser().resolve()

EPHEMERIDES = Path(
    resolved_paths["roman_ephemerides"]
).expanduser().resolve()


CASES = [
    ("FFP", 10, "FSPL"),
    ("BH", 308, "FSPL"),
]


RCONDS = [
    1e-9,
    1e-10,
    1e-11,
    1e-12,
    1e-13,
    1e-14,
    1e-15,
]


# Multipliers around a parameter-dependent reference step.
STEP_MULTIPLIERS = [
    10.0,
    3.0,
    1.0,
    0.3,
    0.1,
]


def as_array(x):
    if hasattr(x, "value"):
        x = x.value

    return np.asarray(
        x,
        dtype=float,
    )


def get_value(params, name):
    try:
        return float(params[name])
    except Exception:
        return float(
            getattr(
                params,
                name,
            )
        )


def vector_from_params(model, params):
    return np.asarray(
        [
            get_value(
                params,
                name,
            )
            for name in parameter_names(model)
        ],
        dtype=float,
    )


def base_step(name, value, params):
    """
    Reference finite-difference step.

    The audit scans factors around this value, so this does not
    assume that one single step is correct.
    """

    value = float(value)

    if name == "t0":
        tE = abs(
            get_value(
                params,
                "tE",
            )
        )

        return 1e-5 * max(
            tE,
            0.1,
        )

    if name == "tE":
        return 1e-5 * max(
            abs(value),
            0.1,
        )

    if name == "u0":
        return 1e-5 * max(
            abs(value),
            0.01,
        )

    if name == "rho":
        return 1e-5 * max(
            abs(value),
            1e-5,
        )

    if name in (
        "piEN",
        "piEE",
    ):
        return 1e-5 * max(
            abs(value),
            1e-3,
        )

    # flux parameters
    return 1e-5 * max(
        abs(value),
        1.0,
    )


def model_flux(
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

    return as_array(
        flux
    )


def five_point_derivative(
    model,
    telescope,
    vector,
    index,
    h,
):
    """
    Five-point central derivative:

       [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / 12h
    """

    vectors = []

    for multiplier in (
        2.0,
        1.0,
        -1.0,
        -2.0,
    ):
        v = np.asarray(
            vector,
            dtype=float,
        ).copy()

        v[index] += (
            multiplier
            * h
        )

        vectors.append(v)

    fp2 = model_flux(
        model,
        telescope,
        vectors[0],
    )

    fp1 = model_flux(
        model,
        telescope,
        vectors[1],
    )

    fm1 = model_flux(
        model,
        telescope,
        vectors[2],
    )

    fm2 = model_flux(
        model,
        telescope,
        vectors[3],
    )

    return (
        -fp2
        + 8.0 * fp1
        - 8.0 * fm1
        + fm2
    ) / (
        12.0 * h
    )


def comparison_metrics(
    reference,
    candidate,
):
    reference = np.asarray(
        reference,
        dtype=float,
    )

    candidate = np.asarray(
        candidate,
        dtype=float,
    )

    diff = (
        candidate
        - reference
    )

    nr = np.linalg.norm(
        reference
    )

    nc = np.linalg.norm(
        candidate
    )

    scale_l2 = max(
        nr,
        nc,
        1e-300,
    )

    max_scale = max(
        float(
            np.max(
                np.abs(
                    reference
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    candidate
                )
            )
        ),
        1e-300,
    )

    if (
        nr > 0
        and nc > 0
    ):
        cosine = float(
            np.dot(
                reference,
                candidate,
            )
            / (
                nr
                * nc
            )
        )
    else:
        cosine = np.nan

    return {
        "relative_l2":
            float(
                np.linalg.norm(
                    diff
                )
                / scale_l2
            ),

        "relative_max":
            float(
                np.max(
                    np.abs(
                        diff
                    )
                )
                / max_scale
            ),

        "cosine":
            cosine,
    }


def audit_jacobian(
    model,
    params,
    label,
):
    print()
    print(
        "=" * 100
    )
    print(
        "JACOBIAN AUDIT:",
        label,
    )
    print(
        "=" * 100
    )

    global_names = parameter_names(
        model
    )

    phys_names = physical_names(
        model
    )

    name_to_index = {
        name: i
        for i, name in enumerate(
            global_names
        )
    }

    vector = vector_from_params(
        model,
        params,
    )

    for telescope in model.event.telescopes:

        if telescope.lightcurve is None:
            continue

        print()
        print(
            "-" * 100
        )
        print(
            "TELESCOPE:",
            telescope.name,
            "N =",
            len(
                telescope.lightcurve
            ),
        )
        print(
            "-" * 100
        )

        py_jac = np.asarray(
            model.photometric_model_Jacobian(
                telescope,
                copy.deepcopy(
                    params
                ),
            ),
            dtype=float,
        )

        names = (
            phys_names
            + local_flux_names(
                model,
                telescope,
            )
        )

        if py_jac.shape == (
            len(
                telescope.lightcurve
            ),
            len(names),
        ):
            py_jac = py_jac.T

        expected = (
            len(names),
            len(
                telescope.lightcurve
            ),
        )

        assert (
            py_jac.shape
            == expected
        ), (
            py_jac.shape,
            expected,
        )

        for row, name in enumerate(
            names
        ):
            index = name_to_index[
                name
            ]

            value = vector[
                index
            ]

            href = base_step(
                name,
                value,
                params,
            )

            best = None

            print()
            print(
                f"PARAMETER {name}"
            )

            print(
                "  value =",
                value,
            )

            print(
                "  base h =",
                href,
            )

            for factor in STEP_MULTIPLIERS:

                h = (
                    href
                    * factor
                )

                fd = five_point_derivative(
                    model,
                    telescope,
                    vector,
                    index,
                    h,
                )

                metrics = comparison_metrics(
                    py_jac[row],
                    fd,
                )

                print(
                    "  factor={:4.1f} "
                    "h={:.6e} "
                    "relL2={:.6e} "
                    "relMax={:.6e} "
                    "cos={:.12f}".format(
                        factor,
                        h,
                        metrics[
                            "relative_l2"
                        ],
                        metrics[
                            "relative_max"
                        ],
                        metrics[
                            "cosine"
                        ],
                    )
                )

                if (
                    best is None
                    or metrics[
                        "relative_l2"
                    ]
                    < best[
                        "relative_l2"
                    ]
                ):
                    best = {
                        **metrics,
                        "factor":
                            factor,
                        "h":
                            h,
                    }

            print(
                "  BEST:",
                best,
            )

            assert np.isfinite(
                best[
                    "relative_l2"
                ]
            )

            assert np.isfinite(
                best[
                    "relative_max"
                ]
            )


def sigma_piE_for_rcond(
    names,
    params,
    Fnative,
    rcond,
):
    (
        q_names,
        D,
    ) = scaled_coordinates(
        names,
        params,
    )

    Fq = (
        D.T
        @ Fnative
        @ D
    )

    Fq = (
        0.5
        * (
            Fq
            + Fq.T
        )
    )

    dec = decompose_fisher(
        Fq,
        rcond=rcond,
    )

    piEN = get_value(
        params,
        "piEN",
    )

    piEE = get_value(
        params,
        "piEE",
    )

    piE = float(
        np.hypot(
            piEN,
            piEE,
        )
    )

    gradient = np.zeros(
        len(names),
        dtype=float,
    )

    gradient[
        names.index(
            "piEN"
        )
    ] = (
        piEN
        / piE
    )

    gradient[
        names.index(
            "piEE"
        )
    ] = (
        piEE
        / piE
    )

    sigma = functional_sigma(
        gradient,
        dec,
    )

    eig = np.asarray(
        dec[
            "eigenvalues"
        ],
        dtype=float,
    )

    return {
        "rank":
            dec[
                "rank"
            ],

        "size":
            dec[
                "size"
            ],

        "threshold":
            dec[
                "threshold"
            ],

        "lambda_min":
            float(
                eig.min()
            ),

        "lambda_max":
            float(
                eig.max()
            ),

        "sigma_piE":
            sigma,

        "fractional_sigma_piE":
            (
                sigma
                / piE
                if np.isfinite(
                    sigma
                )
                else sigma
            ),
    }


def audit_rcond(
    model,
    params,
    label,
):
    print()
    print(
        "=" * 100
    )
    print(
        "RCOND AUDIT:",
        label,
    )
    print(
        "=" * 100
    )

    jac = weighted_jacobian(
        model,
        params,
    )

    names = jac[
        "physical_names"
    ]

    nphys = len(
        names
    )

    # Hold nuisance inversion fixed at a strict tolerance.
    #
    # We want to isolate sensitivity of the PHYSICAL Fisher
    # rank criterion from sensitivity of the flux Schur complement.
    Fnative, nuisance = marginalize_fluxes(
        jac[
            "fisher_full"
        ],
        nphys,
        rcond=1e-14,
    )

    print(
        "nuisance rank / size =",
        nuisance[
            "nuisance_rank"
        ],
        "/",
        nuisance[
            "nuisance_size"
        ],
    )

    for rcond in RCONDS:

        out = sigma_piE_for_rcond(
            names,
            params,
            Fnative,
            rcond,
        )

        print(
            "rcond={:.0e} "
            "rank={}/{} "
            "threshold={:.6e} "
            "lambda_min={:.6e} "
            "sigma_piE={} "
            "frac_sigma_piE={}".format(
                rcond,
                out[
                    "rank"
                ],
                out[
                    "size"
                ],
                out[
                    "threshold"
                ],
                out[
                    "lambda_min"
                ],
                out[
                    "sigma_piE"
                ],
                out[
                    "fractional_sigma_piE"
                ],
            )
        )


def run_case(
    population,
    index,
    model_name,
):
    print()
    print(
        "#" * 110
    )
    print(
        population,
        "catalog_event_index =",
        index,
    )
    print(
        "#" * 110
    )

    event = load_event_from_catalog(
        CATALOG,
        index,
        expected_population=population,
    )

    seed = int(
        event[
            "simulation_seed"
        ]
    )

    (
        true_model,
        truth,
        decision,
    ) = simulate_event_for_fit(
        seed,
        event,
        str(
            EPHEMERIDES
        ),
        model_name,
        time_window=None,
        use_roman=True,
        use_rubin=True,
        truth_parallax=True,
        rubin_pointing_mode="source",
        rubin_cache_cell_deg=None,
        apply_detection_criteria=True,
        apply_photometric_filter=True,
        rubin_saturation_mag=None,
        roman_saturation_mag=None,
    )

    assert decision is True
    assert true_model is not None

    # ------------------------------------------------------------
    # Roman-only
    # ------------------------------------------------------------

    roman_names = [
        tel.name
        for tel in true_model.event.telescopes
        if (
            str(
                tel.name
            ).lower()
            in {
                "w149",
                "roman",
                "f146",
            }
            or
            str(
                tel.filter
            ).lower()
            in {
                "w149",
                "f146",
            }
        )
    ]

    assert len(
        roman_names
    ) == 1

    R_model, R_params = rebuild_model(
        true_model,
        truth,
        roman_names,
        roman_at_earth=False,
    )

    audit_jacobian(
        R_model,
        R_params,
        f"{population} R",
    )

    audit_rcond(
        R_model,
        R_params,
        f"{population} R",
    )

    # ------------------------------------------------------------
    # RR using only Rubin bands with >=5 surviving points
    # ------------------------------------------------------------

    rr_names = list(
        roman_names
    )

    for tel in true_model.event.telescopes:

        if tel.name in roman_names:
            continue

        if (
            tel.lightcurve is not None
            and len(
                tel.lightcurve
            )
            >= 5
        ):
            rr_names.append(
                tel.name
            )

    if rr_names != roman_names:

        RR_model, RR_params = rebuild_model(
            true_model,
            truth,
            rr_names,
            roman_at_earth=False,
        )

        audit_jacobian(
            RR_model,
            RR_params,
            f"{population} RR",
        )

        audit_rcond(
            RR_model,
            RR_params,
            f"{population} RR",
        )

    else:

        print()
        print(
            f"{population}: RR == R "
            "because no Rubin band has >=5 surviving points."
        )


def main():
    print(
        "CONFIG =",
        CONFIG,
    )

    print(
        "CATALOG =",
        CATALOG,
    )

    print(
        "EPHEMERIDES =",
        EPHEMERIDES,
    )

    for (
        population,
        index,
        model_name,
    ) in CASES:

        run_case(
            population,
            index,
            model_name,
        )

    print()
    print(
        "FISHER NUMERICAL AUDIT FINISHED"
    )


if __name__ == "__main__":
    main()
