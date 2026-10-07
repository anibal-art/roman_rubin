#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
set_model_pyLIMA.py

Constructor único de modelos pyLIMA y utilidades asociadas.

Regla de diseño:
- model describe solamente la familia del modelo: PSPL, FSPL o USBL.
- El paralaje se controla con un booleano o con el argumento pyLIMA explícito.
- Esta capa no prepara guesses, bounds, ni corre fits.
"""

import numpy as np

from pyLIMA.models import USBL_model
from pyLIMA.models import FSPLarge_model
from pyLIMA.models import PSPL_model


VALID_MODELS = [
    "PSPL",
    "FSPL",
    "USBL",
]


def normalize_model_name(model_name):
    """
    Normaliza el nombre del modelo a PSPL, FSPL o USBL.

    Se aceptan nombres devueltos por pyLIMA como FSPLarge, pero no se usa
    el nombre del modelo para decidir si hay paralaje.
    """

    if model_name is None:
        raise ValueError("model_name no puede ser None.")

    name = str(model_name).upper()

    if "USBL" in name:
        return "USBL"

    if "FSPL" in name:
        return "FSPL"

    if "PSPL" in name:
        return "PSPL"

    raise ValueError(
        f"Modelo no reconocido: {model_name}. "
        "Usar PSPL, FSPL o USBL."
    )


def make_parallax_arg(
    use_parallax=False,
    t0=None,
    parallax=None,
):
    """
    Construye el argumento de paralaje para pyLIMA.

    Casos permitidos:
    - parallax=["Full", t0] o ["None", 0.0]: se respeta.
    - use_parallax=True: devuelve ["Full", t0].
    - use_parallax=False: devuelve ["None", 0.0].
    """

    if parallax is not None:
        if not isinstance(parallax, (list, tuple)) or len(parallax) != 2:
            raise ValueError(
                "parallax debe ser ['Full', t0] o ['None', 0.0]. "
                f"Recibí: {parallax}"
            )
        return list(parallax)

    if use_parallax:
        if t0 is None:
            raise ValueError(
                "use_parallax=True requiere t0 para definir "
                "la época de referencia del paralaje."
            )
        return [
            "Full",
            float(t0),
        ]

    return [
        "None",
        0.0,
    ]


def choose_usbl_origin(
    origin=None,
    random_origin=False,
    rng=None,
):
    """
    Define el origin de USBL.

    Si rng=None usa np.random.choice para preservar la reproducibilidad
    de código que hace np.random.seed(i) antes de llamar a model_choice().
    """

    if origin is not None:
        return origin

    choices = [
        "central_caustic",
        "second_caustic",
        "third_caustic",
    ]

    if random_origin:
        if rng is None:
            choice = np.random.choice(choices)
        else:
            choice = rng.choice(choices)
    else:
        choice = "central_caustic"

    return [
        choice,
        [
            0,
            0,
        ],
    ]


def build_pyLIMA_model(
    pyLIMA_event,
    model,
    use_parallax=False,
    t0_parallax=None,
    parallax=None,
    origin=None,
    random_origin=False,
    blend_flux_parameter="ftotal",
    rng=None,
):
    """
    Constructor único de modelos pyLIMA.

    Sirve para simular, ajustar y graficar. No prepara parámetros
    iniciales, no aplica bounds y no corre el fit.
    """

    model_base = normalize_model_name(model)

    parallax_arg = make_parallax_arg(
        use_parallax=use_parallax,
        t0=t0_parallax,
        parallax=parallax,
    )

    kwargs = {
        "blend_flux_parameter": blend_flux_parameter,
        "parallax": parallax_arg,
    }

    if model_base == "PSPL":
        return PSPL_model.PSPLmodel(
            pyLIMA_event,
            **kwargs,
        )

    if model_base == "FSPL":
        return FSPLarge_model.FSPLargemodel(
            pyLIMA_event,
            **kwargs,
        )

    if model_base == "USBL":
        kwargs["origin"] = choose_usbl_origin(
            origin=origin,
            random_origin=random_origin,
            rng=rng,
        )

        return USBL_model.USBLmodel(
            pyLIMA_event,
            **kwargs,
        )

    raise ValueError(f"Modelo no reconocido: {model}")


def model_choice(
    new_creation,
    model,
    parallax=["None", 0.0],
    BL_random_origin=True,
    BL_origin=None,
):
    """
    Wrapper compatible con sim_event.

    Antes sim_event llamaba:
        model_choice(new_creation, model, parallax, BL_random_origin)

    Ahora redirige al constructor único.
    """

    return build_pyLIMA_model(
        pyLIMA_event=new_creation,
        model=model,
        parallax=parallax,
        origin=BL_origin,
        random_origin=(
            BL_random_origin if BL_origin is None else False
        ),
        blend_flux_parameter="ftotal",
    )


def physical_parameter_order(
    model,
    use_parallax=False,
):
    """
    Orden canónico de parámetros físicos.

    No incluye parámetros de flujo.
    """

    model_base = normalize_model_name(model)

    order = [
        "t0",
        "u0",
        "tE",
    ]

    if model_base == "FSPL":
        order += [
            "rho",
        ]

    elif model_base == "USBL":
        order += [
            "rho",
            "s",
            "q",
            "alpha",
        ]

    elif model_base == "PSPL":
        pass

    else:
        raise ValueError(f"Modelo no reconocido: {model}")

    if use_parallax:
        order += [
            "piEN",
            "piEE",
        ]

    return order


def parameters_model(
    data,
    pyLIMA_model,
):
    """
    Construye el diccionario de parámetros físicos para pyLIMA.
    """

    model = normalize_model_name(
        pyLIMA_model.model_type()
    )

    parallax_model = getattr(
        pyLIMA_model,
        "parallax_model",
        [
            "None",
            0.0,
        ],
    )

    use_parallax = parallax_model[0] != "None"

    param_order = physical_parameter_order(
        model,
        use_parallax=use_parallax,
    )

    params = {
        key: data[key]
        for key in param_order
    }

    return params, param_order


def _flux_parameters_for_bands(magstar, ZP, pyLIMA_model, band_order, g_for_band):
    """Shared flux-parameter math (single source of truth).

    `g_for_band(band)` returns the blending ratio g = Fblend/Fsource
    for that band; callers decide HOW g is obtained (sampled or
    already materialized) -- this function never samples anything
    itself.
    """

    flux_parameters = []
    fs, G, F = {}, {}, {}

    for band in band_order:

        if band not in magstar:
            raise KeyError(f"La banda {band} no está en magstar")

        if band not in ZP:
            raise KeyError(f"La banda {band} no está en ZP")

        flux_baseline = 10 ** (
            (ZP[band] - magstar[band]) / 2.5
        )

        g = g_for_band(band)

        f_source = flux_baseline / (1 + g)
        f_blend = g * f_source
        f_total = f_source + f_blend

        fs[band] = f_source
        G[band] = g
        F[band] = f_total

        if pyLIMA_model.blend_flux_parameter == "ftotal":
            flux_parameters.append(f_source)
            flux_parameters.append(f_total)
        else:
            flux_parameters.append(f_source)
            flux_parameters.append(f_blend)

    return flux_parameters, fs, G, F


def flux_parameters_model(
    magstar,
    ZP,
    pyLIMA_model,
    band_order=None,
    rng=None,
):
    """
    Construye los parámetros de flujo para pyLIMA.

    Firma e invocación históricas, SIN CAMBIOS: este es el punto que
    LRT reemplaza vía monkey-patching
    (functions_roman_rubin.flux_parameters_model = ...), y su reemplazo
    no acepta blend_ratio ni **kwargs. No agregar parámetros nuevos
    aquí -- ver flux_parameters_from_blend_ratio para la ruta de
    realización explícita.

    Si rng=None usa np.random.uniform para respetar el estado global
    fijado con np.random.seed(i); si rng no es None, usa rng.uniform.
    """

    if band_order is None:
        band_order = list(magstar.keys())

    def _sampled_g(band):
        if rng is None:
            return np.random.uniform(0, 1)
        return rng.uniform(0, 1)

    return _flux_parameters_for_bands(magstar, ZP, pyLIMA_model, band_order, _sampled_g)


def flux_parameters_from_blend_ratio(
    magstar,
    ZP,
    pyLIMA_model,
    band_order,
    blend_ratio,
):
    """
    Construye los parámetros de flujo a partir de un blend_ratio YA
    DECIDIDO (p.ej. por catalog.blending), uno por banda. No sortea
    nada.

    Deliberadamente NO es el nombre que LRT monkey-patchea
    (flux_parameters_model) -- es una API interna explícita para la
    ruta de realización con blending precomputado, usada solo cuando
    esa realización efectivamente lo trae. Comparte la fórmula de
    flujo con flux_parameters_model vía _flux_parameters_for_bands.
    """

    def _decided_g(band):
        return blend_ratio[band]

    return _flux_parameters_for_bands(magstar, ZP, pyLIMA_model, band_order, _decided_g)
