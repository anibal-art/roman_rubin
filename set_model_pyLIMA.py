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
        origin=None,
        random_origin=BL_random_origin,
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


def flux_parameters_model(
    magstar,
    ZP,
    pyLIMA_model,
    band_order=None,
    rng=None,
):
    """
    Construye los parámetros de flujo para pyLIMA.

    Si rng=None usa np.random.uniform para respetar el estado global
    fijado con np.random.seed(i).
    """

    if band_order is None:
        band_order = list(magstar.keys())

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

        if rng is None:
            g = np.random.uniform(0, 1)
        else:
            g = rng.uniform(0, 1)

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
