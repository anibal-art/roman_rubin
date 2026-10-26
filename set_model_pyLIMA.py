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


def choose_usbl_origin(origin=None):
    """
    Define el origin de USBL.

    El origen de la simulacion viene del catalogo (caustic_origin); sin
    origen se usa la caustica central. (La version que sorteaba el origen
    al azar esta en legacy/pure_wrappers.py.)
    """

    if origin is not None:
        return origin

    return [
        "central_caustic",
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
    blend_flux_parameter="ftotal",
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
        )

        return USBL_model.USBLmodel(
            pyLIMA_event,
            **kwargs,
        )

    raise ValueError(f"Modelo no reconocido: {model}")


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

