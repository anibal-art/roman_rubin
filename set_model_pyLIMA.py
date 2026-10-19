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



def make_double_source_arg(
    use_double_source=False,
    t0=None,
    double_source=None,
):
    """
    Construye el argumento double_source de pyLIMA.

    Casos permitidos:
    - double_source=["Circular", t0] o ["None", 0.0]: se respeta.
    - use_double_source=True: devuelve ["Circular", t0].
    - use_double_source=False: devuelve ["None", 0.0].

    El modelo circular usa el formalismo xallarap nativo de pyLIMA.
    """

    if double_source is not None:
        if (
            not isinstance(double_source, (list, tuple))
            or len(double_source) != 2
        ):
            raise ValueError(
                "double_source debe ser ['Circular', t0] "
                "o ['None', 0.0]. "
                f"Recibí: {double_source}"
            )

        mode = str(double_source[0])

        if mode not in {"None", "Circular"}:
            raise ValueError(
                "Por ahora roman_rubin soporta double_source "
                "solo en modos 'None' y 'Circular'. "
                f"Recibí: {double_source}"
            )

        return [
            mode,
            float(double_source[1]),
        ]

    if use_double_source:
        if t0 is None:
            raise ValueError(
                "use_double_source=True requiere t0 para definir "
                "la época de referencia del xallarap."
            )

        return [
            "Circular",
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
    use_double_source=False,
    t0_double_source=None,
    double_source=None,
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

    double_source_arg = make_double_source_arg(
        use_double_source=use_double_source,
        t0=t0_double_source,
        double_source=double_source,
    )

    kwargs = {
        "blend_flux_parameter": blend_flux_parameter,
        "parallax": parallax_arg,
        "double_source": double_source_arg,
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
    double_source=["None", 0.0],
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
        double_source=double_source,
        origin=BL_origin,
        random_origin=(
            BL_random_origin if BL_origin is None else False
        ),
        blend_flux_parameter="ftotal",
    )


def physical_parameter_order(
    model,
    use_parallax=False,
    use_double_source=False,
    double_source_filters=None,
):
    """
    Orden canónico de parámetros físicos.

    No incluye fsource/fblend/ftotal.

    El orden reproduce el orden de construcción de parámetros de pyLIMA:
    parámetros Paczynski -> parallax -> double source.
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
        raise ValueError(
            f"Modelo no reconocido: {model}"
        )

    if use_parallax:
        order += [
            "piEN",
            "piEE",
        ]

    if use_double_source:

        if model_base not in {
            "PSPL",
            "FSPL",
        }:
            raise NotImplementedError(
                "double_source se habilita inicialmente "
                "solo para PSPL/FSPL."
            )

        order += [
            "xi_para",
            "xi_perp",
            "xi_angular_velocity",
            "xi_phase",
            "xi_inclination",
            "xi_mass_ratio",
        ]

        if model_base == "FSPL":
            order += [
                "rho_2",
            ]

        if double_source_filters is None:
            double_source_filters = []

        for band in double_source_filters:
            order.append(
                f"q_flux_{band}"
            )

    return order


def _data_parameter_value(
    data,
    name,
):
    """
    Obtiene un parámetro físico de event_params/data.

    Se aceptan algunos aliases usados en binary_source para que
    la interfaz científica no dependa de los nombres internos de pyLIMA.
    """

    if name in data:
        return data[name]

    aliases = {
        "xi_angular_velocity": [
            "omega",
        ],
        "xi_phase": [
            "phi0",
            "phi",
        ],
        "xi_inclination": [
            "lambda_xi",
            "inclination",
        ],
        "xi_mass_ratio": [
            "q_mass",
            "qM",
        ],
    }

    for alias in aliases.get(
        name,
        [],
    ):
        if alias in data:
            return data[alias]

    # Componentes de xiE a partir de amplitud + orientación.
    if name in {
        "xi_para",
        "xi_perp",
    }:
        if "xiE" in data:
            theta = float(
                data.get(
                    "theta",
                    0.0,
                )
            )

            xiE = float(
                data["xiE"]
            )

            if name == "xi_para":
                return (
                    xiE
                    * np.cos(theta)
                )

            return (
                xiE
                * np.sin(theta)
            )

    # omega = 2 pi / P_days
    if (
        name
        == "xi_angular_velocity"
        and "P_days" in data
    ):
        return (
            2.0
            * np.pi
            / float(data["P_days"])
        )

    # q_flux es común para todas las bandas en el primer estudio.
    if name.startswith(
        "q_flux_"
    ):
        for key in [
            "q_flux",
            "qflux",
        ]:
            if key in data:
                return data[key]

    raise KeyError(
        f"Falta el parámetro '{name}' "
        "en event_params/data."
    )


def parameters_model(
    data,
    pyLIMA_model,
):
    """
    Construye el diccionario de parámetros físicos para pyLIMA.

    Detecta automáticamente si el modelo usa parallax y/o
    double_source circular.
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

    double_source_model = getattr(
        pyLIMA_model,
        "double_source_model",
        [
            "None",
            0.0,
        ],
    )

    use_parallax = (
        parallax_model[0]
        != "None"
    )

    use_double_source = (
        double_source_model[0]
        != "None"
    )

    double_source_filters = []

    if use_double_source:
        double_source_filters = list(
            np.unique(
                [
                    telescope.filter
                    for telescope
                    in pyLIMA_model.event.telescopes
                ]
            )
        )

    param_order = physical_parameter_order(
        model,
        use_parallax=use_parallax,
        use_double_source=use_double_source,
        double_source_filters=double_source_filters,
    )

    params = {
        key: _data_parameter_value(
            data,
            key,
        )
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
