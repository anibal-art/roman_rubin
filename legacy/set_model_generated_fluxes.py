"""LEGACY - not used by production. Kept for reference only.

Flux/blending construction from a magnitude and a sampled or given g,
formerly in set_model_pyLIMA.py. Production now takes fsource/ftotal from
the catalogue (built by catalog/blending.py, with the magnitude as the
SOURCE magnitude; this legacy code treated it as the BASELINE magnitude).
"""


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
