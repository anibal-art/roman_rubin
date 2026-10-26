"""LEGACY - not used by production. Kept for reference only.

On-the-fly EventRealization producer, formerly in simulation/realization.py.
"""


def _resolve_generated_caustic_origin(data, model_name):
    """The legacy/on-the-fly producer's origin contract: if `data`
    already provides an explicit caustic_origin, use it (validated,
    not sampled). Otherwise return None -- "not yet decided" -- so the
    caller falls back to the historical random-origin mechanism. This
    function never samples anything."""
    if str(model_name).upper() != "USBL":
        return None

    if CAUSTIC_ORIGIN_KEY not in data:
        return None

    return _validated_origin_name(str(data[CAUSTIC_ORIGIN_KEY]))


def realization_from_generated_parameters(
    data,
    model_name,
    use_parallax,
    t0_parallax,
    band_order=None,
    blend_ratio=None,
):
    """Build an EventRealization from on-the-fly generated parameters
    (or parameters set manually, e.g. in a test). See module docstring
    for the full contract, including the lenient, legacy-compatible
    caustic_origin resolution for USBL (returns None rather than
    raising when `data` has no explicit caustic_origin -- the caller
    must then use the historical random-origin mechanism).

    Blending defaults to undecided (None): it is then sampled where it
    always has been (`set_model_pyLIMA.flux_parameters_model`, at
    simulation time), using the same global-RNG call site as before
    this refactor. Pass an explicit `blend_ratio` (one value per band
    in `band_order`) only when the caller already decided it -- e.g a
    test fixing every input of a realization -- and nothing is
    sampled.
    """
    if blend_ratio is not None:
        _validate_blend_ratio(blend_ratio, band_order)

    return EventRealization(
        model_name=model_name,
        use_parallax=use_parallax,
        t0_parallax=t0_parallax,
        caustic_origin=_resolve_generated_caustic_origin(data, model_name),
        blend_ratio=blend_ratio,
        physical_params=dict(data),
        metadata=_metadata_from_data(data),
    )
