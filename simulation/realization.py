"""EventRealization: an already-decided event, independent of its origin.

Two independent decisions make up a realization, and neither is
allowed to infer the other (do not couple them):

1. Blending: explicit/materialized `blend_ratio_<band>` columns (from
   `catalog.blending.blending_columns`) vs. legacy/sampled-later.
   `data_has_materialized_blend_ratio(data, band_order)` is this
   decision -- it says nothing about whether `data` "is a catalog
   event"; it only says whether blending is already materialized in
   it. A `catalog_mode="custom_system"` event with no materialized
   blending is legacy for blending purposes, independently of its
   `catalog_mode`.
2. Caustic origin (USBL only): explicit `caustic_origin` in `data` vs.
   legacy/random. This is resolved independently by each producer
   below; it is NOT derived from decision 1.

Two producers build the same `EventRealization` shape:

- `realization_from_catalog`: the materialized-blending producer.
  Requires `blend_ratio_<band>` for every band (raises `KeyError`
  otherwise) -- never samples blending. For USBL, also requires an
  explicit `caustic_origin` in `data` (raises `RuntimeError` otherwise)
  -- a materialized catalog row is expected to carry both, written at
  catalog-construction time by `catalog.blending.blending_columns` and
  `catalog.caustic_origin.choose_catalog_caustic_origin`.
- `realization_from_generated_parameters`: the legacy/on-the-fly
  producer. Blending defaults to undecided (`None`, sampled later by
  `set_model_pyLIMA.flux_parameters_model`, exactly as before this
  refactor) but accepts an explicit `blend_ratio` override for a
  caller (e.g. a test) that already decided every input by hand. For
  USBL, if `data` has an explicit `caustic_origin`, it is used
  (validated, not sampled); if it does not, `caustic_origin` is left
  `None` -- meaning "not yet decided" -- and the caller
  (`functions_roman_rubin.sim_event`) MUST fall back to the exact
  historical random-origin mechanism
  (`set_model_pyLIMA.model_choice(..., BL_random_origin=True,
  BL_origin=None)`, which samples via `np.random.choice` inside
  `choose_usbl_origin`, unmoved from where it has always lived).
  This function never samples a caustic origin itself.

Neither producer samples physical parameters (t0, u0, tE, rho, piEN,
piEE, s, q, alpha, ...): those are already decided by the caller
(`ulens_params.event_param` / a catalog row) before reaching this
module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class EventRealization:
    """Container for what the simulation core actually consumes.

    `caustic_origin` (USBL only) is `None` in two different cases that
    callers must not confuse: (a) the model is not USBL, where origin
    is simply irrelevant, or (b) the model is USBL and this is a
    `realization_from_generated_parameters` realization with no
    explicit origin in its input -- meaning "not yet decided, the
    legacy random mechanism in `set_model_pyLIMA.model_choice` must
    decide it." A `realization_from_catalog` realization never has
    `caustic_origin=None` for USBL: it raises instead (see module
    docstring).

    `physical_params` is currently a full copy of the input mapping
    (`dict(data)`), kept for compatibility with `set_model_pyLIMA
    .parameters_model`, which still looks up physical parameters by
    name from this same mapping -- it is NOT yet restricted to only
    the physical parameters a given model needs. Provenance
    (event_seed, system_type, catalog ids, GENULENS weights, ...) is
    also still present in this copy, in addition to being duplicated
    into `metadata`; narrowing `physical_params` to just the consumed
    keys is a separate, not-yet-made change (see module docstring).
    `metadata` is the field the simulation core is guaranteed to never
    read; `physical_params` is not yet that guarantee.

    `physical_params` is a copy, not the caller's original object: the
    producers below never mutate the caller's `data`, and neither does
    anything built from this realization.
    """

    model_name: str
    use_parallax: bool
    t0_parallax: float
    caustic_origin: str | None = None
    blend_ratio: dict | None = None
    physical_params: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


_METADATA_KEYS = ("event_seed", "system_type", "catalog_mode", "catalog_uid")

CAUSTIC_ORIGIN_KEY = "caustic_origin"


def _validated_origin_name(origin_name):
    from catalog.caustic_origin import CAUSTIC_ORIGINS

    if origin_name not in CAUSTIC_ORIGINS:
        raise ValueError(f"Invalid caustic_origin: {origin_name!r}")

    return origin_name


def _resolve_catalog_caustic_origin(data, model_name):
    """The materialized-blending producer's origin contract: required
    for USBL, never sampled. A materialized catalog row is expected to
    already carry `caustic_origin` (written by `catalog.caustic_origin
    .choose_catalog_caustic_origin` at catalog-construction time)."""
    if str(model_name).upper() != "USBL":
        return None

    if CAUSTIC_ORIGIN_KEY not in data:
        raise RuntimeError(
            "Missing caustic_origin for USBL catalog row: "
            "realization_from_catalog requires it to already be "
            "materialized (catalog.caustic_origin.choose_catalog_"
            "caustic_origin writes it at catalog-construction time) "
            "and never samples it."
        )

    return _validated_origin_name(str(data[CAUSTIC_ORIGIN_KEY]))


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


def data_has_materialized_blend_ratio(data, band_order):
    """Decide whether `data` carries a materialized `blend_ratio_<band>`
    realization -- purely a blending decision, independent of
    `catalog_mode` or any other notion of "is this a catalog event."
    This is the decision the orchestrating layer (`sim_event`) uses to
    pick `realization_from_catalog` vs.
    `realization_from_generated_parameters`.

    - none of the bands have a `blend_ratio_<band>` column -> False
      (legacy/generated: sampled later).
    - every band has one -> True (explicit/materialized realization).
    - some but not all -> raises `ValueError`. A partially-materialized
      realization is not silently completed by sampling the missing
      bands at random: that would mix a fixed and a freshly-sampled
      blend_ratio within the same event with no record of which is
      which.
    """
    if not band_order:
        return False

    present = [f"blend_ratio_{band}" in data for band in band_order]

    if all(present):
        return True

    if not any(present):
        return False

    missing = [b for b, has in zip(band_order, present) if not has]
    have = [b for b, has in zip(band_order, present) if has]
    raise ValueError(
        "Partially-materialized blend_ratio realization: "
        f"present for {have}, missing for {missing}. "
        "Either materialize blend_ratio_<band> for every band in "
        "band_order, or for none of them."
    )


def _validate_blend_ratio(blend_ratio, band_order):
    """g = Fblend/Fsource must be finite and >= 0 for every band.
    No upper bound is imposed (g <= 1 is not a validity requirement).
    """
    for band in band_order:
        g = blend_ratio[band]

        if not math.isfinite(g):
            raise ValueError(f"blend_ratio[{band!r}] is not finite: {g!r}")

        if g < 0:
            raise ValueError(f"blend_ratio[{band!r}] must be >= 0, got {g!r}")


def _blend_ratio_from_data(data, band_order):
    blend_ratio = {band: float(data[f"blend_ratio_{band}"]) for band in band_order}
    _validate_blend_ratio(blend_ratio, band_order)
    return blend_ratio


def _metadata_from_data(data):
    return {key: data[key] for key in _METADATA_KEYS if key in data}


def realization_from_catalog(
    data,
    model_name,
    use_parallax,
    t0_parallax,
    band_order,
):
    """Build an EventRealization from data that ALREADY carries a
    materialized blending realization (see module docstring for the
    full contract, including the stricter, catalog-specific
    caustic_origin requirement for USBL).

    Callers must check `data_has_materialized_blend_ratio(data,
    band_order)` (or otherwise know blending really is materialized in
    `data`) before calling this. It raises `KeyError` rather than
    silently sampling blending when any band's `blend_ratio_<band>` is
    missing.
    """
    if not data_has_materialized_blend_ratio(data, band_order):
        missing = [b for b in band_order if f"blend_ratio_{b}" not in data]
        raise KeyError(
            "realization_from_catalog requires a precomputed "
            f"blend_ratio_<band> for every band; missing for: {missing}. "
            "Use realization_from_generated_parameters for data without "
            "a materialized blending realization."
        )

    return EventRealization(
        model_name=model_name,
        use_parallax=use_parallax,
        t0_parallax=t0_parallax,
        caustic_origin=_resolve_catalog_caustic_origin(data, model_name),
        blend_ratio=_blend_ratio_from_data(data, band_order),
        physical_params=dict(data),
        metadata=_metadata_from_data(data),
    )


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
