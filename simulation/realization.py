"""EventRealization: the already-decided event that sim_event simulates.

Every realization comes from one row of the materialized catalogue
(`realization_from_catalog`). The catalogue is authoritative for:

* physical parameters (t0, u0, tE, rho, piEN, piEE, s, q, alpha, ...);
* blending: `blend_ratio_<band>`, `fsource_<band>`, `ftotal_<band>` for
  every active band, written at catalogue-construction time by
  `catalog.blending.blending_columns`;
* USBL caustic origin: `caustic_origin`, written by
  `catalog.caustic_origin.choose_catalog_caustic_origin`.

Nothing is sampled or reconstructed here. Missing or partial
materialization is an integrity error (KeyError / RuntimeError).

The former on-the-fly producer of generated parameters was moved to
legacy/realization_generated_parameters.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class EventRealization:
    """Container for what the simulation core actually consumes.

    `caustic_origin` is `None` only for non-USBL models; for USBL,
    `realization_from_catalog` raises if the catalogue row lacks it.

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
    materialized_fluxes: dict | None = None
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


def data_has_materialized_blend_ratio(data, band_order):
    """Decide whether `data` carries a complete materialized
    `blend_ratio_<band>` realization.

    This is purely a blending decision, independent of
    `catalog_mode` and independent of whether fsource/ftotal are also
    materialized. `sim_event` combines this result with
    `data_has_materialized_fluxes()` to distinguish catalog,
    generated/manual-explicit, and legacy/generated realizations.

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


def data_has_materialized_fluxes(data, band_order):
    """
    Decide whether `data` carries a complete materialized flux
    realization for every active band.

    This decision is independent of blend_ratio materialization.

    - no fsource/ftotal columns -> False
    - complete fsource+ftotal for every band -> True
    - an incomplete pair for one band -> ValueError
    - complete pairs for only some bands -> ValueError

    A partially-materialized flux realization is never silently
    completed by recomputing missing values.
    """

    if not band_order:
        return False

    present = []

    for band in band_order:

        source_key = f"fsource_{band}"
        total_key = f"ftotal_{band}"

        has_source = source_key in data
        has_total = total_key in data

        if has_source != has_total:
            raise ValueError(
                "Partially-materialized flux pair for "
                f"{band}: fsource present={has_source}, "
                f"ftotal present={has_total}."
            )

        present.append(
            has_source and has_total
        )

    if all(present):
        return True

    if not any(present):
        return False

    have = [
        band
        for band, has in zip(band_order, present)
        if has
    ]

    missing = [
        band
        for band, has in zip(band_order, present)
        if not has
    ]

    raise ValueError(
        "Partially-materialized flux realization: "
        f"present for {have}, missing for {missing}. "
        "Either materialize fsource_<band> and ftotal_<band> "
        "for every active band, or for none of them."
    )


def _blend_ratio_from_data(data, band_order):
    blend_ratio = {band: float(data[f"blend_ratio_{band}"]) for band in band_order}
    _validate_blend_ratio(blend_ratio, band_order)
    return blend_ratio


def _materialized_fluxes_from_data(data, band_order):
    """
    Read the already-materialized catalog flux realization.

    For catalog production rows, fsource_<band> and ftotal_<band>
    are authoritative inputs. They are never reconstructed from
    magnitude or blend_ratio during simulation.
    """

    out = {}

    for band in band_order:

        source_key = f"fsource_{band}"
        total_key = f"ftotal_{band}"

        missing = [
            key
            for key in (source_key, total_key)
            if key not in data
        ]

        if missing:
            raise KeyError(
                "Catalog realization is missing materialized flux "
                f"columns for {band}: {missing}"
            )

        fsource = float(data[source_key])
        ftotal = float(data[total_key])

        if not (
            fsource > 0.0
            and ftotal > 0.0
        ):
            raise ValueError(
                "Catalog materialized fluxes must be positive: "
                f"{band}: fsource={fsource}, ftotal={ftotal}"
            )

        if ftotal < fsource:
            raise ValueError(
                "Catalog materialized fluxes are inconsistent: "
                f"{band}: ftotal={ftotal} < fsource={fsource}"
            )

        out[band] = {
            "fsource": fsource,
            "ftotal": ftotal,
        }

    return out


def _metadata_from_data(data):
    return {key: data[key] for key in _METADATA_KEYS if key in data}


def realization_from_catalog(
    data,
    model_name,
    use_parallax,
    t0_parallax,
    band_order,
):
    """Build an EventRealization from a fully materialized catalog row.

    The catalog is authoritative for physical parameters,
    caustic_origin, blend_ratio, fsource and ftotal. None of those
    quantities are sampled or reconstructed here.

    Callers must provide a fully materialized catalog realization:
    blend_ratio_<band>, fsource_<band>, and ftotal_<band> for every
    active band. Missing or partial materialization is an integrity
    error; this function never samples or reconstructs missing catalog
    quantities.
    """
    if not data_has_materialized_blend_ratio(data, band_order):
        missing = [b for b in band_order if f"blend_ratio_{b}" not in data]
        raise KeyError(
            "realization_from_catalog requires a precomputed "
            f"blend_ratio_<band> for every band; missing for: {missing}. "
            "Every active band must be materialized in the "
            "catalogue (catalog.blending.blending_columns)."
        )

    return EventRealization(
        model_name=model_name,
        use_parallax=use_parallax,
        t0_parallax=t0_parallax,
        caustic_origin=_resolve_catalog_caustic_origin(data, model_name),
        blend_ratio=_blend_ratio_from_data(data, band_order),
        materialized_fluxes=_materialized_fluxes_from_data(
            data,
            band_order,
        ),
        physical_params=dict(data),
        metadata=_metadata_from_data(data),
    )

