"""LEGACY - not used by production. Kept for reference only.

Code removed from functions_roman_rubin.sim_event when the materialized
catalogue became the only source of the event realization. It generated
blending (via flux_parameters_from_blend_ratio / flux_parameters_model)
and, for USBL rows without caustic_origin, drew a random caustic origin.
Reproduced below as comments, in the order it appeared in sim_event.
"""


# ---- comentario del router A/B/C ----
#     # Event realization is selected from two independent pieces of
#     # information already present in `data`:
#     #
#     # A. blend_ratio + fsource/ftotal for every active band:
#     #    fully materialized catalog realization. All values are
#     #    authoritative and are passed through unchanged.
#     #
#     # B. blend_ratio for every active band, but no materialized fluxes:
#     #    generated/manual realization with explicit blending. The
#     #    supplied g values are preserved, but fluxes are constructed by
#     #    the generated/manual path.
#     #
#     # C. neither blend_ratio nor materialized fluxes:
#     #    legacy/generated realization. Blending remains undecided here
#     #    and is sampled later at the historical RNG call site.
#     #
#     # Partial materialization is an integrity error; the helpers below
#     # raise rather than silently mixing fixed and reconstructed values.

# ---- router rutas A/B/C ----
#     if has_materialized_fluxes:
#
#         if not has_materialized_blend:
#             raise ValueError(
#                 "Materialized fsource/ftotal require "
#                 "blend_ratio_<band> for every active band."
#             )
#
#         realization = realization_from_catalog(
#             data,
#             model,
#             truth_parallax,
#             t0,
#             band_order,
#         )
#
#     elif has_materialized_blend:
#
#         explicit_blend_ratio = {
#             band: float(
#                 data[f"blend_ratio_{band}"]
#             )
#             for band in band_order
#         }
#
#         realization = realization_from_generated_parameters(
#             data,
#             model,
#             truth_parallax,
#             t0,
#             band_order,
#             blend_ratio=explicit_blend_ratio,
#         )
#
#     else:
#
#         realization = realization_from_generated_parameters(
#             data,
#             model,
#             truth_parallax,
#             t0,
#             band_order,
#         )

# ---- origen USBL aleatorio ----
#     if realization.caustic_origin is not None:
#         # Explicit origin (from either producer): never sampled.
#         caustic_origin_arg = [realization.caustic_origin, [0, 0]]
#         bl_random_origin = False
#     else:
#         # Not USBL, or legacy/generated USBL with no explicit origin
#         # in `data`: preserve EXACTLY the historical random-origin
#         # mechanism (np.random.choice inside
#         # set_model_pyLIMA.choose_usbl_origin, via model_choice),
#         # unmoved, at the same point in the RNG sequence it has always
#         # occupied -- before flux_parameters_model's blending draws.
#         caustic_origin_arg = None
#         bl_random_origin = True

# ---- flujos generados (blend_ratio / flux_parameters_model) ----
#     elif realization.blend_ratio is not None:
#         # Generated/manual realization with an explicitly supplied
#         # blend ratio but without materialized catalog fluxes.
#         my_own_flux_parameters, fs, G, F = flux_parameters_from_blend_ratio(
#             magstar,
#             ZP,
#             my_own_model,
#             band_order,
#             realization.blend_ratio,
#         )
#
#     else:
#         # Legacy/generated/LRT path: EXACT historical signature and
#         # call site. This global is what LRT monkey-patches
#         # (functions_roman_rubin.flux_parameters_model = ...), and its
#         # replacement does not accept blend_ratio or **kwargs -- do
#         # not add any keyword here.
#         my_own_flux_parameters, fs, G, F = flux_parameters_model(
#             magstar,
#             ZP,
#             my_own_model,
#             band_order=band_order,
#         )
