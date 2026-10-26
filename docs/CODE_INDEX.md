

<!-- BEGIN ROMAN_RUBIN_SOURCE_INDEX -->
# Python source index

Automatically enumerated from locally present Git-tracked/untracked, non-ignored `.py` files.
Descriptions come only from each module's **own top-level docstring**.
Missing descriptions are explicitly flagged, not fabricated.
This index is not an execution/configuration contract: see the root README and production method.

**Indexed Python files:** 191

## `.`

| File | Top-level module documentation |
|---|---|
| [`__init__.py`](../__init__.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`class_analysis.py`](../class_analysis.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`combine_parquets.py`](../combine_parquets.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`connect_CHE.py`](../connect_CHE.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`connect_CHE_private_account.py`](../connect_CHE_private_account.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`create_df_analysis.py`](../create_df_analysis.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`detection_criteria.py`](../detection_criteria.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`fit_lc.py`](../fit_lc.py) | fit_lc.py Rutinas para construir eventos pyLIMA, ajustar curvas Roman/Rubin y reconstruir modelos para graficar. |
| [`functions_roman_rubin.py`](../functions_roman_rubin.py) | Public entry point of the Roman+Rubin simulation/fit pipeline. |
| [`launch_catalog_simfit_from_config.py`](../launch_catalog_simfit_from_config.py) | Run materialized Roman-Rubin events from one JSON configuration file. |
| [`pyLIMA_plots.py`](../pyLIMA_plots.py) | Compatibility facade: the plotting code lives in plotting/. |
| [`read_save.py`](../read_save.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`set_model_pyLIMA.py`](../set_model_pyLIMA.py) | set_model_pyLIMA.py Constructor único de modelos pyLIMA y utilidades asociadas. |
| [`set_telescopes_pyLIMA.py`](../set_telescopes_pyLIMA.py) | Created on Wed Jul 8 13:36:34 2026 @author: anibal. |
| [`ssh_connect.py`](../ssh_connect.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`timing_utils.py`](../timing_utils.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`ulens_params.py`](../ulens_params.py) | **No module docstring: inspect the code before assuming its purpose.** |

## `analysis`

| File | Top-level module documentation |
|---|---|
| [`analysis/__init__.py`](../analysis/__init__.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`analysis/analysis_event.py`](../analysis/analysis_event.py) | Analysis_Event: one simulated + fitted event for the analysis tables. |
| [`analysis/fisher_information.py`](../analysis/fisher_information.py) | Local Fisher information for Roman--Rubin E1. |
| [`analysis/fspl_chain_rule.py`](../analysis/fspl_chain_rule.py) | FSPL/FSPLarge physical derivatives for Roman--Rubin E1. |
| [`analysis/simfit_records.py`](../analysis/simfit_records.py) | Turn a simulated + fitted event into output records. |

## `catalog`

| File | Top-level module documentation |
|---|---|
| [`catalog/__init__.py`](../catalog/__init__.py) | Catalog-realization utilities: blending, caustic origin, Amax prefilter, and event-catalog I/O. |
| [`catalog/amax.py`](../catalog/amax.py) | Catalog-level Amax detectability prefilter. |
| [`catalog/blending.py`](../catalog/blending.py) | Deterministic, precomputed blending for Roman-Rubin catalogs. |
| [`catalog/caustic_origin.py`](../catalog/caustic_origin.py) | Reproducible CROIN-origin realizations. |
| [`catalog/event_reader.py`](../catalog/event_reader.py) | Reader for the materialized GBTDS 300-event Amax production catalog. |

## `decision_tests`

| File | Top-level module documentation |
|---|---|
| [`decision_tests/audit_fisher_numerics.py`](../decision_tests/audit_fisher_numerics.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/audit_fisher_production_events.py`](../decision_tests/audit_fisher_production_events.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/audit_fisher_v21_stability.py`](../decision_tests/audit_fisher_v21_stability.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/audit_fisher_v2_step_stability.py`](../decision_tests/audit_fisher_v2_step_stability.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/audit_pylima_ephemeris_extrapolation.py`](../decision_tests/audit_pylima_ephemeris_extrapolation.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/audit_pylima_periodic_candidate.py`](../decision_tests/audit_pylima_periodic_candidate.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/audit_pylima_space_ephemerides_method.py`](../decision_tests/audit_pylima_space_ephemerides_method.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/audit_roman_ephemerides_raw.py`](../decision_tests/audit_roman_ephemerides_raw.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/audit_roman_ephemeris_provenance.py`](../decision_tests/audit_roman_ephemeris_provenance.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/audit_roman_l2_geometry.py`](../decision_tests/audit_roman_l2_geometry.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/audit_roman_periodic_candidate.py`](../decision_tests/audit_roman_periodic_candidate.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/build_roman_periodic_ephemeris_candidate.py`](../decision_tests/build_roman_periodic_ephemeris_candidate.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/compare_horizons_vs_ideal_l2.py`](../decision_tests/compare_horizons_vs_ideal_l2.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/compare_roman_oem_horizons.py`](../decision_tests/compare_roman_oem_horizons.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/fit_horizons_periodic_halo.py`](../decision_tests/fit_horizons_periodic_halo.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/fit_oem_phase_projected_baseline.py`](../decision_tests/fit_oem_phase_projected_baseline.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/fit_oem_phase_to_horizons.py`](../decision_tests/fit_oem_phase_to_horizons.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/summarize_periodic_halo_blind_windows.py`](../decision_tests/summarize_periodic_halo_blind_windows.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/test_roman_horizons.py`](../decision_tests/test_roman_horizons.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/test_roman_horizons_direct.py`](../decision_tests/test_roman_horizons_direct.py) | **No module docstring: inspect the code before assuming its purpose.** |

## `decision_tests/rubin_information`

| File | Top-level module documentation |
|---|---|
| [`decision_tests/rubin_information/diagnose_sameN_location.py`](../decision_tests/rubin_information/diagnose_sameN_location.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/rubin_information/exp1_cumulative_information.py`](../decision_tests/rubin_information/exp1_cumulative_information.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/rubin_information/exp2_permutation_ladders.py`](../decision_tests/rubin_information/exp2_permutation_ladders.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`decision_tests/rubin_information/exp3_tau_windows.py`](../decision_tests/rubin_information/exp3_tau_windows.py) | **No module docstring: inspect the code before assuming its purpose.** |

## `ephemerides/Roman_positions_2027_2032_periodic_v1_provenance/construction`

| File | Top-level module documentation |
|---|---|
| [`ephemerides/Roman_positions_2027_2032_periodic_v1_provenance/construction/build_roman_periodic_ephemeris_candidate.py`](../ephemerides/Roman_positions_2027_2032_periodic_v1_provenance/construction/build_roman_periodic_ephemeris_candidate.py) | **No module docstring: inspect the code before assuming its purpose.** |

## `ephemerides/Roman_positions_2027_2032_periodic_v1_provenance/validation`

| File | Top-level module documentation |
|---|---|
| [`ephemerides/Roman_positions_2027_2032_periodic_v1_provenance/validation/audit_pylima_periodic_candidate.py`](../ephemerides/Roman_positions_2027_2032_periodic_v1_provenance/validation/audit_pylima_periodic_candidate.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`ephemerides/Roman_positions_2027_2032_periodic_v1_provenance/validation/audit_roman_periodic_candidate.py`](../ephemerides/Roman_positions_2027_2032_periodic_v1_provenance/validation/audit_roman_periodic_candidate.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`ephemerides/Roman_positions_2027_2032_periodic_v1_provenance/validation/compare_horizons_vs_ideal_l2.py`](../ephemerides/Roman_positions_2027_2032_periodic_v1_provenance/validation/compare_horizons_vs_ideal_l2.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`ephemerides/Roman_positions_2027_2032_periodic_v1_provenance/validation/fit_horizons_periodic_halo.py`](../ephemerides/Roman_positions_2027_2032_periodic_v1_provenance/validation/fit_horizons_periodic_halo.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`ephemerides/Roman_positions_2027_2032_periodic_v1_provenance/validation/test_roman_horizons_direct.py`](../ephemerides/Roman_positions_2027_2032_periodic_v1_provenance/validation/test_roman_horizons_direct.py) | **No module docstring: inspect the code before assuming its purpose.** |

## `fitting`

| File | Top-level module documentation |
|---|---|
| [`fitting/__init__.py`](../fitting/__init__.py) | Fit-side helpers of the Roman+Rubin pipeline (inputs, likelihood, LRT, multi-fit). |
| [`fitting/fit_bounds.py`](../fitting/fit_bounds.py) | Bounds of the fitted parameters. |
| [`fitting/fit_event.py`](../fitting/fit_event.py) | Build the pyLIMA Event that is fitted. |
| [`fitting/fit_inputs.py`](../fitting/fit_inputs.py) | Data and bookkeeping for a light-curve fit. |
| [`fitting/fit_parameters.py`](../fitting/fit_parameters.py) | Parameters of the fit model. |
| [`fitting/initial_guess.py`](../fitting/initial_guess.py) | Initial guess of the fit. |
| [`fitting/likelihood.py`](../fitting/likelihood.py) | Photometric chi2 / likelihood and likelihood-ratio tests. |
| [`fitting/multi_fit.py`](../fitting/multi_fit.py) | Several named fits on the same simulated data: specs, entries, summary. |
| [`fitting/run_fits.py`](../fitting/run_fits.py) | Run the pyLIMA fits of one simulated event. |
| [`fitting/trf_optimizer.py`](../fitting/trf_optimizer.py) | pyLIMA fitter, TRF least-squares options and fit diagnostics. |
| [`fitting/true_generator.py`](../fitting/true_generator.py) | chi2 / logL of the TRUE simulated model on the fitted data. |

## `legacy`

| File | Top-level module documentation |
|---|---|
| [`legacy/Sim_Fit_Event.py`](../legacy/Sim_Fit_Event.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/category_constant.py`](../legacy/category_constant.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/class_functions_roman_rubin.py`](../legacy/class_functions_roman_rubin.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/create_chunks_TRILEGAL_GENULENS.py`](../legacy/create_chunks_TRILEGAL_GENULENS.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/dp0_rubin_pylima.py`](../legacy/dp0_rubin_pylima.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/dp0_utils.py`](../legacy/dp0_utils.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/fit_lc_unused.py`](../legacy/fit_lc_unused.py) | Functions of fit_lc.py that nothing calls (moved in step 7; not imported). |
| [`legacy/fixed_field_pointing.py`](../legacy/fixed_field_pointing.py) | LEGACY - not used by production. |
| [`legacy/functions_roman_rubin_old.py`](../legacy/functions_roman_rubin_old.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/functions_roman_rubin_v1.py`](../legacy/functions_roman_rubin_v1.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/parameters_xallarap.py`](../legacy/parameters_xallarap.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/plot_lightcurves.py`](../legacy/plot_lightcurves.py) | Created on Mon Apr 20 22:32:47 2026 @author: anibal. |
| [`legacy/plot_saveLC.py`](../legacy/plot_saveLC.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/pure_wrappers.py`](../legacy/pure_wrappers.py) | Pure wrappers and dead random-origin code removed in step 6.2 (not imported). |
| [`legacy/read_fit_old_sets.py`](../legacy/read_fit_old_sets.py) | LEGACY - not used by production. |
| [`legacy/realization_generated_parameters.py`](../legacy/realization_generated_parameters.py) | LEGACY - not used by production. |
| [`legacy/run_analysis.py`](../legacy/run_analysis.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/run_class_analysis.py`](../legacy/run_class_analysis.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/run_paralelo_read_fit.py`](../legacy/run_paralelo_read_fit.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/run_parallel_fits_binary.py`](../legacy/run_parallel_fits_binary.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/run_plots.py`](../legacy/run_plots.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/scan_rho.py`](../legacy/scan_rho.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/selection_criteria.py`](../legacy/selection_criteria.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/set_model_generated_fluxes.py`](../legacy/set_model_generated_fluxes.py) | LEGACY - not used by production. |
| [`legacy/set_telescopes_unused.py`](../legacy/set_telescopes_unused.py) | Functions of set_telescopes_pyLIMA.py that nothing calls (moved in step 7; not imported). |
| [`legacy/sim_event_generated_realization.py`](../legacy/sim_event_generated_realization.py) | LEGACY - not used by production. |
| [`legacy/sim_fit_generated_inputs.py`](../legacy/sim_fit_generated_inputs.py) | LEGACY - not used by production. |
| [`legacy/sim_fit_parallelization.py`](../legacy/sim_fit_parallelization.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/sims_deviations.py`](../legacy/sims_deviations.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/test_timeseries_long.py`](../legacy/test_timeseries_long.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/tests_generated_realization.py`](../legacy/tests_generated_realization.py) | LEGACY - not used by production. |
| [`legacy/tests_monkeypatch_contract.py`](../legacy/tests_monkeypatch_contract.py) | Tests of the removed monkey-patch contract of functions_roman_rubin. |

## `legacy/chunk_based_simfit_pipeline`

| File | Top-level module documentation |
|---|---|
| [`legacy/chunk_based_simfit_pipeline/launch_simfit_from_config.py`](../legacy/chunk_based_simfit_pipeline/launch_simfit_from_config.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/chunk_based_simfit_pipeline/simfit_parallel_runner.py`](../legacy/chunk_based_simfit_pipeline/simfit_parallel_runner.py) | **No module docstring: inspect the code before assuming its purpose.** |

## `legacy/notebooks`

| File | Top-level module documentation |
|---|---|
| [`legacy/notebooks/Example_fit.py`](../legacy/notebooks/Example_fit.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`legacy/notebooks/Example_sim.py`](../legacy/notebooks/Example_sim.py) | **No module docstring: inspect the code before assuming its purpose.** |

## `notebooks`

| File | Top-level module documentation |
|---|---|
| [`notebooks/filter_df.py`](../notebooks/filter_df.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`notebooks/plot_hist.py`](../notebooks/plot_hist.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`notebooks/roman_rubin_troughtputs.py`](../notebooks/roman_rubin_troughtputs.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`notebooks/test__.py`](../notebooks/test__.py) | Created on Wed Apr 22 11:37:13 2026 @author: anibal-pc. |

## `notebooks/BH-FFP`

| File | Top-level module documentation |
|---|---|
| [`notebooks/BH-FFP/binned_metric.py`](../notebooks/BH-FFP/binned_metric.py) | Created on Wed Apr 1 17:25:35 2026 @author: anibal-pc. |
| [`notebooks/BH-FFP/compute-metrics.py`](../notebooks/BH-FFP/compute-metrics.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`notebooks/BH-FFP/delta_t0_delta_u0.py`](../notebooks/BH-FFP/delta_t0_delta_u0.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`notebooks/BH-FFP/plot_hist_ratio.py`](../notebooks/BH-FFP/plot_hist_ratio.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`notebooks/BH-FFP/plot_thetaE_theta_source.py`](../notebooks/BH-FFP/plot_thetaE_theta_source.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`notebooks/BH-FFP/plot_thetaE_theta_source_parallel.py`](../notebooks/BH-FFP/plot_thetaE_theta_source_parallel.py) | Created on Tue Mar 24 23:00:04 2026 @author: anibal-pc. |
| [`notebooks/BH-FFP/read_results_.py`](../notebooks/BH-FFP/read_results_.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`notebooks/BH-FFP/tE_dist_conditions_on_rho.py`](../notebooks/BH-FFP/tE_dist_conditions_on_rho.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`notebooks/BH-FFP/tE_dist_with_conditions_on_piE.py`](../notebooks/BH-FFP/tE_dist_with_conditions_on_piE.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`notebooks/BH-FFP/tE_distribution_with_conditions_on_metrics.py`](../notebooks/BH-FFP/tE_distribution_with_conditions_on_metrics.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`notebooks/BH-FFP/test_.py`](../notebooks/BH-FFP/test_.py) | **No module docstring: inspect the code before assuming its purpose.** |

## `photometry`

| File | Top-level module documentation |
|---|---|
| [`photometry/__init__.py`](../photometry/__init__.py) | Photometric noise models and shared photometric constants (Roman F146, Rubin). |
| [`photometry/constants.py`](../photometry/constants.py) | Shared photometric constants. |
| [`photometry/roman_f146.py`](../photometry/roman_f146.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`photometry/roman_photometry.py`](../photometry/roman_photometry.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`photometry/roman_rubin_photometry.py`](../photometry/roman_rubin_photometry.py) | Roman + Rubin photometric noise stage of the simulation. |

## `pipeline`

| File | Top-level module documentation |
|---|---|
| [`pipeline/__init__.py`](../pipeline/__init__.py) | End-to-end per-event pipelines (simulation + fits + output records). |
| [`pipeline/sim_fit.py`](../pipeline/sim_fit.py) | Simulate one catalogue event, fit it and save the outputs. |
| [`pipeline/sim_fit_multi_fits.py`](../pipeline/sim_fit_multi_fits.py) | Simulate one catalogue event and run several named fits on it. |

## `plotting`

| File | Top-level module documentation |
|---|---|
| [`plotting/__init__.py`](../plotting/__init__.py) | Plotting code: pyLIMA light curves, geometry, distributions and Roman+Rubin figures. |
| [`plotting/pyLIMA_distributions.py`](../plotting/pyLIMA_distributions.py) | Posterior / sample plots (from pyLIMA's pyLIMA_plots, modified copy). |
| [`plotting/pyLIMA_geometry.py`](../plotting/pyLIMA_geometry.py) | Lens geometry and astrometry plots (from pyLIMA's pyLIMA_plots, modified copy). |
| [`plotting/pyLIMA_plots.py`](../plotting/pyLIMA_plots.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`plotting/roman_rubin_plots.py`](../plotting/roman_rubin_plots.py) | Roman+Rubin light-curve figures from saved simulations and fits. |

## `scripts/validation`

| File | Top-level module documentation |
|---|---|
| [`scripts/validation/validate_catalog_amax.py`](../scripts/validation/validate_catalog_amax.py) | **No module docstring: inspect the code before assuming its purpose.** |

## `simulation`

| File | Top-level module documentation |
|---|---|
| [`simulation/__init__.py`](../simulation/__init__.py) | Event-realization layer: decide WHAT to simulate, before simulating it. |
| [`simulation/core.py`](../simulation/core.py) | Minimal, generic simulation core. |
| [`simulation/event_inputs.py`](../simulation/event_inputs.py) | Catalogue row -> event_params dictionary simulated by sim_event. |
| [`simulation/parallax_cache.py`](../simulation/parallax_cache.py) | Exact process-local cache for pyLIMA Earth ephemerides. |
| [`simulation/realization.py`](../simulation/realization.py) | EventRealization: the already-decided event that sim_event simulates. |
| [`simulation/roman_cadence.py`](../simulation/roman_cadence.py) | Roman GBTDS time sampling. |
| [`simulation/roman_detection.py`](../simulation/roman_detection.py) | Roman-only event detection used by the simulation. |
| [`simulation/rubin_dataslice.py`](../simulation/rubin_dataslice.py) | Helpers for the Rubin OpSim dataslice. |
| [`simulation/simulate_event.py`](../simulation/simulate_event.py) | Simulate one Roman+Rubin microlensing event from a catalogue row. |
| [`simulation/telescope_arrays.py`](../simulation/telescope_arrays.py) | Edit the telescopes of a simulated pyLIMA event. |
| [`simulation/telescope_templates.py`](../simulation/telescope_templates.py) | In-process cache of the Roman and Rubin pyLIMA telescope templates. |
| [`simulation/time_windows.py`](../simulation/time_windows.py) | Optional time window applied to telescopes and to the OpSim dataslice. |

## `stellar_population`

| File | Top-level module documentation |
|---|---|
| [`stellar_population/__init__.py`](../stellar_population/__init__.py) | Stellar-population and survey simulation utilities. |
| [`stellar_population/catalog_amax.py`](../stellar_population/catalog_amax.py) | Compatibility shim: this module moved to ''catalog.amax''. |
| [`stellar_population/catalog_blending.py`](../stellar_population/catalog_blending.py) | Compatibility shim: this module moved to ''catalog.blending''. |
| [`stellar_population/caustic_origin.py`](../stellar_population/caustic_origin.py) | Compatibility shim: this module moved to ''catalog.caustic_origin''. |

## `stellar_population/noise_models`

| File | Top-level module documentation |
|---|---|
| [`stellar_population/noise_models/__init__.py`](../stellar_population/noise_models/__init__.py) | Photometric noise models. |
| [`stellar_population/noise_models/roman_f146.py`](../stellar_population/noise_models/roman_f146.py) | Compatibility shim: this module moved to ''photometry.roman_f146''. |
| [`stellar_population/noise_models/roman_photometry.py`](../stellar_population/noise_models/roman_photometry.py) | Compatibility shim: this module moved to ''photometry.roman_photometry''. |

## `stellar_population/scripts`

| File | Top-level module documentation |
|---|---|
| [`stellar_population/scripts/analyze_trilegal_depth_pilot.py`](../stellar_population/scripts/analyze_trilegal_depth_pilot.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/assemble_amax_event_catalog.py`](../stellar_population/scripts/assemble_amax_event_catalog.py) | Assemble the complete Roman-Rubin precomputed-event catalogue. |
| [`stellar_population/scripts/audit_parallax_mask.py`](../stellar_population/scripts/audit_parallax_mask.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/audit_roman_f146_noise_grid.py`](../stellar_population/scripts/audit_roman_f146_noise_grid.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/build_gbtds_extinction_grid.py`](../stellar_population/scripts/build_gbtds_extinction_grid.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/build_gbtds_pysiaf_footprints.py`](../stellar_population/scripts/build_gbtds_pysiaf_footprints.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/build_gbtds_real_adaptive_cells.py`](../stellar_population/scripts/build_gbtds_real_adaptive_cells.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/build_genulens_trilegal_reservoir.py`](../stellar_population/scripts/build_genulens_trilegal_reservoir.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/build_precomputed_event_catalogs.py`](../stellar_population/scripts/build_precomputed_event_catalogs.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/build_roman_f146_pandeia_grid.py`](../stellar_population/scripts/build_roman_f146_pandeia_grid.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/catalog_event_reader.py`](../stellar_population/scripts/catalog_event_reader.py) | Compatibility shim: this module moved to ''catalog.event_reader''. |
| [`stellar_population/scripts/download_trilegal.py`](../stellar_population/scripts/download_trilegal.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/extract_gbtds_apt_tile_centers.py`](../stellar_population/scripts/extract_gbtds_apt_tile_centers.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/finalize_gbtds_trilegal_cells.py`](../stellar_population/scripts/finalize_gbtds_trilegal_cells.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/inspect_gbtds_apt_pointings.py`](../stellar_population/scripts/inspect_gbtds_apt_pointings.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/inspect_gbtds_special_pointings.py`](../stellar_population/scripts/inspect_gbtds_special_pointings.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/materialize_amax_event_catalogs.py`](../stellar_population/scripts/materialize_amax_event_catalogs.py) | Materialize the complete catalog realization and Amax prefilter. |
| [`stellar_population/scripts/prepare_pandeia_roman_revh.py`](../stellar_population/scripts/prepare_pandeia_roman_revh.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/refine_gbtds_extinction_grid.py`](../stellar_population/scripts/refine_gbtds_extinction_grid.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/refine_roman_f146_saturation_grid.py`](../stellar_population/scripts/refine_roman_f146_saturation_grid.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/run_full_event_catalog_production.py`](../stellar_population/scripts/run_full_event_catalog_production.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/run_trilegal_depth_pilot.py`](../stellar_population/scripts/run_trilegal_depth_pilot.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/select_trilegal_depth_pilot.py`](../stellar_population/scripts/select_trilegal_depth_pilot.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/smoke_roman_rubin.py`](../stellar_population/scripts/smoke_roman_rubin.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`stellar_population/scripts/update_repository_docs.py`](../stellar_population/scripts/update_repository_docs.py) | Maintain root README navigation and exhaustive Python-file index. |

## `tests`

| File | Top-level module documentation |
|---|---|
| [`tests/__init__.py`](../tests/__init__.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`tests/test_dataset_specific_dof.py`](../tests/test_dataset_specific_dof.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`tests/test_fisher_auto_dispatch.py`](../tests/test_fisher_auto_dispatch.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`tests/test_fisher_expected_weights.py`](../tests/test_fisher_expected_weights.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`tests/test_fisher_information.py`](../tests/test_fisher_information.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`tests/test_fisher_information_v2.py`](../tests/test_fisher_information_v2.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`tests/test_regression_interfaces.py`](../tests/test_regression_interfaces.py) | Regression tests freezing the EventRealization architecture and the LRT-facing compatibility contract introduced by the roman_rubin refactor. |
| [`tests/test_roman_detection_chi2.py`](../tests/test_roman_detection_chi2.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`tests/test_roman_only_selection.py`](../tests/test_roman_only_selection.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`tests/test_roman_true_chi2.py`](../tests/test_roman_true_chi2.py) | **No module docstring: inspect the code before assuming its purpose.** |
| [`tests/test_usbl_origin_parameterization.py`](../tests/test_usbl_origin_parameterization.py) | **No module docstring: inspect the code before assuming its purpose.** |

## `utils`

| File | Top-level module documentation |
|---|---|
| [`utils/__init__.py`](../utils/__init__.py) | Leaf utility helpers shared across roman_rubin modules. |
| [`utils/io.py`](../utils/io.py) | Leaf I/O helpers. |
| [`utils/process.py`](../utils/process.py) | Process-level helpers for batch jobs. |
<!-- END ROMAN_RUBIN_SOURCE_INDEX -->
