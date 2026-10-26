# Roman and Rubin microlensing simulations

<!-- BEGIN ROMAN_RUBIN_CODE_MAP -->
## Repository guide: where does each part live?

This section is maintained by `stellar_population/scripts/update_repository_docs.py`.
The full searchable file inventory is in [docs/CODE_INDEX.md](docs/CODE_INDEX.md).
For scientific assumptions see [stellar_population/PRODUCTION_METHOD.md](stellar_population/PRODUCTION_METHOD.md).

### End-to-end data flow

```text
Sky cells + TRILEGAL source stars + GENULENS lens/source population
    -> source/lens matching and FFP/BH/planetary-system event catalogues
       (parameters, RA/Dec, materialized fluxes/blending, caustic origin)
    -> indexed Parquet shards + catalog/event_reader.py (population, index, seed)

launch_catalog_simfit_from_config.py -> pipeline/sim_fit.py, for each event:
    -> simulation/simulate_event.py (sim_event): tel_roman_rubin at the event RA/Dec
       (Roman template + Rubin OpSim), pyLIMA model, noiseless light curves
    -> photometry/: Roman F146/Pandeia noise + Rubin rubin_sim noise + m5/saturation filter
    -> Roman-only detection (simulation/roman_detection.py)
    -> Roman+Rubin and Roman-only fits (fit_lc.fit_rubin_roman, fitting/)
    -> outputs (analysis/simfit_records.py): parquet row, Analysis_Event, saved fits
```

### Production pipeline: entry points

| File | Responsibility |
|---|---|
| [`launch_catalog_simfit_from_config.py`](launch_catalog_simfit_from_config.py) | Active launcher: read one JSON configuration and run sim_fit for each materialized catalogue event. |
| [`functions_roman_rubin.py`](functions_roman_rubin.py) | Public facade: re-exports the pipeline (sim_event, sim_fit, fits, photometry, records); configures IERS and the Earth-ephemerides cache. |
| [`pipeline/sim_fit.py`](pipeline/sim_fit.py) | Simulate one catalogue event, run the Roman+Rubin and Roman-only fits and save the outputs. |
| [`pipeline/sim_fit_multi_fits.py`](pipeline/sim_fit_multi_fits.py) | sim_fit plus several named fits on the same simulated data (e.g. H1/H0) with LRT summaries. |

### Simulation of one event

| File | Responsibility |
|---|---|
| [`simulation/simulate_event.py`](simulation/simulate_event.py) | sim_event: telescopes at the event RA/Dec, true pyLIMA model from the catalogue realization, noiseless light curves, Roman/Rubin photometry and Roman-only detection. |
| [`simulation/event_inputs.py`](simulation/event_inputs.py) | Validate and normalize one materialized catalogue row into event_params (nothing is sampled). |
| [`simulation/realization.py`](simulation/realization.py) | EventRealization: parameters, materialized fluxes, blend ratios and caustic origin taken from the catalogue row. |
| [`simulation/core.py`](simulation/core.py) | pyLIMA compute_pyLIMA_parameters and noiseless simulate_lightcurve for an already-built model. |
| [`simulation/roman_detection.py`](simulation/roman_detection.py) | Roman-only detection criteria (Delta chi2 >= 300 and n3sigma consecutive points). |
| [`simulation/telescope_arrays.py`](simulation/telescope_arrays.py) | Model flux of ground telescopes, parallax arrays after the photometric filter, removal of empty telescopes. |
| [`simulation/parallax_cache.py`](simulation/parallax_cache.py) | Exact cache of the Earth ephemerides used by pyLIMA's parallax. |
| [`set_telescopes_pyLIMA.py`](set_telescopes_pyLIMA.py) | tel_roman_rubin: Roman and Rubin pyLIMA telescopes at the event RA/Dec; Rubin OpSim/MAF dataslice, paths and caches (OpSim global state). |
| [`simulation/telescope_templates.py`](simulation/telescope_templates.py) | In-process cache of the Roman/Rubin pyLIMA telescope templates. |
| [`simulation/roman_cadence.py`](simulation/roman_cadence.py) | Roman GBTDS time sampling (nominal seasons and off-seasons). |
| [`simulation/rubin_dataslice.py`](simulation/rubin_dataslice.py) | Rubin OpSim dataslice helpers: canonical filter names, Rubin time series, sky cell of the dataslice cache. |
| [`simulation/time_windows.py`](simulation/time_windows.py) | Optional time window applied to telescopes and to the OpSim dataslice. |
| [`set_model_pyLIMA.py`](set_model_pyLIMA.py) | build_pyLIMA_model: single constructor of the PSPL/FSPL/USBL pyLIMA models used to simulate, fit and plot. |
| [`detection_criteria.py`](detection_criteria.py) | Detection and observational-selection criteria applied to simulated light curves. |

### Photometry and noise

| File | Responsibility |
|---|---|
| [`photometry/roman_f146.py`](photometry/roman_f146.py) | RomanF146Noise: interpolation of the precomputed Pandeia F146 grid, including saturation/validity state; provides sigma_mag_ab(). |
| [`photometry/roman_photometry.py`](photometry/roman_photometry.py) | Apply the F146 noise model to the simulated Roman telescope photometry. |
| [`photometry/roman_rubin_photometry.py`](photometry/roman_rubin_photometry.py) | Rubin rubin_sim m5/gamma noise, m5 and saturation filters and photometry flags; calls the Roman F146 photometry. |
| [`photometry/constants.py`](photometry/constants.py) | Zero points: simulation band zero points (W149 27.615, Rubin ugrizy) and the pyLIMA fit zero point (27.4). |
| [`photometry/data/roman_f146_pandeia_2026p1.csv`](photometry/data/roman_f146_pandeia_2026p1.csv) | Precomputed Pandeia F146 grid: per-detector S/N, sigma_mag, saturation flags, and magnitude in AB. |
| [`photometry/data/roman_f146_pandeia_2026p1.json`](photometry/data/roman_f146_pandeia_2026p1.json) | Metadata and configuration associated with the Pandeia F146 grid. |
| [`stellar_population/scripts/build_roman_f146_pandeia_grid.py`](stellar_population/scripts/build_roman_f146_pandeia_grid.py) | Rebuild the Pandeia input grid; not required for routine light-curve simulations or plotting. |
| [`stellar_population/scripts/audit_roman_f146_noise_grid.py`](stellar_population/scripts/audit_roman_f146_noise_grid.py) | Audit computed Pandeia grid and saturation transitions. |

### Fits

| File | Responsibility |
|---|---|
| [`fit_lc.py`](fit_lc.py) | fit_rubin_roman (Roman+Rubin and Roman-only TRF/DE fits), save_fit_results, model_rubin_roman (model rebuilt for plots). |
| [`fitting/fit_event.py`](fitting/fit_event.py) | Build the pyLIMA Event that is fitted from lc_to_fit, at the RA/Dec of the simulated event. |
| [`fitting/fit_parameters.py`](fitting/fit_parameters.py) | Fit-model parameters: aliases, pyLIMA order, initial values, fit pyLIMA model. |
| [`fitting/fit_bounds.py`](fitting/fit_bounds.py) | Bounds of the fitted parameters (custom, narrow, wide). |
| [`fitting/initial_guess.py`](fitting/initial_guess.py) | Initial guess of the fit, validated against the bounds. |
| [`fitting/trf_optimizer.py`](fitting/trf_optimizer.py) | pyLIMA fitter, TRF least-squares options, run_fit and optimizer diagnostics. |
| [`fitting/fit_inputs.py`](fitting/fit_inputs.py) | lc_to_fit extraction, fit-only time windows, event coordinates, fit options and result paths. |
| [`fitting/run_fits.py`](fitting/run_fits.py) | run_all_fits (Roman+Rubin and Roman-only) and run_multiple_named_fits. |
| [`fitting/likelihood.py`](fitting/likelihood.py) | Photometric chi2 / logL and likelihood-ratio tests. |
| [`fitting/true_generator.py`](fitting/true_generator.py) | chi2 / logL of the true simulated model on the fitted data. |
| [`fitting/multi_fit.py`](fitting/multi_fit.py) | Named fit specs, result entries and one-row parquet summary. |

### Outputs and analysis

| File | Responsibility |
|---|---|
| [`analysis/simfit_records.py`](analysis/simfit_records.py) | Per-event output records: Analysis_Event, parameter/uncertainty row, parquet. |
| [`analysis/analysis_event.py`](analysis/analysis_event.py) | Analysis_Event: one simulated + fitted event for the analysis tables. |
| [`class_analysis.py`](class_analysis.py) | Metrics helpers for the notebooks; re-exports Analysis_Event and the Roman+Rubin figures. |
| [`read_save.py`](read_save.py) | Simulation/fit serialization and loading of saved results. |
| [`timing_utils.py`](timing_utils.py) | Stage-by-stage runtime diagnostics. |
| [`analysis/fisher_information.py`](analysis/fisher_information.py) | Local Fisher information for Roman-Rubin E1 (physical derivatives by five-point finite differences). |

### Plotting

| File | Responsibility |
|---|---|
| [`plotting/roman_rubin_plots.py`](plotting/roman_rubin_plots.py) | Roman+Rubin light-curve figures from saved simulations and fits (plot_roman_rubin_plt, graph_maker). |
| [`plotting/pyLIMA_plots.py`](plotting/pyLIMA_plots.py) | Light curves, model telescopes and colors (modified copy of pyLIMA's pyLIMA_plots). |
| [`plotting/pyLIMA_geometry.py`](plotting/pyLIMA_geometry.py) | Lens geometry and astrometry plots. |
| [`plotting/pyLIMA_distributions.py`](plotting/pyLIMA_distributions.py) | Sample distributions and parameter table. |
| [`pyLIMA_plots.py`](pyLIMA_plots.py) | Compatibility facade for plotting/ (import pyLIMA_plots keeps working). |

### Stellar populations and event catalogue

| File | Responsibility |
|---|---|
| [`stellar_population/scripts/download_trilegal.py`](stellar_population/scripts/download_trilegal.py) | Generate/download TRILEGAL stellar populations for configured Galactic-bulge cells. |
| [`stellar_population/scripts/build_genulens_trilegal_reservoir.py`](stellar_population/scripts/build_genulens_trilegal_reservoir.py) | Produce GENULENS samples/reservoir for the configured fields. |
| [`stellar_population/scripts/build_precomputed_event_catalogs.py`](stellar_population/scripts/build_precomputed_event_catalogs.py) | Match Galactic lens/source parameters and construct precomputed FFP, BH and binary-lens event Parquet files. |
| [`stellar_population/scripts/run_full_event_catalog_production.py`](stellar_population/scripts/run_full_event_catalog_production.py) | Resumable per-cell orchestration of TRILEGAL, GENULENS, event catalogue generation and assembly. |
| [`ulens_params.py`](ulens_params.py) | Microlensing system parameters, priors and model-specific parameterization used to build the catalogue. |
| [`catalog/blending.py`](catalog/blending.py) | Deterministic, precomputed blending (source/total fluxes per band) for the catalogue. |
| [`catalog/caustic_origin.py`](catalog/caustic_origin.py) | Reproducible caustic-origin realizations for binary lenses. |
| [`catalog/amax.py`](catalog/amax.py) | Catalogue-level Amax detectability prefilter. |
| [`catalog/event_reader.py`](catalog/event_reader.py) | Load a deterministic catalogue event by population and global index, including the simulation seed. |
| [`stellar_population/PRODUCTION_METHOD.md`](stellar_population/PRODUCTION_METHOD.md) | Detailed production method and assumptions; consult before changing catalogue inputs. |

### Validation, diagnostics, and figures

| File | Responsibility |
|---|---|
| [`stellar_population/scripts/smoke_roman_rubin.py`](stellar_population/scripts/smoke_roman_rubin.py) | End-to-end catalogue-event simulation and Roman-only/Roman+Rubin fit smoke tests. |
| [`stellar_population/scripts/audit_parallax_mask.py`](stellar_population/scripts/audit_parallax_mask.py) | Compare masked time-dependent parallax geometry with recomputation after photometry filtering. |
| [`stellar_population/scripts/update_repository_docs.py`](stellar_population/scripts/update_repository_docs.py) | Refresh README's workflow/code map and exhaustive Python source index. |

### Roman F146: read this before modifying noise or figures

- **Current noise input:** saved Pandeia R2026.1 F146 grid, with the project-specific Rev-H `IM_66_6_V2` read pattern. The grid is already computed: do not rerun Pandeia just to make a figure.
- **Use the production interpolation:** `RomanF146Noise` in `photometry/roman_f146.py`, especially `sigma_mag_ab()` for a magnitude-uncertainty curve. The simulation applies the model via `photometry/roman_photometry.py`.
- **Magnitude convention:** the Pandeia noise grid is in **AB**; the TRILEGAL Roman source catalogue uses **Vega**. Use the project's calibrated AB–Vega conversion before mixing magnitudes or plotting thresholds.
- **Catalog selection is not noise sensitivity:** a TRILEGAL extraction cut at F146 = 28 Vega, when configured, is not automatically the Pandeia 5-sigma limiting magnitude.
- **Legacy name:** some pyLIMA telescope objects/data products still use `W149` as an internal band label. This is not a justification for applying the obsolete W149 noise prescription to F146.
- **Observing schedule:** the manuscript uses an approximately **12-minute Roman cadence** during roughly 72-day seasons. Observation timestamps must be checked against the telescope template; changing noise must not silently alter cadence.
- **Rubin:** preserve the same `rubin_sim`/OpSim-dependent uncertainty calculation used by the production simulator. Do not substitute representative m5/gamma constants in publication figures.

### Quick validation and execution

From the repository root, activate the matching environment. Run existing smoke-test modules as Python modules (with `-m`), not by calling deeply nested script paths directly.

```bash
python -m stellar_population.scripts.smoke_roman_rubin --population BH --index 1167
python -m stellar_population.scripts.smoke_roman_rubin --population FFP --index 366
python -m stellar_population.scripts.smoke_roman_rubin --population Planets_systems --index 386
```

For parallax filtering equivalence see `stellar_population/scripts/audit_parallax_mask.py`.

Inspect the catalogue-production driver **before running expensive jobs**:

```bash
python stellar_population/scripts/run_full_event_catalog_production.py --help
```

### Maintenance and reproducibility

1. When modifying a file's scientific role, update the mapping in `update_repository_docs.py` and its module docstring.
2. Regenerate the README and complete source index with `python -m stellar_population.scripts.update_repository_docs` before committing.
3. Confirm output paths, magnitude conventions, cuts, RNG seeds, OpSim database, and noise-grid version for each production run; the README is a navigation map, not a substitute for versioned configuration.
4. A file without a module docstring is flagged in the source index instead of being assigned an invented description.

### Validation work still to close

- Ensure **Roman-only** `chi2_true` uses only the Roman observations, rather than the joint Roman+Rubin true chi-square, before interpreting `delta_chi2_true`.
- Preserve **pre-filter** photometric rejection counts and whole-band removal diagnostics; statistics calculated only after filtering cannot recover removed rows.
- Check any planned publication figure against the actual production F146 interpolator and exact Rubin uncertainty configuration.
<!-- END ROMAN_RUBIN_CODE_MAP -->

Roman and Rubin simulations and analisys.

The repository contains simulations and analyses to study the impact of combining observations of Roman and Rubin microlensing events.

---
# Analysis Results

The `all_results` directory contains analysis results for a set of events corresponding to Free Floating Planets (FFP), Black Holes (BH), and Bound Planets (PB).
---
### File Descriptions

- **`true.csv`**: Contains the true simulation parameters.
- **`fit_rr.csv`**: Contains the estimated parameters and uncertainties from the fit using data from both the Roman and Rubin observatories.
- **`fit_roman.csv`**: Contains similar information as `fit_rr.csv` but using only Roman data.

### Column Descriptions

Each of these files includes the following columns:

- **Event Identifiers**:
  - **`Source`**: Identification number of the event.
  - **`Set`**: Set identifier, as events are generated in different sets.

- **Microlensing Parameters**:
  - **`t0`**: Time of maximum magnification.
  - **`u0`**: Impact parameter.
  - **`te`**: Einstein timescale.
  - **`rho`**: Ratio of the source’s angular radius to the Einstein angular radius.
  - **`s`**: Separation between lenses, in units of Einstein radius (θE).
  - **`q`**: Mass ratio of the lenses.
  - **`alpha`**: Angle between the lens axis and line of sight.
  - **`piEN`**: North component of the parallax.
  - **`piEE`**: East component of the parallax.

- **Uncertainties for Each Parameter**:
  - **`t0_err`**: Uncertainty in `t0`.
  - **`u0_err`**: Uncertainty in `u0`.
  - **`te_err`**: Uncertainty in `te`.
  - **`rho_err`**: Uncertainty in `rho`.
  - **`s_err`**: Uncertainty in `s`.
  - **`q_err`**: Uncertainty in `q`.
  - **`alpha_err`**: Uncertainty in `alpha`.
  - **`piEN_err`**: Uncertainty in `piEN`.
  - **`piEE_err`**: Uncertainty in `piEE`.

- **Additional Parameters**:
  - **`piE`**: Total parallax magnitude.
  - **`piE_err`**: Uncertainty in the total parallax.
  - **`piE_err_MC`**: Monte Carlo-derived uncertainty in the total parallax.

- **Mass-Related Parameters**:
  - **`mass_thetaE`**: Mass estimate derived from θE.
  - **`mass_mu`**: Mass estimate derived from proper motion.
  - **`mass_thetaS`**: Mass estimate derived from the source’s angular radius.
  - **`err_mass_thetaE_NotMC`**: Non-Monte Carlo uncertainty in `mass_thetaE`.
  - **`mass_err_thetaE`**: Uncertainty in `mass_thetaE`.
  - **`mass_err_mu`**: Uncertainty in `mass_mu`.
  - **`mass_err_thetaS`**: Uncertainty in `mass_thetaS`.

- **Fit Quality Metrics**:
  - **`chichi`**: Fit quality parameter.
  - **`dof`**: Degrees of freedom for the fit.
  - **`chi2`**: Chi-squared value of the fit.
---
## Notebooks with metrics
The notebooks in the `notebooks` directory contains three notebooks 
  - **`Binary_Lens_results.ipynb`** 
  - **`FFP_results.ipynb`**
  - **`BH_results.ipynb`**
  these notebooks contain the plot of the metrics
  
  ![Equation](https://latex.codecogs.com/png.latex?\alpha=\frac{|fit-true|}{true}), 
![Equation](https://latex.codecogs.com/png.latex?\beta=\frac{|fit-true|}{\sigma}), 
![Equation](https://latex.codecogs.com/png.latex?\gamma=\frac{\sigma}{fit})

### Parallax uncertainty propagation
In the results you can find two propagation of uncertainty one using the error propagation formulae for a set of functions ![Equation](https://latex.codecogs.com/png.latex?y_1,y_2,y_3...y_m) which all depend on the n random variables ![Equation](https://latex.codecogs.com/png.latex?x_1,x_2,x_3...x_n), thus

![Equation](https://latex.codecogs.com/png.latex?cov_{kl}(\vec{y})=\sum_{i=1}^{n}\sum_{j=1}^{n}\frac{\partial&space;y_k}{\partial&space;x_i}\frac{\partial&space;y_l}{\partial&space;x_j}cov(x_i,x_j))

The second is using a montecarlo aproach by generating samples using the covariance matrix in a multinormal distribution, the covariance matrix is provided by the TRF routine in pyLIMA.

### Mass estimation

We run three test for the mass estimation using

![Equation](https://latex.codecogs.com/png.latex?M=\frac{\theta_E}{\kappa\pi_E}). 

- Assuming known ![Equation](https://latex.codecogs.com/png.latex?\theta_E). We use only the information about the estimation of ![Equation](https://latex.codecogs.com/png.latex?\pi_E) and its uncertainty.
- Assuming known ![Equation](https://latex.codecogs.com/png.latex?\theta_{star}). We use the information about the estimation of ![Equation](https://latex.codecogs.com/png.latex?\pi_E) and its uncertainty and the estimation of ![Equation](https://latex.codecogs.com/png.latex?\rho) and its uncertainty to compute ![Equation](https://latex.codecogs.com/png.latex?\pi_E) and propagate its uncertainty.
- Assuming known ![Equation](https://latex.codecogs.com/png.latex?\mu_{rel}). We use the information about the estimation of ![Equation](https://latex.codecogs.com/png.latex?\pi_E) and its uncertainty and the estimation of ![Equation](https://latex.codecogs.com/png.latex?t_E) and its uncertainty to compute ![Equation](https://latex.codecogs.com/png.latex?\pi_E) and propagate its uncertainty.
---
# Fit and simulation
The code **`functions_roman_rubin.py`** contains the fit routine and the simulation using pyLIMA and rubin_sim.

---
## **Functions**

### 1. **`tel_roman_rubin`**
**Purpose:**  
Simulates telescope observations for Rubin Observatory and Roman Space Telescope, creating synthetic light curves for microlensing events.  

**Inputs:**  
- `path_ephemerides`: Path to ephemerides file for spacecraft positions.  
- `path_dataslice`: Path to Rubin data slice file.  

**Outputs:**  
- A microlensing event object with telescope data.  

---

### 2. **`deviation_from_constant`**
**Purpose:**  
Checks if there are at least four data points within `[t0 - tE, t0 + tE]` that deviate from the constant flux baseline by more than 3σ.  

**Inputs:**  
- `pyLIMA_parameters`: Parameters describing the microlensing model.  
- `pyLIMA_telescopes`: Telescope data objects with light curves.  

**Outputs:**  
- A boolean indicating whether the deviation condition is satisfied.  

---

### 3. **`filter5points`**
**Purpose:**  
Ensures that at least one light curve contains at least five data points within the range `[t0 - tE, t0 + tE]`.  

**Inputs:**  
- `pyLIMA_parameters`: Microlensing model parameters.  
- `pyLIMA_telescopes`: Telescope data objects with light curves.  

**Outputs:**  
- A boolean indicating whether the condition is met.  

---

### 4. **`mag`**
**Purpose:**  
Converts flux measurements into magnitudes.  

**Inputs:**  
- `zp`: Zero-point magnitude.  
- `Flux`: Light curve flux values.  

**Outputs:**  
- Magnitudes corresponding to the input flux values.  

---

### 5. **`filter_band`**
**Purpose:**  
Filters light curve data based on magnitude limits and 5σ depth criteria, ensuring that the curve contains sufficient points for analysis.  

**Inputs:**  
- `mjd`: Modified Julian Dates.  
- `mag`: Magnitudes.  
- `magerr`: Magnitude errors.  
- `m5`: 5σ limiting magnitudes.  
- `fil`: Filter name.  

**Outputs:**  
- Filtered light curve data points and a boolean indicating significant detections.  

---

### 6. **`has_consecutive_numbers`**
**Purpose:**  
Checks if there are at least three consecutive numbers in a list.  

**Inputs:**  
- `lst`: List of integers.  

**Outputs:**  
- A boolean indicating if the condition is met.  

---

### 7. **`set_photometric_parameters`**
**Purpose:**  
Configures photometric parameters, including exposure time and read noise.  

**Inputs:**  
- `exptime`: Exposure time.  
- `nexp`: Number of exposures.  
- `readnoise`: (Optional) Read noise in electrons per pixel.  

**Outputs:**  
- A photometric parameters object.  

---

### 8. **`fit_rubin_roman`**
**Purpose:**  
Performs model fitting for Rubin and Roman telescope data using various microlensing models (e.g., FSPL, USBL, PSPL).  

**Inputs:**  
- Parameters for the event, model type, algorithm, and light curves for Rubin and Roman data.  

**Outputs:**  
- Fit results and associated event data.  

---

### 9. **`save`**
**Purpose:**  
Saves processed event data, light curves, and model parameters to an HDF5 file.  

**Inputs:**  
- Event index, paths to save location, and model parameters.  

**Outputs:**  
- HDF5 file containing the saved data.  

---

### 10. **`read_data`**
**Purpose:**  
Reads event data (simulated) for further processing. 
