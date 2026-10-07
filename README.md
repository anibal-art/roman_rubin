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
    -> indexed Parquet shards + event reader (population, index, seed)
    -> pyLIMA model + Roman scheduled times + Rubin OpSim observations
    -> Roman F146/Pandeia noise + Rubin rubin_sim photometry + filtering
    -> detection selection -> Roman-only / Roman+Rubin fits
    -> saved light curves, parameters and fit diagnostics
```

### Scientific models and orchestration

| File | Responsibility |
|---|---|
| [`ulens_params.py`](ulens_params.py) | Sampling and assembly of microlensing system parameters, priors, and model-specific parameterization. |
| [`set_model_pyLIMA.py`](set_model_pyLIMA.py) | Construct the requested pyLIMA microlensing model and its parameter layout. |
| [`set_telescopes_pyLIMA.py`](set_telescopes_pyLIMA.py) | Construct Roman/Rubin telescope objects, observation times, Rubin OpSim/MAF pointing, and caching. |
| [`functions_roman_rubin.py`](functions_roman_rubin.py) | End-to-end simulation, instrumental photometry, filtering, detection selection, truth chi-square, fitting, and data extraction. |
| [`fit_lc.py`](fit_lc.py) | Configure and execute Roman-only and joint Roman+Rubin light-curve fits. |
| [`detection_criteria.py`](detection_criteria.py) | Detection and observational-selection criteria applied to simulated light curves. |
| [`class_analysis.py`](class_analysis.py) | Event/result representation for analysis and stored outputs. |
| [`read_save.py`](read_save.py) | Simulation/fit serialization and loading of saved results. |
| [`timing_utils.py`](timing_utils.py) | Stage-by-stage runtime diagnostics. |
| [`pyLIMA_plots.py`](pyLIMA_plots.py) | pyLIMA plotting helpers. |

### Stellar populations and event catalogue

| File | Responsibility |
|---|---|
| [`stellar_population/scripts/download_trilegal.py`](stellar_population/scripts/download_trilegal.py) | Generate/download TRILEGAL stellar populations for configured Galactic-bulge cells. |
| [`stellar_population/scripts/build_genulens_trilegal_reservoir.py`](stellar_population/scripts/build_genulens_trilegal_reservoir.py) | Produce GENULENS samples/reservoir for the configured fields. |
| [`stellar_population/scripts/build_precomputed_event_catalogs.py`](stellar_population/scripts/build_precomputed_event_catalogs.py) | Match Galactic lens/source parameters and construct precomputed FFP, BH and binary-lens event Parquet files. |
| [`stellar_population/scripts/run_full_event_catalog_production.py`](stellar_population/scripts/run_full_event_catalog_production.py) | Resumable per-cell orchestration of TRILEGAL, GENULENS, event catalogue generation and assembly. |
| [`stellar_population/scripts/build_event_production_index.py`](stellar_population/scripts/build_event_production_index.py) | Build/audit global event-to-file shard index for production catalogues. |
| [`stellar_population/scripts/catalog_event_reader.py`](stellar_population/scripts/catalog_event_reader.py) | Load a deterministic catalogue event by population and global index, including simulation seed. |
| [`stellar_population/PRODUCTION_METHOD.md`](stellar_population/PRODUCTION_METHOD.md) | Detailed production method and assumptions; consult before changing catalogue inputs. |
| [`stellar_population/config/gbtds_trilegal_cells_production.csv`](stellar_population/config/gbtds_trilegal_cells_production.csv) | Configured sky-cell list; input to stellar-population production. |

### Roman F146 photometric noise — single source of truth

| File | Responsibility |
|---|---|
| [`stellar_population/noise_models/roman_f146.py`](stellar_population/noise_models/roman_f146.py) | RomanF146Noise: load and interpolate the precomputed Pandeia F146 grid, including saturation/validity state; provides sigma_mag_ab(). |
| [`stellar_population/noise_models/roman_photometry.py`](stellar_population/noise_models/roman_photometry.py) | Apply the F146 instrument model to simulated Roman telescope photometry; used by the simulation pipeline. |
| [`stellar_population/scripts/build_roman_f146_pandeia_grid.py`](stellar_population/scripts/build_roman_f146_pandeia_grid.py) | Rebuild the Pandeia input grid; not required for routine light-curve simulations or plotting. |
| [`stellar_population/scripts/audit_roman_f146_noise_grid.py`](stellar_population/scripts/audit_roman_f146_noise_grid.py) | Audit computed Pandeia grid and saturation transitions. |

### Validation, diagnostics, and figures

| File | Responsibility |
|---|---|
| [`stellar_population/scripts/smoke_roman_rubin.py`](stellar_population/scripts/smoke_roman_rubin.py) | End-to-end catalogue-event simulation and Roman-only/Roman+Rubin fit smoke tests. |
| [`stellar_population/scripts/audit_parallax_mask.py`](stellar_population/scripts/audit_parallax_mask.py) | Compare masked time-dependent parallax geometry with recomputation after photometry filtering. |
| [`stellar_population/scripts/plot_pilot_lightcurves.py`](stellar_population/scripts/plot_pilot_lightcurves.py) | Visualize selected pilot-event light curves and empirical observation intervals. |
| [`stellar_population/scripts/plot_updated_photometric_uncertainties.py`](stellar_population/scripts/plot_updated_photometric_uncertainties.py) | Experimental plotting script: verify it calls RomanF146Noise and the exact Rubin pipeline before publication use. |
| [`stellar_population/scripts/update_repository_docs.py`](stellar_population/scripts/update_repository_docs.py) | Refresh README's workflow/code map and exhaustive Python source index. |

### Roman F146: read this before modifying noise or figures

- **Current noise input:** saved Pandeia R2026.1 F146 grid, with the project-specific Rev-H `IM_66_6_V2` read pattern. The grid is already computed: do not rerun Pandeia just to make a figure.
- **Use the production interpolation:** `RomanF146Noise` in `stellar_population/noise_models/roman_f146.py`, especially `sigma_mag_ab()` for a magnitude-uncertainty curve. The simulation applies the model via `roman_photometry.py`.
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
