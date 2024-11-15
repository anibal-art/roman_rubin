# Roman and Rubin microlensing simulations
Roman and Rubin simulations and analisys.

The repository contains simulations and analyses to study the impact of combining observations of Roman and Rubin microlensing events.


# Analysis Results

The `all_results` directory contains analysis results for a set of events corresponding to Free Floating Planets (FFP), Black Holes (BH), and Bound Planets (PB).

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

# Fit and simulation
The code **`functions_roman_rubin.py`** contains the fit routine and the simulation using pyLIMA and rubin_sim.


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
