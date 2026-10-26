"""Pure wrappers and dead random-origin code removed in step 6.2 (not imported).

Each one only forwarded its arguments to another function; callers now
call that function directly."""


# from pipeline/sim_fit.py
def simulate_event_for_fit(
    i,
    event_params,
    path_ephemerides,
    model,
    time_window=None,
    use_roman=True,
    use_rubin=True,
    truth_parallax=True,
    rubin_pointing_mode="source",
    rubin_cache_cell_deg=None,
    apply_detection_criteria=True,
    apply_photometric_filter=True,
    rubin_saturation_mag=None,
    roman_saturation_mag=None,
):
    my_own_model, pyLIMA_parameters, decision = sim_event(
        i,
        event_params,
        path_ephemerides,
        model,
        time_window=time_window,
        use_roman=use_roman,
        use_rubin=use_rubin,
        truth_parallax=truth_parallax,
        rubin_pointing_mode=rubin_pointing_mode,
        rubin_cache_cell_deg=rubin_cache_cell_deg,
        apply_detection_criteria=apply_detection_criteria,
        apply_photometric_filter=apply_photometric_filter,
        rubin_saturation_mag=rubin_saturation_mag,
        roman_saturation_mag=roman_saturation_mag,
    )

    return my_own_model, pyLIMA_parameters, decision


# from set_model_pyLIMA.py
def model_choice(
    new_creation,
    model,
    parallax=["None", 0.0],
    BL_random_origin=True,
    BL_origin=None,
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
        origin=BL_origin,
        random_origin=(
            BL_random_origin if BL_origin is None else False
        ),
        blend_flux_parameter="ftotal",
    )


# from set_model_pyLIMA.py
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


# from fitting/fit_inputs.py
def parallax_suffix_for_filename(use_parallax):
    """
    Sufijo explícito para nombres de archivos de fit.

    Wrapper de compatibilidad sobre `fit_lc.parallax_suffix` (misma lógica).
    """
    return parallax_suffix(use_parallax)


# from fit_lc.py
def _initial_guess_candidate_keys(parameter_name):
    """
    Alias permitidos al pasar un initial_guess como diccionario.

    Esto permite usar indistintamente los nombres físicos cortos
    (t0, u0, s, q) o los nombres internos de pyLIMA para USBL
    (t_center, u_center, separation, mass_ratio).

    Mismo mapeo de alias que `_custom_bound_candidate_keys`.
    """
    return _custom_bound_candidate_keys(parameter_name)


# from analysis/simfit_records.py
def save_simulated_event_if_selected(
    i,
    ROW_G,
    ROW_T,
    path_TRILEGAL_set,
    path_GENULENS_set,
    path_to_save_model,
    my_own_model,
    pyLIMA_parameters,
    event_params,
    GENULENS_row,
    TRILEGAL_row,
):
    """
    Guarda la curva simulada usando tu función save_sim().
    """
    save_sim(
        i,
        ROW_G,
        ROW_T,
        path_TRILEGAL_set,
        path_GENULENS_set,
        path_to_save_model,
        my_own_model,
        pyLIMA_parameters,
        event_params,
        GENULENS_row,
        TRILEGAL_row,
    )


# from fitting/fit_inputs.py
def get_model_origin(pyLIMA_model):
    """
    Devuelve el origen del modelo si existe.

    Para USBL suele existir my_own_model.origin.
    Para PSPL/FSPL puede no ser relevante.
    """
    return getattr(pyLIMA_model, "origin", None)


# from fitting/run_fits.py
def _call_fit_rubin_roman_compatible(*args, **kwargs):
    """
    Call fit_rubin_roman while passing only keyword arguments supported by the
    installed version of fit_lc.py.

    This keeps the function compatible with both older versions and the newer
    version that accepts event_ra/event_dec and random_state.
    """

    try:
        parameters = inspect.signature(fit_rubin_roman).parameters
        supported_kwargs = {
            key: val
            for key, val in kwargs.items()
            if key in parameters
        }

        dropped = sorted(set(kwargs) - set(supported_kwargs))
        if dropped:
            print(
                "[warning] fit_rubin_roman does not accept these kwargs; "
                f"dropping: {dropped}",
                flush=True,
            )

        return fit_rubin_roman(*args, **supported_kwargs)

    except Exception:
        # If inspect failed for any reason, call directly so the original error
        # is not hidden.
        return fit_rubin_roman(*args, **kwargs)
