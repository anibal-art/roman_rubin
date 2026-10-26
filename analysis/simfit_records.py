"""Turn a simulated + fitted event into output records.

* build_analysis_event: Analysis_Event with truth, fits and light curves.
* extract_data_event / new_data: per-event row (true and fitted parameters,
  uncertainties, chi2/dof, photometry summary) for Roman+Rubin and Roman.
* save_extracted_results: write that row as parquet.
"""

import numpy as np
import re
from class_analysis import Analysis_Event
from fitting.fit_inputs import fit_results_path_from_fit_object
from fitting.fit_inputs import resolve_fit_options
from pathlib import Path
from photometry.roman_rubin_photometry import photometry_summary_columns
from utils.io import save_dict_as_parquet as _save_dict_as_parquet


def fit_chi2_dof_from_n_data(data, n_data):
    """
    Compute chi2 and degrees of freedom for the exact fitted dataset.

    dof = N_data - N_fit_parameters

    N_fit_parameters is the size of pyLIMA's best_model vector, which
    contains the fitted physical parameters plus the fitted photometric
    flux parameters for the telescopes active in that fit.
    """
    if n_data is None:
        raise ValueError("n_data must be provided.")

    n_data_float = float(n_data)

    if not np.isfinite(n_data_float):
        raise ValueError(
            f"n_data must be finite, got {n_data!r}."
        )

    n_data_int = int(n_data_float)

    if n_data_int != n_data_float:
        raise ValueError(
            f"n_data must be integer-valued, got {n_data!r}."
        )

    best_model = np.asarray(
        data["best_model"],
        dtype=float,
    ).reshape(-1)

    n_fit_parameters = int(best_model.size)
    dof = n_data_int - n_fit_parameters

    if dof < 0:
        raise ValueError(
            "Negative degrees of freedom: "
            f"N_data={n_data_int}, "
            f"N_fit_parameters={n_fit_parameters}."
        )

    return {
        "chi2": float(data.get("chi2", np.nan)),
        "dof": int(dof),
        "n_data": n_data_int,
        "n_fit_parameters": n_fit_parameters,
    }


def new_data(
    Event,
    nset,
    nevent,
    cols_fit,
    data,
    n_data=None,
):
    """
    Extrae parámetros del fit de forma robusta.

    Si falla piE_MC o alguna estimación de masa por una matriz de
    covarianza mal condicionada, se guardan NaNs en lugar de detener
    toda la corrida.
    """

    import numpy as np

    new_row = dict.fromkeys(cols_fit)

    new_row["Source"] = [nevent]
    new_row["Set"] = [nset]

    # ------------------------------------------------------------
    # Parámetros ajustados y errores
    # ------------------------------------------------------------

    try:
        fit_vals = Event.dict_fit_vals(data)

        for key in fit_vals:
            new_row[key] = [fit_vals[key]]

    except Exception as error:
        print(
            "[warning] Event.dict_fit_vals failed "
            f"for Source={nevent}, Set={nset}: {repr(error)}"
        )

    # ------------------------------------------------------------
    # piE por Monte Carlo
    # ------------------------------------------------------------

    try:
        piemc = Event.piE_MC(data)

        new_row["piE"] = [piemc.get("piE", np.nan)]
        new_row["piE_err"] = [piemc.get("err_piE", np.nan)]

    except Exception as error:
        print(
            "[warning] Event.piE_MC failed "
            f"for Source={nevent}, Set={nset}: {repr(error)}"
        )

        new_row["piE"] = [np.nan]
        new_row["piE_err"] = [np.nan]

    # ------------------------------------------------------------
    # Chi2 / dof
    # ------------------------------------------------------------

    try:
        if n_data is not None:
            chichidof = fit_chi2_dof_from_n_data(
                data,
                n_data,
            )
        else:
            # Legacy fallback for callers that do not provide the
            # exact number of data points used by the fit.
            chichidof = Event.chichi_dof(data)

        new_row["chichi"] = [
            chichidof.get("chi2", np.nan)
        ]
        new_row["dof"] = [
            chichidof.get("dof", np.nan)
        ]

    except Exception as error:
        print(
            "[warning] chi2/dof extraction failed "
            f"for Source={nevent}, Set={nset}: {repr(error)}"
        )

        new_row["chichi"] = [np.nan]
        new_row["dof"] = [np.nan]

    # ------------------------------------------------------------
    # Masas derivadas
    # ------------------------------------------------------------

    if not Event.model == "PSPL":

        try:
            fitmassv3 = Event.fit_mass_v3(data)

            new_row["err_mass_v3"] = [
                fitmassv3.get("err_mass", np.nan)
            ]

            new_row["mass_v3"] = [
                fitmassv3.get("mass", np.nan)
            ]

        except Exception as error:
            print(
                "[warning] Event.fit_mass_v3 failed "
                f"for Source={nevent}, Set={nset}: {repr(error)}"
            )

            new_row["err_mass_v3"] = [np.nan]
            new_row["mass_v3"] = [np.nan]

    try:
        fitmassv2 = Event.fit_mass_v2(data)

        new_row["err_mass_v2"] = [
            fitmassv2.get("err_mass", np.nan)
        ]

        new_row["mass_v2"] = [
            fitmassv2.get("mass", np.nan)
        ]

    except Exception as error:
        print(
            "[warning] Event.fit_mass_v2 failed "
            f"for Source={nevent}, Set={nset}: {repr(error)}"
        )

        new_row["err_mass_v2"] = [np.nan]
        new_row["mass_v2"] = [np.nan]

    try:
        fitmassv1 = Event.fit_mass_v1(data)

        new_row["err_mass_v1"] = [
            fitmassv1.get("err_mass", np.nan)
        ]

        new_row["mass_v1"] = [
            fitmassv1.get("mass", np.nan)
        ]

    except Exception as error:
        print(
            "[warning] Event.fit_mass_v1 failed "
            f"for Source={nevent}, Set={nset}: {repr(error)}"
        )

        new_row["err_mass_v1"] = [np.nan]
        new_row["mass_v1"] = [np.nan]

    # ------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------

    try:
        new_row["ln_likelihood"] = [data.get("ln_likelihood", np.nan)]
    except Exception:
        new_row["ln_likelihood"] = [np.nan]

    return new_row


def extract_data_event(Event, model, nevent, system_type, nset):
    """
    Extrae datos true, fit_rr y opcionalmente fit_roman.

    Esta versión:
    - no fuerza cargar fit_roman si Roman está apagado;
    - guarda chi2_true y n_data_true desde Event.info;
    - agrega delta_chi2_true = chi2_fit - chi2_true;
    - funciona para Rubin-only;
    - tolera fits sin paralaje;
    - guarda coordenadas de fuente y metadata MAF/Rubin para mapas espaciales.
    """

    import numpy as np
    from pathlib import Path

    # ============================================================
    # Cargar simulación
    # ============================================================

    Event.load_data_sim()

    # ============================================================
    # Cargar fit_rr si no viene en memoria
    # ============================================================

    if Event.fit_rr_data is None:
        if Event.path_fit_rr is not None and Path(Event.path_fit_rr).exists():
            Event.fit_rr_data = np.load(
                Event.path_fit_rr,
                allow_pickle=True,
            ).item()
        else:
            raise FileNotFoundError(
                f"No encuentro fit_rr_data ni archivo en {Event.path_fit_rr}"
            )

    # ============================================================
    # Cargar fit_roman solo si existe
    # ============================================================

    if Event.fit_roman_data is None:
        if Event.path_fit_roman is not None and Path(Event.path_fit_roman).exists():
            Event.fit_roman_data = np.load(
                Event.path_fit_roman,
                allow_pickle=True,
            ).item()
        else:
            Event.fit_roman_data = None

    labels_params = Event.labels_params()

    # ============================================================
    # Columnas espaciales / metadata MAF
    # ============================================================

    cols_spatial = [
        "ra",
        "dec",
        "maf_ra",
        "maf_dec",
        "maf_ra_used",
        "maf_dec_used",
        "maf_source_ra",
        "maf_source_dec",
        "rubin_pointing_mode",
        "rubin_cache_cell_deg",
        "maf_mode",
        "maf_cache_mode",
        "maf_cache_source",
        "maf_n_obs",
        "D_L",
        "D_S",
        "D_L_kpc",
        "D_S_kpc",
        "mu_rel",
        "lens_ra",
        "lens_dec",
        "galb",
        "gall",
        "lens_galb",
        "lens_gall",
    ]

    cols_photometry = photometry_summary_columns()

    # ============================================================
    # Columnas true
    # ============================================================

    if "dfit" not in system_type:

        if model == "USBL":

            cols_true = (
                ["Source", "Set"]
                + labels_params
                + [
                    "Category",
                    "Category_p",
                    "mass",
                    "sel_crit",
                    "piE",
                ]
            )

        elif model == "FSPL":

            cols_true = (
                ["Source", "Set"]
                + labels_params
                + [
                    "Category",
                    "mass",
                    "sel_crit",
                    "piE",
                    "crit_FFP_Rubin",
                ]
            )

        else:

            cols_true = (
                ["Source", "Set"]
                + labels_params
                + [
                    "Category",
                    "mass",
                    "sel_crit",
                    "piE",
                    "W149",
                    "u",
                    "g",
                    "r",
                    "i",
                    "z",
                    "y",
                ]
            )

        cols_true = (
            cols_true
            + [
                "chi2_true",
                "n_data_true",
                "chi2_true_roman",
                "n_data_true_roman",
            ]
            + cols_spatial
            + cols_photometry
        )

    # ============================================================
    # Construir true table
    # ============================================================

    if "dfit" not in system_type:

        true = Event.true_values()

        new_data_true = dict.fromkeys(cols_true)

        new_data_true["Source"] = [nevent]
        new_data_true["Set"] = [nset]

        for key in Event.labels_params():
            new_data_true[key] = [true[key]]

        new_data_true["piE"] = [
            np.sqrt(
                true["piEN"] ** 2
                + true["piEE"] ** 2
            )
        ]

        if model == "USBL":

            npts_speak = Event.count_points_secon_peak()

            for f in npts_speak:
                new_data_true["anomaly_" + f] = [
                    npts_speak[f]
                ]

        new_data_true["mass"] = [
            Event.mass_true()
        ]

        npts = Event.count_points()

        for f in npts:
            new_data_true[f] = [
                npts[f]
            ]

        npts_pk = Event.count_points_peak()

        for f in npts_pk:
            new_data_true[f + "_peak"] = [
                npts_pk[f]
            ]

        npts_pk_narrow = Event.count_points_narrow_peak()

        for f in npts_pk:
            new_data_true[f + "_npeak"] = [
                npts_pk_narrow[f]
            ]

        npts_pk_left = Event.count_points_left_peak()

        for f in npts_pk:
            new_data_true[f + "_lpeak"] = [
                npts_pk_left[f]
            ]

        npts_pk_right = Event.count_points_right_peak()

        for f in npts_pk:
            new_data_true[f + "_rpeak"] = [
                npts_pk_right[f]
            ]

        # ========================================================
        # Chi2 del modelo verdadero
        # ========================================================

        if isinstance(Event.info, dict):

            chi2_true = Event.info.get(
                "chi2_true",
                np.nan,
            )

            n_data_true = Event.info.get(
                "n_data_true",
                np.nan,
            )

        else:

            chi2_true = np.nan
            n_data_true = np.nan

        new_data_true["chi2_true"] = [
            chi2_true
        ]

        new_data_true["n_data_true"] = [
            n_data_true
        ]

        if isinstance(Event.info, dict):
            new_data_true["chi2_true_roman"] = [
                Event.info.get(
                    "chi2_true_roman",
                    np.nan,
                )
            ]
            new_data_true["n_data_true_roman"] = [
                Event.info.get(
                    "n_data_true_roman",
                    np.nan,
                )
            ]
        else:
            new_data_true["chi2_true_roman"] = [np.nan]
            new_data_true["n_data_true_roman"] = [np.nan]

        # ========================================================
        # Coordenadas y metadata espacial
        # ========================================================

        computed_params = getattr(Event, "computed_params", {})

        if computed_params is None:
            computed_params = {}

        for key in cols_spatial + cols_photometry:

            try:
                value = computed_params.get(key, np.nan)
            except Exception:
                value = np.nan

            new_data_true[key] = [
                value
            ]

    else:

        new_data_true = []

    # ============================================================
    # Columnas fit
    # ============================================================

    cols_fit = (
        ["Source", "Set"]
        + labels_params
        + [f + "_err" for f in labels_params]
        + ["piE", "piE_err"]
        + ["chichi", "dof"]
        + [
            "mass_v1",
            "mass_v2",
            "err_mass_v1",
            "err_mass_v2",
        ]
    )

    if model != "PSPL":

        cols_fit = cols_fit + [
            "mass_v3",
            "err_mass_v3",
        ]

    cols_fit = cols_fit + [
        "chi2_true",
        "n_data_true",
        "delta_chi2_true",
    ]

    # ============================================================
    # Extraer info true desde Event.info
    # ============================================================

    if isinstance(Event.info, dict):

        chi2_true = Event.info.get(
            "chi2_true",
            np.nan,
        )

        n_data_true = Event.info.get(
            "n_data_true",
            np.nan,
        )

    else:

        chi2_true = np.nan
        n_data_true = np.nan

    # ============================================================
    # Fit RR / telescopios activos
    # ============================================================

    new_data_rr = new_data(
        Event,
        nset,
        nevent,
        cols_fit,
        Event.fit_rr_data,
        n_data=n_data_true,
    )

    new_data_rr["chi2_true"] = [
        chi2_true
    ]

    new_data_rr["n_data_true"] = [
        n_data_true
    ]

    try:
        new_data_rr["delta_chi2_true"] = [
            new_data_rr["chichi"][0] - chi2_true
        ]
    except Exception:
        new_data_rr["delta_chi2_true"] = [
            np.nan
        ]

    # ============================================================
    # Fit Roman-only, si existe
    # ============================================================
    if Event.fit_roman_data is not None:
        try:

            chi2_true_roman = Event.info.get(
                "chi2_true_roman",
                np.nan,
            )

            n_data_true_roman = Event.info.get(
                "n_data_true_roman",
                np.nan,
            )

            new_data_roman = new_data(
                Event,
                nset,
                nevent,
                cols_fit,
                Event.fit_roman_data,
                n_data=n_data_true_roman,
            )

            new_data_roman["chi2_true"] = [
                chi2_true_roman
            ]

            new_data_roman["n_data_true"] = [
                n_data_true_roman
            ]

            try:
                new_data_roman["delta_chi2_true"] = [
                    new_data_roman["chichi"][0]
                    - chi2_true_roman
                ]
            except Exception:
                new_data_roman["delta_chi2_true"] = [
                    np.nan
                ]

        except Exception as error:

            print(
                "[warning] Could not extract Roman-only fit data "
                f"for Source={nevent}, Set={nset}: {repr(error)}"
            )

            new_data_roman = None

    else:

        new_data_roman = None

    return new_data_true, new_data_rr, new_data_roman


def extract_nset_string(path_GENULENS_set, default="manual"):
    """
    Extrae el identificador numérico del path GENULENS.
    Si no encuentra números, devuelve default.
    """
    matches = re.findall(r"\d+", str(path_GENULENS_set))

    if len(matches) == 0:
        return default

    return matches[0]


def build_analysis_event(
    i,
    model,
    algo,
    path_to_save_model,
    path_to_save_fit,
    GENULENS_row,
    TRILEGAL_row,
    event_params,
    fit_rr,
    fit_roman,
    origin,
    lc_to_save,
    pyLIMA_parameters,
    use_roman=True,
    use_rubin=True,
    truth_parallax=True,
    fit_model=None,
    fit_parallax=None,
    fit_time_window=None,
    chi2_true=None,
):
    """
    Construye el objeto Analysis_Event.

    Mantiene model y fit_model separados.
    La opción de paralaje se guarda solamente en fit_parallax,
    no en el nombre del modelo.
    """

    model, fit_model, fit_parallax = resolve_fit_options(
        model,
        fit_model=fit_model,
        fit_parallax=fit_parallax,
        default_fit_parallax=True,
    )

    if chi2_true is None:
        chi2_true = {
            "chi2_true": np.nan,
            "n_data_true": np.nan,
            "chi2_true_by_telescope": {},
        }

    path_model = Path(path_to_save_model) / f"Event_{i}.h5"

    path_fit_rr = fit_results_path_from_fit_object(
        fit_rr,
        path_to_save_fit,
        algo,
        fit_model,
        fit_parallax,
        fallback_event_name=f"Event_RR_{i}",
    )

    if fit_roman is not None:
        path_fit_roman = fit_results_path_from_fit_object(
            fit_roman,
            path_to_save_fit,
            algo,
            fit_model,
            fit_parallax,
            fallback_event_name=f"Event_Roman_{i}",
        )
    else:
        path_fit_roman = None

    chi2_true_roman = np.nan
    n_data_true_roman = np.nan

    if fit_roman is not None:
        try:
            chi2_true_roman = fit_roman.fit_results.get(
                "chi2_true_roman",
                np.nan,
            )
            n_data_true_roman = fit_roman.fit_results.get(
                "n_data_true_roman",
                np.nan,
            )
        except Exception:
            pass

    Event = Analysis_Event(
        model,
        path_model=str(path_model),
        path_fit_rr=str(path_fit_rr),
        path_fit_roman=(
            str(path_fit_roman)
            if path_fit_roman is not None
            else None
        ),
        genulens_params=GENULENS_row.iloc[0],
        trilegal_params=TRILEGAL_row.iloc[0],
        computed_params=event_params,
        fit_rr_data=fit_rr.fit_results,
        fit_roman_data=fit_roman.fit_results if fit_roman is not None else None,
        origin=origin,
        lightcurves=lc_to_save,
        model_params=pyLIMA_parameters,
        info={
            "use_roman": use_roman,
            "use_rubin": use_rubin,
            "sim_model": model,
            "fit_model": fit_model,
            "fit_parallax": fit_parallax,
            "chi2_true": chi2_true["chi2_true"],
            "n_data_true": chi2_true["n_data_true"],
            "chi2_true_roman": chi2_true_roman,
            "n_data_true_roman": n_data_true_roman,
            "chi2_true_by_telescope": chi2_true["chi2_true_by_telescope"],
            "truth_parallax": bool(truth_parallax),
            "fit_time_window": fit_time_window,
            "apply_photometric_filter": event_params.get("apply_photometric_filter", np.nan),
            "phot_n_total": event_params.get("phot_n_total", np.nan),
            "phot_n_keep": event_params.get("phot_n_keep", np.nan),
            "phot_n_too_faint_5sigma": event_params.get("phot_n_too_faint_5sigma", np.nan),
            "phot_n_saturated": event_params.get("phot_n_saturated", np.nan),
            "phot_n_rejected": event_params.get("phot_n_rejected", np.nan),
            "path_fit_rr_actual": str(path_fit_rr),
            "path_fit_roman_actual": (
                str(path_fit_roman)
                if path_fit_roman is not None
                else None
            ),
        },
        indices=None,
    )

    return Event


def save_extracted_results(
    Event,
    model,
    i,
    system_type,
    nset_str,
    path_to_save_results,
    save_roman=True,
):
    """
    Extrae y guarda true, fit_rr y, opcionalmente, fit_roman en parquet.

    Si save_roman=False, no intenta guardar fit_roman.
    """

    new_data_true, new_data_rr, new_data_roman = extract_data_event(
        Event,
        model,
        i,
        system_type,
        nset_str,
    )

    print(new_data_true, new_data_rr, new_data_roman)

    base_dir = Path(path_to_save_results)

    path_true_dir = base_dir / "true"
    path_fitrr_dir = base_dir / "fit_rr"
    path_fitro_dir = base_dir / "fit_roman"

    path_true_file = path_true_dir / f"true_rr_{nset_str}_{i}.parquet"
    path_fitrr_file = path_fitrr_dir / f"fit_rr_{nset_str}_{i}.parquet"
    path_fitro_file = path_fitro_dir / f"fit_roman_{nset_str}_{i}.parquet"

    _save_dict_as_parquet(
        new_data_true,
        path_true_file,
        append=False,
    )

    _save_dict_as_parquet(
        new_data_rr,
        path_fitrr_file,
        append=False,
    )

    if save_roman and new_data_roman is not None:
        _save_dict_as_parquet(
            new_data_roman,
            path_fitro_file,
            append=False,
        )
        roman_msg = f"  {path_fitro_file}"
    else:
        roman_msg = "  fit_roman no guardado"

    print(
        "→ Guardado Parquet:\n"
        f"  {path_true_file}\n"
        f"  {path_fitrr_file}\n"
        f"{roman_msg}"
    )

    return new_data_true, new_data_rr, new_data_roman

