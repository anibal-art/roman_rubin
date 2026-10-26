"""LEGACY - not used by production. Kept for reference only.

Parameter-generating inputs that functions_roman_rubin.sim_fit used to
support before the materialized catalogue became the only source of true
parameters:

* catalog_mode="trilegal_genulens": random TRILEGAL/GENULENS rows plus
  sampled lens parameters (build_event_params_from_rows);
* catalog_mode="astrodatalab_pairs": random row of a lens/source pair
  catalogue (load_pair_catalog_row, build_event_params_from_pair_row).

This module is not imported anywhere and is not expected to run as is.
The sim_fit branches that called these functions are reproduced at the
end of this file as a comment.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from ulens_params import event_param, event_param_from_pair_row


def build_event_params_from_rows(
    i,
    TRILEGAL_row,
    GENULENS_row,
    system_type,
    param_samplers=None,
    t0_range=[2460413.013828608, 2460413.013828608 + 365.25 * 8],
    custom_system=None,
):
    """
    Construye event_params a partir de TRILEGAL + GENULENS,
    permitiendo modificar las distribuciones de parámetros.
    """
    # Roman2024 TRILEGAL uses F146mag in Vega.
    # Keep W149 as the legacy internal Roman channel name.
    TRILEGAL_row = TRILEGAL_row.copy()

    roman_catalog_mag_column = None

    for _candidate in (
        "F146mag",
        "F146",
        "W149",
    ):
        if _candidate in TRILEGAL_row.columns:
            roman_catalog_mag_column = _candidate
            break

    if roman_catalog_mag_column is None:
        raise KeyError(
            "No Roman magnitude column found in TRILEGAL row. "
            "Expected one of: F146mag, F146, W149."
        )

    if "W149" not in TRILEGAL_row.columns:
        TRILEGAL_row["W149"] = (
            TRILEGAL_row[
                roman_catalog_mag_column
            ]
        )

    magstar = TRILEGAL_row[
        [
            "W149",
            "u",
            "g",
            "r",
            "i",
            "z",
            "Y",
        ]
    ].iloc[0]

    event_params = {
        **magstar.to_dict(),
        **event_param(
            i,
            TRILEGAL_row.iloc[0],
            GENULENS_row.iloc[0],
            system_type,
            t0_range=t0_range,
            custom_system=custom_system,
            param_samplers=param_samplers,
        ),
    }

    event_params[
        "roman_catalog_mag_column"
    ] = roman_catalog_mag_column

    if roman_catalog_mag_column in {
        "F146mag",
        "F146",
    }:
        event_params["F146mag"] = float(
            TRILEGAL_row[
                roman_catalog_mag_column
            ].iloc[0]
        )

    return event_params


def build_event_params_from_pair_row(
    i,
    pair_row,
    system_type,
    param_samplers=None,
    t0_range=[2460413.013828608, 2460413.013828608 + 365.25 * 8],
    custom_system=None,
):
    """
    Construye event_params usando una fila del catálogo pareado
    fuente-lente de AstroDataLab.

    Importante:
    - ra, dec son las coordenadas de la fuente.
    - lens_ra, lens_dec son las coordenadas de la lente.
    - maf_ra, maf_dec son las coordenadas que debe usar MAF/Rubin.
    """

    magstar = pair_row[
        [
            "W149",
            "u",
            "g",
            "r",
            "i",
            "z",
            "Y",
        ]
    ]

    event_params = {
        **magstar.to_dict(),
        **event_param_from_pair_row(
            i,
            pair_row,
            system_type,
            t0_range=t0_range,
            custom_system=custom_system,
            param_samplers=param_samplers,
        ),
    }

    # ------------------------------------------------------------
    # Guardar información de coordenadas y cinemática del pair row
    # ------------------------------------------------------------

    keys_to_keep = [
        "ra",
        "dec",
        "lens_ra",
        "lens_dec",
        "D_S",
        "D_L",
        "D_S_kpc",
        "D_L_kpc",
        "mu_rel",
        "theta_rad",
        "mu_source",
        "mu_lens",
        "gc",
        "galb",
        "gall",
        "lens_gc",
        "lens_galb",
        "lens_gall",
    ]

    for key in keys_to_keep:
        if key in pair_row.index:
            try:
                event_params[key] = float(pair_row[key])
            except Exception:
                event_params[key] = pair_row[key]

    if "ra" in pair_row.index and "dec" in pair_row.index:
        event_params["maf_ra"] = float(pair_row["ra"])
        event_params["maf_dec"] = float(pair_row["dec"])

    return event_params


def load_pair_catalog_row(
    pair_catalog=None,
    path_pair_catalog=None,
    row_index=None,
):
    """
    Lee una fila del catálogo pareado.

    Se puede pasar:
    - pair_catalog como DataFrame ya cargado;
    - path_pair_catalog como parquet/csv.

    row_index es 0-based.
    """

    if pair_catalog is None and path_pair_catalog is None:
        raise ValueError(
            "Tenés que pasar pair_catalog o path_pair_catalog."
        )

    if pair_catalog is None:
        path_pair_catalog = Path(path_pair_catalog)

        if path_pair_catalog.suffix == ".parquet":
            pair_catalog = pd.read_parquet(path_pair_catalog)

        elif path_pair_catalog.suffix == ".csv":
            pair_catalog = pd.read_csv(path_pair_catalog)

        else:
            raise ValueError(
                "path_pair_catalog debe ser .parquet o .csv"
            )

    if row_index is None:
        raise ValueError("Tenés que pasar row_index.")

    if row_index >= len(pair_catalog):
        raise IndexError(
            f"row_index={row_index} fuera de rango para catálogo "
            f"con {len(pair_catalog)} filas."
        )

    pair_row = pair_catalog.iloc[row_index]

    return pair_row


# ---- sim_fit branches removed from functions_roman_rubin.sim_fit ----
#     if catalog_mode == "trilegal_genulens":
#
#         ROW_G = np.random.randint(0, 10000)
#         ROW_T = np.random.randint(0, 10000)
#
#         TRILEGAL_row = pd.read_csv(
#             path_TRILEGAL_set,
#             skiprows=lambda x: x not in (0, ROW_T + 1),
#         )
#
#         GENULENS_row = pd.read_csv(
#             path_GENULENS_set,
#             skiprows=lambda x: x not in (0, ROW_G + 1),
#         )
#
#         print(TRILEGAL_row)
#
#         event_params = build_event_params_from_rows(
#             i,
#             TRILEGAL_row,
#             GENULENS_row,
#             system_type,
#             param_samplers=param_samplers,
#             t0_range=t0_range,
#             custom_system=custom_system,
#         )
#
#     elif catalog_mode == "astrodatalab_pairs":
#
#         # Para mantener reproducibilidad:
#         # se usa el mismo seed=i y se elige una fila del catálogo pareado.
#         if pair_catalog is not None:
#             n_pair_rows = len(pair_catalog)
#         else:
#             if path_pair_catalog is None:
#                 raise ValueError(
#                     "Si catalog_mode='astrodatalab_pairs', pasá "
#                     "pair_catalog o path_pair_catalog."
#                 )
#
#             path_pair_catalog_obj = Path(path_pair_catalog)
#
#             if path_pair_catalog_obj.suffix == ".parquet":
#                 n_pair_rows = len(
#                     pd.read_parquet(path_pair_catalog_obj)
#                 )
#             elif path_pair_catalog_obj.suffix == ".csv":
#                 n_pair_rows = (
#                     sum(1 for _ in open(path_pair_catalog_obj)) - 1
#                 )
#             else:
#                 raise ValueError(
#                     "path_pair_catalog debe ser .parquet o .csv."
#                 )
#
#         if n_pair_rows <= 0:
#             raise ValueError("El catálogo pareado está vacío.")
#
#         ROW_P = np.random.randint(0, n_pair_rows)
#
#         pair_row = load_pair_catalog_row(
#             pair_catalog=pair_catalog,
#             path_pair_catalog=path_pair_catalog,
#             row_index=ROW_P,
#         )
#
#         # Compatibilidad con Analysis_Event y save_sim.
#         TRILEGAL_row = pd.DataFrame([pair_row])
#         GENULENS_row = pd.DataFrame([pair_row])
#
#         ROW_T = ROW_P
#         ROW_G = ROW_P
#
#         event_params = build_event_params_from_pair_row(
#             i,
#             pair_row,
#             system_type,
#             param_samplers=param_samplers,
#             t0_range=t0_range,
#             custom_system=custom_system,
#         )
#
#         if path_TRILEGAL_set is None:
#             path_TRILEGAL_set = str(path_pair_catalog)
#
#         if path_GENULENS_set is None:
#             path_GENULENS_set = str(path_pair_catalog)
#
#     else:
#         raise ValueError(
#             "catalog_mode debe ser 'trilegal_genulens', "
#             "'custom_system' o 'astrodatalab_pairs'."
#         )
