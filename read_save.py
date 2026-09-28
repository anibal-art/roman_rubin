import os
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.table import QTable


def hdf5_safe_attr(v):
    """
    Convierte atributos Python/numpy/pandas a tipos compatibles con HDF5.
    """

    if v is None:
        return "None"

    if isinstance(v, Path):
        return str(v)

    if isinstance(v, bytes):
        return v

    if isinstance(v, str):
        return v

    if isinstance(v, (bool, int, float)):
        return v

    if isinstance(v, (np.bool_, np.integer, np.floating)):
        return v.item()

    # pandas NA / NaT
    try:
        if pd.isna(v) and not isinstance(v, (list, tuple, dict, np.ndarray)):
            return "NaN"
    except Exception:
        pass

    if isinstance(v, np.ndarray):
        if v.shape == ():
            return hdf5_safe_attr(v.item())

        if v.dtype.kind in ["i", "u", "f", "b"]:
            return v

        return json.dumps(v.tolist(), default=str)

    if isinstance(v, (list, tuple)):
        arr = np.asarray(v)

        if arr.dtype.kind in ["i", "u", "f", "b"]:
            return arr

        return json.dumps(list(v), default=str)

    if isinstance(v, dict):
        return json.dumps(v, default=str)

    return str(v)


def save_attrs_safe(group, dictionary, group_name=""):
    """
    Guarda un diccionario como attrs HDF5 evitando dtype object.
    """

    for k, v in dictionary.items():
        try:
            group.attrs[str(k)] = hdf5_safe_attr(v)
        except Exception as e:
            print("=" * 80)
            print("[HDF5 ATTR ERROR AFTER SANITIZE]")
            print("group:", group_name)
            print("key:", k)
            print("type original:", type(v))
            print("value original:", repr(v))
            print("safe value:", repr(hdf5_safe_attr(v)))
            print("error:", repr(e))
            print("=" * 80)

            # Último fallback: guardar siempre como string.
            group.attrs[str(k)] = str(v)


def hdf5_safe_dataset_array(x):
    """
    Convierte columnas de astropy/numpy a arrays guardables por HDF5.
    """

    # Astropy Quantity / Column con unidad
    try:
        if hasattr(x, "value"):
            x = x.value
    except Exception:
        pass

    arr = np.asarray(x)

    if arr.dtype.kind in ["i", "u", "f", "b"]:
        return arr

    if arr.dtype.kind in ["S", "U"]:
        return arr.astype(h5py.string_dtype(encoding="utf-8"))

    # Object dtype: convertir a string.
    return arr.astype(str).astype(h5py.string_dtype(encoding="utf-8"))


def save_sim(
    iloc, ROW_G, ROW_T,
    path_TRILEGAL_set, path_GENULENS_set,
    path_to_save, my_own_model,
    pyLIMA_parameters, event_params, GENULENS_row, TRILEGAL_row
):
    print("Saving complete Simulation...")

    # --- Leer filas y convertir a diccionario ---
    genu_dict = GENULENS_row.iloc[0].to_dict()
    trilegal_dict = TRILEGAL_row.iloc[0].to_dict()

    # --- Preparar archivo ---
    os.makedirs(path_to_save, exist_ok=True)
    filename = os.path.join(path_to_save, f"Event_{iloc}.h5")

    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(filename, "w") as f:

        # ---- 1) Enteros ----
        f.create_dataset(
            "indices",
            data=np.array([iloc, ROW_G, ROW_T], dtype=np.int64),
        )

        # ---- 2) Strings ----
        origin_str = str(my_own_model.origin[0])
        f.create_dataset(
            "strings",
            data=np.array(
                [
                    str(path_TRILEGAL_set),
                    str(path_GENULENS_set),
                    origin_str,
                ],
                dtype=str_dt,
            ),
        )

        # ---- 3) Diccionarios como grupos con attrs ----
        g_pylima = f.create_group("pyLIMA_parameters")
        save_attrs_safe(
            g_pylima,
            pyLIMA_parameters,
            group_name="pyLIMA_parameters",
        )

        g_tril = f.create_group("TRILEGAL_params")
        save_attrs_safe(
            g_tril,
            event_params,
            group_name="TRILEGAL_params/event_params",
        )

        # ---- 4) Diccionarios GENULENS/TRILEGAL originales ----
        g_genu = f.create_group("GENULENS_row")
        save_attrs_safe(
            g_genu,
            genu_dict,
            group_name="GENULENS_row",
        )

        g_trilegal = f.create_group("TRILEGAL_row")
        save_attrs_safe(
            g_trilegal,
            trilegal_dict,
            group_name="TRILEGAL_row",
        )

        # ---- 5) Bandas ----
        for telo in my_own_model.event.telescopes:
            table = telo.lightcurve
            tg = f.create_group(telo.name)

            for col in table.colnames:
                data = hdf5_safe_dataset_array(table[col])
                tg.create_dataset(col, data=data)

    print("File saved:", filename)


def save_fit(iloc, path_to_save, fit_results):
    print("Saving Fit results...")

    os.makedirs(path_to_save, exist_ok=True)
    filename = path_to_save + "Event_" + str(iloc) + ".h5"

    with h5py.File(filename, "w") as file:
        dict_group = file.create_group("fit_results_" + str(fit_results["name"]))
        save_attrs_safe(
            dict_group,
            fit_results,
            group_name="fit_results",
        )

    print("File saved:", filename)


def read_data(path_model):
    with h5py.File(path_model, "r") as f:

        # [iloc, ROW_G, ROW_T]
        indices = f["indices"][:].tolist()

        # [path_TRILEGAL_set, path_GENULENS_set, origin]
        strings = [
            s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else s
            for s in f["strings"][:]
        ]

        def decode_attrs(attrs):
            out = {}

            for k in attrs.keys():
                v = attrs[k]

                if isinstance(v, (bytes, np.bytes_)):
                    v = v.decode("utf-8")

                if isinstance(v, np.generic):
                    v = v.item()

                # Recuperar None si fue guardado como string.
                if v == "None":
                    v = None

                out[k] = v

            return out

        pyLIMA_parameters = decode_attrs(f["pyLIMA_parameters"].attrs)
        TRILEGAL_params = decode_attrs(f["TRILEGAL_params"].attrs)

        GENULENS_row = (
            decode_attrs(f["GENULENS_row"].attrs)
            if "GENULENS_row" in f
            else {}
        )

        TRILEGAL_row = (
            decode_attrs(f["TRILEGAL_row"].attrs)
            if "TRILEGAL_row" in f
            else {}
        )

        known = {
            "indices",
            "strings",
            "pyLIMA_parameters",
            "TRILEGAL_params",
            "GENULENS_row",
            "TRILEGAL_row",
        }

        bands = {}

        for key in f.keys():
            if key in known:
                continue

            gband = f[key]
            tbl = QTable()

            for col in gband.keys():
                data = gband[col][:]

                if data.dtype.kind == "S":
                    data = np.array([
                        x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x
                        for x in data
                    ])

                tbl[col] = data

            bands[key] = tbl

    return (
        indices,
        strings,
        pyLIMA_parameters,
        TRILEGAL_params,
        bands,
        GENULENS_row,
        TRILEGAL_row,
    )