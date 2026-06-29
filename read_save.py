import h5py, os
import numpy as  np
import pandas as pd
import astropy.units as u
from astropy.table import QTable
from astropy.time import Time
from astropy.coordinates import SkyCoord

import os
import h5py
import numpy as np
import pandas as pd

# def save_sim(
#     iloc, ROW_G, ROW_T,
#     path_TRILEGAL_set, path_GENULENS_set,
#     path_to_save, my_own_model,
#     pyLIMA_parameters, event_params, GENULENS_row, TRILEGAL_row
# ):
#     print('Saving complete Simulation...')

#     genu_dict = GENULENS_row.to_dict()
#     trilegal_dict = TRILEGAL_row.to_dict()

#     # --- claves a omitir ---
#     skip_keys = {"wtj"}   # agrega aquí cualquier otra clave problemática

#     os.makedirs(path_to_save, exist_ok=True)
#     filename = os.path.join(path_to_save, f'Event_{iloc}.h5')

#     str_dt = h5py.string_dtype(encoding='utf-8')

#     with h5py.File(filename, 'w') as f:
#         # 1) Enteros
#         f.create_dataset('indices', data=np.array([iloc, ROW_G, ROW_T], dtype=np.int64))

#         # 2) Strings
#         origin_str = str(my_own_model.origin[0])
#         f.create_dataset(
#             'strings',
#             data=np.array([str(path_TRILEGAL_set), str(path_GENULENS_set), origin_str], dtype=str_dt)
#         )

#         # 3) Diccionarios existentes
#         g_pylima = f.create_group('pyLIMA_parameters')
#         for k, v in pyLIMA_parameters.items():
#             g_pylima.attrs[k] = v

#         g_tril = f.create_group('TRILEGAL_params')
#         for k, v in event_params.items():
#             g_tril.attrs[k] = v

#         # 4) Nuevos diccionarios: omitir claves en skip_keys
#         g_genu = f.create_group('GENULENS_row')
#         for k, v in genu_dict.items():
#             if k in skip_keys:
#                 print(f"[INFO] GENULENS_row: omito '{k}'")
#                 continue
#             g_genu.attrs[k] = v  # tus valores escalar/str siguen igual

#         g_trilegal = f.create_group('TRILEGAL_row')
#         for k, v in trilegal_dict.items():
#             if k in skip_keys:
#                 print(f"[INFO] TRILEGAL_row: omito '{k}'")
#                 continue
#             g_trilegal.attrs[k] = v

#         # 5) Bandas
#         for telo in my_own_model.event.telescopes:
#             table = telo.lightcurve
#             tg = f.create_group(telo.name)
#             for col in table.colnames:
#                 tg.create_dataset(col, data=table[col])

#     print('File saved:', filename)


def save_sim(
    iloc, ROW_G, ROW_T,
    path_TRILEGAL_set, path_GENULENS_set,
    path_to_save, my_own_model,
    pyLIMA_parameters, event_params,GENULENS_row,TRILEGAL_row
):
    print('Saving complete Simulation...')

    # --- Leer filas y convertir a diccionario ---

    genu_dict = GENULENS_row.iloc[0].to_dict()
    trilegal_dict = TRILEGAL_row.iloc[0].to_dict()

    # --- Preparar archivo ---
    os.makedirs(path_to_save, exist_ok=True)
    filename = os.path.join(path_to_save, f'Event_{iloc}.h5')

    str_dt = h5py.string_dtype(encoding='utf-8')

    with h5py.File(filename, 'w') as f:
        # ---- 1) Enteros ----
        f.create_dataset('indices', data=np.array([iloc, ROW_G, ROW_T], dtype=np.int64))

        # ---- 2) Strings ----
        origin_str = str(my_own_model.origin[0])
        f.create_dataset(
            'strings',
            data=np.array([str(path_TRILEGAL_set), str(path_GENULENS_set), origin_str], dtype=str_dt)
        )

        # ---- 3) Diccionarios como grupos con attrs ----
        g_pylima = f.create_group('pyLIMA_parameters')
        for k, v in pyLIMA_parameters.items():
            g_pylima.attrs[k] = v

        g_tril = f.create_group('TRILEGAL_params')
        for k, v in event_params.items():
            g_tril.attrs[k] = v

        # ---- 4) NUEVOS diccionarios genu_dict y trilegal_dict ----
        g_genu = f.create_group('GENULENS_row')
        for k, v in genu_dict.items():
            g_genu.attrs[k] = v

        g_trilegal = f.create_group('TRILEGAL_row')
        for k, v in trilegal_dict.items():
            g_trilegal.attrs[k] = v

        # ---- 5) Bandas ----
        for telo in my_own_model.event.telescopes:
            table = telo.lightcurve
            tg = f.create_group(telo.name)
            for col in table.colnames:
                tg.create_dataset(col, data=table[col])

    print('File saved:', filename)


def save_fit(iloc , path_to_save, fit_results):
    print('Saving Fit results...')
    # Save to an HDF5 file with specified names
    with h5py.File(path_to_save + 'Event_' + str(iloc) + '.h5', 'w') as file:
        dict_group = file.create_group('fit_results_'+fit_results['name'])
        for key, value in fit_results.items():
            dict_group.attrs[key] = value

    print('File saved:',path_to_save + 'Event_' + str(iloc) + '.h5' )



def read_data(path_model):
    with h5py.File(path_model, 'r') as f:
        # [iloc, ROW_G, ROW_T]
        indices = f['indices'][:].tolist()

        # [path_TRILEGAL_set, path_GENULENS_set, origin]
        strings = [
            s.decode('utf-8') if isinstance(s, (bytes, np.bytes_)) else s
            for s in f['strings'][:]
        ]

        # ---- Helper inline para decodificar diccionarios ----
        def decode_attrs(attrs):
            out = {}
            for k in attrs.keys():
                v = attrs[k]
                if isinstance(v, (bytes, np.bytes_)):
                    v = v.decode('utf-8')
                if isinstance(v, np.generic):
                    v = v.item()
                out[k] = v
            return out

        # Diccionarios existentes
        pyLIMA_parameters = decode_attrs(f['pyLIMA_parameters'].attrs)
        TRILEGAL_params   = decode_attrs(f['TRILEGAL_params'].attrs)

        # Nuevos diccionarios (si existen)
        GENULENS_row = decode_attrs(f['GENULENS_row'].attrs) if 'GENULENS_row' in f else {}
        TRILEGAL_row = decode_attrs(f['TRILEGAL_row'].attrs) if 'TRILEGAL_row' in f else {}

        # Reconstruir tablas por banda (excluyendo grupos conocidos)
        known = {
            'indices', 'strings',
            'pyLIMA_parameters', 'TRILEGAL_params',
            'GENULENS_row', 'TRILEGAL_row'
        }
        bands = {}
        for key in f.keys():
            if key in known:
                continue
            gband = f[key]
            tbl = QTable()
            for col in gband.keys():
                tbl[col] = gband[col][:]
            bands[key] = tbl

    return indices, strings, pyLIMA_parameters, TRILEGAL_params, bands, GENULENS_row, TRILEGAL_row

# def read_data(path_model):
#     with h5py.File(path_model, 'r') as f:
#         indices = f['indices'][:].tolist()              # [iloc, ROW_G, ROW_T]
#         strings = [s.decode('utf-8') if isinstance(s, bytes) else s
#                    for s in f['strings'][:]]            # [path_TRILEGAL_set, path_GENULENS_set, origin]

#         pyLIMA_parameters = {k: f['pyLIMA_parameters'].attrs[k]
#                              for k in f['pyLIMA_parameters'].attrs}
#         TRILEGAL_params   = {k: f['TRILEGAL_params'].attrs[k]
#                              for k in f['TRILEGAL_params'].attrs}

#         # Reconstruir tablas por banda (grupos en raíz excepto los conocidos)
#         known = {'indices', 'strings', 'pyLIMA_parameters', 'TRILEGAL_params'}
#         bands = {}
#         for key in f.keys():
#             if key in known:
#                 continue
#             gband = f[key]
#             tbl = QTable()
#             for col in gband.keys():
#                 tbl[col] = gband[col][:]
#             bands[key] = tbl

#     return indices, strings, pyLIMA_parameters, TRILEGAL_params, bands

# def read_data(path_model):
#     # Open the HDF5 file and load data using specified names
#     with h5py.File(path_model, 'r') as file:
#         # Load array with string with info of dataset using its name
#         info_dataset = file['Data'][:]
#         info_dataset = [file['Data'][:][0].decode('UTF-8'), file['Data'][:][1].decode('UTF-8'),
#                         [file['Data'][:][2].decode('UTF-8'), [0, 0]]]
#         # Dictionary using its name
#         pyLIMA_parameters = {key: file['pyLIMA_parameters'].attrs[key] for key in file['pyLIMA_parameters'].attrs}
#         # Load table using its name
#         bands = {}
#         for band in ("W149", "u", "g", "r", "i", "z", "y"):
#             loaded_table = QTable()
#             for col in file[band]:
#                 loaded_table[col] = file[band][col][:]
#             bands[band] = loaded_table
#         return info_dataset, pyLIMA_parameters, bands