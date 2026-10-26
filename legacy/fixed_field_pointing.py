"""LEGACY - not used by production. Kept for reference only.

Fixed-field Rubin pointing removed from set_telescopes_pyLIMA.py: every
event was observed at one field (l=0.5 deg, b=-1.25 deg) instead of its own
catalogue RA/Dec. Production now always uses rubin_pointing_mode="source".
Not imported anywhere.
"""


def _init_fixed_field_rubin(path_ephemerides, opsim_db_path=None, rubin_throughputs_dir=None):
    """
    Inicializa el dataSlice Rubin del campo fijo Roman+Rubin.

    El archivo OpSim puede venir de:
    - opsim_db_path;
    - RUBIN_OPSIM_DB_PATH / RUBIN_OPSIM_DB;
    - get_baseline() si no se especifica nada.

    El cache en memoria y en disco queda separado por archivo OpSim.
    """

    global _DATASLICE
    global _RUBIN_TS
    global _DATASLICE_OPSIM_CACHE_TAG
    global _DATASLICE_OPSIM_DB_PATH
    global LAST_DATASLICE_INFO

    _init_common_roman_rubin(path_ephemerides, rubin_throughputs_dir=rubin_throughputs_dir)

    opsim_path = _resolve_opsim_db_path(opsim_db_path)
    opsim_tag = _opsim_cache_tag(opsim_path)

    if (
        _DATASLICE is not None
        and _RUBIN_TS is not None
        and _DATASLICE_OPSIM_CACHE_TAG == opsim_tag
    ):
        Ra, Dec = _roman_rubin_field_coordinates()

        LAST_DATASLICE_INFO = {
            "mode": "fixed",
            "Ra": float(Ra),
            "Dec": float(Dec),
            "source": "memory_cache",
            "n_obs": int(len(_DATASLICE)),
            "opsim_db_path": str(opsim_path),
            "opsim_cache_tag": opsim_tag,
        }

        return

    lsst_filterlist = "ugrizy"

    Ra, Dec = _roman_rubin_field_coordinates()

    print("=" * 80)
    print("[MAF] Fixed Roman+Rubin field")
    print(f"[MAF] RA       = {Ra:.8f} deg")
    print(f"[MAF] Dec      = {Dec:.8f} deg")
    print(f"[MAF] OpSim DB = {opsim_path}")
    print(f"[MAF] cache    = {opsim_tag}")
    print("=" * 80)

    dataSlice = _load_or_build_dataslice(
        Ra,
        Dec,
        opsim_db_path=opsim_path,
    )

    rubin_ts = _build_rubin_ts(
        dataSlice,
        lsst_filterlist=lsst_filterlist,
    )

    _DATASLICE = dataSlice
    _RUBIN_TS = rubin_ts
    _DATASLICE_OPSIM_CACHE_TAG = opsim_tag
    _DATASLICE_OPSIM_DB_PATH = str(opsim_path)

    LAST_DATASLICE_INFO = {
        "mode": "fixed",
        "Ra": float(Ra),
        "Dec": float(Dec),
        "source": "disk_or_maf_cache",
        "n_obs": int(len(dataSlice)),
        "opsim_db_path": str(opsim_path),
        "opsim_cache_tag": opsim_tag,
    }


def _roman_rubin_field_coordinates():
    """
    Coordenadas del campo usado para Roman/Rubin.
    """
    gc = SkyCoord(
        l=0.5 * u.degree,
        b=-1.25 * u.degree,
        frame="galactic",
    )

    Ra = gc.icrs.ra.value
    Dec = gc.icrs.dec.value

    return Ra, Dec


def _init_tel_roman_rubin(path_ephemerides, opsim_db_path=None, rubin_throughputs_dir=None):
    """
    Inicializa ingredientes Roman/Rubin una sola vez por proceso.
    No aplica todavía use_roman/use_rubin.

    Mantiene compatibilidad con llamadas viejas. Si no se pasa
    opsim_db_path, usa RUBIN_OPSIM_DB_PATH o get_baseline().
    """
    global _LSST_BANDPASS
    global _DATASLICE
    global _RUBIN_TS
    global _ROMAN_MAG
    global _DATASLICE_OPSIM_CACHE_TAG
    global _DATASLICE_OPSIM_DB_PATH

    opsim_path = _resolve_opsim_db_path(opsim_db_path)
    opsim_tag = _opsim_cache_tag(opsim_path)

    if (
        _LSST_BANDPASS is not None
        and _DATASLICE is not None
        and _RUBIN_TS is not None
        and _ROMAN_MAG is not None
        and _DATASLICE_OPSIM_CACHE_TAG == opsim_tag
    ):
        return

    lsst_filterlist = "ugrizy"

    Ra, Dec = _roman_rubin_field_coordinates()

    LSST_BandPass = _load_lsst_bandpasses(
        lsst_filterlist=lsst_filterlist,
        path_che=rubin_throughputs_dir,
    )

    dataSlice = _load_or_build_dataslice(
        Ra,
        Dec,
        opsim_db_path=opsim_path,
    )

    rubin_ts = _build_rubin_ts(
        dataSlice,
        lsst_filterlist=lsst_filterlist,
    )

    roman_mag = _load_or_build_roman_template()

    _LSST_BANDPASS = LSST_BandPass
    _DATASLICE = dataSlice
    _RUBIN_TS = rubin_ts
    _ROMAN_MAG = roman_mag
    _DATASLICE_OPSIM_CACHE_TAG = opsim_tag
    _DATASLICE_OPSIM_DB_PATH = str(opsim_path)


# def _build_event_template(
#     Ra,
#     Dec,
#     roman_mag,
#     rubin_ts,
#     path_ephemerides,
#     lsst_filterlist="ugrizy",
# ):
#     """
#     Construye el Event base con Roman + Rubin.
#     """
#     my_own_creation = event.Event(
#         ra=Ra,
#         dec=Dec,
#     )

#     my_own_creation.name = "An event observed by Roman"

#     Roman_tot = telescopes.Telescope(
#         name="W149",
#         camera_filter="W149",
#         location="Space",
#         lightcurve=roman_mag,
#         lightcurve_names=[
#             "time",
#             "mag",
#             "err_mag",
#         ],
#         lightcurve_units=[
#             "d",
#             "mag",
#             "mag",
#         ],
#     )

#     ephemerides = np.load(path_ephemerides)

#     Roman_tot.spacecraft_name = "L2"
#     Roman_tot.spacecraft_positions = {
#         "astrometry": [],
#         "photometry": ephemerides,
#     }

#     my_own_creation.telescopes.append(Roman_tot)

#     for band in lsst_filterlist:
#         lsst_telescope = telescopes.Telescope(
#             name=band,
#             camera_filter=band,
#             location="Earth",
#             lightcurve=rubin_ts[band],
#             lightcurve_names=[
#                 "time",
#                 "mag",
#                 "err_mag",
#             ],
#             lightcurve_units=[
#                 "d",
#                 "mag",
#                 "mag",
#             ],
#         )

#         my_own_creation.telescopes.append(lsst_telescope)

#     return my_own_creation


# def _init_tel_roman_rubin(path_ephemerides):
#     """
#     Construye LSST_BandPass, dataSlice, rubin_ts, roman_mag y el Event base
#     una sola vez por proceso.

#     Esta función siempre construye la versión completa.
#     El recorte temporal se aplica después en tel_roman_rubin().
#     """
#     global _LSST_BANDPASS
#     global _DATASLICE
#     global _RUBIN_TS
#     global _ROMAN_MAG
#     global _EVENT_TEMPLATE

#     if _EVENT_TEMPLATE is not None:
#         return

#     lsst_filterlist = "ugrizy"

#     Ra, Dec = _roman_rubin_field_coordinates()

#     LSST_BandPass = _load_lsst_bandpasses(
#         lsst_filterlist=lsst_filterlist,
#         path_che="${RUBIN_SIM_DATA_DIR}/throughputs/baseline/",
#     )

#     dataSlice = _load_or_build_dataslice(
#         Ra,
#         Dec,
#     )

#     rubin_ts = _build_rubin_ts(
#         dataSlice,
#         lsst_filterlist=lsst_filterlist,
#     )

#     roman_mag = _load_or_build_roman_template()

#     my_own_creation = _build_event_template(
#         Ra,
#         Dec,
#         roman_mag,
#         rubin_ts,
#         path_ephemerides,
#         lsst_filterlist=lsst_filterlist,
#     )

#     _LSST_BANDPASS = LSST_BandPass
#     _DATASLICE = dataSlice
#     _RUBIN_TS = rubin_ts
#     _ROMAN_MAG = roman_mag
#     _EVENT_TEMPLATE = my_own_creation


# def tel_roman_rubin(path_ephemerides, time_window=None):
#     """
#     Devuelve una copia del Event base Roman+Rubin.

#     Parameters
#     ----------
#     path_ephemerides : str
#         Path a las efemérides Roman.

#     time_window : None or tuple
#         Si es None, devuelve la curva completa, igual que la función original.
#         Si es una tupla (time_min, time_max), recorta Roman, Rubin y dataSlice.

#         Los tiempos deben estar en JD.

#     Returns
#     -------
#     my_event : pyLIMA event
#         Evento Roman+Rubin.

#     dataSlice : numpy structured array
#         dataSlice completo o recortado.

#     LSST_BandPass : dict
#         Bandpasses Rubin.
#     """
#     _init_tel_roman_rubin(path_ephemerides)

#     time_window = _validate_time_window(time_window)

#     my_event = copy.deepcopy(_EVENT_TEMPLATE)

#     if time_window is not None:
#         my_event = apply_time_window_to_event(
#             my_event,
#             time_window=time_window,
#         )

#         dataSlice = slice_dataslice_by_time(
#             _DATASLICE,
#             time_window=time_window,
#         )

#     else:
#         dataSlice = _DATASLICE

#     return my_event, dataSlice, _LSST_BANDPASS


# ---- rama 'fixed' quitada de tel_roman_rubin ----
#     # ------------------------------------------------------------
#     # Si Roman+Rubin están encendidos, forzamos campo fijo.
#     # Esto preserva tu simulación principal.
#     # ------------------------------------------------------------
#
#     # El modo "fixed" se conserva por defecto.
#     # El modo "source" también se permite con Roman+Rubin.
#
#     # ------------------------------------------------------------
#     # Modo campo fijo Roman+Rubin
#     # ------------------------------------------------------------
#
#     if rubin_pointing_mode == "fixed":
#
#         _init_fixed_field_rubin(
#             path_ephemerides,
#             opsim_db_path=opsim_db_path,
#             rubin_throughputs_dir=rubin_throughputs_dir,
#         )
#
#         event_Ra, event_Dec = _roman_rubin_field_coordinates()
#
#         dataSlice_full = _DATASLICE
#         rubin_ts = _RUBIN_TS
#
#     # ------------------------------------------------------------
#     # Modo coordenadas de fuente TRILEGAL
#     # ------------------------------------------------------------
#
#     elif rubin_pointing_mode == "source":
