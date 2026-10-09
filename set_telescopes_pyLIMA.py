#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:36:34 2026

@author: anibal
"""
import numpy as np
import os, sys, re, copy, math, time, hashlib
import pandas as pd
from pathlib import Path

# Get the directory where the script is located
script_dir = Path(__file__).parent
home_dir = os.path.expanduser("~")

from rubin_sim.phot_utils.photometric_parameters import PhotometricParameters
from rubin_sim.phot_utils.signaltonoise import calc_mag_error_m5
from rubin_sim.phot_utils.bandpass import Bandpass
import rubin_sim.maf as maf
from rubin_sim.data import get_baseline

# astropy
import astropy.units as u
from astropy.table import QTable
from astropy.time import Time
from astropy.coordinates import SkyCoord

# --- Fix para IERS y sidereal_time de Astropy ---
from astropy.utils import iers
iers.conf.auto_max_age = None
iers.conf.auto_download = False  # No intenta descargar
iers.conf.iers_degraded_accuracy = 'warn'

# pyLIMA
from pyLIMA import event
from pyLIMA import telescopes
# from pyLIMA.toolbox import time_series
from pyLIMA.simulations import simulator
# from pyLIMA.models import PSBL_model
# from pyLIMA.models import USBL_model
# from pyLIMA.models import FSPLarge_model
# from pyLIMA.models import PSPL_model
# from pyLIMA.fits import TRF_fit
# from pyLIMA.fits import DE_fit
# from pyLIMA.fits import MCMC_fit
# from pyLIMA.outputs import pyLIMA_plots
# from pyLIMA.outputs import file_outputs

from class_analysis import Analysis_Event
from ulens_params import microlensing_params, event_param
# import multiprocessing as mul
# import h5py
from detection_criteria import filter5points, deviation_from_constant, has_consecutive_numbers, filter_band, mag, debug_nsigma_global
from read_save import save_sim, save_fit, read_data




# ================================================================
#  Guardado a Parquet
# ================================================================
from utils.io import save_dict_as_parquet as _save_dict_as_parquet


# ================================================================
#  Persistencia a disco (cache en archivos .npz)
# ================================================================
_CACHE_DIR = script_dir / ".rr_cache"
_CACHE_DIR.mkdir(exist_ok=True)

def _npz_path(name: str) -> Path:
    return _CACHE_DIR / f"{name}.npz"

def _lock_path(name: str) -> Path:
    return _CACHE_DIR / f"{name}.lock"

def _acquire_lock(name: str, timeout=60.0, sleep=0.1):
    """File-lock best-effort para evitar que múltiples procesos generen el mismo artefacto a la vez."""
    start = time.time()
    lock = _lock_path(name)
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return  # lock adquirido
        except FileExistsError:
            if time.time() - start > timeout:
                return  # seguimos; quizá otro proceso ya lo dejó escrito
            time.sleep(sleep)

def _release_lock(name: str):
    try:
        os.remove(_lock_path(name))
    except FileNotFoundError:
        pass


# ================================================================
#  Helpers de arrays
# ================================================================
def _structured_to_matrix(arr, fields):
    """Convierte un array estructurado 1D a matriz 2D por campos; si ya es 2D, lo retorna."""
    if hasattr(arr, "dtype") and getattr(arr.dtype, "names", None):
        # asegurar que todos los campos existen
        fields = [f for f in fields if f in arr.dtype.names]
        return np.column_stack([arr[f] for f in fields])
    # si no es estructurado pero es 1D, forzamos 2D si se puede
    arr = np.asarray(arr)
    if arr.ndim == 1:
        # no sabemos columnas: devolvemos con shape (N,1) para que falle antes que silently wrong
        return arr.reshape(-1, 1)
    return arr


# ============================================================
# Cache global por proceso
# ============================================================

_LSST_BANDPASS = None

# Cache fijo para el campo Roman+Rubin actual
_DATASLICE = None
_RUBIN_TS = None
_ROMAN_MAG = None


# ============================================================
# Process-local pristine Telescope-template cache
# ============================================================
#
# These objects are cached BEFORE any Event-dependent parallax
# calculation. Every caller receives a deepcopy.
#
# Event RA/Dec, North/East and deltas_positions are never cached.

_ROMAN_TELESCOPE_TEMPLATE_CACHE = {}
_RUBIN_TELESCOPE_TEMPLATE_CACHE = {}
_ROMAN_EPHEMERIDES_TEMPLATE_CACHE = {}

_TELESCOPE_TEMPLATE_CACHE_STATS = {
    "roman_hits": 0,
    "roman_misses": 0,
    "rubin_hits": 0,
    "rubin_misses": 0,
}


def _telescope_array_signature(values):
    """
    Exact SHA256 signature of an input numerical array.
    """
    arr = np.ascontiguousarray(
        np.asarray(values)
    )

    if arr.dtype.hasobject:
        raise TypeError(
            "Object arrays are not valid Telescope-cache inputs."
        )

    h = hashlib.sha256()
    h.update(arr.dtype.str.encode("ascii"))
    h.update(repr(arr.shape).encode("ascii"))
    h.update(arr.tobytes(order="C"))

    return h.hexdigest()


def _load_cached_roman_ephemerides(path_ephemerides):
    """
    Load one unchanged Roman ephemerides file once per process.
    """
    path = os.path.abspath(
        os.path.expanduser(
            str(path_ephemerides)
        )
    )

    st = os.stat(path)

    file_key = (
        path,
        int(st.st_size),
        int(st.st_mtime_ns),
    )

    cached = (
        _ROMAN_EPHEMERIDES_TEMPLATE_CACHE
        .get(file_key)
    )

    if cached is None:

        ephemerides = np.load(path)

        cached = (
            ephemerides,
            _telescope_array_signature(
                ephemerides
            ),
        )

        _ROMAN_EPHEMERIDES_TEMPLATE_CACHE[
            file_key
        ] = cached

    return cached


def _get_cached_roman_telescope(
    roman_mag,
    path_ephemerides,
):
    """
    Return a fresh Roman Telescope copied from a pristine template.
    """
    ephemerides, ephemerides_signature = (
        _load_cached_roman_ephemerides(
            path_ephemerides
        )
    )

    key = (
        _telescope_array_signature(
            roman_mag
        ),
        ephemerides_signature,
    )

    template = (
        _ROMAN_TELESCOPE_TEMPLATE_CACHE
        .get(key)
    )

    if template is None:

        template = telescopes.Telescope(
            name="W149",
            camera_filter="W149",
            location="Space",
            lightcurve=roman_mag,
            lightcurve_names=[
                "time",
                "mag",
                "err_mag",
            ],
            lightcurve_units=[
                "d",
                "mag",
                "mag",
            ],
        )

        template.spacecraft_name = "L2"

        template.spacecraft_positions = {
            "astrometry": [],
            "photometry": ephemerides,
        }

        _ROMAN_TELESCOPE_TEMPLATE_CACHE[
            key
        ] = template

        _TELESCOPE_TEMPLATE_CACHE_STATS[
            "roman_misses"
        ] += 1

    else:

        _TELESCOPE_TEMPLATE_CACHE_STATS[
            "roman_hits"
        ] += 1

    return copy.deepcopy(
        template
    )


def _get_cached_rubin_telescope(
    band,
    lightcurve,
):
    """
    Return a fresh Rubin Telescope copied from a pristine template.

    The cache key uses the complete input [time, mag, err_mag]
    array, not only its timestamps.
    """
    band = str(band)

    key = (
        band,
        _telescope_array_signature(
            lightcurve
        ),
    )

    template = (
        _RUBIN_TELESCOPE_TEMPLATE_CACHE
        .get(key)
    )

    if template is None:

        template = telescopes.Telescope(
            name=band,
            camera_filter=band,
            location="Earth",
            lightcurve=lightcurve,
            lightcurve_names=[
                "time",
                "mag",
                "err_mag",
            ],
            lightcurve_units=[
                "d",
                "mag",
                "mag",
            ],
        )

        _RUBIN_TELESCOPE_TEMPLATE_CACHE[
            key
        ] = template

        _TELESCOPE_TEMPLATE_CACHE_STATS[
            "rubin_misses"
        ] += 1

    else:

        _TELESCOPE_TEMPLATE_CACHE_STATS[
            "rubin_hits"
        ] += 1

    return copy.deepcopy(
        template
    )


def get_telescope_template_cache_stats():
    """
    Process-local Telescope-template cache diagnostics.
    """
    out = dict(
        _TELESCOPE_TEMPLATE_CACHE_STATS
    )

    out["roman_cached_templates"] = len(
        _ROMAN_TELESCOPE_TEMPLATE_CACHE
    )

    out["rubin_cached_templates"] = len(
        _RUBIN_TELESCOPE_TEMPLATE_CACHE
    )

    out["roman_ephemerides_cached_files"] = len(
        _ROMAN_EPHEMERIDES_TEMPLATE_CACHE
    )

    return out


def reset_telescope_template_cache():
    """
    Clear only Telescope-template optimization caches.
    """
    _ROMAN_TELESCOPE_TEMPLATE_CACHE.clear()
    _RUBIN_TELESCOPE_TEMPLATE_CACHE.clear()
    _ROMAN_EPHEMERIDES_TEMPLATE_CACHE.clear()

    for key in (
        "roman_hits",
        "roman_misses",
        "rubin_hits",
        "rubin_misses",
    ):
        _TELESCOPE_TEMPLATE_CACHE_STATS[
            key
        ] = 0

_DATASLICE_OPSIM_CACHE_TAG = None
_DATASLICE_OPSIM_DB_PATH = None

# Cache opcional para Rubin-only en coordenadas de fuente
_RUBIN_SOURCE_CACHE = {}

# Para verificación/debug
LAST_DATASLICE_INFO = {}


# ============================================================
# Paths Rubin configurables
# ============================================================
#
# La idea es que set_telescopes_pyLIMA.py NO tenga paths hardcodeados
# a /home/anibal. En una PC distinta o en cluster, el runner debe leer
# estos paths del config y pasarlos con configure_rubin_paths(...), o bien
# exportarlos como variables de entorno:
#
#   RUBIN_SIM_DATA_DIR=/ruta/rubin_sim_data
#   RUBIN_OPSIM_DB_PATH=/ruta/opsim.db
#   RUBIN_THROUGHPUTS_DIR=/ruta/rubin_sim_data/throughputs/baseline
#
# Si no se define nada, se mantiene un fallback razonable:
#   Path.home()/rubin_sim_data
# y, para la OpSim, get_baseline().
# ============================================================

_RUBIN_SIM_DATA_DIR = None
_RUBIN_THROUGHPUTS_DIR = None
_RUBIN_OPSIM_DB_PATH = None


def _expand_path_string(path_like, extra_env=None):
    """
    Expande ~, variables de entorno y placeholders tipo ${VAR}.

    No depende de que os.environ tenga previamente todas las variables:
    extra_env permite pasar, por ejemplo,
        {"RUBIN_SIM_DATA_DIR": "/cluster/.../rubin_sim_data"}

    Ejemplos
    --------
    ${HOME}/rubin_sim_data
    ${RUBIN_SIM_DATA_DIR}/throughputs/baseline
    $RUBIN_SIM_DATA_DIR/sim_baseline/baseline.db
    """

    if path_like is None:
        return None

    env = dict(os.environ)

    if extra_env is not None:
        env.update({
            str(k): str(v)
            for k, v in extra_env.items()
            if v is not None
        })

    env.setdefault("HOME", str(Path.home()))

    if _RUBIN_SIM_DATA_DIR is not None:
        env.setdefault("RUBIN_SIM_DATA_DIR", str(_RUBIN_SIM_DATA_DIR))

    if _RUBIN_THROUGHPUTS_DIR is not None:
        env.setdefault("RUBIN_THROUGHPUTS_DIR", str(_RUBIN_THROUGHPUTS_DIR))

    if _RUBIN_OPSIM_DB_PATH is not None:
        env.setdefault("RUBIN_OPSIM_DB_PATH", str(_RUBIN_OPSIM_DB_PATH))

    s = str(path_like)

    def repl_braced(match):
        key = match.group(1)
        return env.get(key, match.group(0))

    def repl_plain(match):
        key = match.group(1)
        return env.get(key, match.group(0))

    s = re.sub(r"\$\{([^}]+)\}", repl_braced, s)
    s = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", repl_plain, s)
    s = os.path.expanduser(s)

    return Path(s).resolve()


def _resolve_rubin_sim_data_dir(rubin_sim_data_dir=None):
    """
    Resuelve el directorio raíz de rubin_sim_data.

    Prioridad:
    1. argumento explícito;
    2. configure_rubin_paths(...);
    3. variable RUBIN_SIM_DATA_DIR;
    4. Path.home()/rubin_sim_data.
    """

    candidate = rubin_sim_data_dir

    if candidate in (None, "", "auto", "default"):
        candidate = _RUBIN_SIM_DATA_DIR

    if candidate in (None, "", "auto", "default"):
        candidate = os.environ.get("RUBIN_SIM_DATA_DIR", "")

    if candidate in (None, "", "auto", "default"):
        candidate = Path.home() / "rubin_sim_data"

    return _expand_path_string(candidate)


def _resolve_rubin_throughputs_dir(rubin_throughputs_dir=None):
    """
    Resuelve el directorio de throughputs Rubin.

    Prioridad:
    1. argumento explícito;
    2. configure_rubin_paths(...);
    3. variable RUBIN_THROUGHPUTS_DIR;
    4. ${RUBIN_SIM_DATA_DIR}/throughputs/baseline.
    """

    sim_data_dir = _resolve_rubin_sim_data_dir()

    candidate = rubin_throughputs_dir

    if candidate in (None, "", "auto", "default"):
        candidate = _RUBIN_THROUGHPUTS_DIR

    if candidate in (None, "", "auto", "default"):
        candidate = os.environ.get("RUBIN_THROUGHPUTS_DIR", "")

    if candidate in (None, "", "auto", "default"):
        candidate = sim_data_dir / "throughputs" / "baseline"

    return _expand_path_string(
        candidate,
        extra_env={
            "RUBIN_SIM_DATA_DIR": sim_data_dir,
        },
    )


def configure_rubin_paths(
    rubin_sim_data_dir=None,
    rubin_throughputs_dir=None,
    rubin_opsim_db_path=None,
    reset_caches=True,
    validate=False,
):
    """
    Configura paths Rubin para este módulo.

    Esta función está pensada para ser llamada desde el runner justo después
    de leer el config.

    Parameters
    ----------
    rubin_sim_data_dir : str or Path or None
        Raíz de rubin_sim_data.

    rubin_throughputs_dir : str or Path or None
        Directorio que contiene total_u.dat, total_g.dat, ...

    rubin_opsim_db_path : str or Path or None
        DB OpSim a usar.

    reset_caches : bool
        Si True, limpia los caches en memoria para evitar mezclar bandpasses
        o dataSlices construidos con otra configuración.

    validate : bool
        Si True, verifica existencia de paths y archivos mínimos.
    """

    global _RUBIN_SIM_DATA_DIR
    global _RUBIN_THROUGHPUTS_DIR
    global _RUBIN_OPSIM_DB_PATH

    global _LSST_BANDPASS
    global _DATASLICE
    global _RUBIN_TS
    global _ROMAN_MAG
    global _DATASLICE_OPSIM_CACHE_TAG
    global _DATASLICE_OPSIM_DB_PATH
    global _RUBIN_SOURCE_CACHE

    if rubin_sim_data_dir not in (None, "", "auto", "default"):
        _RUBIN_SIM_DATA_DIR = _expand_path_string(rubin_sim_data_dir)
        os.environ["RUBIN_SIM_DATA_DIR"] = str(_RUBIN_SIM_DATA_DIR)

    elif "RUBIN_SIM_DATA_DIR" in os.environ:
        _RUBIN_SIM_DATA_DIR = _expand_path_string(os.environ["RUBIN_SIM_DATA_DIR"])

    if rubin_throughputs_dir not in (None, "", "auto", "default"):
        _RUBIN_THROUGHPUTS_DIR = _expand_path_string(
            rubin_throughputs_dir,
            extra_env={
                "RUBIN_SIM_DATA_DIR": _resolve_rubin_sim_data_dir(),
            },
        )
        os.environ["RUBIN_THROUGHPUTS_DIR"] = str(_RUBIN_THROUGHPUTS_DIR)

    elif "RUBIN_THROUGHPUTS_DIR" in os.environ:
        _RUBIN_THROUGHPUTS_DIR = _expand_path_string(
            os.environ["RUBIN_THROUGHPUTS_DIR"],
            extra_env={
                "RUBIN_SIM_DATA_DIR": _resolve_rubin_sim_data_dir(),
            },
        )

    elif _RUBIN_SIM_DATA_DIR is not None:
        _RUBIN_THROUGHPUTS_DIR = (
            _RUBIN_SIM_DATA_DIR / "throughputs" / "baseline"
        )
        os.environ["RUBIN_THROUGHPUTS_DIR"] = str(_RUBIN_THROUGHPUTS_DIR)

    if rubin_opsim_db_path not in (None, "", "auto", "default"):
        _RUBIN_OPSIM_DB_PATH = _expand_path_string(
            rubin_opsim_db_path,
            extra_env={
                "RUBIN_SIM_DATA_DIR": _resolve_rubin_sim_data_dir(),
            },
        )
        os.environ["RUBIN_OPSIM_DB_PATH"] = str(_RUBIN_OPSIM_DB_PATH)

    elif "RUBIN_OPSIM_DB_PATH" in os.environ:
        _RUBIN_OPSIM_DB_PATH = _expand_path_string(
            os.environ["RUBIN_OPSIM_DB_PATH"],
            extra_env={
                "RUBIN_SIM_DATA_DIR": _resolve_rubin_sim_data_dir(),
            },
        )

    if reset_caches:
        _LSST_BANDPASS = None
        _DATASLICE = None
        _RUBIN_TS = None
        # Roman no depende de rubin_sim_data; lo conservamos.
        # _ROMAN_MAG = None
        _DATASLICE_OPSIM_CACHE_TAG = None
        _DATASLICE_OPSIM_DB_PATH = None
        _RUBIN_SOURCE_CACHE = {}

    sim_data_dir = _resolve_rubin_sim_data_dir()
    throughputs_dir = _resolve_rubin_throughputs_dir()
    opsim_path = None

    try:
        opsim_path = _resolve_opsim_db_path(rubin_opsim_db_path)
    except Exception:
        if validate:
            raise

    if validate:
        if not sim_data_dir.exists():
            raise FileNotFoundError(
                f"No existe RUBIN_SIM_DATA_DIR: {sim_data_dir}"
            )

        if not throughputs_dir.exists():
            raise FileNotFoundError(
                f"No existe RUBIN_THROUGHPUTS_DIR: {throughputs_dir}"
            )

        for band in "ugrizy":
            dat = throughputs_dir / f"total_{band}.dat"
            gz = Path(str(dat) + ".gz")

            if not dat.exists() and not gz.exists():
                raise FileNotFoundError(
                    "No encontré throughput Rubin:\n"
                    f"{dat}\n"
                    f"ni\n"
                    f"{gz}"
                )

        if opsim_path is not None and not opsim_path.exists():
            raise FileNotFoundError(
                f"No existe RUBIN_OPSIM_DB_PATH: {opsim_path}"
            )

    print("[set_telescopes_pyLIMA] RUBIN_SIM_DATA_DIR    =", sim_data_dir)
    print("[set_telescopes_pyLIMA] RUBIN_THROUGHPUTS_DIR =", throughputs_dir)

    if opsim_path is not None:
        print("[set_telescopes_pyLIMA] RUBIN_OPSIM_DB_PATH   =", opsim_path)

    return {
        "rubin_sim_data_dir": str(sim_data_dir),
        "rubin_throughputs_dir": str(throughputs_dir),
        "rubin_opsim_db_path": str(opsim_path) if opsim_path is not None else "",
    }


def _resolve_opsim_db_path(opsim_db_path=None):
    """
    Devuelve el path del archivo OpSim/MAF a usar.

    Prioridad:
    1. argumento explícito opsim_db_path;
    2. configure_rubin_paths(..., rubin_opsim_db_path=...);
    3. variable de entorno RUBIN_OPSIM_DB_PATH;
    4. variable de entorno RUBIN_OPSIM_DB;
    5. get_baseline() de rubin_sim.data.

    Acepta placeholders tipo:
        ${HOME}
        ${RUBIN_SIM_DATA_DIR}
    """

    if opsim_db_path in (None, "", "default", "auto"):
        opsim_db_path = _RUBIN_OPSIM_DB_PATH

    if opsim_db_path in (None, "", "default", "auto"):
        opsim_db_path = os.environ.get(
            "RUBIN_OPSIM_DB_PATH",
            os.environ.get("RUBIN_OPSIM_DB", ""),
        ).strip()

    if opsim_db_path in (None, "", "default", "auto"):
        opsim_db_path = get_baseline()

    sim_data_dir = _resolve_rubin_sim_data_dir()

    path = _expand_path_string(
        opsim_db_path,
        extra_env={
            "RUBIN_SIM_DATA_DIR": sim_data_dir,
        },
    )

    if not path.exists():
        raise FileNotFoundError(
            "No existe el archivo OpSim especificado:\n"
            f"{path}\n\n"
            "Definí rubin.opsim_db_path en el config, pasá opsim_db_path, "
            "llamá configure_rubin_paths(...), o exportá "
            "RUBIN_OPSIM_DB_PATH=/ruta/al/opsim.db."
        )

    return path.resolve()

def _opsim_cache_tag(opsim_db_path=None):
    """
    Crea una etiqueta corta para el cache del dataSlice.

    Importante: el cache debe depender del archivo OpSim. Si no, al cambiar
    de baseline se podría reutilizar un dataSlice viejo generado con otra DB.
    """

    path = _resolve_opsim_db_path(opsim_db_path)
    stat = path.stat()

    key = (
        f"{path}|"
        f"size={stat.st_size}|"
        f"mtime={stat.st_mtime_ns}"
    )

    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)

    return f"{safe_stem}_{digest}"


def _opsim_info(opsim_db_path=None):
    path = _resolve_opsim_db_path(opsim_db_path)
    return {
        "opsim_db_path": str(path),
        "opsim_cache_tag": _opsim_cache_tag(path),
    }


def _init_common_roman_rubin(path_ephemerides, rubin_throughputs_dir=None):
    """
    Inicializa ingredientes comunes:
    - LSST bandpasses
    - plantilla Roman

    No construye ningún dataSlice de Rubin.
    """

    global _LSST_BANDPASS
    global _ROMAN_MAG

    lsst_filterlist = "ugrizy"

    if _LSST_BANDPASS is None:
        _LSST_BANDPASS = _load_lsst_bandpasses(
            lsst_filterlist=lsst_filterlist,
            path_che=rubin_throughputs_dir,
        )

    if _ROMAN_MAG is None:
        _ROMAN_MAG = _load_or_build_roman_template()


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

def _quantize_sky_position(Ra, Dec, cache_cell_deg=None):
    """
    Devuelve una coordenada para cachear/consultar MAF.

    Si cache_cell_deg is None:
        usa la coordenada exacta.

    Si cache_cell_deg tiene valor, por ejemplo 0.05:
        agrupa las coordenadas en celdas de 0.05 deg.
    """

    Ra = float(Ra)
    Dec = float(Dec)

    if cache_cell_deg is None:
        return Ra, Dec, "exact"

    cache_cell_deg = float(cache_cell_deg)

    if cache_cell_deg <= 0:
        raise ValueError("cache_cell_deg debe ser positivo o None.")

    Ra_q = np.round(Ra / cache_cell_deg) * cache_cell_deg
    Dec_q = np.round(Dec / cache_cell_deg) * cache_cell_deg

    Ra_q = Ra_q % 360.0
    Dec_q = np.clip(Dec_q, -90.0, 90.0)

    return float(Ra_q), float(Dec_q), "cell"
def _get_source_field_rubin(
    path_ephemerides,
    Ra,
    Dec,
    cache_cell_deg=None,
    opsim_db_path=None,
    rubin_throughputs_dir=None,
):
    """
    Devuelve dataSlice y rubin_ts para una coordenada de fuente.

    Importante:
    - Ra, Dec son las coordenadas reales de la fuente.
    - El dataSlice de MAF puede calcularse en una coordenada agrupada
      si cache_cell_deg no es None.
    - El cache incluye el archivo OpSim para no mezclar baselines.
    """

    global _RUBIN_SOURCE_CACHE
    global LAST_DATASLICE_INFO

    _init_common_roman_rubin(path_ephemerides, rubin_throughputs_dir=rubin_throughputs_dir)

    lsst_filterlist = "ugrizy"

    source_Ra = float(Ra)
    source_Dec = float(Dec)

    opsim_path = _resolve_opsim_db_path(opsim_db_path)
    opsim_tag = _opsim_cache_tag(opsim_path)

    maf_Ra, maf_Dec, cache_mode = _quantize_sky_position(
        source_Ra,
        source_Dec,
        cache_cell_deg=cache_cell_deg,
    )

    if cache_cell_deg is None:
        cache_key = (
            "source_exact",
            opsim_tag,
            round(maf_Ra, 5),
            round(maf_Dec, 5),
        )
    else:
        cache_key = (
            "source_cell",
            opsim_tag,
            float(cache_cell_deg),
            round(maf_Ra, 6),
            round(maf_Dec, 6),
        )

    if cache_key not in _RUBIN_SOURCE_CACHE:

        print("=" * 80)
        print("[MAF] Source-coordinate Rubin-only field")
        print(f"[MAF] source RA  = {source_Ra:.10f} deg")
        print(f"[MAF] source Dec = {source_Dec:.10f} deg")
        print(f"[MAF] MAF RA     = {maf_Ra:.10f} deg")
        print(f"[MAF] MAF Dec    = {maf_Dec:.10f} deg")
        print(f"[MAF] cache mode = {cache_mode}")
        print(f"[MAF] cache cell = {cache_cell_deg}")
        print(f"[MAF] OpSim DB   = {opsim_path}")
        print(f"[MAF] cache tag  = {opsim_tag}")
        print("=" * 80)

        dataSlice = _load_or_build_dataslice(
            maf_Ra,
            maf_Dec,
            opsim_db_path=opsim_path,
        )

        rubin_ts = _build_rubin_ts(
            dataSlice,
            lsst_filterlist=lsst_filterlist,
        )

        _RUBIN_SOURCE_CACHE[cache_key] = {
            "source_Ra_example": source_Ra,
            "source_Dec_example": source_Dec,
            "maf_Ra": maf_Ra,
            "maf_Dec": maf_Dec,
            "cache_mode": cache_mode,
            "cache_cell_deg": cache_cell_deg,
            "opsim_db_path": str(opsim_path),
            "opsim_cache_tag": opsim_tag,
            "dataSlice": dataSlice,
            "rubin_ts": rubin_ts,
        }

        source = "disk_or_maf_cache"

    else:
        source = "memory_cache"

    cached = _RUBIN_SOURCE_CACHE[cache_key]

    LAST_DATASLICE_INFO = {
        "mode": "source",
        "cache_mode": cached["cache_mode"],
        "cache_cell_deg": cached["cache_cell_deg"],
        "source_Ra": source_Ra,
        "source_Dec": source_Dec,
        "maf_Ra": float(cached["maf_Ra"]),
        "maf_Dec": float(cached["maf_Dec"]),
        "Ra": float(cached["maf_Ra"]),
        "Dec": float(cached["maf_Dec"]),
        "cache_key": cache_key,
        "source": source,
        "n_obs": int(len(cached["dataSlice"])),
        "opsim_db_path": cached["opsim_db_path"],
        "opsim_cache_tag": cached["opsim_cache_tag"],
    }

    return cached["dataSlice"], cached["rubin_ts"]


# ============================================================
# Utilidades generales
# ============================================================

def _get_col_values(col):
    """
    Devuelve valores numéricos desde una columna astropy Quantity/Column
    o desde un array numpy.
    """
    return col.value if hasattr(col, "value") else np.asarray(col)


def _validate_time_window(time_window):
    """
    Normaliza una ventana temporal en JD.
    """
    if time_window is None:
        return None

    if len(time_window) != 2:
        raise ValueError("time_window debe ser None o una tupla (time_min, time_max).")

    time_min, time_max = time_window

    if time_min is None and time_max is None:
        return None

    if time_min is not None:
        time_min = float(time_min)

    if time_max is not None:
        time_max = float(time_max)

    if time_min is not None and time_max is not None:
        if time_min > time_max:
            raise ValueError("time_min no puede ser mayor que time_max.")

    return time_min, time_max


def slice_lightcurve_by_time(lightcurve, time_window=None):
    """
    Recorta una lightcurve usando la columna 'time'.
    """
    time_window = _validate_time_window(time_window)

    if time_window is None:
        return lightcurve

    time_min, time_max = time_window

    t = _get_col_values(lightcurve["time"])

    mask = np.ones(len(t), dtype=bool)

    if time_min is not None:
        mask &= t >= time_min

    if time_max is not None:
        mask &= t <= time_max

    return lightcurve[mask]


def slice_dataslice_by_time(dataSlice, time_window=None):
    """
    Recorta el dataSlice de Rubin usando observationStartMJD.

    Las curvas de luz están en JD, mientras que OpSim/MAF usa MJD.
    Por eso se convierte:

        JD = MJD + 2400000.5
    """
    time_window = _validate_time_window(time_window)

    if time_window is None:
        return dataSlice

    time_min, time_max = time_window

    jd = dataSlice["observationStartMJD"] + 2400000.5

    mask = np.ones(len(jd), dtype=bool)

    if time_min is not None:
        mask &= jd >= time_min

    if time_max is not None:
        mask &= jd <= time_max

    return dataSlice[mask]


def apply_time_window_to_event(my_event, time_window=None):
    """
    Aplica una ventana temporal a todos los telescopios del evento.
    """
    time_window = _validate_time_window(time_window)

    if time_window is None:
        return my_event

    for tel in my_event.telescopes:
        tel.lightcurve = slice_lightcurve_by_time(
            tel.lightcurve,
            time_window=time_window,
        )

    return my_event


def make_t0_window(data, window_days=None, window_tE=None):
    """
    Construye una ventana simétrica alrededor de t0.

    Ejemplos
    --------
    window_tE=5    -> (t0 - 5*tE, t0 + 5*tE)
    window_days=50 -> (t0 - 50, t0 + 50)
    """
    t0 = data["t0"]
    tE = data["tE"]

    if window_days is not None and window_tE is not None:
        raise ValueError("Usá window_days o window_tE, no ambos.")

    if window_days is not None:
        delta_t = float(window_days)

    elif window_tE is not None:
        delta_t = float(window_tE) * float(tE)

    else:
        raise ValueError("Tenés que pasar window_days o window_tE.")

    return t0 - delta_t, t0 + delta_t


def make_asymmetric_t0_window(data, before_days=None, after_days=None,
                              before_tE=None, after_tE=None):
    """
    Construye una ventana asimétrica alrededor de t0.

    Sirve para probar, por ejemplo:
    - solo datos antes del pico;
    - solo datos después del pico;
    - más baseline de un lado que del otro.
    """
    t0 = data["t0"]
    tE = data["tE"]

    if before_days is not None and before_tE is not None:
        raise ValueError("Usá before_days o before_tE, no ambos.")

    if after_days is not None and after_tE is not None:
        raise ValueError("Usá after_days o after_tE, no ambos.")

    if before_days is None and before_tE is None:
        before = 0.0
    elif before_days is not None:
        before = float(before_days)
    else:
        before = float(before_tE) * float(tE)

    if after_days is None and after_tE is None:
        after = 0.0
    elif after_days is not None:
        after = float(after_days)
    else:
        after = float(after_tE) * float(tE)

    return t0 - before, t0 + after


# ============================================================
# Construcción modular Roman + Rubin
# ============================================================

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


def _load_lsst_bandpasses(
    lsst_filterlist="ugrizy",
    path_che=None,
):
    """
    Carga los bandpasses de Rubin.

    path_che queda como nombre por compatibilidad, pero ya no debe estar
    hardcodeado. Si path_che es None, se usa:

    1. configure_rubin_paths(..., rubin_throughputs_dir=...);
    2. RUBIN_THROUGHPUTS_DIR;
    3. ${RUBIN_SIM_DATA_DIR}/throughputs/baseline;
    4. ${HOME}/rubin_sim_data/throughputs/baseline.
    """

    throughputs_dir = _resolve_rubin_throughputs_dir(path_che)

    if not throughputs_dir.exists():
        raise FileNotFoundError(
            "No existe el directorio de throughputs Rubin:\n"
            f"{throughputs_dir}\n\n"
            "Definilo en el config como rubin.throughputs_dir, "
            "llamá configure_rubin_paths(...), o exportá "
            "RUBIN_THROUGHPUTS_DIR."
        )

    print("[Rubin] throughputs_dir =", throughputs_dir, flush=True)

    LSST_BandPass = {}

    for f in lsst_filterlist:
        bp = Bandpass()

        throughput_file = throughputs_dir / f"total_{f}.dat"
        throughput_gz = Path(str(throughput_file) + ".gz")

        if not throughput_file.exists() and not throughput_gz.exists():
            raise FileNotFoundError(
                "No encontré el throughput Rubin requerido:\n"
                f"{throughput_file}\n"
                f"ni\n"
                f"{throughput_gz}\n\n"
                "Revisá rubin.throughputs_dir en el config."
            )

        # Bandpass.read_throughput acepta el path sin .gz y prueba .gz
        # internamente si el .dat no existe.
        bp.read_throughput(str(throughput_file))
        LSST_BandPass[f] = bp

    return LSST_BandPass

def _load_or_build_dataslice(Ra, Dec, opsim_db_path=None):
    """
    Lee o genera el dataSlice de MAF para el campo elegido.

    El cache incluye una etiqueta del archivo OpSim. Esto evita reutilizar
    accidentalmente un dataSlice generado con otra baseline.
    """

    opsim_path = _resolve_opsim_db_path(opsim_db_path)
    opsim_tag = _opsim_cache_tag(opsim_path)

    ds_name = f"dataslice_{opsim_tag}_ra{Ra:.5f}_dec{Dec:.5f}"
    ds_npz = _npz_path(ds_name)

    if ds_npz.exists():
        packed = np.load(ds_npz, allow_pickle=True)
        dataSlice = packed["dataSlice"]
        dataSlice = _canonicalize_dataslice_filter_column(dataSlice)
        return dataSlice

    conn = str(opsim_path)

    outDir = str(_CACHE_DIR / f"maf_{opsim_tag}_{os.getpid()}")
    os.makedirs(outDir, exist_ok=True)

    resultsDb = maf.db.ResultsDb()

    metric = maf.metrics.PassMetric(
        cols=[
            "filter",
            "observationStartMJD",
            "fiveSigmaDepth",
        ]
    )

    slicer = maf.slicers.UserPointsSlicer(
        ra=[Ra],
        dec=[Dec],
    )

    sql = ""

    metric_bundle = maf.MetricBundle(
        metric,
        slicer,
        sql,
    )

    bundleDict = {
        "my_bundle": metric_bundle,
    }

    bg = maf.MetricBundleGroup(
        bundleDict,
        conn,
        out_dir=outDir,
        results_db=resultsDb,
    )

    _acquire_lock(ds_name)

    try:
        if ds_npz.exists():
            packed = np.load(ds_npz, allow_pickle=True)
            dataSlice = packed["dataSlice"]
            dataSlice = _canonicalize_dataslice_filter_column(dataSlice)

        else:
            print("=" * 80)
            print("[MAF] Building dataSlice")
            print(f"[MAF] RA       = {float(Ra):.8f} deg")
            print(f"[MAF] Dec      = {float(Dec):.8f} deg")
            print(f"[MAF] OpSim DB = {opsim_path}")
            print(f"[MAF] cache    = {ds_npz}")
            print("=" * 80)

            bg.run_all()
            dataSlice = metric_bundle.metric_values[0]
            dataSlice = _canonicalize_dataslice_filter_column(dataSlice)

            try:
                np.savez_compressed(
                    str(ds_npz),
                    dataSlice=dataSlice,
                    opsim_db_path=str(opsim_path),
                    opsim_cache_tag=opsim_tag,
                )
            except Exception as e:
                print(f"[warn] no pude guardar {ds_npz}: {e}")

    finally:
        _release_lock(ds_name)

    return dataSlice


def _canonical_rubin_filter_array(values):
    """
    Convierte nombres de filtro de OpSim a bandas Rubin canónicas.

    Ejemplos:
        'g'    -> 'g'
        'g_6'  -> 'g'
        'r_57' -> 'r'
        b'i_39' -> 'i'
    """

    arr = np.asarray(values)

    if arr.dtype.kind == "S":
        arr = np.char.decode(arr, "utf-8")
    else:
        arr = arr.astype(str)

    out = []

    for value in arr:
        value = str(value).strip()

        if len(value) >= 4 and value.startswith("b'") and value.endswith("'"):
            value = value[2:-1]

        if len(value) >= 4 and value.startswith('b"') and value.endswith('"'):
            value = value[2:-1]

        # Caso nuevo de baseline v5.3.5:
        # g_6, i_39, r_57, u_24, y_10, z_20
        band = value.split("_")[0].lower()

        if band not in {"u", "g", "r", "i", "z", "y"}:
            raise ValueError(
                f"No pude interpretar el filtro Rubin {value!r} como banda ugrizy."
            )

        out.append(band)

    return np.asarray(out, dtype=str)


def _canonicalize_dataslice_filter_column(dataSlice):
    """
    Devuelve una copia del dataSlice con la columna de filtro canonicalizada.

    Esto permite que el resto del pipeline, incluido functions_roman_rubin.py,
    siga usando comparaciones antiguas como:

        dataSlice["filter"] == "r"

    aunque la OpSim nueva tenga valores como:

        r_57, g_6, i_39, u_24, y_10, z_20

    Los telescopios pyLIMA se siguen llamando u,g,r,i,z,y.
    """

    if dataSlice is None:
        return dataSlice

    names = getattr(getattr(dataSlice, "dtype", None), "names", None)

    if names is None:
        return dataSlice

    if "filter" in names:
        column = "filter"
    elif "band" in names:
        column = "band"
    else:
        return dataSlice

    canonical = _canonical_rubin_filter_array(dataSlice[column])

    # Copia explícita: no modifica el objeto original de MAF in-place.
    out = dataSlice.copy()

    try:
        out[column] = canonical
    except Exception:
        # Fallback robusto para dtypes muy restrictivos.
        out[column] = canonical.astype(out[column].dtype, copy=False)

    return out


def _build_rubin_ts(dataSlice, lsst_filterlist="ugrizy"):
    """
    Construye las curvas base de Rubin a partir del dataSlice.

    Cada telescopio Rubin recibe columnas:
        time, mag, err_mag

    donde mag y err_mag se cargan inicialmente con m5.
    """

    rubin_ts = {}

    if "filter" in dataSlice.dtype.names:
        filter_values = _canonical_rubin_filter_array(dataSlice["filter"])
    elif "band" in dataSlice.dtype.names:
        filter_values = _canonical_rubin_filter_array(dataSlice["band"])
    else:
        raise KeyError(
            "dataSlice no tiene columna 'filter' ni 'band'. "
            f"Columnas disponibles: {dataSlice.dtype.names}"
        )

    print(
        "[MAF] Rubin observations by canonical band:",
        {
            fil: int(np.sum(filter_values == fil))
            for fil in lsst_filterlist
        },
        flush=True,
    )

    for fil in lsst_filterlist:
        mask = filter_values == fil

        m5 = dataSlice["fiveSigmaDepth"][mask]
        mjd = dataSlice["observationStartMJD"][mask] + 2400000.5

        int_array = np.column_stack(
            (
                mjd,
                m5,
                m5,
            )
        ).astype(float)

        rubin_ts[fil] = int_array

    return rubin_ts


def _roman_nominal_and_off_seasons():
    """
    Define las temporadas Roman usadas en la simulación.
    """
    nominal_seasons = [
        {"start": "2027-02-11T00:00:00", "end": "2027-04-24T00:00:00"},
        {"start": "2027-08-16T00:00:00", "end": "2027-10-27T00:00:00"},
        {"start": "2028-02-11T00:00:00", "end": "2028-04-24T00:00:00"},
        {"start": "2030-02-11T00:00:00", "end": "2030-04-24T00:00:00"},
        {"start": "2030-08-16T00:00:00", "end": "2030-10-27T00:00:00"},
        {"start": "2031-02-11T00:00:00", "end": "2031-04-24T00:00:00"},
    ]

    off_seasons = [
        {"start": "2028-08-15T00:00:00", "end": "2028-10-27T00:00:00"},
        {"start": "2029-02-11T00:00:00", "end": "2029-04-24T00:00:00"},
        {"start": "2029-08-16T00:00:00", "end": "2029-10-27T00:00:00"},
    ]

    return nominal_seasons, off_seasons


def _simulate_roman_season(season, sampling):
    """
    Simula una temporada Roman W149.
    """
    tstart = Time(season["start"], format="isot").jd
    tend = Time(season["end"], format="isot").jd

    Roman = simulator.simulate_a_telescope(
        name="W149",
        time_start=tstart,
        time_end=tend,
        sampling=sampling,
        location="Space",
        camera_filter="W149",
        uniform_sampling=True,
        astrometry=False,
    )

    return Roman.lightcurve


def _load_or_build_roman_template():
    """
    Lee o genera la plantilla completa Roman W149.
    """
    rt_name = "roman_template_W149"
    rt_npz = _npz_path(rt_name)

    if rt_npz.exists():
        packed = np.load(rt_npz, allow_pickle=True)
        combined_array = packed["combined_array"]

        mat_all = _structured_to_matrix(
            combined_array,
            [
                "time",
                "mag",
                "err_mag",
                "flux",
                "err_flux",
                "inv_err_flux",
            ],
        )

        if mat_all.shape[1] < 3:
            raise ValueError(
                "roman_template_W149 npz no tiene columnas suficientes "
                "(esperaba >=3)."
            )

        roman_mag = mat_all[:, :3]
        return roman_mag

    nominal_seasons, off_seasons = _roman_nominal_and_off_seasons()

    lightcurve_fluxes = []

    for season in nominal_seasons:
        lc = _simulate_roman_season(
            season,
            sampling=121 / 600,
        )
        lightcurve_fluxes.append(lc)

    for season in off_seasons:
        lc = _simulate_roman_season(
            season,
            sampling=24 * 3,
        )
        lightcurve_fluxes.append(lc)

    combined_array = np.concatenate(
        [lc.as_array() for lc in lightcurve_fluxes]
    )

    mat_all = _structured_to_matrix(
        combined_array,
        [
            "time",
            "mag",
            "err_mag",
            "flux",
            "err_flux",
            "inv_err_flux",
        ],
    )

    if mat_all.shape[1] < 3:
        raise ValueError("Roman combinado no tiene columnas suficientes (>=3).")

    roman_mag = mat_all[:, :3]

    try:
        np.savez_compressed(
            str(rt_npz),
            combined_array=combined_array,
        )
    except Exception as e:
        print(f"[warn] no pude guardar {rt_npz}: {e}")

    return roman_mag


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

def _build_event_template(
    Ra,
    Dec,
    roman_mag,
    rubin_ts,
    path_ephemerides,
    lsst_filterlist="ugrizy",
    use_roman=True,
    use_rubin=True,
):
    """
    Construye el Event base con Roman y/o Rubin.
    """

    my_own_creation = event.Event(
        ra=Ra,
        dec=Dec,
    )

    my_own_creation.name = "An event observed by Roman/Rubin"

    if use_roman:

        Roman_tot = _get_cached_roman_telescope(
            roman_mag,
            path_ephemerides,
        )

        my_own_creation.telescopes.append(
            Roman_tot
        )

    if use_rubin:

        for band in lsst_filterlist:

            lsst_telescope = _get_cached_rubin_telescope(
                band,
                rubin_ts[band],
            )

            my_own_creation.telescopes.append(
                lsst_telescope
            )


    if len(my_own_creation.telescopes) == 0:
        raise ValueError("No hay telescopios activos: use_roman=False y use_rubin=False.")

    return my_own_creation

# ============================================================
# API principal
# ============================================================

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
def tel_roman_rubin(
    path_ephemerides,
    time_window=None,
    use_roman=True,
    use_rubin=True,
    Ra=None,
    Dec=None,
    rubin_pointing_mode="fixed",
    rubin_cache_cell_deg=None,
    opsim_db_path=None,
    rubin_sim_data_dir=None,
    rubin_throughputs_dir=None,
):
    """
    Devuelve un Event con Roman y/o Rubin.

    opsim_db_path:
        path a un archivo OpSim/MAF .db. Si es None, se usa
        RUBIN_OPSIM_DB_PATH / RUBIN_OPSIM_DB si existen; si no,
        get_baseline().

    rubin_pointing_mode:
        "fixed"
            Usa siempre el campo fijo Roman+Rubin actual.
            Este es el default y conserva el cache global.

        "source"
            Usa Ra, Dec pasados por argumento.
            Pensado para pruebas Rubin-only con fuentes TRILEGAL.

    Regla recomendada:
        Roman + Rubin  -> fixed
        Rubin-only     -> fixed o source, según el experimento
    """

    if not use_roman and not use_rubin:
        raise ValueError("No podés apagar Roman y Rubin al mismo tiempo.")

    # Permite pasar paths desde el runner/config sin hardcodearlos acá.
    # Si estos argumentos son None, se usan los paths ya configurados
    # por configure_rubin_paths(...) o las variables de entorno.
    if (
        rubin_sim_data_dir not in (None, "", "auto", "default")
        or rubin_throughputs_dir not in (None, "", "auto", "default")
        or opsim_db_path not in (None, "", "auto", "default")
    ):
        configure_rubin_paths(
            rubin_sim_data_dir=rubin_sim_data_dir,
            rubin_throughputs_dir=rubin_throughputs_dir,
            rubin_opsim_db_path=opsim_db_path,
            reset_caches=False,
            validate=False,
        )

    time_window = _validate_time_window(time_window)

    lsst_filterlist = "ugrizy"

    # ------------------------------------------------------------
    # Si Roman+Rubin están encendidos, forzamos campo fijo.
    # Esto preserva tu simulación principal.
    # ------------------------------------------------------------

    # El modo "fixed" se conserva por defecto.
    # El modo "source" también se permite con Roman+Rubin.

    # ------------------------------------------------------------
    # Modo campo fijo Roman+Rubin
    # ------------------------------------------------------------

    if rubin_pointing_mode == "fixed":

        _init_fixed_field_rubin(
            path_ephemerides,
            opsim_db_path=opsim_db_path,
            rubin_throughputs_dir=rubin_throughputs_dir,
        )

        event_Ra, event_Dec = _roman_rubin_field_coordinates()

        dataSlice_full = _DATASLICE
        rubin_ts = _RUBIN_TS

    # ------------------------------------------------------------
    # Modo coordenadas de fuente TRILEGAL
    # ------------------------------------------------------------

    elif rubin_pointing_mode == "source":

        if Ra is None or Dec is None:
            raise ValueError(
                "rubin_pointing_mode='source' requiere Ra y Dec."
            )


        event_Ra = float(Ra)
        event_Dec = float(Dec)

        dataSlice_full, rubin_ts = _get_source_field_rubin(
            path_ephemerides,
            event_Ra,
            event_Dec,
            cache_cell_deg=rubin_cache_cell_deg,
            opsim_db_path=opsim_db_path,
            rubin_throughputs_dir=rubin_throughputs_dir,
        )

    else:
        raise ValueError(
            "rubin_pointing_mode debe ser 'fixed' o 'source'."
        )

    # ------------------------------------------------------------
    # Construcción del evento
    # ------------------------------------------------------------

    my_event = _build_event_template(
        event_Ra,
        event_Dec,
        _ROMAN_MAG,
        rubin_ts,
        path_ephemerides,
        lsst_filterlist=lsst_filterlist,
        use_roman=use_roman,
        use_rubin=use_rubin,
    )

    if time_window is not None:
        my_event = apply_time_window_to_event(
            my_event,
            time_window=time_window,
        )

    if use_rubin:
        if time_window is None:
            dataSlice = dataSlice_full
        else:
            dataSlice = slice_dataslice_by_time(
                dataSlice_full,
                time_window=time_window,
            )
    else:
        dataSlice = None

    return my_event, dataSlice, _LSST_BANDPASS
