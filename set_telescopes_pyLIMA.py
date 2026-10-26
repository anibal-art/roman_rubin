#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:36:34 2026

@author: anibal
"""
import numpy as np
import os, re, time, hashlib
from pathlib import Path

# Get the directory where the script is located
script_dir = Path(__file__).parent
home_dir = os.path.expanduser("~")

from rubin_sim.phot_utils.bandpass import Bandpass
import rubin_sim.maf as maf
from rubin_sim.data import get_baseline

# --- Fix para IERS y sidereal_time de Astropy ---
from astropy.utils import iers
iers.conf.auto_max_age = None
iers.conf.auto_download = False  # No intenta descargar
iers.conf.iers_degraded_accuracy = 'warn'






from simulation.telescope_templates import (  # noqa: F401  (moved; re-exported)
    _telescope_array_signature,
    _load_cached_roman_ephemerides,
    _get_cached_roman_telescope,
    _get_cached_rubin_telescope,
    _build_event_template,
    _ROMAN_TELESCOPE_TEMPLATE_CACHE,
    _RUBIN_TELESCOPE_TEMPLATE_CACHE,
    _ROMAN_EPHEMERIDES_TEMPLATE_CACHE,
    _TELESCOPE_TEMPLATE_CACHE_STATS,
)
from simulation.roman_cadence import (  # noqa: F401  (moved; re-exported)
    _structured_to_matrix,
    _roman_nominal_and_off_seasons,
    _simulate_roman_season,
)
from simulation.time_windows import (  # noqa: F401  (moved; re-exported)
    _get_col_values,
    _validate_time_window,
    slice_lightcurve_by_time,
    slice_dataslice_by_time,
    apply_time_window_to_event,
)
from simulation.rubin_dataslice import (  # noqa: F401  (moved; re-exported)
    _canonical_rubin_filter_array,
    _canonicalize_dataslice_filter_column,
    _build_rubin_ts,
    _quantize_sky_position,
)


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

# ============================================================
# Construcción modular Roman + Rubin
# ============================================================

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


# ============================================================
# API principal
# ============================================================




def tel_roman_rubin(
    path_ephemerides,
    time_window=None,
    use_roman=True,
    use_rubin=True,
    Ra=None,
    Dec=None,
    rubin_pointing_mode="source",
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
        Only "source" is supported: the Event and the Rubin OpSim query use
        the event position Ra, Dec (required).
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

    # Every event is observed at its own catalogue position.
    if rubin_pointing_mode != "source":
        raise ValueError(
            "Only rubin_pointing_mode='source' is supported "
            f"(got {rubin_pointing_mode!r})."
        )

    if rubin_pointing_mode == "source":

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
