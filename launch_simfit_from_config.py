import os

# ============================================================
# Limitar BLAS/OpenMP antes de importar el runner paralelo
# ============================================================

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


from simfit_parallel_runner import run_parallel

import sys
import json
import argparse
import hashlib
import shutil
import signal
import atexit
import time
import multiprocessing as mp
from pathlib import Path


# ============================================================
# Limpieza de procesos hijos
# ============================================================

def _cleanup_children(timeout: float = 2.0):
    """
    Intenta terminar procesos hijos si el proceso principal termina
    por error, interrupción o señal.
    """

    try:
        children = mp.active_children()

        for ch in children:
            try:
                ch.terminate()
            except Exception:
                pass

        t0 = time.time()

        while mp.active_children() and time.time() - t0 < timeout:
            time.sleep(0.05)

        for ch in mp.active_children():
            try:
                ch.kill()
            except Exception:
                pass

    except Exception:
        pass


atexit.register(_cleanup_children)


def _term_handler(signum, frame):
    """
    Manejo explícito de SIGTERM/SIGINT.

    En CHE/SLURM esto evita dejar procesos hijos vivos si el job se cancela.
    """

    print(f"\n[signal] Received signal {signum}. Cleaning children...", file=sys.stderr)
    _cleanup_children()
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, _term_handler)
signal.signal(signal.SIGINT, _term_handler)


# ============================================================
# Helpers
# ============================================================

def _sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Roman-Rubin sim_fit jobs from a JSON config file."
    )

    p.add_argument(
        "--config",
        default=os.environ.get("CFG_PATH"),
        help="Path to JSON configuration file, or define CFG_PATH.",
    )

    return p.parse_args()


def _require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"No existe {label}: {path}")


def _require_writable_dir(path: Path, label: str):
    path.mkdir(parents=True, exist_ok=True)

    if not os.access(path, os.W_OK):
        raise PermissionError(f"No tengo permiso de escritura en {label}: {path}")

def _unlink_if_exists(path: Path):
    """
    Borra un archivo si existe.

    Si el archivo quedó read-only de una corrida anterior, intenta primero
    recuperar permiso de escritura para el usuario y luego lo elimina.
    """

    path = Path(path)

    if path.exists() or path.is_symlink():

        try:
            path.unlink()

        except PermissionError:
            path.chmod(0o644)
            path.unlink()


def safe_copy_config_used(cfg_path: Path, dst: Path):
    """
    Copia el config usado a config_used.json de forma robusta.

    Usamos copyfile, no copy2, para no preservar permisos read-only
    del config congelado. Esto evita que una corrida posterior falle
    al reutilizar el mismo n_db_file.
    """

    cfg_path = Path(cfg_path)
    dst = Path(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)

    _unlink_if_exists(dst)

    shutil.copyfile(cfg_path, dst)

    # Dejamos config_used.json writable para relanzamientos.
    dst.chmod(0o644)


def safe_write_text(path: Path, text: str):
    """
    Escribe un archivo de texto de forma robusta.

    Si el archivo ya existe y quedó protegido, lo elimina antes de escribir.
    """

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    _unlink_if_exists(path)

    path.write_text(text)

    # Dejamos el archivo writable para relanzamientos.
    path.chmod(0o644)



def _build_simfit_kwargs(params, default_model):
    """
    Construye los kwargs científicos que se pasan a sim_fit.

    Estos valores salen del JSON de configuración.
    """

    observing_cfg = params.get("observing", {})
    truth_cfg = params.get("truth", {})
    fit_cfg = params.get("fit", {})
    simulation_cfg = params.get("simulation", {})

    simfit_kwargs = {
        # --------------------------------------------------------
        # Observatorios
        # --------------------------------------------------------
        "use_roman": bool(observing_cfg.get("use_roman", True)),
        "use_rubin": bool(observing_cfg.get("use_rubin", True)),

        # --------------------------------------------------------
        # Modelo de ajuste
        # --------------------------------------------------------
        "fit_model": str(fit_cfg.get("model", default_model)),
        "fit_parallax": fit_cfg.get("parallax", None),

        # --------------------------------------------------------
        # Opciones generales de simulación
        # --------------------------------------------------------
        "catalog_mode": str(simulation_cfg.get("catalog_mode", "trilegal_genulens")),
        "rubin_pointing_mode": str(simulation_cfg.get("rubin_pointing_mode", "fixed")),
        "rubin_cache_cell_deg": simulation_cfg.get("rubin_cache_cell_deg", None),
        "apply_detection_criteria": bool(
            simulation_cfg.get("apply_detection_criteria", True)
        ),

        # time_window afecta simulación/cadencia, no solo fit.
        # Por defecto dejamos la curva completa.
        "time_window": simulation_cfg.get("time_window", None),
    }

    # ------------------------------------------------------------
    # Opcionales: solo se agregan si están en el config
    # ------------------------------------------------------------

    if "defaults" in fit_cfg:
        simfit_kwargs["fit_defaults"] = fit_cfg["defaults"]

    if "bounds" in fit_cfg:
        simfit_kwargs["fit_bounds"] = fit_cfg["bounds"]

    # Para cuando modifiques sim_fit y acepte estos argumentos.
    if "parallax" in truth_cfg:
        simfit_kwargs["truth_parallax"] = bool(truth_cfg["parallax"])

    if "time_window" in fit_cfg:
        simfit_kwargs["fit_time_window"] = fit_cfg["time_window"]

    # ------------------------------------------------------------
    # Validaciones
    # ------------------------------------------------------------

    if not simfit_kwargs["use_roman"] and not simfit_kwargs["use_rubin"]:
        raise ValueError(
            "observing.use_roman y observing.use_rubin no pueden ser ambos false."
        )

    return simfit_kwargs
# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if not args.config:
        print("ERROR: Debes pasar --config o definir CFG_PATH", file=sys.stderr)
        sys.exit(2)

    cfg_path = Path(args.config).expanduser().resolve()

    _require_file(cfg_path, "config")

    with cfg_path.open() as f:
        params = json.load(f)

    required = [
        "path_storage",
        "n_db_file",
        "model",
        "algo",
        "system_type",
        "N_tr",
        "Nevents",
    ]

    missing = [k for k in required if k not in params]

    if missing:
        raise ValueError(f"Faltan claves en config: {missing}")

    # ------------------------------------------------------------
    # Rutas base relativas al código
    # ------------------------------------------------------------

    script_dir = Path(__file__).resolve().parent

    photutils_dir = script_dir / "photutils"

    if photutils_dir.exists():
        sys.path.append(str(photutils_dir))

    path_ephemerides = script_dir / "ephemerides" / "Roman_positions.npy"
    _require_file(path_ephemerides, "Roman ephemerides file")

    # ------------------------------------------------------------
    # Scratch local del nodo
    # ------------------------------------------------------------

    slurm_tmp = os.environ.get("SLURM_TMPDIR", "/tmp")
    os.environ.setdefault("TMPDIR", slurm_tmp)

    # ------------------------------------------------------------
    # Parámetros del config
    # ------------------------------------------------------------

    path_storage = Path(params["path_storage"]).expanduser().resolve()

    j = int(params["n_db_file"])
    model = str(params["model"])
    simfit_kwargs = _build_simfit_kwargs(params, default_model=model)
    algo = str(params["algo"])
    system_type = str(params["system_type"])
    N_tr_cfg = int(params["N_tr"])
    N_events = int(params["Nevents"])

    if N_events <= 0:
        raise ValueError(f"Nevents debe ser > 0. Recibido: {N_events}")

    # Si N_tr <= 0, simfit_parallel_runner decide usando SLURM/cgroups.
    N_tr = N_tr_cfg if N_tr_cfg > 0 else 0

    # ------------------------------------------------------------
    # Salidas desde path_storage del config
    # ------------------------------------------------------------

    _require_writable_dir(path_storage, "path_storage")

    # Nombre heredado: path_dataslice.
    # En realidad es el directorio donde save_extracted_results guarda
    # all_results/<system_type>/true, fit_rr, fit_roman, etc.
    path_dataslice_dir = path_storage / "all_results" / system_type
    _require_writable_dir(path_dataslice_dir, "path_dataslice")

    model_dir = path_storage / system_type / f"set_sim{j}"
    fit_dir = path_storage / system_type / f"set_fit{j}"

    _require_writable_dir(model_dir, "model_dir")
    _require_writable_dir(fit_dir, "fit_dir")

    path_dataslice = str(path_dataslice_dir)
    path_to_save_model = str(model_dir) + os.sep
    path_to_save_fit = str(fit_dir) + os.sep

    # ------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------

    trilegal_file = f"TRILEGAL_chunk_{j}.csv"
    genulens_file = f"Genulens_chunk_{j}.csv"

    path_TRILEGAL_set = script_dir / "chunks_TRILEGAL_GENULENS" / trilegal_file
    path_GENULENS_set = script_dir / "chunks_TRILEGAL_GENULENS" / genulens_file

    _require_file(path_TRILEGAL_set, "TRILEGAL chunk")
    _require_file(path_GENULENS_set, "GENULENS chunk")

    # ------------------------------------------------------------
    # Snapshot del config usado
    # ------------------------------------------------------------

    cfg_sha = _sha256(cfg_path)

    for outdir in (model_dir, fit_dir):
        dst = outdir / "config_used.json"

        safe_copy_config_used(
            cfg_path=cfg_path,
            dst=dst,
        )

        safe_write_text(
            outdir / "CONFIG.SHA256",
            f"{cfg_sha}  config_used.json\n",
        )

    # ------------------------------------------------------------
    # Auditoría
    # ------------------------------------------------------------

    try:
        affinity = (
            os.sched_getaffinity(0)
            if hasattr(os, "sched_getaffinity")
            else "NA"
        )
    except Exception:
        affinity = "NA"

    print("=" * 80)
    print("Run configuration")
    print("=" * 80)
    print("config file:        ", cfg_path)
    print("script_dir:         ", script_dir)
    print("path_storage:       ", path_storage)
    print("system_type:        ", system_type)
    print("model:              ", model)
    print("algo:               ", algo)
    print("n_db_file:          ", j)
    print("Nevents:            ", N_events)
    print("N_tr from config:   ", N_tr_cfg)
    print("N_tr passed:        ", N_tr)
    print("path_dataslice:     ", path_dataslice)
    print("path_to_save_model: ", path_to_save_model)
    print("path_to_save_fit:   ", path_to_save_fit)
    print("TRILEGAL chunk:     ", path_TRILEGAL_set)
    print("GENULENS chunk:     ", path_GENULENS_set)
    print("path_ephemerides:   ", path_ephemerides)
    print("TMPDIR:             ", os.environ.get("TMPDIR"))
    print("Affinity:           ", affinity)
    print("SLURM_CPUS_PER_TASK:", os.environ.get("SLURM_CPUS_PER_TASK"))
    print("SIMFIT_MP_CONTEXT:  ", os.environ.get("SIMFIT_MP_CONTEXT", "spawn"))
    print("SIMFIT_RETRIES:     ", os.environ.get("SIMFIT_RETRIES", "1"))
    print("SIMFIT_SKIP_MODE:   ", os.environ.get("SIMFIT_SKIP_MODE", "none"))
    print("=" * 80)
    print("simfit_kwargs:      ", simfit_kwargs)

    # ------------------------------------------------------------
    # Ejecutar simulación + fit
    # ------------------------------------------------------------

    run_parallel(
        path_ephemerides=str(path_ephemerides),
        path_dataslice=path_dataslice,
        path_TRILEGAL_set=str(path_TRILEGAL_set),
        path_GENULENS_set=str(path_GENULENS_set),
        path_to_save_fit=path_to_save_fit,
        path_to_save_model=path_to_save_model,
        model=model,
        system_type=system_type,
        algo=algo,
        N_tr=N_tr,
        total_events=N_events,
        simfit_kwargs=simfit_kwargs,
    )


if __name__ == "__main__":
    mp.freeze_support()

    try:
        main()

    finally:
        _cleanup_children()
