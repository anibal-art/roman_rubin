from sim_fit_parallelization_v2 import run_parallel
import os, sys, json, argparse, hashlib, shutil, signal, atexit, time
import multiprocessing as mp
from pathlib import Path

# === [AÑADIDO] Limpieza agresiva de procesos hijos al terminar ===
def _cleanup_children(timeout: float = 2.0):
    try:
        # intentar terminar hijos con elegancia
        for ch in mp.active_children():
            try:
                ch.terminate()
            except Exception:
                pass

        t0 = time.time()
        while mp.active_children() and time.time() - t0 < timeout:
            time.sleep(0.05)

        # si aún quedan, matar fuerte
        for ch in mp.active_children():
            try:
                ch.kill()
            except Exception:
                pass
    except Exception:
        pass

# registrar cleanup siempre (salga bien, por error o por señal)
atexit.register(_cleanup_children)

# === [AÑADIDO] Manejo de señales en el proceso padre ===
_STOP = {"flag": False}
def _term_handler(signum, frame):
    # marcar intención de parar; dejamos que main() y run_parallel retornen
    _STOP["flag"] = True
signal.signal(signal.SIGTERM, _term_handler)
signal.signal(signal.SIGINT,  _term_handler)
# =================================================================

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.environ.get("CFG_PATH"),
                   help="Ruta al archivo de configuración (o por env CFG_PATH)")
    return p.parse_args()

def main():
    args = parse_args()
    if not args.config:
        print("ERROR: Debes pasar --config o definir CFG_PATH", file=sys.stderr)
        sys.exit(2)

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"No existe config: {cfg_path}")

    # --- Carga de config congelado ---
    with cfg_path.open() as f:
        params = json.load(f)

    # Validaciones mínimas
    required = ["path_storage", "n_db_file", "model", "algo", "system_type", "N_tr", "Nevents"]
    missing = [k for k in required if k not in params]
    if missing:
        raise ValueError(f"Faltan claves en config: {missing}")

    # --- Rutas base relativas al código (no al config) ---
    script_dir = Path(__file__).resolve().parent
    sys.path.append(str(script_dir / "photutils"))  # solo si realmente lo necesitás

    path_ephemerides = str(script_dir / "ephemerides" / "Roman_positions.npy")

    # [NEW] Usar scratch local del nodo si está disponible (mejora I/O en c00x)
    #       y asegurar que tempfile lo use.
    slurm_tmp = os.environ.get("SLURM_TMPDIR", "/tmp")
    os.environ.setdefault("TMPDIR", slurm_tmp)

    # --- Parámetros del config ---
    path_storage = Path(params["path_storage"])
    j            = int(params["n_db_file"])
    model        = params["model"]
    algo         = params["algo"]
    system_type  = params["system_type"]
    N_tr_cfg     = int(params["N_tr"])
    N_events     = int(params["Nevents"])

    # [CHANGED] Dejar autodetección de workers si N_tr<=0 (afinidad real en el orquestador)
    N_tr = N_tr_cfg if N_tr_cfg > 0 else 0
    save_results_dir = Path('/share/storage3/rubin/microlensing/romanrubin/')
    #save_results_dir = Path('/mnt/almacenamiento')
    path_dataslice   = str(save_results_dir / "all_results" / system_type)

    # --- Archivos de entrada ---
    trilegal_file = f"TRILEGAL_chunk_{j}.csv"
    genulens_file = f"Genulens_chunk_{j}.csv"
    path_TRILEGAL_set = str(script_dir / "chunks_TRILEGAL_GENULENS" / trilegal_file)
    path_GENULENS_set = str(script_dir / "chunks_TRILEGAL_GENULENS" / genulens_file)

    # --- Salidas ---
    model_dir = path_storage / system_type / f"set_sim{j}"
    fit_dir   = path_storage / system_type / f"set_fit{j}"
    model_dir.mkdir(parents=True, exist_ok=True)
    fit_dir.mkdir(parents=True, exist_ok=True)

    # convertir a string y asegurar "/" final
    path_to_save_model = str(model_dir) + os.sep
    path_to_save_fit   = str(fit_dir)   + os.sep

    # --- Snapshot del config usado a las salidas ---
    cfg_sha = _sha256(cfg_path)
    # guardo copias y el hash en ambos destinos
    for outdir in (model_dir, fit_dir):
        dst = outdir / "config_used.json"
        shutil.copy2(cfg_path, dst)
        (outdir / "CONFIG.SHA256").write_text(f"{cfg_sha}  config_used.json\n")

    # --- Evitar oversubscription en el proceso PADRE (los hijos también lo harán) ---
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # [NEW] Auditoría útil: afinidad visible y variables SLURM (si existen)
    try:
        affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else "NA"
    except Exception:
        affinity = "NA"
    print(f"[AUDIT] Affinity={affinity} | SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK')} | TMPDIR={os.environ.get('TMPDIR')}")

    # --- Ejecutar ---
    run_parallel(
        path_ephemerides=path_ephemerides,
        path_dataslice=path_dataslice,
        path_TRILEGAL_set=path_TRILEGAL_set,
        path_GENULENS_set=path_GENULENS_set,
        path_to_save_fit=path_to_save_fit,
        path_to_save_model=path_to_save_model,
        model=model,
        system_type=system_type,
        algo=algo,
        N_tr=0,                 # 0 => el orquestador usa afinidad/cgroups para elegir workers
        total_events=N_events
    )

if __name__ == "__main__":
    # [CHANGED] Mantener 'spawn' explícito
    mp.set_start_method("fork")
    try:
        main()
    finally:
        # asegurarse de barrer cualquier proceso hijo que quede colgado
        _cleanup_children()

