import os
import logging
import re
import concurrent.futures as cf
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional

# ---------- Logging ----------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ---------- Constantes ----------
ENV_NO_OMP = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

_RETRIES = 3  # reintentos internos por job

# ---------- Helpers básicos ----------

def _blas_sanitize_global():
    """
    Fuerza BLAS/OpenMP a 1 hilo en el proceso padre (y por herencia en los hijos).
    """
    for k, v in ENV_NO_OMP.items():
        os.environ[k] = os.environ.get(k, v)



def _with_retries(call, *args, **kwargs):
    """
    Ejecuta call(*args, **kwargs) con hasta _RETRIES intentos.
    Relanza la última excepción si todos fallan.
    """
    last = None
    for k in range(1, _RETRIES + 1):
        try:
            return call(*args, **kwargs)
        except Exception as e:
            last = e
            logging.warning(f"Attempt {k}/{_RETRIES} failed: {e}")
    raise last


# ---------- Workers (procesos hijos) ----------

def _worker_sim_fit(args):
    """
    Worker sencillo para sim_fit.
    Escribe DIRECTO en path_to_save_model (archivos visibles en tiempo real).
    """
    (
        i,
        system_type,
        model,
        algo,
        path_TRILEGAL_set,
        path_GENULENS_set,
        path_to_save_model,
        path_to_save_fit,
        path_ephemerides,
        path_dataslice,
    ) = args

    # Sanitizar hilos BLAS solo por seguridad
    for k, v in ENV_NO_OMP.items():
        os.environ[k] = os.environ.get(k, v)

    try:
        from functions_roman_rubin import sim_fit

        out_final = Path(path_to_save_model) / f"Event_{i}.h5"
        if out_final.exists():
            logging.info(f"[sim:{i}] SKIP (exists)")
            return 0

        # sim_fit escribirá Event_{i}.h5 directamente en path_to_save_model
        _with_retries(
            sim_fit,
            i,
            system_type,
            model,
            algo,
            path_TRILEGAL_set,
            path_GENULENS_set,
            path_to_save_model,
            path_to_save_fit,
            path_ephemerides,
            path_dataslice,
        )

        logging.info(f"[sim:{i}] OK -> {out_final}")
        return 0

    except Exception as e:
        logging.exception(f"[sim:{i}] FAILED: {e}")
        raise


def _worker_read_fit(args):
    """
    Worker sencillo para read_fit.
    """
    (
        nsource,
        nset,
        path_run,
        model,
        algo,
        path_to_save_fit,
        path_ephemerides,
    ) = args

    for k, v in ENV_NO_OMP.items():
        os.environ[k] = os.environ.get(k, v)

    try:
        from functions_roman_rubin import read_fit

        _with_retries(
            read_fit,
            nsource,
            str(nset),
            path_run,
            model,
            algo,
            path_to_save_fit,
            path_ephemerides,
        )

        logging.info(f"[read:{nset}:{nsource}] OK")
        return 0

    except Exception as e:
        logging.exception(f"[read:{nset}:{nsource}] FAILED: {e}")
        raise


# ---------- Utilidades para read_fit ----------

def _list_event_numbers(h5_dir: Path) -> List[int]:
    """
    Devuelve la lista de índices i tales que existen Event_i.h5 en h5_dir.
    """
    patt = re.compile(r"Event_(\d+)\.h5$")
    nums: List[int] = []
    for f in h5_dir.glob("*.h5"):
        m = patt.search(f.name)
        if m:
            nums.append(int(m.group(1)))
    nums.sort()
    return nums


# ---------- API pública simplificada ----------

def _slurm_workers() -> int:
    """
    Determina cuántos workers puede lanzar este job en SLURM.
    Regla:
      1) Si SLURM_CPUS_PER_TASK está definido → usar ese valor.
      2) Si el cgroup fija afinidad → usar la cantidad visible.
      3) Como último recurso → usar 1 (seguro).
    """
    # 1) Lo más confiable: SLURM_CPUS_PER_TASK
    try:
        v = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        if v > 0:
            return v
    except Exception:
        pass

    # 2) Intentar afinidad real
    try:
        if hasattr(os, "sched_getaffinity"):
            return len(os.sched_getaffinity(0))
    except Exception:
        pass

    # 3) Última opción segura: 1
    return 1


def run_parallel(
    path_ephemerides,
    path_dataslice,
    path_TRILEGAL_set,
    path_GENULENS_set,
    path_to_save_fit,
    path_to_save_model,
    model,
    system_type,
    algo,
    N_tr,
    total_events: int = 250_000,
    max_in_flight: Optional[int] = None,
):
    """
    Ejecuta sim_fit en paralelo usando spawn y tantos workers como SLURM asigna.
    """

    _blas_sanitize_global()

    # === NUEVO BLOQUE: determinación robusta de workers ===
    if N_tr > 0:
        workers = N_tr
    else:
        workers = _slurm_workers()

    print(f"[run_parallel] workers={workers}")
    # ======================================================

    args_iter = (
        (
            i,
            system_type,
            model,
            algo,
            path_TRILEGAL_set,
            path_GENULENS_set,
            path_to_save_model,
            path_to_save_fit,
            path_ephemerides,
            path_dataslice,
        )
        for i in range(int(total_events))
    )

    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for k, _ in enumerate(ex.map(_worker_sim_fit, args_iter, chunksize=5), start=1):
            if k % 1000 == 0:
                logging.warning(f"run_parallel: {k} eventos procesados")


def run_parallel_read_fit(
    nset,
    path_run,
    path_ephemerides,
    path_to_save_fit,
    model,
    algo,
    N_tr,
    max_in_flight: Optional[int] = None,  # mantenido solo por compatibilidad
):
    """
    Versión simplificada para la etapa de read_fit:
    - Lanza un worker por Event_i.h5 presente en set_sim{nset}.
    - Progreso básico cada 1000 eventos.
    """
    _blas_sanitize_global()

    workers = N_tr or _slurm_workers()

    logging.warning(
        "run_parallel_read_fit: workers=%d | SLURM_CPUS_PER_TASK=%s",
        workers,
        os.environ.get("SLURM_CPUS_PER_TASK"),
    )

    directory = Path(path_run) / f"set_sim{nset}"
    event_numbers = _list_event_numbers(directory)

    args_iter = (
        (
            nsource,
            nset,
            path_run,
            model,
            algo,
            path_to_save_fit,
            path_ephemerides,
        )
        for nsource in event_numbers
    )

    ctx = mp.get_context("fork")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for k, _ in enumerate(ex.map(_worker_read_fit, args_iter, chunksize=5), start=1):
            if k % 1000 == 0:
                logging.warning(
                    f"run_parallel_read_fit: {k} eventos leídos (set={nset})"
                )


# ---------- Main opcional para pruebas ----------

if __name__ == "__main__":
    mp.freeze_support()
    print("runner_rr (simple) loaded")
