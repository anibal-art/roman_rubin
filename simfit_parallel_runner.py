import os

# ============================================================
# Limitar BLAS/OpenMP ANTES de importar numpy/scipy indirectamente
# ============================================================

ENV_NO_OMP = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

for _k, _v in ENV_NO_OMP.items():
    os.environ.setdefault(_k, _v)


import inspect
import logging
import re
import concurrent.futures as cf
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Iterable, Tuple, Any


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=os.environ.get("SIMFIT_LOGLEVEL", "WARNING").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ============================================================
# Configuración por variables de entorno
# ============================================================

_RETRIES = int(os.environ.get("SIMFIT_RETRIES", "1"))
_CHUNKSIZE = int(os.environ.get("SIMFIT_CHUNKSIZE", "1"))
_PROGRESS_EVERY = int(os.environ.get("SIMFIT_PROGRESS_EVERY", "100"))

# Contexto multiprocessing para sim_fit.
# Opciones: spawn, fork, forkserver.
# spawn es más seguro; fork puede ser más rápido si ya verificaste estabilidad.
_SIMFIT_MP_CONTEXT = os.environ.get("SIMFIT_MP_CONTEXT", "spawn")

# Contexto para read_fit.
_READFIT_MP_CONTEXT = os.environ.get("SIMFIT_READFIT_MP_CONTEXT", "fork")

# skip mode:
#   none          -> nunca saltea eventos
#   model         -> saltea si Event_i.h5 existe
#   model_and_fit -> saltea si Event_i.h5 existe y parece existir algún fit asociado
_SKIP_MODE = os.environ.get("SIMFIT_SKIP_MODE", "none").lower()

# Cantidad de tareas vivas por worker cuando se usa ejecución dinámica.
# max_in_flight = workers * SIMFIT_PREFETCH_FACTOR
_PREFETCH_FACTOR = int(os.environ.get("SIMFIT_PREFETCH_FACTOR", "2"))


# ============================================================
# Helpers generales
# ============================================================

def _blas_sanitize_global():
    """
    Fuerza BLAS/OpenMP a 1 hilo en el proceso actual.
    """
    for k, v in ENV_NO_OMP.items():
        os.environ.setdefault(k, v)


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
            logging.warning("Attempt %d/%d failed: %s", k, _RETRIES, e)

    raise last


def _get_positive_int_env(name: str) -> Optional[int]:
    try:
        value = int(os.environ.get(name, "0"))
        if value > 0:
            return value
    except Exception:
        pass

    return None


def _available_workers() -> int:
    """
    Determina cuántos workers son razonables para el job actual.

    Usa el mínimo entre:
      - afinidad real del proceso, si existe;
      - SLURM_CPUS_PER_TASK, si existe;
      - SIMFIT_MAX_WORKERS, si existe.

    Si nada está disponible, usa 1 por seguridad.
    """

    candidates = []

    # Afinidad real/cgroup.
    try:
        if hasattr(os, "sched_getaffinity"):
            candidates.append(len(os.sched_getaffinity(0)))
    except Exception:
        pass

    # SLURM.
    slurm_cpus = _get_positive_int_env("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        candidates.append(slurm_cpus)

    # Override manual.
    max_workers_env = _get_positive_int_env("SIMFIT_MAX_WORKERS")
    if max_workers_env is not None:
        candidates.append(max_workers_env)

    if len(candidates) == 0:
        return 1

    return max(1, min(candidates))


def _resolve_workers(N_tr: int, total_tasks: int) -> int:
    """
    Usa como máximo:
      - N_tr si fue pedido;
      - CPUs disponibles;
      - número de eventos/tareas.
    """

    available = _available_workers()

    total_tasks = int(total_tasks)
    if total_tasks <= 0:
        return 1

    if N_tr is not None and int(N_tr) > 0:
        workers = min(int(N_tr), available, total_tasks)
    else:
        workers = min(available, total_tasks)

    return max(1, workers)


def _fit_exists_for_event(path_to_save_fit: str, i: int) -> bool:
    """
    Intenta detectar si ya existe algún fit asociado al evento i.

    Es necesariamente flexible porque diferentes versiones del pipeline
    pueden usar nombres como:
      Event_i.h5
      Event_RR_i_TRF.npy
      Event_Roman_i_TRF.npy
    """

    fit_dir = Path(path_to_save_fit)

    if not fit_dir.exists():
        return False

    patterns = [
        f"Event_{i}.h5",
        f"Event_{i}_*.h5",
        f"Event_RR_{i}_*",
        f"Event_Roman_{i}_*",
        f"*_{i}_TRF*",
        f"*_{i}_DE*",
        f"*_{i}_LM*",
    ]

    for patt in patterns:
        if any(fit_dir.glob(patt)):
            return True

    return False


def _should_skip_event(
    i: int,
    path_to_save_model: str,
    path_to_save_fit: str,
) -> bool:
    """
    Decide si saltear un evento existente.

    Controlado por SIMFIT_SKIP_MODE:
      none          -> no salta nada
      model         -> salta si existe Event_i.h5
      model_and_fit -> salta si existe modelo y algún fit
    """

    if _SKIP_MODE in ["", "none", "false", "0", "no"]:
        return False

    out_model = Path(path_to_save_model) / f"Event_{i}.h5"

    if _SKIP_MODE == "model":
        return out_model.exists()

    if _SKIP_MODE == "model_and_fit":
        return out_model.exists() and _fit_exists_for_event(path_to_save_fit, i)

    raise ValueError(
        "SIMFIT_SKIP_MODE debe ser uno de: none, model, model_and_fit. "
        f"Valor recibido: {_SKIP_MODE}"
    )


def _filter_kwargs_for_callable(call, kwargs: Optional[dict], task_label: str = "") -> dict:
    """
    Filtra kwargs para pasar solamente aquellos aceptados por la firma de call.

    Esto permite que launch_simfit_from_config.py/config tenga claves nuevas
    sin romper corridas con versiones de sim_fit que todavía no aceptan todas
    esas opciones.
    """

    if kwargs is None:
        return {}

    if not isinstance(kwargs, dict):
        raise TypeError(
            f"simfit_kwargs debe ser dict o None. Recibido: {type(kwargs)}"
        )

    params = inspect.signature(call).parameters

    # Si la función acepta **kwargs, se puede pasar todo.
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in params.values()
    )

    if accepts_var_kwargs:
        return dict(kwargs)

    safe_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in params
    }

    ignored_kwargs = sorted(
        set(kwargs.keys()) - set(safe_kwargs.keys())
    )

    if ignored_kwargs:
        logging.warning(
            "%s Ignoring unsupported kwargs for %s: %s",
            task_label,
            getattr(call, "__name__", str(call)),
            ignored_kwargs,
        )

    return safe_kwargs


# ============================================================
# Workers
# ============================================================

def _worker_sim_fit(args):
    """
    Worker para sim_fit.

    Cada proceso hijo importa functions_roman_rubin localmente.
    Esto evita cargar pyLIMA/rubin_sim en el proceso padre.
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
        simfit_kwargs,
    ) = args

    _blas_sanitize_global()

    try:
        from functions_roman_rubin import sim_fit

        if _should_skip_event(i, path_to_save_model, path_to_save_fit):
            logging.info(
                "[sim:%s] SKIP existing according to mode=%s",
                i,
                _SKIP_MODE,
            )
            return 0

        safe_simfit_kwargs = _filter_kwargs_for_callable(
            sim_fit,
            simfit_kwargs,
            task_label=f"[sim:{i}]",
        )

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
            **safe_simfit_kwargs,
        )

        logging.info("[sim:%s] OK", i)
        return 0

    except Exception as e:
        logging.exception("[sim:%s] FAILED: %s", i, e)
        raise


def _worker_read_fit(args):
    """
    Worker para read_fit.
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

    _blas_sanitize_global()

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

        logging.info("[read:%s:%s] OK", nset, nsource)
        return 0

    except Exception as e:
        logging.exception("[read:%s:%s] FAILED: %s", nset, nsource, e)
        raise


# ============================================================
# Utilidades para read_fit
# ============================================================

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


# ============================================================
# Ejecución dinámica
# ============================================================

def _run_dynamic(
    executor: cf.ProcessPoolExecutor,
    worker_fn,
    args_iter: Iterable[Tuple[Any, ...]],
    total_tasks: int,
    max_in_flight: int,
    progress_every: int,
):
    """
    Ejecuta tareas con una cola acotada de futures.

    Ventajas frente a ex.map para muchos eventos:
      - no intenta registrar todos los futures de golpe;
      - mantiene una cantidad controlada de tareas en vuelo;
      - balancea mejor cuando algunos eventos tardan más que otros.
    """

    args_iterator = iter(args_iter)

    futures = set()
    submitted = 0
    completed = 0

    def submit_one():
        nonlocal submitted

        try:
            args = next(args_iterator)
        except StopIteration:
            return False

        fut = executor.submit(worker_fn, args)
        futures.add(fut)
        submitted += 1
        return True

    # Llenar cola inicial.
    for _ in range(min(max_in_flight, total_tasks)):
        ok = submit_one()
        if not ok:
            break

    # Consumir dinámicamente.
    while futures:
        done, futures = cf.wait(
            futures,
            return_when=cf.FIRST_COMPLETED,
        )

        for fut in done:
            # Si hubo excepción, se relanza acá.
            fut.result()

            completed += 1

            if progress_every > 0:
                if completed % progress_every == 0 or completed == total_tasks:
                    logging.warning(
                        "progress: %d/%d tasks completed",
                        completed,
                        total_tasks,
                    )

            # Mantener cola llena.
            if submitted < total_tasks:
                submit_one()


# ============================================================
# API pública: sim_fit paralelo
# ============================================================

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
    simfit_kwargs: Optional[dict] = None,
):
    """
    Ejecuta sim_fit en paralelo.

    Mejoras principales:
      - limita workers por CPUs reales, N_tr y total_events;
      - usa BLAS/OpenMP con 1 hilo por proceso;
      - permite controlar retries/contexto por env;
      - usa ejecución dinámica acotada para no crear todos los futures de golpe;
      - pasa kwargs científicos desde el config hacia sim_fit.
    """

    _blas_sanitize_global()

    total_events = int(total_events)
    if total_events <= 0:
        logging.warning("run_parallel: total_events <= 0, no hago nada.")
        return

    if simfit_kwargs is None:
        simfit_kwargs = {}

    if not isinstance(simfit_kwargs, dict):
        raise TypeError(
            f"simfit_kwargs debe ser dict o None. Recibido: {type(simfit_kwargs)}"
        )

    workers = _resolve_workers(N_tr, total_events)

    if max_in_flight is None:
        max_in_flight = max(workers, workers * _PREFETCH_FACTOR)
    else:
        max_in_flight = int(max_in_flight)

    max_in_flight = max(workers, min(max_in_flight, total_events))

    progress_every = _PROGRESS_EVERY

    print("=" * 80)
    print("[run_parallel]")
    print("=" * 80)
    print("available workers: ", _available_workers())
    print("requested N_tr:     ", N_tr)
    print("workers used:       ", workers)
    print("total_events:       ", total_events)
    print("max_in_flight:      ", max_in_flight)
    print("retries:            ", _RETRIES)
    print("chunksize env:      ", _CHUNKSIZE)
    print("mp_context:         ", _SIMFIT_MP_CONTEXT)
    print("skip mode:          ", _SKIP_MODE)
    print("progress_every:     ", progress_every)
    print("path_to_save_model: ", path_to_save_model)
    print("path_to_save_fit:   ", path_to_save_fit)
    print("path_dataslice:     ", path_dataslice)
    print("simfit_kwargs:      ", simfit_kwargs)
    print("=" * 80)

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
            simfit_kwargs,
        )
        for i in range(total_events)
    )

    ctx = mp.get_context(_SIMFIT_MP_CONTEXT)

    with cf.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
    ) as ex:
        _run_dynamic(
            executor=ex,
            worker_fn=_worker_sim_fit,
            args_iter=args_iter,
            total_tasks=total_events,
            max_in_flight=max_in_flight,
            progress_every=progress_every,
        )


# ============================================================
# API pública: read_fit paralelo
# ============================================================

def run_parallel_read_fit(
    nset,
    path_run,
    path_ephemerides,
    path_to_save_fit,
    model,
    algo,
    N_tr,
    max_in_flight: Optional[int] = None,
):
    """
    Ejecuta read_fit en paralelo.

    Lanza un worker por Event_i.h5 presente en set_sim{nset}.
    """

    _blas_sanitize_global()

    directory = Path(path_run) / f"set_sim{nset}"
    event_numbers = _list_event_numbers(directory)

    total_tasks = len(event_numbers)

    if total_tasks == 0:
        logging.warning(
            "run_parallel_read_fit: no encontré Event_i.h5 en %s",
            directory,
        )
        return

    workers = _resolve_workers(N_tr, total_tasks)

    if max_in_flight is None:
        max_in_flight = max(workers, workers * _PREFETCH_FACTOR)
    else:
        max_in_flight = int(max_in_flight)

    max_in_flight = max(workers, min(max_in_flight, total_tasks))

    progress_every = _PROGRESS_EVERY

    print("=" * 80)
    print("[run_parallel_read_fit]")
    print("=" * 80)
    print("directory:          ", directory)
    print("events found:       ", total_tasks)
    print("available workers:  ", _available_workers())
    print("requested N_tr:     ", N_tr)
    print("workers used:       ", workers)
    print("max_in_flight:      ", max_in_flight)
    print("retries:            ", _RETRIES)
    print("mp_context:         ", _READFIT_MP_CONTEXT)
    print("progress_every:     ", progress_every)
    print("=" * 80)

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

    ctx = mp.get_context(_READFIT_MP_CONTEXT)

    with cf.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
    ) as ex:
        _run_dynamic(
            executor=ex,
            worker_fn=_worker_read_fit,
            args_iter=args_iter,
            total_tasks=total_tasks,
            max_in_flight=max_in_flight,
            progress_every=progress_every,
        )


# ============================================================
# Main opcional para pruebas
# ============================================================

if __name__ == "__main__":
    mp.freeze_support()

    print("simfit_parallel_runner.py loaded")
    print("Available workers:", _available_workers())
    print("SIMFIT_RETRIES:", _RETRIES)
    print("SIMFIT_CHUNKSIZE:", _CHUNKSIZE)
    print("SIMFIT_MP_CONTEXT:", _SIMFIT_MP_CONTEXT)
    print("SIMFIT_READFIT_MP_CONTEXT:", _READFIT_MP_CONTEXT)
    print("SIMFIT_SKIP_MODE:", _SKIP_MODE)
