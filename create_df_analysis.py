from pathlib import Path
from class_analysis import Analysis_Event, labels_params, event_fits
import pandas as pd
import os, re, time
from tqdm.auto import tqdm
import warnings
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from ulens_params import event_param
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import traceback

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Helpers de E/S en formato dataset particionado (Parquet, partición Set)
# ---------------------------------------------------------------------

def _flush_dataset(rows, base_dir, schema=None):
    """
    Escribe 'rows' (list[dict]) en un dataset Parquet particionado por 'Set'.
    Si el dataset/partición no existe, se crea; si existe, se agrega (append).
    """
    if not rows:
        return
    table = pa.Table.from_pylist(rows, schema=schema)
    ds.write_dataset(
        data=table,
        base_dir=base_dir,
        format="parquet",
        partitioning=["Set"],             # crea carpetas Set=<n>
        existing_data_behavior="overwrite_or_append"
    )
    rows.clear()


def _existing_sources_in_set(ds_root: str, nset: int) -> set[int]:
    """
    Devuelve los 'Source' ya presentes en la partición Set=<nset> del dataset.
    No asume que la columna 'Set' esté materializada dentro del archivo Parquet.
    """
    part_dir = os.path.join(ds_root, f"Set={nset}")
    if not os.path.isdir(part_dir):
        return set()
    try:
        dset = ds.dataset(part_dir, format="parquet")
        # leer solo 'Source' si existe; si no, leer todo y validar
        try:
            tbl = dset.to_table(columns=["Source"])
        except Exception:
            tbl = dset.to_table()
    except Exception:
        return set()

    if tbl.num_rows == 0 or "Source" not in tbl.schema.names:
        return set()

    vals = tbl.column("Source").to_pylist()
    return {int(x) for x in vals if x is not None}

# ---------------------------------------------------------------------
# Rutas y construcción de paths
# ---------------------------------------------------------------------

def get_fit_rr_path(path_run, nset, nevent):
    return f"{path_run}/set_fit{nset}/Event_RR_{nevent}_TRF.npy"

def get_fit_roman_path(path_run, nset, nevent):
    return f"{path_run}/set_fit{nset}/Event_Roman_{nevent}_TRF.npy"

def get_model_path(path_run, nset, nevent):
    return f"{path_run}/set_sim{nset}/Event_{nevent}.h5"

def make_paths(script_dir, path_run_model, path_run_fit):
    return {
        "fit_rr": get_fit_rr_path,
        "fit_roman": get_fit_roman_path,
        "model": get_model_path,
        "script_dir": script_dir,
        "path_run_model": path_run_model,
        "path_run_fit": path_run_fit,
    }

# ---------------------------------------------------------------------
# Lógica científica (tu código existente, sin cambios sustantivos)
# ---------------------------------------------------------------------

def check_source_set_in_csv(file_path, source_value, set_value):
    chunksize = 200
    for chunk in pd.read_csv(file_path, usecols=['Source', 'Set'], chunksize=chunksize):
        if ((chunk['Source'] == source_value) & (chunk['Set'] == set_value)).any():
            return True
    return False

def new_data(Event, nset, nevent, cols_fit, data):
    new_row = pd.DataFrame(columns=cols_fit)
    new_row['Source'] = [nevent]
    new_row['Set'] = [nset]
    fit_vals = Event.dict_fit_vals(data)
    for key in fit_vals:
        new_row[key] = [fit_vals[key]]
    piemc = Event.piE_MC(data)
    new_row['piE'] = [piemc['piE']]
    new_row['piE_err'] = [piemc['err_piE']]

    chichidof = Event.chichi_dof(data)
    new_row['chichi'] = [chichidof['chi2']]
    new_row['dof'] = [chichidof['dof']]

    if not Event.model == 'PSPL':
        fitmassv3 = Event.fit_mass_v3(data)
        new_row['err_mass_v3'] = [fitmassv3['err_mass']]
        new_row['mass_v3'] = [fitmassv3['mass']]
    fitmassv2 = Event.fit_mass_v2(data)
    fitmassv1 = Event.fit_mass_v1(data)
    new_row['err_mass_v2'] = [fitmassv2['err_mass']]
    new_row['mass_v2'] = [fitmassv2['mass']]
    new_row['err_mass_v1'] = [fitmassv1['err_mass']]
    new_row['mass_v1'] = [fitmassv1['mass']]
    new_row['ln_likelihood'] = [data['ln_likelihood']]

    return new_row

def extract_data_event(model, nset, nevent, system_type, paths):
    script_dir = paths["script_dir"]
    path_run_model = paths["path_run_model"]
    path_model = paths["model"](path_run_model, nset, nevent)

    if 'dfit' not in system_type:
        if model == 'USBL':
            cols_true = ['Source', 'Set'] + labels_params(model) + ['Category', 'Category_p', 'mass', 'sel_crit', 'piE']
        elif model == 'FSPL':
            cols_true = ['Source', 'Set'] + labels_params(model) + ['Category', 'mass', 'sel_crit', 'piE', 'crit_FFP_Rubin']
        else:
            cols_true = ['Source', 'Set'] + labels_params(model) + ['Category', 'mass', 'sel_crit', 'piE', 'W149', 'u', 'g', 'r', 'i', 'z', 'y']

    if "Planets_systems" in system_type:
        st = "Planets_systems"
    elif "FFP" in system_type:
        st = "FFP"
    elif "BH" in system_type:
        st = "BH"

    path_run_fit = paths["path_run_fit"]
    path_fit_rr = paths["fit_rr"](path_run_fit, nset, nevent)
    path_fit_roman = paths["fit_roman"](path_run_fit, nset, nevent)

    Event = Analysis_Event(model, path_model, path_fit_rr, path_fit_roman)
    Event.load_data_fit()
    Event.load_data_sim()

    # df true
    if 'dfit' not in system_type:
        true = Event.true_values()
        fit_rr, fit_roman = Event.fit_values()

        new_data_true = pd.DataFrame(columns=cols_true)
        new_data_true['Source'] = [nevent]
        new_data_true['Set'] = [nset]
        for key in Event.labels_params():
            new_data_true[key] = [true[key]]
        new_data_true['piE'] = [np.sqrt(true['piEN']**2 + true['piEE']**2)]

        if model == 'USBL':
            npts_speak = Event.count_points_secon_peak()
            for f in npts_speak:
                new_data_true['anomaly_' + f] = [npts_speak[f]]

        new_data_true['mass'] = [Event.mass_true()]

        npts = Event.count_points()
        for f in npts:
            new_data_true[f] = [npts[f]]

        npts_pk = Event.count_points_peak()
        for f in npts_pk:
            new_data_true[f + '_peak'] = [npts_pk[f]]

        npts_pk_narrow = Event.count_points_narrow_peak()
        for f in npts_pk:
            new_data_true[f + '_npeak'] = [npts_pk_narrow[f]]

        npts_pk_left = Event.count_points_left_peak()
        for f in npts_pk:
            new_data_true[f + '_lpeak'] = [npts_pk_left[f]]

        npts_pk_right = Event.count_points_right_peak()
        for f in npts_pk:
            new_data_true[f + '_rpeak'] = [npts_pk_right[f]]

    else:
        new_data_true = []

    cols_fit = ['Source', 'Set'] + labels_params(model) + \
               [f + '_err' for f in labels_params(model)] + \
               ['piE', 'piE_err'] + ['chichi', 'dof'] + \
               ['mass_v1', 'mass_v2', 'err_mass_v1', 'err_mass_v2']
    if not model == 'PSPL':
        cols_fit = cols_fit + ['mass_v3', 'err_mass_v3']

    new_data_rr = new_data(Event, nset, nevent, cols_fit, Event.fit_rr_data)
    new_data_roman = new_data(Event, nset, nevent, cols_fit, Event.fit_roman_data)

    return new_data_true, new_data_rr, new_data_roman



def process_chunk_worker(chunk, model, system_type, paths):
    """
    Procesa un LOTE de eventos (lista de (nset, nevent)).
    Devuelve dicts por set: {'set': nset, 'true': [...], 'rr': [...], 'roman': [...], 'errors': int, 'skips': int}
    """
    out_by_set = {}
    errors = 0
    skips = 0

    for (nset, nevent) in chunk:
        try:
            if 'dfit' not in system_type:
                path_model = paths["model"](paths["path_run_model"], nset, nevent)
                if not os.path.isfile(path_model) or os.path.getsize(path_model) == 0:
                    skips += 1
                    continue

            new_data_true, new_data_rr, new_data_roman = extract_data_event(model, nset, nevent, system_type, paths)

            ob = out_by_set.setdefault(nset, {"set": nset, "true": [], "rr": [], "roman": []})

            if 'dfit' not in system_type and isinstance(new_data_true, pd.DataFrame) and not new_data_true.empty:
                ob["true"].append(new_data_true.iloc[0].to_dict())
            if isinstance(new_data_rr, pd.DataFrame) and not new_data_rr.empty:
                ob["rr"].append(new_data_rr.iloc[0].to_dict())
            if isinstance(new_data_roman, pd.DataFrame) and not new_data_roman.empty:
                ob["roman"].append(new_data_roman.iloc[0].to_dict())

        except Exception:
            errors += 1
            continue

    # anexar contadores
    for nset, ob in out_by_set.items():
        ob["errors"] = errors
        ob["skips"] = skips
    return out_by_set


# ---------------------------------------------------------------------
# Worker por set (paralelizado por ProcessPoolExecutor)
# ---------------------------------------------------------------------

SHOW_INNER = False  # mantener False para evitar múltiples barras en paralelo
def process_set(nset, model, system_type, paths, out_dirs_ds):
    """
    Procesa todos los eventos de un set en paralelo (nivel nodo).
    """
    import os, time, sys, traceback
    from concurrent.futures import ProcessPoolExecutor, as_completed

    MAX_WORKERS = min(os.cpu_count() or 1, 32)
    REPORT_INTERVAL_SEC = 120
    BLOCK_FLUSH = 2000

    t0 = time.perf_counter()
    set_dir_fit = os.path.join(paths["path_run_fit"], f"set_fit{nset}")
    events_all = event_fits(set_dir_fit)
    total_events = len(events_all)

    existing_rr = _existing_sources_in_set(out_dirs_ds["rr_ds"], nset)
    existing_roman = _existing_sources_in_set(out_dirs_ds["roman_ds"], nset)
    existing_true = _existing_sources_in_set(out_dirs_ds["true_ds"], nset) if 'dfit' not in system_type else set()

    def _already_done(ev: int) -> bool:
        if 'dfit' not in system_type:
            return (ev in existing_true) and (ev in existing_rr) and (ev in existing_roman)
        else:
            return (ev in existing_rr) and (ev in existing_roman)

    events = [e for e in events_all if not _already_done(e)]
    n_to_process = len(events)
    if n_to_process == 0:
        print(f"[DONE-SET {nset}] Ningún evento nuevo. Total={total_events}")
        return {"set": nset, "nuevos": 0, "elapsed_sec": 0.0, "total_scan": total_events}

    buf_true, buf_rr, buf_roman = [], [], []
    processed, errors = 0, 0
    last_report_time = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_event_worker, ev, nset, model, system_type, paths): ev for ev in events}
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            if "error" in result:
                errors += 1
                continue

            if 'true' in result:
                buf_true.append(result['true'])
            if 'rr' in result:
                buf_rr.append(result['rr'])
            if 'roman' in result:
                buf_roman.append(result['roman'])

            processed += 1

            now = time.time()
            if (processed % 1000 == 0) or (now - last_report_time >= REPORT_INTERVAL_SEC):
                pct = (processed / n_to_process) * 100
                elapsed = now - t0
                print(f"[PROGRESS-SET {nset}] {processed}/{n_to_process} ({pct:.2f}%) "
                      f"| elapsed={elapsed/60:.2f} min | evt/s={processed/max(elapsed,1):.1f}",
                      flush=True)
                last_report_time = now

            if processed % BLOCK_FLUSH == 0:
                if 'dfit' not in system_type:
                    _flush_dataset(buf_true, out_dirs_ds["true_ds"])
                _flush_dataset(buf_rr, out_dirs_ds["rr_ds"])
                _flush_dataset(buf_roman, out_dirs_ds["roman_ds"])

    # Escritura final
    if 'dfit' not in system_type:
        _flush_dataset(buf_true, out_dirs_ds["true_ds"])
    _flush_dataset(buf_rr, out_dirs_ds["rr_ds"])
    _flush_dataset(buf_roman, out_dirs_ds["roman_ds"])

    elapsed = time.perf_counter() - t0
    eps = processed / max(elapsed, 1.0)

    print(f"[DONE-SET {nset}] {processed} eventos nuevos en {elapsed/60:.2f} min "
          f"({eps:.1f} evt/s) | errores={errors}", flush=True)

    return {
        "set": nset,
        "nuevos": processed,
        "errores": errors,
        "elapsed_sec": elapsed,
        "total_scan": total_events,
        "evt_per_sec": eps,
    }

# ==============================================================
# FUNCIÓN DE NIVEL SUPERIOR — ahora picklable por multiprocessing
# ==============================================================

# ---------------------------------------------------------------------
# Orquestación: crear y ejecutar jobs por set (barra global única)
# ---------------------------------------------------------------------
def create_df(path_run, save_results, model, system_type):
    """
    Paraleliza a nivel global con COLA de eventos y envía LOTES (chunks) al pool.
    Muestra progreso inmediato y también cada REPORT_INTERVAL_SEC aunque no se complete ningún lote,
    usando concurrent.futures.wait(..., timeout=...).
    """
    import os, re, sys
    from time import perf_counter
    from math import ceil
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

    # --- paths base ---
    script_dir = str(Path(__file__).parent)
    if "Planets_systems" in system_type:
        st = "Planets_systems"
    elif "FFP" in system_type:
        st = "FFP"
    elif "BH" in system_type:
        st = "BH"
    else:
        st = system_type

    path_run_model = os.path.join(path_run, st)
    path_run_fit   = os.path.join(path_run, system_type)
    paths = make_paths(script_dir, path_run_model, path_run_fit)

    # --- output datasets particionados ---
    out_dirs_ds = {
        "true_ds":  f"{save_results}/true_ds",
        "rr_ds":    f"{save_results}/fit_rr_ds",
        "roman_ds": f"{save_results}/fit_roman_ds",
    }
    for d in out_dirs_ds.values():
        os.makedirs(d, exist_ok=True)

    # --- params desde env ---
    MAX_WORKERS = int(os.environ.get("MAX_WORKERS", str(os.cpu_count() or 1)))
    BLOCK_SIZE  = int(os.environ.get("BLOCK_SIZE", "1000"))   # sugerencia: 200–1000
    REPORT_INTERVAL_SEC = int(os.environ.get("REPORT_INTERVAL_SEC", "120"))
    NO_IDEMP = os.environ.get("NO_IDEMPOTENCY_SCAN", "0") == "1"

    is_tty = sys.stdout.isatty()
    if not is_tty:
        print(f"[INFO] STDOUT no es TTY: modo 'log lines' activo. Sets=?, workers={MAX_WORKERS}", flush=True)

    # --- descubrir sets ---
    strings = os.listdir(path_run_fit)
    sets = []
    for s in strings:
        m = re.search(r'\d+', s)
        if m:
            sets.append(int(m.group()))
    SETS = sorted(set(sets))

    # --- construir cola global (nset, nevent) ---
    global_jobs = []
    total_scan_by_set = {}

    for nset in SETS:
        events = event_fits(os.path.join(path_run_fit, f"set_fit{nset}"))
        total_scan_by_set[nset] = len(events)
        if NO_IDEMP:
            for ev in events:
                global_jobs.append((nset, ev))
        else:
            existing_rr    = _existing_sources_in_set(out_dirs_ds["rr_ds"], nset)
            existing_roman = _existing_sources_in_set(out_dirs_ds["roman_ds"], nset)
            existing_true  = _existing_sources_in_set(out_dirs_ds["true_ds"], nset) if ('dfit' not in system_type) else set()
            def _already_done(ev: int) -> bool:
                if 'dfit' not in system_type:
                    return (ev in existing_true) and (ev in existing_rr) and (ev in existing_roman)
                else:
                    return (ev in existing_rr) and (ev in existing_roman)
            for ev in events:
                if not _already_done(ev):
                    global_jobs.append((nset, ev))

    if not global_jobs:
        print("[INFO] No hay eventos para procesar (tras idempotencia).", flush=True)
        return

    global_jobs.sort()
    num_jobs = len(global_jobs)
    num_chunks = ceil(num_jobs / BLOCK_SIZE)
    chunks = [global_jobs[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE] for i in range(num_chunks)]

    # buffers por set
    buf_true_by_set  = {nset: [] for nset in SETS} if ('dfit' not in system_type) else {}
    buf_rr_by_set    = {nset: [] for nset in SETS}
    buf_roman_by_set = {nset: [] for nset in SETS}

    # métricas globales
    t0 = perf_counter()
    last_report = t0
    done_events = 0
    errors = 0
    skips = 0

    # Mensaje inicial inmediato (para que veas algo al principio)
    print(f"[QUEUE] sets={len(SETS)} | jobs={num_jobs} | chunks={num_chunks} | "
          f"BLOCK_SIZE={BLOCK_SIZE} | workers={MAX_WORKERS}", flush=True)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_idx = {}
        for idx, chunk in enumerate(chunks):
            future = pool.submit(process_chunk_worker, chunk, model, system_type, paths)
            future_to_idx[future] = idx

        pending = set(future_to_idx.keys())

        while pending:
            # Espera a que se complete al menos 1 futuro, o bien timeout para reporte periódico
            done_set, pending = wait(pending, timeout=REPORT_INTERVAL_SEC, return_when=FIRST_COMPLETED)

            # Si venció el timeout y no terminó ninguno, igual imprimimos latido/progreso
            now = perf_counter()
            if not done_set:
                elapsed = now - t0
                eps = done_events / max(elapsed, 1.0)
                pct = (done_events / num_jobs) * 100.0
                print(f"[PROGRESS-ALL] {done_events}/{num_jobs} ({pct:.2f}%) | "
                      f"elapsed={elapsed/60:.2f} min | evt/s={eps:.1f} | errors={errors} | skips={skips}",
                      flush=True)
                last_report = now
                continue

            # Procesar todos los que se completaron en este ciclo
            for fut in done_set:
                idx = future_to_idx[fut]
                try:
                    res = fut.result()
                except Exception:
                    # Si falló el lote completo, contamos como errores = tamaño del lote
                    errors += len(chunks[idx])
                    continue

                # Merge por set
                for nset, ob in res.items():
                    if 'dfit' not in system_type and ob.get("true"):
                        buf_true_by_set[nset].extend(ob["true"])
                    if ob.get("rr"):
                        buf_rr_by_set[nset].extend(ob["rr"])
                    if ob.get("roman"):
                        buf_roman_by_set[nset].extend(ob["roman"])
                    errors += ob.get("errors", 0)
                    skips  += ob.get("skips", 0)

                # Avance en eventos (tamaño del chunk terminado)
                done_events += len(chunks[idx])

            # Reporte periódico (aunque sea tras completar alguno)
            now = perf_counter()
            if (now - last_report) >= REPORT_INTERVAL_SEC:
                elapsed = now - t0
                eps = done_events / max(elapsed, 1.0)
                pct = (done_events / num_jobs) * 100.0
                print(f"[PROGRESS-ALL] {done_events}/{num_jobs} ({pct:.2f}%) | "
                      f"elapsed={elapsed/60:.2f} min | evt/s={eps:.1f} | errors={errors} | skips={skips}",
                      flush=True)
                last_report = now

            # Flush parcial por memoria (opcional)
            if (done_events % (BLOCK_SIZE*4)) == 0:
                if 'dfit' not in system_type:
                    for s, rows in list(buf_true_by_set.items()):
                        _flush_dataset(rows, out_dirs_ds["true_ds"])
                for s, rows in list(buf_rr_by_set.items()):
                    _flush_dataset(rows, out_dirs_ds["rr_ds"])
                for s, rows in list(buf_roman_by_set.items()):
                    _flush_dataset(rows, out_dirs_ds["roman_ds"])

    # Flush final
    if 'dfit' not in system_type:
        for s, rows in buf_true_by_set.items():
            _flush_dataset(rows, out_dirs_ds["true_ds"])
    for s, rows in buf_rr_by_set.items():
        _flush_dataset(rows, out_dirs_ds["rr_ds"])
    for s, rows in buf_roman_by_set.items():
        _flush_dataset(rows, out_dirs_ds["roman_ds"])

    elapsed = perf_counter() - t0
    eps = done_events / max(elapsed, 1.0)
    print(f"[DONE-ALL] {done_events}/{num_jobs} eventos escritos | elapsed={elapsed/60:.2f} min | "
          f"evt/s={eps:.1f} | errors={errors} | skips={skips}", flush=True)

