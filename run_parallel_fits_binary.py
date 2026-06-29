#!/usr/bin/env python3
import os, sys, signal, logging, tempfile, concurrent.futures as cf, multiprocessing as mp
from math import ceil

from functions_roman_rubin import sim_fit,sim_event
# from functions_roman_rubin import model_rubin_roman
from functions_roman_rubin import read_data, save_sim
from fit_lc import fit_rubin_roman, model_rubin_roman
import pyLIMA_plots
from fit_lc import fit_rubin_roman
# ---------- YOUR IMPORTS ----------
# from your_module import read_data, fit_rubin_roman

path_to_save_model = '/home/anibal/microlensing/simulation_Rubin/roman_rubin/test_sim_fit/sim/'
path_to_save_fit   = '/home/anibal/microlensing/simulation_Rubin/roman_rubin/test_sim_fit/fit/'
path_ephemerides   = '/home/anibal/microlensing/simulation_Rubin/roman_rubin/ephemerides/Roman_positions.npy'
# ----------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# (Optional but recommended on clusters) avoid BLAS oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Use node-local scratch if available (good on HPC to avoid NFS thrash)
tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
tempfile.tempdir = tmp

# ---- config you likely already have somewhere ----
# path_to_save_model = "/path/to/save/model"      # <- set me
# path_to_save_fit   = "/path/to/save/fit"        # <- set me
# path_ephemerides   = "/path/to/ephemerides"     # <- set me
algo               = "TRF"
indices            = range(0, 100)  # 1..99 inclusive

# max parallel workers = 16 (or respect Slurm if fewer CPUs were allocated)
SLURM_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", "16") or "16")
MAX_WORKERS = min(16, max(1, SLURM_CPUS))

# Graceful shutdown flag for SIGTERM (Slurm sends a TERM before KILL if configured)
STOP = {"flag": False}
def _term_handler(signum, frame):
    logging.warning("Received SIGTERM; finishing in-flight tasks and stopping new ones.")
    STOP["flag"] = True
signal.signal(signal.SIGTERM, _term_handler)

def _safe_get_band(bands, name):
    """Return numpy array [time,mag,err_mag] or [] if unavailable/empty."""
    try:
        b = bands.get(name)
        if b is None: 
            return []
        # 'to_pandas' exists in your original snippet; keep that pathway.
        if hasattr(b, "to_pandas"):
            df = b[['time','mag','err_mag']].to_pandas()
            return df.values if len(df) else []
        # Fallback if it's already a pandas DataFrame-like
        if hasattr(b, "loc") and hasattr(b, "values"):
            df = b[['time','mag','err_mag']]
            return df.values if len(df) else []
        # Last resort: assume dict of arrays
        if all(k in b for k in ("time","mag","err_mag")):
            import numpy as np
            if len(b["mag"]) == 0: 
                return []
            return np.c_[b["time"], b["mag"], b["err_mag"]]
    except Exception as e:
        logging.error(f"Error preparing band {name}: {e}")
    return []

def run_one(i: int):
    """
    Do the work for a single event index i.
    Returns a small status dict you can log or ignore.
    """
    if STOP["flag"]:
        return {"i": i, "status": "skipped_due_to_stop"}

    try:
        info_dataset, pyLIMA_parameters, bands = read_data(f"{path_to_save_model}/Event_{int(i)}.h5")

        # Build lc_to_fit with safe defaults
        band_names = ["W149","u","g","r","i","z","y"]
        lc_to_fit = {name: _safe_get_band(bands, name) for name in band_names}

        origin = info_dataset[2]
        rango = 1

        # Unique prefixes per event to avoid filename collisions
        usbl_prefix = f"USBL_{int(i)}_"
        pspl_prefix = f"PSPL_{int(i)}_"

        # USBL fit
        fit_usbl, event_fit_usbl, pyLIMAmodel_usbl = fit_rubin_roman(
            usbl_prefix, pyLIMA_parameters, path_to_save_fit, path_ephemerides,
            "USBL_NoPiE", algo, origin, rango,
            lc_to_fit["W149"], lc_to_fit["u"], lc_to_fit["g"], lc_to_fit["r"],
            lc_to_fit["i"], lc_to_fit["z"], lc_to_fit["y"]
        )

        # PSPL fit
        fit_pspl, event_fit_pspl, pyLIMAmodel_pspl = fit_rubin_roman(
            pspl_prefix, pyLIMA_parameters, path_to_save_fit, path_ephemerides,
            "PSPL_NoPiE", algo, origin, rango,
            lc_to_fit["W149"], lc_to_fit["u"], lc_to_fit["g"], lc_to_fit["r"],
            lc_to_fit["i"], lc_to_fit["z"], lc_to_fit["y"]
        )

        return {"i": i, "status": "ok"}

    except Exception as e:
        logging.exception(f"[i={i}] failed: {e}")
        return {"i": i, "status": "error", "msg": str(e)}

def main():
    # Prefer spawn on HPC so workers don't inherit weird state
    ctx = mp.get_context("spawn")

    logging.info(f"Running {len(list(indices))} events with up to {MAX_WORKERS} parallel workers.")
    results = []

    # ------- Option A (recommended): bounded pool of 16 workers --------
    # This keeps at most 16 running at once; as one finishes, another starts.
    with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as ex:
        future_to_i = {ex.submit(run_one, i): i for i in indices}
        for fut in cf.as_completed(future_to_i):
            res = fut.result()
            results.append(res)
            if res["status"] == "ok":
                logging.info(f"i={res['i']} done")
            elif res["status"] == "skipped_due_to_stop":
                logging.warning(f"i={res['i']} skipped due to shutdown")
            else:
                logging.error(f"i={res['i']} error: {res.get('msg','')}")

            if STOP["flag"]:
                # Stop scheduling new work (already all submitted), but we can break early if desired
                logging.warning("Stop flag set; breaking from result collection.")
                break

    # ------- Option B (strict 'chunks of 16' batches) -------
    # If you prefer to *wait* for each batch of 16 to finish before launching
    # the next batch, comment Option A and uncomment this block:
    #
    # from itertools import islice
    # it = iter(indices)
    # batch_num = 0
    # while True:
    #     batch = list(islice(it, MAX_WORKERS))
    #     if not batch:
    #         break
    #     batch_num += 1
    #     logging.info(f"Starting batch {batch_num} with {len(batch)} items: {batch}")
    #     with cf.ProcessPoolExecutor(max_workers=len(batch), mp_context=ctx) as ex:
    #         for res in ex.map(run_one, batch, chunksize=1):
    #             results.append(res)
    #             if res["status"] == "ok":
    #                 logging.info(f"i={res['i']} done")
    #             else:
    #                 logging.error(f"i={res['i']} status={res['status']} msg={res.get('msg','')}")
    #     if STOP["flag"]:
    #         logging.warning("Stop flag set; exiting after this batch.")
    #         break

    # Summarize
    ok = sum(r["status"] == "ok" for r in results)
    err = sum(r["status"] == "error" for r in results)
    skip = sum(r["status"] == "skipped_due_to_stop" for r in results)
    logging.info(f"Done. ok={ok}, error={err}, skipped={skip}")

if __name__ == "__main__":
    main()
