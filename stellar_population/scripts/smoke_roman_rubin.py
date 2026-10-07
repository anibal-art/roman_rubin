import argparse
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

from catalog.event_reader import (
    ROOT, load_event, load_index,
)
from functions_roman_rubin import sim_fit
import fit_lc

MODELS = {
    "BH": "PSPL",
    "FFP": "FSPL",
    "Planets_systems": "USBL",
}

SEASONS = [
    ("2027-02-11", "2027-04-24"),
    ("2027-08-16", "2027-10-27"),
    ("2028-02-11", "2028-04-24"),
    ("2030-02-11", "2030-04-24"),
    ("2030-08-16", "2030-10-27"),
    ("2031-02-11", "2031-04-24"),
]


def select_candidate(population):
    index = load_index()
    index = index[index.population == population]

    columns = ["t0", "u0", "tE", "W149", "i", "z"]

    if population == "FFP":
        columns.append("rho")

    if population == "Planets_systems":
        columns += ["q", "s"]

    for _, shard in index.iterrows():
        path = ROOT / str(shard["catalog_path"])
        df = pd.read_parquet(path, columns=columns)

        ok = (
            (df["W149"] < 23)
            & (df["z"] < 24)
            & (df["i"] < 25)
            & (df["u0"].abs() < 0.4)
        )

        if population == "BH":
            ok &= df["tE"].between(20, 200)

        elif population == "FFP":
            ok &= (
                df["tE"].between(0.05, 10)
                & df["rho"].between(0.001, 1)
            )

        else:
            ok &= (
                df["tE"].between(5, 150)
                & df["q"].between(1e-5, 0.1)
                & df["s"].between(0.5, 2.0)
            )

        season_ok = np.zeros(len(df), dtype=bool)

        for start, end in SEASONS:
            lo = Time(start).jd + 10
            hi = Time(end).jd - 10
            season_ok |= df["t0"].between(lo, hi).to_numpy()

        candidates = np.flatnonzero(
            ok.to_numpy() & season_ok
        )

        if len(candidates):
            return int(shard["row_start"]) + int(candidates[0])

    raise RuntimeError(
        "No encontré un candidato con estos cortes. "
        "No modificaré ni remuestrearé el catálogo."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--population",
        choices=list(MODELS),
        default="BH",
    )
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()

    population = args.population

    global_i = (
        args.index if args.index is not None
        else select_candidate(population)
    )

    event = load_event(population, global_i)

    model = MODELS[population]
    seed = int(event["simulation_seed"])

    assert "event_ra" in inspect.signature(
        fit_lc.fit_rubin_roman
    ).parameters
    assert "event_dec" in inspect.signature(
        fit_lc.fit_rubin_roman
    ).parameters

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    out = (
        ROOT / "stellar_population/pilots"
        / "roman_rubin_integration"
        / f"{population}_{global_i}_{stamp}"
    )

    sim_dir = out / "simulation"
    fit_dir = out / "fits"
    results_dir = out / "results"

    for directory in (sim_dir, fit_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metadata = {
        "population": population,
        "global_index": global_i,
        "simulation_seed": seed,
        "genulens_event_id": str(event["genulens_event_id"]),
        "star_id": str(event["star_id"]),
        "field_id": str(event["field_id"]),
        "ra": float(event["ra"]),
        "dec": float(event["dec"]),
        "model": model,
        "output": str(out),
    }

    (out / "event_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("\nPILOT EVENT")
    print(json.dumps(metadata, indent=2), flush=True)

    t0 = float(event["t0"])

    # Ventana corta exclusivamente para la prueba de integración.
    half_window = 3.0 if population == "FFP" else 30.0

    result = sim_fit(
        i=seed,
        system_type=population,
        model=model,
        algo="TRF",
        path_TRILEGAL_set="custom_system",
        path_GENULENS_set="custom_system",
        path_to_save_model=str(sim_dir) + "/",
        path_to_save_fit=str(fit_dir) + "/",
        path_ephemerides=str(
            ROOT / "ephemerides/Roman_positions.npy"
        ),
        path_to_save_results=str(results_dir),
        catalog_mode="custom_system",
        custom_system=event,
        use_roman=True,
        use_rubin=True,
        truth_parallax=True,
        fit_model=model,
        fit_parallax=True,
        rubin_pointing_mode="source",
        rubin_cache_cell_deg=None,
        apply_detection_criteria=True,
        apply_photometric_filter=True,
        time_window=(t0 - half_window, t0 + half_window),
        fit_time_window=None,
        return_data=True,
        initial_guess="truth",
    )

    status = result.get("status")
    print("\nSIMULATION STATUS:", status)

    params = result.get("event_params", {})

    if status == "fitted":
        assert result["fit_rr"] is not None
        assert result["fit_roman"] is not None

        for key, expected in (
            ("event_ra_used", event["ra"]),
            ("event_dec_used", event["dec"]),
            ("maf_ra_used", event["ra"]),
            ("maf_dec_used", event["dec"]),
        ):
            assert np.isclose(
                float(params[key]),
                float(expected),
                atol=1e-8,
            ), key

        print("\nROMAN + RUBIN FIT: OK")
        print("ROMAN-ONLY FIT: OK")
        print("COORDINATES: OK")
        print("\nINTEGRATION SMOKE TEST PASSED")

    elif status == "rejected":
        print(
            "Event rejected by the scientific selection. "
            "This is not necessarily a software error."
        )

    else:
        raise RuntimeError(f"Unexpected status: {status}")

    (out / "status.json").write_text(
        json.dumps(
            {**metadata, "status": status},
            indent=2,
        )
    )

    print("RESULTS:", out)


if __name__ == "__main__":
    main()
