"""Reader for the frozen GBTDS 2000-event production catalog."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = (
    ROOT / "stellar_population/production"
    / "gbtds_2000_v1_event_shards.parquet"
)

N_PER_POPULATION = 798_135

POPULATIONS = (
    "FFP",
    "BH",
    "Planets_systems",
)

MODEL_FOR_VALIDATION = {
    "FFP": "FSPL",
    "BH": "FSPL",
    "Planets_systems": "USBL",
}

REQUIRED = (
    "field_id", "genulens_event_id", "star_id",
    "system_type", "t0", "u0", "tE",
    "piEN", "piEE", "W149",
    "u", "g", "r", "i", "z", "Y",
)


@lru_cache(maxsize=1)
def load_index():
    df = pd.read_parquet(INDEX_PATH)

    for pop in POPULATIONS:
        part = df[df["population"] == pop]

        assert len(part) == 510, (pop, len(part))

        starts = part["row_start"].to_numpy(dtype=np.int64)
        stops = part["row_stop"].to_numpy(dtype=np.int64)

        assert starts[0] == 0
        assert stops[-1] == N_PER_POPULATION
        assert np.all(starts[1:] == stops[:-1])

    return df


def read_parquet_row(path, row_number):
    """Read a single physical row from its Parquet row group."""
    pf = pq.ParquetFile(path)

    remaining = int(row_number)

    for group in range(pf.num_row_groups):
        n = pf.metadata.row_group(group).num_rows

        if remaining < n:
            return (
                pf.read_row_group(group)
                .slice(remaining, 1)
                .to_pylist()[0]
            )

        remaining -= n

    raise IndexError((path, row_number))


def load_event(population, global_index):
    if population not in POPULATIONS:
        raise ValueError(f"Unknown population: {population}")

    global_index = int(global_index)

    if not 0 <= global_index < N_PER_POPULATION:
        raise IndexError(global_index)

    index = load_index()
    subset = (
        index[index["population"] == population]
        .reset_index(drop=True)
    )

    starts = subset["row_start"].to_numpy(dtype=np.int64)

    shard_number = int(
        np.searchsorted(starts, global_index, side="right") - 1
    )

    shard = subset.iloc[shard_number]

    assert global_index < int(shard["row_stop"])

    local_index = global_index - int(shard["row_start"])

    path = ROOT / str(shard["catalog_path"])

    event = read_parquet_row(path, local_index)

    assert event["field_id"] == shard["field_id"]
    assert event["system_type"] == population

    missing = [key for key in REQUIRED if key not in event]
    if missing:
        raise KeyError(f"Missing event columns: {missing}")

    if population == "Planets_systems":
        for key in ("s", "q", "alpha", "rho"):
            if key not in event:
                raise KeyError(key)

    if population in ("FFP", "BH") and "rho" not in event:
        raise KeyError("rho")

    for key in ("t0", "u0", "tE", "piEN", "piEE",
                "W149", "u", "g", "r", "i", "z", "Y"):
        assert np.isfinite(float(event[key])), (population, key)

    assert float(event["tE"]) > 0

    # Physical event direction = exact OpSim query direction.
    event["ra"] = float(shard["ra"])
    event["dec"] = float(shard["dec"])
    event["maf_ra"] = event["ra"]
    event["maf_dec"] = event["dec"]

    # Distinct, collision-free noise seeds across all three populations.
    population_number = POPULATIONS.index(population)

    event["simulation_seed"] = (
        population_number * N_PER_POPULATION + global_index
    )

    # Keep the original event_seed from the physical catalog unchanged.
    event["catalog_global_index"] = global_index
    event["catalog_local_index"] = local_index
    event["catalog_uid"] = (
        f"gbtds_2000_v1:{population}:{global_index:07d}"
    )

    return event


if __name__ == "__main__":
    from functions_roman_rubin import (
        build_event_params_from_custom_system,
    )

    test_indices = (0, N_PER_POPULATION // 2,
                    N_PER_POPULATION - 1)

    for pop in POPULATIONS:
        for global_i in test_indices:
            event = load_event(pop, global_i)

            # Validate the actual interface used by sim_fit.
            validated = build_event_params_from_custom_system(
                custom_system=event,
                model=MODEL_FOR_VALIDATION[pop],
                use_roman=True,
                use_rubin=True,
                rubin_pointing_mode="source",
            )

            assert validated["tE"] == event["tE"]
            assert validated["piEN"] == event["piEN"]
            assert validated["piEE"] == event["piEE"]
            assert validated["ra"] == event["ra"]
            assert validated["dec"] == event["dec"]

            print(
                pop,
                global_i,
                event["field_id"],
                f"tE={event['tE']:.6g}",
                f"RA={event['ra']:.7f}",
                f"Dec={event['dec']:.7f}",
                f"seed={event['simulation_seed']}",
            )

    print("\nCATALOG READER AUDIT PASSED")
