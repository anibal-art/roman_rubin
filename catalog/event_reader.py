"""Reader for the materialized GBTDS 300-event Amax production catalog."""

from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from astropy.coordinates import SkyCoord
import astropy.units as u


ROOT = Path(__file__).resolve().parents[1]


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


POPULATION_FILES = {
    "FFP": "ffp_events_amax.parquet",
    "BH": "bh_events_amax.parquet",
    "Planets_systems": "binary_lens_events_amax.parquet",
}


REQUIRED = (
    "catalog_event_index",
    "field_id",
    "genulens_event_id",
    "star_id",
    "system_type",
    "t0",
    "u0",
    "tE",
    "piEN",
    "piEE",
    "W149",
    "u",
    "g",
    "r",
    "i",
    "z",
    "Y",
)


GEOMETRY_PATH = (
    ROOT
    / "stellar_population"
    / "config"
    / "gbtds_trilegal_cells_production.csv"
)


def _resolve_catalog_dir():
    """
    Resolve the frozen materialized Amax catalogue location.

    Priority:
    1. ROMAN_RUBIN_EVENT_CATALOG_DIR
    2. CHE shared production location
    3. repository-local assembled catalogue
    """

    env = os.environ.get(
        "ROMAN_RUBIN_EVENT_CATALOG_DIR"
    )

    candidates = []

    if env:
        candidates.append(
            Path(env).expanduser()
        )

    candidates.extend(
        [
            Path(
                "/share/storage3/rubin/microlensing/"
                "romanrubin/catalogs/"
                "gbtds_300_v1_amax_v1"
            ),
            (
                ROOT
                / "stellar_population"
                / "precomputed_events"
                / "assembled"
                / "gbtds_300_v1_amax_v1"
            ),
        ]
    )

    for path in candidates:
        if all(
            (path / filename).is_file()
            for filename in POPULATION_FILES.values()
        ):
            return path.resolve()

    raise FileNotFoundError(
        "Could not locate the GBTDS 300-event Amax catalogue. "
        "Set ROMAN_RUBIN_EVENT_CATALOG_DIR explicitly. "
        f"Checked: {[str(p) for p in candidates]}"
    )


CATALOG_DIR = _resolve_catalog_dir()


CATALOG_PATHS = {
    population: CATALOG_DIR / filename
    for population, filename
    in POPULATION_FILES.items()
}


def _population_row_count(population):
    return int(
        pq.ParquetFile(
            CATALOG_PATHS[population]
        ).metadata.num_rows
    )


_population_counts = {
    pop: _population_row_count(pop)
    for pop in POPULATIONS
}

if len(set(_population_counts.values())) != 1:
    raise RuntimeError(
        "Population catalogues have different row counts: "
        f"{_population_counts}"
    )


N_PER_POPULATION = next(
    iter(_population_counts.values())
)


@lru_cache(maxsize=1)
def _load_geometry():
    if not GEOMETRY_PATH.is_file():
        raise FileNotFoundError(
            f"Missing geometry file: {GEOMETRY_PATH}"
        )

    geometry = pd.read_csv(
        GEOMETRY_PATH,
        usecols=[
            "field_id",
            "l_deg",
            "b_deg",
        ],
    )

    geometry["field_id"] = (
        geometry["field_id"].astype(str)
    )

    if geometry["field_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate field_id in geometry table."
        )

    if len(geometry) != 510:
        raise RuntimeError(
            f"Expected 510 geometry rows, "
            f"found {len(geometry)}."
        )

    sky = SkyCoord(
        l=geometry["l_deg"].to_numpy(dtype=float)
        * u.deg,
        b=geometry["b_deg"].to_numpy(dtype=float)
        * u.deg,
        frame="galactic",
    ).icrs

    geometry["ra"] = sky.ra.deg
    geometry["dec"] = sky.dec.deg

    return geometry.set_index(
        "field_id",
        verify_integrity=True,
    )


@lru_cache(maxsize=1)
def load_index():
    """
    Compatibility index for existing smoke/validation callers.

    The new production catalogue has one materialized parquet per
    population rather than 510 physical event shards per population.
    """

    rows = []

    for population in POPULATIONS:
        rows.append(
            {
                "population": population,
                "row_start": 0,
                "row_stop": N_PER_POPULATION,
                "catalog_path": str(
                    CATALOG_PATHS[population]
                ),
            }
        )

    return pd.DataFrame(rows)


def read_parquet_row(path, row_number):
    """Read one physical row without loading the full parquet."""

    path = Path(path)
    row_number = int(row_number)

    pf = pq.ParquetFile(path)

    if not 0 <= row_number < pf.metadata.num_rows:
        raise IndexError(
            (str(path), row_number)
        )

    remaining = row_number

    for group in range(pf.num_row_groups):
        n = pf.metadata.row_group(
            group
        ).num_rows

        if remaining < n:
            return (
                pf.read_row_group(group)
                .slice(remaining, 1)
                .to_pylist()[0]
            )

        remaining -= n

    raise IndexError(
        (str(path), row_number)
    )


def load_event(population, global_index):
    """
    Load one event by zero-based index within a population.

    `catalog_event_index` remains the unique global catalogue identity.
    """

    if population not in POPULATIONS:
        raise ValueError(
            f"Unknown population: {population}"
        )

    global_index = int(global_index)

    if not 0 <= global_index < N_PER_POPULATION:
        raise IndexError(global_index)

    event = read_parquet_row(
        CATALOG_PATHS[population],
        global_index,
    )

    if event["system_type"] != population:
        raise RuntimeError(
            "Population mismatch: "
            f"requested={population!r}, "
            f"row={event['system_type']!r}"
        )

    missing = [
        key
        for key in REQUIRED
        if key not in event
    ]

    if missing:
        raise KeyError(
            f"Missing event columns: {missing}"
        )

    if population == "Planets_systems":
        for key in (
            "s",
            "q",
            "alpha",
            "rho",
        ):
            if key not in event:
                raise KeyError(key)

    if (
        population in ("FFP", "BH")
        and "rho" not in event
    ):
        raise KeyError("rho")

    for key in (
        "t0",
        "u0",
        "tE",
        "piEN",
        "piEE",
        "W149",
        "u",
        "g",
        "r",
        "i",
        "z",
        "Y",
    ):
        if not np.isfinite(
            float(event[key])
        ):
            raise RuntimeError(
                f"Non-finite {key} "
                f"for {population} "
                f"index {global_index}"
            )

    if float(event["tE"]) <= 0:
        raise RuntimeError(
            "Non-positive tE."
        )

    field_id = str(
        event["field_id"]
    )

    geometry = _load_geometry()

    if field_id not in geometry.index:
        raise KeyError(
            f"Unknown field_id: {field_id}"
        )

    g = geometry.loc[field_id]

    # Physical direction of the event/cell.
    event["l_deg"] = float(g["l_deg"])
    event["b_deg"] = float(g["b_deg"])
    event["ra"] = float(g["ra"])
    event["dec"] = float(g["dec"])

    # Exact same direction is used for the OpSim query.
    event["maf_ra"] = event["ra"]
    event["maf_dec"] = event["dec"]

    # Collision-free simulation identity of this assembled catalogue.
    event["simulation_seed"] = int(
        event["catalog_event_index"]
    )

    # Preserve event_seed exactly as materialized in the physical catalog.

    return event


# ============================================================
# Config-driven materialized catalogue API
# ============================================================

def load_event_from_catalog(
    catalog_path,
    catalog_event_index,
    expected_population=None,
):
    """
    Load one event from a materialized unified event catalogue.

    The catalogue is expected to satisfy:

        physical row number == catalog_event_index

    which is the invariant established by
    assemble_amax_event_catalog.py.

    No catalogue location or sky geometry is inferred here.
    ra/dec are read directly from the materialized catalogue.
    """

    catalog_path = Path(
        catalog_path
    ).expanduser().resolve()

    if not catalog_path.is_file():
        raise FileNotFoundError(
            f"Missing event catalogue: {catalog_path}"
        )

    catalog_event_index = int(
        catalog_event_index
    )

    event = read_parquet_row(
        catalog_path,
        catalog_event_index,
    )

    actual_index = int(
        event["catalog_event_index"]
    )

    if actual_index != catalog_event_index:
        raise RuntimeError(
            "Catalogue physical-row/index invariant failed: "
            f"requested row={catalog_event_index}, "
            f"row catalog_event_index={actual_index}"
        )

    population = str(
        event["system_type"]
    )

    if (
        expected_population is not None
        and population != str(expected_population)
    ):
        raise RuntimeError(
            "Population mismatch: "
            f"expected={expected_population!r}, "
            f"catalog={population!r}"
        )

    for key in (
        "ra",
        "dec",
        "l_deg",
        "b_deg",
    ):
        if key not in event:
            raise KeyError(
                f"Materialized catalogue is missing {key!r}"
            )

        if not np.isfinite(
            float(event[key])
        ):
            raise RuntimeError(
                f"Non-finite {key} for "
                f"catalog_event_index={catalog_event_index}"
            )

    # One collision-free identity across the complete assembled catalogue.
    event["simulation_seed"] = catalog_event_index

    # The physical Event direction and OpSim query direction are identical.
    event["maf_ra"] = float(event["ra"])
    event["maf_dec"] = float(event["dec"])

    return event
