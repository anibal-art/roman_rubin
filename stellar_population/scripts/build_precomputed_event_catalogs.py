#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repository root:
#   roman_rubin/
#     ulens_params.py
#     stellar_population/
#       scripts/
#         build_precomputed_event_catalogs.py
REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ulens_params import (
    event_param,
    M_JUP_TO_M_SUN,
    PLANET_MASS_MIN_MJUP,
    PLANET_MASS_MAX_MJUP,
    STELLAR_MASS_MIN_MSUN,
    STELLAR_MASS_MAX_MSUN,
    PLANET_HOST_MASS_MIN_MSUN,
    PLANET_HOST_MASS_MAX_MSUN,
    PLANET_Q_MIN,
    PLANET_Q_MAX,
    PLANET_S_MIN,
    PLANET_S_MAX,
)


# ============================================================
# Configuration
# ============================================================

MAG_COLUMNS = [
    "umag",
    "gmag",
    "rmag",
    "imag",
    "zmag",
    "ymag",
    "F062mag",
    "F087mag",
    "F106mag",
    "F129mag",
    "F158mag",
    "F184mag",
    "F146mag",
    "F213mag",
]

# GENULENS source component -> TRILEGAL component
#
# GENULENS:
#   0..6 thin disk
#   7    thick disk
#   8    bar/bulge
#   9    NSD
#   10   stellar halo
#
# TRILEGAL:
#   1 thin disk
#   2 thick disk
#   3 halo
#   4 bulge
#
GENULENS_TO_TRILEGAL = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 2,
    8: 4,
    10: 3,
}


SYSTEMS = {
    "ffp": "FFP",
    "bh": "BH",
    "binary_lens": "Planets_systems",
}


# ============================================================
# Utilities
# ============================================================

def stable_seed(base_seed: int, *labels) -> int:
    payload = "|".join(
        [str(base_seed)] + [str(x) for x in labels]
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="little",
        signed=False,
    ) % (2**31 - 1)


def distance_from_mu0(mu0):
    return 10.0 ** ((float(mu0) + 5.0) / 5.0)


def mu0_from_distance(distance_pc):
    distance_pc = float(distance_pc)

    if not np.isfinite(distance_pc) or distance_pc <= 0:
        raise ValueError(
            f"Invalid distance: {distance_pc}"
        )

    return 5.0 * np.log10(distance_pc) - 5.0


def scalarize(value):
    """
    Convert numpy / astropy-like scalar values into parquet-safe
    Python scalars where possible.
    """
    if hasattr(value, "value"):
        value = value.value

    if isinstance(value, np.generic):
        return value.item()

    return value


# ============================================================
# Matching
# ============================================================

def prepare_trilegal(tri: pd.DataFrame):
    required = {
        "field_id",
        "star_id",
        "Gc",
        "mu0",
        "logL",
        "logTe",
        "F146mag",
    }

    missing = required - set(tri.columns)

    if missing:
        raise RuntimeError(
            "TRILEGAL missing columns: "
            f"{sorted(missing)}"
        )

    tri = tri.copy()

    tri["Gc"] = tri["Gc"].astype(int)

    tri["D_S_trilegal"] = (
        10.0
        **
        (
            (
                tri["mu0"].astype(float)
                + 5.0
            )
            / 5.0
        )
    )

    return tri


def prepare_genulens(gen: pd.DataFrame):
    required = {
        "field_id",
        "genulens_event_id",
        "D_L",
        "D_S",
        "mu_rel",
        "mu_rel_N",
        "mu_rel_E",
        "iS",
    }

    missing = required - set(gen.columns)

    if missing:
        raise RuntimeError(
            "GENULENS missing columns: "
            f"{sorted(missing)}"
        )

    gen = gen.copy()

    gen["iS"] = gen["iS"].astype(int)

    gen["mu0_genulens"] = (
        5.0
        * np.log10(
            gen["D_S"].astype(float)
        )
        - 5.0
    )

    return gen


def build_component_indices(tri):
    """
    For each TRILEGAL Galactic component, store rows sorted by mu0.
    """
    out = {}

    for gc in sorted(tri["Gc"].unique()):

        sub = (
            tri.loc[tri["Gc"] == gc]
            .sort_values("mu0")
            .copy()
        )

        out[int(gc)] = {
            "frame": sub,
            "mu0": sub["mu0"]
            .astype(float)
            .to_numpy(copy=True),
        }

    return out


def match_genulens_to_trilegal(
    tri,
    gen,
    max_dmu=0.05,
    seed=20260930,
):
    components = build_component_indices(tri)

    matched = []
    unmatched = []

    for _, grow in gen.iterrows():

        iS = int(grow["iS"])

        base_audit = {
            "field_id": str(grow["field_id"]),
            "genulens_event_id":
                str(grow["genulens_event_id"]),
            "iS": iS,
            "D_S_genulens":
                float(grow["D_S"]),
            "mu0_genulens":
                float(grow["mu0_genulens"]),
        }

        # ----------------------------------------------------
        # NSD currently has no direct TRILEGAL component
        # ----------------------------------------------------

        if iS == 9:
            unmatched.append(
                {
                    **base_audit,
                    "reason":
                        "unsupported_NSD_source_component",
                    "n_candidates": 0,
                }
            )
            continue

        gc = GENULENS_TO_TRILEGAL.get(iS)

        if gc is None:
            unmatched.append(
                {
                    **base_audit,
                    "reason":
                        "unsupported_GENULENS_source_component",
                    "n_candidates": 0,
                }
            )
            continue

        if gc not in components:
            unmatched.append(
                {
                    **base_audit,
                    "reason":
                        "TRILEGAL_component_absent",
                    "target_Gc": gc,
                    "n_candidates": 0,
                }
            )
            continue

        q = components[gc]
        mu = q["mu0"]
        mu_gen = float(grow["mu0_genulens"])

        lo = np.searchsorted(
            mu,
            mu_gen - max_dmu,
            side="left",
        )

        hi = np.searchsorted(
            mu,
            mu_gen + max_dmu,
            side="right",
        )

        n_candidates = int(hi - lo)

        if n_candidates == 0:
            unmatched.append(
                {
                    **base_audit,
                    "reason":
                        "no_TRILEGAL_match_within_dmu",
                    "target_Gc": gc,
                    "n_candidates": 0,
                }
            )
            continue

        # Deterministic random analogue selection.
        event_seed = stable_seed(
            seed,
            grow["field_id"],
            grow["genulens_event_id"],
            "source_match",
        )

        rng = np.random.default_rng(
            event_seed
        )

        local_idx = int(
            rng.integers(
                lo,
                hi,
            )
        )

        trow = q["frame"].iloc[
            local_idx
        ]

        mu_tri = float(trow["mu0"])
        delta_mu = mu_gen - mu_tri

        if abs(delta_mu) > max_dmu + 1e-12:
            raise RuntimeError(
                "Internal matching error: "
                f"|delta_mu|={abs(delta_mu)} "
                f"> {max_dmu}"
            )

        # ----------------------------------------------------
        # Build one joint source + geometry row
        # ----------------------------------------------------

        row = {}

        # Primary IDs
        row["field_id"] = str(
            grow["field_id"]
        )

        row["genulens_event_id"] = str(
            grow["genulens_event_id"]
        )

        row["star_id"] = str(
            trow["star_id"]
        )

        # Source components
        row["iS"] = iS
        row["Gc"] = int(trow["Gc"])

        # Distances
        row["D_S"] = float(
            grow["D_S"]
        )

        row["D_L"] = float(
            grow["D_L"]
        )

        row["D_S_trilegal"] = float(
            trow["D_S_trilegal"]
        )

        row["mu0_genulens"] = mu_gen
        row["mu0_trilegal"] = mu_tri
        row["delta_mu0"] = delta_mu

        # GENULENS kinematics
        for col in [
            "mu_rel",
            "mu_rel_N",
            "mu_rel_E",
            "mu_Sl",
            "mu_Sb",
            "iL",
            "fREM",
        ]:
            if col in grow.index:
                row[col] = scalarize(
                    grow[col]
                )

        # Keep GENULENS rate/mass outputs only as provenance.
        # They are NOT the physical lens mass/weight used by
        # our characterization experiment.
        provenance_map = {
            "wtj":
                "wtj_genulens",
            "M_L":
                "M_L_genulens",
            "t_E":
                "t_E_genulens",
            "theta_E":
                "theta_E_genulens",
            "pi_E":
                "pi_E_genulens",
            "pi_EN":
                "pi_EN_genulens",
            "pi_EE":
                "pi_EE_genulens",
        }

        for src, dst in provenance_map.items():
            if src in grow.index:
                row[dst] = scalarize(
                    grow[src]
                )

        # Intrinsic source properties
        for col in [
            "logAge",
            "M_H",
            "m_ini",
            "Mass",
            "logL",
            "logTe",
            "logg",
            "Av",
        ]:
            if col in trow.index:
                row[col] = scalarize(
                    trow[col]
                )

        # ----------------------------------------------------
        # Apparent source photometry
        #
        # TRILEGAL magnitudes correspond to its own mu0.
        # Move the analogue to the GENULENS source distance by
        # adding delta_mu0. Extinction is held fixed because
        # the allowed displacement is very small.
        # ----------------------------------------------------

        for col in MAG_COLUMNS:

            if col not in trow.index:
                continue

            raw = float(trow[col])

            row[f"{col}_trilegal"] = raw

            if np.isfinite(raw):
                row[col] = raw + delta_mu
            else:
                row[col] = np.nan

        # Compatibility aliases used by current simulation code.
        if "F146mag" in row:
            row["W149"] = row["F146mag"]

        if "umag" in row:
            row["u"] = row["umag"]

        if "gmag" in row:
            row["g"] = row["gmag"]

        if "rmag" in row:
            row["r"] = row["rmag"]

        if "imag" in row:
            row["i"] = row["imag"]

        if "zmag" in row:
            row["z"] = row["zmag"]

        if "ymag" in row:
            row["Y"] = row["ymag"]

        row["match_n_candidates"] = (
            n_candidates
        )

        row["match_max_dmu"] = float(
            max_dmu
        )

        row["match_seed"] = int(
            event_seed
        )

        matched.append(row)

    return (
        pd.DataFrame(matched),
        pd.DataFrame(unmatched),
    )


# ============================================================
# Precompute physical event parameters
# ============================================================

def build_system_catalog(
    matched,
    system_key,
    base_seed,
    t0_range=None,
):
    system_type = SYSTEMS[
        system_key
    ]

    records = []

    for row_index, row in matched.iterrows():

        event_seed = stable_seed(
            base_seed,
            row["field_id"],
            row["genulens_event_id"],
            system_key,
        )

        # ----------------------------------------------------
        # Build exactly the two Series expected by event_param
        # ----------------------------------------------------

        tri_row = row.copy()

        gen_row = pd.Series(
            {
                "D_L":
                    float(row["D_L"]),
                "D_S":
                    float(row["D_S"]),
                "mu_rel":
                    float(row["mu_rel"]),
                "mu_rel_N":
                    float(row["mu_rel_N"]),
                "mu_rel_E":
                    float(row["mu_rel_E"]),
            }
        )

        kwargs = {}

        if t0_range is not None:
            kwargs["t0_range"] = (
                list(t0_range)
            )

        params = event_param(
            random_seed=event_seed,
            data_TRILEGAL=tri_row,
            data_Genulens=gen_row,
            system_type=system_type,
            **kwargs,
        )

        out = row.to_dict()

        out["event_seed"] = int(
            event_seed
        )

        out["system_key"] = (
            system_key
        )

        out["system_type"] = (
            system_type
        )

        # Current event_param is the single authority for
        # microlensing parameter calculation.
        for key, value in params.items():
            out[key] = scalarize(
                value
            )

        # Useful explicit derived quantities
        out["piE"] = float(
            np.hypot(
                float(out["piEN"]),
                float(out["piEE"]),
            )
        )

        out["mu_rel_vector_norm"] = float(
            np.hypot(
                float(out["mu_rel_N"]),
                float(out["mu_rel_E"]),
            )
        )

        # Direction audit
        mu_norm = (
            out["mu_rel_vector_norm"]
        )

        pi_norm = out["piE"]

        if mu_norm <= 0 or pi_norm <= 0:
            raise RuntimeError(
                "Invalid vector norm while "
                f"building {system_key}: "
                f"row={row_index}"
            )

        cosang = (
            float(out["piEN"])
            * float(out["mu_rel_N"])
            +
            float(out["piEE"])
            * float(out["mu_rel_E"])
        ) / (
            pi_norm
            * mu_norm
        )

        out[
            "piE_mu_rel_direction_cosine"
        ] = float(cosang)

        if not np.isclose(
            cosang,
            1.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                "piE is not aligned with "
                "GENULENS mu_rel: "
                f"cos={cosang}, "
                f"row={row_index}"
            )

        records.append(out)

    return pd.DataFrame(records)



# ============================================================
# Production catalogue validation
# ============================================================

def validate_system_catalog(
    df: pd.DataFrame,
    system_key: str,
):
    """
    Validate the class-specific controlled priors after
    event_param() has materialized the physical event catalogue.

    This is intentionally a design-prior validation, not a
    Galactic event-rate validation.
    """

    generic_required = {
        "D_L",
        "D_S",
        "mu_rel",
        "mu_rel_N",
        "mu_rel_E",
        "tE",
        "thetaE",
        "piEN",
        "piEE",
        "piE",
        "mass_star",
        "mass_planet",
    }

    missing = generic_required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{system_key}: missing required columns: "
            f"{sorted(missing)}"
        )

    if len(df) == 0:
        raise RuntimeError(
            f"{system_key}: empty catalogue."
        )

    if not np.all(
        np.isfinite(df["D_L"])
        & np.isfinite(df["D_S"])
        & np.isfinite(df["mu_rel"])
        & np.isfinite(df["tE"])
        & np.isfinite(df["thetaE"])
    ):
        raise RuntimeError(
            f"{system_key}: non-finite physical parameters."
        )

    if not np.all(
        df["D_L"].to_numpy(float)
        <
        df["D_S"].to_numpy(float)
    ):
        raise RuntimeError(
            f"{system_key}: found D_L >= D_S."
        )

    if system_key == "ffp":

        mp = df[
            "mass_planet"
        ].to_numpy(float)

        ms = df[
            "mass_star"
        ].to_numpy(float)

        if not np.all(ms == 0.0):
            raise RuntimeError(
                "FFP catalogue contains non-zero host masses."
            )

        if not np.all(
            (mp >= PLANET_MASS_MIN_MJUP)
            &
            (mp <= PLANET_MASS_MAX_MJUP)
        ):
            raise RuntimeError(
                "FFP mass outside production prior."
            )

    elif system_key == "bh":

        ms = df[
            "mass_star"
        ].to_numpy(float)

        if not np.all(
            (ms >= STELLAR_MASS_MIN_MSUN)
            &
            (ms <= STELLAR_MASS_MAX_MSUN)
        ):
            raise RuntimeError(
                "BH/compact-lens mass outside "
                "production prior."
            )

    elif system_key == "binary_lens":

        required = {
            "q",
            "s",
            "a_perp_au",
        }

        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(
                "binary_lens: missing new primary/derived "
                f"parameters: {sorted(missing)}"
            )

        ms = df[
            "mass_star"
        ].to_numpy(float)

        mp = df[
            "mass_planet"
        ].to_numpy(float)

        q = df[
            "q"
        ].to_numpy(float)

        sep = df[
            "s"
        ].to_numpy(float)

        aperp = df[
            "a_perp_au"
        ].to_numpy(float)

        if not np.all(
            (ms >= PLANET_HOST_MASS_MIN_MSUN)
            &
            (ms <= PLANET_HOST_MASS_MAX_MSUN)
        ):
            raise RuntimeError(
                "binary_lens: host mass outside "
                "production prior."
            )

        if not np.all(
            (q >= PLANET_Q_MIN)
            &
            (q <= PLANET_Q_MAX)
        ):
            raise RuntimeError(
                "binary_lens: q outside production prior."
            )

        if not np.all(
            (sep >= PLANET_S_MIN)
            &
            (sep <= PLANET_S_MAX)
        ):
            raise RuntimeError(
                "binary_lens: s outside production prior."
            )

        # q = Mcompanion / Mhost
        q_from_mass = (
            mp
            * M_JUP_TO_M_SUN
            / ms
        )

        if not np.allclose(
            q,
            q_from_mass,
            rtol=1e-12,
            atol=0.0,
        ):
            raise RuntimeError(
                "binary_lens: q is inconsistent with "
                "the generated masses."
            )

        # a_perp [AU] = s thetaE[mas] D_L[kpc]
        aperp_from_s = (
            sep
            * df["thetaE"].to_numpy(float)
            * df["D_L"].to_numpy(float)
            / 1000.0
        )

        if not np.allclose(
            aperp,
            aperp_from_s,
            rtol=1e-12,
            atol=0.0,
        ):
            raise RuntimeError(
                "binary_lens: a_perp is inconsistent "
                "with s, thetaE and D_L."
            )

    else:
        raise ValueError(
            f"Unknown system_key={system_key!r}"
        )

    print(
        f"[validation] {system_key}: "
        f"{len(df):,} rows OK"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Match GENULENS geometries to TRILEGAL sources "
            "and precompute FFP, BH and planetary binary-lens "
            "microlensing event catalogues."
        )
    )

    p.add_argument(
        "--trilegal",
        required=True,
    )

    p.add_argument(
        "--genulens",
        required=True,
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    p.add_argument(
        "--max-dmu",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260930,
    )

    p.add_argument(
        "--t0-start",
        type=float,
        default=None,
    )

    p.add_argument(
        "--t0-end",
        type=float,
        default=None,
    )

    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    outdir = Path(
        args.output_dir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        (args.t0_start is None)
        !=
        (args.t0_end is None)
    ):
        raise ValueError(
            "--t0-start and --t0-end must "
            "be supplied together."
        )

    t0_range = None

    if args.t0_start is not None:
        if args.t0_end <= args.t0_start:
            raise ValueError(
                "Require t0_end > t0_start."
            )

        t0_range = (
            float(args.t0_start),
            float(args.t0_end),
        )

    print("=" * 72)
    print("PRECOMPUTED MICROLENSING EVENT CATALOGUES")
    print("=" * 72)

    print("TRILEGAL :", args.trilegal)
    print("GENULENS :", args.genulens)
    print("Output   :", outdir)
    print("max dmu  :", args.max_dmu)
    print("seed     :", args.seed)

    tri = prepare_trilegal(
        pd.read_parquet(
            args.trilegal
        )
    )

    gen = prepare_genulens(
        pd.read_parquet(
            args.genulens
        )
    )

    print()
    print("TRILEGAL rows =", len(tri))
    print("GENULENS rows =", len(gen))

    matched, unmatched = (
        match_genulens_to_trilegal(
            tri=tri,
            gen=gen,
            max_dmu=args.max_dmu,
            seed=args.seed,
        )
    )

    print()
    print("=======================================")
    print("MATCHING")
    print("=======================================")

    print("matched   =", len(matched))
    print("unmatched =", len(unmatched))

    if len(gen):
        print(
            "matched fraction =",
            len(matched) / len(gen),
        )

    if len(unmatched):
        print()
        print("Unmatched reasons:")
        print(
            unmatched["reason"]
            .value_counts(
                dropna=False
            )
        )

    print()
    print("|delta mu0|:")
    print(
        matched["delta_mu0"]
        .abs()
        .describe(
            percentiles=[
                0.5,
                0.9,
                0.95,
                0.99,
                0.999,
            ]
        )
    )

    matched_path = (
        outdir
        / "matched_geometry_sources.parquet"
    )

    unmatched_path = (
        outdir
        / "unmatched_genulens.parquet"
    )

    matched.to_parquet(
        matched_path,
        index=False,
        compression="zstd",
    )

    unmatched.to_parquet(
        unmatched_path,
        index=False,
        compression="zstd",
    )

    # --------------------------------------------------------
    # Precompute all three physical populations using the
    # same matched Galactic geometries.
    # --------------------------------------------------------

    outputs = {}

    for system_key in [
        "ffp",
        "bh",
        "binary_lens",
    ]:

        print()
        print("=" * 72)
        print(
            "BUILDING:",
            system_key,
        )
        print("=" * 72)

        df = build_system_catalog(
            matched=matched,
            system_key=system_key,
            base_seed=args.seed,
            t0_range=t0_range,
        )

        validate_system_catalog(
            df=df,
            system_key=system_key,
        )

        path = (
            outdir
            / f"{system_key}_events.parquet"
        )

        df.to_parquet(
            path,
            index=False,
            compression="zstd",
        )

        outputs[system_key] = str(
            path
        )

        print(
            "rows =",
            len(df),
        )

        print(
            "saved:",
            path,
        )

        print(
            "tE [d]:"
        )
        print(
            df["tE"].describe(
                percentiles=[
                    0.5,
                    0.9,
                    0.99,
                ]
            )
        )

        print(
            "piE:"
        )
        print(
            df["piE"].describe(
                percentiles=[
                    0.5,
                    0.9,
                    0.99,
                ]
            )
        )

        print(
            "direction cosine min/max =",
            df[
                "piE_mu_rel_direction_cosine"
            ].min(),
            df[
                "piE_mu_rel_direction_cosine"
            ].max(),
        )

    metadata = {
        "trilegal": str(
            args.trilegal
        ),
        "genulens": str(
            args.genulens
        ),
        "seed": int(
            args.seed
        ),
        "max_delta_mu0_mag": float(
            args.max_dmu
        ),
        "n_trilegal": int(
            len(tri)
        ),
        "n_genulens": int(
            len(gen)
        ),
        "n_matched": int(
            len(matched)
        ),
        "n_unmatched": int(
            len(unmatched)
        ),
        "source_authority":
            "TRILEGAL",
        "geometry_kinematic_authority":
            "GENULENS",
        "mass_authority":
            "class-specific priors in ulens_params.event_param",
        "genulens_weight_usage":
            "provenance_only",
        "genulens_mass_usage":
            "provenance_only",
        "parallax_direction":
            "parallel_to_GENULENS_mu_rel_vector",
        "source_match":
            "same Galactic component and |delta_mu0| <= max_delta_mu0",
        "source_photometry_distance_correction":
            "m_final = m_TRILEGAL + delta_mu0",
        "nsd_policy":
            "GENULENS iS=9 saved as unmatched",

        "catalog_interpretation":
            "controlled_characterization_parameter_scan_not_event_rate_prediction",

        "event_parameterization": {

            "ffp": {
                "system_type":
                    "FFP",
                "primary_parameters": [
                    "mass_planet"
                ],
                "mass_sampling":
                    "log_uniform",
                "mass_range_mjup": [
                    float(PLANET_MASS_MIN_MJUP),
                    float(PLANET_MASS_MAX_MJUP),
                ],
                "host_mass_msun":
                    0.0,
            },

            "bh": {
                "system_type":
                    "BH",
                "primary_parameters": [
                    "mass_star"
                ],
                "mass_sampling":
                    "log_uniform",
                "mass_range_msun": [
                    float(STELLAR_MASS_MIN_MSUN),
                    float(STELLAR_MASS_MAX_MSUN),
                ],
                "interpretation":
                    "controlled compact-lens mass scan; not a BH population mass function",
            },

            "binary_lens": {
                "system_type":
                    "Planets_systems",
                "primary_parameters": [
                    "mass_star",
                    "q",
                    "s",
                ],
                "primary_parameter_sampling": {
                    "mass_star":
                        "log_uniform",
                    "q":
                        "log_uniform",
                    "s":
                        "log_uniform",
                },
                "host_mass_range_msun": [
                    float(PLANET_HOST_MASS_MIN_MSUN),
                    float(PLANET_HOST_MASS_MAX_MSUN),
                ],
                "q_range": [
                    float(PLANET_Q_MIN),
                    float(PLANET_Q_MAX),
                ],
                "s_range": [
                    float(PLANET_S_MIN),
                    float(PLANET_S_MAX),
                ],
                "companion_mass":
                    "derived: mass_planet = q * mass_star",
                "projected_separation":
                    "derived: a_perp_au = s * thetaE_mas * D_L_kpc",
                "semi_major_axis":
                    "not sampled for Planets_systems",
            },
        },

        "outputs":
            outputs,
    }

    (
        outdir
        / "precomputed_event_catalogs.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print("Matched :", matched_path)
    print("Rejected:", unmatched_path)

    for key, path in outputs.items():
        print(
            f"{key:12s}: {path}"
        )


if __name__ == "__main__":
    main()
