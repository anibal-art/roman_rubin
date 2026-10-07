"""
Catalog-level Amax detectability prefilter.

Purpose
-------
Decide whether an already-realized catalog event is worth sending to the
full Roman+Rubin simulation.

This module DOES NOT use cadence, observing seasons, noise realizations,
or the final deviation-from-constant criterion.

The prefilter is intentionally one-sided:

    simulate_amax == False

means:

    neither the baseline nor the estimated maximum-magnification state
    is bright enough to reach the most favorable single-visit faint
    limit of Roman or Rubin.

Such an event can be discarded before generating a light curve.

IMPORTANT
---------
The final detection decision remains the full simulation pipeline.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BANDS = ("W149", "u", "g", "r", "i", "z", "y")

MAG_COLUMN = {
    "W149": "W149",
    "u": "u",
    "g": "g",
    "r": "r",
    "i": "i",
    "z": "z",
    "y": "Y",
}

# Internal flux zero points used by the Roman-Rubin simulation.
ZERO_POINTS = {
    "W149": 27.615,
    "u": 27.03,
    "g": 28.38,
    "r": 28.16,
    "i": 27.85,
    "z": 27.46,
    "y": 26.68,
}


# ============================================================
# Detection limits
# ============================================================

@dataclass(frozen=True)
class CatalogDetectionLimits:
    limits: dict

    def __getitem__(self, band):
        return float(self.limits[band])


def rubin_deepest_m5(opsim_db):
    """
    Return the deepest single-visit fiveSigmaDepth for each Rubin band.

    The production OpSim database stores canonical Rubin bands in:

        observations.band

    We deliberately use MAX(fiveSigmaDepth) for each band because this
    catalog filter is a conservative necessary-condition prefilter:

    if an event cannot reach even the deepest available single visit,
    it cannot become photometrically observable in that band.
    """

    path = Path(
        opsim_db
    ).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    canonical = (
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
    )

    conn = sqlite3.connect(
        f"file:{path}?mode=ro",
        uri=True,
    )

    try:
        rows = conn.execute(
            """
            SELECT band, MAX(fiveSigmaDepth)
            FROM observations
            WHERE fiveSigmaDepth IS NOT NULL
            GROUP BY band
            """
        ).fetchall()

    finally:
        conn.close()

    result = {
        str(band): float(m5)
        for band, m5 in rows
        if str(band) in canonical
    }

    missing = [
        band
        for band in canonical
        if band not in result
    ]

    if missing:
        raise RuntimeError(
            "Faltan bandas Rubin en observations.band: "
            f"{missing}. Encontradas: {sorted(result)}"
        )

    print(
        "[Amax] Rubin OpSim table : observations"
    )
    print(
        "[Amax] Rubin band column: band"
    )

    for band in canonical:
        print(
            f"[Amax] Rubin m5 max {band}: "
            f"{result[band]:.6f}"
        )

    return result

def roman_f146_5sigma_vega():
    """
    Roman F146 single-epoch S/N=5 faint limit.

    Pandeia magnitudes are AB.
    TRILEGAL F146 magnitudes in the event catalog are Vega.
    """

    import json

    from photometry.roman_f146 import (
        RomanF146Noise,
    )

    data_dir = (
        Path(__file__).resolve().parent.parent
        / "photometry"
        / "data"
    )

    grid_path = (
        data_dir
        / "roman_f146_pandeia_2026p1.csv"
    )

    metadata_path = (
        data_dir
        / "roman_f146_pandeia_2026p1.json"
    )

    if not grid_path.exists():
        raise FileNotFoundError(grid_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    # --------------------------------------------------------
    # Read AB-Vega conversion from production metadata
    # --------------------------------------------------------

    metadata = json.loads(
        metadata_path.read_text()
    )

    def find_key(obj, key):
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]

            for value in obj.values():
                found = find_key(
                    value,
                    key,
                )

                if found is not None:
                    return found

        elif isinstance(obj, list):
            for value in obj:
                found = find_key(
                    value,
                    key,
                )

                if found is not None:
                    return found

        return None

    delta = find_key(
        metadata,
        "ab_minus_vega_f146",
    )

    if delta is None:
        raise KeyError(
            "No encontré 'ab_minus_vega_f146' "
            "en roman_f146_pandeia_2026p1.json"
        )

    delta = float(delta)

    # --------------------------------------------------------
    # Restrict evaluation to the faint, unsaturated region.
    # No reason to evaluate the saturated bright end.
    # --------------------------------------------------------

    grid = pd.read_csv(
        grid_path,
        usecols=[
            "mag_ab",
            "snr",
            "full_saturated",
        ],
    )

    faint_grid = grid[
        (
            ~grid["full_saturated"].astype(bool)
        )
        & np.isfinite(grid["snr"])
        & (grid["mag_ab"] >= 23.0)
    ]

    mag_min = float(
        faint_grid["mag_ab"].min()
    )

    mag_max = float(
        faint_grid["mag_ab"].max()
    )

    noise = RomanF146Noise()

    mag_ab = np.linspace(
        mag_min,
        mag_max,
        14001,
    )

    snr = np.asarray(
        noise.snr_ab(
            mag_ab
        ),
        dtype=float,
    )

    finite = (
        np.isfinite(snr)
        & (snr > 0.0)
    )

    if not np.any(finite):
        raise RuntimeError(
            "RomanF146Noise returned no finite S/N "
            "in the faint unsaturated regime."
        )

    good = (
        finite
        & (snr >= 5.0)
    )

    if not np.any(good):
        raise RuntimeError(
            "No Roman F146 point reaches S/N >= 5."
        )

    i1 = int(
        np.flatnonzero(good)[-1]
    )

    if i1 >= len(mag_ab) - 1:
        raise RuntimeError(
            "S/N=5 boundary is outside the Pandeia grid."
        )

    i2 = i1 + 1

    m1 = float(mag_ab[i1])
    m2 = float(mag_ab[i2])

    s1 = float(snr[i1])
    s2 = float(snr[i2])

    if not (
        s1 >= 5.0
        and s2 < 5.0
    ):
        raise RuntimeError(
            "Could not bracket Roman S/N=5 boundary: "
            f"({m1}, {s1}) -> ({m2}, {s2})"
        )

    # log(S/N) is smoother than S/N itself.
    m5_ab = (
        m1
        + (
            np.log(5.0)
            - np.log(s1)
        )
        * (m2 - m1)
        / (
            np.log(s2)
            - np.log(s1)
        )
    )

    m5_vega = (
        float(m5_ab)
        - delta
    )

    print(
        f"[Amax] Roman F146 m5 AB   : "
        f"{m5_ab:.6f}"
    )

    print(
        f"[Amax] F146 AB-Vega       : "
        f"{delta:.6f}"
    )

    print(
        f"[Amax] Roman F146 m5 Vega : "
        f"{m5_vega:.6f}"
    )

    return m5_vega

def build_detection_limits(opsim_db):
    rubin = rubin_deepest_m5(
        opsim_db,
    )

    limits = {
        "W149": roman_f146_5sigma_vega(),
        **rubin,
    }

    return CatalogDetectionLimits(
        limits=limits,
    )


# ============================================================
# Blending
# ============================================================

def _blend_ratio_array(df, band):
    """
    Return g = F_blend / F_source for `band`.

    Single authority for the column convention: `catalog.blending.
    blending_columns`, the only function that writes these columns,
    always produces `fsource_{band}` and `ftotal_{band}` (from which
    g = ftotal/fsource - 1). No other naming convention exists in
    production.
    """
    fs_col = f"fsource_{band}"
    ft_col = f"ftotal_{band}"

    if fs_col not in df or ft_col not in df:
        raise KeyError(
            f"No encontré {fs_col}/{ft_col} para {band}. "
            "El catálogo Amax debe construirse después de incorporar "
            "la realización de blending (catalog.blending.blending_columns)."
        )

    fs = df[fs_col].to_numpy(float)
    ft = df[ft_col].to_numpy(float)

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        return ft / fs - 1.0


def baseline_magnitude_arrays(df):
    """
    Total observed baseline magnitude:

        F_base = F_s (1 + g)

        m_base = m_source - 2.5 log10(1 + g)
    """
    result = {}

    for band in BANDS:
        mag_col = MAG_COLUMN[band]

        if mag_col not in df:
            result[band] = np.full(
                len(df),
                np.nan,
            )
            continue

        ms = df[
            mag_col
        ].to_numpy(float)

        g = _blend_ratio_array(
            df,
            band,
        )

        ok = (
            np.isfinite(ms)
            & np.isfinite(g)
            & (g >= 0.0)
        )

        mbase = np.full(
            len(df),
            np.nan,
        )

        mbase[ok] = (
            ms[ok]
            - 2.5
            * np.log10(
                1.0 + g[ok]
            )
        )

        result[band] = mbase

    return result


def peak_magnitude_arrays(
    df,
    amax,
):
    """
    Total observed magnitude at Amax:

        F_peak = F_s (Amax + g)

        m_peak = m_source - 2.5 log10(Amax + g)
    """
    amax = np.asarray(
        amax,
        dtype=float,
    )

    result = {}

    for band in BANDS:
        mag_col = MAG_COLUMN[band]

        if mag_col not in df:
            result[band] = np.full(
                len(df),
                np.nan,
            )
            continue

        ms = df[
            mag_col
        ].to_numpy(float)

        g = _blend_ratio_array(
            df,
            band,
        )

        ok = (
            np.isfinite(ms)
            & np.isfinite(g)
            & np.isfinite(amax)
            & (g >= 0.0)
            & (amax > 0.0)
        )

        mpeak = np.full(
            len(df),
            np.nan,
        )

        mpeak[ok] = (
            ms[ok]
            - 2.5
            * np.log10(
                amax[ok]
                + g[ok]
            )
        )

        result[band] = mpeak

    return result


# ============================================================
# Amax: PSPL / FSPL
# ============================================================

def pspl_amax(u0):
    u = np.abs(
        np.asarray(
            u0,
            dtype=float,
        )
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        A = (
            u**2 + 2.0
        ) / (
            u
            * np.sqrt(
                u**2 + 4.0
            )
        )

    A[u == 0.0] = np.inf

    return A


def fspl_amax(u0, rho):
    """
    Exact rectilinear FSPL maximum at tau=0.

    Evaluate gamma=0 and gamma=1 and keep the larger value.
    This makes the catalog criterion conservative with respect to
    a linear limb-darkening coefficient in [0,1].
    """
    from pyLIMA.magnification import (
        magnification_FSPL,
    )

    fn = getattr(
        magnification_FSPL,
        "magnification_FSPL_Yoo",
    )

    beta = np.abs(
        np.asarray(
            u0,
            dtype=float,
        )
    )

    rho = np.asarray(
        rho,
        dtype=float,
    )

    tau = np.zeros_like(
        beta,
    )

    A0 = np.asarray(
        fn(
            tau,
            beta,
            rho,
            0.0,
            return_impact_parameter=False,
        ),
        dtype=float,
    )

    A1 = np.asarray(
        fn(
            tau,
            beta,
            rho,
            1.0,
            return_impact_parameter=False,
        ),
        dtype=float,
    )

    return np.maximum(
        A0,
        A1,
    )


# ============================================================
# Amax: USBL
# ============================================================

def _make_usbl_model_and_parameters(
    row,
    times,
):
    """
    Create a temporary pyLIMA USBL object only for magnification
    evaluation. No photometric simulation/noise is generated.
    """
    from pyLIMA import event
    from pyLIMA import telescopes
    from pyLIMA.models import USBL_model

    t_center = float(
        row["t0"]
    )

    u_center = float(
        row["u0"]
    )

    tE = float(
        row["tE"]
    )

    rho = float(
        row["rho"]
    )

    s = float(
        row["s"]
    )

    q = float(
        row["q"]
    )

    alpha = float(
        row["alpha"]
    )

    origin = str(
        row["caustic_origin"]
    )

    ra = float(
        row.get(
            "ra",
            267.8,
        )
    )

    dec = float(
        row.get(
            "dec",
            -30.4,
        )
    )

    times = np.asarray(
        times,
        dtype=float,
    )

    lc = np.column_stack(
        [
            times,
            np.full_like(
                times,
                20.0,
            ),
            np.full_like(
                times,
                0.01,
            ),
        ]
    )

    ev = event.Event()
    ev.name = "catalog_amax_usbl"
    ev.ra = ra
    ev.dec = dec

    tel = telescopes.Telescope(
        name="Amax",
        camera_filter="I",
        lightcurve=lc,
        lightcurve_names=[
            "time",
            "mag",
            "err_mag",
        ],
        lightcurve_units=[
            "JD",
            "mag",
            "mag",
        ],
        location="Earth",
    )

    tel.ld_gamma = 0.0

    ev.telescopes.append(
        tel,
    )

    model = USBL_model.USBLmodel(
        ev,
        parallax=[
            "None",
            0.0,
        ],
        origin=[
            origin,
            [0.0, 0.0],
        ],
        blend_flux_parameter="ftotal",
    )

    values = {
        "t_center": t_center,
        "t0": t_center,
        "u_center": u_center,
        "u0": u_center,
        "tE": tE,
        "rho": rho,
        "separation": s,
        "s": s,
        "mass_ratio": q,
        "q": q,
        "alpha": alpha,
        "fsource_Amax": 1.0,
        "ftotal_Amax": 1.0,
    }

    vector = []

    for name, index in sorted(
        model.model_dictionnary.items(),
        key=lambda x: x[1],
    ):
        if name.startswith(
            "fsource_"
        ):
            vector.append(1.0)

        elif name.startswith(
            "ftotal_"
        ):
            vector.append(1.0)

        elif name.startswith(
            "fblend_"
        ):
            vector.append(0.0)

        elif name in values:
            vector.append(
                values[name]
            )

        else:
            raise KeyError(
                f"No sé construir parámetro pyLIMA {name!r}."
            )

    pyparams = (
        model.compute_pyLIMA_parameters(
            vector
        )
    )

    return model, tel, pyparams


def usbl_amax(row):
    """
    Multiscale Amax estimate for a finite-source binary lens.

    The catalog trajectory is caustic-centered, therefore we combine:

      1. broad search for the global/baseline lens geometry;
      2. medium search around the chosen caustic;
      3. rho-scaled dense search around the caustic;
      4. exact evaluation at t_center;
      5. exact evaluation at pyLIMA's transformed host t0 when available.

    This quantity MUST be validated against the full pipeline before
    turning simulate_amax=False into a hard production rejection.
    """
    t_center = float(
        row["t0"]
    )

    tE = float(
        row["tE"]
    )

    rho = max(
        float(row["rho"]),
        1.0e-8,
    )

    s = max(
        float(row["s"]),
        1.0e-6,
    )

    # Broad enough to include the host-lens peak even when the selected
    # origin is a planetary caustic.
    x_extent = max(
        4.0,
        2.0 + s + 1.0 / s,
    )

    x_broad = np.linspace(
        -x_extent,
        x_extent,
        257,
    )

    x_medium = np.linspace(
        -0.25,
        0.25,
        257,
    )

    # Resolves finite-source-smoothed caustic structure on the rho scale.
    x_local = (
        rho
        * np.linspace(
            -64.0,
            64.0,
            513,
        )
    )

    x = np.unique(
        np.concatenate(
            [
                x_broad,
                x_medium,
                x_local,
                np.array([0.0]),
            ]
        )
    )

    times = (
        t_center
        + tE * x
    )

    model, tel, pyparams = (
        _make_usbl_model_and_parameters(
            row,
            times,
        )
    )

    A = np.asarray(
        model.model_magnification(
            tel,
            pyparams,
        ),
        dtype=float,
    )

    finite = np.isfinite(A)

    if not np.any(finite):
        raise RuntimeError(
            "USBL Amax returned no finite magnifications."
        )

    best = float(
        np.nanmax(A)
    )

    # Refine around the strongest sampled locations.
    order = np.argsort(
        np.nan_to_num(
            A,
            nan=-np.inf,
        )
    )[-8:]

    candidate_x = x[
        order
    ]

    local_half_width = max(
        2.0 * rho,
        1.0e-5,
    )

    x_refine = []

    for xc in candidate_x:
        x_refine.append(
            np.linspace(
                xc - local_half_width,
                xc + local_half_width,
                65,
            )
        )

    x_refine = np.unique(
        np.concatenate(
            x_refine
        )
    )

    times_refine = (
        t_center
        + tE * x_refine
    )

    model2, tel2, pyparams2 = (
        _make_usbl_model_and_parameters(
            row,
            times_refine,
        )
    )

    A2 = np.asarray(
        model2.model_magnification(
            tel2,
            pyparams2,
        ),
        dtype=float,
    )

    best = max(
        best,
        float(
            np.nanmax(A2)
        ),
    )

    return best


# ============================================================
# Catalog annotation
# ============================================================

def annotate_catalog(
    df,
    population,
    limits,
    margin_mag=0.0,
):
    """
    Add catalog-level Amax prefilter columns.

    simulate_amax=True:
        send event to the full simulation.

    simulate_amax=False:
        baseline AND Amax are fainter than every relevant survey limit.
    """
    df = df.copy()

    population = str(
        population
    )

    mbase = baseline_magnitude_arrays(
        df
    )

    baseline_detect = np.zeros(
        len(df),
        dtype=bool,
    )

    for band in BANDS:
        values = mbase[
            band
        ]

        limit = (
            limits[band]
            + float(margin_mag)
        )

        flag = (
            np.isfinite(values)
            & (values <= limit)
        )

        df[
            f"m_base_{band}"
        ] = values

        df[
            f"baseline_detectable_{band}"
        ] = flag

        baseline_detect |= flag

    df[
        "baseline_detectable_any"
    ] = baseline_detect

    Amax = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    method = np.full(
        len(df),
        "",
        dtype=object,
    )

    # --------------------------------------------------------
    # PSPL / FSPL: cheap, calculate for every event
    # --------------------------------------------------------
    if population in {
        "FFP",
        "BH",
    }:
        if (
            "rho" in df
            and np.all(
                np.isfinite(
                    df["rho"].to_numpy(float)
                )
            )
        ):
            Amax[:] = fspl_amax(
                df["u0"].to_numpy(float),
                df["rho"].to_numpy(float),
            )

            method[:] = (
                "FSPL_tau0_gamma_envelope_v1"
            )

        else:
            Amax[:] = pspl_amax(
                df["u0"].to_numpy(float)
            )

            method[:] = (
                "PSPL_analytic_v1"
            )

    # --------------------------------------------------------
    # USBL: expensive. Only needed when baseline itself is
    # invisible; otherwise the final flag is already True.
    # --------------------------------------------------------
    elif population == "Planets_systems":

        required = {
            "t0",
            "u0",
            "tE",
            "rho",
            "s",
            "q",
            "alpha",
            "caustic_origin",
        }

        missing = (
            required
            - set(df.columns)
        )

        if missing:
            raise KeyError(
                "USBL Amax requires catalog columns: "
                f"{sorted(missing)}"
            )

        indices = np.flatnonzero(
            ~baseline_detect
        )

        for counter, i in enumerate(
            indices,
            start=1,
        ):
            Amax[i] = usbl_amax(
                df.iloc[i]
            )

            method[i] = (
                "USBL_multiscale_caustic_v1"
            )

            if (
                counter % 100 == 0
                or counter == len(indices)
            ):
                print(
                    "[Amax USBL]",
                    counter,
                    "/",
                    len(indices),
                )

        method[
            baseline_detect
        ] = (
            "not_needed_baseline_detectable"
        )

    else:
        raise ValueError(
            f"Population not supported: {population}"
        )

    df["Amax_catalog"] = Amax
    df["amax_method"] = method

    mpeak = peak_magnitude_arrays(
        df,
        Amax,
    )

    peak_detect = np.zeros(
        len(df),
        dtype=bool,
    )

    for band in BANDS:
        values = mpeak[
            band
        ]

        limit = (
            limits[band]
            + float(margin_mag)
        )

        flag = (
            np.isfinite(values)
            & (values <= limit)
        )

        df[
            f"m_peak_{band}"
        ] = values

        df[
            f"peak_detectable_{band}"
        ] = flag

        peak_detect |= flag

        df[
            f"limit_mag_{band}"
        ] = float(
            limits[band]
        )

    df[
        "peak_detectable_any"
    ] = peak_detect

    simulate = (
        baseline_detect
        | peak_detect
    )

    df[
        "simulate_amax"
    ] = simulate

    reason = np.full(
        len(df),
        "reject_baseline_and_peak_too_faint",
        dtype=object,
    )

    reason[
        baseline_detect
    ] = (
        "keep_baseline_detectable"
    )

    reason[
        (~baseline_detect)
        & peak_detect
    ] = (
        "keep_peak_detectable"
    )

    df[
        "amax_prefilter_reason"
    ] = reason

    df[
        "amax_margin_mag"
    ] = float(
        margin_mag
    )

    return df
