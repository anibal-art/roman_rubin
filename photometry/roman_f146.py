from __future__ import annotations

import json
from pathlib import Path

import warnings

import numpy as np
import pandas as pd


DEFAULT_GRID = (
    Path(__file__).parent
    / "data"
    / "roman_f146_pandeia_2026p1.csv"
)

DEFAULT_METADATA = (
    Path(__file__).parent
    / "data"
    / "roman_f146_pandeia_2026p1.json"
)


FRAC_TO_MAG = 2.5 / np.log(10.0)


def _bool_series(series):
    """
    Robust conversion of CSV boolean columns.
    """
    if series.dtype == bool:
        return series

    out = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
            }
        )
    )

    if out.isna().any():
        bad = series[out.isna()].unique()

        raise ValueError(
            "Could not parse boolean values: "
            f"{bad}"
        )

    return out.astype(bool)


def _restore_scalar(value, scalar):
    arr = np.asarray(value)

    if scalar:
        return arr.reshape(-1)[0].item()

    return arr



def _nanmedian_18sca_axis0_exact(
    values,
):
    """
    Exact fast nanmedian for the Roman median-SCA 18 x N stacks.

    The Roman median-detector approximation combines the 18 real
    SCAs independently at every epoch. NumPy's generic nanmedian()
    has substantial overhead for this small fixed detector axis.

    For the validated 18 x N floating-point case, sort the 18
    detector values explicitly and select the middle finite value(s).
    Outside that exact case, fall back to NumPy's implementation.

    The all-NaN RuntimeWarning is preserved.
    """

    arr = np.asarray(
        values
    )

    if (
        arr.ndim != 2
        or arr.shape[0] != 18
        or arr.dtype.kind != "f"
    ):
        return np.nanmedian(
            arr,
            axis=0,
        )

    # Put the fixed 18-SCA detector axis last, then sort it.
    # NaNs sort to the end, so the first `counts` entries are
    # exactly the finite values used by nanmedian.
    ordered = np.sort(
        arr.T,
        axis=1,
    )

    counts = np.sum(
        ~np.isnan(
            ordered
        ),
        axis=1,
    )

    result = np.full(
        arr.shape[1],
        np.nan,
        dtype=arr.dtype,
    )

    columns = np.arange(
        arr.shape[1]
    )

    nonzero = (
        counts > 0
    )

    odd = (
        nonzero
        & ((counts % 2) == 1)
    )

    even = (
        nonzero
        & ((counts % 2) == 0)
    )

    middle = (
        counts // 2
    )

    if np.any(
        odd
    ):
        result[
            odd
        ] = ordered[
            columns[odd],
            middle[odd],
        ]

    if np.any(
        even
    ):
        lower = ordered[
            columns[even],
            middle[even] - 1,
        ]

        upper = ordered[
            columns[even],
            middle[even],
        ]

        result[
            even
        ] = (
            lower
            + upper
        ) / 2.0

    n_all_nan = int(
        np.count_nonzero(
            counts == 0
        )
    )

    for _ in range(
        n_all_nan
    ):
        warnings.warn(
            "All-NaN slice encountered",
            RuntimeWarning,
            stacklevel=2,
        )

    return result


class RomanF146Noise:
    """
    Roman GBTDS F146 single-epoch photometric-noise model.

    Base model
    ----------
    Pandeia Engine/RefData/PSFs R2026.1 with the local
    IM_66_6_V2 Revision-H GBTDS read-pattern override.

    The Pandeia grid is interpolated separately within each
    instrumental regime:

        full saturated
        partial saturated, N_resultants = 3
        partial saturated, N_resultants = 4
        partial saturated, N_resultants = 5
        unsaturated

    No interpolation is ever performed across a change in
    saturation state or sat_nresultants.

    Parameters
    ----------
    detector : str
        wfi01 ... wfi18, or "median".

        "median" is intended for quick tests. Production should
        use the actual SCA whenever geometry provides it.

    background_level : str
        Usually "medium".

    systematic_floor_frac : float
        Optional fractional flux floor, added in quadrature.
        Default is zero: no empirical floor is assumed.
    """

    def __init__(
        self,
        grid_path=DEFAULT_GRID,
        metadata_path=DEFAULT_METADATA,
        detector="median",
        background_level="medium",
        systematic_floor_frac=0.0,
    ):

        self.grid_path = Path(grid_path)
        self.metadata_path = Path(
            metadata_path
        )

        self.detector = str(detector)
        self.background_level = str(
            background_level
        )

        self.systematic_floor_frac = float(
            systematic_floor_frac
        )

        if self.systematic_floor_frac < 0:
            raise ValueError(
                "systematic_floor_frac must be >= 0."
            )

        self.grid = pd.read_csv(
            self.grid_path
        )

        with open(
            self.metadata_path
        ) as f:
            self.metadata = json.load(f)

        self.ab_minus_vega = (
            self.metadata.get(
                "ab_minus_vega_f146"
            )
        )

        self._prepare()


    # ========================================================
    # Grid preparation
    # ========================================================

    @staticmethod
    def _segment_key(
        full,
        partial,
        sat_nresultants,
    ):

        if full:
            return "full"

        if partial:

            if pd.isna(
                sat_nresultants
            ):
                raise ValueError(
                    "Partial-saturation row without "
                    "sat_nresultants."
                )

            n = int(
                round(
                    float(
                        sat_nresultants
                    )
                )
            )

            return f"partial_{n}"

        return "unsaturated"


    def _prepare(self):

        required = {
            "detector",
            "background_level",
            "mag_ab",
            "snr",
            "partial_saturated",
            "full_saturated",
            "sat_nresultants",
        }

        missing = (
            required
            - set(self.grid.columns)
        )

        if missing:
            raise ValueError(
                "Missing columns in Roman noise grid: "
                f"{sorted(missing)}"
            )

        g = self.grid[
            self.grid[
                "background_level"
            ]
            == self.background_level
        ].copy()

        if len(g) == 0:
            raise ValueError(
                "No Pandeia grid entries for "
                f"background_level="
                f"{self.background_level}"
            )

        g[
            "partial_saturated"
        ] = _bool_series(
            g[
                "partial_saturated"
            ]
        )

        g[
            "full_saturated"
        ] = _bool_series(
            g[
                "full_saturated"
            ]
        )

        detectors = sorted(
            g[
                "detector"
            ].unique()
        )

        if len(detectors) != 18:
            raise RuntimeError(
                "Expected 18 WFI detectors, "
                f"found {len(detectors)}."
            )

        self.detectors = detectors

        if (
            self.detector
            != "median"
            and self.detector
            not in detectors
        ):
            raise ValueError(
                "detector must be 'median' "
                "or one of "
                f"{detectors}"
            )

        self._tables = {}

        for detector in detectors:

            d = (
                g[
                    g["detector"]
                    == detector
                ]
                .copy()
                .sort_values(
                    "mag_ab"
                )
                .reset_index(
                    drop=True
                )
            )

            if d[
                "mag_ab"
            ].duplicated().any():
                raise RuntimeError(
                    "Duplicate magnitudes for "
                    f"{detector}"
                )

            d[
                "segment"
            ] = [
                self._segment_key(
                    full=full,
                    partial=partial,
                    sat_nresultants=nres,
                )
                for full, partial, nres
                in zip(
                    d[
                        "full_saturated"
                    ],
                    d[
                        "partial_saturated"
                    ],
                    d[
                        "sat_nresultants"
                    ],
                )
            ]

            mags = d[
                "mag_ab"
            ].to_numpy(
                dtype=float
            )

            # Midpoints define nearest-grid-state boundaries.
            boundaries = (
                0.5
                * (
                    mags[:-1]
                    + mags[1:]
                )
            )

            segments = {}

            for key, s in d.groupby(
                "segment",
                sort=False,
            ):

                s = (
                    s
                    .sort_values(
                        "mag_ab"
                    )
                    .copy()
                )

                good = (
                    np.isfinite(
                        s["snr"]
                    )
                    &
                    (
                        s["snr"]
                        > 0
                    )
                )

                s_valid = (
                    s.loc[
                        good
                    ]
                    .copy()
                )

                if (
                    key != "full"
                    and len(
                        s_valid
                    ) == 0
                ):
                    raise RuntimeError(
                        f"{detector}: "
                        f"segment {key} "
                        "contains no valid S/N."
                    )

                segments[
                    key
                ] = s_valid

            self._tables[
                detector
            ] = {
                "table":
                    d,

                "magnitudes":
                    mags,

                "boundaries":
                    boundaries,

                "segments":
                    segments,

                "mag_min":
                    float(
                        mags.min()
                    ),

                "mag_max":
                    float(
                        mags.max()
                    ),
            }

        self.mag_min = min(
            x[
                "mag_min"
            ]
            for x in (
                self._tables.values()
            )
        )

        self.mag_max = max(
            x[
                "mag_max"
            ]
            for x in (
                self._tables.values()
            )
        )


    # ========================================================
    # Magnitude conversions
    # ========================================================

    def vega_to_ab(
        self,
        mag_vega,
    ):

        if self.ab_minus_vega is None:
            raise RuntimeError(
                "F146 AB-Vega conversion "
                "is absent from metadata."
            )

        return (
            np.asarray(
                mag_vega,
                dtype=float,
            )
            +
            float(
                self.ab_minus_vega
            )
        )


    def ab_to_vega(
        self,
        mag_ab,
    ):

        if self.ab_minus_vega is None:
            raise RuntimeError(
                "F146 AB-Vega conversion "
                "is absent from metadata."
            )

        return (
            np.asarray(
                mag_ab,
                dtype=float,
            )
            -
            float(
                self.ab_minus_vega
            )
        )


    # ========================================================
    # Single-detector evaluation
    # ========================================================

    def _evaluate_detector_ab(
        self,
        mag_ab,
        detector,
    ):
        """
        Evaluate one Roman detector using the precomputed Pandeia grid.

        Static detector-table information is converted to NumPy arrays
        once per process and then reused. Only the event-dependent
        magnitude lookup and piecewise interpolation are repeated.

        This preserves the original regime classification,
        interpolation rules, saturation policy, and returned values.
        """

        scalar = (
            np.asarray(
                mag_ab
            ).ndim
            == 0
        )

        mag = np.atleast_1d(
            np.asarray(
                mag_ab,
                dtype=float,
            )
        )

        data = self._tables[
            detector
        ]

        # ====================================================
        # Static detector preprocessing
        #
        # The Pandeia grid and segment definitions never change
        # between events. Avoid rebuilding pandas objects,
        # converting strings, and extracting interpolation arrays
        # for every simulated light curve.
        # ====================================================

        cache = data.get(
            "_roman_fast_eval_cache"
        )

        if cache is None:

            table = data[
                "table"
            ]

            row_segment = (
                table[
                    "segment"
                ]
                .astype(str)
                .to_numpy()
            )

            row_nres = (
                table[
                    "sat_nresultants"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            row_full = (
                row_segment
                == "full"
            )

            row_partial = (
                np.char.startswith(
                    row_segment.astype(str),
                    "partial_",
                )
            )

            row_unsaturated = (
                row_segment
                == "unsaturated"
            )

            segment_arrays = {}

            for key in sorted(
                data[
                    "segments"
                ].keys()
            ):

                segment_table = (
                    data[
                        "segments"
                    ][key]
                )

                segment_arrays[
                    str(key)
                ] = (
                    segment_table[
                        "mag_ab"
                    ].to_numpy(
                        dtype=float
                    ),
                    segment_table[
                        "snr"
                    ].to_numpy(
                        dtype=float
                    ),
                )

            cache = {
                "row_segment":
                    row_segment,

                "row_nres":
                    row_nres,

                "row_full":
                    row_full,

                "row_partial":
                    row_partial,

                "row_unsaturated":
                    row_unsaturated,

                "segment_arrays":
                    segment_arrays,
            }

            data[
                "_roman_fast_eval_cache"
            ] = cache

        # ====================================================
        # Event-dependent evaluation
        # ====================================================

        faint = (
            mag
            > data[
                "mag_max"
            ]
        )

        if np.any(faint):

            raise ValueError(
                "Magnitude fainter than the "
                "precomputed Pandeia grid "
                f"for {detector}. "
                f"Maximum = "
                f"{data['mag_max']:.3f} AB; "
                f"requested maximum = "
                f"{mag.max():.3f} AB."
            )

        # Anything brighter than the grid minimum retains the
        # exact historical full-saturation policy.
        bright = (
            mag
            < data[
                "mag_min"
            ]
        )

        clipped = np.clip(
            mag,
            data[
                "mag_min"
            ],
            data[
                "mag_max"
            ],
        )

        nearest_index = (
            np.searchsorted(
                data[
                    "boundaries"
                ],
                clipped,
                side="right",
            )
        )

        segment = (
            cache[
                "row_segment"
            ][
                nearest_index
            ]
            .copy()
        )

        full = (
            cache[
                "row_full"
            ][
                nearest_index
            ]
            .copy()
        )

        partial = (
            cache[
                "row_partial"
            ][
                nearest_index
            ]
            .copy()
        )

        unsaturated = (
            cache[
                "row_unsaturated"
            ][
                nearest_index
            ]
            .copy()
        )

        segment[
            bright
        ] = "full"

        full[
            bright
        ] = True

        partial[
            bright
        ] = False

        unsaturated[
            bright
        ] = False

        valid = (
            ~full
        )

        nres = np.full(
            len(mag),
            np.nan,
            dtype=float,
        )

        if np.any(
            valid
        ):

            nres[
                valid
            ] = (
                cache[
                    "row_nres"
                ][
                    nearest_index[
                        valid
                    ]
                ]
            )

        snr = np.zeros(
            len(mag),
            dtype=float,
        )

        # ====================================================
        # Piecewise interpolation
        #
        # Preserve the original rule that each instrumental
        # segment is interpolated independently.
        # ====================================================

        for key, (
            x,
            y,
        ) in (
            cache[
                "segment_arrays"
            ].items()
        ):

            mask = (
                valid
                &
                (
                    segment
                    == key
                )
            )

            if not np.any(
                mask
            ):
                continue

            if len(
                x
            ) == 1:

                snr[
                    mask
                ] = y[
                    0
                ]

            else:

                snr[
                    mask
                ] = (
                    10.0
                    **
                    np.interp(
                        mag[
                            mask
                        ],
                        x,
                        np.log10(
                            y
                        ),
                    )
                )

        result = {
            "mag_ab":
                mag,

            "snr":
                snr,

            "valid":
                valid,

            "full_saturated":
                full,

            "partial_saturated":
                partial,

            "unsaturated":
                unsaturated,

            "sat_nresultants":
                nres,

            "segment":
                segment,
        }

        return {
            key:
                _restore_scalar(
                    value,
                    scalar,
                )
            for key, value
            in result.items()
        }

    def evaluate_ab(
        self,
        mag_ab,
    ):

        if self.detector != "median":

            result = (
                self._evaluate_detector_ab(
                    mag_ab,
                    self.detector,
                )
            )

            scalar = (
                np.asarray(
                    mag_ab
                ).ndim
                == 0
            )

            snr = np.atleast_1d(
                np.asarray(
                    result[
                        "snr"
                    ],
                    dtype=float,
                )
            )

            valid = np.atleast_1d(
                np.asarray(
                    result[
                        "valid"
                    ],
                    dtype=bool,
                )
            )

            frac = np.full(
                len(snr),
                np.nan,
                dtype=float,
            )

            good = (
                valid
                & (snr > 0)
            )

            frac[
                good
            ] = np.sqrt(
                (
                    1.0
                    / snr[
                        good
                    ]
                )**2
                +
                self.systematic_floor_frac**2
            )

            sigma_mag = (
                FRAC_TO_MAG
                * frac
            )

            result[
                "fractional_flux_error"
            ] = _restore_scalar(
                frac,
                scalar,
            )

            result[
                "sigma_mag"
            ] = _restore_scalar(
                sigma_mag,
                scalar,
            )

            return result


        # ====================================================
        # Median-SCA approximation.
        #
        # Evaluate every real detector first. We never average
        # the raw grid before classifying its instrumental
        # regime.
        # ====================================================

        scalar = (
            np.asarray(
                mag_ab
            ).ndim
            == 0
        )

        mag = np.atleast_1d(
            np.asarray(
                mag_ab,
                dtype=float,
            )
        )

        evaluations = [
            self._evaluate_detector_ab(
                mag,
                detector,
            )
            for detector
            in self.detectors
        ]

        snr_stack = np.vstack(
            [
                np.asarray(
                    x["snr"],
                    dtype=float,
                )
                for x
                in evaluations
            ]
        )

        valid_stack = np.vstack(
            [
                np.asarray(
                    x["valid"],
                    dtype=bool,
                )
                for x
                in evaluations
            ]
        )

        full_stack = np.vstack(
            [
                np.asarray(
                    x[
                        "full_saturated"
                    ],
                    dtype=bool,
                )
                for x
                in evaluations
            ]
        )

        partial_stack = np.vstack(
            [
                np.asarray(
                    x[
                        "partial_saturated"
                    ],
                    dtype=bool,
                )
                for x
                in evaluations
            ]
        )

        nres_stack = np.vstack(
            [
                np.asarray(
                    x[
                        "sat_nresultants"
                    ],
                    dtype=float,
                )
                for x
                in evaluations
            ]
        )

        valid_fraction = (
            valid_stack.mean(
                axis=0
            )
        )

        full_fraction = (
            full_stack.mean(
                axis=0
            )
        )

        partial_fraction = (
            partial_stack.mean(
                axis=0
            )
        )

        # A median detector is only a diagnostic approximation.
        # Require at least half the SCAs to provide a valid
        # measurement.
        valid = (
            valid_fraction
            >= 0.5
        )

        masked_snr = np.where(
            valid_stack,
            snr_stack,
            np.nan,
        )

        with np.errstate(
            all="ignore"
        ):
            snr = _nanmedian_18sca_axis0_exact(
                masked_snr
            )

        snr[
            ~valid
        ] = 0.0

        with np.errstate(
            all="ignore"
        ):
            nres = _nanmedian_18sca_axis0_exact(
                np.where(
                    valid_stack,
                    nres_stack,
                    np.nan,
                )
            )

        full = (
            full_fraction
            > 0.5
        )

        partial = (
            valid
            & ~full
            & (
                partial_fraction
                > 0.5
            )
        )

        unsaturated = (
            valid
            & ~partial
            & ~full
        )

        frac = np.full(
            len(mag),
            np.nan,
            dtype=float,
        )

        good = (
            valid
            & (snr > 0)
        )

        frac[
            good
        ] = np.sqrt(
            (
                1.0
                / snr[
                    good
                ]
            )**2
            +
            self.systematic_floor_frac**2
        )

        result = {
            "mag_ab":
                mag,

            "snr":
                snr,

            "valid":
                valid,

            "full_saturated":
                full,

            "partial_saturated":
                partial,

            "unsaturated":
                unsaturated,

            "sat_nresultants":
                nres,

            "fractional_flux_error":
                frac,

            "sigma_mag":
                FRAC_TO_MAG
                * frac,

            "valid_detector_fraction":
                valid_fraction,

            "full_saturated_fraction":
                full_fraction,

            "partial_saturated_fraction":
                partial_fraction,
        }

        return {
            key:
                _restore_scalar(
                    value,
                    scalar,
                )
            for key, value
            in result.items()
        }


    # ========================================================
    # Convenience wrappers
    # ========================================================

    def snr_ab(
        self,
        mag_ab,
    ):
        return self.evaluate_ab(
            mag_ab
        )["snr"]


    def fractional_flux_error_ab(
        self,
        mag_ab,
    ):
        return self.evaluate_ab(
            mag_ab
        )[
            "fractional_flux_error"
        ]


    def sigma_mag_ab(
        self,
        mag_ab,
    ):
        return self.evaluate_ab(
            mag_ab
        )[
            "sigma_mag"
        ]


    # ========================================================
    # Microlensing flux
    # ========================================================

    def microlensing_flux_error(
        self,
        source_mag,
        magnification,
        blend_ratio=0.0,
        mag_system="vega",
    ):
        """
        Evaluate Roman F146 noise for a microlensing light curve.

        Flux units are normalized such that Fs = 1:

            F(t) / Fs = A(t) + Fb/Fs

        Parameters
        ----------
        source_mag : float or array
            Unmagnified source magnitude.

        magnification : array
            A(t).

        blend_ratio : float or array
            Fb / Fs.

        mag_system : {"vega", "ab"}
            Magnitude system of source_mag.

        Returns
        -------
        dict
            Includes total instantaneous magnitude, S/N,
            flux uncertainty, saturation flags and validity.
        """

        source_mag = np.asarray(
            source_mag,
            dtype=float,
        )

        A = np.asarray(
            magnification,
            dtype=float,
        )

        fb_fs = np.asarray(
            blend_ratio,
            dtype=float,
        )

        if (
            mag_system.lower()
            == "vega"
        ):

            source_mag_ab = (
                self.vega_to_ab(
                    source_mag
                )
            )

        elif (
            mag_system.lower()
            == "ab"
        ):

            source_mag_ab = (
                source_mag
            )

        else:
            raise ValueError(
                "mag_system must be "
                "'vega' or 'ab'."
            )

        total_flux_fs = (
            A
            + fb_fs
        )

        if np.any(
            total_flux_fs
            <= 0
        ):
            raise ValueError(
                "A + Fb/Fs must be > 0."
            )

        total_mag_ab = (
            source_mag_ab
            -
            2.5
            * np.log10(
                total_flux_fs
            )
        )

        noise = self.evaluate_ab(
            total_mag_ab
        )

        frac = np.asarray(
            noise[
                "fractional_flux_error"
            ],
            dtype=float,
        )

        sigma_flux_fs = (
            total_flux_fs
            * frac
        )

        return {
            "total_flux_fs":
                total_flux_fs,

            "total_mag_ab":
                total_mag_ab,

            "snr":
                noise[
                    "snr"
                ],

            "sigma_flux_fs":
                sigma_flux_fs,

            "fractional_flux_error":
                noise[
                    "fractional_flux_error"
                ],

            "sigma_mag":
                noise[
                    "sigma_mag"
                ],

            "valid":
                noise[
                    "valid"
                ],

            "full_saturated":
                noise[
                    "full_saturated"
                ],

            "partial_saturated":
                noise[
                    "partial_saturated"
                ],

            "unsaturated":
                noise[
                    "unsaturated"
                ],

            "sat_nresultants":
                noise[
                    "sat_nresultants"
                ],
        }
