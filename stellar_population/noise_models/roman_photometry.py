from __future__ import annotations

import os

import astropy.units as u
import numpy as np

from .roman_f146 import RomanF146Noise


_ROMAN_NOISE_CACHE = {}


def _env_float(name, default):
    return float(
        os.environ.get(
            name,
            str(default),
        )
    )


def get_roman_f146_noise_model(
    detector=None,
    background_level=None,
    systematic_floor_frac=None,
):
    """
    Cached Roman F146 Pandeia-grid noise model.

    Defaults can be overridden through:

        ROMAN_F146_DETECTOR
        ROMAN_F146_BACKGROUND_LEVEL
        ROMAN_F146_SYSTEMATIC_FLOOR_FRAC
    """

    if detector is None:
        detector = os.environ.get(
            "ROMAN_F146_DETECTOR",
            "median",
        )

    if background_level is None:
        background_level = os.environ.get(
            "ROMAN_F146_BACKGROUND_LEVEL",
            "medium",
        )

    if systematic_floor_frac is None:
        systematic_floor_frac = _env_float(
            "ROMAN_F146_SYSTEMATIC_FLOOR_FRAC",
            0.0,
        )

    key = (
        str(detector),
        str(background_level),
        float(systematic_floor_frac),
    )

    if key not in _ROMAN_NOISE_CACHE:
        _ROMAN_NOISE_CACHE[key] = RomanF146Noise(
            detector=detector,
            background_level=background_level,
            systematic_floor_frac=systematic_floor_frac,
        )

    return _ROMAN_NOISE_CACHE[key]


def _plain(x, dtype=float):
    if hasattr(x, "value"):
        x = x.value

    return np.asarray(
        x,
        dtype=dtype,
    )


def _replace_column_preserve_unit(
    table,
    name,
    values,
):
    """
    Replace values preserving the existing Astropy unit, if any.
    """

    values = np.asarray(values)

    if name in table.colnames:
        unit = getattr(
            table[name],
            "unit",
            None,
        )

        if unit is not None:
            table[name] = values * unit
        else:
            table[name] = values

    else:
        table[name] = values


def apply_roman_f146_photometry(
    telescope,
    zero_point,
    apply_photometric_filter=True,
    detector=None,
    background_level=None,
    systematic_floor_frac=None,
    snr_min=None,
):
    """
    Replace the Roman/W149 placeholder photometry by the
    Roman F146 Pandeia-grid noise model.

    Important
    ---------
    The telescope flux entering this function must be the
    *theoretical* pyLIMA flux, i.e. no pyLIMA noise.

    The historical telescope/channel name "W149" is preserved
    internally, but magnitudes are interpreted as Roman F146 Vega.

    Policies
    --------
    full saturation:
        invalid/rejected

    partial saturation:
        retained with the Pandeia uncertainty

    unsaturated:
        retained normally

    faint:
        S/N < snr_min, default 5, rejected when
        apply_photometric_filter=True
    """

    if snr_min is None:
        snr_min = _env_float(
            "ROMAN_F146_SNR_MIN",
            5.0,
        )

    noise_model = get_roman_f146_noise_model(
        detector=detector,
        background_level=background_level,
        systematic_floor_frac=systematic_floor_frac,
    )

    lc = telescope.lightcurve

    if lc is None or len(lc) == 0:
        return telescope


    # ========================================================
    # Theoretical instantaneous total flux
    # ========================================================

    model_flux = _plain(
        lc["flux"],
        dtype=float,
    )

    if np.any(~np.isfinite(model_flux)):
        raise ValueError(
            "Roman theoretical flux contains non-finite values."
        )

    if np.any(model_flux <= 0.0):
        raise ValueError(
            "Roman theoretical flux contains non-positive values."
        )


    # pyLIMA model flux was built from the source/blend
    # magnitudes with the simulation zero point.
    #
    # Therefore this gives the instantaneous TOTAL
    # F146 Vega magnitude:
    #
    #   F = Fs A + Fb
    #
    #   m = ZP - 2.5 log10(F)
    #
    model_mag_vega = (
        float(zero_point)
        -
        2.5
        * np.log10(model_flux)
    )

    model_mag_ab = noise_model.vega_to_ab(
        model_mag_vega
    )


    # ========================================================
    # Pandeia grid
    # ========================================================

    result = noise_model.evaluate_ab(
        model_mag_ab
    )

    snr = np.asarray(
        result["snr"],
        dtype=float,
    )

    sigma_mag = np.asarray(
        result["sigma_mag"],
        dtype=float,
    )

    frac_flux_error = np.asarray(
        result["fractional_flux_error"],
        dtype=float,
    )

    valid = np.asarray(
        result["valid"],
        dtype=bool,
    )

    full_sat = np.asarray(
        result["full_saturated"],
        dtype=bool,
    )

    partial_sat = np.asarray(
        result["partial_saturated"],
        dtype=bool,
    )

    sat_raw = np.asarray(
        result["sat_nresultants"],
        dtype=float,
    )


    # ========================================================
    # Photometric selection
    # ========================================================

    finite_sigma = (
        np.isfinite(sigma_mag)
        &
        (sigma_mag > 0.0)
    )

    too_faint = (
        valid
        &
        np.isfinite(snr)
        &
        (snr < float(snr_min))
    )

    keep = (
        valid
        &
        (~full_sat)
        &
        finite_sigma
        &
        (~too_faint)
    )


    # ========================================================
    # Gaussian realization in magnitude
    #
    # This mirrors the Rubin branch of the existing pipeline.
    # ========================================================

    observed_mag_vega = model_mag_vega.copy()

    if apply_photometric_filter:
        draw_mask = keep
    else:
        draw_mask = (
            valid
            &
            (~full_sat)
            &
            finite_sigma
        )

    if np.any(draw_mask):
        observed_mag_vega[draw_mask] = np.random.normal(
            model_mag_vega[draw_mask],
            sigma_mag[draw_mask],
        )


    # ========================================================
    # Keep magnitude and flux representations synchronized
    # ========================================================

    observed_flux = (
        10.0
        **
        (
            0.4
            *
            (
                float(zero_point)
                -
                observed_mag_vega
            )
        )
    )

    err_flux = np.full(
        len(model_flux),
        np.nan,
        dtype=float,
    )

    good_error = (
        valid
        &
        (~full_sat)
        &
        finite_sigma
    )

    err_flux[good_error] = (
        observed_flux[good_error]
        *
        np.log(10.0)
        /
        2.5
        *
        sigma_mag[good_error]
    )

    inv_err_flux = np.zeros(
        len(model_flux),
        dtype=float,
    )

    good_inverse = (
        np.isfinite(err_flux)
        &
        (err_flux > 0.0)
    )

    inv_err_flux[good_inverse] = (
        1.0
        /
        err_flux[good_inverse]
    )

    _replace_column_preserve_unit(
        lc,
        "flux",
        observed_flux,
    )

    _replace_column_preserve_unit(
        lc,
        "err_flux",
        err_flux,
    )

    if "inv_err_flux" in lc.colnames:
        _replace_column_preserve_unit(
            lc,
            "inv_err_flux",
            inv_err_flux,
        )

    lc["mag"] = (
        observed_mag_vega
        * u.mag
    )

    lc["err_mag"] = (
        sigma_mag
        * u.mag
    )


    # ========================================================
    # Existing generic diagnostic convention
    # ========================================================

    lc["mag_model"] = model_mag_vega

    # m5 is no longer an input to Roman. Keep this column only
    # for compatibility with existing diagnostics.
    lc["photometry_m5"] = np.full(
        len(lc),
        np.nan,
    )

    # Saturation is not represented by a single hard magnitude
    # anymore; it is detector/Pandeia-state dependent.
    lc["photometry_saturation_mag"] = np.full(
        len(lc),
        np.nan,
    )

    lc["photometry_too_faint_5sigma"] = too_faint
    lc["photometry_saturated"] = full_sat
    lc["photometry_keep"] = keep
    lc["photometry_rejected"] = ~keep


    reason = np.full(
        len(lc),
        "ok",
        dtype=object,
    )

    reason[
        partial_sat
        &
        keep
    ] = "partial_saturated_valid"

    reason[
        too_faint
    ] = "too_faint_snr"

    reason[
        full_sat
    ] = "full_saturated"

    invalid_other = (
        (~valid)
        &
        (~full_sat)
    )

    reason[
        invalid_other
    ] = "invalid_noise_model"

    lc["photometry_flag_reason"] = reason


    # ========================================================
    # Roman/Pandeia-specific diagnostics
    # ========================================================

    lc["roman_model_mag_vega"] = (
        model_mag_vega
    )

    lc["roman_model_mag_ab"] = (
        model_mag_ab
    )

    lc["roman_snr"] = snr

    lc["roman_fractional_flux_error"] = (
        frac_flux_error
    )

    lc["roman_full_saturated"] = (
        full_sat
    )

    lc["roman_partial_saturated"] = (
        partial_sat
    )

    lc["roman_sat_nresultants_raw"] = (
        sat_raw
    )


    # ========================================================
    # Production filtering
    # ========================================================

    if apply_photometric_filter:
        telescope.lightcurve = lc[
            keep
        ]

    return telescope
