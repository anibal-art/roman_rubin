"""Edit the telescopes of a simulated pyLIMA event.

* inject_model_flux_for_ground_telescopes: fill the model flux of the
  non-Roman telescopes after simulate_lightcurve(add_noise=False).
* _capture_prefilter_telescope_times / _mask_time_indexed_telescope_arrays:
  keep pyLIMA's per-epoch parallax arrays aligned with the light curves
  after the photometric filter removes epochs (_PARALLAX_TIME_ARRAY_AXES
  lists those arrays and their epoch axis).
* remove_empty_telescopes: drop telescopes without epochs.
"""

import numpy as np

_PARALLAX_TIME_ARRAY_AXES = {
    "Earth_positions": 0,
    "Earth_speeds": 0,
    "sidereal_times": 0,
    "telescope_positions": 0,
    "Earth_positions_projected": 1,
    "Earth_speeds_projected": 1,
    "deltas_positions": 1,
}


def inject_model_flux_for_ground_telescopes(
    new_creation,
    pyLIMA_model,
    pyLIMA_parameters,
    roman_band_name="W149",
):
    """
    Refuerza el flujo teórico del modelo para telescopios Rubin.

    Roman también contiene ahora flujo teórico porque
    simulator.simulate_lightcurve() se llama con add_noise=False.
    Su ruido instrumental se agrega después con RomanF146Noise.
    """

    for tel in new_creation.telescopes:

        if tel.name == roman_band_name:
            continue

        model_flux = pyLIMA_model.compute_the_microlensing_model(
            tel,
            pyLIMA_parameters,
        )["photometry"]

        tel.lightcurve["flux"] = model_flux

    return new_creation


def _lightcurve_time_values(telescope):
    times = telescope.lightcurve["time"]

    if hasattr(times, "value"):
        times = times.value

    return np.asarray(times, dtype=float)


def _capture_prefilter_telescope_times(event):
    """Save only observation times needed to reconstruct filter masks."""

    out = {}

    for tel in event.telescopes:

        if tel.lightcurve is None:
            continue

        out[tel.name] = _lightcurve_time_values(
            tel
        ).copy()

    return out


def _mask_time_indexed_telescope_arrays(
    event,
    prefilter_times,
):
    """
    Restrict already-computed pyLIMA time-indexed arrays to the
    photometric rows that survived filtering.

    This is mathematically equivalent to recomputing parallax at the
    surviving times, but avoids the expensive ephemeris calculation.

    Fail loudly if filtering changed times rather than only removing
    rows.
    """

    for tel in event.telescopes:

        if tel.lightcurve is None or len(tel.lightcurve) == 0:
            continue

        if tel.name not in prefilter_times:
            raise RuntimeError(
                "Missing pre-filter telescope times: "
                f"{tel.name}"
            )

        before = np.asarray(
            prefilter_times[tel.name],
            dtype=float,
        )

        after = _lightcurve_time_values(tel)

        n_before = len(before)
        n_after = len(after)

        if n_after > n_before:
            raise RuntimeError(
                "Photometric filtering increased point count: "
                f"{tel.name}: {n_before} -> {n_after}"
            )

        # Fast no-op when nothing was removed.
        if (
            n_before == n_after
            and np.array_equal(before, after)
        ):
            continue

        # Filtering preserves order, so surviving rows are recovered
        # exactly by indexing into the original sorted timestamps.
        idx = np.searchsorted(before, after)

        if np.any(idx < 0) or np.any(idx >= n_before):
            raise RuntimeError(
                "Filtered times are not a subset of original times: "
                f"{tel.name}"
            )

        if not np.array_equal(before[idx], after):
            max_diff = float(
                np.max(np.abs(before[idx] - after))
            )

            raise RuntimeError(
                "Photometric filtering modified observation times "
                f"for {tel.name}; max difference={max_diff}"
            )

        for attr, axis in _PARALLAX_TIME_ARRAY_AXES.items():

            if not hasattr(tel, attr):
                continue

            value = getattr(tel, attr)

            if value is None:
                continue

            if isinstance(value, dict):

                for key, item in list(value.items()):

                    if item is None:
                        continue

                    shape = np.shape(item)

                    if len(shape) <= axis:
                        continue

                    n_axis = shape[axis]

                    if n_axis == n_before:
                        value[key] = np.take(
                            item,
                            idx,
                            axis=axis,
                        )

                    elif n_axis == n_after:
                        # Already synchronized by another operation.
                        continue

                    else:
                        raise RuntimeError(
                            "Unexpected time-axis length: "
                            f"{tel.name}.{attr}[{key!r}] "
                            f"has {n_axis}; expected "
                            f"{n_before} or {n_after}"
                        )

            else:

                shape = np.shape(value)

                if len(shape) <= axis:
                    continue

                n_axis = shape[axis]

                if n_axis == n_before:
                    setattr(
                        tel,
                        attr,
                        np.take(
                            value,
                            idx,
                            axis=axis,
                        ),
                    )

                elif n_axis == n_after:
                    continue

                else:
                    raise RuntimeError(
                        "Unexpected time-axis length: "
                        f"{tel.name}.{attr} has {n_axis}; "
                        f"expected {n_before} or {n_after}"
                    )


def remove_empty_telescopes(event, verbose=True):
    """
    Elimina del evento los telescopios/bandas sin observaciones.

    pyLIMA construye un parámetro de flujo por telescopio y ejecuta
    np.max(flux). Si la curva está vacía, la creación del modelo falla.
    """

    valid_telescopes = []
    removed_telescopes = []

    for telescope in event.telescopes:

        lightcurve = getattr(
            telescope,
            "lightcurve",
            None,
        )

        if lightcurve is None:
            n_points = 0
        else:
            try:
                n_points = len(lightcurve)
            except TypeError:
                n_points = 0

        if n_points > 0:
            valid_telescopes.append(telescope)
        else:
            removed_telescopes.append(telescope.name)

    event.telescopes = valid_telescopes

    if verbose and removed_telescopes:
        print(
            "Removed empty telescopes before model creation:",
            removed_telescopes,
        )

    return event, removed_telescopes
