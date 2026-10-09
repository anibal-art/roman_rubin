import argparse
import copy
import numpy as np

from pathlib import Path
from pyLIMA.fits.objective_functions import (
    all_telescope_photometric_chi2,
)

import functions_roman_rubin as frr
from catalog.event_reader import (
    ROOT, load_event,
)

# Attributes indexed by observation time.
# 0 = observations along rows
# 1 = observations along columns
ARRAY_AXES = {
    "Earth_positions": 0,
    "Earth_speeds": 0,
    "sidereal_times": 0,
    "telescope_positions": 0,
    "Earth_positions_projected": 1,
    "Earth_speeds_projected": 1,
    "deltas_positions": 1,
}

ATOL_GEOMETRY = 1e-11
RTOL_GEOMETRY = 1e-12


def values(column):
    return np.asarray(
        column.value if hasattr(column, "value") else column
    )


def filter_telescope(telescope, keep, filter_geometry):
    """Apply one photometry mask consistently."""

    keep = np.asarray(keep, dtype=bool)
    n_before = len(telescope.lightcurve)

    assert keep.shape == (n_before,), (
        telescope.name, keep.shape, n_before
    )

    if filter_geometry:
        for attr_name, axis in ARRAY_AXES.items():
            container = getattr(telescope, attr_name)

            if "photometry" not in container:
                continue

            original = np.asarray(
                container["photometry"]
            )

            if original.size == 0:
                continue

            if (
                original.ndim <= axis
                or original.shape[axis] != n_before
            ):
                raise RuntimeError(
                    f"{telescope.name}/{attr_name}: "
                    f"shape {original.shape} incompatible "
                    f"with N={n_before}"
                )

            container["photometry"] = np.take(
                original,
                np.flatnonzero(keep),
                axis=axis,
            )

    # Apply the exact same mask to every lightcurve column.
    telescope.lightcurve = telescope.lightcurve[keep]


def apply_masks(model, masks, recalculate):
    """Produce a filtered model using either strategy."""

    kept = []

    for tel in model.event.telescopes:
        filter_telescope(
            tel,
            masks[tel.name],
            filter_geometry=not recalculate,
        )

        if len(tel.lightcurve) == 0:
            continue

        if recalculate:
            tel.compute_parallax(
                model.parallax_model,
                model.event.North,
                model.event.East,
            )

        kept.append(tel)

    model.event.telescopes = kept
    return model


def compare_array(label, left, right, atol=ATOL_GEOMETRY):
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)

    if a.shape != b.shape:
        raise AssertionError(
            f"{label}: shape {a.shape} != {b.shape}"
        )

    np.testing.assert_allclose(
        a,
        b,
        rtol=RTOL_GEOMETRY,
        atol=atol,
        equal_nan=False,
        err_msg=label,
    )

    difference = (
        float(np.max(np.abs(a - b)))
        if a.size else 0.0
    )

    return difference


def run_simulation(event, model_name, half_window, filtered):
    t0 = float(event["t0"])

    model, truth, decision = frr.simulate_event_for_fit(
        i=int(event["simulation_seed"]),
        event_params=event,
        path_ephemerides=str(
            ROOT / "ephemerides/Roman_positions.npy"
        ),
        model=model_name,
        time_window=(
            t0 - half_window,
            t0 + half_window,
        ),
        use_roman=True,
        use_rubin=True,
        truth_parallax=True,
        rubin_pointing_mode="source",
        rubin_cache_cell_deg=None,
        apply_detection_criteria=False,
        apply_photometric_filter=filtered,
    )

    if model is None:
        raise RuntimeError("Simulation returned no model")

    return model, truth


def compare_models(masked, recalculated):
    tels_a = {t.name: t for t in masked.event.telescopes}
    tels_b = {t.name: t for t in recalculated.event.telescopes}

    assert tels_a.keys() == tels_b.keys()

    print("\nGEOMETRY COMPARISON")
    print("-" * 70)

    for name in tels_a:
        a = tels_a[name]
        b = tels_b[name]

        compare_array(
            f"{name}/time",
            values(a.lightcurve["time"]),
            values(b.lightcurve["time"]),
            atol=0.0,
        )

        maximum = 0.0

        for attr_name, axis in ARRAY_AXES.items():
            container_a = getattr(a, attr_name)
            container_b = getattr(b, attr_name)

            has_a = "photometry" in container_a
            has_b = "photometry" in container_b

            assert has_a == has_b, (
                name, attr_name, "missing geometry"
            )

            if not has_a:
                continue

            if np.asarray(container_a["photometry"]).size == 0:
                continue

            error = compare_array(
                f"{name}/{attr_name}",
                container_a["photometry"],
                container_b["photometry"],
            )
            maximum = max(maximum, error)

        assert np.asarray(
            a.deltas_positions["photometry"]
        ).shape == (2, len(a.lightcurve))

        # Important: spacecraft_positions is an ephemeris table,
        # not a per-observation array, so it was not masked.
        if a.location == "Space":
            compare_array(
                f"{name}/spacecraft_positions",
                a.spacecraft_positions["photometry"],
                b.spacecraft_positions["photometry"],
            )

        print(
            f"{name:6s} "
            f"N={len(a.lightcurve):6d} "
            f"max_geometry_difference={maximum:.3e}"
        )


def calculate_chi2(model, truth):
    # Convert the existing noisy magnitudes to observed fluxes
    # just as sim_fit does. Do not generate noise again.
    frr.replace_flux_by_noisy_magnitude_flux_ZP(
        model,
        verbose=False,
    )

    chi2 = float(
        all_telescope_photometric_chi2(model, truth)
    )

    n = sum(
        len(t.lightcurve)
        for t in model.event.telescopes
    )

    assert n > 0
    assert np.isfinite(chi2)

    return chi2, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--population",
        choices=("FFP", "BH", "Planets_systems"),
        default="Planets_systems",
    )
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()

    defaults = {
        "Planets_systems": 386,
        "FFP": 366,
        "BH": 1167,
    }

    models = {
        "Planets_systems": "USBL",
        "FFP": "FSPL",
        "BH": "FSPL",
    }

    index = (
        args.index
        if args.index is not None
        else defaults[args.population]
    )

    event = load_event(args.population, index)
    model_name = models[args.population]
    half_window = 3.0 if args.population == "FFP" else 30.0

    print("\nEVENT:", args.population, index)
    print("MODEL:", model_name)
    print("SEED:", event["simulation_seed"])
    print("FIELD:", event["field_id"])

    # 1. Generate full photometry and geometry.
    original, truth = run_simulation(
        event, model_name, half_window, filtered=False
    )

    masks = {}
    counts = {}

    for tel in original.event.telescopes:
        if "photometry_keep" not in tel.lightcurve.colnames:
            raise RuntimeError(
                f"Missing photometry_keep in {tel.name}"
            )

        keep = values(
            tel.lightcurve["photometry_keep"]
        ).astype(bool)

        masks[tel.name] = keep

        counts[tel.name] = (
            len(keep),
            int(keep.sum()),
        )

    print("\nFILTER MASKS (before -> after)")
    for name, (before, after) in counts.items():
        print(f"{name:6s} {before:6d} -> {after:6d}")

    # 2. Two independent copies of the SAME simulation.
    masked = copy.deepcopy(original)
    recalculated = copy.deepcopy(original)

    masked = apply_masks(
        masked, masks, recalculate=False
    )

    recalculated = apply_masks(
        recalculated, masks, recalculate=True
    )

    # 3. Compare positions, projections and parallax.
    compare_models(masked, recalculated)

    # 4. Compare actual model photometry point by point.
    print("\nMODEL PHOTOMETRY COMPARISON")

    for a, b in zip(
        masked.event.telescopes,
        recalculated.event.telescopes,
    ):
        assert a.name == b.name

        flux_a = masked.compute_the_microlensing_model(
            a, truth
        )["photometry"]

        flux_b = recalculated.compute_the_microlensing_model(
            b, truth
        )["photometry"]

        error = compare_array(
            f"{a.name}/model_flux",
            flux_a, flux_b,
            atol=1e-9,
        )

        print(
            f"{a.name:6s} "
            f"max_model_flux_difference={error:.3e}"
        )

    # 5. Calculate official pyLIMA chi2.
    chi2_mask, n_mask = calculate_chi2(masked, truth)
    chi2_recalc, n_recalc = calculate_chi2(
        recalculated, truth
    )

    assert n_mask == n_recalc

    np.testing.assert_allclose(
        chi2_mask,
        chi2_recalc,
        rtol=1e-11,
        atol=1e-8,
    )

    print("\nOFFICIAL PYLIMA CHI2")
    print("Mask only :", chi2_mask)
    print("Recompute :", chi2_recalc)
    print("N         :", n_mask)
    print("Difference:", abs(chi2_mask - chi2_recalc))

    # 6. Third, independent run through the current pipeline.
    current, current_truth = run_simulation(
        event, model_name, half_window, filtered=True
    )

    current_tels = {
        t.name: t for t in current.event.telescopes
        if len(t.lightcurve) > 0
    }

    mask_tels = {
        t.name: t for t in masked.event.telescopes
    }

    assert current_tels.keys() == mask_tels.keys()

    for name in mask_tels:
        left = mask_tels[name]
        right = current_tels[name]

        for column in ("time", "mag", "err_mag"):
            compare_array(
                f"{name}/current_{column}",
                values(left.lightcurve[column]),
                values(right.lightcurve[column]),
                atol=1e-10 if column != "time" else 0.0,
            )

        compare_array(
            f"{name}/current_parallax",
            left.deltas_positions["photometry"],
            right.deltas_positions["photometry"],
        )

    chi2_current, n_current = calculate_chi2(
        current, current_truth
    )

    assert n_current == n_mask

    np.testing.assert_allclose(
        chi2_current,
        chi2_mask,
        rtol=1e-11,
        atol=1e-8,
    )

    print("\nCURRENT PIPELINE CHI2:", chi2_current)

    print("\n" + "=" * 70)
    print("PARALLAX MASK EQUIVALENCE AUDIT PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
