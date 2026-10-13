#!/usr/bin/env python3

import inspect

from pyLIMA.parallax import parallax


def header(title):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def main():

    header("PYLIMA PARALLAX MODULE")

    print(
        "module file =",
        inspect.getsourcefile(parallax),
    )

    header("space_ephemerides SOURCE")

    print(
        inspect.getsource(
            parallax.space_ephemerides
        )
    )

    # ------------------------------------------------------------
    # Print source of functions referenced by space_ephemerides
    # when they are defined in the same module.
    # ------------------------------------------------------------

    source = inspect.getsource(
        parallax.space_ephemerides
    )

    candidates = [
        name
        for name in dir(parallax)
        if callable(
            getattr(parallax, name)
        )
        and not name.startswith("_")
    ]

    header("POSSIBLE HELPER FUNCTIONS")

    found = []

    for name in candidates:

        if name == "space_ephemerides":
            continue

        if name not in source:
            continue

        obj = getattr(
            parallax,
            name,
        )

        try:
            obj_source = inspect.getsource(
                obj
            )
        except Exception:
            continue

        found.append(name)

        print()
        print("-" * 88)
        print(name)
        print("-" * 88)
        print(obj_source)

    if not found:
        print(
            "No same-module helper function names "
            "were detected in space_ephemerides."
        )

    header("GLOBALS USED BY space_ephemerides")

    closure = inspect.getclosurevars(
        parallax.space_ephemerides
    )

    print("globals:")

    for name, value in sorted(
        closure.globals.items()
    ):
        print(
            f"  {name}: "
            f"{type(value).__module__}."
            f"{type(value).__name__}"
        )

        # For scipy/numpy interpolation objects, show repr too.
        if any(
            token in name.lower()
            for token in (
                "interp",
                "spline",
                "curve",
            )
        ):
            print(
                "    repr =",
                repr(value),
            )

    header("INTERPRETATION")

    print(
        "Look specifically for scipy.interpolate.interp1d, "
        "InterpolatedUnivariateSpline, np.interp, "
        "fill_value='extrapolate', or an equivalent explicit "
        "extrapolation rule."
    )


if __name__ == "__main__":
    main()
