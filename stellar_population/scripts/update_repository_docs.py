#!/usr/bin/env python3
"""Maintain root README navigation and exhaustive Python-file index.

Run from repository root:
    python -m stellar_population.scripts.update_repository_docs

The command replaces only its own delimited Markdown sections. It never
modifies the scientific pipeline or guesses an undocumented module's purpose.
"""

from __future__ import annotations

import ast
import argparse
from collections import defaultdict
from pathlib import Path
import subprocess
import sys

README_BEGIN = "<!-- BEGIN ROMAN_RUBIN_CODE_MAP -->"
README_END = "<!-- END ROMAN_RUBIN_CODE_MAP -->"
INDEX_BEGIN = "<!-- BEGIN ROMAN_RUBIN_SOURCE_INDEX -->"
INDEX_END = "<!-- END ROMAN_RUBIN_SOURCE_INDEX -->"

# Only factual, human-reviewed descriptions belong in this mapping.
# Missing files are omitted. Any other .py appears in docs/CODE_INDEX.md.
GROUPS = [
    ("Production pipeline: entry points", [
        ("launch_catalog_simfit_from_config.py", "Active launcher: read one JSON configuration and run sim_fit for each materialized catalogue event."),
        ("functions_roman_rubin.py", "Public facade: re-exports the pipeline (sim_event, sim_fit, fits, photometry, records); configures IERS and the Earth-ephemerides cache."),
        ("pipeline/sim_fit.py", "Simulate one catalogue event, run the Roman+Rubin and Roman-only fits and save the outputs."),
        ("pipeline/sim_fit_multi_fits.py", "sim_fit plus several named fits on the same simulated data (e.g. H1/H0) with LRT summaries."),
    ]),
    ("Simulation of one event", [
        ("simulation/simulate_event.py", "sim_event: telescopes at the event RA/Dec, true pyLIMA model from the catalogue realization, noiseless light curves, Roman/Rubin photometry and Roman-only detection."),
        ("simulation/event_inputs.py", "Validate and normalize one materialized catalogue row into event_params (nothing is sampled)."),
        ("simulation/realization.py", "EventRealization: parameters, materialized fluxes, blend ratios and caustic origin taken from the catalogue row."),
        ("simulation/core.py", "pyLIMA compute_pyLIMA_parameters and noiseless simulate_lightcurve for an already-built model."),
        ("simulation/roman_detection.py", "Roman-only detection criteria (Delta chi2 >= 300 and n3sigma consecutive points)."),
        ("simulation/telescope_arrays.py", "Model flux of ground telescopes, parallax arrays after the photometric filter, removal of empty telescopes."),
        ("simulation/parallax_cache.py", "Exact cache of the Earth ephemerides used by pyLIMA's parallax."),
        ("set_telescopes_pyLIMA.py", "tel_roman_rubin: Roman and Rubin pyLIMA telescopes at the event RA/Dec; Rubin OpSim/MAF dataslice, paths and caches (OpSim global state)."),
        ("simulation/telescope_templates.py", "In-process cache of the Roman/Rubin pyLIMA telescope templates."),
        ("simulation/roman_cadence.py", "Roman GBTDS time sampling (nominal seasons and off-seasons)."),
        ("simulation/rubin_dataslice.py", "Rubin OpSim dataslice helpers: canonical filter names, Rubin time series, sky cell of the dataslice cache."),
        ("simulation/time_windows.py", "Optional time window applied to telescopes and to the OpSim dataslice."),
        ("set_model_pyLIMA.py", "build_pyLIMA_model: single constructor of the PSPL/FSPL/USBL pyLIMA models used to simulate, fit and plot."),
        ("detection_criteria.py", "Detection and observational-selection criteria applied to simulated light curves."),
    ]),
    ("Photometry and noise", [
        ("photometry/roman_f146.py", "RomanF146Noise: interpolation of the precomputed Pandeia F146 grid, including saturation/validity state; provides sigma_mag_ab()."),
        ("photometry/roman_photometry.py", "Apply the F146 noise model to the simulated Roman telescope photometry."),
        ("photometry/roman_rubin_photometry.py", "Rubin rubin_sim m5/gamma noise, m5 and saturation filters and photometry flags; calls the Roman F146 photometry."),
        ("photometry/constants.py", "Zero points: simulation band zero points (W149 27.615, Rubin ugrizy) and the pyLIMA fit zero point (27.4)."),
        ("photometry/data/roman_f146_pandeia_2026p1.csv", "Precomputed Pandeia F146 grid: per-detector S/N, sigma_mag, saturation flags, and magnitude in AB."),
        ("photometry/data/roman_f146_pandeia_2026p1.json", "Metadata and configuration associated with the Pandeia F146 grid."),
        ("stellar_population/scripts/build_roman_f146_pandeia_grid.py", "Rebuild the Pandeia input grid; not required for routine light-curve simulations or plotting."),
        ("stellar_population/scripts/audit_roman_f146_noise_grid.py", "Audit computed Pandeia grid and saturation transitions."),
    ]),
    ("Fits", [
        ("fit_lc.py", "fit_rubin_roman (Roman+Rubin and Roman-only TRF/DE fits), save_fit_results, model_rubin_roman (model rebuilt for plots)."),
        ("fitting/fit_event.py", "Build the pyLIMA Event that is fitted from lc_to_fit, at the RA/Dec of the simulated event."),
        ("fitting/fit_parameters.py", "Fit-model parameters: aliases, pyLIMA order, initial values, fit pyLIMA model."),
        ("fitting/fit_bounds.py", "Bounds of the fitted parameters (custom, narrow, wide)."),
        ("fitting/initial_guess.py", "Initial guess of the fit, validated against the bounds."),
        ("fitting/trf_optimizer.py", "pyLIMA fitter, TRF least-squares options, run_fit and optimizer diagnostics."),
        ("fitting/fit_inputs.py", "lc_to_fit extraction, fit-only time windows, event coordinates, fit options and result paths."),
        ("fitting/run_fits.py", "run_all_fits (Roman+Rubin and Roman-only) and run_multiple_named_fits."),
        ("fitting/likelihood.py", "Photometric chi2 / logL and likelihood-ratio tests."),
        ("fitting/true_generator.py", "chi2 / logL of the true simulated model on the fitted data."),
        ("fitting/multi_fit.py", "Named fit specs, result entries and one-row parquet summary."),
    ]),
    ("Outputs and analysis", [
        ("analysis/simfit_records.py", "Per-event output records: Analysis_Event, parameter/uncertainty row, parquet."),
        ("analysis/analysis_event.py", "Analysis_Event: one simulated + fitted event for the analysis tables."),
        ("class_analysis.py", "Metrics helpers for the notebooks; re-exports Analysis_Event and the Roman+Rubin figures."),
        ("read_save.py", "Simulation/fit serialization and loading of saved results."),
        ("timing_utils.py", "Stage-by-stage runtime diagnostics."),
        ("analysis/fisher_information.py", "Local Fisher information for Roman-Rubin E1 (physical derivatives by five-point finite differences)."),
    ]),
    ("Plotting", [
        ("plotting/roman_rubin_plots.py", "Roman+Rubin light-curve figures from saved simulations and fits (plot_roman_rubin_plt, graph_maker)."),
        ("plotting/pyLIMA_plots.py", "Light curves, model telescopes and colors (modified copy of pyLIMA's pyLIMA_plots)."),
        ("plotting/pyLIMA_geometry.py", "Lens geometry and astrometry plots."),
        ("plotting/pyLIMA_distributions.py", "Sample distributions and parameter table."),
        ("pyLIMA_plots.py", "Compatibility facade for plotting/ (import pyLIMA_plots keeps working)."),
    ]),
    ("Stellar populations and event catalogue", [
        ("stellar_population/scripts/download_trilegal.py", "Generate/download TRILEGAL stellar populations for configured Galactic-bulge cells."),
        ("stellar_population/scripts/build_genulens_trilegal_reservoir.py", "Produce GENULENS samples/reservoir for the configured fields."),
        ("stellar_population/scripts/build_precomputed_event_catalogs.py", "Match Galactic lens/source parameters and construct precomputed FFP, BH and binary-lens event Parquet files."),
        ("stellar_population/scripts/run_full_event_catalog_production.py", "Resumable per-cell orchestration of TRILEGAL, GENULENS, event catalogue generation and assembly."),
        ("ulens_params.py", "Microlensing system parameters, priors and model-specific parameterization used to build the catalogue."),
        ("catalog/blending.py", "Deterministic, precomputed blending (source/total fluxes per band) for the catalogue."),
        ("catalog/caustic_origin.py", "Reproducible caustic-origin realizations for binary lenses."),
        ("catalog/amax.py", "Catalogue-level Amax detectability prefilter."),
        ("catalog/event_reader.py", "Load a deterministic catalogue event by population and global index, including the simulation seed."),
        ("stellar_population/PRODUCTION_METHOD.md", "Detailed production method and assumptions; consult before changing catalogue inputs."),
    ]),
    ("Validation, diagnostics, and figures", [
        ("stellar_population/scripts/smoke_roman_rubin.py", "End-to-end catalogue-event simulation and Roman-only/Roman+Rubin fit smoke tests."),
        ("stellar_population/scripts/audit_parallax_mask.py", "Compare masked time-dependent parallax geometry with recomputation after photometry filtering."),
        ("stellar_population/scripts/update_repository_docs.py", "Refresh README's workflow/code map and exhaustive Python source index."),
    ]),
]


def rel_link(path: str) -> str:
    return f"[`{path}`]({path})"


def replace_marked(existing: str, begin: str, end: str, content: str) -> str:
    block = f"{begin}\n{content.rstrip()}\n{end}"
    if begin in existing or end in existing:
        if existing.count(begin) != 1 or existing.count(end) != 1:
            raise ValueError(f"Inconsistent markers for {begin}")
        left = existing.index(begin)
        right = existing.index(end) + len(end)
        if left > right:
            raise ValueError("Section markers are out of order")
        return existing[:left] + block + existing[right:]
    # New README navigation should be visible immediately on GitHub:
    # insert after the first top-level title, without deleting other text.
    if begin == README_BEGIN:
        lines = existing.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if line.startswith("# "):
                before = "".join(lines[:i + 1]).rstrip()
                after = "".join(lines[i + 1:]).lstrip("\n")
                return before + "\n\n" + block + "\n\n" + after
        return block + "\n\n" + existing
    return existing.rstrip() + "\n\n" + block + "\n"


def tracked_python_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, check=True, stdout=subprocess.PIPE,
    )
    paths = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        name = raw.decode("utf-8", "surrogateescape")
        p = root / name
        if p.is_file() and p.suffix == ".py" and not p.is_symlink():
            paths.append(name)
    return sorted(set(paths))


def module_description(path: Path) -> str:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        doc = ast.get_docstring(tree)
    except (OSError, UnicodeError, SyntaxError):
        return "**Needs documentation or contains unparsable Python.**"
    if not doc:
        return "**No module docstring: inspect the code before assuming its purpose.**"
    first = " ".join(doc.strip().split())
    first = first.split(". ", 1)[0].rstrip(".")
    first = first.replace("|", "\\|").replace("`", "'")
    return first[:240] + ("..." if len(first) > 240 else "") + "."


def render_readme(root: Path, paths: list[str]) -> str:
    lines = [
        "## Repository guide: where does each part live?",
        "",
        "This section is maintained by `stellar_population/scripts/update_repository_docs.py`.",
        "The full searchable file inventory is in [docs/CODE_INDEX.md](docs/CODE_INDEX.md).",
        "For scientific assumptions see [stellar_population/PRODUCTION_METHOD.md](stellar_population/PRODUCTION_METHOD.md).",
        "",
        "### End-to-end data flow",
        "",
        "```text",
        "Sky cells + TRILEGAL source stars + GENULENS lens/source population",
        "    -> source/lens matching and FFP/BH/planetary-system event catalogues",
        "       (parameters, RA/Dec, materialized fluxes/blending, caustic origin)",
        "    -> indexed Parquet shards + catalog/event_reader.py (population, index, seed)",
        "",
        "launch_catalog_simfit_from_config.py -> pipeline/sim_fit.py, for each event:",
        "    -> simulation/simulate_event.py (sim_event): tel_roman_rubin at the event RA/Dec",
        "       (Roman template + Rubin OpSim), pyLIMA model, noiseless light curves",
        "    -> photometry/: Roman F146/Pandeia noise + Rubin rubin_sim noise + m5/saturation filter",
        "    -> Roman-only detection (simulation/roman_detection.py)",
        "    -> Roman+Rubin and Roman-only fits (fit_lc.fit_rubin_roman, fitting/)",
        "    -> outputs (analysis/simfit_records.py): parquet row, Analysis_Event, saved fits",
        "```",
        "",
    ]
    for title, files in GROUPS:
        found = [(p, desc) for p, desc in files if (root / p).is_file()]
        if not found:
            continue
        lines += [f"### {title}", "", "| File | Responsibility |", "|---|---|"]
        for p, desc in found:
            lines.append(f"| {rel_link(p)} | {desc} |")
        lines.append("")

    lines += [
        "### Roman F146: read this before modifying noise or figures",
        "",
        "- **Current noise input:** saved Pandeia R2026.1 F146 grid, with the project-specific Rev-H `IM_66_6_V2` read pattern. The grid is already computed: do not rerun Pandeia just to make a figure.",
        "- **Use the production interpolation:** `RomanF146Noise` in `photometry/roman_f146.py`, especially `sigma_mag_ab()` for a magnitude-uncertainty curve. The simulation applies the model via `photometry/roman_photometry.py`.",
        "- **Magnitude convention:** the Pandeia noise grid is in **AB**; the TRILEGAL Roman source catalogue uses **Vega**. Use the project's calibrated AB–Vega conversion before mixing magnitudes or plotting thresholds.",
        "- **Catalog selection is not noise sensitivity:** a TRILEGAL extraction cut at F146 = 28 Vega, when configured, is not automatically the Pandeia 5-sigma limiting magnitude.",
        "- **Legacy name:** some pyLIMA telescope objects/data products still use `W149` as an internal band label. This is not a justification for applying the obsolete W149 noise prescription to F146.",
        "- **Observing schedule:** the manuscript uses an approximately **12-minute Roman cadence** during roughly 72-day seasons. Observation timestamps must be checked against the telescope template; changing noise must not silently alter cadence.",
        "- **Rubin:** preserve the same `rubin_sim`/OpSim-dependent uncertainty calculation used by the production simulator. Do not substitute representative m5/gamma constants in publication figures.",
        "",
        "### Quick validation and execution",
        "",
        "From the repository root, activate the matching environment. Run existing smoke-test modules as Python modules (with `-m`), not by calling deeply nested script paths directly.",
        "",
    ]
    if (root / "stellar_population/scripts/smoke_roman_rubin.py").is_file():
        lines += [
            "```bash",
            "python -m stellar_population.scripts.smoke_roman_rubin --population BH --index 1167",
            "python -m stellar_population.scripts.smoke_roman_rubin --population FFP --index 366",
            "python -m stellar_population.scripts.smoke_roman_rubin --population Planets_systems --index 386",
            "```",
            "",
        ]
    if (root / "stellar_population/scripts/audit_parallax_mask.py").is_file():
        lines += ["For parallax filtering equivalence see `stellar_population/scripts/audit_parallax_mask.py`.", ""]
    if (root / "stellar_population/scripts/run_full_event_catalog_production.py").is_file():
        lines += [
            "Inspect the catalogue-production driver **before running expensive jobs**:",
            "",
            "```bash",
            "python stellar_population/scripts/run_full_event_catalog_production.py --help",
            "```",
            "",
        ]
    lines += [
        "### Maintenance and reproducibility",
        "",
        "1. When modifying a file's scientific role, update the mapping in `update_repository_docs.py` and its module docstring.",
        "2. Regenerate the README and complete source index with `python -m stellar_population.scripts.update_repository_docs` before committing.",
        "3. Confirm output paths, magnitude conventions, cuts, RNG seeds, OpSim database, and noise-grid version for each production run; the README is a navigation map, not a substitute for versioned configuration.",
        "4. A file without a module docstring is flagged in the source index instead of being assigned an invented description.",
        "",
        "### Validation work still to close",
        "",
        "- Ensure **Roman-only** `chi2_true` uses only the Roman observations, rather than the joint Roman+Rubin true chi-square, before interpreting `delta_chi2_true`.",
        "- Preserve **pre-filter** photometric rejection counts and whole-band removal diagnostics; statistics calculated only after filtering cannot recover removed rows.",
        "- Check any planned publication figure against the actual production F146 interpolator and exact Rubin uncertainty configuration.",
        "",
    ]
    return "\n".join(lines)


def render_index(root: Path, paths: list[str]) -> str:
    groups: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        parent = str(Path(path).parent)
        groups[parent].append(path)
    lines = [
        "# Python source index",
        "",
        "Automatically enumerated from locally present Git-tracked/untracked, non-ignored `.py` files.",
        "Descriptions come only from each module's **own top-level docstring**.",
        "Missing descriptions are explicitly flagged, not fabricated.",
        "This index is not an execution/configuration contract: see the root README and production method.",
        "",
        f"**Indexed Python files:** {len(paths)}",
        "",
    ]
    for directory, filenames in sorted(groups.items()):
        lines += [f"## `{directory}`", "", "| File | Top-level module documentation |", "|---|---|"]
        for name in filenames:
            rel_from_docs = "../" + name
            doc = module_description(root / name)
            lines.append(f"| [`{name}`]({rel_from_docs}) | {doc} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if outputs are out of date, without writing")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if not (root / ".git").exists() or not (root / "functions_roman_rubin.py").is_file():
        raise SystemExit("Run from the root of the roman_rubin Git repository")
    if not (root / "photometry/roman_f146.py").is_file():
        raise SystemExit("Missing photometry/roman_f146.py: verify checkout")
    paths = tracked_python_paths(root)
    readme = root / "README.md"
    current_readme = readme.read_text(encoding="utf-8") if readme.exists() else "# Roman + Rubin Microlensing\n"
    new_readme = replace_marked(current_readme, README_BEGIN, README_END, render_readme(root, paths))

    dest = root / "docs/CODE_INDEX.md"
    current_index = dest.read_text(encoding="utf-8") if dest.exists() else ""
    new_index = replace_marked(current_index, INDEX_BEGIN, INDEX_END, render_index(root, paths))

    updates = [(readme, current_readme, new_readme), (dest, current_index, new_index)]
    stale = [p for p, old, new in updates if old != new]
    if args.check:
        if stale:
            print("Documentation out of date:", ", ".join(str(p.relative_to(root)) for p in stale))
            raise SystemExit(1)
        print(f"Documentation up-to-date; {len(paths)} Python files indexed")
        return
    for path, old, new in updates:
        if old != new:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new, encoding="utf-8")
            print("Updated:", path.relative_to(root))
        else:
            print("Unchanged:", path.relative_to(root))
    print(f"Indexed {len(paths)} Python source files")


if __name__ == "__main__":
    main()
