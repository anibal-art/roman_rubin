import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_results_ import read_results

def fraction_curve_for_metric(
    met_rr: pd.DataFrame,
    met_roman: pd.DataFrame,
    true: pd.DataFrame,
    cat: str,
    bins: np.ndarray,
    param: str = "piE",
    thresh: float = 0.5,
    min_n: int = 10,
) -> pd.DataFrame:
    """
    Por bin de tE, calcula la fracción de eventos con (met_rr[param]/met_roman[param]) < 1,
    restringiendo a pares (Source, Set) tales que met_rr[param] < thresh.
    Devuelve: bin_left, bin_right, t_center, frac, frac_err, n.
    """
    bins = np.asarray(bins, dtype=float)
    if not np.all(np.diff(bins) > 0):
        raise ValueError("Los bins de tE deben ser crecientes (p.ej., np.logspace(1,3,50)).")

    if "peak_flag" not in true.columns:
        raise KeyError("La columna 'peak_flag' no está en 'true'.")
    if "tE" not in true.columns:
        raise KeyError("La columna 'tE' no está en 'true'.")

    true_cat = true.loc[true["peak_flag"] == cat, ["Source", "Set", "tE"]].copy()

    out = {"bin_left": [], "bin_right": [], "t_center": [], "frac": [], "frac_err": [], "n": []}

    for i in range(len(bins) - 1):
        ti, tf = bins[i], bins[i + 1]
        true_cat_bin = true_cat[(true_cat["tE"] > ti) & (true_cat["tE"] <= tf)]

        # defaults del bin
        out["bin_left"].append(ti)
        out["bin_right"].append(tf)
        out["t_center"].append(np.sqrt(ti * tf))

        if true_cat_bin.empty:
            out["frac"].append(np.nan)
            out["frac_err"].append(np.nan)
            out["n"].append(0)
            continue

        m_rr_bin = met_rr.merge(true_cat_bin[["Source", "Set"]], on=["Source", "Set"], how="inner")
        m_rom_bin = met_roman.merge(true_cat_bin[["Source", "Set"]], on=["Source", "Set"], how="inner")

        if param not in m_rr_bin.columns or param not in m_rom_bin.columns:
            out["frac"].append(np.nan)
            out["frac_err"].append(np.nan)
            out["n"].append(0)
            continue

        m_rr_sel = m_rr_bin[m_rr_bin[param] < thresh]
        if m_rr_sel.empty:
            out["frac"].append(np.nan)
            out["frac_err"].append(np.nan)
            out["n"].append(0)
            continue

        m_rom_sel = m_rom_bin.merge(m_rr_sel[["Source", "Set"]], on=["Source", "Set"], how="inner")

        pair = m_rr_sel.merge(
            m_rom_sel[["Source", "Set", param]],
            on=["Source", "Set"],
            suffixes=("_rr", "_roman"),
        )

        if pair.empty:
            out["frac"].append(np.nan)
            out["frac_err"].append(np.nan)
            out["n"].append(0)
            continue

        ratio = pair[f"{param}_rr"] / pair[f"{param}_roman"]
        n = int(ratio.size)

        if n < min_n:
            out["frac"].append(np.nan)
            out["frac_err"].append(np.nan)
            out["n"].append(n)
            continue

        frac = float(np.mean(ratio < 1.0))
        frac_err = float(np.sqrt(frac * (1.0 - frac) / n))  # binomial (Wald)

        out["frac"].append(frac)
        out["frac_err"].append(frac_err)
        out["n"].append(n)

    return pd.DataFrame(out)


def plot_three_metrics_fraction(
    mets_list,
    true: pd.DataFrame,
    bins: np.ndarray,
    param: str = "piE",
    thresh: float = 0.5,
    min_n: int = 10,
    label_rmet_lat=None,
    dict_cat_peak=None,
    xlim=(2, 50),
    ylim=(0.0, 1.05),
    outname="system_type_ratio",
    outdir=None,
    add_grid=True,
):
    """
    1x3 paneles (tres métricas). Un solo xlabel y ylabel globales.
    Guarda PNG y PDF.
    """
    if label_rmet_lat is None:
        label_rmet_lat = [f"métrica {i+1}" for i in range(3)]
    if dict_cat_peak is None:
        dict_cat_peak = {"A": "A", "B": "B", "C": "C"}

    # --- estilo: legible a tamaño de paper ---
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
    })

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)

    cat_styles = {
        "A": dict(marker="o", markersize=3.0, linestyle="-",  linewidth=1.0, alpha=1.0),
        "B": dict(marker="o", markersize=3.0, linestyle="--", linewidth=1.0, alpha=0.95),
        "C": dict(marker="o", markersize=3.0, linestyle="-.", linewidth=1.0, alpha=0.9),
    }

    for k, (met_rr, met_rom) in enumerate(mets_list):
        ax = axes[k]

        for cat in ["A", "B", "C"]:
            df = fraction_curve_for_metric(
                met_rr=met_rr,
                met_roman=met_rom,
                true=true,
                cat=cat,
                bins=bins,
                param=param,
                thresh=thresh,
                min_n=min_n,
            )

            m = np.isfinite(df["frac"].values)
            if not np.any(m):
                continue

            style = cat_styles.get(cat, {})
            ax.errorbar(
                df["t_center"].values[m],
                df["frac"].values[m],
                yerr=df["frac_err"].values[m],
                capsize=2,
                elinewidth=1.0,
                **style,
                label=dict_cat_peak.get(cat, cat),
            )

        ax.set_xscale("log")
        ax.set_title(f"{label_rmet_lat[k]} < 1")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if add_grid:
            ax.grid(True, which="major", ls=":", lw=0.7, alpha=0.7)
            ax.grid(True, which="minor", ls=":", lw=0.4, alpha=0.5)

        # ticks hacia adentro, estilo paper
        ax.tick_params(which="both", direction="in", top=True, right=True)

        # leyenda una sola vez
        if k == 1:
            ax.legend(
                loc="upper center",
                ncols=3,
                bbox_to_anchor=(0.5, 1.28),
                frameon=False,
                handlelength=2.0,
                columnspacing=1.2,
            )

    # labels globales con padding controlado (más estable que y=-0.04)
    fig.supxlabel(r"$t_E$ [days]", fontsize=16)
    fig.supylabel("Fraction of events", fontsize=16)

    # ajustar márgenes: controla separación de labels/ticks
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.88, wspace=0.25)

    if outdir is None:
        outdir = os.getcwd()

    png_path = os.path.join(outdir, f"{outname}.png")
    pdf_path = os.path.join(outdir, f"{outname}.pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    return fig, axes


label_rmet_lat = [r"$\frac{RB(\pi_E)_{Roman+Rubin}}{RB(\pi_E)_{Roman}}$", r"$\frac{SB(\pi_E)_{Roman+Rubin}}{SB(\pi_E)_{Roman}}$", 
                 r"$\frac{NU(\pi_E)_{Roman+Rubin}}{NU(\pi_E)_{Roman}}$"]
dict_cat_peak = {"A":"peak in Rubin+Roman",
                "B":"peak in Rubin",
                "C":"peak in Roman"}
# =============================================================================
model = 'USBL'
system_type = "PB"
true, fit_roman, fit_rr, met_1_roman, met_1_rr, met_2_roman, met_2_rr, met_3_roman, met_3_rr, met1_ratio, met2_ratio, met3_ratio =read_results(system_type, model)
mets_list = [
    (met_1_rr, met_1_roman),
    (met_2_rr, met_2_roman),
    (met_3_rr, met_3_roman),]
# =============================================================================

param = "piE"
thresh = 1
time_bins = np.logspace(-2, 4, 30)

plot_three_metrics_fraction(
    mets_list=mets_list,
    true=true,
    bins=time_bins,
    param=param,
    thresh=thresh,
    min_n=10,
    outname=f"hist{system_type}_ratio",
    label_rmet_lat=label_rmet_lat if 'label_rmet_lat' in globals() else None,
    dict_cat_peak=dict_cat_peak if 'dict_cat_peak' in globals() else None,
    xlim=(5,10000))
