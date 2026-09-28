import numpy as np
import matplotlib.pyplot as plt


def make_adaptive_bins_te(tE, min_count=30):
    """
    Construye bins adaptativos en tE garantizando aproximadamente
    min_count eventos por bin.

    Parameters
    ----------
    tE : array-like
        Einstein timescales. Deben ser positivos.
    min_count : int
        Número mínimo aproximado de eventos por bin.

    Returns
    -------
    edges : np.ndarray
        Bordes de bins en unidades originales de tE.
    """
    tE = np.asarray(tE, dtype=float)
    tE = tE[np.isfinite(tE) & (tE > 0)]

    if len(tE) == 0:
        raise ValueError("No hay valores válidos de tE > 0.")

    tE_sorted = np.sort(tE)
    n = len(tE_sorted)

    # --------- [NUEVO BLOQUE] cortes por bloques ----------
    split_idx = list(range(0, n, min_count))
    if split_idx[-1] != n:
        split_idx.append(n)
    # -----------------------------------------------------

    # --------- [NUEVO BLOQUE] construir bordes -----------
    edges = [tE_sorted[0]]
    for i in split_idx[1:-1]:
        left = tE_sorted[i - 1]
        right = tE_sorted[i]
        edges.append(np.sqrt(left * right))  # centro geométrico
    edges.append(tE_sorted[-1] * (1 + 1e-10))
    # -----------------------------------------------------

    edges = np.array(edges, dtype=float)
    edges = np.unique(edges)

    if len(edges) < 2:
        raise ValueError("No fue posible construir bins adaptativos únicos.")

    return edges


def compute_hist_ratio_adaptive(top, bottom, bins, min_bin_count=20):
    """
    Calcula la fracción N_top / N_bottom por bin y su error binomial,
    descartando bins con pocos eventos en el denominador.

    Parameters
    ----------
    top : array-like
        Valores de tE de los eventos que cumplen el criterio.
    bottom : array-like
        Valores de tE del total de eventos de referencia.
    bins : array-like
        Bordes de bins.
    min_bin_count : int
        Número mínimo de eventos en el denominador para reportar el bin.

    Returns
    -------
    x : np.ndarray
        Centros geométricos de los bins válidos.
    frac : np.ndarray
        Fracción por bin.
    err : np.ndarray
        Error binomial por bin.
    counts : np.ndarray
        Número de eventos del denominador por bin.
    """
    top = np.asarray(top, dtype=float)
    bottom = np.asarray(bottom, dtype=float)
    bins = np.asarray(bins, dtype=float)

    N_top, _ = np.histogram(top, bins=bins)
    N_bot, _ = np.histogram(bottom, bins=bins)

    x_all = np.sqrt(bins[:-1] * bins[1:])

    frac = np.full_like(N_top, np.nan, dtype=float)
    err = np.full_like(N_top, np.nan, dtype=float)

    # --------- [NUEVO BLOQUE] mínimo número de eventos por bin ----
    mask = N_bot >= min_bin_count
    frac[mask] = N_top[mask] / N_bot[mask]
    err[mask] = np.sqrt(frac[mask] * (1 - frac[mask]) / N_bot[mask])
    # ----------------------------------------------------------------

    return x_all[mask], frac[mask], err[mask], N_bot[mask]

from read_results_ import read_results

model = 'FSPL'
system_type = "FFP"
true, fit_roman, fit_rr, met_1_roman, met_1_rr, met_2_roman, met_2_rr, met_3_roman, met_3_rr, met1_ratio, met2_ratio, met3_ratio =read_results(system_type, model)
mets_list = [
    (met_1_rr, met_1_roman),
    (met_2_rr, met_2_roman),
    (met_3_rr, met_3_roman),]
true_filtered= true
# ============================================================
# Parámetros de binning adaptativo
# ============================================================
min_count_for_edges = 30   # aprox. eventos por bin al construir bordes
min_count_to_plot   = 20   # mínimo N en denominador para mostrar el bin

# ============================================================
# [NUEVO BLOQUE] seleccionar eventos con RB(piE)<0.5 y NU(piE)<0.5
# met_1_* = RB
# met_3_* = NU
# ============================================================
cond_rr = (met_1_rr['piE'] < 0.5) & (met_3_rr['piE'] < 0.5)
cond_rom = (met_1_roman['piE'] < 0.5) & (met_3_roman['piE'] < 0.5)

rr = true_filtered.merge(
    met_1_rr.loc[cond_rr, ['Source', 'Set']],
    on=['Source', 'Set'],
    how='inner'
)

rom = true_filtered.merge(
    met_1_roman.loc[cond_rom, ['Source', 'Set']],
    on=['Source', 'Set'],
    how='inner'
)
# ============================================================

print("RR events satisfying RB(piE)<0.5 and NU(piE)<0.5:", len(rr))
print("Roman events satisfying RB(piE)<0.5 and NU(piE)<0.5:", len(rom))

# ============================================================
# Figura
# ============================================================
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True, dpi=200)

handles_global = None
cat_labels = {
    'A': 'Roman+Rubin peak',
    'B': 'Rubin-only peak',
    'C': 'Roman-only peak'
}

for ax, cat in zip(axs, ['A', 'B', 'C']):


    # Denominador: todos los eventos de esa categoría
    histbot = true_filtered[true_filtered['peak_flag'] == cat]['tE'].values
    bins_cat = make_adaptive_bins_te(histbot, min_count=min_count_for_edges)
    # histogram del denominador (fondo gris)
    N_bot, _ = np.histogram(histbot, bins=bins_cat)
    centers = np.sqrt(bins_cat[:-1] * bins_cat[1:])

    
    if len(histbot) < max(min_count_for_edges, min_count_to_plot):
        ax.set_xscale('log')
        ax.set_xlabel(r'$t_E$ [day]')
        ax.set_ylim(0, 1)
        ax.set_title(cat_labels[cat], fontsize=9)
        ax.text(
            0.5, 0.5, 'Not enough events',
            transform=ax.transAxes,
            ha='center', va='center', fontsize=8
        )
        continue

    # --------- [NUEVO BLOQUE] bins adaptativos por categoría ------
    
    # --------------------------------------------------------------

    
    # Roman+Rubin
    histtop_rr = rr[rr['peak_flag'] == cat]['tE'].values
    x_rr, frac_rr, err_rr, N_rr = compute_hist_ratio_adaptive(
        histtop_rr,
        histbot,
        bins_cat,
        min_bin_count=min_count_to_plot
    )

    line1 = ax.errorbar(
        x_rr, frac_rr, yerr=err_rr,
        fmt='o-', ms=3,
        label='Roman+Rubin'
    )

    # Roman
    histtop_rom = rom[rom['peak_flag'] == cat]['tE'].values
    x_rom, frac_rom, err_rom, N_rom = compute_hist_ratio_adaptive(
        histtop_rom,
        histbot,
        bins_cat,
        min_bin_count=min_count_to_plot
    )

    line2 = ax.errorbar(
        x_rom, frac_rom, yerr=err_rom,
        fmt='s--', ms=3,alpha=0.6,
        label='Roman'
    )

    if handles_global is None:
        handles_global = [line1, line2]

    ax.set_xscale('log')
    ax.set_title(cat_labels[cat], fontsize=9)
    ax.set_xlabel(r'$t_E$ [day]')
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=8)

    # --------- [NUEVO BLOQUE] anotar N total por categoría --------
    ax.text(
        0.03, 0.5,
        f'N={len(histbot)}',
        transform=ax.transAxes,
        fontsize=7,
        ha='left',
        va='bottom'
    )
    # --------------------------------------------------------------

axs[0].set_ylabel(r'Fraction with $RB(\pi_E)<0.5$ and $NU(\pi_E)<0.5$', fontsize=9)

fig.legend(
    handles_global, ['Roman+Rubin', 'Roman'],
    loc='upper center', ncol=2, frameon=False
)

plt.tight_layout(rect=[0, 0, 1, 0.9])
# plt.savefig("/home/anibal/RB_NU_piE_vstE_adaptive.png", dpi=200, bbox_inches='tight')
plt.show()