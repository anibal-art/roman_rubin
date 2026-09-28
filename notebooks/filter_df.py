import numpy as np
import pandas as pd


def prepare_and_compute_metrics(
    true: pd.DataFrame,
    fit_rr: pd.DataFrame,
    fit_roman: pd.DataFrame,
    model,
    metrics_creator,
    compute_mass_metrics,
    *,
    key_cols=("Source", "Set"),
    roman_band="W149",
    rubin_bands=("u", "g", "r", "i", "z", "y"),
    fillna_value=0,
    # Relevancia (por npeak)
    require_relevance=True,
    # Opcional: crear flag por total_npeak > threshold
    make_total_npeak_flag=False,
    total_npeak_threshold=10,
) -> tuple:
    """
    Prepara los dataframes y computa métricas para:
      - fit_rr (Roman+Rubin)
      - fit_roman (Roman-only)

    Retorna (en este orden):
      fit_rr_f, fit_roman_f, true_f,
      met_1_rr, met_2_rr, met_3_rr,
      met_1_roman, met_2_roman, met_3_roman

    Notas:
      - peak_flag (A/B/C/D) se define usando columnas *_peak
      - flag_relevance se define usando columnas *_npeak
      - Filtrado por inner join en key_cols
    """

    # -------------------------
    # 0) Copias defensivas
    # -------------------------
    true = true.copy()
    fit_rr = fit_rr.copy()
    fit_roman = fit_roman.copy()

    # -------------------------
    # 1) Validaciones mínimas
    # -------------------------
    for k in key_cols:
        if k not in true.columns:
            raise KeyError(f"true no contiene la columna clave '{k}'.")
        if k not in fit_rr.columns:
            raise KeyError(f"fit_rr no contiene la columna clave '{k}'.")
        if k not in fit_roman.columns:
            raise KeyError(f"fit_roman no contiene la columna clave '{k}'.")

    roman_peak_col = f"{roman_band}_peak"
    roman_npeak_col = f"{roman_band}_npeak"

    needed_peak_cols = [roman_peak_col] + [f"{b}_peak" for b in rubin_bands]
    needed_npeak_cols = [roman_npeak_col] + [f"{b}_npeak" for b in rubin_bands]

    missing_peak = [c for c in needed_peak_cols if c not in true.columns]
    missing_npeak = [c for c in needed_npeak_cols if c not in true.columns]
    if missing_peak:
        raise KeyError(f"Faltan columnas *_peak en true: {missing_peak}")
    if missing_npeak:
        raise KeyError(f"Faltan columnas *_npeak en true: {missing_npeak}")

    # -------------------------
    # 2) fillna SOLO donde corresponde (cols de conteo)
    # -------------------------
    count_cols = needed_peak_cols + needed_npeak_cols
    true[count_cols] = true[count_cols].fillna(fillna_value)

    # -------------------------
    # 3) peak_flag A/B/C/D (por *_peak)
    # -------------------------
    roman_detected_peak = true[roman_peak_col] != 0
    rubin_detected_peak = np.zeros(len(true), dtype=bool)
    for b in rubin_bands:
        rubin_detected_peak |= (true[f"{b}_peak"] != 0)

    cat_A = roman_detected_peak & rubin_detected_peak
    cat_B = (~roman_detected_peak) & rubin_detected_peak
    cat_C = roman_detected_peak & (~rubin_detected_peak)
    cat_D = (~roman_detected_peak) & (~rubin_detected_peak)

    true["peak_flag"] = np.select(
        [cat_A, cat_B, cat_C, cat_D],
        ["A", "B", "C", "D"],
        default=None,
    )

    # -------------------------
    # 4) flag_relevance (por *_npeak)
    # -------------------------
    roman_detected_npeak = true[roman_npeak_col] != 0
    rubin_detected_npeak = np.zeros(len(true), dtype=bool)
    for b in rubin_bands:
        rubin_detected_npeak |= (true[f"{b}_npeak"] != 0)

    true["flag_relevance"] = roman_detected_npeak | rubin_detected_npeak

    # -------------------------
    # 5) Opcional: total_npeak y flag_npeak_gtN (bien definido)
    # -------------------------
    if make_total_npeak_flag:
        true["total_npeak"] = true[roman_npeak_col].astype(float)
        for b in rubin_bands:
            true["total_npeak"] += true[f"{b}_npeak"].astype(float)
        true[f"flag_total_npeak_gt{int(total_npeak_threshold)}"] = (
            true["total_npeak"] > float(total_npeak_threshold)
        ).astype(int)

    # -------------------------
    # 6) Filtrado a relevantes (opcional)
    # -------------------------
    if require_relevance:
        true_f = true[true["flag_relevance"]].copy()
    else:
        true_f = true.copy()

    # Si el filtrado deja vacío, devolvemos estructuras vacías coherentes
    if len(true_f) == 0:
        empty = true_f.copy()
        return (
            fit_rr.iloc[0:0].copy(),
            fit_roman.iloc[0:0].copy(),
            true_f,
            empty, empty, empty,
            empty, empty, empty,
        )

    # -------------------------
    # 7) Filtra fits por inner join en keys
    # -------------------------
    keys_df = true_f[list(key_cols)].drop_duplicates()
    fit_rr_f = fit_rr.merge(keys_df, on=list(key_cols), how="inner")
    fit_roman_f = fit_roman.merge(keys_df, on=list(key_cols), how="inner")

    # -------------------------
    # 8) Métricas (recalculadas desde el subset coherente)
    # -------------------------
    met_1_rr, met_2_rr, met_3_rr = metrics_creator(true_f, fit_rr_f, model)
    met_1_roman, met_2_roman, met_3_roman = metrics_creator(true_f, fit_roman_f, model)

    # -------------------------
    # 9) Métricas de masa (¡usar fits filtrados!)
    # -------------------------
    met_1_rr, met_2_rr, met_3_rr = compute_mass_metrics(
        fit_rr_f, true_f, met_1_rr, met_2_rr, met_3_rr, model
    )
    met_1_roman, met_2_roman, met_3_roman = compute_mass_metrics(
        fit_roman_f, true_f, met_1_roman, met_2_roman, met_3_roman, model
    )

    # -------------------------
    # 10) Return: SOLO tus objetos de trabajo
    # -------------------------
    return (
        fit_rr_f, fit_roman_f, true_f,
        met_1_rr, met_2_rr, met_3_rr,
        met_1_roman, met_2_roman, met_3_roman,
    )
