#!/usr/bin/env bash

# [MOVIDO ARRIBA] modo estricto
set -euo pipefail

# [SIN CAMBIOS DE PATH] directorio donde guardar los configs
# CONFIG_DIR="/share/storage3/rubin/microlensing/romanrubin/config_file"
CONFIG_DIR="/mnt/almacenamiento/config_files"
mkdir -p "$CONFIG_DIR"

CFG_SRC="sim_fit_config_file.json"
[[ ! -f "$CFG_SRC" ]] && { echo "No se encontró $CFG_SRC"; exit 1; }

# ===== generar copia con timestamp =====
STAMP=$(date +'%Y%m%d_%H%M%S')
CFG_BASE="$(basename "$CFG_SRC" .json)"

# [CAMBIO IMPORTANTE] ahora guardamos el config en CONFIG_DIR
CFG_DST="${CONFIG_DIR}/${CFG_BASE}_${STAMP}.json"

cp -a "$CFG_SRC" "$CFG_DST"
HASH=$(sha256sum "$CFG_DST" | awk '{print $1}')
echo "$HASH" > "${CFG_DST}.SHA256"
chmod a-w "$CFG_DST" "${CFG_DST}.SHA256"

# ===== extraer model y n_db_file del JSON (sin jq) =====
# [CAMBIO] antes usabas system_type; ahora usamos "model"
MODEL=$(grep -E '"model"' "$CFG_DST" | head -n1 | sed -E 's/.*"model"\s*:\s*"([^"]+)".*/\1/')
N_DB_FILE=$(grep -E '"n_db_file"' "$CFG_DST" | head -n1 | sed -E 's/.*"n_db_file"\s*:\s*([0-9]+).*/\1/')

# Validar extracción
if [[ -z "$MODEL" || -z "$N_DB_FILE" ]]; then
  echo "ERROR: no se pudieron leer model o n_db_file del config"
  echo "MODEL='$MODEL' N_DB_FILE='$N_DB_FILE'"
  exit 1
fi

# ===== nombre automático del job =====
# [CAMBIO] ahora el job se llama, p.ej., PSPL_1
JOBNAME="${MODEL}_${N_DB_FILE}"

# ===== directorio de logs =====
LOG_DIR="/mnt/almacenamiento/"
mkdir -p "$LOG_DIR"

# ===== enviar a SLURM =====
# [CAMBIO IMPORTANTE] nombres de logs fijos: PSPL_1.out / PSPL_1.err
sbatch --chdir="$(pwd)" \
       --job-name="$JOBNAME" \
       --output="${LOG_DIR}/${JOBNAME}.out" \
       --error="${LOG_DIR}/${JOBNAME}.err" \
       --export=ALL,CFG_PATH="${CFG_DST}" \
       job_simfit.slurm

echo "Enviado."
echo "  CFG_PATH : ${CFG_DST}"
echo "  JOBNAME  : ${JOBNAME}"
echo "  LOG_OUT  : ${LOG_DIR}/${JOBNAME}.out"
echo "  LOG_ERR  : ${LOG_DIR}/${JOBNAME}.err"


