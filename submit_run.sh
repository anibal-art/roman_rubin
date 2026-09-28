#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Config base
# ============================================================

CFG_SRC="${1:-sim_fit_config_file.json}"

if [[ ! -f "$CFG_SRC" ]]; then
  echo "ERROR: no se encontró config: $CFG_SRC" >&2
  exit 1
fi

# ============================================================
# Leer campos del JSON de forma robusta usando Python
# ============================================================

read -r PATH_STORAGE MODEL SYSTEM_TYPE N_DB_FILE NEVENTS N_TR <<< "$(
python - <<EOF
import json
from pathlib import Path

cfg_path = Path("$CFG_SRC")

with cfg_path.open() as f:
    cfg = json.load(f)

required = ["path_storage", "model", "system_type", "n_db_file", "Nevents", "N_tr"]
missing = [k for k in required if k not in cfg]

if missing:
    raise SystemExit(f"ERROR: faltan claves en config: {missing}")

print(
    cfg["path_storage"],
    cfg["model"],
    cfg["system_type"],
    cfg["n_db_file"],
    cfg["Nevents"],
    cfg["N_tr"],
)
EOF
)"

# ============================================================
# Directorios de configs congelados y logs
# ============================================================

CONFIG_DIR="${PATH_STORAGE}/config_file"
LOG_DIR="${PATH_STORAGE}/slurm_logs"

mkdir -p "$CONFIG_DIR"
mkdir -p "$LOG_DIR"

# Verificar permisos antes de enviar
if [[ ! -w "$CONFIG_DIR" ]]; then
  echo "ERROR: no tengo permiso de escritura en CONFIG_DIR=$CONFIG_DIR" >&2
  exit 1
fi

if [[ ! -w "$LOG_DIR" ]]; then
  echo "ERROR: no tengo permiso de escritura en LOG_DIR=$LOG_DIR" >&2
  exit 1
fi

# ============================================================
# Congelar config con timestamp
# ============================================================

STAMP="$(date +'%Y%m%d_%H%M%S')"
CFG_BASE="$(basename "$CFG_SRC" .json)"

CFG_DST="${CONFIG_DIR}/${CFG_BASE}_${SYSTEM_TYPE}_${MODEL}_chunk${N_DB_FILE}_${STAMP}.json"

cp -a "$CFG_SRC" "$CFG_DST"

HASH="$(sha256sum "$CFG_DST" | awk '{print $1}')"
echo "$HASH  $(basename "$CFG_DST")" > "${CFG_DST}.SHA256"

chmod a-w "$CFG_DST" "${CFG_DST}.SHA256"

# ============================================================
# Nombre del job y logs
# ============================================================

JOBNAME="${SYSTEM_TYPE}_${MODEL}_chunk${N_DB_FILE}"

LOG_OUT="${LOG_DIR}/${JOBNAME}_${STAMP}_%j.out"
LOG_ERR="${LOG_DIR}/${JOBNAME}_${STAMP}_%j.err"

# ============================================================
# Enviar a SLURM
# ============================================================

# ============================================================
# Enviar a SLURM
# ============================================================

JOB_SCRIPT="slurm_files/job_simfit.slurm"

if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "ERROR: no se encontró JOB_SCRIPT=$JOB_SCRIPT" >&2
  exit 1
fi

sbatch --chdir="$(pwd)" \
       --job-name="$JOBNAME" \
       --output="$LOG_OUT" \
       --error="$LOG_ERR" \
       --export=ALL,CFG_PATH="$CFG_DST" \
       "$JOB_SCRIPT"

echo "Enviado."
echo "  CFG_SRC      : $CFG_SRC"
echo "  CFG_PATH     : $CFG_DST"
echo "  SHA256       : $HASH"
echo "  PATH_STORAGE : $PATH_STORAGE"
echo "  SYSTEM_TYPE  : $SYSTEM_TYPE"
echo "  MODEL        : $MODEL"
echo "  N_DB_FILE    : $N_DB_FILE"
echo "  NEVENTS      : $NEVENTS"
echo "  N_TR         : $N_TR"
echo "  JOBNAME      : $JOBNAME"
echo "  LOG_OUT      : $LOG_OUT"
echo "  LOG_ERR      : $LOG_ERR"
