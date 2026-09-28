#!/bin/bash
#SBATCH --partition=local
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@90
#SBATCH --mail-type=FAIL,END

set -euo pipefail

echo "============================================================"
echo "SLURM job info"
echo "============================================================"
echo "HOST    : $(hostname)"
echo "CWD     : $(pwd)"
echo "JOB_ID  : ${SLURM_JOB_ID:-NA}"
echo "CFG_PATH: ${CFG_PATH:-UNSET}"
echo "============================================================"

if [[ -z "${CFG_PATH:-}" ]]; then
  echo "ERROR: CFG_PATH is not defined." >&2
  exit 1
fi

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "ERROR: config file does not exist: ${CFG_PATH}" >&2
  exit 1
fi

# ------------------------------------------------------------
# Verify frozen config checksum
# ------------------------------------------------------------

if [[ -f "${CFG_PATH}.SHA256" ]]; then
  EXPECTED="$(awk '{print $1}' "${CFG_PATH}.SHA256")"
  ACTUAL="$(sha256sum "${CFG_PATH}" | awk '{print $1}')"

  echo "EXPECTED SHA256: ${EXPECTED}"
  echo "ACTUAL SHA256  : ${ACTUAL}"

  if [[ "${EXPECTED}" != "${ACTUAL}" ]]; then
    echo "ERROR: checksum mismatch for ${CFG_PATH}" >&2
    exit 1
  fi
else
  echo "WARNING: checksum file not found: ${CFG_PATH}.SHA256"
fi

# ------------------------------------------------------------
# Numerical-library thread control
# ------------------------------------------------------------

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# ------------------------------------------------------------
# simfit_parallel_runner configuration
# ------------------------------------------------------------

export SIMFIT_RETRIES="${SIMFIT_RETRIES:-1}"
export SIMFIT_MP_CONTEXT="${SIMFIT_MP_CONTEXT:-spawn}"
export SIMFIT_SKIP_MODE="${SIMFIT_SKIP_MODE:-none}"
export SIMFIT_PROGRESS_EVERY="${SIMFIT_PROGRESS_EVERY:-100}"

# Optional:
# export SIMFIT_MAX_WORKERS="${SLURM_CPUS_PER_TASK}"

# ------------------------------------------------------------
# CPU binding and local scratch
# ------------------------------------------------------------

export SLURM_CPU_BIND=cores

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export MPLCONFIGDIR="${TMPDIR}/mplconfig_${SLURM_JOB_ID:-manual}"
mkdir -p "${MPLCONFIGDIR}"

echo "============================================================"
echo "Resource audit"
echo "============================================================"
lscpu | egrep 'Model name|Socket|Core|Thread|CPU\(s\)|MHz' || true
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-NA}"
echo "SLURM_CPU_BIND=${SLURM_CPU_BIND:-NA}"
echo "TMPDIR=${TMPDIR}"
echo "MPLCONFIGDIR=${MPLCONFIGDIR}"
echo "SIMFIT_RETRIES=${SIMFIT_RETRIES}"
echo "SIMFIT_MP_CONTEXT=${SIMFIT_MP_CONTEXT}"
echo "SIMFIT_SKIP_MODE=${SIMFIT_SKIP_MODE}"
echo "SIMFIT_PROGRESS_EVERY=${SIMFIT_PROGRESS_EVERY}"
echo "============================================================"

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

srun --cpu-bind=cores -u python -u launch_simfit_from_config.py --config "${CFG_PATH}"
