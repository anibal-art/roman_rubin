#!/bin/bash
#SBATCH --partition=local
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         
#SBATCH --hint=nomultithread        
#SBATCH --exclusive                 
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@90
#SBATCH --mail-type=FAIL,END


set -euo pipefail

echo "CWD     : $(pwd)"
echo "CFG_PATH: ${CFG_PATH}"

if [[ -f "${CFG_PATH}.SHA256" ]]; then
  EXPECTED=$(cat "${CFG_PATH}.SHA256")
  ACTUAL=$(sha256sum "${CFG_PATH}" | awk '{print $1}')
  [[ "$EXPECTED" == "$ACTUAL" ]] || { echo "ERROR: checksum mismatch"; exit 1; }
fi

# 1 hilo por librería numérica (padre + hijos)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Fijar binding a cores físicos
export SLURM_CPU_BIND=cores

# Scratch local para reducir I/O en red
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export MPLCONFIGDIR="$TMPDIR"   # por si Matplotlib crea cache

# Log útil de auditoría
lscpu | egrep 'Model name|Socket|Core|Thread|CPU\(s\)|MHz' || true
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-NA}"
echo "SLURM_CPU_BIND=${SLURM_CPU_BIND:-NA}"

srun --cpu-bind=cores -u python -u run_paralelo_simfit.py --config "${CFG_PATH}"

