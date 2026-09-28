#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --mem=250G
#SBATCH --partition=milliways-int
#SBATCH --job-name="proc"
#SBATCH --output=out_%j.log
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=anibalvarela134@gmail.com

export BLOCK_SIZE=200
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

# Parámetros de análisis
export NO_IDEMPOTENCY_SCAN=1
export MAX_WORKERS=40
export BLOCK_SIZE=1000
export REPORT_INTERVAL_SEC=120
export BLOCK_FLUSH=8000

python run_class_analysis.py

