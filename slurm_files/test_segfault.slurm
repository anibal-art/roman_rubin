#!/usr/bin/env bash
#SBATCH --job-name=FFP
#SBATCH --partition=milliways-int
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40           # igual que tu script original
#SBATCH --hint=multithread             # como tenías
#SBATCH --mem=250G
#SBATCH --time=02:00:00                # ajusta si querés
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=anibalvarela134@gmail.com

# === LOGS SEPARADOS ===
#SBATCH -o FFP.%j.out
#SBATCH -e FFP.%j.err

# === OPCIONAL (recomendado mientras depuras): aísla el nodo para no afectar a otros ===
# #SBATCH --exclusive

set -euo pipefail

echo "[ENV] $(date)"
echo "HOSTNAME: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

# === Activa tu conda env (ajusta la ruta si difiere) ===
# Si usás miniconda en $HOME:
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
# O si usás otra instalación de conda, ajustá esa ruta arriba.
conda activate pyLIMA_test

# === Medidas anti-mezclas y estabilidad ===
export PYTHONNOUSERSITE=1       # evita que ~/.local se cuele en sys.path
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ulimit -c unlimited             # habilita core dumps (si el cluster lo permite)

# === Backtrace inmediato de C al stderr ante SIGSEGV (si existe libSegFault) ===
if [[ -f /lib64/libSegFault.so ]]; then
  export LD_PRELOAD=/lib64/libSegFault.so
  export SEGFAULT_SIGNALS=all
  echo "[INFO] LD_PRELOAD=libSegFault habilitado"
fi

# === Info rápida del entorno (queda en .out) ===
python - << 'PY'
import sys, site, numpy as np
print("[PYTHON]", sys.executable, sys.version.split()[0])
print("[NUMPY ]", np.__version__, np.__file__)
print("[USER_SITE_ENABLED]", site.ENABLE_USER_SITE)
try:
    import VBBinaryLensing as VBB, os, glob
    print("[VBB   ]", VBB.__file__)
    pkg = os.path.dirname(VBB.__file__)
    sos = glob.glob(os.path.join(pkg, "**", "*.so"), recursive=True)
    print("[VBB_SO]", sos[:1])
except Exception as e:
    print("[VBB IMPORT ERROR]", repr(e))
PY

# === Habilita faulthandler para capturar stack de Python en stderr si hay SIGSEGV ===
# (inyectamos esto y luego ejecutamos tu script)
cat > _faulthandler_wrapper.py << 'PY'
import faulthandler, signal, runpy, os, sys
faulthandler.register(signal.SIGSEGV)
faulthandler.enable(all_threads=True)
target = "run_paralelo_set1.py"  # <-- tu script
if not os.path.exists(target):
    print(f"[ERROR] No se encuentra {target}", file=sys.stderr); sys.exit(2)
runpy.run_path(target, run_name="__main__")
PY

echo "[RUN] $(date)"
python _faulthandler_wrapper.py
echo "[DONE] $(date)"
