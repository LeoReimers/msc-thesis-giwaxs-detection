#!/bin/bash
set -euo pipefail

: "${WORK:=$HOME}"

# --------- SETUP ----------
# Wir nutzen das vom Submit-Skript übergebene OUTPUT_DIR
OUTPUT_DIR="${OUTPUT_DIR:-$WORK/DINO/output/debug_run}"
CONFIG_FILE="config/DINO/DINO_4scale_swin.py" # Dein Default Hardcode

echo "[INFO] Starting Single-GPU Training"
echo "[INFO] Output Dir: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

export PYTHONUNBUFFERED=1

# ---- Environment Setup ----
# (Pfade ggf. anpassen, falls abweichend von deinem Upload)
set +u  # <--- HIER: Strenge Prüfung AUSSCHALTEN
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate "$WORK/miniconda3/envs/dino"
set -u  # <--- HIER: Strenge Prüfung wieder EINSCHALTEN

export CUDA_HOME="/usr/local/cuda-11.8"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"

# PYTHONPATH
export PYTHONPATH="${PYTHONPATH:-}:$WORK/DINO/models/dino/ops"
export PYTHONPATH="$WORK/DINO:${PYTHONPATH}"

cd "$WORK/DINO"

# ---- START ----
# Kein torch.distributed.run, einfach python!
# Wir übergeben NUR output_dir und config, damit Resume klappt.
# Alles andere kommt aus defaults in main.py oder der config Datei.

python main.py \
  --config_file "${CONFIG_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda

# HINWEIS:
# Die main.py prüft selbstständig:
# if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
#     args.resume = ...
# Daher müssen wir --resume hier NICHT explizit übergeben.