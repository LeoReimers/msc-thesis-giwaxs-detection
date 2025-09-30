#!/bin/bash
# File: $WORK/DINO/scripts/start_dino.sh
# Robust start: setzt Env, findet letzten Checkpoint, startet Training.
set -euo pipefail

# --------- PARAMS (kommen von außen per ENV) ----------
OUTPUT_DIR="${OUTPUT_DIR:-$WORK/DINO/output/run_$(date +%Y%m%d-%H%M%S)}"
CONFIG_FILE="${CONFIG_FILE:-config/DINO/DINO_4scale_swin.py}"
WINDOW_H="${WINDOW_H:-4}"
WINDOW_W="${WINDOW_W:-4}"
BATCH_SIZE_ENV="${BATCH_SIZE_ENV:-}"

echo "[INFO] OUTPUT_DIR: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Logs ohne Buffer (schneller im File)
export PYTHONUNBUFFERED=1

# ---- Umgebung setzen ----
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate "$WORK/miniconda3/envs/dino"

export CUDA_VER="${CUDA_VER:-11.8}"
export CUDA_HOME="/usr/local/cuda-${CUDA_VER}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"

# Pythonpfade für DINO (ops + repo-root)
export PYTHONPATH="${PYTHONPATH:-}:$WORK/DINO/models/dino/ops"
export PYTHONPATH="$WORK/DINO:${PYTHONPATH}"

cd "$WORK/DINO"

# ---- Resume-Logik ----
RESUME_ARG=()
if [ -f "${OUTPUT_DIR}/checkpoint.pth" ]; then
  echo "[INFO] Found checkpoint: ${OUTPUT_DIR}/checkpoint.pth"
  RESUME_ARG=(--resume "${OUTPUT_DIR}/checkpoint.pth")
elif [ -n "${INITIAL_CHECKPOINT:-}" ] && [ -f "${INITIAL_CHECKPOINT}" ]; then
  echo "[INFO] Using initial checkpoint: ${INITIAL_CHECKPOINT}"
  RESUME_ARG=(--resume "${INITIAL_CHECKPOINT}")
else
  echo "[INFO] No checkpoint found; starting fresh."
fi

if [ -n "${BATCH_SIZE_ENV}" ]; then
  echo "[INFO] Using BATCH_SIZE_ENV=${BATCH_SIZE_ENV}"
fi

# ---- Start: Single-GPU Distributed (standalone) ----
python -m torch.distributed.run \
  --nproc_per_node=1 \
  --standalone \
  main.py \
  --config_file "${CONFIG_FILE}" \
  --options window_size_h="${WINDOW_H}" window_size_w="${WINDOW_W}" \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda \
  "${RESUME_ARG[@]}"
