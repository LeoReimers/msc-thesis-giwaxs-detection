#!/bin/bash
# File: $WORK/DINO/scripts/start_dino.sh
# Robust start: sets env, resumes from checkpoint if present, starts training.
set -euo pipefail

# Make sure WORK exists (safety net; normally set by cluster)
: "${WORK:=$HOME}"

# LD_LIBRARY_PATH safe init (for set -u)
: "${LD_LIBRARY_PATH:=}"

# --------- PARAMS (from ENV or defaults) ----------
OUTPUT_DIR="${OUTPUT_DIR:-$WORK/DINO/output/run_$(date +%Y%m%d-%H%M%S)}"
CONFIG_FILE="${CONFIG_FILE:-config/DINO/DINO_4scale_swin.py}"
WINDOW_H="${WINDOW_H:-4}"
WINDOW_W="${WINDOW_W:-4}"
BATCH_SIZE_ENV="${BATCH_SIZE_ENV:-}"

echo "[INFO] OUTPUT_DIR: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Unbuffered logs
export PYTHONUNBUFFERED=1

# ---- SLURM / GPU quick diagnostics ----
echo "[INFO] host=$(hostname)"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "[INFO] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "[WARN] nvidia-smi not available"

# ---- Activate conda env ----
source "$WORK/miniconda3/etc/profile.d/conda.sh"
conda activate "$WORK/miniconda3/envs/dino"

# ---- CUDA env ----
export CUDA_VER="${CUDA_VER:-11.8}"
export CUDA_HOME="/usr/local/cuda-${CUDA_VER}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

# ---- PYTHONPATH for DINO (ops + repo-root) ----
export PYTHONPATH="${PYTHONPATH:-}:$WORK/DINO/models/dino/ops"
export PYTHONPATH="$WORK/DINO:${PYTHONPATH}"

cd "$WORK/DINO"

# ---- Resume logic ----
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

# ---- Base options ----
SEED="${SEED:-42}"
: "${CFG_OPTS:=}"   # e.g. epochs=120 lr=...
: "${PY_ARGS:=}"    # extra flags, e.g. --eval

# Options merged into config (array for safe quoting)
OPTS=()
if [ -n "${EPOCHS:-}" ]; then
  OPTS+=("epochs=${EPOCHS}")
fi
if [ -n "${BATCH_SIZE_ENV:-}" ]; then
  OPTS+=("batch_size=${BATCH_SIZE_ENV}")
fi

echo "[INFO] Using LR schedule from config (no extra CLI scheduler args)."

python -m torch.distributed.run \
  --nproc_per_node=1 \
  --standalone \
  main.py \
    --config_file "${CONFIG_FILE}" \
    --options \
      "window_size_h=${WINDOW_H}" \
      "window_size_w=${WINDOW_W}" \
      ${CFG_OPTS} \
      "${OPTS[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda \
    --seed "${SEED}" \
    "${RESUME_ARG[@]}" \
    ${PY_ARGS}
