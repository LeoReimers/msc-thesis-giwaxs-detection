#!/bin/bash
# File: $WORK/DINO/scripts/start_dino.sh
# Robust start: setzt Env, findet letzten Checkpoint, startet Training.
set -euo pipefail

# --------- PARAMS (kommen von au�en per ENV) ----------
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

# Pythonpfade f�r DINO (ops + repo-root)
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
# ---- Start: Single-GPU Distributed (standalone) ----
LR_T0="${LR_T0:-10}"
LR_TMULT="${LR_TMULT:-2}"
LR_MIN="${LR_MIN:-1e-7}"
SEED="${SEED:-42}"
: "${FLATCOS:=1}"
: "${LR_WARMUP_EPOCHS:=3}"
: "${LR_HOLD_EPOCHS:=97}"
: "${LR_COSINE_EPOCHS:=40}"
: "${LR_WARMUP_START_FACTOR:=0.3}"
: "${CFG_OPTS:=}"
: "${PY_ARGS:=}"


# Options, die in die Config gemergt werden (als Array, damit sauber gequotet)
OPTS=()
if [ -n "${EPOCHS:-}" ]; then
  OPTS+=("epochs=${EPOCHS}")
fi
if [ -n "${BATCH_SIZE_ENV:-}" ]; then
  OPTS+=("batch_size=${BATCH_SIZE_ENV}")
fi

echo "[INFO] OUTPUT_DIR: ${OUTPUT_DIR}"
echo "[INFO] Using Flat→Hold→Cosine schedule (warmup=${LR_WARMUP_EPOCHS}, hold=${LR_HOLD_EPOCHS}, cosine=${LR_COSINE_EPOCHS}, lr_min=${LR_MIN}, seed=${SEED})"

: "${CFG_OPTS:=}"   # key=val Paare für --options
: "${PY_ARGS:=}"    # zusätzliche Python-Flags (--lr_mode, --flatcos, ...)

SCHED_OPTS=()
if [ "$FLATCOS" = "1" ]; then
  SCHED_OPTS=( --flatcos
               --lr_warmup_epochs "$LR_WARMUP_EPOCHS"
               --lr_hold_epochs   "$LR_HOLD_EPOCHS"
               --lr_cosine_epochs "$LR_COSINE_EPOCHS"
               --lr_warmup_start_factor "$LR_WARMUP_START_FACTOR" )
fi

python -m torch.distributed.run \
  --nproc_per_node=1 \
  --standalone \
  main.py \
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