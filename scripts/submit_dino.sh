#!/bin/bash
# File: $WORK/DINO/scripts/submit_dino.sh
# Robust SLURM submit wrapper for DINO:
# - excludes bad nodes by default
# - ensures srun step gets GPU allocation
# - keeps SLURM stdout/err AND writes a full log
set -euo pipefail

# Optional debug for this wrapper only
if [ "${DEBUG_SUBMIT:-0}" = "1" ]; then
  set -x
  trap 'echo "[ERR] line=$LINENO cmd=$BASH_COMMAND exit=$?"; echo "host=$(hostname) pwd=$(pwd)"; env | sort' ERR
fi

OUTPUT_DIR="${1:-}"
if [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: submit_dino.sh <OUTPUT_DIR> [WINDOW_H] [WINDOW_W]"
  exit 1
fi

WINDOW_H="${2:-4}"
WINDOW_W="${3:-4}"

# -------- SLURM defaults (overridable via env) --------
PARTITION="${PARTITION:-a100-galvani,a100-fat-galvani,a100-preemptable-galvani}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
TIME_LIMIT="${TIME_LIMIT:-40:00:00}"
GPUS="${GPUS:-1}"
JOB_NAME="${JOB_NAME:-dino_swin}"

# Exclude problematic node(s) by default (can override: EXCLUDE_NODES="")
EXCLUDE_NODES="${EXCLUDE_NODES:-galvani-cn223}"
# Optional: force a node list if you want (NODELIST="galvani-cn226")
NODELIST="${NODELIST:-}"

INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-}"
KEEP_LINES="${KEEP_LINES:-200}"

mkdir -p "${OUTPUT_DIR}"
echo "[INFO] Submitting SLURM job... Output & logs: ${OUTPUT_DIR}"
echo "[INFO] Partition=${PARTITION} GPUs=${GPUS} CPUs=${CPUS} MEM=${MEM} TIME=${TIME_LIMIT}"
if [ -n "${EXCLUDE_NODES}" ]; then
  echo "[INFO] Excluding nodes: ${EXCLUDE_NODES}"
fi
if [ -n "${NODELIST}" ]; then
  echo "[INFO] Forcing nodelist: ${NODELIST}"
fi

# Inner sbatch script (kept in OUTPUT_DIR for reproducibility)
SBATCH_FILE="${OUTPUT_DIR}/sbatch_inner.sh"

cat > "$SBATCH_FILE" << 'EOS'
#!/bin/bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:?}"
WINDOW_H="${WINDOW_H:?}"
WINDOW_W="${WINDOW_W:?}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-}"
KEEP_LINES="${KEEP_LINES:-200}"
GPUS="${GPUS:-1}"

LOG_FULL="${OUTPUT_DIR}/slurm-${SLURM_JOB_ID}.full.out"
LOG_SHORT="${OUTPUT_DIR}/slurm-${SLURM_JOB_ID}.out"

# Mirror all output to LOG_FULL while still letting SLURM capture stdout/err
exec > >(tee -a "$LOG_FULL") 2>&1

echo "[INFO] Starting DINO job on node $(hostname) with SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "[INFO] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR} WINDOW_H=${WINDOW_H} WINDOW_W=${WINDOW_W} GPUS=${GPUS}"

# IMPORTANT: ensure the *step* gets GPU allocation too
srun --unbuffered --gres="gpu:${GPUS}" bash "$WORK/DINO/scripts/start_dino.sh"
EXIT_CODE=$?

echo "[INFO] DINO job finished with exit code ${EXIT_CODE}, truncating logs to last ${KEEP_LINES} lines..."

# Create a short log copy (optional convenience)
if [ -f "$LOG_FULL" ]; then
  tail -n "$KEEP_LINES" "$LOG_FULL" > "$LOG_SHORT" || true
fi

exit "$EXIT_CODE"
EOS

chmod +x "$SBATCH_FILE"

# Submit job
sbatch \
  --job-name="${JOB_NAME}" \
  --partition="${PARTITION}" \
  ${NODELIST:+--nodelist="${NODELIST}"} \
  ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
  --gres="gpu:${GPUS}" \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --time="${TIME_LIMIT}" \
  --output="${OUTPUT_DIR}/sbatch-%j.out" \
  --error="${OUTPUT_DIR}/sbatch-%j.err" \
  --export=ALL,OUTPUT_DIR="${OUTPUT_DIR}",WINDOW_H="${WINDOW_H}",WINDOW_W="${WINDOW_W}",INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT}",KEEP_LINES="${KEEP_LINES}",GPUS="${GPUS}" \
  "$SBATCH_FILE"