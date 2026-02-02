#!/bin/bash
# File: $WORK/DINO/scripts/submit_dino.sh
set -euo pipefail

OUTPUT_DIR="${1:-}"
if [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: submit_dino.sh <OUTPUT_DIR>"
  exit 1
fi

# Defaults
PARTITION="${PARTITION:-a100-galvani,a100-fat-galvani}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
GPUS=1  # Fest auf 1 GPU
JOB_NAME="${JOB_NAME:-dino_single}"

mkdir -p "${OUTPUT_DIR}"

SBATCH_FILE="${OUTPUT_DIR}/sbatch_inner.sh"

cat > "$SBATCH_FILE" << 'EOS'
#!/bin/bash
set -euo pipefail

# Variablen aus dem äußeren Skript übernehmen
OUTPUT_DIR="${OUTPUT_DIR:?}"
export OUTPUT_DIR

LOG_FULL="${OUTPUT_DIR}/slurm-${SLURM_JOB_ID}.full.out"

# Output Redirection
exec > >(tee -a "$LOG_FULL") 2>&1

echo "[INFO] Job running on $(hostname)"

# Starte das Trainingsskript
srun --unbuffered --gres="gpu:1" bash "$WORK/DINO/scripts/start_dino.sh"
EOS

chmod +x "$SBATCH_FILE"

sbatch \
  --job-name="${JOB_NAME}" \
  --partition="${PARTITION}" \
  --gres="gpu:${GPUS}" \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --time="${TIME_LIMIT}" \
  --output="${OUTPUT_DIR}/sbatch-%j.out" \
  --error="${OUTPUT_DIR}/sbatch-%j.err" \
  --export=ALL,OUTPUT_DIR="${OUTPUT_DIR}" \
  "$SBATCH_FILE"