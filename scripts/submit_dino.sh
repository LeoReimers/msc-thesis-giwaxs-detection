#!/bin/bash
# File: $WORK/DINO/scripts/submit_dino.sh
# Reicht einen robusten SLURM-Job ein, der Preemption/Crashes abfängt und neu startet.
set -euo pipefail

# ------------------ Pflicht-/Standard-Parameter --------------------
OUTPUT_DIR="${1:-}"
if [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: submit_dino.sh <OUTPUT_DIR> [WINDOW_H] [WINDOW_W]"
  exit 1
fi
WINDOW_H="${2:-4}"
WINDOW_W="${3:-4}"

# Ressourcen (ggf. via ENV überschreiben)
PARTITION="${PARTITION:-a100-preemptable-galvani}"
CPUS="${CPUS:-12}"
MEM="${MEM:-256G}"
TIME_LIMIT="${TIME_LIMIT:-70:00:00}"
GPUS="${GPUS:-1}"
JOB_NAME="${JOB_NAME:-dino_swin}"

# Optionale Start-Checkpoint-Variable
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-}"

# Wie viele Zeilen am Ende behalten? (0 = gar keine Logdatei behalten)
KEEP_LINES="${KEEP_LINES:-100}"

# ------------------ SBATCH Script dynamisch erstellen --------------
mkdir -p "${OUTPUT_DIR}"
SBATCH_FILE="$(mktemp)"

cat > "$SBATCH_FILE" <<'EOS'
#!/bin/bash
# Innerer Job-Skript (läuft auf dem Knoten)
set -euo pipefail

# Variablen kommen aus sbatch --export
OUTPUT_DIR="${OUTPUT_DIR:?}"
WINDOW_H="${WINDOW_H:?}"
WINDOW_W="${WINDOW_W:?}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-}"
KEEP_LINES="${KEEP_LINES:-100}"

LOG_DIR="${OUTPUT_DIR}"
mkdir -p "$LOG_DIR"
LOG_FULL="${LOG_DIR}/slurm-${SLURM_JOB_ID}.full.out"
LOG_SHORT="${LOG_DIR}/slurm-${SLURM_JOB_ID}.out"

cleanup_logs() {
  if [ "${KEEP_LINES}" = "0" ]; then
    rm -f "$LOG_FULL" "$LOG_SHORT" || true
    return
  fi
  if [ -f "$LOG_FULL" ]; then
    tail -n "$KEEP_LINES" "$LOG_FULL" > "$LOG_SHORT" || true
    rm -f "$LOG_FULL" || true
  fi
}

trap 'echo "[INFO] Truncating logs..."; cleanup_logs' EXIT TERM INT

# Ab hier alles in unsere eigene Logdatei
exec >"$LOG_FULL" 2>&1

export OUTPUT_DIR WINDOW_H WINDOW_W INITIAL_CHECKPOINT

TRIES=0
MAX_TRIES=5
while true; do
  echo "[INFO] Try $((TRIES+1))/${MAX_TRIES} starting at $(date)"
  EXIT_CODE=0
  srun --unbuffered bash "$WORK/DINO/scripts/start_dino.sh" || EXIT_CODE=$?

  if [ "$EXIT_CODE" -eq 0 ]; then
    echo "[INFO] Training finished successfully."
    break
  else
    echo "[ERROR] Training crashed with exit code ${EXIT_CODE}."
  fi

  TRIES=$((TRIES+1))
  if [ "$TRIES" -ge "$MAX_TRIES" ]; then
    echo "[ERROR] Max retries reached. Exiting."
    exit 1
  fi

  echo "[INFO] Sleeping 60s before respawn…"
  sleep 60
done
EOS

# ------------------ Job absenden (Parameter direkt an sbatch) ------
echo "[INFO] Submitting SLURM job… Output & logs: ${OUTPUT_DIR}"


sbatch \
  --job-name="${JOB_NAME}" \
  --partition="${PARTITION}" \
  --gres="gpu:${GPUS}" \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --time="${TIME_LIMIT}" \
  --signal=B:USR1@120 \
  --output=/dev/null \
  --error=/dev/null \
  --requeue \
  --export=ALL,OUTPUT_DIR="${OUTPUT_DIR}",WINDOW_H="${WINDOW_H}",WINDOW_W="${WINDOW_W}",INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT}",KEEP_LINES="${KEEP_LINES}" \
  "$SBATCH_FILE"

rm -f "$SBATCH_FILE"

