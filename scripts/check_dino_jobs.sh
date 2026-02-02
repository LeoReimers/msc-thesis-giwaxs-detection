#!/usr/bin/env bash
# Prüft Status der DINO-Jobs und folgt den Logs, sobald sie existieren.

set -euo pipefail

# Deine JobIDs von gestern
JOBS=("1789354" "1789355")

# Zugehörige Output-Ordner
OUTDIRS=(
  "$WORK/DINO/output/Patch_size_4x4_p4_r1"
  "$WORK/DINO/output/Patch_size_4x4_p4_r3"
)

echo "=== Aktueller SLURM-Status ==="
squeue -u $USER -o "%.18i %.9P %.22j %.2t %.10M %R"
echo

for i in "${!JOBS[@]}"; do
  jid="${JOBS[$i]}"
  odir="${OUTDIRS[$i]}"
  logfile=$(ls -t "$odir"/slurm-${jid}.out 2>/dev/null | head -n1 || true)

  echo "--- Job $jid (Ordner: $odir) ---"
  if [[ -z "$logfile" ]]; then
    echo "Noch kein Logfile für Job $jid gefunden. (Job evtl. noch PENDING)"
  else
    echo "Starte tail -F auf: $logfile"
    echo "Zum Abbrechen: Ctrl+C"
    tail -n 50 -F "$logfile"
  fi
  echo
done
