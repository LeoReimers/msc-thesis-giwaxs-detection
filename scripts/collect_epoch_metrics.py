#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect & Plot (Cluster-Version)

Für jeden Run unter /mnt/lustre/work/schreiber/szb559/DINO/output/<RUN>/:
  - sammelt AP_total (aus exp_log_single.txt oder exp_ap_40_polar.txt)
  - sammelt loss_giou (training_stats.txt)
  - schreibt /output/<RUN>/<RUN>_metrics.txt  mit:  epoch ap_total loss_giou

Erzeugt EINEN Vergleichs-Plot (AP links, loss_giou rechts) für alle angegebenen Runs:
  /mnt/lustre/work/schreiber/szb559/DINO/Plots/compare_<runs>.png

Beispiel:
  python collect_epoch_metrics.py --run hold_70 hold_60 hold_50

Optionale Flags:
  --no-take-second-ap   -> nimmt die letzte statt der 2. AP-Messung pro Epoche
  --title "Mein Plot"
  --lr-drops 85 125
  --no-log              -> rechte Achse (loss_giou) linear statt log
  --log-y-min 0.2 --log-y-max 2.0
  --linear-y-min 0.2 --linear-y-max 2.0
"""

import argparse
import os
import re
import ast
import math
import matplotlib.pyplot as plt
from collections import defaultdict

# Pfad-Defaults
OUTPUT_BASE_DEFAULT = "/mnt/lustre/work/schreiber/szb559/DINO/output"
PLOTS_BASE_DEFAULT  = "/mnt/lustre/work/schreiber/szb559/DINO/Plots"

# Histogram-Logging-Defaults
HISTOGRAM_BASE_DEFAULT = "/mnt/lustre/work/schreiber/szb559/DINO/Histograms"
HISTOGRAM_METRIX_NAME  = "Metrix.txt"

# Anzahl der letzten Epochen für die Mittelwerte:
# - separat einstellbar für loss_giou und AP_total
MEAN_LAST_EPOCHS_LOSS = 8
MEAN_LAST_EPOCHS_AP   = 13

# Regex
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
EPOCH_GENERIC_RE = re.compile(r"epoch\s+(\d+)")
EPOCH_TRAINSTATS_RE = re.compile(r"^epoch:\s*(\d+)\s*(\{.*\})\s*$")


# ----------------------------- Parsers ---------------------------------------
def _parse_ap_from_exp_log(path: str, take_second: bool = True):
    """
    Liest AP_total aus exp_log_single.txt.
    Erwartet Zeilen mit 'exp metrics' und 'epoch N', nimmt den letzten Float in der Zeile.
    """
    if not os.path.isfile(path):
        return {}

    ep_to_vals = defaultdict(list)
    current_epoch = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Epoche setzen, falls 'epoch N' in der Zeile vorkommt
            m_ep = EPOCH_GENERIC_RE.search(line)
            if m_ep:
                current_epoch = int(m_ep.group(1))

            # Nur Zeilen mit exp metrics auswerten
            if "exp metrics" in line and current_epoch is not None:
                floats = FLOAT_RE.findall(line)
                if floats:
                    ap_val = float(floats[-1])
                    ep_to_vals[current_epoch].append(ap_val)

    out = {}
    for e in sorted(ep_to_vals):
        vals = ep_to_vals[e]
        if not vals:
            continue
        # 2. Messung nehmen (wie früher), ansonsten die letzte
        out[e] = vals[1] if (take_second and len(vals) >= 2) else vals[-1]
    return out


def _parse_ap_from_polar_txt(path: str):
    """
    Liest AP_total aus exp_ap_40_polar.txt.
    Format: pro Zeile genau ein Float (AP_total), ohne Epoch-Angabe.
    Epoch wird hier als Zeilenindex 0,1,2,... angenommen.
    """
    if not os.path.isfile(path):
        return {}

    ap_map = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                val = float(line)
            except ValueError:
                continue
            # Epoch = Zeilenindex (0-basierend)
            ap_map[idx] = val
    return ap_map


def parse_ap_total(run_dir: str, take_second: bool = True):
    """
    Sucht im Run-Ordner zuerst exp_log_single.txt (altes Format),
    falls dort nichts gefunden wird, exp_ap_40_polar.txt (neues Format).

    Rückgabe: dict {epoch -> ap_total}
    """
    # 1) altes Format: exp_log_single.txt
    exp_log_path = os.path.join(run_dir, "exp_log_single.txt")
    ap_map = _parse_ap_from_exp_log(exp_log_path, take_second=take_second)

    # 2) wenn leer oder Datei nicht vorhanden, neues Format verwenden
    if not ap_map:
        polar_path = os.path.join(run_dir, "exp_ap_40_polar.txt")
        ap_map = _parse_ap_from_polar_txt(polar_path)

    return ap_map



def parse_loss_giou(training_stats_path: str):
    """loss_giou aus training_stats.txt (Dict-String nach 'epoch: N')."""
    if not os.path.isfile(training_stats_path):
        return {}
    out = {}
    with open(training_stats_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = EPOCH_TRAINSTATS_RE.match(line.strip())
            if not m:
                continue
            ep = int(m.group(1))
            dict_str = m.group(2)
            try:
                d = ast.literal_eval(dict_str)
                if "loss_giou" in d:
                    out[ep] = float(d["loss_giou"])
            except Exception:
                continue
    return out


# ----------------------------- Plot helpers ----------------------------------
def floor_for_log(values, ymin):
    """Werte <=0 minimal anheben, damit log-Scale robust bleibt (nur Darstellung)."""
    out, bumped = [], False
    floor_val = ymin * 1.001
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(v)
        elif v <= 0:
            out.append(floor_val)
            bumped = True
        else:
            if v < ymin:
                bumped = True
            out.append(max(v, floor_val))
    return out, bumped


# ----------------------------- Main ------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True,
                    help="Run-Ordner unter output/, z.B. hold_70 hold_60 …")
    ap.add_argument("--output-base", default=OUTPUT_BASE_DEFAULT,
                    help="Basisverzeichnis der Runs")
    ap.add_argument("--plots-base",  default=PLOTS_BASE_DEFAULT,
                    help="Zielordner für Plots")
    ap.add_argument("--outfile-name", default=None,
                    help="Name der je-Run-Ausgabedatei (überschreibt Default '<RUN>_metrics.txt')")
    ap.add_argument("--take-second-ap", dest="take_second_ap",
                    action="store_true", default=True,
                    help="2. AP-Messung pro Epoche verwenden (Default)")
    ap.add_argument("--no-take-second-ap", dest="take_second_ap",
                    action="store_false",
                    help="Stattdessen letzte AP-Messung nehmen")
    ap.add_argument("--title", default=None,
                    help="Optionaler Plot-Titel")
    ap.add_argument("--lr-drops", nargs="*", type=float, default=[],
                    help="Epochen mit LR-Drop (vertikale Linien)")
    ap.add_argument("--log", dest="use_log", action="store_true", default=True,
                    help="rechte Achse (loss_giou) log (Default)")
    ap.add_argument("--no-log", dest="use_log", action="store_false",
                    help="rechte Achse linear")
    # Gewünschter Bereich: 0.2 .. 2 (egal ob log oder linear)
    ap.add_argument("--log-y-min", type=float, default=0.2,
                    help="untere Grenze für log-Skala (loss_giou)")
    ap.add_argument("--log-y-max", type=float, default=2.3,
                    help="obere Grenze für log-Skala (loss_giou)")
    ap.add_argument("--linear-y-min", type=float, default=0.2,
                    help="untere Grenze für lineare Skala (loss_giou)")
    ap.add_argument("--linear-y-max", type=float, default=2.0,
                    help="obere Grenze für lineare Skala (loss_giou)")
    args = ap.parse_args()

    # Zielplot-Ordner
    os.makedirs(args.plots_base, exist_ok=True)

    # Figure/Axes
    plt.figure(figsize=(11, 6.5))
    ax_left  = plt.gca()          # AP links
    ax_right = ax_left.twinx()    # loss_giou rechts

    colors = plt.rcParams['axes.prop_cycle'].by_key().get(
        'color',
        ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
         "#9467bd", "#8c564b"]
    )

    # Für Dateinamen
    run_list_str = ",".join(args.run)
    short_name = "_".join(args.run)
    short_name = (short_name[:80] + "...") if len(short_name) > 80 else short_name

    any_bumped = False
    ap_lines, ap_labels = [], []
    lg_lines, lg_labels = [], []

    # Sammlung für die übersichtliche Mean-Ausgabe (in Aufruf-Reihenfolge)
    mean_loss_rows = []
    mean_ap_rows   = []

    for i, run in enumerate(args.run):
        run_dir = os.path.join(args.output_base, run)
        if not os.path.isdir(run_dir):
            print(f"[WARN] Run-Verzeichnis nicht gefunden: {run_dir}")
            continue

        training_stats_path = os.path.join(run_dir, "training_stats.txt")

        ap_total_map  = parse_ap_total(run_dir, take_second=args.take_second_ap)
        loss_giou_map = parse_loss_giou(training_stats_path)

        epochs = sorted(set(ap_total_map) | set(loss_giou_map))
        if not epochs:
            print(f"[WARN] Keine Epochen gefunden in: {run}")
            continue

        ap_vals = [ap_total_map.get(e, float("nan")) for e in epochs]
        lg_vals = [loss_giou_map.get(e, float("nan")) for e in epochs]

        # Je-Run TXT schreiben
        out_name = args.outfile_name or f"{run}_metrics.txt"
        out_path = os.path.join(run_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as w:
            w.write("epoch ap_total loss_giou\n")
            for e, a, g in zip(epochs, ap_vals, lg_vals):
                w.write(f"{e} {a} {g}\n")
        print(f"[OK] {run}: {len(epochs)} Epochen -> {out_path}")

        # Serienname aus Datei ohne "_metrics.txt"
        _series = os.path.splitext(os.path.basename(out_path))[0]
        if _series.endswith("_metrics"):
            _series = _series[:-8]  # 8 Zeichen ("_metrics") entfernen

        # Mean loss_giou der letzten MEAN_LAST_EPOCHS_LOSS Epochen sammeln (NaNs ignorieren)
        _recent_loss = []
        for v in reversed(lg_vals):
            if isinstance(v, (int, float)) and math.isfinite(v):
                _recent_loss.append(v)
                if len(_recent_loss) >= MEAN_LAST_EPOCHS_LOSS:
                    break
        _mean_loss = (sum(_recent_loss) / len(_recent_loss)) if _recent_loss else float('nan')
        mean_loss_rows.append((_series, _mean_loss))

        # Mean AP_total der letzten MEAN_LAST_EPOCHS_AP Epochen sammeln (NaNs ignorieren)
        _recent_ap = []
        for v in reversed(ap_vals):
            if isinstance(v, (int, float)) and math.isfinite(v):
                _recent_ap.append(v)
                if len(_recent_ap) >= MEAN_LAST_EPOCHS_AP:
                    break
        _mean_ap = (sum(_recent_ap) / len(_recent_ap)) if _recent_ap else float('nan')
        mean_ap_rows.append((_series, _mean_ap))

        # Plotten
        color = colors[i % len(colors)]
        # AP (links)
        line_ap, = ax_left.plot(epochs, ap_vals, linewidth=1.9,
                                color=color, label=f"{run} AP")
        ap_lines.append(line_ap)
        ap_labels.append(f"{run} AP")

        # loss_giou (rechts)
        if args.use_log:
            lg_plot, bumped = floor_for_log(lg_vals, args.log_y_min)
            any_bumped = any_bumped or bumped
        else:
            lg_plot = lg_vals

        line_lg, = ax_right.plot(
            epochs, lg_plot, linestyle="--", linewidth=1.6,
            color=color, alpha=0.9, label=f"{run} loss_giou"
        )
        lg_lines.append(line_lg)
        lg_labels.append(f"{run} loss_giou")

    # LR-Drops
    lr_line = None
    if args.lr_drops:
        for j, drop in enumerate(args.lr_drops):
            lr_line = ax_left.axvline(drop, color="red",
                                      linestyle=":", linewidth=1.2)

    # Rechte Achse skalieren (fix 0.2..2 wie gewünscht)
    if args.use_log:
        ax_right.set_yscale("log")
        ax_right.set_ylim(args.log_y_min, args.log_y_max)  # Default 0.2 .. 2.3
        right_label = "loss_giou [log]"
    else:
        ax_right.set_ylim(args.linear_y_min, args.linear_y_max)  # Default 0.2 .. 2.0
        right_label = "loss_giou"

    # Labels & Titel
    ax_left.set_xlabel("Epoch", fontsize=12)
    ax_left.set_ylabel("AP total", fontsize=12)
    ax_right.set_ylabel(right_label, fontsize=12)
    ax_left.grid(True, linestyle="--", alpha=0.6)
    plot_title = args.title or f"AP vs. loss_giou over epochs: {run_list_str}"
    ax_left.set_title(plot_title, fontsize=14, weight="bold")

    # Legende: erst alle AP, dann alle loss_giou, dann lr_drop
    legend_lines = ap_lines + lg_lines + ([lr_line] if lr_line is not None else [])
    legend_labels = ap_labels + lg_labels + (["lr_drop"] if lr_line is not None else [])
    ax_left.legend(legend_lines, legend_labels, fontsize=9,
                   loc="center right", ncol=2)

    # ---------------- Gruppenstatistik & Ausgabe ----------------
    if mean_loss_rows:
        # Initialisierung der Gruppenstatistik-Container
        group_mean_loss = None
        sem_group_loss  = None
        n_group_loss    = 0

        group_mean_ap = None
        sem_group_ap  = None
        n_group_ap    = 0

        # --- loss_giou ---
        w = max(len(name) for name, _ in mean_loss_rows)
        print(f"\n=== Mean loss_giou (letzte {MEAN_LAST_EPOCHS_LOSS} Epochen) je Messreihe ===")
        for name, val in mean_loss_rows:
            val_str = f"{val:.6f}" if math.isfinite(val) else "n/a"
            print(f"{name:<{w}} : {val_str}")
        print("=" * (w + 15))

        # Gruppenstatistik über alle gültigen Mean(loss_giou)
        valid_vals_loss = [val for _, val in mean_loss_rows if math.isfinite(val)]

        if valid_vals_loss:
            n = len(valid_vals_loss)
            n_group_loss = n
            group_mean_loss = sum(valid_vals_loss) / n

            if n > 1:
                # Stichprobenvarianz mit Division durch (n-1)
                var_group_loss = sum((v - group_mean_loss) ** 2 for v in valid_vals_loss) / (n - 1)
                sd_group_loss  = math.sqrt(var_group_loss)
                sem_group_loss = sd_group_loss / math.sqrt(n)

                print(f"\n=== Gruppenstatistik über {n} Messreihen "
                      f"(Mean(loss_giou) der letzten {MEAN_LAST_EPOCHS_LOSS} Epochen) ===")
                print(f"Gruppenmittelwert      : {group_mean_loss:.6f}")
                print(f"Standardabweichung (s) : {sd_group_loss:.6f}")
                print(f"Standardfehler (SEM)   : {sem_group_loss:.6f}")
            else:
                print(f"\n=== Gruppenstatistik über {n} Messreihe "
                      f"(Mean(loss_giou) der letzten {MEAN_LAST_EPOCHS_LOSS} Epochen) ===")
                print(f"Gruppenmittelwert      : {group_mean_loss:.6f}")
                print("Standardabweichung (s) : n/a (n < 2)")
                print("Standardfehler (SEM)   : n/a (n < 2)")

        # --- AP_total ---
        if mean_ap_rows:
            w_ap = max(len(name) for name, _ in mean_ap_rows)
            print(f"\n=== Mean AP_total (letzte {MEAN_LAST_EPOCHS_AP} Epochen) je Messreihe ===")
            for name, val in mean_ap_rows:
                val_str = f"{val:.6f}" if math.isfinite(val) else "n/a"
                print(f"{name:<{w_ap}} : {val_str}")
            print("=" * (w_ap + 15))

            valid_vals_ap = [val for _, val in mean_ap_rows if math.isfinite(val)]

            if valid_vals_ap:
                n_ap = len(valid_vals_ap)
                n_group_ap = n_ap
                group_mean_ap = sum(valid_vals_ap) / n_ap

                if n_ap > 1:
                    var_group_ap = sum((v - group_mean_ap) ** 2 for v in valid_vals_ap) / (n_ap - 1)
                    sd_group_ap  = math.sqrt(var_group_ap)
                    sem_group_ap = sd_group_ap / math.sqrt(n_ap)

                    print(f"\n=== Gruppenstatistik über {n_ap} Messreihen "
                          f"(Mean(AP_total) der letzten {MEAN_LAST_EPOCHS_AP} Epochen) ===")
                    print(f"Gruppenmittelwert      : {group_mean_ap:.6f}")
                    print(f"Standardabweichung (s) : {sd_group_ap:.6f}")
                    print(f"Standardfehler (SEM)   : {sem_group_ap:.6f}")
                else:
                    print(f"\n=== Gruppenstatistik über {n_ap} Messreihe "
                          f"(Mean(AP_total) der letzten {MEAN_LAST_EPOCHS_AP} Epochen) ===")
                    print(f"Gruppenmittelwert      : {group_mean_ap:.6f}")
                    print("Standardabweichung (s) : n/a (n < 2)")
                    print("Standardfehler (SEM)   : n/a (n < 2)")

        # ---------------- Histogram-Logging in Metrix.txt ----------------
        # Nur wenn:
        #  - mindestens 2 gültige Werte für beide Metriken (n_group_loss > 1, n_group_ap > 1),
        #  - group_mean_* und sem_group_* definiert,
        #  - und alle Run-Namen sich nur in den letzten 3 Zeichen unterscheiden.
        if (group_mean_loss is not None) and (sem_group_loss is not None) and (n_group_loss > 1) \
           and (group_mean_ap is not None) and (sem_group_ap is not None) and (n_group_ap > 1):
            runs = args.run
            if len(runs) >= 2:
                base = runs[0]
                group_name = None
                if len(base) > 3:
                    prefix = base[:-3]  # z.B. "5_32x8_r1" -> "5_32x8"
                    same_pattern = all(
                        (len(r) == len(base)) and r.startswith(prefix)
                        for r in runs
                    )
                    if same_pattern:
                        group_name = prefix

                if group_name is not None:
                    # Ordner und Datei anlegen (falls nötig)
                    hist_dir  = HISTOGRAM_BASE_DEFAULT
                    hist_file = os.path.join(hist_dir, HISTOGRAM_METRIX_NAME)
                    os.makedirs(hist_dir, exist_ok=True)
                    file_exists = os.path.isfile(hist_file)

                    with open(hist_file, "a", encoding="utf-8") as f:
                        if not file_exists:
                            # Headerzeile für sauberes Parsen in anderem Skript
                            f.write("# group_name\tgroup_mean_loss_giou\tSEM_loss_giou\tgroup_mean_AP_total\tSEM_AP_total\n")
                        f.write(f"{group_name}\t{group_mean_loss:.6f}\t{sem_group_loss:.6f}\t"
                                f"{group_mean_ap:.6f}\t{sem_group_ap:.6f}\n")

                    print(f"[OK] Gruppenstatistik für Histogramm nach {hist_file} geschrieben "
                          f"(Gruppe: {group_name})")

    plt.tight_layout()

    # Speichern
    os.makedirs(PLOTS_BASE_DEFAULT, exist_ok=True)
    out_plot = os.path.join(args.plots_base, f"compare_{short_name}.png")
    plt.savefig(out_plot, dpi=300)
    plt.show()
    print(f"[OK] Vergleichs-Plot gespeichert unter: {out_plot}")

    if args.use_log and any_bumped:
        print("[Hinweis] Mindestens ein loss_giou-Wert <= 0 – für Log-Skala minimal angehoben (nur Darstellung).")


if __name__ == "__main__":
    main()
