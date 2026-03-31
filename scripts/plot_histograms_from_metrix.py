#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot Histogram from MetrixNew.txt

Reads from:
  /mnt/lustre/work/schreiber/szb559/DINO/Histograms/MetrixNew.txt

Expected format:
  # group_name\tgroup_mean_loss_giou\tSEM_loss_giou\tgroup_mean_AP_total\tSEM_AP_total\tbest_AP_total_all_runs
  <name> <mean_loss> <sem_loss> <mean_ap> <sem_ap> <best_ap>

Example:
  python plot_histograms_from_metrix_new.py --run NEW_8x8 NEW_8x32 NEW_32x8 NEW_32x32
"""

import argparse
import os
import math
import matplotlib.pyplot as plt


# ============================================================
# USER-FACING PLOT SETTINGS (edit only this block)
# ============================================================

# Plot title (use None to fall back to a default)
PLOT_TITLE = "Performance Metrics Across Window Sizes (350 Epochs)"

# Axis labels
X_AXIS_LABEL = "Window Size"
LEFT_Y_LABEL = "loss_giou"
RIGHT_Y_LABEL = "AP_total"

# Legend labels (what appears in the legend)
LEGEND_LABEL_LOSS = "Mean loss_giou (± SEM)"
LEGEND_LABEL_AP_MEAN = "Mean AP_total (± SEM)"
LEGEND_LABEL_AP_BEST = "Max AP_total single value"

# Optional: show ±2 SEM error bars (True/False)
SHOW_SEM2 = True

# Optional: map internal group names to clean labels for the x-axis
# If a key is missing, the code will auto-clean the name using CLEAN_PREFIXES below.
LABEL_MAP = {
    # "NEW_8x8": "8×8",
    # "NEW_8x32": "8×32",
    # "NEW_32x8": "32×8",
    # "NEW_32x32": "32×32",
}

# Optional: remove these prefixes from group names before displaying
CLEAN_PREFIXES = ["NEW_", "Bl_", "Blank_", "EMA_", "Fin_", "Bs1_", "APMFin_"]

# Optional: replace underscore separators after cleaning (e.g., "8x32" -> "8×32")
REPLACE_X = True  # if True: "8x32" -> "8×32"

# ---------------- adjustable y-axis ranges ----------------
# If *_MIN / *_MAX = None -> automatic scaling
Y_MIN_LOSS = 0.2
Y_MAX_LOSS = 0.45

Y_MIN_AP = 0.6
Y_MAX_AP = 0.8

# ============================================================


# Path defaults
HISTOGRAM_BASE_DEFAULT = "/mnt/lustre/work/schreiber/szb559/DINO/Histograms"
HISTOGRAM_METRIX_NAME  = "MetrixNew.txt"


def prettify_label(name: str) -> str:
    """
    Convert internal run/group names into clean x-axis labels.
    Priority:
      1) LABEL_MAP
      2) strip CLEAN_PREFIXES
      3) optionally replace 'x' with '×'
    """
    if name in LABEL_MAP:
        return LABEL_MAP[name]

    s = name
    for p in CLEAN_PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
            break

    # Common patterns: "NEW_8x32" -> "8x32" -> "8×32"
    s = s.replace("_", " ").strip()

    if REPLACE_X:
        # replace occurrences like "8x32" with "8×32"
        s = s.replace("x", "×")

    return s


def read_metrix_file(metrix_path):
    """
    Read MetrixNew.txt and return dict:
      { group_name: { mean_loss, sem_loss, mean_ap, sem_ap, best_ap } }
    """
    if not os.path.isfile(metrix_path):
        raise FileNotFoundError(f"Metrix file not found: {metrix_path}")

    group_stats = {}
    with open(metrix_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue

            name = parts[0]
            try:
                mean_loss = float(parts[1])
                sem_loss  = float(parts[2])
                mean_ap   = float(parts[3])
                sem_ap    = float(parts[4])
                best_ap   = float(parts[5])
            except ValueError:
                continue

            group_stats[name] = {
                "mean_loss": mean_loss,
                "sem_loss":  sem_loss,
                "mean_ap":   mean_ap,
                "sem_ap":    sem_ap,
                "best_ap":   best_ap,
            }
    return group_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        nargs="+",
        required=True,
        help="Group names that appear in MetrixNew.txt (e.g. NEW_8x8 NEW_8x32 ...)"
    )
    ap.add_argument(
        "--hist-base",
        default=HISTOGRAM_BASE_DEFAULT,
        help="Base directory for histograms (contains MetrixNew.txt and output plot)"
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of the output plot (dpi)"
    )
    ap.add_argument(
        "--outfile-name",
        default=None,
        help="Optional plot filename (overrides default based on group names)"
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Optional plot title (overrides PLOT_TITLE)"
    )
    args = ap.parse_args()

    hist_dir    = args.hist_base
    metrix_path = os.path.join(hist_dir, HISTOGRAM_METRIX_NAME)

    try:
        group_stats = read_metrix_file(metrix_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # Collect requested groups
    raw_labels    = []
    display_labels = []
    means_loss    = []
    sems_loss     = []
    means_ap      = []
    sems_ap       = []
    best_aps      = []

    for g in args.run:
        if g not in group_stats:
            print(f"[WARN] Group '{g}' not found in {metrix_path} – skipped.")
            continue

        stats = group_stats[g]
        mean_loss = stats.get("mean_loss", float("nan"))
        sem_loss  = stats.get("sem_loss",  float("nan"))
        mean_ap   = stats.get("mean_ap",   float("nan"))
        sem_ap    = stats.get("sem_ap",    float("nan"))
        best_ap   = stats.get("best_ap",   float("nan"))

        if not (math.isfinite(mean_loss) and math.isfinite(sem_loss)
                and math.isfinite(mean_ap) and math.isfinite(sem_ap)
                and math.isfinite(best_ap)):
            print(f"[WARN] Invalid values for group '{g}' – skipped.")
            continue

        raw_labels.append(g)
        display_labels.append(prettify_label(g))
        means_loss.append(mean_loss)
        sems_loss.append(sem_loss)
        means_ap.append(mean_ap)
        sems_ap.append(sem_ap)
        best_aps.append(best_ap)

    if not raw_labels:
        print("[ERROR] None of the specified groups could be read from MetrixNew.txt.")
        return

    # X positions
    x = list(range(len(display_labels)))
    bar_width = 0.25

    x_loss    = [xi - bar_width for xi in x]
    x_ap_mean = [xi for xi in x]
    x_ap_best = [xi + bar_width for xi in x]

    plt.figure(figsize=(9, 5))
    ax_loss = plt.gca()
    ax_ap   = ax_loss.twinx()

    # Loss bars (left axis)
    ax_loss.bar(
        x_loss,
        means_loss,
        width=bar_width,
        alpha=0.8,
        label=LEGEND_LABEL_LOSS
    )
    ax_loss.errorbar(
        x_loss,
        means_loss,
        yerr=sems_loss,
        fmt='none',
        ecolor='black',
        elinewidth=1.5,
        capsize=5,
        label="_nolegend_"
    )
    if SHOW_SEM2:
        sem2_loss = [2.0 * s for s in sems_loss]
        ax_loss.errorbar(
            x_loss,
            means_loss,
            yerr=sem2_loss,
            fmt='none',
            ecolor='gray',
            elinewidth=1.0,
            capsize=3,
            label="_nolegend_"
        )

    # Mean AP bars (right axis)
    ax_ap.bar(
        x_ap_mean,
        means_ap,
        width=bar_width,
        alpha=0.8,
        color="orange",
        label=LEGEND_LABEL_AP_MEAN
    )
    ax_ap.errorbar(
        x_ap_mean,
        means_ap,
        yerr=sems_ap,
        fmt='none',
        ecolor='black',
        elinewidth=1.5,
        capsize=5,
        label="_nolegend_"
    )
    if SHOW_SEM2:
        sem2_ap = [2.0 * s for s in sems_ap]
        ax_ap.errorbar(
            x_ap_mean,
            means_ap,
            yerr=sem2_ap,
            fmt='none',
            ecolor='gray',
            elinewidth=1.0,
            capsize=3,
            label="_nolegend_"
        )

    # Best AP bars (right axis)
    ax_ap.bar(
        x_ap_best,
        best_aps,
        width=bar_width,
        alpha=0.8,
        color="tab:green",
        label=LEGEND_LABEL_AP_BEST
    )

    # X ticks + labels
    ax_loss.set_xticks(x)
    ax_loss.set_xticklabels(display_labels, rotation=45, ha='right')

    # Axis labels
    ax_loss.set_xlabel(X_AXIS_LABEL)
    ax_loss.set_ylabel(LEFT_Y_LABEL)
    ax_ap.set_ylabel(RIGHT_Y_LABEL)

    # Title
    plot_title = args.title if args.title is not None else PLOT_TITLE
    if plot_title is None:
        plot_title = "Window Size Comparison"
    ax_loss.set_title(plot_title)

    # Y-limits
    if (Y_MIN_LOSS is not None) or (Y_MAX_LOSS is not None):
        cur_min, cur_max = ax_loss.get_ylim()
        ax_loss.set_ylim(
            Y_MIN_LOSS if Y_MIN_LOSS is not None else cur_min,
            Y_MAX_LOSS if Y_MAX_LOSS is not None else cur_max
        )

    if (Y_MIN_AP is not None) or (Y_MAX_AP is not None):
        cur_min, cur_max = ax_ap.get_ylim()
        ax_ap.set_ylim(
            Y_MIN_AP if Y_MIN_AP is not None else cur_min,
            Y_MAX_AP if Y_MAX_AP is not None else cur_max
        )

    ax_loss.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Legend
    handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
    handles_ap, labels_ap     = ax_ap.get_legend_handles_labels()

    handles = handles_loss + handles_ap
    labels_legend = labels_loss + labels_ap

    seen = set()
    handles_unique = []
    labels_unique = []
    for h, lab in zip(handles, labels_legend):
        if lab not in seen and not lab.startswith("_"):
            seen.add(lab)
            handles_unique.append(h)
            labels_unique.append(lab)

    ax_loss.legend(handles_unique, labels_unique, loc="upper left", fontsize=8)

    plt.tight_layout()

    # Output file
    os.makedirs(hist_dir, exist_ok=True)
    if args.outfile_name is not None:
        out_file = os.path.join(hist_dir, args.outfile_name)
    else:
        name_join = "_".join(raw_labels)
        if len(name_join) > 80:
            name_join = name_join[:80] + "..."
        out_file = os.path.join(hist_dir, f"hist_{name_join}.png")

    plt.savefig(out_file, dpi=args.dpi)
    plt.close()
    print(f"[OK] Histogram plot saved to: {out_file}")


if __name__ == "__main__":
    main()