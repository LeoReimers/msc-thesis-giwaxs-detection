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

# Plot title

PLOT_TITLE = "Batch-size effects under the full stabilization configuration with APM (350 epochs)"

#PLOT_TITLE = "Final comparison of the benchmark and optimized full-training configurations"
#PLOT_TITLE = "Batch-size effects under the full stabilization configuration with APM (350 epochs)"
#PLOT_TITLE = "Comparison of window geometries after 350 epochs with APM"
#PLOT_TITLE = "APM under progressively stabilized training (120 epochs)"

# Optional: Plot giou_loss alongside AP_total
PLOT_GIOU_LOSS = True

# Axis labels
X_AXIS_LABEL = "Training configuration"
Y_AXIS_LABEL_AP = "Mean AP_total"
Y_AXIS_LABEL_LOSS = "Mean loss_giou"

# Legend labels
LEGEND_LABEL_AP_MEAN = "Mean AP_total (+/- SEM)"
LEGEND_LABEL_LOSS = "Mean loss_giou (+/- SEM)"

# Optional: show +/- 2 SEM error bars (True/False)
SHOW_SEM2 = False

# Optional: map internal group names to clean labels for the x-axis
LABEL_MAP = {}

# Optional: remove these prefixes from group names before displaying
CLEAN_PREFIXES = ["NEW_", "Bl_", "Blank_", "EMA_", "Fin_", "Bs1_", "APMFin_"]

# ---------------- adjustable y-axis ranges ----------------
# If Y_MIN / Y_MAX = None -> automatic scaling
Y_MIN_AP = 0.6
Y_MAX_AP = 0.8

Y_MIN_LOSS = 0.2
Y_MAX_LOSS = 0.3

# ============================================================


# Path defaults
HISTOGRAM_BASE_DEFAULT = "/mnt/lustre/work/schreiber/szb559/DINO/Histograms"
HISTOGRAM_METRIX_NAME  = "MetrixNew.txt"


def prettify_label(name: str) -> str:
    """
    Convert internal run/group names into clean x-axis labels.
    """
    if name in LABEL_MAP:
        return LABEL_MAP[name]

    s = name
    for p in CLEAN_PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
            break

    s = s.replace("_", " ").strip()
    return s


def read_metrix_file(metrix_path):
    """
    Read MetrixNew.txt and return dict with stats.
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
            except ValueError:
                continue

            group_stats[name] = {
                "mean_loss": mean_loss,
                "sem_loss":  sem_loss,
                "mean_ap":   mean_ap,
                "sem_ap":    sem_ap,
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
    raw_labels     = []
    display_labels = []
    means_ap       = []
    sems_ap        = []
    means_loss     = []
    sems_loss      = []

    for g in args.run:
        if g not in group_stats:
            print(f"[WARN] Group '{g}' not found in {metrix_path} - skipped.")
            continue

        stats = group_stats[g]
        mean_ap   = stats.get("mean_ap",  float("nan"))
        sem_ap    = stats.get("sem_ap",   float("nan"))
        mean_loss = stats.get("mean_loss", float("nan"))
        sem_loss  = stats.get("sem_loss",  float("nan"))

        if not (math.isfinite(mean_ap) and math.isfinite(sem_ap)):
            print(f"[WARN] Invalid AP values for group '{g}' - skipped.")
            continue

        raw_labels.append(g)
        display_labels.append(prettify_label(g))
        means_ap.append(mean_ap)
        sems_ap.append(sem_ap)
        means_loss.append(mean_loss)
        sems_loss.append(sem_loss)

    if not raw_labels:
        print("[ERROR] None of the specified groups could be read from MetrixNew.txt.")
        return

    # X positions
    x = list(range(len(display_labels)))
    
    plt.figure(figsize=(9, 5))
    ax_ap = plt.gca()

    if PLOT_GIOU_LOSS:
        # Two variables to plot -> narrower and shifted bars
        bar_width = 0.35
        x_ap   = [xi - bar_width/2 for xi in x]
        x_loss = [xi + bar_width/2 for xi in x]
        
        ax_loss = ax_ap.twinx()
        
        # Mean Loss bars (right axis)
        ax_loss.bar(x_loss, means_loss, width=bar_width, alpha=0.8, color="tab:blue", label=LEGEND_LABEL_LOSS)
        ax_loss.errorbar(x_loss, means_loss, yerr=sems_loss, fmt='none', ecolor='black', elinewidth=1.5, capsize=5, label="_nolegend_")
        
        if SHOW_SEM2:
            sem2_loss = [2.0 * s for s in sems_loss]
            ax_loss.errorbar(x_loss, means_loss, yerr=sem2_loss, fmt='none', ecolor='gray', elinewidth=1.0, capsize=3, label="_nolegend_")
            
        ax_loss.set_ylabel(Y_AXIS_LABEL_LOSS)
        
        if (Y_MIN_LOSS is not None) or (Y_MAX_LOSS is not None):
            cur_min, cur_max = ax_loss.get_ylim()
            ax_loss.set_ylim(
                Y_MIN_LOSS if Y_MIN_LOSS is not None else cur_min,
                Y_MAX_LOSS if Y_MAX_LOSS is not None else cur_max
            )

    else:
        # Only AP to plot -> wider and centered bars
        bar_width = 0.5
        x_ap = x

    # Mean AP bars (left axis)
    ax_ap.bar(
        x_ap,
        means_ap,
        width=bar_width,
        alpha=0.8,
        color="orange",
        label=LEGEND_LABEL_AP_MEAN
    )
    
    # 1x SEM errorbars (AP)
    ax_ap.errorbar(
        x_ap,
        means_ap,
        yerr=sems_ap,
        fmt='none',
        ecolor='black',
        elinewidth=1.5,
        capsize=5,
        label="_nolegend_"
    )
    
    # Optional 2x SEM errorbars (AP)
    if SHOW_SEM2:
        sem2_ap = [2.0 * s for s in sems_ap]
        ax_ap.errorbar(
            x_ap,
            means_ap,
            yerr=sem2_ap,
            fmt='none',
            ecolor='gray',
            elinewidth=1.0,
            capsize=3,
            label="_nolegend_"
        )

    # X ticks + labels (rotation=0 für waagerechte Schrift)
    ax_ap.set_xticks(x)
    ax_ap.set_xticklabels(display_labels, rotation=0, ha='center')

    # Axis labels (Hier wurde labelpad=15 hinzugefügt)
    ax_ap.set_xlabel(X_AXIS_LABEL, labelpad=15)
    ax_ap.set_ylabel(Y_AXIS_LABEL_AP)

    # Title
    plot_title = args.title if args.title is not None else PLOT_TITLE
    if plot_title is None:
        plot_title = "Window Size Comparison"
    ax_ap.set_title(plot_title)

    # Y-limits (AP)
    if (Y_MIN_AP is not None) or (Y_MAX_AP is not None):
        cur_min, cur_max = ax_ap.get_ylim()
        ax_ap.set_ylim(
            Y_MIN_AP if Y_MIN_AP is not None else cur_min,
            Y_MAX_AP if Y_MAX_AP is not None else cur_max
        )

    # Grid & Legend
    ax_ap.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    if PLOT_GIOU_LOSS:
        # Combine legends from both axes
        handles_ap, labels_ap     = ax_ap.get_legend_handles_labels()
        handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
        ax_ap.legend(handles_ap + handles_loss, labels_ap + labels_loss, loc="upper right", fontsize=10)
    else:
        ax_ap.legend(loc="upper right", fontsize=10)

    plt.tight_layout()

    # Output file
    os.makedirs(hist_dir, exist_ok=True)
    if args.outfile_name is not None:
        out_file = os.path.join(hist_dir, args.outfile_name)
    else:
        name_join = "_".join(raw_labels)
        if len(name_join) > 80:
            name_join = name_join[:80] + "..."
        out_file = os.path.join(hist_dir, f"hist_AP_{name_join}.png")

    plt.savefig(out_file, dpi=args.dpi)
    plt.close()
    print(f"[OK] Histogram plot saved to: {out_file}")


if __name__ == "__main__":
    main()