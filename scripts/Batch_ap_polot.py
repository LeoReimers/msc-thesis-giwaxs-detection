#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot Grouped Histogram from MetrixNew.txt for b2 and b4

Reads from:
  /mnt/lustre/work/schreiber/szb559/DINO/Histograms/MetrixNew.txt

Example:
  python Batch_ap_plot.py --run-b2 WS_8x8_b2_ep120 ... --run-b4 WS_8x8_b4_ep120 ...
"""

import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# USER-FACING PLOT SETTINGS (edit only this block)
# ============================================================

# Plot title
PLOT_TITLE = "Comparison of window geometries and batch sizes (120 epochs)"

# Axis labels
X_AXIS_LABEL = "Window geometry"
Y_AXIS_LABEL = "Mean AP_total"

# Legend labels and Colors for the two groups
COLOR_B2 = "tab:blue"
COLOR_B4 = "tab:orange"
LEGEND_LABEL_B2 = "Batch Size 2 (Mean AP +/- SEM)"
LEGEND_LABEL_B4 = "Batch Size 4 (Mean AP +/- SEM)"

# Optional: show +/- 2 SEM error bars (True/False)
SHOW_SEM2 = False

# Optional: map internal group names to clean labels for the x-axis
LABEL_MAP = {}

# ---------------- adjustable y-axis ranges ----------------
# If Y_MIN / Y_MAX = None -> automatic scaling
Y_MIN = 0.6
Y_MAX = 0.75

# ============================================================

# Path defaults
HISTOGRAM_BASE_DEFAULT = "/mnt/lustre/work/schreiber/szb559/DINO/Histograms"
HISTOGRAM_METRIX_NAME  = "MetrixNew.txt"


def prettify_label(name: str) -> str:
    """
    Convert internal run/group names into clean x-axis labels.
    Removes the batch size indicator to group them correctly.
    Example: "WS_8x8_b4_ep120" -> "WS_8x8_ep120"
    """
    if name in LABEL_MAP:
        return LABEL_MAP[name]

    # Remove batch size specific strings to get the base name
    s = name.replace("_b2", "").replace("_b4", "")

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
                mean_ap   = float(parts[3])
                sem_ap    = float(parts[4])
            except ValueError:
                continue

            group_stats[name] = {
                "mean_ap":   mean_ap,
                "sem_ap":    sem_ap,
            }
    return group_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-b2",
        nargs="+",
        required=True,
        help="Group names for batch size 2 (e.g. WS_8x8_b2_ep120 ...)"
    )
    ap.add_argument(
        "--run-b4",
        nargs="+",
        required=True,
        help="Group names for batch size 4 (e.g. WS_8x8_b4_ep120 ...)"
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
        help="Optional plot filename"
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Optional plot title (overrides PLOT_TITLE)"
    )
    args = ap.parse_args()

    if len(args.run_b2) != len(args.run_b4):
        print("[ERROR] You must provide the same number of arguments for --run-b2 and --run-b4.")
        return

    hist_dir    = args.hist_base
    metrix_path = os.path.join(hist_dir, HISTOGRAM_METRIX_NAME)

    try:
        group_stats = read_metrix_file(metrix_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # Collect requested groups
    display_labels = []
    
    means_b2 = []
    sems_b2  = []
    
    means_b4 = []
    sems_b4  = []

    for g2, g4 in zip(args.run_b2, args.run_b4):
        if g2 not in group_stats:
            print(f"[WARN] Group '{g2}' not found in {metrix_path} - skipped pair.")
            continue
        if g4 not in group_stats:
            print(f"[WARN] Group '{g4}' not found in {metrix_path} - skipped pair.")
            continue

        st2 = group_stats[g2]
        st4 = group_stats[g4]

        m2, s2 = st2.get("mean_ap", float("nan")), st2.get("sem_ap", float("nan"))
        m4, s4 = st4.get("mean_ap", float("nan")), st4.get("sem_ap", float("nan"))

        if not (math.isfinite(m2) and math.isfinite(s2) and math.isfinite(m4) and math.isfinite(s4)):
            print(f"[WARN] Invalid values for group pair '{g2}'/'{g4}' - skipped.")
            continue

        # Add data to lists
        display_labels.append(prettify_label(g2))
        means_b2.append(m2)
        sems_b2.append(s2)
        means_b4.append(m4)
        sems_b4.append(s4)

    if not display_labels:
        print("[ERROR] None of the specified groups could be read or paired from MetrixNew.txt.")
        return

    # X positions for grouped bars
    x = np.arange(len(display_labels))
    bar_width = 0.35  

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Plot Batch Size 2
    ax.bar(
        x - bar_width/2,
        means_b2,
        width=bar_width,
        alpha=0.8,
        color=COLOR_B2,
        label=LEGEND_LABEL_B2
    )
    ax.errorbar(
        x - bar_width/2,
        means_b2,
        yerr=sems_b2,
        fmt='none',
        ecolor='black',
        elinewidth=1.5,
        capsize=5,
        label="_nolegend_"
    )
    if SHOW_SEM2:
        ax.errorbar(
            x - bar_width/2,
            means_b2,
            yerr=[2.0 * s for s in sems_b2],
            fmt='none',
            ecolor='gray',
            elinewidth=1.0,
            capsize=3,
            label="_nolegend_"
        )

    # Plot Batch Size 4
    ax.bar(
        x + bar_width/2,
        means_b4,
        width=bar_width,
        alpha=0.8,
        color=COLOR_B4,
        label=LEGEND_LABEL_B4
    )
    ax.errorbar(
        x + bar_width/2,
        means_b4,
        yerr=sems_b4,
        fmt='none',
        ecolor='black',
        elinewidth=1.5,
        capsize=5,
        label="_nolegend_"
    )
    if SHOW_SEM2:
        ax.errorbar(
            x + bar_width/2,
            means_b4,
            yerr=[2.0 * s for s in sems_b4],
            fmt='none',
            ecolor='gray',
            elinewidth=1.0,
            capsize=3,
            label="_nolegend_"
        )

    # X ticks + labels
    ax.set_xticks(x)
    # HIER IST DIE ÄNDERUNG: rotation=0 und ha='center' für waagerechte Schrift
    ax.set_xticklabels(display_labels, rotation=0, ha='center')

    # Axis labels
    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel(Y_AXIS_LABEL)

    # Title
    plot_title = args.title if args.title is not None else PLOT_TITLE
    if plot_title is None:
        plot_title = "Window Size Comparison"
    ax.set_title(plot_title)

    # Y-limits
    if (Y_MIN is not None) or (Y_MAX is not None):
        cur_min, cur_max = ax.get_ylim()
        ax.set_ylim(
            Y_MIN if Y_MIN is not None else cur_min,
            Y_MAX if Y_MAX is not None else cur_max
        )

    # Grid & Legend
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()

    # Output file
    os.makedirs(hist_dir, exist_ok=True)
    if args.outfile_name is not None:
        out_file = os.path.join(hist_dir, args.outfile_name)
    else:
        out_file = os.path.join(hist_dir, "hist_AP_grouped_b2_vs_b4.png")

    plt.savefig(out_file, dpi=args.dpi)
    plt.close()
    print(f"[OK] Grouped histogram plot saved to: {out_file}")


if __name__ == "__main__":
    main()