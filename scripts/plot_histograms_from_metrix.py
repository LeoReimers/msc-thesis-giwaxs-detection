#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot Histogram from MetrixNew.txt

Reads from:
  /mnt/lustre/work/schreiber/szb559/DINO/Histograms/MetrixNew.txt

Expected format (produced by collect_epoch_metrics.py, updated):
  # group_name\tgroup_mean_loss_giou\tSEM_loss_giou\tgroup_mean_AP_total\tSEM_AP_total\tbest_AP_total_all_runs
  5_32x8    <mean_loss>  <sem_loss>  <mean_ap>  <sem_ap>  <best_ap>
  ...

Example call:
  python plot_histograms_from_metrix_new.py --run 5_32x8 7_24x4

The script creates a bar chart in which:
  - each group has three bars:
      * mean loss_giou (left y-axis)
      * mean AP_total (right y-axis)
      * max/best AP_total over all runs in that parameter set (right y-axis)
  - error bars for ±1 SEM and ±2 SEM are drawn for:
      * mean loss_giou
      * mean AP_total
    (no SEM is assumed for best_AP_total_all_runs)

The plot is saved (dpi=300) under:
  /mnt/lustre/work/schreiber/szb559/DINO/Histograms/
with a filename constructed from the group names.
"""

import argparse
import os
import math
import matplotlib.pyplot as plt

# Path defaults (consistent with collect_epoch_metrics.py)
HISTOGRAM_BASE_DEFAULT = "/mnt/lustre/work/schreiber/szb559/DINO/Histograms"
HISTOGRAM_METRIX_NAME  = "MetrixNew.txt"

# ---------------- adjustable y-axis ranges ----------------
# If *_MIN / *_MAX = None -> automatic scaling
# Otherwise the range is explicitly set to [*_MIN, *_MAX].

# Loss axis (left)
Y_MIN_LOSS = 0.2
Y_MAX_LOSS = 0.45

# AP axis (right)
Y_MIN_AP = 0.6
Y_MAX_AP = 0.8


def read_metrix_file(metrix_path):
    """
    Read MetrixNew.txt and return a dict:
      {
        group_name: {
          "mean_loss": float,
          "sem_loss":  float,
          "mean_ap":   float,
          "sem_ap":    float,
          "best_ap":   float,
        }
      }

    Expected per data line: at least 6 columns:
      name mean_loss sem_loss mean_ap sem_ap best_ap

    If a group appears multiple times, later entries overwrite earlier ones.
    Effectively, the *last* entry per group is used.
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
                # Ignore malformed lines (need the new 6th column)
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
        help="Group names that appear in MetrixNew.txt (e.g. 5_32x8 7_24x4 …)"
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
        help="Optional plot title"
    )
    args = ap.parse_args()

    # Read Metrix file
    hist_dir    = args.hist_base
    metrix_path = os.path.join(hist_dir, HISTOGRAM_METRIX_NAME)

    try:
        group_stats = read_metrix_file(metrix_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # Collect requested groups from --run
    labels       = []
    means_loss   = []
    sems_loss    = []
    means_ap     = []
    sems_ap      = []
    best_aps     = []

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

        # Required values must be finite
        if not (math.isfinite(mean_loss) and math.isfinite(sem_loss)
                and math.isfinite(mean_ap) and math.isfinite(sem_ap)
                and math.isfinite(best_ap)):
            print(f"[WARN] Invalid values for group '{g}' – skipped.")
            continue

        labels.append(g)
        means_loss.append(mean_loss)
        sems_loss.append(sem_loss)
        means_ap.append(mean_ap)
        sems_ap.append(sem_ap)
        best_aps.append(best_ap)

    if not labels:
        print("[ERROR] None of the specified groups could be read from MetrixNew.txt.")
        return

    # X-positions for groups and bars
    x = list(range(len(labels)))
    bar_width = 0.25

    # Three bars per group:
    #   loss slightly left, mean AP centered, best AP slightly right
    x_loss    = [xi - bar_width for xi in x]
    x_ap_mean = [xi for xi in x]
    x_ap_best = [xi + bar_width for xi in x]

    # Figure and axes (two y-axes)
    plt.figure(figsize=(9, 5))
    ax_loss = plt.gca()
    ax_ap   = ax_loss.twinx()

    # Bars for loss_giou (left axis)
    ax_loss.bar(
        x_loss,
        means_loss,
        width=bar_width,
        alpha=0.8,
        label="Mean loss_giou"
    )

    # Error bars ±1 SEM for loss_giou (no legend entry)
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

    # Error bars ±2 SEM for loss_giou (no legend entry)
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

    # Bars for mean AP_total (right axis)
    ax_ap.bar(
        x_ap_mean,
        means_ap,
        width=bar_width,
        alpha=0.8,
        color="orange",
        label="Mean AP_total"
    )

    # Error bars ±1 SEM for mean AP_total
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

    # Error bars ±2 SEM for mean AP_total
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

    # Bars for best/max AP_total (right axis) — no SEM assumed
    ax_ap.bar(
        x_ap_best,
        best_aps,
        width=bar_width,
        alpha=0.8,
        color="tab:green",
        label="Max AP_total (best)"
    )

    # Axis labels
    ax_loss.set_xticks(x)
    ax_loss.set_xticklabels(labels, rotation=45, ha='right')

    ax_loss.set_ylabel("Mean loss_giou")
    ax_ap.set_ylabel("AP_total")

    plot_title = args.title or "Group comparison: mean loss_giou, mean AP_total, and max AP_total"
    ax_loss.set_title(plot_title)

    # Set y-axis ranges if desired
    if (Y_MIN_LOSS is not None) or (Y_MAX_LOSS is not None):
        current_ymin_loss, current_ymax_loss = ax_loss.get_ylim()
        ymin_loss = Y_MIN_LOSS if Y_MIN_LOSS is not None else current_ymin_loss
        ymax_loss = Y_MAX_LOSS if Y_MAX_LOSS is not None else current_ymax_loss
        ax_loss.set_ylim(ymin_loss, ymax_loss)

    if (Y_MIN_AP is not None) or (Y_MAX_AP is not None):
        current_ymin_ap, current_ymax_ap = ax_ap.get_ylim()
        ymin_ap = Y_MIN_AP if Y_MIN_AP is not None else current_ymin_ap
        ymax_ap = Y_MAX_AP if Y_MAX_AP is not None else current_ymax_ap
        ax_ap.set_ylim(ymin_ap, ymax_ap)

    ax_loss.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Legend: only main metrics
    handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
    handles_ap, labels_ap     = ax_ap.get_legend_handles_labels()

    handles = handles_loss + handles_ap
    labels_legend = labels_loss + labels_ap

    # Remove duplicates while preserving order
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

    # Output filename
    os.makedirs(hist_dir, exist_ok=True)
    if args.outfile_name is not None:
        out_file = os.path.join(hist_dir, args.outfile_name)
    else:
        name_join = "_".join(labels)
        if len(name_join) > 80:
            name_join = name_join[:80] + "..."
        out_file = os.path.join(hist_dir, f"hist_{name_join}.png")

    plt.savefig(out_file, dpi=args.dpi)
    plt.close()

    print(f"[OK] Histogram plot saved to: {out_file}")


if __name__ == "__main__":
    main()