#!/usr/bin/env python3
"""Extended Data Fig: the redox-coupled fine-tune advantage replicates across sizes.

Grouped bars of Delta(fine-tuned - Random Forest) AUC by mechanism group
(redox-coupled As/Fe/Mn/PO4, conservative NO3/F, uranium), one bar per encoder
size (Small 1.47M / Base 8.99M / Large 48.4M). Shows that the redox advantage is
positive at every size, conservative is ~0 at every size, and uranium is a wash --
i.e. the mechanism-located win is a property of the paradigm, not one encoder.

Reads the per-size headline-stat JSONs produced by src/30_redox_dissociation.py:
  results/redox_dissociation_small.json
  results/redox_dissociation_base.json
  results/redox_dissociation.json          (Large; default src/30 output)

Usage:  python src/32_fig_crosssize_replication.py
Writes: figures/fig14_crosssize_replication.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# seaborn "deep" palette — consistent with fig7 (attention scaling)
COLORS = {"Small": "#4c72b0", "Base": "#dd8452", "Large": "#55a868"}
SIZE_LABEL = {"Small": "Small (1.47M)", "Base": "Base (8.99M)", "Large": "Large (48.4M)"}

GROUPS = [
    ("Redox-coupled\n(As, Fe, Mn, PO$_4$)", "ft_vs_rf__REDOX_AsFeMnPO4"),
    ("Conservative\n(NO$_3$, F)",           "ft_vs_rf__CONSERVATIVE_NO3F"),
    ("Uranium",                             "ft_vs_rf__URANIUM"),
]


def stars(p):
    if p is None:
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p <= 0.06: return "(*)"   # mark the borderline (Base redox p=0.050)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="results/redox_dissociation_small.json")
    ap.add_argument("--base",  default="results/redox_dissociation_base.json")
    ap.add_argument("--large", default="results/redox_dissociation.json")
    ap.add_argument("--out",   default="figures/fig14_crosssize_replication.png")
    args = ap.parse_args()

    stats = {"Small": json.load(open(args.small)),
             "Base":  json.load(open(args.base)),
             "Large": json.load(open(args.large))}

    x = np.arange(len(GROUPS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 5.4))

    for i, size in enumerate(["Small", "Base", "Large"]):
        deltas = [stats[size][key]["delta"] for _, key in GROUPS]
        ps     = [stats[size][key]["wilcoxon_p"] for _, key in GROUPS]
        xs = x + (i - 1) * w
        bars = ax.bar(xs, deltas, w, label=SIZE_LABEL[size],
                      color=COLORS[size], edgecolor="white", linewidth=0.5)
        for xi, d, p in zip(xs, deltas, ps):
            s = stars(p)
            if s:
                ax.text(xi, d + (0.0015 if d >= 0 else -0.0015), s,
                        ha="center", va="bottom" if d >= 0 else "top",
                        fontsize=11, weight="bold", color="#333333")

    ax.axhline(0, color="#444444", lw=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([g for g, _ in GROUPS])
    ax.set_ylabel(r"$\Delta$ AUC  (fine-tuned encoder $-$ random forest)")
    ax.set_title("Redox-coupled advantage replicates across the 33$\\times$ size range",
                 weight="bold")
    ax.legend(title="Encoder", loc="lower left", frameon=True, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.985, 0.02,
            "*** p<.001  ** p<.01  * p<.05  (*) p=.05  |  paired Wilcoxon, 3 seeds",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="#777777", style="italic")

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")

    # echo the numbers the figure encodes
    for size in ["Small", "Base", "Large"]:
        for g, key in GROUPS:
            s = stats[size][key]
            print(f"  {size:5s} {g.splitlines()[0]:14s} "
                  f"Δ{s['delta']:+.3f}  p={s['wilcoxon_p']:.4f}  {s['wins']}/{s['n']}")


if __name__ == "__main__":
    main()
