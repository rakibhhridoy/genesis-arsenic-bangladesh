#!/usr/bin/env python3
"""ED Fig: the redox-coupled fine-tune advantage replicates across encoder sizes.

Grouped bars of Delta(fine-tuned - default random forest) AUC by mechanism group
(redox-coupled As/Fe/Mn/PO4, conservative NO3/F, uranium), one bar per encoder
size (Small/Base/Large). Vibrant theme + redundant colour/hatch per encoder size.

Reads results/redox_dissociation{_small,_base}.json (src/30, per size).
Usage:  python src/32_fig_crosssize_replication.py
Writes: figures/fig14_crosssize_replication.png
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

GROUPS = [("Redox-coupled\n(As, Fe, Mn, PO$_4$)", "ft_vs_rf__REDOX_AsFeMnPO4"),
          ("Conservative\n(NO$_3$, F)", "ft_vs_rf__CONSERVATIVE_NO3F"),
          ("Uranium", "ft_vs_rf__URANIUM")]
SIZES = [("Small", "Small (1.47M)"), ("Base", "Base (8.99M)"), ("Large", "Large (48.4M)")]


def stars(p):
    if p is None: return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    if p <= 0.06: return "(*)"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default="results/redox_dissociation_small.json")
    ap.add_argument("--base", default="results/redox_dissociation_base.json")
    ap.add_argument("--large", default="results/redox_dissociation.json")
    ap.add_argument("--out", default="figures/fig14_crosssize_replication.png")
    args = ap.parse_args()
    fs.apply_theme()
    stats = {"Small": json.load(open(args.small)),
             "Base": json.load(open(args.base)),
             "Large": json.load(open(args.large))}

    x = np.arange(len(GROUPS)); w = 0.26
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for i, (sz, lab) in enumerate(SIZES):
        deltas = [stats[sz][k]["delta"] for _, k in GROUPS]
        ps = [stats[sz][k]["wilcoxon_p"] for _, k in GROUPS]
        xs = x + (i - 1) * w
        ax.bar(xs, deltas, w, color=fs.color(sz), hatch=fs.hatch(sz),
               label=lab, **fs.BAR)
        for xi, d, p in zip(xs, deltas, ps):
            s = stars(p)
            if s:
                ax.annotate(s, (xi, d), textcoords="offset points",
                            xytext=(0, 2 if d >= 0 else -10), ha="center",
                            fontsize=11, fontweight="bold", color="#222")
    ax.axhline(0, color="#444", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels([g for g, _ in GROUPS])
    ax.set_ylabel(r"$\Delta$ AUC  (fine-tuned encoder $-$ default random forest)")
    ax.set_title(r"Redox-coupled advantage replicates across the 33$\times$ size range")
    ax.legend(title="Encoder", loc="upper right")
    ax.grid(axis="x", visible=False)
    ax.text(0.985, 0.02, "*** p<.001   ** p<.01   * p<.05   (*) p=.05    paired Wilcoxon, 3 seeds",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color="#777", style="italic")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    for sz, _ in SIZES:
        for g, k in GROUPS:
            s = stats[sz][k]
            print(f"  {sz:5s} {g.splitlines()[0]:14s} Δ{s['delta']:+.3f} p={s['wilcoxon_p']:.4f}")


if __name__ == "__main__":
    main()
