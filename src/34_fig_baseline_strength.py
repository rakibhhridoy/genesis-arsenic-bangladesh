#!/usr/bin/env python3
"""Fig: redox-suite fine-tune advantage vanishes against a strong tree.
Bars = tree baselines (weak->strong); vibrant theme, ordered palette + hatch.
Reads results/baseline_strength.json (src/33). Writes figures/fig15_baseline_strength.png
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

ORDER = [("rf_default", "RF\n(default)"), ("rf_reg", "RF\n(regularized)"),
         ("rf_leaf", "RF\n(leaf=20)"), ("histgb", "HistGradient-\nBoosting"),
         ("xgb_strong", "XGBoost\n(strong)"), ("best_tree", "per-cell\nbest tree")]
# ordered weak->strong colours from the palette (+grey for the composite best)
BARCOLORS = [fs.NAVY, fs.TEAL, fs.YELLOW, fs.ORANGE, fs.RED, fs.GREY]


def stars(p):
    if p is None: return "n.s."
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "n.s."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="results/baseline_strength.json")
    ap.add_argument("--out", default="figures/fig15_baseline_strength.png")
    args = ap.parse_args()
    fs.apply_theme()
    g = json.load(open(args.stats))["group_summary"]["REDOX_AsFeMnPO4"]
    keys = [k for k, _ in ORDER]
    deltas = [g[k]["delta"] for k in keys]
    ps = [g[k]["wilcoxon_p"] for k in keys]
    n = g["rf_default"]["n"]

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for xi, d, c, h in zip(x, deltas, BARCOLORS, fs.HATCH_CYCLE):
        ax.bar(xi, d, 0.66, color=c, hatch=h, **fs.BAR)
    ax.axhline(0, color="#444", lw=1.0)
    for xi, d, p in zip(x, deltas, ps):
        lab = f"{stars(p)}\np={p:.3f}" if p is not None else stars(p)
        ax.text(xi, d + 0.0012, lab, ha="center", va="bottom", fontsize=8.5,
                color="#222" if (p is not None and p < 0.05) else "#888")
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in ORDER])
    ax.set_ylabel(r"$\Delta$ AUC  (fine-tuned encoder $-$ tree baseline)")
    ax.set_ylim(min(0, min(deltas)) - 0.006, max(deltas) + 0.012)
    ax.set_title(f"Redox-suite fine-tune advantage vs. baseline strength (As, Fe, Mn, PO$_4$; n={n})")
    ax.text(0.5, 0.97, "advantage is baseline-sensitive: a tie against the strongest trees (HistGB, per-cell best)",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, style="italic", color="#555")
    ax.grid(axis="x", visible=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
