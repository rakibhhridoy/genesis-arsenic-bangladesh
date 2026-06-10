#!/usr/bin/env python3
"""Fig: the redox-suite fine-tune advantage vanishes against a strong tree.

Bar chart of the fine-tuned encoder's mean AUC advantage over each tree baseline
on the redox-coupled suite (As, Fe, Mn, PO4; n=34 cells), ordered by baseline
strength: default RF -> regularized RF -> leaf-regularized RF -> HistGradient-
Boosting -> strong XGBoost -> per-cell best tree. Significance (paired Wilcoxon)
marked per bar. The advantage is significant only against the default random
forest and falls to a statistical tie against any strong tree -- the empirical
core of the paper's strong-baseline message.

Reads results/baseline_strength.json (src/33).
Usage:  python src/34_fig_baseline_strength.py
Writes: figures/fig15_baseline_strength.png
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ORDER = [("rf_default", "RF\n(default)"), ("rf_reg", "RF\n(regularized)"),
         ("rf_leaf", "RF\n(leaf=20)"), ("histgb", "HistGradient-\nBoosting"),
         ("xgb_strong", "XGBoost\n(strong)"), ("best_tree", "per-cell\nbest tree")]


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
    g = json.load(open(args.stats))["group_summary"]["REDOX_AsFeMnPO4"]

    keys = [k for k, _ in ORDER]
    deltas = [g[k]["delta"] for k in keys]
    ps = [g[k]["wilcoxon_p"] for k in keys]
    sig = [g[k]["significant"] for k in keys]
    n = g["rf_default"]["n"]

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = ["#4c72b0" if s else "#b0b0b0" for s in sig]
    ax.bar(x, deltas, 0.62, color=colors, edgecolor="white", linewidth=0.6)
    ax.axhline(0, color="#444444", lw=1.0)

    for xi, d, p in zip(x, deltas, ps):
        lab = f"{stars(p)}\np={p:.3f}" if p is not None else stars(p)
        ax.text(xi, d + 0.0012, lab, ha="center", va="bottom", fontsize=8.5,
                color="#333333" if p is not None and p < 0.05 else "#777777")

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in ORDER], fontsize=9)
    ax.set_ylabel(r"$\Delta$ AUC  (fine-tuned encoder $-$ tree baseline)")
    ax.set_ylim(min(0, min(deltas)) - 0.006, max(deltas) + 0.012)
    ax.set_title("Redox-suite fine-tune advantage vs. baseline strength (As, Fe, Mn, PO$_4$; "
                 f"n={n})", weight="bold", fontsize=11)
    ax.text(0.5, 0.96, "advantage is baseline-sensitive: a tie against the strongest trees (HistGB, per-cell best)",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, style="italic", color="#555555")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#4c72b0", label="p < 0.05"),
                       Patch(facecolor="#b0b0b0", label="not significant")],
              loc="upper right", frameon=True, fontsize=9)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")
    for k, lab in ORDER:
        s = g[k]
        print(f"  {k:11s} Δ{s['delta']:+.3f} p={s['wilcoxon_p']:.4f} {'SIG' if s['significant'] else 'ns'}")


if __name__ == "__main__":
    main()
