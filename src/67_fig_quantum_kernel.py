#!/usr/bin/env python3
"""Figure 6, the simulated quantum-kernel entrant against its matched controls and the field.

  (a) cell AUC in the held-out region, quantum kernel against the classical RBF kernel
  (b) quantum kernel against the same circuit without entangling terms
  (c) mean AUC against miss rate on exceedances for all eleven classifiers, 95% bootstrap
      intervals over cells

Reads results/quantum_kernel_summary.json (src/66).

  python src/67_fig_quantum_kernel.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import evalstyle as es

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
GROUP = {"As": "redox", "Fe": "redox", "Mn": "redox", "PO4": "redox",
         "NO3": "conservative", "F": "conservative", "U": "uranium"}
GMARK = {"redox": "^", "conservative": "o", "uranium": "s"}
ORDER = ["finetune_base", "frozen_base", "tabpfn", "xgb", "rf", "logreg", "rf_aux", "knn_space",
         "qkernel", "qkernel_prod", "rbf_svm"]
rng = np.random.default_rng(0)

es.apply()
S = json.load(open(ROOT / "results/quantum_kernel_summary.json"))
M = S["methods"]


def boot_ci(x, n=5000):
    m = rng.choice(np.asarray(x, float), (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def paired(ax, other, key):
    q, o = M["qkernel"]["cells"], M[other]["cells"]
    for c in q:
        g = GROUP[c.split("/")[0]]
        ax.scatter(o[c]["auc"], q[c]["auc"], marker=GMARK[g], s=16, lw=0.5, zorder=3,
                   facecolor=es.GROUP_FILL[g] if g != "uranium" else "#BDBDBD", edgecolor=es.INK)
    ax.plot([0.3, 1], [0.3, 1], color=es.INK, lw=0.6, ls="--", zorder=1)
    ax.set_xlim(0.3, 1)
    ax.set_ylim(0.3, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    p = S["pairs"][f"qkernel_vs_{other}_all"]["auc"]
    ax.text(0.97, 0.05, f"mean {p['a']:.3f} vs {p['b']:.3f}\nhigher in {p['a_wins']} of 47 cells\n"
            f"Wilcoxon $p$ = {p['p']:.2f}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6)
    ax.set_xlabel(f"AUC, {key}")
    ax.set_ylabel("AUC, quantum kernel")


fig = plt.figure(figsize=(es.DOUBLE, 2.45))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.45)
a, b, c = (fig.add_subplot(gs[i]) for i in range(3))
paired(a, "rbf_svm", "classical RBF kernel")
es.label(a, "a", x=-0.3)
paired(b, "qkernel_prod", "circuit without entanglement")
es.label(b, "b", x=-0.3)

z = c.inset_axes([0.07, 0.07, 0.42, 0.36])
for k in ORDER:
    lab, col, mk = es.METHOD[k]
    cells = M[k]["cells"].values()
    auc = [v["auc"] for v in cells]
    miss = [1 - v["cov_pos"] for v in cells]
    lo, hi = boot_ci(auc)
    mu = np.mean(auc)
    c.errorbar(mu, np.mean(miss), xerr=[[mu - lo], [hi - mu]], fmt="none", ecolor=es.GREY,
               elinewidth=0.6, capsize=1.5, zorder=2)
    c.plot(mu, np.mean(miss), marker=mk, color=col, ms=8.5 if mk == "*" else 5.5, lw=0,
           markeredgecolor=es.edge(col), markeredgewidth=0.6, zorder=4 if k == "qkernel" else 3)
    z.plot(mu, np.mean(miss), marker=mk, color=col, ms=7 if mk == "*" else 4.5, lw=0,
           markeredgecolor=es.edge(col), markeredgewidth=0.6, zorder=4 if k == "qkernel" else 3)
z.set_xlim(0.7055, 0.7125)
z.set_ylim(0.193, 0.209)
z.set_xticks([0.707, 0.709, 0.711])
z.set_yticks([0.195, 0.200, 0.205])
z.tick_params(labelleft=False, labelbottom=False, length=1.5)
for sp in z.spines.values():
    sp.set_visible(True)
    sp.set_linewidth(0.5)
z.set_facecolor("white")
c.indicate_inset_zoom(z, edgecolor=es.GREY, lw=0.5)
c.axhline(0.10, color=es.INK, lw=0.6, ls="--")
c.text(0.795, 0.107, "nominal 0.10", fontsize=5.8, ha="right", va="bottom")
c.set_xlim(0.5, 0.8)
c.set_ylim(0.0, 0.4)
c.set_box_aspect(1)
c.set_xlabel("Mean exceedance AUC, held-out region")
c.set_ylabel("Miss rate on exceedances")
es.label(c, "c", x=-0.2)

hs = es.handles(ORDER) + [
    Line2D([], [], marker=GMARK[g], lw=0, ms=4, markeredgecolor=es.INK, markeredgewidth=0.5,
           markerfacecolor=es.GROUP_FILL[g] if g != "uranium" else "#BDBDBD",
           label=f"{es.GROUP_LABEL[g].lower()} cell") for g in GMARK]
es.shared_legend(fig, hs, ncol=7, y=0.93)
es.save(fig, FIG, "fig_eval_Q_quantum")
