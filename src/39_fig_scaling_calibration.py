#!/usr/bin/env python3
"""Rebuild fig3 (scaling) and fig4 (calibration) with the shared vibrant theme.

fig3: BD-As frozen-MLP transfer AUC vs encoder parameter count (non-monotonic);
      chemistry-only LogReg reference band. NOTE: Large value follows the
      manuscript table/caption (Stage-1, 0.690); the Stage-2 refresh JSON differs
      (0.618) and that discrepancy is tracked separately as a content decision.
fig4: operating-point analysis for chemistry-only LogReg on BD-As transfer:
      (a) confusion matrix at F1-best threshold; (b) 10-bin reliability diagram.

Reads results/calibration_logreg_as_chemonly.json (fig4). fig3 values are taken
to match the manuscript (Small 0.632, Base 0.559, Large 0.690; LogReg 0.694+/-0.009).
Writes figures/fig3_scaling.png, figures/fig4a_confusion_matrix_chemonly.png,
       figures/fig4b_reliability_chemonly.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs


def fig_scaling():
    # manuscript-consistent frozen-MLP BD-As AUC (mean, std) per size
    sizes = [("Small", 1.47, 0.632, 0.004), ("Base", 8.99, 0.559, 0.019),
             ("Large", 48.4, 0.690, 0.022)]
    logreg_mean, logreg_std = 0.694, 0.009
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    # LogReg reference band
    ax.axhspan(logreg_mean - logreg_std, logreg_mean + logreg_std,
               color=fs.GREY, alpha=0.25, zorder=0)
    ax.axhline(logreg_mean, color=fs.GREY, ls="--", lw=1.2, zorder=1,
               label=f"chemistry-only LogReg ({logreg_mean:.3f})")
    for name, p, m, s in sizes:
        ax.errorbar(p, m, yerr=s, fmt="o", ms=12, color=fs.color(name),
                    ecolor="#444", elinewidth=1.2, capsize=4, zorder=3,
                    markeredgecolor="#333", markeredgewidth=0.7, label=f"{name} ({p}M)")
    ax.plot([s[1] for s in sizes], [s[2] for s in sizes], color="#999", lw=1.2,
            ls=":", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Pretrained-encoder parameters (millions, log scale)")
    ax.set_ylabel("Bangladesh-transfer AUC (frozen + MLP)")
    ax.set_title("Pretraining benefit is non-monotonic in encoder capacity")
    ax.set_xticks([s[1] for s in sizes]); ax.set_xticklabels([str(s[1]) for s in sizes])
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout(); fig.savefig("figures/fig3_scaling.png")
    print("wrote figures/fig3_scaling.png")


def fig_calibration():
    c = json.load(open("results/calibration_logreg_as_chemonly.json"))
    cm = c["avg_confusion_at_f1_best"]
    rb = c["reliability_quantile_10bin"]

    # (a) confusion matrix
    M = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    im = ax.imshow(M, cmap="Blues", aspect="equal")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{int(round(v))}", ha="center", va="center",
                fontsize=18, fontweight="bold",
                color="white" if v > M.max() * 0.5 else "#222")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Safe (pred)", "Unsafe (pred)"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Safe (true)", "Unsafe (true)"])
    ax.set_title(f"Confusion at F1-best (recall {cm['recall']:.0%}, prec. {cm['precision']:.0%})",
                 fontsize=10.5)
    ax.grid(False)
    fig.tight_layout(); fig.savefig("figures/fig4a_confusion_matrix_chemonly.png")
    print("wrote figures/fig4a_confusion_matrix_chemonly.png")

    # (b) reliability diagram
    pp = rb["bin_mean_predicted_prob"]; op = rb["bin_observed_positive_rate"]
    brier = rb.get("brier_avg_probs")
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot([0, 1], [0, 1], ls="--", color="#888", lw=1.2, label="perfect calibration")
    ax.plot(pp, op, "o-", color=fs.color("As"), ms=7, lw=2.0,
            markeredgecolor="#333", markeredgewidth=0.6, label="chemistry-only LogReg")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed exceedance rate")
    ttl = "Over-confident under transfer"
    if brier: ttl += f" (Brier {brier:.2f})"
    ax.set_title(ttl, fontsize=10.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout(); fig.savefig("figures/fig4b_reliability_chemonly.png")
    print("wrote figures/fig4b_reliability_chemonly.png")


if __name__ == "__main__":
    fs.apply_theme()
    fig_scaling()
    fig_calibration()
