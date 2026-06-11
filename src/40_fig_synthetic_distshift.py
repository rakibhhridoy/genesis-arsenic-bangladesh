#!/usr/bin/env python3
"""Rebuild fig12 (synthetic decomposition) and fig6 (distribution shift).

fig12 (3 panels): (a) encoder advantage over alpha (nonlinearity) x delta
      (covariate shift) heatmap; (b) advantage vs nonlinearity collapsed over
      delta; (c) concept shift collapses both models toward chance.
fig6 (2 panels): (a) PC1-2 and PC3-4 distribution ellipses (mean +/- std) of
      pretrain corpus vs Bangladesh (they do not overlap); (b) per-parameter
      standardized 1-Wasserstein distance, ranked.

Reads results/synthetic_law.json, results/dist_shift.json.
Writes figures/fig12_synthetic_law.png, fig6_dist_shift_pca.png,
       fig6b_dist_shift_per_param.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import figstyle as fs

REDOX = {"As", "Fe", "Mn", "PO4", "Eh", "U", "NO3", "SO4"}


def fig_synthetic():
    d = json.load(open("results/synthetic_law.json"))
    cov = d["covariate_grid"]; con = d["concept_shift"]
    alphas = sorted(set(r["alpha"] for r in cov))
    deltas = sorted(set(r["delta"] for r in cov))
    A = np.full((len(alphas), len(deltas)), np.nan)
    for r in cov:
        A[alphas.index(r["alpha"]), deltas.index(r["delta"])] = r["advantage"]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(14, 4.4))
    # (a) heatmap
    im = a.imshow(A, origin="lower", aspect="auto", cmap="RdBu_r",
                  vmin=-abs(np.nanmax(A)), vmax=abs(np.nanmax(A)),
                  extent=[min(deltas), max(deltas), min(alphas), max(alphas)])
    a.set_xlabel(r"Covariate shift $\delta$"); a.set_ylabel(r"Target nonlinearity $\alpha$")
    a.set_title("(a)  Encoder advantage ($\\Delta$AUC)", loc="left", fontsize=11)
    a.grid(False); fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    # (b) advantage vs nonlinearity (mean over delta)
    adv_by_alpha = [np.nanmean([r["advantage"] for r in cov if r["alpha"] == al]) for al in alphas]
    b.plot(alphas, adv_by_alpha, "o-", color=fs.color("redox"), ms=7, lw=2.2,
           markeredgecolor="#333", markeredgewidth=0.6)
    b.axhline(0, color="#888", ls=":", lw=1.0)
    b.set_xlabel(r"Target nonlinearity $\alpha$"); b.set_ylabel(r"Encoder advantage ($\Delta$AUC)")
    b.set_title("(b)  Advantage is created by nonlinearity", loc="left", fontsize=11)
    # (c) concept shift collapse
    g = [r["gamma"] for r in con]
    c.plot(g, [r["auc_enc"] for r in con], "o-", color=fs.color("finetuned"), ms=7, lw=2.2,
           markeredgecolor="#333", markeredgewidth=0.6, label="encoder")
    c.plot(g, [r["auc_base"] for r in con], "s--", color=fs.color("logreg"), ms=6, lw=2.0,
           markeredgecolor="#333", markeredgewidth=0.6, label="logistic regression")
    c.axhline(0.5, color="#888", ls=":", lw=1.0)
    c.text(max(g), 0.505, "chance", ha="right", va="bottom", fontsize=8.5, color="#777", style="italic")
    c.set_xlabel(r"Concept shift $\gamma$"); c.set_ylabel("Zero-shot AUC")
    c.set_title("(c)  Concept shift collapses transfer", loc="left", fontsize=11)
    c.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout(); fig.savefig("figures/fig12_synthetic_law.png")
    print("wrote figures/fig12_synthetic_law.png")


def _ellipse(ax, mx, my, sx, sy, color, hatch, label):
    ax.add_patch(Ellipse((mx, my), 2 * sx, 2 * sy, facecolor=color, alpha=0.30,
                         edgecolor=color, lw=1.6, hatch=hatch, label=label))
    ax.plot(mx, my, "o", color=color, ms=7, markeredgecolor="#333", markeredgewidth=0.6)


def fig_distshift():
    d = json.load(open("results/dist_shift.json"))
    pm, ps = d["pretrain_pc_mean"], d["pretrain_pc_std"]
    bm, bs = d["bd_pc_mean"], d["bd_pc_std"]
    # (a) PC1-2 and PC3-4 ellipses
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (i, j) in [(ax1, (0, 1)), (ax2, (2, 3))]:
        _ellipse(ax, pm[i], pm[j], ps[i], ps[j], fs.color("Base"), "", "Pretraining corpus")
        _ellipse(ax, bm[i], bm[j], bs[i], bs[j], fs.color("As"), "....", "Bangladesh")
        ax.set_xlabel(f"PC{i+1}"); ax.set_ylabel(f"PC{j+1}")
        ax.grid(True)
    ax1.set_title("(a)  PC1 vs PC2", loc="left", fontsize=11)
    ax2.set_title("(b)  PC3 vs PC4", loc="left", fontsize=11)
    ax1.legend(loc="upper left", fontsize=8.5)
    ed = d["energy_distance_pca4d"]
    val = ed["value"] if isinstance(ed, dict) else ed
    fig.suptitle(f"Pretraining corpus and Bangladesh do not overlap (4-D energy distance {val:.1f})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig("figures/fig6_dist_shift_pca.png")
    print("wrote figures/fig6_dist_shift_pca.png")

    # (b) per-parameter W1, ranked
    w = d["per_param_w1_standardized"]
    items = sorted(((k, v["w1_standardized"]) for k, v in w.items()), key=lambda kv: kv[1])
    params = [k for k, _ in items]; vals = [v for _, v in items]
    y = np.arange(len(params))
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    for yi, p, v in zip(y, params, vals):
        red = p in REDOX
        ax.barh(yi, v, color=fs.color("redox_active") if red else fs.color("conservative_ion"),
                hatch=fs.hatch("redox_active") if red else "", **fs.BAR)
    ax.set_yticks(y); ax.set_yticklabels(
        [p + (r" $^\ast$" if p in REDOX else "") for p in params])
    ax.set_xlabel("Standardized 1-Wasserstein distance (corpus vs Bangladesh)")
    ax.set_title("Redox-sensitive species shift most")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig("figures/fig6b_dist_shift_per_param.png")
    print("wrote figures/fig6b_dist_shift_per_param.png")


if __name__ == "__main__":
    fs.apply_theme()
    fig_synthetic()
    fig_distshift()
