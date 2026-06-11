#!/usr/bin/env python3
"""Composite (multi-panel) main figures for the compressed 5-figure layout.

F2 instrument readout  = (a) attention couplings  (b) reconstruction R2 hierarchy
F3 benchmark result    = (a) transfer vs distance (b) FT vs RF by group
                         (c) FT vs RF per-cell     (d) baseline-strength escalation
F4 concept-shift why   = (a) PC ellipses (b) per-param W1 (c) confusion (d) reliability

(F1 architecture and F5 synthetic are existing standalone figures.)
All panels reuse src/figstyle.py and read the released JSONs.

Writes: figures/figF2_instrument.png, figF3_benchmark.png, figF4_conceptshift.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
import figstyle as fs

REDOX = {"As", "Fe", "Mn", "PO4", "Eh", "U", "NO3", "SO4"}
PANEL = dict(fontsize=13, fontweight="bold", loc="left", pad=8)


def panel_tag(ax, s):
    ax.set_title(s, **PANEL)


# ---------------------------------------------------------------- attention (F2a)
COUPLINGS = [("Ca-HCO$_3$", ("Ca", "HCO3")), ("Na-Cl", ("Na", "Cl")),
             ("Mg-HCO$_3$", ("Mg", "HCO3")), ("As-Fe", ("As", "Fe")),
             ("As-Eh", ("As", "Eh")), ("As-PO$_4$", ("As", "PO4"))]


def _load_attn(d):
    A = np.load(Path(d) / "attention_mean.npy")
    labels = json.load(open(Path(d) / "param_labels.json"))
    if isinstance(labels, dict):
        labels = labels.get("labels", list(labels.values()))
    return A, {l: i for i, l in enumerate(labels)}


def draw_attention(ax):
    dirs = {"Small": "results/attention_analysis_small",
            "Base": "results/attention_analysis",
            "Large": "results/attention_analysis_large_stage2"}
    data = {k: _load_attn(v) for k, v in dirs.items()}
    coup = lambda A, idx, a, b: (A[idx[a], idx[b]] + A[idx[b], idx[a]]) / 2
    floors = [float(np.median(A[~np.eye(A.shape[0], dtype=bool)])) for A, _ in data.values()]
    vals = {k: [coup(A, idx, a, b) for _, (a, b) in COUPLINGS] for k, (A, idx) in data.items()}
    x = np.arange(len(COUPLINGS)); w = 0.27
    for i, sz in enumerate(["Small", "Base", "Large"]):
        ax.bar(x + (i - 1) * w, vals[sz], w, color=fs.color(sz), hatch=fs.hatch(sz),
               label={"Small": "Small (1.47M)", "Base": "Base (8.99M)", "Large": "Large (48.4M)"}[sz], **fs.BAR)
    ax.axhspan(min(floors), max(floors), color="#999", alpha=0.22, zorder=0)
    ax.axhline(float(np.median(floors)), color="#666", ls="--", lw=0.9)
    ax.axvline(2.5, color="#bbb", ls="--", lw=1.0)
    ax.text(1, 0.075, "Oxic major-ion", ha="center", fontsize=9.5, color="#333", fontweight="bold")
    ax.text(4, 0.075, "Reducing arsenic", ha="center", fontsize=9.5, color=fs.RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in COUPLINGS], fontsize=8)
    ax.set_ylabel("Mean learned attention weight"); ax.set_ylim(0, 0.08)
    ax.legend(title="Encoder", loc="upper right", fontsize=7.5); ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------- reconstruction (F2b)
RECON_ORDER = ["Eh", "As", "U", "Mn", "PO4", "Fe", "SO4", "NO3",
               "Ca", "HCO3", "Mg", "Na", "K", "Cl", "TDS", "EC"]


def draw_reconstruction(ax):
    d = json.load(open("results/reconstruction_probe.json"))
    corp, bd = d["corpus"], d["bangladesh"]
    ps = [p for p in RECON_ORDER if p in corp and p in bd]
    y = np.arange(len(ps))[::-1]; h = 0.38
    for yi, p in zip(y, ps):
        red = p in REDOX
        col = fs.color("redox_active") if red else fs.color("conservative_ion")
        hat = fs.hatch("redox_active") if red else ""
        ax.barh(yi + h/2, corp[p]["r2"], h, color=col, hatch=hat, **fs.BAR)
        ax.barh(yi - h/2, bd[p]["r2"], h, color=col, hatch=hat, alpha=0.45, **fs.BAR)
    ax.axvline(0, color="#444", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels([p + (r" $^\ast$" if p in REDOX else "") for p in ps], fontsize=8)
    ax.set_xlabel(r"Reconstruction $R^2$ (mask-one)"); ax.set_xlim(-5, 1.05)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(facecolor=fs.color("redox_active"), hatch="....", edgecolor="white", label=r"redox-active ($^\ast$)"),
                       Patch(facecolor=fs.color("conservative_ion"), edgecolor="white", label="conservative"),
                       Patch(facecolor="#999", label="corpus | BD (faded)")],
              loc="lower left", fontsize=7.5)


def build_F2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6.2))
    draw_attention(a); panel_tag(a, "(a)  Learned attention couplings vs. encoder scale")
    draw_reconstruction(b); panel_tag(b, "(b)  Masked-reconstruction learnability hierarchy")
    fig.tight_layout(); fig.savefig("figures/figF2_instrument.png")
    print("wrote figures/figF2_instrument.png")


# ---------------------------------------------------------------- F3 benchmark
def draw_transfer_distance(ax):
    rt = json.load(open("results/region_transfer.json"))["results"]
    for t in ["As", "Fe", "Mn", "PO4", "U", "NO3", "F"]:
        pts = [(c["energy_distance_4d"], c["transfer_auc_mean"], c["held_out_region"])
               for c in rt if c["target"] == t and c.get("energy_distance_4d") is not None]
        gw = [(x, y) for x, y, r in pts if r != "Bangladesh"]
        bd = [(x, y) for x, y, r in pts if r == "Bangladesh"]
        if gw:
            ax.scatter([x for x, _ in gw], [y for _, y in gw], s=34, color=fs.color(t),
                       edgecolor="white", linewidth=0.5, alpha=0.9, label=t, zorder=3)
        if bd:
            ax.scatter([x for x, _ in bd], [y for _, y in bd], s=120, color=fs.color(t),
                       edgecolor="#222", linewidth=0.8, marker="*", zorder=4)
    ax.axhline(0.5, color="#999", ls=":", lw=1.0)
    ax.set_xlabel("Distribution distance (4-D energy)"); ax.set_ylabel("Chemistry-only transfer AUC")
    ax.legend(title="Contaminant", loc="upper right", fontsize=7, ncol=2)


def draw_ft_vs_rf_bars(ax):
    s = json.load(open("results/redox_dissociation.json"))
    groups = [("Redox\n(As,Fe,Mn,PO$_4$)", "ft_vs_rf__REDOX_AsFeMnPO4"),
              ("Conserv.\n(NO$_3$,F)", "ft_vs_rf__CONSERVATIVE_NO3F"),
              ("Uranium", "ft_vs_rf__URANIUM"), ("All", "ft_vs_rf__ALL_47")]
    x = np.arange(len(groups)); w = 0.36
    ft = [s[k]["mean_a"] for _, k in groups]; rf = [s[k]["mean_b"] for _, k in groups]
    ps = [s[k]["wilcoxon_p"] for _, k in groups]
    ax.bar(x - w/2, ft, w, color=fs.color("finetuned"), hatch=fs.hatch("finetuned"), label="Fine-tuned", **fs.BAR)
    ax.bar(x + w/2, rf, w, color=fs.color("rf"), hatch=fs.hatch("rf"), label="Random forest", **fs.BAR)
    ax.axhline(0.5, color="#999", ls=":", lw=1.0)
    for i, p in enumerate(ps):
        st = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
        ax.text(i, max(ft[i], rf[i]) + 0.02, st, ha="center", fontsize=10, fontweight="bold",
                color="#222" if p < .05 else "#999")
    ax.set_xticks(x); ax.set_xticklabels([g for g, _ in groups], fontsize=8)
    ax.set_ylabel("Mean transfer AUC"); ax.set_ylim(0.5, 0.95)
    ax.legend(loc="upper center", fontsize=7.5); ax.grid(axis="x", visible=False)


def draw_ft_vs_rf_scatter(ax):
    ft = {(r["target"], r["held_out_region"]): r["finetune_auc_mean"]
          for r in json.load(open("results/loro_finetune_large_multiseed.json"))["per_cell_seedmean"]}
    enc = {(c["target"], c["held_out_region"]): c
           for c in json.load(open("results/region_transfer_encoder_large.json"))["results"]}
    grp = lambda t: "redox" if t in {"As", "Fe", "Mn", "PO4"} else ("conservative" if t in {"NO3", "F"} else "uranium")
    mark = {"redox": "o", "conservative": "s", "uranium": "D"}
    seen = set()
    for k, fa in ft.items():
        c = enc.get(k)
        if not c or c.get("rf_auc") is None:
            continue
        g = grp(k[0])
        ax.scatter(c["rf_auc"], fa, s=44, color=fs.color(g), marker=mark[g],
                   edgecolor="#333", linewidth=0.5, alpha=0.9,
                   label=g if g not in seen else None, zorder=3)
        seen.add(g)
    ax.plot([0.3, 1], [0.3, 1], ls="--", color="#888", lw=1.0)
    ax.set_xlim(0.3, 1); ax.set_ylim(0.3, 1); ax.set_aspect("equal")
    ax.set_xlabel("Random-forest AUC"); ax.set_ylabel("Fine-tuned AUC")
    ax.legend(loc="lower right", fontsize=7.5)


def draw_baseline_strength(ax):
    g = json.load(open("results/baseline_strength.json"))["group_summary"]["REDOX_AsFeMnPO4"]
    order = [("rf_default", "RF\ndefault"), ("rf_reg", "RF\nreg."), ("rf_leaf", "RF\nleaf"),
             ("histgb", "HistGB"), ("xgb_strong", "XGB"), ("best_tree", "best\ntree")]
    cols = [fs.NAVY, fs.TEAL, fs.YELLOW, fs.ORANGE, fs.RED, fs.GREY]
    x = np.arange(len(order))
    for xi, (k, _), c, h in zip(x, order, cols, fs.HATCH_CYCLE):
        ax.bar(xi, g[k]["delta"], 0.66, color=c, hatch=h, **fs.BAR)
    ax.axhline(0, color="#444", lw=1.0)
    for xi, (k, _) in zip(x, order):
        p = g[k]["wilcoxon_p"]; st = "**" if p < .01 else "*" if p < .05 else "n.s."
        ax.text(xi, g[k]["delta"] + 0.0012, st, ha="center", fontsize=8.5,
                color="#222" if p < .05 else "#999")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in order], fontsize=8)
    ax.set_ylabel(r"$\Delta$AUC (FT $-$ tree)"); ax.grid(axis="x", visible=False)


def build_F3():
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    draw_transfer_distance(axes[0, 0]); panel_tag(axes[0, 0], "(a)  Transfer degrades with distance")
    draw_ft_vs_rf_bars(axes[0, 1]); panel_tag(axes[0, 1], "(b)  Fine-tuned vs. default random forest")
    draw_ft_vs_rf_scatter(axes[1, 0]); panel_tag(axes[1, 0], "(c)  Per-cell comparison (47 cells)")
    draw_baseline_strength(axes[1, 1]); panel_tag(axes[1, 1], "(d)  Advantage vs. baseline strength")
    fig.tight_layout(); fig.savefig("figures/figF3_benchmark.png")
    print("wrote figures/figF3_benchmark.png")


# ---------------------------------------------------------------- F4 concept shift
def draw_pc_ellipses(ax):
    d = json.load(open("results/dist_shift.json"))
    pm, ps, bm, bs = d["pretrain_pc_mean"], d["pretrain_pc_std"], d["bd_pc_mean"], d["bd_pc_std"]
    for (mx, my, sx, sy, col, hat, lab) in [
            (pm[0], pm[1], ps[0], ps[1], fs.color("Base"), "", "Corpus"),
            (bm[0], bm[1], bs[0], bs[1], fs.color("As"), "....", "Bangladesh")]:
        ax.add_patch(Ellipse((mx, my), 2*sx, 2*sy, facecolor=col, alpha=0.30,
                             edgecolor=col, lw=1.6, hatch=hat, label=lab))
        ax.plot(mx, my, "o", color=col, ms=7, markeredgecolor="#333", markeredgewidth=0.6)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(loc="upper left", fontsize=8)


def draw_w1(ax):
    w = json.load(open("results/dist_shift.json"))["per_param_w1_standardized"]
    items = sorted(((k, v["w1_standardized"]) for k, v in w.items()), key=lambda kv: kv[1])
    y = np.arange(len(items))
    for yi, (p, v) in zip(y, items):
        red = p in REDOX
        ax.barh(yi, v, color=fs.color("redox_active") if red else fs.color("conservative_ion"),
                hatch="...." if red else "", **fs.BAR)
    ax.set_yticks(y); ax.set_yticklabels([p + (r" $^\ast$" if p in REDOX else "") for p, _ in items], fontsize=7.5)
    ax.set_xlabel("Standardized 1-Wasserstein distance"); ax.grid(axis="y", visible=False)


def draw_confusion(ax):
    cm = json.load(open("results/calibration_logreg_as_chemonly.json"))["avg_confusion_at_f1_best"]
    M = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    im = ax.imshow(M, cmap="Blues")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{int(round(v))}", ha="center", va="center", fontsize=16,
                fontweight="bold", color="white" if v > M.max()*0.5 else "#222")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Safe", "Unsafe"]); ax.set_xlabel("Predicted")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Safe", "Unsafe"]); ax.set_ylabel("True")
    ax.text(0.5, -0.32, f"recall {cm['recall']:.0%} · precision {cm['precision']:.0%}",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555")
    ax.grid(False)


def draw_reliability(ax):
    rb = json.load(open("results/calibration_logreg_as_chemonly.json"))["reliability_quantile_10bin"]
    ax.plot([0, 1], [0, 1], ls="--", color="#888", lw=1.2, label="perfect")
    ax.plot(rb["bin_mean_predicted_prob"], rb["bin_observed_positive_rate"], "o-",
            color=fs.color("As"), ms=6, lw=2.0, markeredgecolor="#333", markeredgewidth=0.6, label="LogReg")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted prob."); ax.set_ylabel("Observed rate")
    b = rb.get("brier_avg_probs")
    if b:
        ax.text(0.05, 0.92, f"Brier {b:.2f}", fontsize=9, color="#555")
    ax.legend(loc="lower right", fontsize=8)


def build_F4():
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
    draw_pc_ellipses(axes[0, 0]); panel_tag(axes[0, 0], "(a)  Corpus vs. Bangladesh (PC1-2)")
    draw_w1(axes[0, 1]); panel_tag(axes[0, 1], "(b)  Per-parameter distribution shift")
    draw_confusion(axes[1, 0]); panel_tag(axes[1, 0], "(c)  Triage operating point")
    draw_reliability(axes[1, 1]); panel_tag(axes[1, 1], "(d)  Reliability under transfer")
    fig.tight_layout(); fig.savefig("figures/figF4_conceptshift.png")
    print("wrote figures/figF4_conceptshift.png")


if __name__ == "__main__":
    fs.apply_theme()
    build_F2()
    build_F3()
    build_F4()
