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
    """Cleveland dot plot: 3 encoder-size dots per coupling, with noise-floor band."""
    dirs = {"Small": "results/attention_analysis_small",
            "Base": "results/attention_analysis",
            "Large": "results/attention_analysis_large_stage2"}
    data = {k: _load_attn(v) for k, v in dirs.items()}
    coup = lambda A, idx, a, b: (A[idx[a], idx[b]] + A[idx[b], idx[a]]) / 2
    floors = [float(np.median(A[~np.eye(A.shape[0], dtype=bool)])) for A, _ in data.values()]
    vals = {k: [coup(A, idx, a, b) for _, (a, b) in COUPLINGS] for k, (A, idx) in data.items()}
    y = np.arange(len(COUPLINGS))[::-1]
    mk = {"Small": "o", "Base": "s", "Large": "D"}
    lbl = {"Small": "Small (1.47M)", "Base": "Base (8.99M)", "Large": "Large (48.4M)"}
    ax.set_ylim(-0.5, len(COUPLINGS) - 0.5)        # first 3 couplings oxic, last 3 reducing
    # semantic region shading: oxic (teal) top, reducing-As (red) bottom
    ax.axhspan(2.5, len(COUPLINGS) - 0.5, color=fs.TEAL, alpha=0.10, zorder=0)
    ax.axhspan(-0.5, 2.5, color=fs.RED, alpha=0.08, zorder=0)
    ax.axhline(2.5, color="#bbb", ls="-", lw=0.8, zorder=1)
    # noise-floor vertical band + line + label (kept inside the axes)
    ax.axvspan(min(floors), max(floors), color="#999", alpha=0.15, zorder=0)
    ax.axvline(float(np.median(floors)), color="#666", ls="--", lw=0.9, zorder=1)
    ax.text(float(np.median(floors)) + 0.001, 5.35, "noise floor", fontsize=7.5,
            color="#555", style="italic", ha="left", va="top", zorder=5)
    for yi, ci in zip(y, range(len(COUPLINGS))):
        triple = [vals[s][ci] for s in ["Small", "Base", "Large"]]
        ax.plot([min(triple), max(triple)], [yi, yi], color="#ccc", lw=1.5, zorder=2)
        for s in ["Small", "Base", "Large"]:
            ax.scatter(vals[s][ci], yi, s=68, color=fs.color(s), marker=mk[s],
                       edgecolor="#333", linewidth=0.5, zorder=3,
                       label=lbl[s] if ci == 0 else None)
    ax.set_yticks(y); ax.set_yticklabels([c[0] for c in COUPLINGS], fontsize=8.5)
    ax.set_xlabel("Mean learned attention weight"); ax.set_xlim(0, 0.082)
    # region labels inside their bands (right side, clear of dots)
    ax.text(0.080, 4.0, "oxic", ha="right", va="center", fontsize=9, color=fs.NAVY, fontweight="bold")
    ax.text(0.080, 0.6, "reducing As", ha="right", va="center", fontsize=9, color=fs.RED, fontweight="bold")
    ax.legend(title="Encoder", loc="upper left", fontsize=7.5); ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------- reconstruction (F2b)
RECON_ORDER = ["Eh", "As", "U", "Mn", "PO4", "Fe", "SO4", "NO3",
               "Ca", "HCO3", "Mg", "Na", "K", "Cl", "TDS", "EC"]


def draw_reconstruction(ax):
    """Diverging drop bars: corpus R2 - Bangladesh R2 per parameter (the redox collapse)."""
    d = json.load(open("results/reconstruction_probe.json"))
    corp, bd = d["corpus"], d["bangladesh"]
    ps = [p for p in RECON_ORDER if p in corp and p in bd]
    drops = {p: corp[p]["r2"] - bd[p]["r2"] for p in ps}
    ps = sorted(ps, key=lambda p: drops[p])          # smallest drop at bottom
    y = np.arange(len(ps))
    for yi, p in zip(y, ps):
        red = p in REDOX
        col = fs.color("redox_active") if red else fs.color("conservative_ion")
        hat = fs.hatch("redox_active") if red else fs.hatch("conservative_ion")
        ax.barh(yi, drops[p], color=col, hatch=hat, **fs.BAR)
        ax.text(drops[p] + 0.08, yi, f"{drops[p]:.1f}", va="center", fontsize=7, color="#555")
    ax.axvline(0, color="#444", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels([p + (r" $^\ast$" if p in REDOX else "") for p in ps], fontsize=8)
    ax.set_xlabel(r"Reconstruction $R^2$ collapse (corpus $-$ Bangladesh)")
    ax.set_xlim(-0.4, max(drops.values()) + 0.6); ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(facecolor=fs.color("redox_active"), hatch=fs.hatch("redox_active"),
                             edgecolor="white", label=r"redox-active ($^\ast$)"),
                       Patch(facecolor=fs.color("conservative_ion"), hatch=fs.hatch("conservative_ion"),
                             edgecolor="white", label="conservative")],
              loc="lower right", fontsize=7.5)


def build_F2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6.2))
    draw_attention(a); panel_tag(a, "(a)  Learned attention couplings vs. encoder scale")
    draw_reconstruction(b); panel_tag(b, "(b)  Reconstruction collapse on the reducing aquifer")
    fig.tight_layout(); fig.savefig("figures/figF2_instrument.png")
    print("wrote figures/figF2_instrument.png")


# ---------------------------------------------------------------- F3 benchmark
def draw_transfer_distance(ax):
    rt = json.load(open("results/region_transfer.json"))["results"]
    marks = {"As": "o", "Fe": "s", "Mn": "D", "PO4": "^", "U": "v", "NO3": "P", "F": "X"}
    for t in ["As", "Fe", "Mn", "PO4", "U", "NO3", "F"]:
        pts = [(c["energy_distance_4d"], c["transfer_auc_mean"], c["held_out_region"])
               for c in rt if c["target"] == t and c.get("energy_distance_4d") is not None]
        gw = [(x, y) for x, y, r in pts if r != "Bangladesh"]
        bd = [(x, y) for x, y, r in pts if r == "Bangladesh"]
        if gw:
            ax.scatter([x for x, _ in gw], [y for _, y in gw], s=40, color=fs.color(t),
                       marker=marks[t], edgecolor="#333", linewidth=0.5, alpha=0.9,
                       label=t, zorder=3)
        if bd:
            ax.scatter([x for x, _ in bd], [y for _, y in bd], s=150, color=fs.color(t),
                       marker=marks[t], edgecolor="#000", linewidth=1.3, zorder=4)
    ax.axhline(0.5, color="#999", ls=":", lw=1.0)
    ax.set_xlabel("Distribution distance (4-D energy)"); ax.set_ylabel("Chemistry-only transfer AUC")
    ax.legend(title="Contaminant (bold edge = Bangladesh)", loc="upper right", fontsize=7, ncol=2)


def draw_ft_vs_rf_bars(ax):
    """Box + strip: per-cell FT vs RF AUC distributions per mechanism group."""
    s = json.load(open("results/redox_dissociation.json"))
    ft = {(r["target"], r["held_out_region"]): r["finetune_auc_mean"]
          for r in json.load(open("results/loro_finetune_large_multiseed.json"))["per_cell_seedmean"]}
    rf = {(c["target"], c["held_out_region"]): c.get("rf_auc")
          for c in json.load(open("results/region_transfer_encoder_large.json"))["results"]}
    groups = [("Redox\n(As,Fe,Mn,PO$_4$)", {"As", "Fe", "Mn", "PO4"}, "ft_vs_rf__REDOX_AsFeMnPO4"),
              ("Conserv.\n(NO$_3$,F)", {"NO3", "F"}, "ft_vs_rf__CONSERVATIVE_NO3F"),
              ("Uranium", {"U"}, "ft_vs_rf__URANIUM")]
    rng = np.random.default_rng(0)
    xc = np.arange(len(groups)); w = 0.34
    for gi, (lab, sel, kk) in enumerate(groups):
        keys = [k for k in ft if k[0] in sel and rf.get(k) is not None]
        ftv = np.array([ft[k] for k in keys]); rfv = np.array([rf[k] for k in keys])
        for off, vals, col, who in [(-w/2, ftv, fs.color("finetuned"), "FT"),
                                    (+w/2, rfv, fs.color("rf"), "RF")]:
            bp = ax.boxplot(vals, positions=[gi + off], widths=w*0.8, patch_artist=True,
                            showfliers=False, medianprops=dict(color="#222", lw=1.3),
                            whiskerprops=dict(color="#666"), capprops=dict(color="#666"))
            for box in bp["boxes"]:
                box.set(facecolor=col, alpha=0.45, edgecolor="#333", linewidth=0.8)
            jx = gi + off + (rng.random(len(vals)) - 0.5) * w * 0.5
            ax.scatter(jx, vals, s=18, color=col, edgecolor="#333", linewidth=0.3, zorder=3, alpha=0.9)
        p = s[kk]["wilcoxon_p"]; st = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
        top = max(ftv.max(), rfv.max())
        ax.text(gi, top + 0.03, st, ha="center", fontsize=9.5, fontweight="bold",
                color="#222" if p < .05 else "#999")
    ax.axhline(0.5, color="#999", ls=":", lw=1.0)
    ax.set_xticks(xc); ax.set_xticklabels([g for g, _, _ in groups], fontsize=8.5)
    ax.set_ylabel("Per-cell transfer AUC"); ax.set_ylim(0.3, 1.02)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=fs.color("finetuned"), alpha=0.45, edgecolor="#333", label="Fine-tuned"),
                       Patch(facecolor=fs.color("rf"), alpha=0.45, edgecolor="#333", label="Random forest")],
              loc="lower left", fontsize=7.5)
    ax.grid(axis="x", visible=False)


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
    ax.set_xlim(0.3, 1); ax.set_ylim(0.3, 1)  # fill cell width (no forced square)
    ax.set_xlabel("Random-forest AUC"); ax.set_ylabel("Fine-tuned AUC")
    ax.legend(loc="lower right", fontsize=7.5)


def draw_baseline_strength(ax):
    """Horizontal bars + 95% CI whiskers: redox-suite advantage (FT - tree) per baseline."""
    cells = [c for c in json.load(open("results/baseline_strength.json"))["per_cell"]
             if c["target"] in {"As", "Fe", "Mn", "PO4"}]
    gsum = json.load(open("results/baseline_strength.json"))["group_summary"]["REDOX_AsFeMnPO4"]
    order = [("rf_default", "RF (default)"), ("rf_reg", "RF (regularized)"),
             ("rf_leaf", "RF (leaf=20)"), ("histgb", "HistGradientBoosting"),
             ("xgb_strong", "XGBoost (strong)"), ("best_tree", "per-cell best tree")]
    cols = [fs.NAVY, fs.TEAL, fs.YELLOW, fs.ORANGE, fs.RED, fs.GREY]
    hatches = [fs.hatch("frozen"), fs.hatch("rf"), "----", fs.hatch("histgb"),
               fs.hatch("xgb"), fs.hatch("redox")]
    y = np.arange(len(order))[::-1]
    n = len(cells)
    for yi, (k, lab), c, h in zip(y, order, cols, hatches):
        diffs = np.array([cc["ft"] - cc[k] for cc in cells])
        m = diffs.mean(); se = diffs.std(ddof=1) / np.sqrt(n); ci = 1.96 * se
        p = gsum[k]["wilcoxon_p"]; sig = p < 0.05
        ax.barh(yi, m, color=c, hatch=h, zorder=2, **fs.BAR)
        ax.errorbar(m, yi, xerr=ci, fmt="none", ecolor="#333", elinewidth=1.2,
                    capsize=3.5, zorder=3)
        st = "**" if p < .01 else "*" if p < .05 else "n.s."
        ax.text(0.043, yi, st, va="center", ha="right", fontsize=8.5,
                color="#222" if sig else "#999")
    ax.axvline(0, color="#444", lw=1.2, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in order], fontsize=8.5)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(r"$\Delta$AUC (fine-tuned $-$ tree)  $\pm$95\% CI")
    ax.set_xlim(-0.02, 0.045); ax.grid(axis="y", visible=False)


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
    """Sample-level PCA scatter: corpus cloud vs Bangladesh points (PC1-2)."""
    s = json.load(open("results/fig4_panels.json"))["pca_scatter"]
    ax.scatter(s["corpus_pc1"], s["corpus_pc2"], s=6, color=fs.color("Base"),
               alpha=0.18, edgecolor="none", zorder=2, label="Corpus (n=%d shown)" % len(s["corpus_pc1"]), rasterized=True)
    ax.scatter(s["bd_pc1"], s["bd_pc2"], s=9, color=fs.color("As"), alpha=0.55,
               edgecolor="none", zorder=3, label="Bangladesh", rasterized=True)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    leg = ax.legend(loc="upper left", fontsize=8, markerscale=2)
    for lh in leg.legend_handles:
        lh.set_alpha(1)


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
    import matplotlib.colors as mcolors
    cm = json.load(open("results/calibration_logreg_as_chemonly.json"))["avg_confusion_at_f1_best"]
    M = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    # semantic colouring: predicted-Safe column = teal, predicted-Unsafe column =
    # light red; cell intensity (alpha) scaled by count
    teal = np.array(mcolors.to_rgb(fs.TEAL))
    lred = np.array(mcolors.to_rgb("#E8836F"))   # light red
    mx = M.max()
    rgba = np.ones((2, 2, 4))
    for i in range(2):
        for j in range(2):
            base = teal if j == 0 else lred
            a = 0.20 + 0.70 * (M[i, j] / mx)
            rgba[i, j, :3] = 1 - a * (1 - base)   # blend base over white
            rgba[i, j, 3] = 1.0
    ax.imshow(rgba, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        dark = M[i, j] > 0.55 * mx
        ax.text(j, i, f"{int(round(v))}", ha="center", va="center", fontsize=16,
                fontweight="bold", color="white" if dark else "#222")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Safe", "Unsafe"]); ax.set_xlabel("Predicted")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Safe", "Unsafe"]); ax.set_ylabel("True")
    ax.text(0.5, -0.32, f"recall {cm['recall']:.0%} · precision {cm['precision']:.0%}",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555")
    ax.grid(False)


def draw_reliability(ax):
    """Calibration curve + inset histogram of predicted probabilities."""
    rb = json.load(open("results/calibration_logreg_as_chemonly.json"))["reliability_quantile_10bin"]
    ax.plot([0, 1], [0, 1], ls="--", color="#888", lw=1.2, label="perfect calibration")
    ax.plot(rb["bin_mean_predicted_prob"], rb["bin_observed_positive_rate"], "o-",
            color=fs.color("As"), ms=6, lw=2.0, markeredgecolor="#333", markeredgewidth=0.6,
            label="chemistry-only LogReg")
    # shade the overconfidence gap (curve below diagonal)
    pp = np.array(rb["bin_mean_predicted_prob"]); op = np.array(rb["bin_observed_positive_rate"])
    ax.fill_between(pp, op, pp, color=fs.color("As"), alpha=0.12, zorder=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed exceedance rate")
    b = rb.get("brier_avg_probs")
    if b:
        ax.text(0.04, 0.94, f"Brier {b:.2f}\n(over-confident)", fontsize=8.5, color="#555", va="top")
    ax.legend(loc="lower right", fontsize=7.5)
    # inset: predicted-probability histogram (predictions pile up near 1.0)
    h = json.load(open("results/fig4_panels.json"))["pred_hist"]
    iax = ax.inset_axes([0.04, 0.46, 0.40, 0.34])
    edges = np.array(h["edges"]); ctr = (edges[:-1] + edges[1:]) / 2
    iax.bar(ctr, h["counts"], width=0.092, color=fs.color("As"), hatch=fs.hatch("redox_active"),
            edgecolor="white", linewidth=0.4)
    iax.set_title("predicted prob.", fontsize=7, pad=2)
    iax.tick_params(labelsize=6, length=2); iax.set_yticks([])
    iax.set_xlim(0, 1); iax.grid(False)
    for sp in ("top", "right", "left"):
        iax.spines[sp].set_visible(False)


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
