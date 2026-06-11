#!/usr/bin/env python3
"""Fig: fine-tuned encoder vs Random Forest, by mechanism group (paper headline).

Two panels, both read straight from the released result JSONs (no model/GPU):

  (a) Grouped bars of mean zero-shot transfer AUC, fine-tuned encoder vs Random
      Forest, for the mechanism-defined groups (redox-coupled As/Fe/Mn/PO4,
      conservative NO3/F, uranium) and all 47 cells. Group means + paired-Wilcoxon
      p-values are taken verbatim from results/redox_dissociation.json (src/30) so
      the figure cannot drift from the numbers in the text.

  (b) Per-cell scatter of fine-tuned AUC (y) vs Random-Forest AUC (x), one point
      per leave-one-region-out cell, coloured by mechanism group. Points above the
      diagonal are cells where fine-tuning leads the default forest. The redox-coupled
      cells cluster above the line; uranium sits below; Bangladesh arsenic (the
      concept-shift pole, where every model fails) is labelled.

Inputs (all released):
  results/redox_dissociation.json              (group stats, src/30)
  results/loro_finetune_large_multiseed.json   (3-seed fine-tune, src/29)
  results/region_transfer_encoder_large.json   (frozen encoder + RF/XGB cols)

Usage:  python src/31_fig_finetune_vs_rf.py
Writes: figures/fig13_finetune_vs_rf.png
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

# mechanism grouping — identical to src/30_redox_dissociation.py (pre-specified)
REDOX = {"As", "Fe", "Mn", "PO4"}
CONSERVATIVE = {"NO3", "F"}
URANIUM = {"U"}

# methods (color+hatch) and mechanism groups from the shared design system
C_FT = fs.color("finetuned")
C_RF = fs.color("rf")
C_GROUP = {"redox": fs.color("redox"), "conservative": fs.color("conservative"),
           "uranium": fs.color("uranium")}
GROUP_LABEL = {"redox": "Redox-coupled (As, Fe, Mn, PO$_4$)",
               "conservative": "Conservative (NO$_3$, F)",
               "uranium": "Uranium"}


def group_of(target):
    if target in REDOX:
        return "redox"
    if target in CONSERVATIVE:
        return "conservative"
    if target in URANIUM:
        return "uranium"
    return None


def load_rows(path):
    d = json.load(open(path))
    if isinstance(d, list):
        return d
    for k in ("per_seed_rows", "results", "rows"):
        if k in d and isinstance(d[k], list):
            return d[k]
    raise ValueError(f"no row list in {path}")


def build_cells(ft_path, enc_path):
    """Seed-average fine-tune per cell, join to frozen + RF; mirrors src/30."""
    by = defaultdict(list)
    for r in load_rows(ft_path):
        by[(r["target"], r["held_out_region"])].append(r["finetune_auc"])
    ft = {k: float(np.mean(v)) for k, v in by.items()}
    enc = {(c["target"], c["held_out_region"]): c for c in load_rows(enc_path)}
    cells = []
    for k, c in enc.items():
        if c.get("rf_auc") is None or k not in ft:
            continue
        cells.append(dict(target=k[0], region=k[1], rf=c["rf_auc"], ft=ft[k],
                          group=group_of(k[0])))
    return cells


def p_to_stars(p):
    if p is None:
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="results/redox_dissociation.json")
    ap.add_argument("--ft", default="results/loro_finetune_large_multiseed.json")
    ap.add_argument("--enc", default="results/region_transfer_encoder_large.json")
    ap.add_argument("--out", default="figures/fig13_finetune_vs_rf.png")
    args = ap.parse_args()

    stats = json.load(open(args.stats))
    cells = build_cells(args.ft, args.enc)

    fs.apply_theme()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.6))

    # ---------------- panel (a): grouped bars by mechanism group ----------------
    groups = [
        ("Redox-coupled\n(As, Fe, Mn, PO$_4$)", "ft_vs_rf__REDOX_AsFeMnPO4"),
        ("Conservative\n(NO$_3$, F)",           "ft_vs_rf__CONSERVATIVE_NO3F"),
        ("Uranium",                             "ft_vs_rf__URANIUM"),
        ("All cells",                           "ft_vs_rf__ALL_47"),
    ]
    x = np.arange(len(groups))
    w = 0.36
    ft_means = [stats[k]["mean_a"] for _, k in groups]
    rf_means = [stats[k]["mean_b"] for _, k in groups]
    ns = [stats[k]["n"] for _, k in groups]
    ps = [stats[k]["wilcoxon_p"] for _, k in groups]

    axa.bar(x - w / 2, ft_means, w, label="Fine-tuned encoder",
            color=C_FT, hatch=fs.hatch("finetuned"), **fs.BAR)
    axa.bar(x + w / 2, rf_means, w, label="Random forest",
            color=C_RF, hatch=fs.hatch("rf"), **fs.BAR)

    axa.axhline(0.5, color="#999999", ls=":", lw=1.0, zorder=0)
    axa.text(len(groups) - 0.45, 0.505, "chance", fontsize=8.5,
             color="#777777", style="italic", va="bottom", ha="right")
    axa.grid(axis="x", visible=False)

    top = max(max(ft_means), max(rf_means))
    for i, (_, k) in enumerate(groups):
        bar_top = max(ft_means[i], rf_means[i])
        stars = p_to_stars(ps[i])
        if ps[i] is not None and ps[i] >= 0.999:
            plab = "parity"
        elif ps[i] is not None and ps[i] < 0.01:
            plab = f"p={ps[i]:.4f}"      # match manuscript precision (e.g. 0.0096)
        else:
            plab = f"p={ps[i]:.3f}"
        axa.text(i, bar_top + 0.018, stars, ha="center", va="bottom",
                 fontsize=13, weight="bold",
                 color=("#333333" if stars != "n.s." else "#999999"))
        axa.text(i, bar_top + 0.052, f"{plab}\n(n={ns[i]})", ha="center",
                 va="bottom", fontsize=8.5, color="#555555")

    axa.set_xticks(x)
    axa.set_xticklabels([g for g, _ in groups], fontsize=9.5)
    axa.set_ylabel("Mean zero-shot transfer AUC")
    axa.set_ylim(0.5, top + 0.13)
    axa.set_title("(a)  Advantage over the default random forest, by mechanism group",
                  fontsize=11, weight="bold", loc="left")
    axa.legend(loc="upper right")

    # ---------------- panel (b): per-cell scatter FT vs RF ----------------
    MARK = {"redox": "o", "conservative": "s", "uranium": "D"}
    for g in ["redox", "conservative", "uranium"]:
        pts = [c for c in cells if c["group"] == g]
        axb.scatter([c["rf"] for c in pts], [c["ft"] for c in pts],
                    s=58, color=C_GROUP[g], edgecolor="#333333", linewidth=0.6,
                    marker=MARK[g], alpha=0.92,
                    label=f"{GROUP_LABEL[g]} (n={len(pts)})", zorder=3)

    lo, hi = 0.30, 1.0
    axb.plot([lo, hi], [lo, hi], ls="--", color="#888888", lw=1.0, zorder=1)
    axb.fill_between([lo, hi], [lo, hi], [hi, hi], color=C_FT, alpha=0.05, zorder=0)
    axb.text(0.40, 0.66, "fine-tuning wins", fontsize=9.5, color=C_FT,
             style="italic", ha="left", va="bottom", rotation=45,
             rotation_mode="anchor")
    axb.text(0.97, 0.40, "forest wins", fontsize=9.5, color="#888888",
             style="italic", ha="right", va="bottom")

    # label Bangladesh arsenic (the concept-shift pole)
    for c in cells:
        if c["target"] == "As" and c["region"] == "Bangladesh":
            axb.annotate("As / Bangladesh\n(concept shift:\nall models fail)",
                         (c["rf"], c["ft"]), textcoords="offset points",
                         xytext=(18, -6), fontsize=8, color="#c44e52",
                         ha="left", va="center",
                         arrowprops=dict(arrowstyle="->", color="#c44e52", lw=0.9))

    axb.set_xlim(lo, hi)
    axb.set_ylim(lo, hi)
    axb.set_aspect("equal")
    axb.set_xlabel("Random-forest AUC")
    axb.set_ylabel("Fine-tuned encoder AUC")
    axb.set_title("(b)  Per-cell comparison (47 leave-one-region-out cells)",
                  fontsize=11, weight="bold", loc="left")
    axb.legend(loc="upper left")
    axb.grid(True)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")

    # echo the headline numbers the figure encodes
    for g, k in groups:
        s = stats[k]
        print(f"  {g.replace(chr(10), ' '):28s} FT {s['mean_a']:.3f}  RF {s['mean_b']:.3f}  "
              f"Δ{s['delta']:+.3f}  p={s['wilcoxon_p']:.4f}  n={s['n']}")
    above = sum(1 for c in cells if c["ft"] > c["rf"])
    print(f"  per-cell: fine-tuning above diagonal in {above}/{len(cells)} cells")


if __name__ == "__main__":
    main()
