#!/usr/bin/env python3
"""Regenerate Fig 7: learned parameter couplings vs. encoder scale.

Grouped bar chart of the mean learned MGM attention weight for the canonical
oxic major-ion couplings (left of divider) and the reducing-aquifer arsenic
couplings (right), for the Small / Base / Large encoders. Grey band = the range
of per-encoder off-diagonal median attention (the noise floor); dashed line =
their median.

Reads the per-encoder attention matrices produced by src/07_attention_analysis.py
(attention_mean.npy + param_labels.json). The Large encoder reads its STAGE-2
checkpoint output by default; pass --large_dir to override.

This script reproduces the original (Stage-1) figure exactly when pointed at
results/attention_analysis_large, and was written to refresh Large to Stage-2.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# seaborn "deep" palette — matches the original figure
COLORS = {"Small": "#4c72b0", "Base": "#dd8452", "Large": "#55a868"}
SIZE_LABEL = {"Small": "Small\n(1.47M)", "Base": "Base\n(8.99M)", "Large": "Large\n(48.4M)"}

# (label for x-axis, (param_a, param_b)); first 3 oxic, last 3 reducing-arsenic
COUPLINGS = [
    ("Ca–HCO$_3$", ("Ca", "HCO3")),
    ("Na–Cl",      ("Na", "Cl")),
    ("Mg–HCO$_3$", ("Mg", "HCO3")),
    ("As–Fe",      ("As", "Fe")),
    ("As–Eh",      ("As", "Eh")),
    ("As–PO$_4$",  ("As", "PO4")),
]
N_OXIC = 3


def load(d):
    A = np.load(Path(d) / "attention_mean.npy")
    labels = json.load(open(Path(d) / "param_labels.json"))
    if isinstance(labels, dict):
        labels = labels.get("labels", list(labels.values()))
    return A, {l: i for i, l in enumerate(labels)}


def coupling(A, idx, a, b):
    return (A[idx[a], idx[b]] + A[idx[b], idx[a]]) / 2.0


def median_floor(A):
    off = A[~np.eye(A.shape[0], dtype=bool)]
    return float(np.median(off))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small_dir", default="results/attention_analysis_small")
    ap.add_argument("--base_dir",  default="results/attention_analysis")
    ap.add_argument("--large_dir", default="results/attention_analysis_large_stage2")
    ap.add_argument("--out", default="figures/fig7_attention_scaling.png")
    args = ap.parse_args()

    dirs = {"Small": args.small_dir, "Base": args.base_dir, "Large": args.large_dir}
    data = {k: load(d) for k, d in dirs.items()}
    floors = {k: median_floor(A) for k, (A, _) in data.items()}

    # bar values: [size][coupling_index]
    vals = {k: [coupling(A, idx, a, b) for _, (a, b) in COUPLINGS]
            for k, (A, idx) in data.items()}

    x = np.arange(len(COUPLINGS))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, size in enumerate(["Small", "Base", "Large"]):
        ax.bar(x + (i - 1) * w, vals[size], w, label=SIZE_LABEL[size],
               color=COLORS[size], edgecolor="white", linewidth=0.4)

    # noise-floor band = range of per-encoder medians; dashed line = their median
    fl = list(floors.values())
    ax.axhspan(min(fl), max(fl), color="#999999", alpha=0.25, zorder=0)
    ax.axhline(float(np.median(fl)), color="#666666", ls="--", lw=0.9, zorder=1)
    ax.text(len(COUPLINGS) - 0.05, np.median(fl), " attention noise floor",
            va="center", ha="left", fontsize=9, color="#666666", style="italic")

    # divider between oxic and reducing groups
    ax.axvline(N_OXIC - 0.5, color="#bbbbbb", ls="--", lw=1.0)

    ymax = 0.08
    ax.text((N_OXIC - 1) / 2, ymax * 0.94, "Oxic major-ion\n(corpus chemistry)",
            ha="center", va="top", fontsize=11, color="#333333", weight="bold")
    ax.text(N_OXIC + (len(COUPLINGS) - N_OXIC - 1) / 2, ymax * 0.94,
            "Reducing arsenic\n(Bangladesh mechanism)",
            ha="center", va="top", fontsize=11, color="#8c2d2d", weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in COUPLINGS])
    ax.set_ylabel("Mean learned attention weight")
    ax.set_ylim(0, ymax)
    ax.set_title("Learned parameter couplings vs. encoder scale", weight="bold")
    ax.legend(title="Encoder", loc="upper right", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")

    # report the As-redox composite (manuscript number) for the Large encoder
    for size in ["Small", "Base", "Large"]:
        A, idx = data[size]
        asred = np.mean([coupling(A, idx, "As", b) for b in ("Fe", "Eh", "PO4")])
        print(f"  {size:<6} As-redox composite = {asred:.5f}  "
              f"({asred / floors[size]:.2f}x floor={floors[size]:.5f})")


if __name__ == "__main__":
    main()
