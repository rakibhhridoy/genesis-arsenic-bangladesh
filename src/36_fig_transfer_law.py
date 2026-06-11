#!/usr/bin/env python3
"""Regenerate fig8a/fig8b (the transfer-vs-distance + vs-logistic-advantage panels)
with honest, parity-consistent titles. The original generator was lost and the old
PNGs carried stale pre-reframe titles ("Where pretraining beats logistic regression").

fig8a: chemistry-only logistic-regression zero-shot transfer AUC vs the 4-D PCA
       energy distance between held-out region and training pool, coloured by
       contaminant; Bangladesh cells marked as stars at the extreme-distance corner.
fig8b: per-cell difference (frozen-encoder AUC - chemistry-only logistic AUC),
       sorted; bars darkened where the encoder's 95% CI clears the chemistry CI.
       This is the encoder's edge over the LINEAR baseline (a nonlinearity edge,
       per the main text), not a pretraining edge.

Inputs: results/region_transfer.json (chem-only AUC + energy distance per cell)
        results/region_transfer_encoder.json (Base: encoder_minus_chem, encoder_beats)
Usage:  python src/36_fig_transfer_law.py
Writes: figures/fig8a_transfer_vs_distance.png, figures/fig8b_encoder_advantage.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

PALETTE = {t: fs.color(t) for t in ["As", "Fe", "Mn", "PO4", "U", "NO3", "F"]}


def short(region):
    return region.replace("GEMStat:", "").replace("Netherlands (-the )", "Netherlands")


def main():
    fs.apply_theme()
    rt = json.load(open("results/region_transfer.json"))["results"]
    enc = json.load(open("results/region_transfer_encoder.json"))["results"]

    # ---- fig8a: chem-only transfer AUC vs distance ----
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for tgt in PALETTE:
        pts = [(c["energy_distance_4d"], c["transfer_auc_mean"], c["held_out_region"])
               for c in rt if c["target"] == tgt and c.get("energy_distance_4d") is not None]
        if not pts:
            continue
        gw = [(x, y) for x, y, r in pts if r != "Bangladesh"]
        bd = [(x, y) for x, y, r in pts if r == "Bangladesh"]
        if gw:
            ax.scatter([x for x, _ in gw], [y for _, y in gw], s=42, color=PALETTE[tgt],
                       edgecolor="white", linewidth=0.5, alpha=0.9, label=tgt, zorder=3)
        if bd:
            ax.scatter([x for x, _ in bd], [y for _, y in bd], s=150, color=PALETTE[tgt],
                       edgecolor="black", linewidth=0.8, marker="*", zorder=4)
    ax.axhline(0.5, color="#999", ls=":", lw=1.0)
    ax.text(ax.get_xlim()[1], 0.505, "chance", ha="right", va="bottom", fontsize=8.5,
            color="#777", style="italic")
    ax.set_xlabel("Distribution distance (4-D PCA energy distance)")
    ax.set_ylabel("Chemistry-only transfer AUC")
    ax.set_title("Zero-shot transfer degrades with distribution distance",
                 weight="bold", fontsize=11)
    ax.text(0.5, 0.02, "stars = Bangladesh (the extreme-distance corner)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.5, color="#555", style="italic")
    ax.legend(title="Contaminant", loc="upper right", frameon=True, fontsize=8, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/fig8a_transfer_vs_distance.png", dpi=200, bbox_inches="tight")
    print("wrote figures/fig8a_transfer_vs_distance.png")

    # ---- fig8b: encoder-minus-chem advantage per cell ----
    cells = [c for c in enc if c.get("encoder_minus_chem") is not None]
    cells.sort(key=lambda c: c["encoder_minus_chem"])
    deltas = [c["encoder_minus_chem"] for c in cells]
    labels = [f"{c['target']}/{short(c['held_out_region'])}" for c in cells]
    beats = [bool(c.get("encoder_beats")) for c in cells]
    colors = []
    for c, b in zip(cells, beats):
        base = PALETTE.get(c["target"], "#888")
        colors.append(base if b else base + "66")  # lighter if not significant

    fig, ax = plt.subplots(figsize=(8.6, 9.0))
    y = np.arange(len(cells))
    ax.barh(y, deltas, color=colors, edgecolor="white", linewidth=0.3, zorder=3)
    ax.axvline(0, color="#444", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.2)
    ax.set_ylim(-0.8, len(cells) - 0.2)
    ax.set_xlabel(r"Encoder AUC $-$ chemistry-only AUC  ($>0$: encoder leads logistic)")
    ax.set_title("Encoder's edge over logistic regression is largest for nonlinear contaminants",
                 weight="bold", fontsize=10.5)
    ax.text(0.03, 0.97, "saturated bars: encoder 95% CI clears the chemistry CI",
            transform=ax.transAxes, fontsize=7.5, color="#555", style="italic", va="top")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/fig8b_encoder_advantage.png", dpi=200, bbox_inches="tight")
    print("wrote figures/fig8b_encoder_advantage.png")

    # report key numbers vs caption
    uusa = next((c for c in enc if c["target"] == "U" and c["held_out_region"] == "USA"), None)
    if uusa:
        print(f"  U/USA encoder_minus_chem = {uusa['encoder_minus_chem']:+.3f} (caption says +0.34)")
    bd = [c for c in rt if c["held_out_region"] == "Bangladesh"]
    if bd:
        print(f"  Bangladesh distances: {sorted(round(c['energy_distance_4d'],2) for c in bd)}")


if __name__ == "__main__":
    main()
