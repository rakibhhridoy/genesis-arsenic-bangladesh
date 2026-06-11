#!/usr/bin/env python3
"""Rebuild fig10 (aux-reliance) and fig11 (layer-wise decodability) with the
shared vibrant theme. Both from released JSONs.

fig10: per-parameter reconstruction R2 with vs without the 18 auxiliary tokens
       (connector = environmental-context contribution); redox-active depend on it.
fig11: layer-wise [CLS] ridge-decodability per species across Base's 8 layers;
       conservative ions rise with depth, redox info is discarded.

Reads results/aux_reliance.json, results/layerwise_decodability.json.
Writes figures/fig10_aux_reliance.png, figures/fig11_layerwise.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

REDOX = {"As", "Fe", "Mn", "PO4", "Eh", "U", "NO3"}


def fig_aux():
    d = json.load(open("results/aux_reliance.json"))
    # order by the drop (with - without), largest first
    items = sorted(d.items(), key=lambda kv: -(kv[1]["with"] - kv[1]["without"]))
    params = [k for k, _ in items]
    wi = [v["with"] for _, v in items]
    wo = [v["without"] for _, v in items]
    y = np.arange(len(params))[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for yi, p, a, b in zip(y, params, wi, wo):
        red = p in REDOX
        col = fs.color("redox_active") if red else fs.color("conservative_ion")
        ax.plot([b, a], [yi, yi], color="#bbb", lw=1.5, zorder=1)
        ax.scatter(a, yi, s=70, color=col, edgecolor="#333", linewidth=0.6, zorder=3,
                   marker="o", label="with env. context" if yi == y[0] else None)
        ax.scatter(b, yi, s=70, color=col, edgecolor="#333", linewidth=0.6, zorder=3,
                   marker="X", label="without env. context" if yi == y[0] else None)
    ax.axvline(0, color="#888", ls=":", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels(
        [p + (r" $^\ast$" if p in REDOX else "") for p in params])
    ax.set_xlabel(r"Masked-reconstruction $R^2$")
    ax.set_title("Redox species rely on environmental context; conservative ions do not")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout(); fig.savefig("figures/fig10_aux_reliance.png")
    print("wrote figures/fig10_aux_reliance.png")


def fig_layerwise():
    d = json.load(open("results/layerwise_decodability.json"))
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for p, series in d.items():
        red = p in REDOX
        col = fs.color(p) if p in fs.COLORS else (fs.color("redox_active") if red else fs.color("conservative_ion"))
        x = np.arange(1, len(series) + 1)
        ax.plot(x, series, marker="o" if red else "s", ms=5, lw=2.0,
                color=col, ls="-" if red else "--",
                label=p + (r" $^\ast$" if red else ""))
    ax.axhline(0, color="#888", ls=":", lw=1.0)
    ax.set_xlabel("Transformer layer (depth)")
    ax.set_ylabel(r"[CLS] decodability ($R^2$, 3-fold CV)")
    ax.set_title("Depth consolidates conservative chemistry and discards redox signal")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5,
              title="Species")
    fig.tight_layout(); fig.savefig("figures/fig11_layerwise.png")
    print("wrote figures/fig11_layerwise.png")


if __name__ == "__main__":
    fs.apply_theme()
    fig_aux()
    fig_layerwise()
