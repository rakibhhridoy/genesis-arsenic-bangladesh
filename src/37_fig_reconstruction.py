#!/usr/bin/env python3
"""Fig 9 rebuild: masked-reconstruction R2 as a model-internal probe.

Per-parameter mask-one reconstruction R2 on the corpus test split vs the
Bangladesh held-out set. Redox-active species (red, hatched) collapse on
Bangladesh; conservative/bulk ions (navy) are stable. Vibrant shared theme.

Reads results/reconstruction_probe.json. Writes figures/fig9_reconstruction_ood.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

REDOX = {"As", "Fe", "Mn", "PO4", "Eh", "U", "SO4", "NO3"}  # redox-active/-sensitive
# display order: redox-active first (by corpus R2), then conservative/bulk
ORDER = ["Eh", "As", "U", "Mn", "PO4", "Fe", "SO4", "NO3",
         "Ca", "HCO3", "Mg", "Na", "K", "Cl", "TDS", "EC"]


def main():
    fs.apply_theme()
    d = json.load(open("results/reconstruction_probe.json"))
    corpus, bd = d["corpus"], d["bangladesh"]

    params = [p for p in ORDER if p in corpus and p in bd]
    cr = [corpus[p]["r2"] for p in params]
    br = [bd[p]["r2"] for p in params]

    y = np.arange(len(params))[::-1]   # top-to-bottom
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    h = 0.38
    for yi, p, c, b in zip(y, params, cr, br):
        red = p in REDOX
        col = fs.color("redox_active") if red else fs.color("conservative_ion")
        hat = fs.hatch("redox_active") if red else fs.hatch("conservative_ion")
        ax.barh(yi + h/2, c, h, color=col, hatch=hat, **fs.BAR)            # corpus
        ax.barh(yi - h/2, b, h, color=col, hatch=hat, alpha=0.45, **fs.BAR)  # bangladesh
    ax.axvline(0, color="#444", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels(
        [p + (r" $^\ast$" if p in REDOX else "") for p in params])
    ax.set_xlabel(r"Masked-reconstruction $R^2$  (mask-one, z-scored)")
    ax.set_title("Redox chemistry becomes unreconstructable on the reducing Bengal aquifer")
    ax.set_xlim(min(-5, min(br) - 0.3), 1.05)
    ax.grid(axis="y", visible=False)

    from matplotlib.patches import Patch
    leg = [Patch(facecolor=fs.color("redox_active"), hatch=fs.hatch("redox_active"),
                 edgecolor="white", label=r"redox-active ($^\ast$)"),
           Patch(facecolor=fs.color("conservative_ion"), edgecolor="white",
                 label="conservative / bulk"),
           Patch(facecolor="#888", label="corpus (solid) vs Bangladesh (faded)")]
    ax.legend(handles=leg, loc="lower left", fontsize=8)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig9_reconstruction_ood.png")
    print("wrote figures/fig9_reconstruction_ood.png")
    print(f"  corpus R2>0.9: {[p for p in params if corpus[p]['r2']>0.9]}")
    print(f"  Eh corpus {corpus['Eh']['r2']:.2f} -> BD {bd['Eh']['r2']:.2f}")


if __name__ == "__main__":
    main()
