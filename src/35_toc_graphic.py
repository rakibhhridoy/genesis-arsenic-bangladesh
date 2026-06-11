#!/usr/bin/env python3
"""EST Table-of-Contents graphic (3.25 x 1.75 in, 300 dpi).

Captures the paper's headline: read the foundation model as an instrument, and
when benchmarked fairly its apparent advantage over a weak baseline vanishes
against a strong one -> parity. Built from results/baseline_strength.json.

Usage:  python src/35_toc_graphic.py
Writes: figures/toc_graphic.png  and  figures/toc_graphic.tiff
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

fs.apply_theme()
plt.rcParams["savefig.bbox"] = "standard"   # exact canvas size for the TOC (not 'tight')
g = json.load(open("results/baseline_strength.json"))["group_summary"]["REDOX_AsFeMnPO4"]
order = [("rf_default", "default\nRF"), ("rf_reg", "regularized\nRF"), ("histgb", "HistGB"),
         ("best_tree", "best\ntree")]
deltas = [g[k]["delta"] for k, _ in order]
sig = [g[k]["significant"] for k, _ in order]

fig, ax = plt.subplots(figsize=(3.25, 1.75), dpi=300)
x = np.arange(len(order))
# weak->strong ordered palette; faded for non-significant
barcols = [fs.NAVY, fs.TEAL, fs.ORANGE, fs.GREY]
ax.bar(x, deltas, 0.66, color=barcols, hatch=["", "....", "----", "xxxx"],
       edgecolor="white", linewidth=0.6, zorder=3)
ax.grid(False)
ax.axhline(0, color="#444", lw=0.7)
ax.set_xticks(x)
ax.set_xticklabels([lab for _, lab in order], fontsize=6)
ax.set_yticks([0.0, 0.03])
ax.set_yticklabels(["0", "+0.03"], fontsize=6)
ax.set_ylabel(r"$\Delta$AUC vs tree", fontsize=6.5)
ax.tick_params(length=2, pad=1)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("A geochemical foundation model, benchmarked fairly",
             fontsize=7.2, weight="bold", pad=3)
ax.text(0.5, 0.93, "apparent win over a weak baseline $\\rightarrow$ parity with a strong one",
        transform=ax.transAxes, ha="center", va="top", fontsize=5.8, style="italic", color="#555")
ax.annotate("", xy=(3, 0.004), xytext=(0, 0.030),
            arrowprops=dict(arrowstyle="->", color="#8c2d2d", lw=1.0, connectionstyle="arc3,rad=-0.15"))
ax.set_ylim(-0.004, 0.045)
fig.tight_layout(pad=0.4)
Path("figures").mkdir(exist_ok=True)
# exact 3.25 x 1.75 in canvas (no tight bbox, which would expand it)
fig.savefig("figures/toc_graphic.png", dpi=300, bbox_inches=None)
fig.savefig("figures/toc_graphic.tiff", dpi=300, bbox_inches=None, pil_kwargs={"compression": "tiff_lzw"})
print("wrote figures/toc_graphic.png and .tiff")
# report rendered size
from PIL import Image
for ext in ("png", "tiff"):
    im = Image.open(f"figures/toc_graphic.{ext}")
    print(f"  {ext}: {im.size[0]}x{im.size[1]} px = {im.size[0]/300:.2f}x{im.size[1]/300:.2f} in @300dpi")
