"""Figure 1 of the evaluation reframe: data coverage and study design (one full-page figure).

(a) Global Robinson map on the Seminar-paper grey basemap (land #f3f2ef, borders #c9c8c3).
    Held-out countries are tinted by the data source of their held-out region, pretraining-site
    density is a grey hexbin layer, and two zoom insets (Europe, Bengal Basin) show the
    temporal-pair sites coloured by source with each region's leave-one-region-out cell count.
(b) Workflow with icons: databases -> harmonized corpus -> a method x axis matrix (inputs,
    pretraining, which axes score each method) -> a schematic of the data splits behind each
    axis. Method colours come from evalstyle, so they match Figs. 2-5.

Every count is read from data or results files. Outputs figures/fig_eval_1_design.{pdf,png}.
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyBboxPatch, Patch, Polygon, Rectangle

import evalstyle as es
import make_global_maps_v2 as gm

ROOT, OUT = gm.ROOT, gm.OUT
region_of = __import__("19_region_transfer_encoder").region_of
es.apply()

LAND, BORDER = "#f3f2ef", "#c9c8c3"            # Seminar-paper basemap
INK2, INK3 = "#52514e", "#8a8984"
SRC_COL = {"NWIS": "#5B6C9D", "ADES": "#8E5A9B", "EEA": "#3A8F85", "GEMStat": "#B08A2E",
           "Bangladesh": es.VERMILLION}
DENSITY = LinearSegmentedColormap.from_list("dens", ["#dcdad3", "#8f8d86", "#2e2d2a"])
W, H = es.DOUBLE, 7.5
H_A = 6.6            # panel (a) is laid out for a 6.6 in figure; A() keeps it fixed in inches


def A(y):
    """Figure fraction for a panel-(a) y position given in the 6.6 in layout."""
    return 1 - (1 - y) * H_A / H


def Ah(h):
    return h * H_A / H

# ---------------------------------------------------------------- data, all counts from files
meta = pd.read_parquet(ROOT / "data/processed/genesis_pretrain_meta.parquet",
                       columns=["source", "lat", "lon"])
src_n = meta.source.value_counts()


def count(*keys):
    return int(sum(v for k, v in src_n.items() if any(s in str(k).upper() for s in keys)))


N = {"GEMStat": count("GEMS"), "ADES": count("ADES", "BRGM"), "EEA": count("EEA", "WISE"),
     "NWIS": count("NWIS", "USGS")}
N_TOTAL = len(meta)
assert sum(N.values()) == N_TOTAL, (N, N_TOTAL)
pairs = pd.read_parquet(ROOT / "data/processed/genesis_temporal_pairs_meta.parquet",
                        columns=["lat", "lon", "source", "country"])
N_PAIRS = len(pairs)
bd = pd.read_parquet(ROOT / "data/processed/genesis_held_out_bd_meta.parquet")
N_BD, N_BD_SITES = len(bd), bd.site_id.nunique()
bdp = pd.read_parquet(ROOT / "data/processed/genesis_held_out_bd_pairs_meta.parquet")
cells = json.load(open(ROOT / "results/conformal_coverage.json"))["methods"]["rf"]["cells"]
ncell = Counter(c["region"] for c in cells)
N_TEST_A = 1173                        # site-disjoint groundwater validation samples (src/56 log)

pairs["region"] = [region_of(s, c) for s, c in zip(pairs.source, pairs.country)]
pairs = pairs[pairs.region.isin(ncell)]


def source_of(region):
    return {"USA": "NWIS", "France": "ADES", "EU": "EEA", "Bangladesh": "Bangladesh"}.get(
        region, "GEMStat")


pairs["src"] = pairs.region.map(source_of)
bdp = bdp[["lat", "lon"]].dropna().assign(region="Bangladesh", src="Bangladesh")

# held-out country polygons, priority national > GEMStat > EEA, coloured by source
GEM_NAME = {"Netherlands (-the )": "Netherlands"}
ctry = gm.CTRY.copy()
ctry["iso"] = gm.gpd.read_file(gm.GIS / "ne_50m_admin_0_countries.shp")[
    "ISO_A2_EH"].reindex(ctry.index).values
held = {}
for iso in pairs.loc[pairs.region == "EU", "country"].unique():
    held.update({n: "EEA" for n in ctry.loc[ctry.iso == iso, "NAME"]})
for r in ncell:
    if r.startswith("GEMStat:"):
        held[GEM_NAME.get(r[8:], r[8:])] = "GEMStat"
held.update({"United States of America": "NWIS", "France": "ADES", "Bangladesh": "Bangladesh"})
ctry["src"] = ctry.NAME.map(held)
ctry_r = ctry.to_crs(gm.ROB)

fig = plt.figure(figsize=(W, H))


def basemap(ax, alpha):
    gm.LANDr.plot(ax=ax, color=LAND, ec="none", zorder=1, rasterized=True)
    for s, g in ctry_r.dropna(subset=["src"]).groupby("src"):
        g.plot(ax=ax, color=SRC_COL[s], alpha=alpha, ec="none", zorder=1.2, rasterized=True)
    ctry_r.boundary.plot(ax=ax, color=BORDER, lw=0.25, zorder=1.5, rasterized=True)
    gm.LANDr.boundary.plot(ax=ax, color=BORDER, lw=0.35, zorder=1.6, rasterized=True)
    ax.set_axis_off()
    ax.set_aspect("equal")


def lims(ax, lon0, lon1, lat0, lat1):
    xs, ys = gm.to_rob([lon0, lon1, lon0, lon1], [lat0, lat0, lat1, lat1])
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))
    return min(xs), max(xs), min(ys), max(ys)


# ================================================================ (a) global map
_xr = 2 * gm.to_rob(180, 0)[0]
_yr = gm.to_rob(0, 84)[1] - gm.to_rob(0, -46)[1]
GA_W = 0.645
GA_H = GA_W * W * _yr / _xr / H_A
ga = fig.add_axes([0.0, A(0.975 - GA_H), GA_W, Ah(GA_H)])
basemap(ga, 0.30)
sx, sy = gm.to_rob(meta.lon.values, meta.lat.values)
ok = np.isfinite(sx) & np.isfinite(sy)
hb = ga.hexbin(sx[ok], sy[ok], gridsize=260, bins="log", mincnt=1, cmap=DENSITY,
               linewidths=0, alpha=0.75, zorder=2, rasterized=True)
x, y = gm.to_rob(bdp.lon.values, bdp.lat.values)
ga.scatter(x, y, s=1.2, color=es.VERMILLION, lw=0, zorder=4, rasterized=True)
lims(ga, -180, 180, -46, 84)
xl0, _ = gm.to_rob(-180, 0)
xl1, _ = gm.to_rob(180, 0)
ga.set_xlim(xl0, xl1)

# inset frames on the global map, labelled to match the insets
INSETS = {"Europe": (-11, 34, 34, 71.5), "Bengal Basin": (66, 99, 7, 31)}
for name, (lo0, lo1, la0, la1) in INSETS.items():
    xs, ys = gm.to_rob([lo0, lo1, lo1, lo0], [la0, la0, la1, la1])
    ga.add_patch(Polygon(np.c_[xs, ys], closed=True, fill=False, ec=es.INK, lw=0.6, zorder=5))
    ga.text(xs[3], ys[3], f" {name}", fontsize=5.8, va="bottom", ha="left", zorder=6)
for r, (lo, la) in {"USA": (-128, 28), "GEMStat:Mexico": (-124, 14),
                    "GEMStat:Canada": (-104, 64)}.items():
    g = pairs[pairs.region == r]
    x0, y0 = gm.to_rob(float(np.median(g.lon)), float(np.median(g.lat)))
    x1, y1 = gm.to_rob(lo, la)
    name = {"USA": "USA (NWIS)"}.get(r, r.replace("GEMStat:", ""))
    ga.annotate(f"{name}  {ncell[r]}", (x0, y0), (x1, y1), fontsize=5.8, ha="right",
                va="center", zorder=7,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=INK2, shrinkA=0, shrinkB=1),
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
fig.text(0.003, A(0.995), "(a)", fontsize=8, fontweight="bold", va="top")

# legend and density bar below the global map
leg = [Patch(fc=SRC_COL[s], alpha=0.75, ec="none", label=l) for s, l in
       [("NWIS", "USGS NWIS"), ("ADES", "French ADES"), ("EEA", "EEA Waterbase"),
        ("GEMStat", "GEMStat"), ("Bangladesh", "Bangladesh field data")]]
fig.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.005, A(0.975 - GA_H - 0.012)), ncol=2,
           frameon=False, fontsize=5.8, handlelength=1.1, handletextpad=0.35,
           columnspacing=0.9, title="Held-out region by data source (number = cells)",
           title_fontsize=5.8, alignment="left")
cax = fig.add_axes([0.02, A(0.545), 0.22, Ah(0.009)])
cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
cb.set_label("Pretraining sites per hexagon", fontsize=5.4, labelpad=1)
cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
cb.outline.set_linewidth(0.3)

# ---------------------------------------------------------------- insets
LAB = {"EU": ("EU (EEA)", 11.5, 49.6), "France": ("France ADES", 1.5, 47.6),
       "GEMStat:France": ("France GEMStat", 1.0, 45.0), "GEMStat:Ireland": ("Ireland", -8.0, 53.2),
       "GEMStat:Netherlands (-the )": ("Netherlands", 1.5, 55.2),
       "GEMStat:Sweden": ("Sweden", 15.0, 62.5), "GEMStat:Lithuania": ("Lithuania", 24.0, 55.4),
       "GEMStat:Poland": ("Poland", 19.2, 52.0), "GEMStat:Italy": ("Italy", 13.0, 42.6),
       "GEMStat:Greece": ("Greece", 22.0, 39.4), "GEMStat:India": ("India", 79.0, 21.0),
       "Bangladesh": ("Bangladesh", 93.6, 25.6)}
for (name, (lo0, lo1, la0, la1)), rect, letter in zip(
        INSETS.items(), ([0.66, A(0.505), 0.335, Ah(0.47)],
                         [0.40, A(0.505), 0.245, Ah(0.975 - GA_H - 0.515)]), "bc"):
    ia = fig.add_axes(rect)
    basemap(ia, 0.45)
    pts = pd.concat([pairs[["lat", "lon", "src"]], bdp[["lat", "lon", "src"]]])
    for s, g in pts.groupby("src"):
        x, y = gm.to_rob(g.lon.values, g.lat.values)
        ia.scatter(x, y, s=0.45 if s != "Bangladesh" else 1.4, color=SRC_COL[s], lw=0,
                   alpha=0.85, zorder=3, rasterized=True)
    lims(ia, lo0, lo1, la0, la1)
    ia.set_anchor("NW" if name == "Bengal Basin" else "N")
    for r, (lab, lo, la) in LAB.items():
        if not (lo0 <= lo <= lo1 and la0 <= la <= la1):
            continue
        x, y = gm.to_rob(lo, la)
        ia.text(x, y, f"{lab} {ncell[r]}", fontsize=5.6, ha="center", va="center", zorder=6,
                color=es.VERMILLION if r == "Bangladesh" else es.INK,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
    for sp in ia.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
        sp.set_edgecolor(es.INK)
    ia.set_axis_on()
    ia.set_xticks([])
    ia.set_yticks([])
    ia.text(0.015, 0.97, name, transform=ia.transAxes, fontsize=6, va="top", fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))

# ================================================================ (b) workflow
# Laid out on a 0-100 grid; panel (b) is ~4 in tall, so one y unit is ~2.9 pt. Row heights
# and bar widths below were sized for the text they hold, at 6.2-7.2 pt.
B_TOP = A(0.47)
bx = fig.add_axes([0.0, 0.0, 1.0, B_TOP])
bx.set_xlim(0, 100)
bx.set_ylim(0, 100)
bx.set_axis_off()
bx.text(0.0, 100, "(b)", fontsize=8, fontweight="bold", va="top")
ASPECT = W / (H * B_TOP)                        # x-unit length / y-unit length on paper
F_HEAD, F_NAME, F_TXT, F_SMALL = 7.2, 6.8, 6.5, 6.2


def box(x, y, w, h, fc=LAND, ec=BORDER, ls="-", lw=0.6, r=0.9):
    bx.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, mutation_aspect=ASPECT))


def arrow(x0, y0, x1, y1, col=INK3):
    bx.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=col,
                                                        shrinkA=0, shrinkB=0))


def cylinder(x, y, w, h, col):
    """Database icon: body plus elliptical top, sized in axis units."""
    e = h * 0.28
    bx.add_patch(Rectangle((x, y + e / 2), w, h - e, fc=col, ec="none"))
    bx.add_patch(Ellipse((x + w / 2, y + e / 2), w, e, fc=col, ec="none"))
    bx.add_patch(Ellipse((x + w / 2, y + h - e / 2), w, e, fc="white", ec=col, lw=0.6))


def droplet(x, y, s, col):
    """Water-drop icon, round part centred at (x, y); s = width in x units."""
    r = s / 2
    t = np.linspace(-np.pi / 6, 7 * np.pi / 6, 50)
    px = list(x + r * np.cos(t)) + [x]
    py = list(y - r * np.sin(t) * ASPECT) + [y + 2.1 * r * ASPECT]
    bx.add_patch(Polygon(np.c_[px, py], closed=True, fc=col, ec="none"))


def badge(x, y, letter, r=1.35):
    bx.add_patch(Ellipse((x, y), 2 * r, 2 * r * ASPECT, color=es.INK))
    bx.text(x, y, letter, ha="center", va="center", fontsize=6.4, color="white",
            fontweight="bold")


def head(x, text):
    bx.text(x, 94.5, text, fontsize=F_HEAD, fontweight="bold", color=INK2, va="center")


# ---------------------------------------------------------------- column 1: databases
C1, C1W = 1.0, 19.0
head(C1, "Monitoring data")
for i, (k, name) in enumerate([("GEMStat", "GEMStat v3"), ("ADES", "French ADES"),
                               ("EEA", "EEA Waterbase"), ("NWIS", "USGS NWIS")]):
    y = 76 - i * 16.0
    box(C1, y, C1W, 13.0)
    cylinder(C1 + 1.1, y + 2.6, 2.4, 7.8, SRC_COL[k])
    bx.text(C1 + 4.6, y + 8.8, name, fontsize=F_NAME, fontweight="bold", va="center")
    bx.text(C1 + 4.6, y + 4.0, f"{N[k]:,} samples", fontsize=F_TXT, color=INK2, va="center")
    arrow(C1 + C1W, y + 6.5, 22.6, y + 6.5)
box(C1, 3.0, C1W, 21.0, fc="white", ec=es.VERMILLION, ls="--", lw=0.8)
droplet(C1 + 2.3, 18.2, 2.2, es.VERMILLION)
bx.text(C1 + 4.4, 19.2, "Bangladesh field data", fontsize=F_NAME, fontweight="bold",
        va="center", color=es.VERMILLION)
bx.text(C1 + 1.2, 13.4, f"{N_BD:,} samples", fontsize=F_TXT, color=INK2, va="center")
bx.text(C1 + 1.2, 9.6, f"{N_BD_SITES:,} sites", fontsize=F_TXT, color=INK2, va="center")
bx.text(C1 + 1.2, 5.8, "never used for training", fontsize=F_SMALL, color=es.VERMILLION,
        va="center", style="italic")

# ---------------------------------------------------------------- column 2: corpus
C2, C2W = 23.0, 13.0
head(C2, "Harmonized corpus")
box(C2, 21.0, C2W, 68.0, fc="#e7e6e2")
droplet(C2 + C2W / 2, 82.0, 2.6, "#7c7a73")
for k, (big, small) in enumerate([(f"{N_TOTAL:,}", "samples"),
                                  ("20", "chemical\nparameters"),
                                  ("18", "environmental\nrasters"),
                                  (f"{N_PAIRS:,}", "temporal pairs\n≥ 5 years apart")]):
    yy = 74.0 - k * 13.5
    bx.text(C2 + 1.0, yy, big, fontsize=7.6, fontweight="bold", va="center")
    bx.text(C2 + 1.0, yy - 2.8, small, fontsize=F_SMALL, color=INK2, va="top",
            linespacing=1.0)
arrow(C2 + C2W, 55.0, 39.0, 55.0)

# ---------------------------------------------------------------- column 3: method x axis matrix
X0, X1 = 39.5, 99.5
head(X0, "Methods and the axes that score them")
XI, XP = X0 + 21.0, X0 + 32.5
AXC = {"A": X0 + 45.0, "B": X0 + 49.3, "C": X0 + 53.6, "D": X0 + 57.9}
ROWS = [("No pretraining", None),
        ("logreg", ("Logistic regression", "chemistry", "none", "BC")),
        ("knn_space", ("Spatial kNN", "coordinates", "none", "BC")),
        ("rf_aux", ("Raster-only RF", "rasters", "none", "BC")),
        ("rf", ("Random forest", "chemistry*", "none", "ABCD")),
        ("xgb", ("XGBoost / gradient boosting", "chemistry*", "none", "ABCD")),
        ("qkernel", ("Quantum kernel (simulated)", "chem., 12 PCs", "none", "ABCD")),
        ("Generic pretraining", None),
        ("tabpfn", ("TabPFN", "chemistry*", "synthetic tables", "ABCD")),
        ("Domain pretraining (GENESIS)", None),
        ("frozen_base", ("Frozen probe", "chem. + rasters", "2.1 M samples", "BC")),
        ("finetune_base", ("Fine-tuned classifier", "chem. + rasters", "2.1 M samples", "BC")),
        ("fm", ("Masked reconstruction", "chem. + rasters", "2.1 M samples", "AD"))]
TOP = 88.5
for lab, x in (("Method", X0 + 0.5), ("Inputs", XI), ("Pretraining", XP)):
    bx.text(x, TOP, lab, fontsize=F_TXT, color=INK2, va="center", fontweight="bold")
for L, x in AXC.items():
    badge(x, TOP, L)
RULE1 = TOP - 4.4                               # clear of the badges (radius ~3.3 y units)
bx.plot([X0, X1], [RULE1, RULE1], color=INK2, lw=0.5)
y = RULE1 - 3.0
RH, GH = 4.2, 3.6
for key, row in ROWS:
    if row is None:
        bx.text(X0 + 0.5, y, key, fontsize=F_SMALL, color=INK3, style="italic", va="center")
        y -= GH
        continue
    name, inp, pre, axes = row
    sw = 1.3
    bx.add_patch(Rectangle((X0 + 0.5, y - sw * ASPECT / 2), sw, sw * ASPECT,
                           fc=es.METHOD[key][1], ec="none"))
    bx.text(X0 + 2.6, y, name, fontsize=F_TXT, va="center")
    bx.text(XI, y, inp, fontsize=F_SMALL, va="center", color=INK2)
    bx.text(XP, y, pre, fontsize=F_SMALL, va="center", color=INK2)
    for L, x in AXC.items():
        d = 1.5 if L in axes else 0.55
        bx.add_patch(Ellipse((x, y), d, d * ASPECT, fc=es.INK if L in axes else BORDER,
                             ec="none"))
    y -= RH
RULE2 = y + RH - 2.6
bx.plot([X0, X1], [RULE2, RULE2], color=INK2, lw=0.5)
bx.text(X0 + 0.5, RULE2 - 2.8, "* plus the rasters in axes A and D.   A inferability · "
        "B transfer · C calibration · D physical consistency", fontsize=6.0, color=INK3,
        va="center")


# ---------------------------------------------------------------- column 3, bottom: splits
def bar(x, y, segs, h=3.6):
    """segs = [(width, colour, label, text colour)]; returns the right edge."""
    for w, c, lab, tc in segs:
        bx.add_patch(Rectangle((x, y), w, h, fc=c, ec="white", lw=0.6))
        if lab:
            bx.text(x + w / 2, y + h / 2, lab, fontsize=F_SMALL, ha="center", va="center",
                    color=tc)
        x += w
    return x


SPLIT = RULE2 - 8.5
bx.text(X0, SPLIT, "How the data are split", fontsize=F_HEAD, fontweight="bold", color=INK2,
        va="center")
YB, YA = SPLIT - 11.5, 3.5
XB = X0 + 5.5
bx.text(X0, YB + 1.8, "B, C", fontsize=F_TXT, fontweight="bold", va="center")
regs = len(ncell)
e1 = bar(XB, YB, [(1.45, "#d6d4cd", "", es.INK)] * (regs - 1) + [(1.45, es.INK, "", "")])
bx.text(XB, YB + 5.6, f"{regs} regions, one held out ({len(cells)} cells)",
        fontsize=F_SMALL, color=INK2, va="center")
arrow(e1 + 0.8, YB + 1.8, e1 + 3.6, YB + 1.8)
XS = e1 + 4.2
bar(XS, YB, [(11.0, "#d6d4cd", "train 70%", es.INK), (4.6, "#bdbbb4", "val", es.INK),
             (4.6, "#a6a49d", "cal", es.INK), (5.6, "#8f8d86", "test", "white")])
bx.text(XS, YB + 5.6, "remaining regions, per cell", fontsize=F_SMALL, color=INK2,
        va="center")
bx.text(X0, YA + 1.8, "A, D", fontsize=F_TXT, fontweight="bold", va="center")
e3 = bar(XB, YA, [(12.5, "#d6d4cd", "training sites", es.INK),
                  (13.5, "#7c7a73", f"{N_TEST_A:,} unseen wells", "white")])
BX0 = e3 + 1.5
bar(BX0, YA, [(9.5, es.VERMILLION, "Bangladesh", "white")])
# Bangladesh feeds the test sets only, routed below everything else
bdx = BX0 + 4.75
bx.plot([C1 + C1W / 2, C1 + C1W / 2, bdx], [3.0, 1.0, 1.0], color=es.VERMILLION, lw=0.8,
        ls="--")
bx.annotate("", (bdx, YA), (bdx, 1.0), arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                                        color=es.VERMILLION, shrinkA=0,
                                                        shrinkB=0))
bx.text(29.5, 1.6, "test sets only", fontsize=F_SMALL, color=es.VERMILLION, ha="center",
        va="bottom", style="italic")

es.save(fig, OUT, "fig_eval_1_design")
