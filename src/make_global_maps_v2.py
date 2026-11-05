#!/usr/bin/env python3
"""
Nature-tier upgrade of the two global figures (static, publication PDF):
  Fig 1  fig_global_arsenic_podgorski : observed As hazard, by continent
  Fig 2  fig_global_transfer          : 7-contaminant leave-one-region-out AUC

Upgrades over v1 (src/make_global_maps.py):
  - 50 m Natural Earth basemap: crisp coastlines, subtle ocean fill, neutral land
  - perceptually-uniform, colour-blind-safe colormaps (colorcet)
  - finer smoothing grid for hotspot detail; rasterised field + vector text
  - named-basin hotspot annotations on the global panel
  - larger, cleaner legends and titles

Honesty unchanged: hazard = OBSERVED well-exceedance fraction (not a model
prediction); transfer = MEASURED leave-one-region-out AUC at 15 regions only.
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize, TwoSlopeNorm, BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.cm as mcm
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString
from pyproj import Transformer
import colorcet as cc

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
GIS = ROOT / "data/gis/ne_50m"
OUT = ROOT / "figures"
ROB = "ESRI:54030"
T = Transformer.from_crs("EPSG:4326", ROB, always_xy=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 450,
})

OCEAN = "#eaf0f4"      # very pale blue-grey sea
LAND = "#e7e4dd"       # warm neutral land
COAST = "#9aa6ad"      # coastline / border
GRAT = "#d9dee2"
INK = "#222222"
# hazard: DISCRETE bins, dark blue (low) -> dark red (high), through a saturated
# purple middle (no washed-out tone), so every bin contrasts with the beige land
HAZ_BINS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
HAZ = ListedColormap(["#0a2f6e", "#2b6fb0", "#7b4ea3", "#cf3a2c", "#6d0a13"])
HAZ_NORM = BoundaryNorm(HAZ_BINS, HAZ.N)
AUC = cc.cm["CET_D1A_r"]               # high AUC -> blue (transfers), low -> red (fails)

print("Loading 50 m basemap ...")
SIMP = 0.04   # deg; thins coastline vertices ~10x, invisible at print size


def _simplify(gdf):
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.simplify(SIMP, preserve_topology=True)
    return gdf


LANDg = _simplify(gpd.read_file(GIS / "ne_50m_land.shp"))
LANDg = LANDg[LANDg.geometry.bounds.miny > -60]      # drop Antarctica
OCEANg = _simplify(gpd.read_file(GIS / "ne_50m_ocean.shp"))
CTRY = gpd.read_file(GIS / "ne_50m_admin_0_countries.shp")[["NAME", "geometry"]]
CTRY = _simplify(CTRY[CTRY.NAME != "Antarctica"])
LAND_UNION = LANDg.geometry.buffer(0).union_all()
LANDr = LANDg.to_crs(ROB); OCEANr = OCEANg.to_crs(ROB); CTRYr = CTRY.to_crs(ROB)


def to_rob(lon, lat):
    return T.transform(np.asarray(lon), np.asarray(lat))


def globe(ax):
    """Robinson base: ocean fill, land, crisp coast, light graticule."""
    grat = []
    for lo in range(-180, 181, 30):
        grat.append([(lo, la) for la in np.linspace(-85, 85, 60)])
    for la in range(-60, 61, 30):
        grat.append([(lo, la) for lo in np.linspace(-180, 180, 120)])
    gpd.GeoSeries([LineString(g) for g in grat], crs="EPSG:4326").to_crs(
        ROB).plot(ax=ax, color=GRAT, lw=0.25, zorder=0, rasterized=True)
    OCEANr.plot(ax=ax, color=OCEAN, ec="none", zorder=0.5, rasterized=True)
    LANDr.plot(ax=ax, color=LAND, ec=COAST, lw=0.18, zorder=1, rasterized=True)
    ax.set_axis_off(); ax.set_aspect("equal")


def _edges(lo, hi, res):
    e = np.arange(lo, hi, res)
    return np.append(e, hi) if e[-1] < hi else e


def smooth_field(lon, lat, values=None, res=0.3, val_s=2.0, sup_s=3.0, thr=0.004):
    lon, lat = np.asarray(lon), np.asarray(lat)
    le, la = _edges(-180, 180, res), _edges(-90, 90, res)
    cnt, _, _ = np.histogram2d(lat, lon, [la, le])
    sup = gaussian_filter((cnt > 0).astype(float), sup_s) > thr
    if values is None:
        f = np.ma.masked_where(~sup | (cnt == 0), np.log10(gaussian_filter(cnt, val_s) + 1))
    else:
        vs, _, _ = np.histogram2d(lat, lon, [la, le], weights=np.asarray(values))
        with np.errstate(invalid="ignore"):
            f = np.where(gaussian_filter(cnt, val_s) > 0,
                         gaussian_filter(vs, val_s) / gaussian_filter(cnt, val_s), 0.0)
        f = np.ma.masked_where(~sup, f)
    clon = 0.5 * (le[:-1] + le[1:]); clat = 0.5 * (la[:-1] + la[1:])
    from shapely.vectorized import contains
    onland = contains(LAND_UNION, *np.meshgrid(clon, clat))
    return le, la, np.ma.masked_where(~onland, f)


# ============================================================ FIGURE 1
CONTINENTS = [("North America", (-168, -52, 7, 73)),
              ("South America", (-82, -34, -56, 13)),
              ("Africa", (-19, 52, -36, 38)),
              ("Europe", (-12, 45, 34, 72)),
              ("Asia", (40, 150, 5, 78)),
              ("Australia & Oceania", (110, 180, -48, 0))]
HOTSPOTS = [("Bengal\nBasin", 89, 24), ("Indus\nPlain", 71, 30),
            ("Mekong &\nRed River", 106, 16), ("North China\nPlain", 116, 36),
            ("Pampas", -63, -33), ("U.S.\nSouthwest", -110, 35)]


def fig_arsenic():
    print("Fig 1: arsenic hazard (Nature-tier) ...")
    p = ROOT / "data/Podgorski/arsenic_data/data/arsenic_concentrations_predictor_data.csv"
    a = pd.read_csv(p, usecols=["Latitude", "Longitude", "As_ppb"]).dropna()
    a["exc"] = (a.As_ppb >= 10).astype(float)
    le, la, F = smooth_field(a.Longitude, a.Latitude, a.exc, res=0.28, val_s=1.9)

    fig = plt.figure(figsize=(7.4, 8.8))
    gs = GridSpec(4, 3, height_ratios=[1.65, 0.8, 0.8, 0.12], hspace=0.14,
                  wspace=0.05, left=0.012, right=0.988, top=0.985, bottom=0.025)
    axg = fig.add_subplot(gs[0, :]); globe(axg)
    LO, LA = np.meshgrid(le, la); X, Y = T.transform(LO, LA)
    pm = axg.pcolormesh(X, Y, F, cmap=HAZ, norm=HAZ_NORM, shading="auto",
                        zorder=2, rasterized=True)
    for i, (_, e) in enumerate(CONTINENTS, 1):
        cx, cy = T.transform((e[0] + e[1]) / 2, (e[2] + e[3]) / 2)
        axg.annotate(str(i), (cx, cy), fontsize=6.5, fontweight="bold",
                     color=INK, ha="center", va="center", zorder=7,
                     bbox=dict(boxstyle="circle,pad=0.14", fc="white", ec=INK, lw=0.4))
    for name, lon, lat in HOTSPOTS:                     # named-basin callouts
        x, y = T.transform(lon, lat)
        axg.annotate(name, (x, y), fontsize=6.0, color="white", ha="center",
                     va="center", fontweight="bold", linespacing=0.9, zorder=8,
                     path_effects=[pe.withStroke(linewidth=1.7, foreground="#1a1a1a")])
    # In-panel title and subtitle removed (2026-09-06, 2026-09-12) at the author's
    # request. The same information is carried by the LaTeX caption.

    cells = [gs[1, 0], gs[1, 1], gs[1, 2], gs[2, 0], gs[2, 1], gs[2, 2]]
    for i, ((name, e), c) in enumerate(zip(CONTINENTS, cells), 1):
        ax = fig.add_subplot(c)
        sub = CTRY.cx[e[0]:e[1], e[2]:e[3]]
        gpd.clip(OCEANg, (e[0], e[2], e[1], e[3])).plot(ax=ax, color=OCEAN, zorder=0)
        sub.plot(ax=ax, color=LAND, ec=COAST, lw=0.2, zorder=1)
        ax.pcolormesh(le, la, F, cmap=HAZ, norm=HAZ_NORM, shading="auto",
                      zorder=2, rasterized=True)
        ax.set_xlim(e[0], e[1]); ax.set_ylim(e[2], e[3])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect(1 / np.cos(np.radians(np.clip((e[2] + e[3]) / 2, -60, 60))))
        for s in ax.spines.values():
            s.set_edgecolor(INK); s.set_linewidth(0.5)
        ax.set_title(f"{i}  {name}", fontsize=7.3, fontweight="bold", loc="left",
                     color=INK, pad=2)

    band = fig.add_subplot(gs[3, :]); band.set_axis_off()
    cb = fig.colorbar(pm, cax=band.inset_axes([0.28, 0.45, 0.44, 0.42]),
                      orientation="horizontal", ticks=HAZ_BINS, spacing="uniform")
    cb.set_label("Observed fraction of wells > 10 µg/L (arsenic hazard)", fontsize=7)
    cb.ax.set_xticklabels(["0", "10%", "25%", "50%", "75%", "100%"])
    cb.ax.tick_params(labelsize=6.3, length=2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_global_arsenic_podgorski.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  -> fig_global_arsenic_podgorski")


# ============================================================ FIGURE 2
R2C = {"France": "France", "USA": "United States of America", "Bangladesh": "Bangladesh",
       "GEMStat:Canada": "Canada", "GEMStat:France": "France", "GEMStat:Greece": "Greece",
       "GEMStat:India": "India", "GEMStat:Ireland": "Ireland", "GEMStat:Italy": "Italy",
       "GEMStat:Lithuania": "Lithuania", "GEMStat:Mexico": "Mexico",
       "GEMStat:Netherlands (-the )": "Netherlands", "GEMStat:Poland": "Poland",
       "GEMStat:Sweden": "Sweden"}
EURO = {"France", "Italy", "Greece", "Poland", "Netherlands", "Sweden", "Ireland", "Lithuania"}
EEA_BLOC = ["Germany", "Spain", "Portugal", "Austria", "Belgium", "Denmark", "Finland",
            "Norway", "Czechia", "Slovakia", "Hungary", "Romania", "Bulgaria", "Croatia",
            "Slovenia", "Estonia", "Latvia", "Switzerland", "Luxembourg", "United Kingdom",
            "France", "Italy", "Greece", "Poland", "Netherlands", "Sweden", "Ireland", "Lithuania"]
EU_EXT = (-12, 31, 35, 71)
TARGETS = ["As", "Fe", "Mn", "PO4", "NO3", "F", "U"]
TLAB = {"As": "Arsenic", "Fe": "Iron", "Mn": "Manganese", "PO4": "Phosphate",
        "NO3": "Nitrate", "F": "Fluoride", "U": "Uranium"}
REDOX = {"As", "Fe", "Mn", "PO4"}
VMIN, VMAX = 0.40, 0.85
NORM = TwoSlopeNorm(vmin=VMIN, vcenter=0.5, vmax=VMAX)   # white = chance (0.5)


def aggregate(sub):
    by = {}
    for _, r in sub.iterrows():
        if r.held_out_region == "EEA_WISE":
            continue
        c = R2C.get(r.held_out_region)
        if c:
            by.setdefault(c, []).append((r.transfer_auc_mean, r.n_test))
    out = {c: float(np.average([a for a, _ in v], weights=[n for _, n in v])) for c, v in by.items()}
    e = sub[sub.held_out_region == "EEA_WISE"]
    return out, (float(e.transfer_auc_mean.iloc[0]) if len(e) else None)


def shade(ax, agg, eea, inset_rect):
    rows = {c: v for c, v in agg.items() if c not in EURO and c in set(CTRYr.NAME)}
    if rows:
        g = CTRYr[CTRYr.NAME.isin(rows)].copy(); g["v"] = g.NAME.map(rows)
        g.plot(ax=ax, color=[AUC(NORM(x)) for x in g.v], ec="white", lw=0.3, zorder=3)
    if "Bangladesh" in rows:
        bd = CTRYr[CTRYr.NAME == "Bangladesh"]
        bd.plot(ax=ax, facecolor=AUC(NORM(rows["Bangladesh"])), ec="#111", lw=1.0, zorder=4)
        x, y = to_rob(89, 24)
        ax.annotate("Bangladesh", (x, y), xytext=(x + 2.0e6, y - 1.9e6), fontsize=5.4,
                    color="#111", zorder=6, arrowprops=dict(arrowstyle="-", color="#111", lw=0.4))
    # Europe inset
    iax = ax.inset_axes(inset_rect); lo0, lo1, la0, la1 = EU_EXT
    gpd.clip(OCEANg, (lo0, la0, lo1, la1)).plot(ax=iax, color=OCEAN)
    CTRY.cx[lo0:lo1, la0:la1].plot(ax=iax, color=LAND, ec=COAST, lw=0.2)
    if eea is not None:
        CTRY[CTRY.NAME.isin(EEA_BLOC)].plot(ax=iax, color=AUC(NORM(eea)), ec="white",
                                            lw=0.2, alpha=0.9)
    er = {c: v for c, v in agg.items() if c in EURO and c in set(CTRY.NAME)}
    if er:
        g = CTRY[CTRY.NAME.isin(er)].copy(); g["v"] = g.NAME.map(er)
        g.plot(ax=iax, color=[AUC(NORM(x)) for x in g.v], ec="#777", lw=0.3)
    iax.set_xlim(lo0, lo1); iax.set_ylim(la0, la1); iax.set_xticks([]); iax.set_yticks([])
    for s in iax.spines.values():
        s.set_edgecolor("#999"); s.set_linewidth(0.5)


def fig_transfer():
    print("Fig 2: transfer (Nature-tier) ...")
    df = pd.DataFrame(json.load(open(ROOT / "results/region_transfer.json"))["results"])
    fig = plt.figure(figsize=(7.4, 8.7))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1], hspace=0.13,
                  wspace=0.03, left=0.01, right=0.99, top=0.965, bottom=0.02)
    pos = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)]
    for tgt, (r, c) in zip(TARGETS, pos):
        ax = fig.add_subplot(gs[r, c]); globe(ax)
        agg, eea = aggregate(df[df.target == tgt])
        shade(ax, agg, eea, [0.0, 0.0, 0.30, 0.52])
        tag = " (redox)" if tgt in REDOX else ""
        ax.set_title(f"{TLAB[tgt]}{tag}", fontsize=8.5, fontweight="bold", loc="left",
                     color=INK, pad=2)
    axl = fig.add_subplot(gs[3, 1]); axl.set_axis_off()
    cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=AUC),
                      cax=axl.inset_axes([0.08, 0.80, 0.84, 0.05]),
                      orientation="horizontal", ticks=[0.40, 0.5, 0.6, 0.7, 0.8, 0.85])
    cb.set_label("Zero-shot transfer AUC (leave-one-region-out, measured)", fontsize=7.5, labelpad=2)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.ax.axvline(0.5, color="#111", lw=0.9, ls="--")          # chance, = white centre
    axl.text(0.08, 0.60, "dashed = chance (0.5); blue transfers, red fails; "
             "Bangladesh outlined", fontsize=6.4, color="#555", transform=axl.transAxes)
    axl.text(0.08, 0.44, "Each country is a region actually held out and tested. The reducing\n"
             "Bengal aquifer (Bangladesh, India) collapses toward chance on the redox\n"
             "suite (As, Fe, Mn, PO4); oxic regions transfer well. Inset zooms Europe.",
             fontsize=6.6, color=INK, transform=axl.transAxes, va="top")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_global_transfer.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  -> fig_global_transfer")


if __name__ == "__main__":
    fig_arsenic()
    fig_transfer()
    print("Done.")
