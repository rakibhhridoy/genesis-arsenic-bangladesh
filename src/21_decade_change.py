"""
GENESIS Paper 5 (second-paper seed) — Step 21: decade-scale groundwater-quality
change atlas.

Mines the 58,722 temporal pairs (median 14-yr gap) for where groundwater
contamination is rising vs falling, by country and spatially, with significance.

Key findings explored here:
  - Regional/country As divergence (France declining, GEMStat-Mexico rising).
  - The "improving-mean / worsening-tail" paradox: bulk As falls while the
    fraction above the WHO 10 ug/L threshold can still grow.
  - NO3 rising in agricultural regions (France, USA).

Produces:
  results/decade_change.json          (country-level stats)
  figures/fig_decade_as_map.png       (global As-trend map)
  figures/fig_decade_tail_paradox.png (mean vs tail change)

Usage:
  python src/21_decade_change.py
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB = "data/genesis.duckdb"
WHO = {"As": 10.0, "NO3": 50.0}


def main():
    con = duckdb.connect(DB, read_only=True)
    df = con.execute(
        "SELECT country, lat, lon, gap_years, As_t0, As_t1, NO3_t0, NO3_t1 "
        "FROM genesis_temporal_pairs"
    ).df()
    con.close()

    # ---- country-level trends ----
    out = {"targets": {}}
    for p, thr in WHO.items():
        rows = []
        for c, g in df.groupby("country"):
            d = (g[f"{p}_t1"] - g[f"{p}_t0"]).dropna()
            if len(d) < 100:
                continue
            try:
                _, pv = wilcoxon(d)
            except ValueError:
                pv = np.nan
            t0_unsafe = int((g[f"{p}_t0"] > thr).sum())
            t1_unsafe = int((g[f"{p}_t1"] > thr).sum())
            rows.append({
                "country": str(c), "n": int(len(d)),
                "median_delta": float(d.median()), "mean_delta": float(d.mean()),
                "pct_worsening": float(100 * (d > 0).mean()),
                "wilcoxon_p": float(pv),
                "tail_t0": t0_unsafe, "tail_t1": t1_unsafe,
                "tail_change": t1_unsafe - t0_unsafe,
            })
        rows.sort(key=lambda r: r["median_delta"])
        out["targets"][p] = rows
        print(f"\n{p} decade change by country (sorted by median delta):")
        print(f"  {'country':10s}{'n':>7s}{'medΔ':>9s}{'%worse':>8s}{'p':>10s}  tail t0→t1")
        for r in rows:
            print(f"  {r['country']:10s}{r['n']:7d}{r['median_delta']:+9.3f}"
                  f"{r['pct_worsening']:7.0f}%{r['wilcoxon_p']:10.1e}  "
                  f"{r['tail_t0']}→{r['tail_t1']} ({r['tail_change']:+d})")

    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/decade_change.json", "w"), indent=2)

    # ---- Fig 1: global As-trend map ----
    a = df.dropna(subset=["As_t0", "As_t1", "lat", "lon"]).copy()
    a["dAs"] = a["As_t1"] - a["As_t0"]
    # bin spatially to reduce overplotting
    a["latb"] = (a["lat"] / 2).round() * 2
    a["lonb"] = (a["lon"] / 2).round() * 2
    cell = a.groupby(["latb", "lonb"])["dAs"].median().reset_index()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    vmax = np.percentile(np.abs(cell["dAs"]), 95)
    sc = ax.scatter(cell["lonb"], cell["latb"], c=cell["dAs"], cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, s=18, edgecolor="none")
    plt.colorbar(sc, label="median ΔAs (µg/L), t1 − t0  (red = rising)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Decade-scale arsenic change in global groundwater\n"
                 "(2°×2° cells, median of ~14-yr temporal pairs)", fontweight="bold")
    ax.set_xlim(-130, 100); ax.set_ylim(0, 70)
    plt.tight_layout(); plt.savefig("figures/fig_decade_as_map.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("\nsaved figures/fig_decade_as_map.png")

    # ---- Fig 2: mean-vs-tail paradox by country ----
    rows = out["targets"]["As"]
    big = [r for r in rows if r["n"] >= 200]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for r in big:
        tail_pct = 100 * r["tail_change"] / max(r["n"], 1)
        ax.scatter(r["median_delta"], tail_pct, s=60, zorder=3)
        ax.annotate(r["country"], (r["median_delta"], tail_pct),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.axhline(0, color="gray", lw=0.7); ax.axvline(0, color="gray", lw=0.7)
    ax.set_xlabel("median ΔAs (µg/L)  ←improving | worsening→")
    ax.set_ylabel("change in % of wells above WHO 10 µg/L")
    ax.set_title("Improving-mean / worsening-tail paradox\n"
                 "(lower-right quadrant: bulk improves but dangerous tail grows)",
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig("figures/fig_decade_tail_paradox.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved figures/fig_decade_tail_paradox.png")


if __name__ == "__main__":
    main()
