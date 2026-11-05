#!/usr/bin/env python3
"""Verdict on src/52 + src/53: does raster-only hazard skill out of region follow redox control?

Reads the seed-level results and reports, in the order the confounds must be cleared:
  1. across-parameter Spearman of out-of-region skill vs reconstruction R2 (seed means)
  2. WITHIN each held-out region, parameters ranked against R2, so region difficulty and
     which regions measure what cancel (the region-composition confound)
  3. within one data source (src/53), so the source/protocol boundary is removed
  4. a permutation test on (1), shuffling R2 across parameters
Eh rests on few sites and few regions; every test is repeated without it.

  python3 src/54_transfer_gradient_analysis.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
REDOX = ["Eh", "Fe", "Mn", "As", "PO4", "U"]
CONS = ["Ca", "Mg", "Na", "Cl", "HCO3", "EC", "TDS", "SO4", "K"]


def perm_p(x, y, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    obs = spearmanr(x, y)[0]
    null = np.array([spearmanr(rng.permutation(x), y)[0] for _ in range(n)])
    return obs, float((np.abs(null) >= abs(obs)).mean())


def main():
    r = pd.DataFrame(json.load(open(ROOT/"results/hazard_transfer_gradient.json")))
    p = r[r["mode"] == "p80"].copy()
    m = p.groupby("param").mean(numeric_only=True)
    for k in ["block1", "block5", "block10", "loro_mean"]:
        m[f"gap_{k}"] = m["random"] - m[k]
    m["seeds"] = p.groupby("param").size()
    saved = {"across_params": {}, "within_region": {}, "within_source": {}}
    print("== 1. seed means, 80th-percentile labels (sorted by R2)")
    print(m.sort_values("r2")[["r2", "seeds", "random", "block10", "loro_mean",
                               "gap_block10", "gap_loro_mean"]].round(3).to_string())

    print("\n== 1b. across-parameter Spearman vs R2 (permutation p), all / without Eh")
    for k in ["random", "block1", "block5", "block10", "loro_mean", "gap_block10", "gap_loro_mean"]:
        for lab, q in [("all", m), ("noEh", m.drop(index="Eh", errors="ignore"))]:
            q = q.dropna(subset=[k, "r2"])
            rho, pv = perm_p(q.r2.values, q[k].values)
            print(f"  {k:14s} {lab:5s} rho={rho:+.2f} perm-p={pv:.4f} n={len(q)}")
            saved["across_params"][f"{k}_{lab}"] = {"rho": float(rho), "perm_p": pv, "n": len(q)}

    print("\n== 1c. redox vs conservative group means")
    for k in ["random", "block10", "loro_mean", "gap_loro_mean"]:
        a, b = m.loc[m.index.intersection(REDOX), k], m.loc[m.index.intersection(CONS), k]
        print(f"  {k:14s} redox {a.mean():.3f} (n={len(a)})  conservative {b.mean():.3f} (n={len(b)})")

    print("\n== 2. within each held-out region: parameters vs R2 (seed-mean AUC)")
    rows = []
    for _, o in p.iterrows():
        for R, auc in o["loro"].items():
            rows.append((R, o["param"], o["seed"], auc, o["r2"]))
    w = pd.DataFrame(rows, columns=["region", "param", "seed", "auc", "r2"])
    w = w.groupby(["region", "param"]).mean(numeric_only=True).reset_index()
    rhos = []
    for R, g in w.groupby("region"):
        if len(g) < 6:
            print(f"  {R:22s} {len(g)} params, skipped")
            continue
        rho, pv = spearmanr(g.r2, g.auc)
        g2 = g[g.param != "Eh"]
        rho2 = spearmanr(g2.r2, g2.auc)[0]
        rhos.append((R, rho, rho2, len(g)))
        saved["within_region"][R] = {"rho": float(rho), "p": float(pv), "rho_noEh": float(rho2),
                                     "n_params": len(g)}
        print(f"  {R:22s} n={len(g):2d} rho={rho:+.2f} (p={pv:.3f})  without Eh {rho2:+.2f}")
    if rhos:
        z = np.array([x[1] for x in rhos])
        print(f"  mean within-region rho {z.mean():+.2f}; positive in {(z > 0).sum()}/{len(z)} regions")

    f = ROOT/"results/hazard_transfer_within_source.json"
    if f.exists():
        print("\n== 3. within one data source (block hold-out, one protocol)")
        s = pd.DataFrame(json.load(open(f)))
        t = s.groupby(["source", "param"]).mean(numeric_only=True).reset_index()
        for src, g in t.groupby("source"):
            for k in ["random", "block", "gap"]:
                for lab, q in [("all", g), ("noEh", g[g.param != "Eh"])]:
                    q = q.dropna(subset=[k, "r2"])
                    rho, pv = perm_p(q.r2.values, q[k].values)
                    print(f"  {src:6s} {k:6s} {lab:5s} rho={rho:+.2f} perm-p={pv:.4f} n={len(q)}")
                    saved["within_source"][f"{src}_{k}_{lab}"] = {"rho": float(rho), "perm_p": pv,
                                                                  "n": len(q)}

    g = r[r["mode"] == "guideline"].groupby("param").mean(numeric_only=True)
    print("\n== 4. WHO/operational guideline labels (seed means)")
    print(g[["thr", "prev", "random", "block10", "loro_mean"]].round(3).to_string())
    json.dump(saved, open(ROOT/"results/transfer_gradient_controls.json", "w"), indent=1)
    print("  -> results/transfer_gradient_controls.json")


if __name__ == "__main__":
    main()
