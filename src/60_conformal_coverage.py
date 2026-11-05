#!/usr/bin/env python3
"""Does a calibrated exceedance guarantee survive the move into an unseen region? (E3)

Class-conditional (Mondrian) split conformal on the per-sample probabilities of src/59.
For each class c the threshold q_c is the ceil((n_c+1)(1-alpha))/n_c quantile of the
calibration scores 1 - p_c, and class c enters the prediction set when 1 - p_c <= q_c.
Calibration uses out-of-region rows only. The guarantee is then checked twice
  test_in   out-of-region rows held aside (exchangeable with calibration, should hold)
  test_out  the held-out region (the deployment case)
The quantity that matters for health is coverage of the exceedance class, since a
positive sample whose set excludes "exceeds" is a missed exceedance the method claimed
to cover. Mondrian conformal stays valid under pure label shift, so prevalence
differences between regions cannot by themselves cause undercoverage. A loss of
positive-class coverage in test_out therefore means p(x | exceedance) changed.

Remedy check: recalibrate on k labelled samples from the held-out region (k = 25, 50,
100, 200; 50 random draws) and score on the remaining in-region rows.

Cell metrics are averaged over seeds before any test. Groups follow src/30: redox
(As, Fe, Mn, PO4), conservative (NO3, F), uranium alone.

  python src/60_conformal_coverage.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
ALPHA = 0.10
GROUP = {"As": "redox", "Fe": "redox", "Mn": "redox", "PO4": "redox",
         "NO3": "conservative", "F": "conservative", "U": "uranium"}
KS = [25, 50, 100, 200]


def thresholds(p, y):
    q = {}
    for c in (0, 1):
        s = np.sort(1 - (p[y == c] if c else 1 - p[y == c]))
        n = len(s)
        k = int(np.ceil((n + 1) * (1 - ALPHA)))
        q[c] = s[k - 1] if 0 < k <= n else np.inf
    return q


def sets(p, q):
    return (p >= 1 - q[1]), ((1 - p) >= 1 - q[0])      # contains "exceeds", contains "below"


def score(p, y, q):
    has1, has0 = sets(p, q)
    cov = np.where(y == 1, has1, has0)
    return {"cov": float(cov.mean()), "cov_pos": float(has1[y == 1].mean()),
            "cov_neg": float(has0[y == 0].mean()),
            "size": float((has1.astype(int) + has0).mean()),
            "both": float((has1 & has0).mean()), "empty": float((~has1 & ~has0).mean())}


def local(p, y, rng, k, draws=50):
    out = []
    for _ in range(draws):
        if len(y) <= k + 20:
            break
        i = rng.permutation(len(y))
        c, t = i[:k], i[k:]
        if y[t].sum() == 0:
            continue
        out.append(score(p[t], y[t], thresholds(p[c], y[c])))
    return {m: float(np.mean([o[m] for o in out])) for m in out[0]} if out else None


def cell_metrics(f, rng):
    z = np.load(f, allow_pickle=True)
    q = thresholds(z["p_cal"], z["y_cal"])
    r = {"target": str(z["target"]), "region": str(z["region"]), "seed": int(z["seed"]),
         "n_out": int(len(z["y_out"])), "n_pos_out": int(z["y_out"].sum()),
         "n_pos_cal": int(z["y_cal"].sum()),
         "prev_cal": float(z["y_cal"].mean()), "prev_out": float(z["y_out"].mean()),
         "in": score(z["p_in"], z["y_in"], q), "out": score(z["p_out"], z["y_out"], q),
         "auc_out": float(roc_auc_score(z["y_out"], z["p_out"]))}
    r["local"] = {k: local(z["p_out"], z["y_out"], rng, k) for k in KS}
    return r


def seed_mean(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["target"], r["region"])].append(r)
    out = []
    for (t, R), rs in by.items():
        m = {"target": t, "region": R, "group": GROUP[t], "n_seeds": len(rs),
             "n_out": rs[0]["n_out"], "n_pos_out": rs[0]["n_pos_out"],
             "prev_cal": float(np.mean([x["prev_cal"] for x in rs])), "prev_out": rs[0]["prev_out"],
             "auc_out": float(np.mean([x["auc_out"] for x in rs]))}
        for s in ("in", "out"):
            m[s] = {k: float(np.mean([x[s][k] for x in rs])) for k in rs[0][s]}
        m["local"] = {}
        for k in KS:
            v = [x["local"][k] for x in rs if x["local"][k]]
            m["local"][k] = {kk: float(np.mean([x[kk] for x in v])) for kk in v[0]} if v else None
        out.append(m)
    return out


def summarise(cells):
    s = {}
    for g in ["all", "redox", "conservative", "uranium"]:
        c = [x for x in cells if g == "all" or x["group"] == g]
        if not c:
            continue
        pin = np.array([x["in"]["cov_pos"] for x in c]); pout = np.array([x["out"]["cov_pos"] for x in c])
        nin = np.array([x["in"]["cov_neg"] for x in c]); nout = np.array([x["out"]["cov_neg"] for x in c])
        d = {"n_cells": len(c), "auc_out": float(np.mean([x["auc_out"] for x in c])),
             "cov_pos_in": float(pin.mean()), "cov_pos_out": float(pout.mean()),
             "cov_neg_in": float(nin.mean()), "cov_neg_out": float(nout.mean()),
             "size_out": float(np.mean([x["out"]["size"] for x in c])),
             "frac_cells_pos_below_nominal": float((pout < 1 - ALPHA - 0.05).mean()),
             "worst_cells": sorted([(x["target"], x["region"], round(x["out"]["cov_pos"], 3))
                                    for x in c], key=lambda t: t[2])[:5]}
        if len(c) >= 6:
            d["p_pos_out_vs_in"] = float(wilcoxon(pout, pin).pvalue)
        for k in KS:
            v = [x["local"][k] for x in c if x["local"][k]]
            if v:
                d[f"local{k}_cov_pos"] = float(np.mean([x["cov_pos"] for x in v]))
                d[f"local{k}_size"] = float(np.mean([x["size"] for x in v]))
        s[g] = d
    r = [x["out"]["cov_pos"] - x["in"]["cov_pos"] for x in cells if x["group"] == "redox"]
    c = [x["out"]["cov_pos"] - x["in"]["cov_pos"] for x in cells if x["group"] == "conservative"]
    if r and c:
        s["redox_vs_conservative_gap_p"] = float(mannwhitneyu(r, c, alternative="less").pvalue)
    return s


def main():
    rng = np.random.default_rng(0)
    res = {"alpha": ALPHA, "methods": {}}
    dirs = sorted(d for d in (ROOT / "results/conformal_preds").iterdir() if d.is_dir())
    methods = defaultdict(list)
    for d in dirs:
        methods[d.name.rsplit("_s", 1)[0]].extend(sorted(f for f in d.glob("*.npz") if not f.name.startswith("._")))
    for m, files in methods.items():
        rows = [cell_metrics(f, rng) for f in files]
        cells = seed_mean(rows)
        res["methods"][m] = {"cells": cells, "summary": summarise(cells)}
        print(f"\n=== {m}: {len(cells)} cells, {len(rows)} cell-seeds ===")
        for g, d in res["methods"][m]["summary"].items():
            if isinstance(d, dict):
                print(f"  {g:13s} n={d['n_cells']:2d}  AUC {d['auc_out']:.3f}  pos cov in {d['cov_pos_in']:.3f} -> out "
                      f"{d['cov_pos_out']:.3f} (p={d.get('p_pos_out_vs_in', float('nan')):.4f})  "
                      f"neg out {d['cov_neg_out']:.3f}  size {d['size_out']:.2f}  "
                      f"local50 {d.get('local50_cov_pos', float('nan')):.3f}  "
                      f"local200 {d.get('local200_cov_pos', float('nan')):.3f}")
            else:
                print(f"  {g}: {d:.4f}")
    # head-to-head on positive-class coverage under shift, seed-averaged, paired by cell
    names = list(res["methods"])
    h2h = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A = {(c["target"], c["region"]): c for c in res["methods"][a]["cells"]}
            B = {(c["target"], c["region"]): c for c in res["methods"][b]["cells"]}
            for g in ["all", "redox", "conservative"]:
                k = [x for x in A if x in B and (g == "all" or GROUP[x[0]] == g)]
                if len(k) < 6:
                    continue
                da = np.array([A[x]["out"]["cov_pos"] for x in k]); db = np.array([B[x]["out"]["cov_pos"] for x in k])
                h2h[f"{a}_vs_{b}_{g}"] = {"n": len(k), "a": float(da.mean()), "b": float(db.mean()),
                                         "p": float(wilcoxon(da, db).pvalue) if np.any(da != db) else 1.0}
                print(f"  {a} vs {b} [{g}] pos cov out {da.mean():.3f} vs {db.mean():.3f} "
                      f"p={h2h[f'{a}_vs_{b}_{g}']['p']:.4f}")
    res["head_to_head"] = h2h
    json.dump(res, open(ROOT / "results/conformal_coverage.json", "w"), indent=1)
    print("\nwrote results/conformal_coverage.json")


if __name__ == "__main__":
    main()
