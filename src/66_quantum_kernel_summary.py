#!/usr/bin/env python3
"""Score the src/65 quantum-kernel entrant and its controls exactly as src/60 scores the others.

Reads results/conformal_preds_quantum/<method>_s<seed>/ (qkernel, qkernel_prod, rbf_svm) and
the eight existing methods in results/conformal_preds/, computes cell AUC and exceedance
coverage with the src/60 functions, averages seeds within cell, and runs paired Wilcoxon tests
over the cells common to each pair. The decisive comparison is qkernel against rbf_svm (same
rows, same features), and qkernel against qkernel_prod isolates the entangling ZZ terms.
Also reports kernel concentration (variance of off-diagonal train kernel values at the
selected bandwidth) and how often the selected bandwidth sits at the edge of the grid.

  python src/66_quantum_kernel_summary.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
_m60 = __import__("60_conformal_coverage")
Q = ["qkernel", "qkernel_prod", "rbf_svm"]
REF = ["finetune_base", "tabpfn", "rf", "xgb", "logreg", "frozen_base", "rf_aux", "knn_space"]


def method_cells(root, m, rng):
    files = sorted(f for d in (ROOT / root).glob(f"{m}_s*") for f in d.glob("*.npz")
                   if not f.name.startswith("._"))
    return {(c["target"], c["region"]): c
            for c in _m60.seed_mean([_m60.cell_metrics(f, rng) for f in files])}


def main():
    rng = np.random.default_rng(0)
    M = {m: method_cells("results/conformal_preds_quantum", m, rng) for m in Q}
    M.update({m: method_cells("results/conformal_preds", m, rng) for m in REF})
    out = {"methods": {}, "pairs": {}, "selection": {}}
    M = {m: c for m, c in M.items() if c}
    for m, cells in M.items():
        g = defaultdict(list)
        for (t, _), c in cells.items():
            for k in ("all", _m60.GROUP[t]):
                g[k].append(c)
        out["methods"][m] = {k: {"n": len(v), "auc_out": float(np.mean([c["auc_out"] for c in v])),
                                 "cov_pos_out": float(np.mean([c["out"]["cov_pos"] for c in v])),
                                 "both_out": float(np.mean([c["out"]["both"] for c in v]))}
                             for k, v in g.items()}
        out["methods"][m]["cells"] = {f"{t}/{R}": {"auc": c["auc_out"], "cov_pos": c["out"]["cov_pos"],
                                                   "both": c["out"]["both"]} for (t, R), c in cells.items()}
        a = out["methods"][m]["all"]
        print(f"{m:14s} n={a['n']:2d} AUC {a['auc_out']:.3f}  exceedance coverage {a['cov_pos_out']:.3f}")
    for a in [m for m in ["qkernel"] if m in M]:
        for b in [m for m in ["rbf_svm", "qkernel_prod"] + REF if m in M]:
            for g in ["all", "redox", "conservative"]:
                k = [x for x in M[a] if x in M[b] and (g == "all" or _m60.GROUP[x[0]] == g)]
                if len(k) < 6:
                    continue
                r = {"n": len(k)}
                for met, get in [("auc", lambda c: c["auc_out"]), ("cov_pos", lambda c: c["out"]["cov_pos"])]:
                    da, db = np.array([get(M[a][x]) for x in k]), np.array([get(M[b][x]) for x in k])
                    r[met] = {"a": float(da.mean()), "b": float(db.mean()), "a_wins": int((da > db).sum()),
                              "p": float(wilcoxon(da, db).pvalue) if np.any(da != db) else 1.0}
                out["pairs"][f"{a}_vs_{b}_{g}"] = r
                print(f"  {a} vs {b:13s} [{g:12s}] n={len(k):2d} AUC {r['auc']['a']:.3f} vs {r['auc']['b']:.3f} "
                      f"(wins {r['auc']['a_wins']}, p={r['auc']['p']:.4f})  cov {r['cov_pos']['a']:.3f} vs "
                      f"{r['cov_pos']['b']:.3f} (p={r['cov_pos']['p']:.4f})")
    grid = {"qkernel": 0.01, "qkernel_prod": 0.01, "rbf_svm": 0.003}
    for m in Q:
        sel = [v for f in (ROOT / "results/conformal_preds_quantum").glob(f"{m}_s*/_selection.json")
               for v in json.load(open(f)).values()]
        if not sel:
            continue
        h = np.array([s["h"] for s in sel])
        out["selection"][m] = {"n": len(sel), "h_counts": {str(u): int((h == u).sum()) for u in np.unique(h)},
                               "frac_at_lower_edge": float((h == grid[m]).mean()),
                               "k_var_median": float(np.median([s["concentration"]["k_var"] for s in sel])),
                               "pca_var_median": float(np.median([s["pca_var"] for s in sel])),
                               "sec_median": float(np.median([s["sec"] for s in sel]))}
        print(m, out["selection"][m])
    json.dump(out, open(ROOT / "results/quantum_kernel_summary.json", "w"), indent=1)
    print("wrote results/quantum_kernel_summary.json")


if __name__ == "__main__":
    main()
