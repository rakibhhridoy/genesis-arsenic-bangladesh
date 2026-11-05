#!/usr/bin/env python3
"""Axes A (inferability) and D (physical consistency) for the simulated quantum kernel.

Kernel ridge regression with the three kernels of src/65 on identical rows and features
  qkernel       IQP-type ZZ fidelity kernel, 12 qubits, ring, 2 layers, bandwidth-scaled
  qkernel_prod  the same circuit without the ZZ terms (no entanglement)
  rbf_svm       classical Gaussian kernel (key kept from src/65 so labels and colours match)

Test rows are those of src/56/src/61. The ladder replays src/61's random-number sequence call
for call (one generator, seed 0), and the thermodynamic check rebuilds src/58's site-disjoint
and Bangladesh rows the same way, so every entrant is scored on the same samples. Training
pools are also the src/61 pools; from each, a kernel method can only use a random subsample
(2,000 rows for fitting, 500 further rows for choosing bandwidth and ridge penalty by R2).

Features per fit: the src/56 inputs of that level (z-scored chemistry and rasters), median
imputation and z-scoring fitted on the 2,000 rows, PCA to 12 components, z-scored and clipped
to [-3, 3]. The target is the z-scored parameter, centred on the training mean.

Writes NEW files only (src/61's outputs are untouched):
  results/inferability_ladder_quantum.json
  results/thermodynamic_consistency_quantum.json

  python src/68_quantum_kernel_regression.py --mode both
  python src/68_quantum_kernel_regression.py --mode ladder --targets Ca,As --levels L2,L4   # pilot
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = load_mod("b61", "src/61_axes_AD_baselines.py")
Q = load_mod("q65", "src/65_quantum_kernel_loro.py")
L, T = B.L, B.T
METHODS = ["qkernel", "qkernel_prod", "rbf_svm"]
QUBITS, N_FIT, N_SEL = 12, 2000, 500
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]


class KRR:
    """Kernel ridge on 12 principal components, hyperparameters chosen on held-aside rows."""

    def __init__(self, method, Xtr, ytr, Xsel, ysel):
        self.k = Q.kernel_fn(method, QUBITS)
        X = Xtr.copy()
        self.med = np.nanmedian(X, 0)
        self.med = np.where(np.isnan(self.med), 0, self.med)
        X = np.where(np.isnan(X), self.med, X)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd = np.where(self.sd > 0, self.sd, 1)
        self.pca = PCA(QUBITS, random_state=0).fit((X - self.mu) / self.sd)
        P = self.pca.transform((X - self.mu) / self.sd)
        self.pm, self.ps = P.mean(0), P.std(0)
        self.ps = np.where(self.ps > 0, self.ps, 1)
        self.Ztr, self.ym = self.feat(Xtr), ytr.mean()
        Zs = self.feat(Xsel)
        grid = Q.GAMMA_GRID if method == "rbf_svm" else Q.BW_GRID
        best = (-np.inf, None, None, None)
        for h in grid:
            K, Ks = self.k(self.Ztr, self.Ztr, h), self.k(Zs, self.Ztr, h)
            for a in ALPHAS:
                c = solve(K + a * np.eye(len(K)), ytr - self.ym, assume_a="pos")
                r = L.r2(ysel, Ks @ c + self.ym)
                if r > best[0]:
                    best = (r, h, a, c)
        self.sel_r2, self.h, self.alpha, self.c = best

    def feat(self, X):
        X = np.where(np.isnan(X), self.med, X)
        P = (self.pca.transform((X - self.mu) / self.sd) - self.pm) / self.ps
        return np.clip(P, -3, 3)

    def predict(self, X, bs=2000):
        return np.concatenate([self.k(self.feat(X[i:i + bs]), self.Ztr, self.h) @ self.c + self.ym
                               for i in range(0, len(X), bs)])


def split(pool, seed):
    r = np.random.default_rng(seed)
    p = r.permutation(pool)[:N_FIT + N_SEL]
    return p[:N_FIT], p[N_FIT:]


def ladder(ns, chem, aux, bchem, baux, gw_tr, test, targets, levels, n_train=150000, max_test=3000):
    Z, AZ, BZ, BAZ = L.znorm(chem, ns), L.aux_z(aux, ns), L.znorm(bchem, ns), L.aux_z(baux, ns)
    col = {p: i for i, p in enumerate(L.GENESIS_PARAMS)}
    req = [col[p] for p in L.REQUIRED]
    stored = {(r["target"], r["level"], r["test"]): r for r in
              json.load(open(ROOT / "results/inferability_ladder_baselines.json"))["rows"]}
    out_f = ROOT / "results/inferability_ladder_quantum.json"
    rows = json.load(open(out_f))["rows"] if out_f.exists() else []
    done = {(r["target"], r["level"], r["test"], r["method"]) for r in rows}
    rng = np.random.default_rng(0)                       # replays src/56 / src/61 call for call
    n_mismatch = 0
    for t in L.GENESIS_PARAMS:
        ti = col[t]
        tr = gw_tr[np.isfinite(chem[gw_tr, ti])]
        if len(tr) > n_train:
            tr = rng.choice(tr, n_train, replace=False)
        sets = {}
        for name, Zs, AZs, cs, idx in [("site_disjoint", Z, AZ, chem, test),
                                       ("bangladesh", BZ, BAZ, bchem, np.arange(len(bchem)))]:
            ok = idx[np.isfinite(cs[idx, ti]) & np.isfinite(cs[idx][:, req]).all(1)]
            if len(ok) > max_test:
                ok = rng.choice(ok, max_test, replace=False)
            sets[name] = (Zs, AZs, ok)
        if targets and t not in targets:
            continue
        fit, sel = split(tr, 1000 + ti)
        for lv, obs in L.LEVELS.items():
            if levels and lv not in levels:
                continue
            feats = [col[p] for p in obs if p != t]
            X = lambda Zs, AZs, r: np.c_[Zs[r][:, feats], AZs[r]] if feats else AZs[r]
            for m in METHODS:
                if all((t, lv, n, m) in done for n in sets if len(sets[n][2]) >= 50):
                    continue
                t0 = time.time()
                mdl = KRR(m, X(Z, AZ, fit), Z[fit, ti], X(Z, AZ, sel), Z[sel, ti])
                for name, (Zs, AZs, ok) in sets.items():
                    if len(ok) < 50 or (t, lv, name, m) in done:
                        continue
                    r2 = L.r2(Zs[ok, ti], mdl.predict(X(Zs, AZs, ok)))
                    s = stored.get((t, lv, name))
                    if s is None or s["n"] != len(ok):
                        n_mismatch += 1
                    rows.append(dict(target=t, level=lv, test=name, method=m, n=int(len(ok)), r2=r2,
                                     h=mdl.h, alpha=mdl.alpha, sel_r2=mdl.sel_r2,
                                     n_matches_src61=bool(s and s["n"] == len(ok))))
                    print(f"  {t:5s} {lv} {name:13s} {m:12s} R2={r2:+.3f}  h={mdl.h} a={mdl.alpha} "
                          f"{time.time() - t0:5.1f}s", flush=True)
                json.dump({"rows": rows}, open(out_f, "w"), indent=1)
    print(f"row-count mismatches against src/61: {n_mismatch}", flush=True)


def thermo(ns, chem, aux, bchem, baux, gw_tr, test, n_train=150000):
    Z, AZ, BZ, BAZ = L.znorm(chem, ns), L.aux_z(aux, ns), L.znorm(bchem, ns), L.aux_z(baux, ns)
    col = {p: i for i, p in enumerate(L.GENESIS_PARAMS)}
    need = [col[p] for p in T.RECON if p != "F"]
    clean = test[np.isfinite(chem[test][:, need]).all(1)]
    rng = np.random.default_rng(0)                       # replays src/58 / src/61
    bidx = np.where(np.isfinite(bchem[:, need]).all(1))[0]
    if len(bidx) > 2000:
        bidx = rng.choice(bidx, 2000, replace=False)
    out = {}
    for m in METHODS:
        models = {}
        for p in T.RECON:
            ti = col[p]
            pool = gw_tr[np.isfinite(chem[gw_tr, ti])]
            fit, sel = split(pool, 2000 + ti)
            feats = [col[q] for q in L.GENESIS_PARAMS if q != p]
            models[p] = (KRR(m, np.c_[Z[fit][:, feats], AZ[fit]], Z[fit, ti],
                             np.c_[Z[sel][:, feats], AZ[sel]], Z[sel, ti]), feats)

        def swap(model, ds, rows, target, allowed):
            mdl, feats = models[target]
            Zs, AZs = ds
            return mdl.predict(np.c_[Zs[rows][:, feats], AZs[rows]])
        T.L.fm_predict = swap
        r = np.random.default_rng(0)
        out[m] = [T.evaluate("site_disjoint", chem, None, (Z, AZ), clean, None, ns, col, r),
                  T.evaluate("bangladesh", bchem, None, (BZ, BAZ), bidx, None, ns, col, r)]
        for res in out[m]:
            a = res["all"]
            print(f"  {m:12s} {res['set']:13s} n={res['n']:5d} calcite {a['SI_calcite']['r2']:+.3f} "
                  f"gypsum {a['SI_gypsum']['r2']:+.3f} fluorite {a.get('SI_fluorite', {}).get('r2', float('nan')):+.3f} "
                  f"|CBE| obs {a['abs_CBE_obs']:.1f} rec {a['abs_CBE_rec']:.1f}", flush=True)
        json.dump(out, open(ROOT / "results/thermodynamic_consistency_quantum.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["ladder", "thermo", "both"])
    ap.add_argument("--targets", default="")
    ap.add_argument("--levels", default="")
    a = ap.parse_args()
    data = B.load()
    if a.mode in ("ladder", "both"):
        ladder(*data, {x for x in a.targets.split(",") if x}, {x for x in a.levels.split(",") if x})
    if a.mode in ("thermo", "both"):
        thermo(*data)


if __name__ == "__main__":
    main()
