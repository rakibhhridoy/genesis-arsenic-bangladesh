#!/usr/bin/env python3
"""Simulated quantum-kernel SVM as a ninth entrant on the LORO cells, with matched controls.

Feasibility memo: RewriteGenesisRewrite/Quantum_feasibility_review.md. Every method below
sees the SAME rows and the SAME features, so any difference is due to the kernel alone.

  qkernel       fidelity kernel |<psi(x)|psi(x')>|^2 of an IQP-type ZZ feature map
                (Havlicek et al. 2019) on q qubits, linear-ring ZZ entanglement, 2 layers,
                exact statevector simulation, bandwidth-scaled angles a = c z
                (Shaydulin & Wild 2022) to counter exponential kernel concentration
  qkernel_prod  the same circuit without the ZZ terms, i.e. a product state with no
                entanglement, whose kernel is classically trivial (prod_i of a cosine factor)
  rbf_svm       classical Gaussian-kernel SVM

Features: t0 chemistry, log(1 + max(x, 0)) except pH and Eh, median imputation and
z-scoring fitted on train, PCA to q components, z-scored and clipped to [-3, 3].
Kernel SVMs cannot use the full training set, so each cell trains on a stratified
subsample of N rows (default 2,000) of the src/59 train partition. The bandwidth (or gamma)
and C are chosen by AUC on up to 2,000 rows of the src/59 validation partition, and the
decision values are mapped to [0, 1] by Platt scaling fitted on those validation rows.
Conformal sets depend only on the rank of the scores, so the mapping does not change them.

Output files match src/59 (one .npz per method, seed and cell), so src/60 scores them
unchanged. Write to a separate --out_root for pilots so src/60 does not pick them up.

  python src/65_quantum_kernel_loro.py --method qkernel --seeds 42 \
      --cells "As/Bangladesh,Mn/USA,NO3/France" --out_root results/conformal_preds_qpilot
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent))
from model.genesis_encoder import GENESIS_PARAMS

_m59 = __import__("59_loro_conformal_preds")
NOLOG = [GENESIS_PARAMS.index(p) for p in ("pH", "Eh")]
C_GRID = [0.1, 1.0, 10.0]
BW_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]           # angle scale c for the two circuit kernels
GAMMA_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]        # RBF gamma, in units of 1/q


def features(t0, tr, q):
    X = t0.astype(np.float64).copy()
    lg = np.setdiff1d(np.arange(X.shape[1]), NOLOG)
    X[:, lg] = np.log1p(np.clip(X[:, lg], 0, None))
    med = np.nanmedian(X[tr], 0)
    med = np.where(np.isnan(med), 0, med)
    X = np.where(np.isnan(X), med, X)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    X = (X - mu) / np.where(sd > 0, sd, 1)
    pca = PCA(q, random_state=0).fit(X[tr])
    Z = pca.transform(X)
    Z = (Z - Z[tr].mean(0)) / Z[tr].std(0)
    return np.clip(Z, -3, 3), float(pca.explained_variance_ratio_.sum())


class Circuit:
    """Exact statevectors of (U_Z(a) H^q)^reps |0>, U_Z = exp(i sum a_i Z_i + i sum a_i a_j Z_i Z_j)."""

    def __init__(self, q, reps=2, entangle=True):
        self.q, self.reps, self.D = q, reps, 2 ** q
        b = (np.arange(self.D)[:, None] >> np.arange(q)[None, :]) & 1
        self.s = (1 - 2 * b).astype(np.float64)                       # Z eigenvalues, D x q
        self.pairs = [(i, (i + 1) % q) for i in range(q)] if entangle else []
        self.ss = (np.stack([self.s[:, i] * self.s[:, j] for i, j in self.pairs], 1)
                   if self.pairs else None)

    def fwht(self, v):
        n = v.shape[0]
        for k in range(self.q):
            v = v.reshape(n, -1, 2, 2 ** k)
            v = np.concatenate([v[:, :, :1] + v[:, :, 1:], v[:, :, :1] - v[:, :, 1:]], 2)
        return v.reshape(n, self.D) / np.sqrt(2) ** self.q

    def states(self, a):
        ph = a @ self.s.T
        if self.pairs:
            ph += np.stack([a[:, i] * a[:, j] for i, j in self.pairs], 1) @ self.ss.T
        u = np.exp(1j * ph).astype(np.complex64)
        v = np.full((len(a), self.D), 1 / np.sqrt(self.D), np.complex64)  # H^q |0>
        for r in range(self.reps):
            v = v * u
            if r < self.reps - 1:
                v = self.fwht(v)
        return v


def kernel_fn(method, q):
    if method == "rbf_svm":
        def k(A, B, g):
            d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
            return np.exp(-(g / q) * np.clip(d, 0, None))
        return k
    circ = Circuit(q, entangle=(method == "qkernel"))

    def k(A, B, c, chunk=2000):
        SB = circ.states(c * B).conj().T
        return np.concatenate([np.abs(circ.states(c * A[i:i + chunk]) @ SB) ** 2
                               for i in range(0, len(A), chunk)], 0).astype(np.float64)
    return k


def subsample(idx, y, n, seed):
    if len(idx) <= n:
        return idx
    r = np.random.default_rng(seed)
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    k = min(max(int(round(n * len(pos) / len(idx))), min(len(pos), 50)), len(pos))
    return np.concatenate([r.choice(pos, k, replace=False), r.choice(neg, n - k, replace=False)])


def run_cell(method, Z, y, tr, va, score, q, seed):
    K = kernel_fn(method, q)
    grid = GAMMA_GRID if method == "rbf_svm" else BW_GRID
    va = subsample(va, y, 2000, seed + 1)
    best, diag = (-1, None, None), {}
    for h in grid:
        Ktr, Kva = K(Z[tr], Z[tr], h), K(Z[va], Z[tr], h)
        off = Ktr[~np.eye(len(tr), dtype=bool)]
        diag[h] = {"k_mean": float(off.mean()), "k_var": float(off.var())}
        for C in C_GRID:
            m = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(Ktr, y[tr])
            auc = roc_auc_score(y[va], m.decision_function(Kva)) if 0 < y[va].sum() < len(va) else 0.5
            if auc > best[0]:
                best = (auc, h, C)
    _, h, C = best
    Ktr = K(Z[tr], Z[tr], h)
    m = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(Ktr, y[tr])
    platt = LogisticRegression().fit(m.decision_function(K(Z[va], Z[tr], h))[:, None], y[va])
    p = platt.predict_proba(m.decision_function(K(Z[score], Z[tr], h))[:, None])[:, 1]
    return p, {"val_auc": best[0], "h": h, "C": C, "concentration": diag[h], "grid": diag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["qkernel", "qkernel_prod", "rbf_svm"])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--qubits", type=int, default=12)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--cells", default="", help="comma list 'target/region' to run a subset")
    ap.add_argument("--out_root", default=str(_m59.OUT))
    a = ap.parse_args()
    want = {c.strip() for c in a.cells.split(",") if c.strip()}

    _, _, _, t0, t1, region = _m59.load()
    for seed in [int(s) for s in a.seeds.split(",")]:
        d = Path(a.out_root) / f"{a.method}_s{seed}"
        d.mkdir(parents=True, exist_ok=True)
        log = d / "_selection.json"
        sel = json.load(open(log)) if log.exists() else {}
        for target, R, y, tr_all, te in _m59.cells(t1, region):
            if want and f"{target}/{R}" not in want:
                continue
            f = d / f"{target}__{R.replace(':', '-').replace('/', '-')}.npz"
            if f.exists():
                continue
            tr, va, cal, tin = _m59.partition(tr_all, target, R, seed)
            score = np.concatenate([cal, tin, te])
            t_start = time.time()
            Z, evr = features(t0, tr, a.qubits)
            trs = subsample(tr, y, a.n_train, seed)
            p, info = run_cell(a.method, Z, y, trs, va, score, a.qubits, seed)
            n1, n2 = len(cal), len(cal) + len(tin)
            np.savez_compressed(f, target=target, region=R, seed=seed,
                                p_cal=p[:n1], y_cal=y[cal], p_in=p[n1:n2], y_in=y[tin],
                                p_out=p[n2:], y_out=y[te], idx_out=te)
            info.update(pca_var=evr, n_train=len(trs), sec=time.time() - t_start,
                        auc_out=float(roc_auc_score(y[te], p[n2:])))
            sel[f"{target}/{R}"] = info
            json.dump(sel, open(log, "w"), indent=1)
            print(f"{a.method} s{seed} {target:4s} {R:20s} h={info['h']:<5} C={info['C']:<4} "
                  f"val {info['val_auc']:.3f} out {info['auc_out']:.3f} "
                  f"kvar {info['concentration']['k_var']:.2e} {info['sec']:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
