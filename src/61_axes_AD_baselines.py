#!/usr/bin/env python3
"""Evaluation axes A (inferability) and D (physical consistency) for the non-GENESIS entrants.

src/56 and src/58 scored the masked model against gradient boosting only. This adds random
forest and TabPFN regressors on the SAME rows.

Axis A  The ladder rows are rebuilt by replaying src/56's random-number sequence (one
        generator, seed 0, the same calls in the same order). Gradient boosting is refit with
        the src/56 recipe and its R2 must equal the stored gb_r2, which proves the rows match.
Axis D  The predictor inside src/58's evaluate() is swapped for each entrant, so the
        reconstruction, the thermodynamics and the null are computed by the same code that
        scored GENESIS. Every parameter in src/58's RECON set is reconstructed at L4 (all other
        parameters plus the rasters observed).

  rf      RandomForestRegressor, native NaN, 200 trees, max_samples 0.3
  tabpfn  TabPFNRegressor v2.2.1, default ensemble, 10,000-row random context (GPU)
  gb      HistGradientBoosting, src/56 recipe (row-identity check and axis D entrant)

  python src/61_axes_AD_baselines.py --mode both --methods gb,rf,tabpfn
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


L = load_mod("l56", "src/56_inferability_ladder.py")
T = load_mod("t58", "src/58_thermodynamic_consistency.py")


def make(method, Xtr, ytr, seed=0):
    if method == "gb":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=0).fit(Xtr, ytr)
    if method == "rf":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_samples=0.3,
                                     n_jobs=-1, random_state=0).fit(Xtr, ytr)
    from tabpfn import TabPFNRegressor
    if len(Xtr) > 10000:
        i = np.random.default_rng(seed).choice(len(Xtr), 10000, replace=False)
        Xtr, ytr = Xtr[i], ytr[i]
    return TabPFNRegressor(device="cuda" if torch.cuda.is_available() else "cpu",
                           random_state=seed).fit(Xtr, ytr)


def predict(m, X, bs=2000):
    return np.concatenate([m.predict(X[i:i + bs]) for i in range(0, len(X), bs)])


def load():
    ns = L.load_stats(str(ROOT/"data/processed/normalization_stats.json"))
    chem = torch.load(ROOT/"data/processed/genesis_pretrain.pt", map_location="cpu", weights_only=False).numpy()
    aux = torch.load(ROOT/"data/processed/genesis_pretrain_aux.pt", map_location="cpu", weights_only=False)
    aux = (aux["aux"] if isinstance(aux, dict) else aux)
    aux = aux.numpy() if hasattr(aux, "numpy") else aux
    meta = pd.read_parquet(ROOT/"data/processed/genesis_pretrain_meta.parquet",
                           columns=["site_id", "lat", "lon", "water_body_type"])
    bchem = torch.load(ROOT/"data/processed/genesis_held_out_bd.pt", map_location="cpu", weights_only=False).numpy()
    baux = torch.load(ROOT/"data/processed/genesis_held_out_bd_aux.pt", map_location="cpu", weights_only=False)
    baux = baux.numpy() if hasattr(baux, "numpy") else baux
    N = len(chem); n_val = min(50000, max(1, int(N * 0.05)))
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(42)).numpy()
    tr_idx, va_idx = perm[:N - n_val], perm[N - n_val:]
    train_sites = set(meta.site_id.values[tr_idx])
    gw = meta.water_body_type.values == "groundwater"
    test = np.array([i for i in va_idx if gw[i] and meta.site_id.values[i] not in train_sites])
    return ns, chem, aux, bchem, baux, tr_idx[gw[tr_idx]], test


def ladder(methods, ns, chem, aux, bchem, baux, gw_tr, test, n_train=150000, max_test=3000):
    Z, AZ, BZ, BAZ = L.znorm(chem, ns), L.aux_z(aux, ns), L.znorm(bchem, ns), L.aux_z(baux, ns)
    col = {p: i for i, p in enumerate(L.GENESIS_PARAMS)}
    req = [col[p] for p in L.REQUIRED]
    stored = {(r["target"], r["level"], r["test"]): r for r in json.load(open(ROOT/"results/inferability_ladder.json"))}
    rng = np.random.default_rng(0)                       # replays src/56 call for call
    rows, maxdiff = [], 0.0
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
        for lv, obs in L.LEVELS.items():
            feats = [col[p] for p in obs if p != t]
            Xtr = np.c_[Z[tr][:, feats], AZ[tr]] if feats else AZ[tr]
            for m in methods:
                mdl = make(m, Xtr, Z[tr, ti])
                for name, (Zs, AZs, ok) in sets.items():
                    if len(ok) < 50:
                        continue
                    Xte = np.c_[Zs[ok][:, feats], AZs[ok]] if feats else AZs[ok]
                    r2 = L.r2(Zs[ok, ti], predict(mdl, Xte))
                    s = stored.get((t, lv, name))
                    if m == "gb" and s:
                        maxdiff = max(maxdiff, abs(r2 - s["gb_r2"]))
                    rows.append(dict(target=t, level=lv, test=name, method=m, n=int(len(ok)), r2=r2,
                                     fm_r2=s["fm_r2"] if s else None, gb_r2_stored=s["gb_r2"] if s else None))
                    print(f"  {t:5s} {lv} {name:13s} {m:6s} R2={r2:+.3f}  (FM {rows[-1]['fm_r2']})", flush=True)
            json.dump({"row_identity_max_abs_diff_gb": maxdiff, "rows": rows},
                      open(ROOT/"results/inferability_ladder_baselines.json", "w"), indent=1)
    print(f"row-identity check: max |GB R2 - stored| = {maxdiff:.2e}", flush=True)


def thermo(methods, ns, chem, aux, bchem, baux, gw_tr, test, n_train=150000):
    Z, AZ, BZ, BAZ = L.znorm(chem, ns), L.aux_z(aux, ns), L.znorm(bchem, ns), L.aux_z(baux, ns)
    col = {p: i for i, p in enumerate(L.GENESIS_PARAMS)}
    need = [col[p] for p in T.RECON if p != "F"]
    clean = test[np.isfinite(chem[test][:, need]).all(1)]
    rng = np.random.default_rng(0)                       # replays src/58
    bidx = np.where(np.isfinite(bchem[:, need]).all(1))[0]
    if len(bidx) > 2000:
        bidx = rng.choice(bidx, 2000, replace=False)
    sel = np.random.default_rng(1)
    out = {}
    for m in methods:
        models = {}
        for p in T.RECON:
            ti = col[p]
            tr = gw_tr[np.isfinite(chem[gw_tr, ti])]
            if len(tr) > n_train:
                tr = sel.choice(tr, n_train, replace=False)
            feats = [col[q] for q in L.GENESIS_PARAMS if q != p]
            models[p] = (make(m, np.c_[Z[tr][:, feats], AZ[tr]], Z[tr, ti]), feats)

        def swap(model, ds, rows, target, allowed):
            mdl, feats = models[target]
            Zs, AZs = ds
            return predict(mdl, np.c_[Zs[rows][:, feats], AZs[rows]])
        T.L.fm_predict = swap
        r = np.random.default_rng(0)
        out[m] = [T.evaluate("site_disjoint", chem, None, (Z, AZ), clean, None, ns, col, r),
                  T.evaluate("bangladesh", bchem, None, (BZ, BAZ), bidx, None, ns, col, r)]
        for res in out[m]:
            a = res["all"]
            print(f"  {m:6s} {res['set']:13s} n={res['n']:5d} calcite {a['SI_calcite']['r2']:+.3f} "
                  f"gypsum {a['SI_gypsum']['r2']:+.3f} fluorite {a.get('SI_fluorite', {}).get('r2', float('nan')):+.3f} "
                  f"|CBE| obs {a['abs_CBE_obs']:.1f} rec {a['abs_CBE_rec']:.1f}", flush=True)
        json.dump(out, open(ROOT/"results/thermodynamic_consistency_baselines.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["ladder", "thermo", "both"])
    ap.add_argument("--methods", default="gb,rf,tabpfn")
    a = ap.parse_args()
    torch.set_num_threads(8)
    methods = a.methods.split(",")
    data = load()
    if a.mode in ("thermo", "both"):
        thermo(methods, *data)
    if a.mode in ("ladder", "both"):
        ladder(methods, *data)


if __name__ == "__main__":
    main()
