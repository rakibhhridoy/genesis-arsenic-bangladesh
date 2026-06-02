"""
GENESIS Paper 5 — Step 18: Leave-one-region-out (LORO) transfer benchmark.

Tests whether chemistry-only models transfer ACROSS regions, not just to
Bangladesh. For each held-out region R and each WHO target, we train on all
pairs OUTSIDE R and evaluate zero-shot on R, then relate the transfer AUC to how
far R sits from the training pool in standardized chemistry space (a
multivariate energy-distance proxy on the shared parameters).

Scientific question: is the Bangladesh transfer failure specific to its extreme
redox-regime shift, or do simple models fail to transfer everywhere? If LogReg
transfers well to low-shift regions (France, USA) but poorly to Bangladesh, the
transfer gap is governed by distribution distance — a quantitative law rather
than a flat negative.

This is the non-neural backbone; the frozen-encoder layer is added separately.

Usage:
  python src/18_region_transfer.py
"""
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model.genesis_encoder import GENESIS_PARAMS

# Clean, license-safe sources: the released processed tensors
# (genesis_temporal_pairs.pt + _meta.parquet, row-aligned) + the clean BD
# held-out pairs. We deliberately do NOT use genesis.duckdb (it contains raw
# GEMStat rows that cannot be redistributed) nor
# final_temporal_pairs_enriched_global.parquet (a pre-remediation artifact with
# France Eh = temperature and U inflated 1000x; see src/17). These derived
# tensors carry the correct remediated Eh/Cl/TDS/U values.
PAIRS_PT = Path("data/processed/genesis_temporal_pairs.pt")
PAIRS_META = Path("data/processed/genesis_temporal_pairs_meta.parquet")
BD_PT = Path("data/processed/genesis_held_out_bd_pairs.pt")
OUT = Path("results/region_transfer.json")


def load_clean_pairs():
    gp = torch.load(PAIRS_PT, map_location="cpu", weights_only=False)
    gm = pd.read_parquet(PAIRS_META)
    t0, t1 = gp["t0"].numpy(), gp["t1"].numpy()
    g = {f"{p}_t0": t0[:, i] for i, p in enumerate(GENESIS_PARAMS)}
    g.update({f"{p}_t1": t1[:, i] for i, p in enumerate(GENESIS_PARAMS)})
    g = pd.DataFrame(g)
    g["source"] = gm["source"].values
    g["country"] = gm["country"].values

    b = torch.load(BD_PT, map_location="cpu", weights_only=False)
    params = list(b["params"])
    bt0, bt1 = b["t0"].numpy(), b["t1"].numpy()
    bd = {f"{p}_t0": bt0[:, i] for i, p in enumerate(params)}
    bd.update({f"{p}_t1": bt1[:, i] for i, p in enumerate(params)})
    bd = pd.DataFrame(bd)
    bd["source"] = "bangladesh"; bd["country"] = "Bangladesh"
    return pd.concat([g, bd], ignore_index=True)
# WHO health-based drinking-water thresholds...
HEALTH = {"As": 10.0, "NO3": 50.0, "F": 1.5, "U": 30.0}
# ...plus the reducing-aquifer redox-indicator suite (Fe/Mn WHO operational,
# PO4 no WHO limit) — measured in Bangladesh and mechanistically coupled to As
# (reductive dissolution releases As, Fe, Mn, PO4 together). These give the
# Bangladesh transfer evaluation more than a single target.
REDOX = {"Fe": 0.3, "Mn": 0.4, "PO4": 0.5}
WHO = {**HEALTH, **REDOX}
CATEGORY = {**{k: "health" for k in HEALTH}, **{k: "redox-indicator" for k in REDOX}}
MIN_POS, MIN_NEG, MIN_TRAIN = 10, 10, 200  # viability gates for a held-out region


def region_of(row):
    s = str(row["source"])
    if "NWIS" in s or "USGS" in s:
        return "USA"
    if "ADES" in s or "BRGM" in s:
        return "France"
    if "bangla" in s.lower():
        return "Bangladesh"
    if "GEMS" in s.upper():
        return f"GEMStat:{row['country']}"
    return s


def boot_auc(y, p, n=200, seed=0):
    rng = np.random.default_rng(seed)
    aucs = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        aucs.append(roc_auc_score(y[s], p[s]))
    if not aucs:
        return (float("nan"), float("nan"), float("nan"))
    return (float(np.mean(aucs)),
            float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)))


def energy_distance(A, B, max_n=400, seed=0):
    """Standardized 4-D PCA energy distance proxy between two sample sets."""
    rng = np.random.default_rng(seed)
    def sub(X):
        return X[rng.choice(len(X), min(len(X), max_n), replace=False)] if len(X) > max_n else X
    A, B = sub(A), sub(B)
    def mpd(X, Y):
        d = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
        return d.mean()
    return float(2 * mpd(A, B) - mpd(A, A) - mpd(B, B))


def main():
    df = load_clean_pairs()
    df["region"] = df.apply(region_of, axis=1)
    feat_cols = [f"{p}_t0" for p in GENESIS_PARAMS]
    X_all = df[feat_cols].to_numpy(np.float64)

    # standardized matrix for distance (median-impute + z-score on full pool)
    imp = SimpleImputer(strategy="median").fit(X_all)
    sc = StandardScaler().fit(imp.transform(X_all))
    Xs = sc.transform(imp.transform(X_all))
    # 4-D PCA for energy distance
    Xc = Xs - Xs.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P4 = Xc @ Vt[:4].T

    results = []
    region_counts = df["region"].value_counts()
    regions = [r for r in region_counts.index if region_counts[r] >= MIN_TRAIN or r == "Bangladesh"]

    for target, thr in WHO.items():
        t1 = df[f"{target}_t1"].to_numpy(np.float64)
        valid = ~np.isnan(t1)
        y = (t1 > thr).astype(int)
        for R in regions:
            in_R = (df["region"] == R).to_numpy()
            te = valid & in_R
            tr = valid & ~in_R
            yte, ytr = y[te], y[tr]
            if yte.sum() < MIN_POS or (len(yte) - yte.sum()) < MIN_NEG:
                continue
            if ytr.sum() < MIN_POS or tr.sum() < MIN_TRAIN:
                continue
            # chem-only LogReg, train outside R -> test on R (zero-shot)
            clf = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            ).fit(X_all[tr], ytr)
            p = clf.predict_proba(X_all[te])[:, 1]
            auc_m, lo, hi = boot_auc(yte, p)
            ed = energy_distance(P4[te], P4[tr])
            results.append({
                "target": target, "category": CATEGORY[target], "held_out_region": R,
                "n_test": int(te.sum()), "n_pos": int(yte.sum()),
                "n_train": int(tr.sum()),
                "transfer_auc_mean": auc_m, "auc_ci95": [lo, hi],
                "energy_distance_4d": ed,
            })

    OUT.parent.mkdir(exist_ok=True)
    json.dump({"min_pos": MIN_POS, "results": results}, open(OUT, "w"), indent=2)

    # print table
    print(f"\n{'target':5s} {'held-out region':18s} {'n_pos/n':>10s} {'transfer AUC':>16s} {'energy_dist':>12s}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: (x["target"], -x["transfer_auc_mean"])):
        ci = r["auc_ci95"]
        print(f"{r['target']:5s} {r['held_out_region']:18s} "
              f"{str(r['n_pos'])+'/'+str(r['n_test']):>10s} "
              f"{r['transfer_auc_mean']:.3f} [{ci[0]:.3f},{ci[1]:.3f}]  "
              f"{r['energy_distance_4d']:11.3f}")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
