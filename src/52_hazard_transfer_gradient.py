#!/usr/bin/env python3
"""Do raster-only hazard maps lose skill out of region in proportion to redox control?

Global groundwater hazard maps (Podgorski & Berg 2020 As, 2022 F, 2026 Mn) fit a random
forest on surface rasters alone and project it everywhere. The Nature Water reframing
predicts that this works for the equilibrium-controlled majors and fails out of region
for the redox-controlled species, in the order of the GENESIS learnability gradient.

For every one of the 20 parameters we fit the same raster-only RF (the 18 GENESIS aux
rasters, no chemistry) to a site-level "high" label and compare in-distribution skill
(random CV) with out-of-region skill (spatial block CV at 1/5/10 degrees, leave-one-
region-out, Bangladesh test-only). The optimism gap is random minus out-of-region AUC.

Confounds fixed by design, before any robustness pass:
  prevalence  - "high" = above the parameter's own 80th percentile of site medians, so
                every parameter has 20% positives (guideline thresholds run separately)
  sample size - each parameter subsampled to the same number of sites per seed
  region mix  - a within-region analysis compares parameters inside one held-out
                region, so region difficulty cancels

  python3 src/52_hazard_transfer_gradient.py --seeds 3 --n 12000
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model.genesis_encoder import GENESIS_PARAMS, AUX_FEATURES

GUIDE = {"As": 10.0, "F": 1.5, "U": 30.0, "NO3": 50.0, "Fe": 0.3, "Mn": 0.08, "PO4": 0.5}
MIN_SITES, MIN_POS, MIN_NEG, MIN_TRAIN = 2000, 10, 10, 200
BLOCKS = [1.0, 5.0, 10.0]


def region_of(source, country):
    s = str(source)
    if "NWIS" in s or "USGS" in s:
        return "USA"
    if "ADES" in s or "BRGM" in s:
        return "France"
    if "EEA" in s or "WISE" in s:
        return "EU"
    if "GEMS" in s.upper():
        return f"GEMStat:{country}"
    return s


def load_sites():
    """Groundwater site medians for chemistry, plus the 18 rasters and a region label."""
    X = torch.load(ROOT/"data/processed/genesis_pretrain.pt", map_location="cpu",
                   weights_only=False).numpy()
    m = pd.read_parquet(ROOT/"data/processed/genesis_pretrain_meta.parquet",
                        columns=["site_id", "lat", "lon", "country", "source",
                                 "water_body_type"] + AUX_FEATURES)
    gw = (m.water_body_type == "groundwater").values
    c = pd.DataFrame(X[gw], columns=GENESIS_PARAMS)
    c["site_id"] = m.site_id.values[gw]
    chem = c.groupby("site_id")[GENESIS_PARAMS].median()
    first = m[gw].groupby("site_id")[["lat", "lon", "country", "source"] + AUX_FEATURES].first()
    s = chem.join(first)
    s["region"] = [region_of(a, b) for a, b in zip(s.source, s.country)]

    bx = torch.load(ROOT/"data/processed/genesis_held_out_bd.pt", map_location="cpu",
                    weights_only=False).numpy()
    bm = pd.read_parquet(ROOT/"data/processed/genesis_held_out_bd_meta.parquet")
    b = pd.DataFrame(bx, columns=GENESIS_PARAMS)
    b["site_id"] = bm.site_id.values
    bchem = b.groupby("site_id")[GENESIS_PARAMS].median()
    bfirst = bm.groupby("site_id")[["lat", "lon", "country", "source"] + AUX_FEATURES].first()
    bs = bchem.join(bfirst)
    bs["region"] = "Bangladesh"
    return s.reset_index(), bs.reset_index()


def rf(seed):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                  class_weight="balanced", n_jobs=-1, random_state=seed)


def cv_auc(X, y, splitter, groups, seed):
    pred = np.full(len(y), np.nan)
    for tr, te in splitter.split(X, y, groups):
        if y[tr].sum() < MIN_POS:
            continue
        pred[te] = rf(seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    ok = np.isfinite(pred)
    return float(roc_auc_score(y[ok], pred[ok])) if len(np.unique(y[ok])) == 2 else np.nan


def evaluate(d, bd, p, thr, n, seed):
    rng = np.random.default_rng(seed)
    d = d[d[p].notna()]
    if len(d) > n:
        d = d.iloc[rng.choice(len(d), n, replace=False)]
    X = d[AUX_FEATURES].to_numpy(np.float64)
    X = np.where(np.isfinite(X), X, np.nanmedian(X, 0))
    y = (d[p].to_numpy() > thr).astype(int)
    if y.sum() < 50 or (len(y) - y.sum()) < 50:
        return None
    out = {"param": p, "seed": seed, "n": int(len(y)), "prev": float(y.mean())}
    out["random"] = cv_auc(X, y, StratifiedKFold(5, shuffle=True, random_state=seed), None, seed)
    for bdeg in BLOCKS:
        g = (np.floor(d.lat.values / bdeg).astype(int) * 10000
             + np.floor(d.lon.values / bdeg).astype(int))
        k = min(5, len(np.unique(g)))
        out[f"block{bdeg:g}"] = cv_auc(X, y, GroupKFold(k), g, seed) if k >= 2 else np.nan

    reg = d.region.values
    per = {}
    for R in pd.unique(reg):
        te, tr = reg == R, reg != R
        if y[te].sum() < MIN_POS or (te.sum() - y[te].sum()) < MIN_NEG:
            continue
        if tr.sum() < MIN_TRAIN or y[tr].sum() < MIN_POS:
            continue
        pr = rf(seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        per[R] = float(roc_auc_score(y[te], pr))
    b = bd[bd[p].notna()]
    yb = (b[p].to_numpy() > thr).astype(int)
    if yb.sum() >= MIN_POS and (len(yb) - yb.sum()) >= MIN_NEG:
        Xb = b[AUX_FEATURES].to_numpy(np.float64)
        Xb = np.where(np.isfinite(Xb), Xb, np.nanmedian(X, 0))
        per["Bangladesh"] = float(roc_auc_score(yb, rf(seed).fit(X, y).predict_proba(Xb)[:, 1]))
    out["loro"] = per
    out["loro_mean"] = float(np.mean(list(per.values()))) if per else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--out", default=str(ROOT/"results/hazard_transfer_gradient.json"))
    a = ap.parse_args()

    s, bd = load_sites()
    print(f"groundwater sites {len(s):,} in {s.region.nunique()} regions; Bangladesh {len(bd):,}",
          flush=True)
    r2 = {k: v["r2"] for k, v in json.load(open(ROOT/"results/reconstruction_probe.json"))["corpus"].items()}

    rows = []
    for mode in ["p80", "guideline"]:
        params = GENESIS_PARAMS if mode == "p80" else list(GUIDE)
        for p in params:
            if s[p].notna().sum() < MIN_SITES:
                print(f"  skip {p}: {s[p].notna().sum()} sites", flush=True)
                continue
            thr = float(np.nanpercentile(s[p], 80)) if mode == "p80" else GUIDE[p]
            for seed in range(a.seeds):
                o = evaluate(s, bd, p, thr, a.n, seed)
                if o is None:
                    continue
                o.update(mode=mode, thr=thr, r2=r2.get(p))
                rows.append(o)
                print(f"  {mode:9s} {p:5s} seed{seed} n={o['n']:>6,} random={o['random']:.3f} "
                      f"b1={o['block1']:.3f} b5={o['block5']:.3f} b10={o['block10']:.3f} "
                      f"loro={o['loro_mean']:.3f} ({len(o['loro'])} regions)", flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
