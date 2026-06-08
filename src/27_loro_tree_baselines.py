"""
GENESIS Paper 5 — Step 27: nonlinear non-pretrained baselines for the LORO law.

The headline positive claim (uranium: pretraining beats logistic regression) is
tested in src/19 only against a LINEAR baseline. The essential control is whether
a NONLINEAR non-pretrained model also beats logistic regression there: if XGBoost
or Random Forest match the encoder on uranium, the advantage is attributable to
nonlinearity, not to pretraining. To credit pretraining, the encoder must beat the
tree baselines too.

These baselines are fit on RAW chemistry (the same 20-feature vector the chem-only
LogReg uses) on the SAME leave-one-region-out splits as src/19 — so they are
encoder-independent and identical across all three encoder sizes; we compute them
once. As a correctness check we also recompute chem_auc with the exact src/19
recipe and assert it matches the stored values per cell (proving identical splits).

No encoder, no checkpoint, no MPS — pure CPU. Merges xgb_*/rf_* columns into the
existing results/region_transfer_encoder*.json files and writes a standalone
results/region_transfer_tree_baselines.json.

Usage:
  python src/27_loro_tree_baselines.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from model.genesis_encoder import GENESIS_PARAMS

# reuse src/19's exact constants + split helpers so splits are provably identical
sys.path.insert(0, str(Path(__file__).parent))
_m = __import__("19_region_transfer_encoder")
region_of, boot_auc = _m.region_of, _m.boot_auc
HEALTH, REDOX, TARGETS = _m.HEALTH, _m.REDOX, _m.TARGETS
MIN_POS, MIN_NEG, MIN_TRAIN = _m.MIN_POS, _m.MIN_NEG, _m.MIN_TRAIN

ENCODER_FILES = {
    "small": "results/region_transfer_encoder_small.json",
    "base":  "results/region_transfer_encoder.json",
    "large": "results/region_transfer_encoder_large.json",
}


def fit_predict_xgb(Xtr, ytr, Xte):
    # raw chemistry with native NaN handling — same recipe as the BD-As table
    pos = max(int(ytr.sum()), 1)
    neg = max(len(ytr) - int(ytr.sum()), 1)
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=neg / pos, eval_metric="logloss",
        tree_method="hist", n_jobs=-1, random_state=0,
    )
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def fit_predict_rf(Xtr, ytr, Xte):
    # RF cannot ingest NaN — impute (median) + scale, matching the chem-LogReg pre
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RandomForestClassifier(n_estimators=400, class_weight="balanced",
                               n_jobs=-1, random_state=0),
    )
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def fit_predict_chem(Xtr, ytr, Xte):
    # identical to src/19 chem-only baseline — used as the split-identity check
    clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def main():
    # ---- load raw pairs (global + BD); no encoder ----
    gp = torch.load("data/processed/genesis_temporal_pairs.pt", map_location="cpu", weights_only=False)
    gm = pd.read_parquet("data/processed/genesis_temporal_pairs_meta.parquet")
    bp = torch.load("data/processed/genesis_held_out_bd_pairs.pt", map_location="cpu", weights_only=False)

    g_t0, g_t1 = gp["t0"].numpy(), gp["t1"].numpy()
    b_t0, b_t1 = bp["t0"].numpy(), bp["t1"].numpy()

    chem = np.concatenate([g_t0, b_t0], 0)
    t1 = np.concatenate([g_t1, b_t1], 0)
    region = np.array([region_of(s, c) for s, c in zip(gm["source"], gm["country"])]
                      + ["Bangladesh"] * len(b_t0))

    counts = pd.Series(region).value_counts()
    regions = [r for r in counts.index if counts[r] >= MIN_TRAIN or r == "Bangladesh"]
    print(f"chem {chem.shape}, regions {len(set(region))}")

    rows = []
    for target, thr in TARGETS.items():
        ti = GENESIS_PARAMS.index(target)
        tv = t1[:, ti]
        valid = ~np.isnan(tv)
        y = (tv > thr).astype(int)
        for R in regions:
            inR = region == R
            te, tr = valid & inR, valid & ~inR
            yte, ytr = y[te], y[tr]
            if yte.sum() < MIN_POS or (len(yte) - yte.sum()) < MIN_NEG:
                continue
            if ytr.sum() < MIN_POS or tr.sum() < MIN_TRAIN:
                continue
            Xtr, Xte = chem[tr], chem[te]
            pc = fit_predict_chem(Xtr, ytr, Xte)
            px = fit_predict_xgb(Xtr, ytr, Xte)
            pr = fit_predict_rf(Xtr, ytr, Xte)
            ca, clo, chi = boot_auc(yte, pc)
            xa, xlo, xhi = boot_auc(yte, px)
            ra, rlo, rhi = boot_auc(yte, pr)
            rows.append({
                "target": target, "held_out_region": R,
                "n_test": int(te.sum()), "n_pos": int(yte.sum()),
                "chem_auc_recomputed": ca,
                "xgb_auc": xa, "xgb_ci95": [xlo, xhi],
                "rf_auc": ra, "rf_ci95": [rlo, rhi],
            })
            print(f"{target:5s} {R:18s} n={int(te.sum()):5d} "
                  f"chem={ca:.3f} xgb={xa:.3f} rf={ra:.3f}")

    base = {(r["target"], r["held_out_region"]): r for r in rows}
    json.dump({"note": "encoder-independent nonlinear baselines on raw chemistry, "
                       "same LORO splits as src/19", "rows": rows},
              open("results/region_transfer_tree_baselines.json", "w"), indent=2)
    print(f"\nSaved results/region_transfer_tree_baselines.json ({len(rows)} cells)")

    # ---- split-identity check + merge into each encoder file ----
    print("\n=== split-identity check (recomputed chem_auc vs stored) ===")
    for size, path in ENCODER_FILES.items():
        if not Path(path).exists():
            print(f"  {size}: {path} absent, skipping merge")
            continue
        doc = json.load(open(path))
        cells = doc["results"] if isinstance(doc, dict) and "results" in doc else doc
        max_diff, merged, beats_xgb, beats_rf = 0.0, 0, 0, 0
        for c in cells:
            b = base.get((c["target"], c["held_out_region"]))
            if not b:
                continue
            max_diff = max(max_diff, abs(c["chem_auc"] - b["chem_auc_recomputed"]))
            c["xgb_auc"] = b["xgb_auc"]; c["xgb_ci95"] = b["xgb_ci95"]
            c["rf_auc"] = b["rf_auc"]; c["rf_ci95"] = b["rf_ci95"]
            c["encoder_minus_xgb"] = c["encoder_auc"] - b["xgb_auc"]
            c["encoder_minus_rf"] = c["encoder_auc"] - b["rf_auc"]
            # non-overlapping CI & higher, same rule as encoder_beats (vs chem)
            c["encoder_beats_xgb"] = bool(c["encoder_ci95"][0] > b["xgb_ci95"][1])
            c["encoder_beats_rf"] = bool(c["encoder_ci95"][0] > b["rf_ci95"][1])
            beats_xgb += c["encoder_beats_xgb"]; beats_rf += c["encoder_beats_rf"]
            merged += 1
        json.dump(doc, open(path, "w"), indent=2)
        print(f"  {size:5s}: merged {merged} cells | max |Δchem_auc| = {max_diff:.4f} "
              f"({'IDENTICAL splits' if max_diff < 1e-9 else 'CHECK'}) | "
              f"encoder beats xgb {beats_xgb}/{merged}, beats rf {beats_rf}/{merged}")

    # ---- focused report: does pretraining survive the nonlinear control? ----
    print("\n=== uranium + positive cells: encoder vs chem vs xgb vs rf ===")
    doc = json.load(open(ENCODER_FILES["large"])) if Path(ENCODER_FILES["large"]).exists() \
        else json.load(open(ENCODER_FILES["base"]))
    cells = doc["results"] if isinstance(doc, dict) and "results" in doc else doc
    focus = [c for c in cells if c.get("encoder_beats") or c["target"] in ("U", "PO4")]
    print(f"{'target':5s} {'region':16s} {'enc':>6s} {'chem':>6s} {'xgb':>6s} {'rf':>6s}"
          f" {'e-xgb':>7s} {'>xgb?':>6s}")
    for c in sorted(focus, key=lambda x: (x["target"], -x.get("encoder_minus_chem", 0))):
        if "xgb_auc" not in c:
            continue
        print(f"{c['target']:5s} {c['held_out_region']:16s} {c['encoder_auc']:6.3f} "
              f"{c['chem_auc']:6.3f} {c['xgb_auc']:6.3f} {c['rf_auc']:6.3f} "
              f"{c['encoder_minus_xgb']:+7.3f} {str(c['encoder_beats_xgb']):>6s}")


if __name__ == "__main__":
    main()
