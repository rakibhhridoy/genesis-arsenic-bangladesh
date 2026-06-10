"""GENESIS Paper 5 — Step 33: does the fine-tune advantage survive a STRONG tree?

The redox-suite fine-tune advantage reported against the default random forest
(src/30) is re-tested against progressively stronger tree baselines on the
identical leave-one-region-out splits:

  rf_default   RandomForest(n_estimators=400)            -- the src/27 baseline
  rf_reg       RandomForest(max_depth=12, leaf=5, sqrt)  -- regularized
  rf_leaf      RandomForest(min_samples_leaf=20, sqrt)   -- leaf-regularized
  histgb       HistGradientBoosting (native NaN, l2=1)   -- strong modern GBT
  xgb_strong   XGBoost(600, depth 5, lr 0.03)            -- strong GBT
  best_tree    per-cell max over the above               -- most adversarial

Conclusion (see manuscript): the advantage is significant only against the
default RF; against a regularized RF or HistGradientBoosting it falls to a
statistical tie. This is the empirical core of the paper's strong-baseline
message: apparent wins over a default tree need not survive a strong one.

Inputs (released, CPU-only, no encoder):
  data/processed/genesis_temporal_pairs*.{pt,parquet}, genesis_held_out_bd_pairs.pt
  results/loro_finetune_large_multiseed.json   (seed-averaged fine-tune AUC)

Usage:  python src/33_baseline_strength.py
Writes: results/baseline_strength.json
"""
import sys, json, warnings
from collections import defaultdict
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
import xgboost as xgb

m19 = __import__("19_region_transfer_encoder")
region_of, TARGETS = m19.region_of, m19.TARGETS
MIN_POS, MIN_NEG, MIN_TRAIN = m19.MIN_POS, m19.MIN_NEG, m19.MIN_TRAIN
from model.genesis_encoder import GENESIS_PARAMS

REDOX = {"As", "Fe", "Mn", "PO4"}; CONS = {"NO3", "F"}; URAN = {"U"}
TREES = ["rf_default", "rf_reg", "rf_leaf", "histgb", "xgb_strong", "best_tree"]


def _pre(): return [SimpleImputer(strategy="median"), StandardScaler()]
def rf(Xtr, ytr, Xte, **kw):
    p = make_pipeline(*_pre(), RandomForestClassifier(
        n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=0, **kw))
    p.fit(Xtr, ytr); return p.predict_proba(Xte)[:, 1]
def histgb(Xtr, ytr, Xte):
    pos = max(ytr.sum(), 1); neg = max(len(ytr) - ytr.sum(), 1)
    sw = np.where(ytr == 1, neg / pos, 1.0)
    c = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
        max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
    c.fit(Xtr, ytr, sample_weight=sw); return c.predict_proba(Xte)[:, 1]
def xgb_strong(Xtr, ytr, Xte):
    pos = max(int(ytr.sum()), 1); neg = max(len(ytr) - int(ytr.sum()), 1)
    c = xgb.XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, scale_pos_weight=neg / pos,
        eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=0)
    c.fit(Xtr, ytr); return c.predict_proba(Xte)[:, 1]


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    out = dict(n=len(a), mean_ft=float(a.mean()), mean_tree=float(b.mean()),
               delta=float((a - b).mean()), wins=int((a > b).sum()))
    try:
        _, p = wilcoxon(a, b); out["wilcoxon_p"] = float(p); out["significant"] = bool(p < 0.05)
    except ValueError:
        out["wilcoxon_p"] = None; out["significant"] = False
    return out


def main():
    gp = torch.load("data/processed/genesis_temporal_pairs.pt", map_location="cpu", weights_only=False)
    gm = pd.read_parquet("data/processed/genesis_temporal_pairs_meta.parquet")
    bp = torch.load("data/processed/genesis_held_out_bd_pairs.pt", map_location="cpu", weights_only=False)
    chem = np.concatenate([gp["t0"].numpy(), bp["t0"].numpy()], 0)
    t1 = np.concatenate([gp["t1"].numpy(), bp["t1"].numpy()], 0)
    region = np.array([region_of(s, c) for s, c in zip(gm["source"], gm["country"])]
                      + ["Bangladesh"] * len(bp["t0"]))
    regions = [r for r in pd.Series(region).value_counts().index
               if (region == r).sum() >= MIN_TRAIN or r == "Bangladesh"]

    ftrows = json.load(open("results/loro_finetune_large_multiseed.json"))["per_cell_seedmean"]
    ft = {(r["target"], r["held_out_region"]): r["finetune_auc_mean"] for r in ftrows}
    # Use the SAME stored default-RF as the rest of the paper (src/19+27 -> redox_dissociation,
    # fig13) so the default-RF column is identical everywhere; only the STRONGER trees are re-fit here.
    rf_stored = {(c["target"], c["held_out_region"]): c.get("rf_auc")
                 for c in json.load(open("results/region_transfer_encoder_large.json"))["results"]}

    cells = []
    for target, thr in TARGETS.items():
        ti = GENESIS_PARAMS.index(target); tv = t1[:, ti]
        valid = ~np.isnan(tv); y = (tv > thr).astype(int)
        for R in regions:
            inR = region == R; te, tr = valid & inR, valid & ~inR
            yte, ytr = y[te], y[tr]
            if yte.sum() < MIN_POS or (len(yte) - yte.sum()) < MIN_NEG: continue
            if ytr.sum() < MIN_POS or tr.sum() < MIN_TRAIN: continue
            if (target, R) not in ft: continue
            if rf_stored.get((target, R)) is None: continue
            Xtr, Xte = chem[tr], chem[te]
            d = dict(target=target, region=R, ft=ft[(target, R)])
            d["rf_default"] = rf_stored[(target, R)]   # stored, paper-consistent default RF
            d["rf_reg"] = roc_auc_score(yte, rf(Xtr, ytr, Xte, max_depth=12, min_samples_leaf=5, max_features="sqrt"))
            d["rf_leaf"] = roc_auc_score(yte, rf(Xtr, ytr, Xte, min_samples_leaf=20, max_features="sqrt"))
            d["histgb"] = roc_auc_score(yte, histgb(Xtr, ytr, Xte))
            d["xgb_strong"] = roc_auc_score(yte, xgb_strong(Xtr, ytr, Xte))
            d["best_tree"] = max(d[k] for k in ["rf_default", "rf_reg", "rf_leaf", "histgb", "xgb_strong"])
            cells.append(d)
            print(f"{target:4s} {R:18s} FT={d['ft']:.3f} | "
                  f"RFdef={d['rf_default']:.3f} RFreg={d['rf_reg']:.3f} "
                  f"HistGB={d['histgb']:.3f} XGB={d['xgb_strong']:.3f} best={d['best_tree']:.3f}")

    groups = {"REDOX_AsFeMnPO4": REDOX, "CONSERVATIVE_NO3F": CONS, "URANIUM": URAN, "ALL_47": None}
    summary = {}
    print("\n=== fine-tuned encoder vs each tree baseline, by group ===")
    for gname, sel in groups.items():
        cc = [c for c in cells if sel is None or c["target"] in sel]
        summary[gname] = {}
        for tk in TREES:
            st = paired([c["ft"] for c in cc], [c[tk] for c in cc])
            summary[gname][tk] = st
        r = summary[gname]
        print(f"\n{gname} (n={r['rf_default']['n']}):")
        for tk in TREES:
            s = r[tk]
            print(f"  FT vs {tk:11s} Δ{s['delta']:+.3f} win {s['wins']}/{s['n']} "
                  f"p={s['wilcoxon_p']:.4f} {'SIG' if s['significant'] else 'ns'}")

    json.dump({"per_cell": cells, "group_summary": summary}, open("results/baseline_strength.json", "w"), indent=1)
    print("\nSaved results/baseline_strength.json")


if __name__ == "__main__":
    main()
