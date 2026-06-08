"""
GENESIS Paper 5 — Step 28: can the encoder beat Random Forest when given a FAIR fight?

src/19 scored the encoder with a LINEAR probe (LogReg on [CLS]); src/27 showed a
non-pretrained Random Forest on raw chemistry matches it. But that comparison
handicaps the encoder twice over: (i) a linear head cannot exploit a nonlinear
representation, and (ii) the baselines saw chemistry only while the encoder also
ingests 18 satellite aux features + location.

This script gives the encoder its strongest fair shot, on identical LORO splits:
  - matched HEAD: RF and XGBoost on the [CLS] embedding (not just LogReg);
  - matched INFORMATION: the raw baselines now get chem + aux + lat/lon, the same
    inputs the encoder sees (we also keep chem-only for the gap);
  - a concat config: [embedding ++ raw] -> tree, to test whether the representation
    adds anything ON TOP of the raw inputs.

The decisive cell is RF-on-embedding vs RF-on-raw(chem+aux+loc): same head, same
information source. If the embedding wins, the pretrained nonlinear integration
beats a tree's integration of the same inputs.

Embeds once (guarded for 8GB MPS) and caches to results/loro_emb_<size>.npz.
CPU after that. Usage:
  python src/28_encoder_vs_rf_tournament.py --encoder_size large \
      --ckpt checkpoints/large_stage2_frompod/best.pt
"""
import argparse
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
from model.genesis_encoder import GENESISForMGM, GENESIS_PARAMS
from model.temporal_dataset import TemporalPairDataset, temporal_collate_fn
from model.dataset import load_stats

_m = __import__("19_region_transfer_encoder")
region_of, boot_auc, extract_embeddings = _m.region_of, _m.boot_auc, _m.extract_embeddings
TARGETS = _m.TARGETS
MIN_POS, MIN_NEG, MIN_TRAIN = _m.MIN_POS, _m.MIN_NEG, _m.MIN_TRAIN
NORM = "data/processed/normalization_stats.json"
LOC_COLS = ["lat", "lon"]


def rf():
    return RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                  n_jobs=-1, random_state=0)


def xgbc(ytr):
    pos = max(int(ytr.sum()), 1); neg = max(len(ytr) - int(ytr.sum()), 1)
    return xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9,
                             scale_pos_weight=neg / pos, eval_metric="logloss",
                             tree_method="hist", n_jobs=-1, random_state=0)


def pipe(model):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_size", default="large")
    ap.add_argument("--ckpt", default="checkpoints/large_stage2_frompod/best.pt")
    ap.add_argument("--out", default="results/encoder_vs_rf_tournament_large.json")
    args = ap.parse_args()
    cache = f"results/loro_emb_{args.encoder_size}.npz"

    ns = load_stats(NORM)
    gp = torch.load("data/processed/genesis_temporal_pairs.pt", map_location="cpu", weights_only=False)
    ga = torch.load("data/processed/genesis_temporal_pairs_aux.pt", map_location="cpu", weights_only=False)
    ga = (ga["aux"] if isinstance(ga, dict) else ga).numpy()
    gm = pd.read_parquet("data/processed/genesis_temporal_pairs_meta.parquet")
    bp = torch.load("data/processed/genesis_held_out_bd_pairs.pt", map_location="cpu", weights_only=False)
    ba = torch.load("data/processed/genesis_held_out_bd_pairs_aux.pt", map_location="cpu", weights_only=False)
    ba = (ba["aux"] if isinstance(ba, dict) else ba).numpy()
    bm = pd.read_parquet("data/processed/genesis_held_out_bd_pairs_meta.parquet")

    g_t0, g_t1 = gp["t0"].numpy(), gp["t1"].numpy()
    g_gap = gp["gap_years"].numpy()
    b_t0, b_t1 = bp["t0"].numpy(), bp["t1"].numpy()
    b_gap = np.full(len(b_t0), float(bp["gap_years"]))

    chem = np.concatenate([g_t0, b_t0], 0)
    aux = np.concatenate([ga, ba], 0)
    loc = np.concatenate([gm[LOC_COLS].to_numpy(float), bm[LOC_COLS].to_numpy(float)], 0)
    raw = np.concatenate([chem, aux, loc], 1)               # chem+aux+loc (matched to encoder)
    t1 = np.concatenate([g_t1, b_t1], 0)
    region = np.array([region_of(s, c) for s, c in zip(gm["source"], gm["country"])]
                      + ["Bangladesh"] * len(b_t0))

    # ---- embeddings (cached) ----
    if Path(cache).exists():
        emb = np.load(cache)["emb"]
        print(f"loaded cached embeddings {emb.shape} from {cache}")
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        enc = GENESISForMGM(args.encoder_size)
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        enc.load_state_dict(ck.get("model_state_dict", ck), strict=False)
        enc.to(device)
        print(f"Encoder {args.encoder_size} on {device}; embedding (guarded)...")
        ds_g = TemporalPairDataset(t0_values=g_t0, t1_values=g_t1, gap_years=g_gap,
                                   norm_stats=ns, meta_df=gm, aux_values=ga)
        ds_b = TemporalPairDataset(t0_values=b_t0, t1_values=b_t1, gap_years=b_gap,
                                   norm_stats=ns, meta_df=bm, aux_values=ba)
        emb = np.concatenate([extract_embeddings(enc, ds_g, device),
                              extract_embeddings(enc, ds_b, device)], 0)
        np.savez_compressed(cache, emb=emb)
        print(f"cached embeddings {emb.shape} -> {cache}")

    counts = pd.Series(region).value_counts()
    regions = [r for r in counts.index if counts[r] >= MIN_TRAIN or r == "Bangladesh"]

    # configs: (name, feature_matrix, head_factory)   head_factory(ytr)->fitted-pipe-spec
    def configs(ytr):
        return [
            ("rf_chem",     chem, rf()),            # original RF baseline (chem only)
            ("rf_raw",      raw,  rf()),            # RF on matched inputs chem+aux+loc
            ("rf_emb",      emb,  rf()),            # RF on the representation
            ("rf_emb_raw",  None, rf()),            # RF on [emb ++ raw]
            ("xgb_raw",     raw,  xgbc(ytr)),       # XGB on matched raw
            ("xgb_emb",     emb,  xgbc(ytr)),       # XGB on representation
            ("logreg_emb",  emb,  LogisticRegression(max_iter=2000, class_weight="balanced")),  # = src/19 encoder_auc
            ("logreg_chem", chem, LogisticRegression(max_iter=2000, class_weight="balanced")),  # = src/19 chem_auc
        ]

    rows = []
    for target, thr in TARGETS.items():
        ti = GENESIS_PARAMS.index(target)
        tv = t1[:, ti]; valid = ~np.isnan(tv); y = (tv > thr).astype(int)
        for R in regions:
            inR = region == R
            te, tr = valid & inR, valid & ~inR
            yte, ytr = y[te], y[tr]
            if yte.sum() < MIN_POS or (len(yte) - yte.sum()) < MIN_NEG: continue
            if ytr.sum() < MIN_POS or tr.sum() < MIN_TRAIN: continue
            cell = {"target": target, "held_out_region": R,
                    "n_test": int(te.sum()), "n_pos": int(yte.sum())}
            for name, F, model in configs(ytr):
                if name == "rf_emb_raw":
                    F = np.concatenate([emb, raw], 1)
                clf = pipe(model)
                clf.fit(F[tr], ytr)
                p = clf.predict_proba(F[te])[:, 1]
                a, lo, hi = boot_auc(yte, p)
                cell[name] = a; cell[name + "_ci"] = [lo, hi]
            rows.append(cell)
            print(f"{target:4s} {R:18s} "
                  f"rf_chem={cell['rf_chem']:.3f} rf_raw={cell['rf_raw']:.3f} "
                  f"rf_emb={cell['rf_emb']:.3f} xgb_emb={cell['xgb_emb']:.3f} "
                  f"logreg_emb={cell['logreg_emb']:.3f}")

    json.dump({"encoder_size": args.encoder_size, "ckpt": args.ckpt,
               "note": "matched-head/matched-input encoder-vs-tree tournament, LORO splits",
               "rows": rows}, open(args.out, "w"), indent=2)
    print(f"\nSaved {args.out} ({len(rows)} cells)")

    # ---- verdict ----
    def best_enc(c): return max(c["rf_emb"], c["xgb_emb"], c["logreg_emb"])
    def best_rf(c):  return max(c["rf_raw"], c["rf_chem"])
    enc_mean = np.mean([best_enc(c) for c in rows])
    rf_mean = np.mean([best_rf(c) for c in rows])
    enc_beats = sum(1 for c in rows if c["rf_emb_ci"][0] > c["rf_raw_ci"][1])
    rf_beats = sum(1 for c in rows if c["rf_raw_ci"][0] > c["rf_emb_ci"][1])
    print("\n=== VERDICT (best encoder-head vs best RF-on-raw) ===")
    print(f"  mean best-encoder AUC {enc_mean:.3f}  vs  mean best-RF AUC {rf_mean:.3f}")
    print(f"  rf_emb significantly beats rf_raw: {enc_beats}/{len(rows)}  |  "
          f"rf_raw beats rf_emb: {rf_beats}/{len(rows)}")
    print("\n=== U/USA + positive cells ===")
    for c in sorted(rows, key=lambda x: (x["target"], x["held_out_region"])):
        if c["target"] in ("U", "PO4") or c["held_out_region"] == "Bangladesh":
            print(f"  {c['target']:4s} {c['held_out_region']:16s} "
                  f"rf_chem={c['rf_chem']:.3f} rf_raw={c['rf_raw']:.3f} "
                  f"rf_emb={c['rf_emb']:.3f} xgb_emb={c['xgb_emb']:.3f} "
                  f"logreg_emb={c['logreg_emb']:.3f} | "
                  f"best_enc={best_enc(c):.3f} best_rf={best_rf(c):.3f}")


if __name__ == "__main__":
    main()
