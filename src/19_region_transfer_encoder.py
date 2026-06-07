"""
GENESIS Paper 5 — Step 19: Leave-one-region-out transfer, FROZEN ENCODER vs
chemistry-only LogReg, head-to-head on identical splits.

The decisive global test: does the pretrained foundation-model representation beat
a chemistry-only logistic regression on ANY region x target cell, or is the
negative result global across the whole benchmark?

For each pair we extract the frozen encoder's [CLS] embedding once (the same
feature the paper's "frozen + linear probe" config uses). Then, per held-out
region and target, we train a LogReg on out-of-region data and evaluate zero-shot
on the held-out region, for BOTH the encoder embedding and the raw chemistry,
on the SAME split — so the comparison is apples-to-apples.

Usage:
  python src/19_region_transfer_encoder.py --encoder_size base \
      --ckpt base_full/checkpoints/base_stage2/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from model.genesis_encoder import GENESISForMGM, GENESIS_PARAMS
from model.temporal_dataset import TemporalPairDataset, temporal_collate_fn
from model.dataset import load_stats

HEALTH = {"As": 10.0, "NO3": 50.0, "F": 1.5, "U": 30.0}
REDOX = {"Fe": 0.3, "Mn": 0.4, "PO4": 0.5}
TARGETS = {**HEALTH, **REDOX}
MIN_POS, MIN_NEG, MIN_TRAIN = 10, 10, 200
NORM = "data/processed/normalization_stats.json"


def region_of(source, country):
    s = str(source)
    if "NWIS" in s or "USGS" in s:
        return "USA"
    if "ADES" in s or "BRGM" in s:
        return "France"
    if "EEA" in s or "WISE" in s:
        return "EU"
    if "bangla" in s.lower():
        return "Bangladesh"
    if "GEMS" in s.upper():
        return f"GEMStat:{country}"
    return s


def boot_auc(y, p, n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        aucs.append(roc_auc_score(y[s], p[s]))
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def extract_embeddings(enc, ds, device, bs=128):
    # bs kept small + MPS pool released each batch: on an 8GB M1, MPS unified
    # memory competes with the OS, and PyTorch caches its pool across batches.
    # Without empty_cache() the high-water mark of a full embed run stays
    # resident and can starve WindowServer into a watchdog freeze.
    dl = DataLoader(ds, batch_size=bs, collate_fn=temporal_collate_fn, num_workers=0)
    out = []
    enc.eval()
    with torch.no_grad():
        for b in dl:
            cur = {k: v.to(device) for k, v in b["current"].items()}
            z = enc.get_latent(cur["param_ids"], cur["values"], cur["padding_mask"])
            out.append(z.cpu().numpy())
            del cur, z
            if device == "mps":
                torch.mps.empty_cache()
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_size", default="base")
    ap.add_argument("--ckpt", default="base_full/checkpoints/base_stage2/best.pt")
    ap.add_argument("--out", default="results/region_transfer_encoder.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ns = load_stats(NORM)

    # ---- load clean pairs (global + BD) ----
    gp = torch.load("data/processed/genesis_temporal_pairs.pt", map_location="cpu", weights_only=False)
    ga = torch.load("data/processed/genesis_temporal_pairs_aux.pt", map_location="cpu", weights_only=False)
    ga = ga["aux"] if isinstance(ga, dict) else ga
    gm = pd.read_parquet("data/processed/genesis_temporal_pairs_meta.parquet")
    bp = torch.load("data/processed/genesis_held_out_bd_pairs.pt", map_location="cpu", weights_only=False)
    ba = torch.load("data/processed/genesis_held_out_bd_pairs_aux.pt", map_location="cpu", weights_only=False)
    ba = ba["aux"] if isinstance(ba, dict) else ba
    bm = pd.read_parquet("data/processed/genesis_held_out_bd_pairs_meta.parquet")

    g_t0, g_t1 = gp["t0"].numpy(), gp["t1"].numpy()
    g_gap = gp["gap_years"].numpy()
    b_t0, b_t1 = bp["t0"].numpy(), bp["t1"].numpy()
    b_gap = np.full(len(b_t0), float(bp["gap_years"]))

    # ---- build datasets + extract frozen embeddings ----
    enc = GENESISForMGM(args.encoder_size)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    enc.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    enc.to(device)
    print(f"Encoder {args.encoder_size} loaded on {device}; extracting embeddings...")

    ds_g = TemporalPairDataset(t0_values=g_t0, t1_values=g_t1, gap_years=g_gap,
                               norm_stats=ns, meta_df=gm, aux_values=ga.numpy())
    ds_b = TemporalPairDataset(t0_values=b_t0, t1_values=b_t1, gap_years=b_gap,
                               norm_stats=ns, meta_df=bm, aux_values=ba.numpy())
    emb = np.concatenate([extract_embeddings(enc, ds_g, device),
                          extract_embeddings(enc, ds_b, device)], 0)

    chem = np.concatenate([g_t0, b_t0], 0)
    t1 = np.concatenate([g_t1, b_t1], 0)
    region = np.array([region_of(s, c) for s, c in zip(gm["source"], gm["country"])]
                      + ["Bangladesh"] * len(b_t0))
    print(f"Embeddings {emb.shape}, chem {chem.shape}, regions {len(set(region))}")

    counts = pd.Series(region).value_counts()
    regions = [r for r in counts.index if counts[r] >= MIN_TRAIN or r == "Bangladesh"]

    results = []
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
            # encoder (frozen linear probe = LogReg on [CLS] embedding)
            enc_clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=2000, class_weight="balanced"))
            enc_clf.fit(emb[tr], ytr)
            pe = enc_clf.predict_proba(emb[te])[:, 1]
            # chemistry-only baseline (same split)
            chem_clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                     LogisticRegression(max_iter=2000, class_weight="balanced"))
            chem_clf.fit(chem[tr], ytr)
            pc = chem_clf.predict_proba(chem[te])[:, 1]
            ea, elo, ehi = boot_auc(yte, pe)
            ca, clo, chi = boot_auc(yte, pc)
            results.append({
                "target": target, "held_out_region": R,
                "n_test": int(te.sum()), "n_pos": int(yte.sum()),
                "encoder_auc": ea, "encoder_ci95": [elo, ehi],
                "chem_auc": ca, "chem_ci95": [clo, chi],
                "encoder_minus_chem": ea - ca,
                "encoder_beats": bool(elo > chi),  # non-overlapping & higher
            })

    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump({"encoder_size": args.encoder_size, "ckpt": args.ckpt, "results": results},
              open(args.out, "w"), indent=2)

    print(f"\n{'target':5s} {'region':16s} {'n_pos/n':>9s} {'encoder':>8s} {'chem':>8s} {'Δ':>7s}  verdict")
    print("-" * 72)
    wins = 0
    for r in sorted(results, key=lambda x: (x["target"], -x["encoder_minus_chem"])):
        v = "ENCODER WINS" if r["encoder_beats"] else ("chem wins" if r["chem_ci95"][0] > r["encoder_ci95"][1] else "tie")
        wins += r["encoder_beats"]
        print(f"{r['target']:5s} {r['held_out_region']:16s} "
              f"{str(r['n_pos'])+'/'+str(r['n_test']):>9s} "
              f"{r['encoder_auc']:8.3f} {r['chem_auc']:8.3f} {r['encoder_minus_chem']:+7.3f}  {v}")
    print("-" * 72)
    print(f"Encoder beats chem-only (non-overlapping CI) in {wins}/{len(results)} cells.")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
