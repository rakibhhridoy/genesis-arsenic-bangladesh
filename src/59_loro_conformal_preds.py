#!/usr/bin/env python3
"""Per-sample exceedance probabilities on the LORO cells, for conformal coverage (E3).

src/27 and src/29 saved only cell AUCs. Conformal prediction needs per-sample scores on
three disjoint sets, so each cell is re-split. Out-of-region rows are partitioned once,
deterministically per (target, region, seed), into
  train 70% | val 10% (early stopping, fine-tune only) | cal 10% | test_in 10%
and the held-out region is test_out. Every method sees the SAME partition, so the trees
and the fine-tuned encoder are compared on identical calibration and test rows.

  rf, xgb   - raw t0 chemistry, recipes of src/27 (CPU, minutes)
  finetune  - full encoder + head, recipe of src/29 (CUDA; RunPod)
  logreg    - logistic regression on raw t0 chemistry (src/19 chem-only baseline)
  rf_aux    - random forest on the 18 auxiliary rasters only (a Podgorski-style hazard map)
  knn_space - mean label of the 25 nearest training sites by great-circle distance (pure
              spatial interpolation; under LORO the neighbours lie outside the region)
  frozen    - logistic probe on the frozen Base encoder's latent (src/19 recipe), embeddings
              cached at results/loro_emb_base.npz (CPU, minutes)
  tabpfn    - TabPFN v2 (Hollmann et al. 2025), default 4-member ensemble, raw t0 chemistry
              with native NaN, in-context on a stratified 10,000-row subsample of train (its
              design limit). Needs tabpfn==2.2.1 and a GPU (CPU is ~15 h per seed).

One .npz per (method, seed, cell) under results/conformal_preds/, so the run resumes
after a disconnect and the folder can be sent back from the pod as is.

  python src/59_loro_conformal_preds.py --method rf --seeds 42
  python src/59_loro_conformal_preds.py --method finetune --encoder_size base \
      --ckpt checkpoints/base_stage2/best.pt --seeds 42,43,44
"""
import argparse
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model.genesis_encoder import GENESIS_PARAMS

_m19 = __import__("19_region_transfer_encoder")
region_of, TARGETS = _m19.region_of, _m19.TARGETS
MIN_POS, MIN_NEG, MIN_TRAIN = _m19.MIN_POS, _m19.MIN_NEG, _m19.MIN_TRAIN
OUT = Path("results/conformal_preds")


def load():
    gp = torch.load("data/processed/genesis_temporal_pairs.pt", map_location="cpu", weights_only=False)
    gm = pd.read_parquet("data/processed/genesis_temporal_pairs_meta.parquet")
    bp = torch.load("data/processed/genesis_held_out_bd_pairs.pt", map_location="cpu", weights_only=False)
    t0 = np.concatenate([gp["t0"].numpy(), bp["t0"].numpy()], 0)
    t1 = np.concatenate([gp["t1"].numpy(), bp["t1"].numpy()], 0)
    region = np.array([region_of(s, c) for s, c in zip(gm["source"], gm["country"])]
                      + ["Bangladesh"] * len(bp["t0"]))
    return gp, gm, bp, t0, t1, region


def cells(t1, region):
    counts = pd.Series(region).value_counts()
    regions = [r for r in counts.index if counts[r] >= MIN_TRAIN or r == "Bangladesh"]
    for target, thr in TARGETS.items():
        ti = GENESIS_PARAMS.index(target)
        valid = ~np.isnan(t1[:, ti]); y = (t1[:, ti] > thr).astype(int)
        for R in regions:
            inR = region == R
            te, tr = valid & inR, valid & ~inR
            if y[te].sum() < MIN_POS or (te.sum() - y[te].sum()) < MIN_NEG:
                continue
            if y[tr].sum() < MIN_POS or tr.sum() < MIN_TRAIN:
                continue
            yield target, R, y, np.where(tr)[0], np.where(te)[0]


def frozen_embeddings(gp, gm, bp, size, ckpt):
    f = Path(f"results/loro_emb_{size}.npz")
    if f.exists():
        return np.load(f)["emb"]
    _m19 = __import__("19_region_transfer_encoder")
    from model.dataset import load_stats
    from model.genesis_encoder import GENESISForMGM
    ga = torch.load("data/processed/genesis_temporal_pairs_aux.pt", map_location="cpu", weights_only=False)
    ba = torch.load("data/processed/genesis_held_out_bd_pairs_aux.pt", map_location="cpu", weights_only=False)
    ns = load_stats(_m19.NORM)
    enc = GENESISForMGM(size)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    print(enc.load_state_dict(ck.get("model_state_dict", ck), strict=False), flush=True)
    torch.set_num_threads(8)
    bm = pd.read_parquet("data/processed/genesis_held_out_bd_pairs_meta.parquet")
    out = []
    for p, m, aux, gap in [(gp, gm, ga, gp["gap_years"].numpy()),
                           (bp, bm, ba, np.full(len(bp["t0"]), float(bp["gap_years"])))]:
        ds = _m19.TemporalPairDataset(t0_values=p["t0"].numpy(), t1_values=p["t1"].numpy(), gap_years=gap,
                                      norm_stats=ns, meta_df=m,
                                      aux_values=(aux["aux"] if isinstance(aux, dict) else aux).numpy())
        out.append(_m19.extract_embeddings(enc, ds, "cpu", bs=256))
    emb = np.concatenate(out, 0)
    np.savez_compressed(f, emb=emb)
    return emb


def partition(idx, target, R, seed):
    key = zlib.crc32(f"{target}/{R}/{seed}".encode())
    p = idx[np.random.default_rng(key).permutation(len(idx))]
    a, b, c = int(.7 * len(p)), int(.8 * len(p)), int(.9 * len(p))
    return p[:a], p[a:b], p[b:c], p[c:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["rf", "xgb", "logreg", "rf_aux", "knn_space", "frozen", "finetune", "tabpfn"])
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--encoder_size", default="base")
    ap.add_argument("--ckpt", default="checkpoints/base_stage2/best.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--context", type=int, default=10000)
    ap.add_argument("--cells", default="", help="comma list 'target/region' to run a subset")
    ap.add_argument("--out_root", default=str(OUT))
    a = ap.parse_args()
    want = {c.strip() for c in a.cells.split(",") if c.strip()}

    gp, gm, bp, t0, t1, region = load()
    if a.method == "finetune":
        _m29 = __import__("29_loro_finetune")
        from model.dataset import load_stats
        ga = torch.load("data/processed/genesis_temporal_pairs_aux.pt", map_location="cpu", weights_only=False)
        ba = torch.load("data/processed/genesis_held_out_bd_pairs_aux.pt", map_location="cpu", weights_only=False)
        aux = np.concatenate([(ga["aux"] if isinstance(ga, dict) else ga).numpy(),
                              (ba["aux"] if isinstance(ba, dict) else ba).numpy()], 0)
        gap = np.concatenate([gp["gap_years"].numpy(), np.full(len(bp["t0"]), float(bp["gap_years"]))])
        meta = pd.concat([gm, pd.read_parquet("data/processed/genesis_held_out_bd_pairs_meta.parquet")],
                         ignore_index=True)
        base_ds = _m29.TemporalPairDataset(t0_values=t0, t1_values=t1, gap_years=gap,
                                           norm_stats=load_stats(_m29.NORM), meta_df=meta, aux_values=aux)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device={device} encoder={a.encoder_size} ckpt={a.ckpt}", flush=True)
    elif a.method == "tabpfn":
        from tabpfn import TabPFNClassifier
        torch.set_num_threads(8)
    elif a.method == "knn_space":
        from sklearn.neighbors import BallTree
        ll = np.radians(np.r_[gm[["lat", "lon"]].values,
                              pd.read_parquet("data/processed/genesis_held_out_bd_pairs_meta.parquet")[["lat", "lon"]].values])
        for R in np.unique(region):                 # one Bangladesh pair lacks coordinates
            m = (region == R) & np.isnan(ll).any(1)
            if m.any():
                ll[m] = np.nanmedian(ll[region == R], 0)
    else:
        _m27 = __import__("27_loro_tree_baselines")
        fit = {"rf": _m27.fit_predict_rf, "xgb": _m27.fit_predict_xgb, "logreg": _m27.fit_predict_chem,
               "rf_aux": _m27.fit_predict_rf, "frozen": _m27.fit_predict_chem}[a.method]
        X = t0
        if a.method == "rf_aux":
            ga = torch.load("data/processed/genesis_temporal_pairs_aux.pt", map_location="cpu", weights_only=False)
            ba = torch.load("data/processed/genesis_held_out_bd_pairs_aux.pt", map_location="cpu", weights_only=False)
            X = np.concatenate([(ga["aux"] if isinstance(ga, dict) else ga).numpy(),
                                (ba["aux"] if isinstance(ba, dict) else ba).numpy()], 0)
        elif a.method == "frozen":
            X = frozen_embeddings(gp, gm, bp, a.encoder_size, a.ckpt)

    for seed in [int(s) for s in a.seeds.split(",")]:
        d = Path(a.out_root) / f"{a.method}{'_' + a.encoder_size if a.method in ('finetune', 'frozen') else ''}_s{seed}"
        d.mkdir(parents=True, exist_ok=True)
        for target, R, y, tr_all, te in cells(t1, region):
            if want and f"{target}/{R}" not in want:
                continue
            f = d / f"{target}__{R.replace(':', '-').replace('/', '-')}.npz"
            if f.exists():
                continue
            tr, va, cal, tin = partition(tr_all, target, R, seed)
            score = np.concatenate([cal, tin, te])
            t_start = time.time()
            if a.method == "finetune":
                _m29.set_seed(seed * 1000 + GENESIS_PARAMS.index(target))
                lab = _m29.LabeledTemporal(base_ds, y.astype(np.float32))
                _, p, _ = _m29.finetune_cell(lab, tr, va, score, y[tr], a.encoder_size, a.ckpt, device,
                                             a.epochs, a.batch_size, a.lr, a.patience)
            elif a.method == "tabpfn":
                ctx = tr
                if len(tr) > a.context:
                    r = np.random.default_rng(seed)
                    pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
                    k = max(int(round(a.context * len(pos) / len(tr))), min(len(pos), 50))
                    ctx = np.concatenate([r.choice(pos, min(k, len(pos)), replace=False),
                                          r.choice(neg, a.context - min(k, len(pos)), replace=False)])
                clf = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu", random_state=seed).fit(t0[ctx], y[ctx])
                p = np.concatenate([clf.predict_proba(t0[score[i:i + 5000]])[:, 1]
                                    for i in range(0, len(score), 5000)])
            elif a.method == "knn_space":
                nb = BallTree(ll[tr], metric="haversine").query(ll[score], k=25, return_distance=False)
                p = y[tr][nb].mean(1)
            else:
                p = fit(X[tr], y[tr], X[score])
            n1, n2 = len(cal), len(cal) + len(tin)
            np.savez_compressed(f, target=target, region=R, seed=seed,
                                p_cal=p[:n1], y_cal=y[cal], p_in=p[n1:n2], y_in=y[tin],
                                p_out=p[n2:], y_out=y[te], idx_out=te)
            print(f"{a.method} s{seed} {target:4s} {R:20s} cal={len(cal):5d} in={len(tin):5d} "
                  f"out={len(te):5d} {time.time() - t_start:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
