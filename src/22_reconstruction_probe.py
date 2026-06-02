"""
GENESIS Paper 5 — Step 22: MGM reconstruction-error probe (foundation-model-native).

Uses the pretraining objective itself as a scientific instrument. For each
chemistry parameter P, we mask P ALONE in each sample and ask how well the
encoder reconstructs its (z-scored) value from the remaining chemistry +
environmental context. This yields:

  1. A geochemical LEARNABILITY ranking: which species are determined by their
     co-occurring chemistry (low reconstruction error) versus quasi-independent
     / anthropogenic (high error). A property of the corpus, read from the model.

  2. A model-internal OUT-OF-DISTRIBUTION signal: comparing per-parameter
     reconstruction error on the corpus test split versus the strongly reducing
     Bangladesh held-out set. Parameters the model reconstructs much worse in
     Bangladesh are exactly those whose geochemical context differs — an
     independent, FM-native corroboration of the redox-regime mismatch
     (complementary to the input-space energy distance and the attention map).

This is distinct from the companion supervised-classifier work: it concerns what
the self-supervised pretraining objective did and did not internalize.

Usage:
  python src/22_reconstruction_probe.py --ckpt base_full/checkpoints/base_stage2/best.pt --size base
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model.genesis_encoder import (
    GENESISForMGM, GENESIS_PARAMS, PARAM_TO_ID, MASK_TOKEN,
)
from model.temporal_dataset import TemporalPairDataset
from model.dataset import load_stats

NORM = "data/processed/normalization_stats.json"
MASK_ID = PARAM_TO_ID[MASK_TOKEN]


def tokenized_currents(t0, t1, gap, meta, aux, ns, n):
    """Return list of (param_ids, values, padding_mask) for the current (t0) state."""
    ds = TemporalPairDataset(t0_values=t0, t1_values=t1, gap_years=gap,
                             norm_stats=ns, meta_df=meta, aux_values=aux)
    idx = np.random.default_rng(0).choice(len(ds), min(n, len(ds)), replace=False)
    items = []
    for i in idx:
        cur = ds[int(i)]["current"]
        items.append((cur["param_ids"], cur["values"], cur["padding_mask"]))
    return items


@torch.no_grad()
def per_param_recon(model, items, device, bs=128):
    """For each parameter, mask it alone and collect (pred, true) z-scored pairs."""
    sse = {p: 0.0 for p in GENESIS_PARAMS}
    cnt = {p: 0 for p in GENESIS_PARAMS}
    sum_y = {p: 0.0 for p in GENESIS_PARAMS}
    sum_y2 = {p: 0.0 for p in GENESIS_PARAMS}
    model.eval()
    for pname in GENESIS_PARAMS:
        pid = PARAM_TO_ID[pname]
        batch_pid, batch_val, batch_pad, batch_true = [], [], [], []
        for param_ids, values, pad in items:
            pos = (param_ids == pid)
            if not pos.any():
                continue
            orig = param_ids.clone()
            true_val = values[pos][0].item()
            mids = param_ids.clone()
            mvals = values.clone()
            mids[pos] = MASK_ID
            mvals[pos] = 0.0
            mpos = torch.zeros_like(param_ids, dtype=torch.bool)
            mpos[pos] = True
            batch_pid.append((mids, mvals, pad, orig, mpos, true_val))
        # run in mini-batches
        for k in range(0, len(batch_pid), bs):
            chunk = batch_pid[k:k + bs]
            mids = torch.stack([c[0] for c in chunk]).to(device)
            mvals = torch.stack([c[1] for c in chunk]).to(device)
            pads = torch.stack([c[2] for c in chunk]).to(device)
            origs = torch.stack([c[3] for c in chunk]).to(device)
            mposs = torch.stack([c[4] for c in chunk]).to(device)
            preds = model(mids, mvals, pads, origs, mposs)
            if pname not in preds or preds[pname] is None or len(preds[pname]) == 0:
                continue
            pv = preds[pname].detach().cpu().numpy().ravel()
            tv = np.array([c[5] for c in chunk[:len(pv)]])
            e = pv - tv
            sse[pname] += float((e ** 2).sum())
            cnt[pname] += len(e)
            sum_y[pname] += float(tv.sum())
            sum_y2[pname] += float((tv ** 2).sum())
    out = {}
    for p in GENESIS_PARAMS:
        if cnt[p] < 20:
            continue
        mse = sse[p] / cnt[p]
        var = sum_y2[p] / cnt[p] - (sum_y[p] / cnt[p]) ** 2
        r2 = 1 - mse / var if var > 1e-9 else float("nan")
        out[p] = {"n": cnt[p], "mse_zscore": mse, "r2": r2}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="base_full/checkpoints/base_stage2/best.pt")
    ap.add_argument("--size", default="base")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", default="results/reconstruction_probe.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ns = load_stats(NORM)
    model = GENESISForMGM(args.size)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    model.to(device)

    import pandas as pd
    # corpus sample
    gp = torch.load("data/processed/genesis_temporal_pairs.pt", map_location="cpu", weights_only=False)
    ga = torch.load("data/processed/genesis_temporal_pairs_aux.pt", map_location="cpu", weights_only=False)
    ga = ga["aux"] if isinstance(ga, dict) else ga
    gm = pd.read_parquet("data/processed/genesis_temporal_pairs_meta.parquet")
    corpus = tokenized_currents(gp["t0"].numpy(), gp["t1"].numpy(), gp["gap_years"].numpy(),
                                gm, ga.numpy(), ns, args.n)
    # Bangladesh
    bp = torch.load("data/processed/genesis_held_out_bd_pairs.pt", map_location="cpu", weights_only=False)
    ba = torch.load("data/processed/genesis_held_out_bd_pairs_aux.pt", map_location="cpu", weights_only=False)
    ba = ba["aux"] if isinstance(ba, dict) else ba
    bm = pd.read_parquet("data/processed/genesis_held_out_bd_pairs_meta.parquet")
    bgap = np.full(len(bp["t0"]), float(bp["gap_years"]))
    bd = tokenized_currents(bp["t0"].numpy(), bp["t1"].numpy(), bgap, bm, ba.numpy(), ns, len(bp["t0"]))

    print(f"Probing reconstruction: corpus n={len(corpus)}, Bangladesh n={len(bd)} ...")
    rc = per_param_recon(model, corpus, device)
    rb = per_param_recon(model, bd, device)

    rows = []
    for p in GENESIS_PARAMS:
        if p in rc:
            rows.append({"param": p, "corpus_r2": rc[p]["r2"], "corpus_mse": rc[p]["mse_zscore"],
                         "bd_r2": rb.get(p, {}).get("r2"), "bd_mse": rb.get(p, {}).get("mse_zscore")})
    json.dump({"corpus": rc, "bangladesh": rb}, open(args.out, "w"), indent=2)

    print("\n=== Geochemical learnability (corpus): how well is each species "
          "reconstructed from its context? (R2, higher=more determined) ===")
    for r in sorted(rows, key=lambda x: -(x["corpus_r2"] if x["corpus_r2"] is not None else -9)):
        bd_r2 = r["bd_r2"]
        dd = (f"  BD R2={bd_r2:+.2f}  Δ(corpus−BD)={r['corpus_r2']-bd_r2:+.2f}"
              if bd_r2 is not None else "  BD: n/a")
        print(f"  {r['param']:5s} corpus R2={r['corpus_r2']:+.2f}{dd}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
