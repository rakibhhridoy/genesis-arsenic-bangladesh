"""
GENESIS Paper 5 — Step 29: END-TO-END FINE-TUNED encoder on the LORO benchmark.

src/19 scored the encoder frozen (linear probe); src/28 showed that even with a
nonlinear head and matched inputs the FROZEN representation does not beat Random
Forest (best frozen-encoder mean 0.747 vs best tree 0.773). Fine-tuning is the one
remaining lever: unfreeze the encoder and let pretraining reshape to each target.

This is the strongest possible use of the foundation model. For each LORO cell we
fine-tune the full encoder + head on the out-of-region training data and evaluate
zero-shot on the held-out region, against the same RF/XGBoost baselines (read from
results/region_transfer_tree_baselines.json) on identical splits.

DESIGNED FOR RUNPOD (CUDA). Heavy: ~47 cells x backprop on up to ~55k samples.
Writes results INCREMENTALLY to --out (resumable: re-run skips finished cells), so
a disconnect never loses completed work. Use --cells for a cheap pilot first.

Usage (RunPod A40):
  # cheap pilot — the decisive cells (~6, ~30 min): does fine-tuning beat RF on U/USA?
  python src/29_loro_finetune.py --encoder_size large \
      --ckpt checkpoints/large_stage2/best.pt \
      --cells "U/USA,PO4/GEMStat:Italy,PO4/GEMStat:India,As/Bangladesh,Fe/Bangladesh,U/GEMStat:Canada"
  # full benchmark (all 47 cells, ~3-4 h on A40):
  python src/29_loro_finetune.py --encoder_size large --ckpt checkpoints/large_stage2/best.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from model.genesis_encoder import GENESISForMGM, GENESIS_PARAMS
from model.dataset import load_stats

_c10 = __import__("10_classify_exceedance")
ExceedanceClassifier = _c10.ExceedanceClassifier
LabeledTemporal = _c10.LabeledTemporal
labeled_collate_fn = _c10.labeled_collate_fn
TemporalPairDataset = _c10.TemporalPairDataset

_m19 = __import__("19_region_transfer_encoder")
region_of, boot_auc = _m19.region_of, _m19.boot_auc
TARGETS = _m19.TARGETS
MIN_POS, MIN_NEG, MIN_TRAIN = _m19.MIN_POS, _m19.MIN_NEG, _m19.MIN_TRAIN
NORM = "data/processed/normalization_stats.json"


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ps, ys = [], []
    for b in loader:
        cur = {k: v.to(device) for k, v in b["current"].items()}
        logits = model(cur)
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(b["label"].numpy())
    return np.concatenate(ys), np.concatenate(ps)


def finetune_cell(base_ds, tr_idx, va_idx, te_idx, ytr, encoder_size, ckpt,
                  device, epochs, bs, lr, patience):
    enc = GENESISForMGM(encoder_size)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    enc.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    model = ExceedanceClassifier(enc, freeze=False, linear_probe=False).to(device)

    tr = DataLoader(Subset(base_ds, tr_idx), batch_size=bs, shuffle=True,
                    collate_fn=labeled_collate_fn, num_workers=2, pin_memory=True)
    va = DataLoader(Subset(base_ds, va_idx), batch_size=bs, shuffle=False,
                    collate_fn=labeled_collate_fn, num_workers=2)
    te = DataLoader(Subset(base_ds, te_idx), batch_size=bs, shuffle=False,
                    collate_fn=labeled_collate_fn, num_workers=2)

    pos = float(ytr.sum()); neg = float(len(ytr) - ytr.sum())
    pw = torch.tensor(neg / max(pos, 1.0), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)

    best_va, best_state, wait = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for b in tr:
            cur = {k: v.to(device) for k, v in b["current"].items()}
            y = b["label"].to(device)
            opt.zero_grad()
            if amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    loss = F.binary_cross_entropy_with_logits(model(cur), y, pos_weight=pw)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss = F.binary_cross_entropy_with_logits(model(cur), y, pos_weight=pw)
                loss.backward(); opt.step()
        yv, pv = predict(model, va, device)
        va_auc = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else float("nan")
        if va_auc == va_auc and va_auc > best_va:
            best_va, wait = va_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    yte, pte = predict(model, te, device)
    return yte, pte, best_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_size", default="large")
    ap.add_argument("--ckpt", default="checkpoints/large_stage2/best.pt")
    ap.add_argument("--out", default="results/loro_finetune_large.json")
    ap.add_argument("--cells", default="", help="comma list 'target/region' to run a subset")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)  # gentle: pretrained encoder
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--val_frac", type=float, default=0.15)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"device={device}  encoder={args.encoder_size}  ckpt={args.ckpt}")
    if device.type == "mps":
        print("WARNING: fine-tuning 47 cells on MPS/8GB is freeze-prone — RunPod/CUDA intended.")

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
    t0 = np.concatenate([g_t0, b_t0], 0); t1 = np.concatenate([g_t1, b_t1], 0)
    gap = np.concatenate([g_gap, b_gap], 0)
    aux = np.concatenate([ga, ba], 0)
    meta = pd.concat([gm, bm], ignore_index=True)
    region = np.array([region_of(s, c) for s, c in zip(gm["source"], gm["country"])]
                      + ["Bangladesh"] * len(b_t0))

    base_ds = TemporalPairDataset(t0_values=t0, t1_values=t1, gap_years=gap,
                                  norm_stats=ns, meta_df=meta, aux_values=aux)

    counts = pd.Series(region).value_counts()
    regions = [r for r in counts.index if counts[r] >= MIN_TRAIN or r == "Bangladesh"]
    want = set(c.strip() for c in args.cells.split(",") if c.strip()) if args.cells else None

    # tree baselines for head-to-head (computed encoder-independent in src/27)
    tb = {}
    tbp = Path("results/region_transfer_tree_baselines.json")
    if tbp.exists():
        for r in json.load(open(tbp))["rows"]:
            tb[(r["target"], r["held_out_region"])] = r

    # resume: load any finished cells
    out = json.load(open(args.out)) if Path(args.out).exists() else {"rows": []}
    done = {(r["target"], r["held_out_region"]) for r in out["rows"]}

    rng = np.random.default_rng(0)
    for target, thr in TARGETS.items():
        ti = GENESIS_PARAMS.index(target)
        tv = t1[:, ti]; valid = ~np.isnan(tv); y = (tv > thr).astype(int)
        # carry per-target labels so labeled_collate_fn finds b['label'];
        # invalid (NaN-target) rows are never indexed (we mask on `valid`).
        lab_ds = LabeledTemporal(base_ds, y.astype(np.float32))
        for R in regions:
            key = f"{target}/{R}"
            if want is not None and key not in want:
                continue
            if (target, R) in done:
                print(f"skip {key} (done)"); continue
            inR = region == R
            te_mask, tr_mask = valid & inR, valid & ~inR
            yte_all, ytr_all = y[te_mask], y[tr_mask]
            if yte_all.sum() < MIN_POS or (len(yte_all) - yte_all.sum()) < MIN_NEG:
                continue
            if ytr_all.sum() < MIN_POS or tr_mask.sum() < MIN_TRAIN:
                continue
            tr_idx_all = np.where(tr_mask)[0]
            te_idx = np.where(te_mask)[0]
            # carve a stratified val split from train for early stopping
            perm = rng.permutation(len(tr_idx_all))
            n_val = max(int(args.val_frac * len(perm)), 50)
            va_idx = tr_idx_all[perm[:n_val]]; tr_idx = tr_idx_all[perm[n_val:]]
            ytr = y[tr_idx]

            t_start = time.time()
            yte, pte, best_va = finetune_cell(
                lab_ds, tr_idx, va_idx, te_idx, ytr,
                args.encoder_size, args.ckpt, device,
                args.epochs, args.batch_size, args.lr, args.patience)
            ft_auc, lo, hi = boot_auc(yte, pte)
            b = tb.get((target, R), {})
            row = {"target": target, "held_out_region": R,
                   "n_test": int(te_mask.sum()), "n_pos": int(yte_all.sum()),
                   "finetune_auc": ft_auc, "finetune_ci95": [lo, hi],
                   "val_auc": float(best_va),
                   "rf_auc": b.get("rf_auc"), "xgb_auc": b.get("xgb_auc"),
                   "elapsed_sec": round(time.time() - t_start, 1)}
            if b.get("rf_ci95"):
                row["finetune_beats_rf"] = bool(lo > b["rf_ci95"][1])
            out["rows"].append(row)
            json.dump(out, open(args.out, "w"), indent=2)  # incremental write
            print(f"{key:24s} n={row['n_test']:5d}  ft={ft_auc:.3f} "
                  f"rf={b.get('rf_auc', float('nan')):.3f} xgb={b.get('xgb_auc', float('nan')):.3f} "
                  f"val={best_va:.3f}  {row['elapsed_sec']}s "
                  f"{'FT BEATS RF' if row.get('finetune_beats_rf') else ''}")

    rows = out["rows"]
    if rows:
        ft = np.array([r["finetune_auc"] for r in rows])
        rf = np.array([r["rf_auc"] for r in rows if r["rf_auc"] is not None])
        beats = sum(1 for r in rows if r.get("finetune_beats_rf"))
        print(f"\n=== {len(rows)} cells | mean fine-tune AUC {ft.mean():.3f} "
              f"vs mean RF {rf.mean():.3f} | fine-tune significantly beats RF {beats}/{len(rows)} ===")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
