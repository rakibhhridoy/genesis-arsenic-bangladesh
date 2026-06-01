#!/usr/bin/env python3
"""
GENESIS encoder → WHO-threshold exceedance classifier (As default).

Diagnostic answer to: did the encoder learn anything that the regression head
couldn't extract? Loads the pretrained MGM encoder, optionally freezes it,
trains a small MLP head (or a linear probe) on top of the [CLS] embedding to
predict whether the *future* concentration of a target parameter (default As,
10 µg/L WHO) will exceed a safety threshold given the *current* chemistry.

Supports the experimental matrix used by run_classify_sweep.sh:
  - frozen encoder + MLP head      (default)
  - --unfreeze_encoder             (fine-tune encoder end-to-end)
  - --no_pretrain                  (random encoder, train all)
  - --linear_probe                 (single Linear, no MLP)
  - --n_seeds N                    (repeat with N seeds; aggregate metrics)

Outputs:
  results/classify_<param>_<tag>.json   (or default tag from args)
  checkpoints/classify_<param>_<tag>/seed<S>/best.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent))
from model.genesis_encoder import (
    GENESISForMGM, GENESIS_PARAMS, MODEL_CONFIGS,
)
from model.temporal_dataset import (
    TemporalPairDataset, _wide_pairs_to_arrays, temporal_collate_fn,
)


WHO_THRESHOLDS = {
    'As': 10.0,    # µg/L
    'F':  1.5,     # mg/L
    'NO3': 50.0,   # mg/L
    'PO4': 0.1,    # eutrophication, not health-based
    'U':  30.0,    # µg/L
}


# ---------------------------------------------------------------------------
# Dataset wrapper that carries binary labels
# ---------------------------------------------------------------------------

class LabeledTemporal(Dataset):
    def __init__(self, base_ds: TemporalPairDataset, labels: np.ndarray):
        self.base = base_ds
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        s = self.base[idx]
        s['label'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return s


def labeled_collate_fn(batch):
    out = temporal_collate_fn(batch)
    out['label'] = torch.stack([b['label'] for b in batch])
    return out


def build_labeled_dataset(t0, t1, gap, df, aux, norm_stats,
                          target_param='As', threshold=10.0, source_tag=''):
    """Filter to samples with non-NaN raw t1 target, build LabeledTemporal."""
    p_idx = GENESIS_PARAMS.index(target_param)
    raw_t1 = t1[:, p_idx]
    valid = ~np.isnan(raw_t1)
    n_total, n_valid = len(raw_t1), int(valid.sum())
    labels = (raw_t1[valid] > threshold).astype(np.float32)
    n_pos = int(labels.sum())
    pos_rate = n_pos / max(1, n_valid)
    print(f'  [{source_tag}] {target_param}>{threshold}: '
          f'{n_valid}/{n_total} valid, {n_pos} positive ({pos_rate:.3f})')

    df_f = df[valid].reset_index(drop=True) if df is not None else None
    aux_f = aux[valid] if aux is not None else None

    base = TemporalPairDataset(
        t0_values=t0[valid], t1_values=t1[valid], gap_years=gap[valid],
        norm_stats=norm_stats, meta_df=df_f, aux_values=aux_f,
    )
    return LabeledTemporal(base, labels), labels


# ---------------------------------------------------------------------------
# Classifier model
# ---------------------------------------------------------------------------

class ExceedanceClassifier(nn.Module):
    def __init__(self, encoder_module: GENESISForMGM, freeze=True,
                 linear_probe=False, hidden=64, dropout=0.2):
        super().__init__()
        self.encoder = encoder_module
        d_model = MODEL_CONFIGS[encoder_module.config_name]['d_model']
        if linear_probe:
            self.head = nn.Linear(d_model, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        self.freeze = freeze
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def forward(self, current):
        if self.freeze:
            with torch.no_grad():
                latent = self.encoder.get_latent(
                    current['param_ids'], current['values'],
                    current['padding_mask'],
                )
        else:
            latent = self.encoder.get_latent(
                current['param_ids'], current['values'],
                current['padding_mask'],
            )
        return self.head(latent).squeeze(-1)


# ---------------------------------------------------------------------------
# Eval + bootstrap CIs
# ---------------------------------------------------------------------------

def _safe_auc(y, p):
    if len(y) < 2 or y.sum() == 0 or y.sum() == len(y):
        return float('nan')
    return float(roc_auc_score(y, p))


def _safe_ap(y, p):
    if y.sum() == 0:
        return float('nan')
    return float(average_precision_score(y, p))


def bootstrap_metrics(labels, probs, n_boot=1000, seed=0):
    """Stratified bootstrap of (AUC, AP, F1_best). Returns dict of CI tuples."""
    if len(labels) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return {'auc_ci95': [float('nan'), float('nan')],
                'ap_ci95':  [float('nan'), float('nan')],
                'f1_best_ci95': [float('nan'), float('nan')]}
    rng = np.random.default_rng(seed)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    aucs, aps, f1bs = [], [], []
    ts = np.linspace(0.05, 0.95, 91)
    for _ in range(n_boot):
        ip = rng.choice(pos, size=len(pos), replace=True)
        in_ = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([ip, in_])
        y = labels[idx]; p = probs[idx]
        aucs.append(_safe_auc(y, p))
        aps.append(_safe_ap(y, p))
        f1s_t = [f1_score(y, p >= t, zero_division=0) for t in ts]
        f1bs.append(float(np.max(f1s_t)))
    return {
        'auc_ci95':     [float(np.nanpercentile(aucs, 2.5)),
                         float(np.nanpercentile(aucs, 97.5))],
        'ap_ci95':      [float(np.nanpercentile(aps, 2.5)),
                         float(np.nanpercentile(aps, 97.5))],
        'f1_best_ci95': [float(np.nanpercentile(f1bs, 2.5)),
                         float(np.nanpercentile(f1bs, 97.5))],
    }


def evaluate(model, loader, device, n_boot=0, seed=0):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            current = {k: v.to(device) for k, v in batch['current'].items()}
            logits.append(model(current).cpu().numpy())
            labels.append(batch['label'].numpy())
    logits = np.concatenate(logits) if logits else np.zeros(0)
    labels = np.concatenate(labels) if labels else np.zeros(0)
    probs = 1.0 / (1.0 + np.exp(-logits))

    out = {
        'n_samples': int(len(labels)),
        'n_positive': int(labels.sum()),
        'positive_rate': float(labels.mean()) if len(labels) else float('nan'),
    }
    if 0 < labels.sum() < len(labels):
        out['auc']     = _safe_auc(labels, probs)
        out['ap']      = _safe_ap(labels, probs)
        out['brier']   = float(brier_score_loss(labels, probs))
        out['f1_at_0.5'] = float(f1_score(labels, probs >= 0.5, zero_division=0))
        ts = np.linspace(0.05, 0.95, 91)
        f1s = [f1_score(labels, probs >= t, zero_division=0) for t in ts]
        best_i = int(np.argmax(f1s))
        out['f1_best'] = float(f1s[best_i])
        out['f1_best_threshold'] = float(ts[best_i])
        tn, fp, fn, tp = confusion_matrix(labels, probs >= ts[best_i]).ravel()
        out['confusion_at_best'] = {'tn': int(tn), 'fp': int(fp),
                                    'fn': int(fn), 'tp': int(tp)}
        if n_boot > 0:
            out.update(bootstrap_metrics(labels, probs, n_boot=n_boot, seed=seed))
    else:
        out['note'] = 'single-class split — AUC undefined'
    return out, probs, labels


# ---------------------------------------------------------------------------
# One full training run for one seed
# ---------------------------------------------------------------------------

def run_one_seed(seed, args, full_ds, full_labels, bd_ds, n_total, encoder_state,
                 device, ckpt_dir):
    print(f'\n{"="*70}\n  SEED {seed}\n{"="*70}')
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_test = max(50, int(n_total * 0.15))
    n_val = max(50, int(n_total * 0.15))
    n_train = n_total - n_val - n_test
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    test_ds  = Subset(full_ds, test_idx)
    train_pos = int(full_labels[train_idx].sum())
    val_pos = int(full_labels[val_idx].sum())
    test_pos = int(full_labels[test_idx].sum())
    print(f'  splits: train={n_train} ({train_pos}+), '
          f'val={n_val} ({val_pos}+), test={n_test} ({test_pos}+)')

    # Skip degenerate splits (rare-contaminant edge case, e.g. U).
    if train_pos == 0 or val_pos == 0 or test_pos == 0:
        print(f'  [skip] seed {seed}: degenerate split '
              f'(train_pos={train_pos}, val_pos={val_pos}, test_pos={test_pos})')
        return {
            'seed': seed,
            'best_val_auc': float('nan'),
            'global_test': {'note': 'degenerate split — skipped',
                            'n_samples': n_test, 'n_positive': test_pos},
            'bangladesh_transfer': {'note': 'skipped due to global split',
                                    'n_samples': len(bd_ds),
                                    'n_positive': int(bd_ds.labels.sum())
                                    if hasattr(bd_ds, 'labels') else 0},
            'history': [],
            'elapsed_sec': 0.0,
            'n_train_params': 0,
            'frozen_encoder': False,
            'linear_probe': bool(args.linear_probe),
            'no_pretrain':  bool(encoder_state is None),
        }

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=labeled_collate_fn, num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=labeled_collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=labeled_collate_fn, num_workers=0)
    bd_loader = DataLoader(bd_ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=labeled_collate_fn, num_workers=0)

    # Build encoder fresh per seed (so head init + encoder init can vary
    # for --no_pretrain; for pretrained runs, weights are deterministic).
    encoder = GENESISForMGM(args.encoder_size)
    if encoder_state is not None:
        encoder.load_state_dict(encoder_state)
        print('  Encoder: pretrained weights loaded')
    else:
        print('  Encoder: random init (--no_pretrain)')

    freeze = (not args.unfreeze_encoder) and (encoder_state is not None)
    # If no_pretrain, must train encoder; freezing random encoder is meaningless
    if encoder_state is None:
        freeze = False

    model = ExceedanceClassifier(
        encoder, freeze=freeze, linear_probe=args.linear_probe,
    ).to(device)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Model: linear_probe={args.linear_probe}, frozen_enc={freeze}, '
          f'trainable={n_train_params:,}')

    pos_weight = None
    if 0 < train_pos < n_train:
        pos_weight = torch.tensor((n_train - train_pos) / train_pos,
                                  dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    use_amp = args.fp16 and device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)

    seed_ckpt_dir = ckpt_dir / f'seed{seed}'
    seed_ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = -1.0
    patience_counter = 0
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n_batches = 0.0, 0
        for batch in train_loader:
            current = {k: v.to(device) for k, v in batch['current'].items()}
            label = batch['label'].to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(current)
                    loss = F.binary_cross_entropy_with_logits(
                        logits, label, pos_weight=pos_weight)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(current)
                loss = F.binary_cross_entropy_with_logits(
                    logits, label, pos_weight=pos_weight)
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(1, n_batches)

        val_metrics, _, _ = evaluate(model, val_loader, device)
        bd_metrics,  _, _ = evaluate(model, bd_loader, device)
        val_auc = val_metrics.get('auc', float('nan'))
        bd_auc  = bd_metrics.get('auc', float('nan'))
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_auc': val_auc, 'bd_auc': bd_auc})
        print(f'  Epoch {epoch:3d}/{args.epochs} '
              f'loss={train_loss:.4f}  val_AUC={val_auc:.4f}  '
              f'BD_AUC={bd_auc:.4f}')

        improved = (val_auc == val_auc) and (val_auc > best_val_auc)
        if improved:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'state_dict': model.state_dict(),
                'val_auc': val_auc, 'bd_auc': bd_auc,
                'seed': seed, 'args': vars(args),
            }, seed_ckpt_dir / 'best.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'  Early stop at epoch {epoch} '
                      f'(best val AUC {best_val_auc:.4f})')
                break

    # Final eval with best ckpt + bootstrap CIs
    best = torch.load(seed_ckpt_dir / 'best.pt', map_location=device,
                      weights_only=False)
    model.load_state_dict(best['state_dict'])
    test_metrics, _, _ = evaluate(model, test_loader, device,
                                  n_boot=args.bootstrap_n, seed=seed)
    bd_metrics,   _, _ = evaluate(model, bd_loader, device,
                                  n_boot=args.bootstrap_n, seed=seed)
    elapsed = time.time() - t0

    return {
        'seed': seed,
        'best_val_auc': float(best_val_auc),
        'global_test': test_metrics,
        'bangladesh_transfer': bd_metrics,
        'history': history,
        'elapsed_sec': float(elapsed),
        'n_train_params': int(n_train_params),
        'frozen_encoder': bool(freeze),
        'linear_probe': bool(args.linear_probe),
        'no_pretrain':  bool(encoder_state is None),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def aggregate_seeds(per_seed_results, key='global_test'):
    """Mean ± std across seeds for AUC/AP/F1_best. Drops None and NaN."""
    def _good(v):
        return v is not None and v == v
    aucs = [r[key].get('auc') for r in per_seed_results
            if _good(r[key].get('auc'))]
    aps  = [r[key].get('ap')  for r in per_seed_results
            if _good(r[key].get('ap'))]
    f1s  = [r[key].get('f1_best') for r in per_seed_results
            if _good(r[key].get('f1_best'))]
    def _ms(v):
        if not v:
            return [float('nan'), float('nan')]
        return [float(np.mean(v)), float(np.std(v))]
    return {
        'n_seeds_used': len(aucs),
        'auc_mean_std':  _ms(aucs),
        'ap_mean_std':   _ms(aps),
        'f1_best_mean_std': _ms(f1s),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_size', default='base',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--encoder_ckpt', default=None,
                        help='Required unless --no_pretrain')
    parser.add_argument('--target_param', default='As')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--unfreeze_encoder', action='store_true')
    parser.add_argument('--no_pretrain', action='store_true',
                        help='Random encoder init; ignores --encoder_ckpt')
    parser.add_argument('--linear_probe', action='store_true',
                        help='Single Linear head (no MLP)')
    parser.add_argument('--n_seeds', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42,
                        help='Base seed (seeds = [seed, seed+1, ..., seed+n_seeds-1])')
    parser.add_argument('--bootstrap_n', type=int, default=1000)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--out_tag', default=None,
                        help='Tag for output filenames (default: derived from flags)')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    target = args.target_param
    threshold = (args.threshold if args.threshold is not None
                 else WHO_THRESHOLDS.get(target, 10.0))

    if args.out_tag is None:
        bits = []
        bits.append('linprobe' if args.linear_probe else 'mlp')
        if args.no_pretrain:
            bits.append('nopretrain')
        elif args.unfreeze_encoder:
            bits.append('unfrozen')
        else:
            bits.append('frozen')
        args.out_tag = '_'.join(bits)

    project_root = Path(__file__).parent.parent
    data_dir = (Path(args.data_dir) if args.data_dir
                else project_root / 'data' / 'processed')
    out_dir = (Path(args.out_dir) if args.out_dir
               else project_root / 'results')
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = (project_root / 'checkpoints'
                / f'classify_{target.lower()}_{args.out_tag}')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print('=' * 70)
    print(f'GENESIS classification — target {target} > {threshold}  '
          f'tag={args.out_tag}')
    print('=' * 70)
    print(f'  Encoder: {args.encoder_size}, no_pretrain={args.no_pretrain}, '
          f'unfreeze={args.unfreeze_encoder}, linear_probe={args.linear_probe}')
    print(f'  Seeds: {args.n_seeds}  Device: {device}  fp16={args.fp16}')
    if not args.no_pretrain and not args.encoder_ckpt:
        raise ValueError('--encoder_ckpt required unless --no_pretrain set')

    # ---- Load + prep data once (shared across seeds) ----
    with open(data_dir / 'normalization_stats.json') as f:
        ns_raw = json.load(f)
    norm_stats = {
        'chem_mean': np.array(ns_raw['chem_mean'], dtype=np.float32),
        'chem_std':  np.array(ns_raw['chem_std'],  dtype=np.float32),
        'chem_log_mask': np.array(ns_raw['chem_log_mask'], dtype=bool),
        'aux_mean':  np.array(ns_raw['aux_mean'],  dtype=np.float32),
        'aux_std':   np.array(ns_raw['aux_std'],   dtype=np.float32),
    }

    pairs_path = data_dir / 'final_temporal_pairs.parquet'
    print(f'\nLoading global pairs: {pairs_path}')
    df = pd.read_parquet(pairs_path)
    t0, t1, gap = _wide_pairs_to_arrays(df)
    full_ds, full_labels = build_labeled_dataset(
        t0, t1, gap, df, None, norm_stats,
        target_param=target, threshold=threshold, source_tag='global',
    )
    n_total = len(full_ds)

    print(f'\nLoading BD held-out:')
    bd_bundle = torch.load(data_dir / 'genesis_held_out_bd_pairs.pt',
                           weights_only=True)
    bd_t0 = bd_bundle['t0'].numpy().astype(np.float32)
    bd_t1 = bd_bundle['t1'].numpy().astype(np.float32)
    bd_gap = bd_bundle.get('gap_years')
    if bd_gap is None or bd_gap.numel() == 1:
        scalar = float(bd_gap.item()) if bd_gap is not None else 0.0
        bd_gap = np.full(bd_t0.shape[0], scalar, dtype=np.float32)
    else:
        bd_gap = bd_gap.numpy().astype(np.float32)
    bd_meta = pd.read_parquet(data_dir / 'genesis_held_out_bd_pairs_meta.parquet')
    bd_aux_path = data_dir / 'genesis_held_out_bd_pairs_aux.pt'
    bd_aux = (torch.load(bd_aux_path, weights_only=True).numpy().astype(np.float32)
              if bd_aux_path.exists() else None)
    bd_ds, _ = build_labeled_dataset(
        bd_t0, bd_t1, bd_gap, bd_meta, bd_aux, norm_stats,
        target_param=target, threshold=threshold, source_tag='bd_held_out',
    )

    # Encoder weights (None = random init each seed)
    encoder_state = None
    if not args.no_pretrain:
        print(f'\nLoading encoder: {args.encoder_ckpt}')
        ckpt = torch.load(args.encoder_ckpt, map_location='cpu',
                          weights_only=False)
        encoder_state = ckpt['model_state_dict']
        print(f"  (val_loss={ckpt.get('val_loss', 'N/A')})")

    # ---- Seed loop ----
    seeds = [args.seed + i for i in range(args.n_seeds)]
    per_seed = []
    for s in seeds:
        r = run_one_seed(s, args, full_ds, full_labels, bd_ds, n_total,
                         encoder_state, device, ckpt_dir)
        per_seed.append(r)

    # ---- Aggregate + save ----
    out = {
        'target_param': target,
        'threshold': threshold,
        'tag': args.out_tag,
        'encoder_size': args.encoder_size,
        'encoder_ckpt': args.encoder_ckpt,
        'no_pretrain': args.no_pretrain,
        'unfreeze_encoder': args.unfreeze_encoder,
        'linear_probe': args.linear_probe,
        'seeds': seeds,
        'aggregate_global_test': aggregate_seeds(per_seed, 'global_test'),
        'aggregate_bangladesh':  aggregate_seeds(per_seed, 'bangladesh_transfer'),
        'per_seed': per_seed,
    }
    out_path = out_dir / f'classify_{target.lower()}_{args.out_tag}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    # ---- Summary ----
    print('\n' + '=' * 70)
    print(f'Summary — target {target} > {threshold}  tag={args.out_tag}')
    print('=' * 70)
    g = out['aggregate_global_test']; b = out['aggregate_bangladesh']
    print(f"  Global test  AUC: {g['auc_mean_std'][0]:.4f} ± {g['auc_mean_std'][1]:.4f}  "
          f"(n_seeds={g['n_seeds_used']})")
    print(f"  Global test  AP : {g['ap_mean_std'][0]:.4f} ± {g['ap_mean_std'][1]:.4f}")
    print(f"  Global test  F1 : {g['f1_best_mean_std'][0]:.4f} ± {g['f1_best_mean_std'][1]:.4f}")
    print(f"  BD held-out  AUC: {b['auc_mean_std'][0]:.4f} ± {b['auc_mean_std'][1]:.4f}")
    print(f"  BD held-out  AP : {b['ap_mean_std'][0]:.4f} ± {b['ap_mean_std'][1]:.4f}")
    print(f"  BD held-out  F1 : {b['f1_best_mean_std'][0]:.4f} ± {b['f1_best_mean_std'][1]:.4f}")
    print(f'\n  Wrote: {out_path}')


if __name__ == '__main__':
    main()
