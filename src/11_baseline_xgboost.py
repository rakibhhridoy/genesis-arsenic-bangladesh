#!/usr/bin/env python3
"""
Non-neural baseline for the WHO-exceedance classification task.

Trains XGBoost and Logistic Regression on raw t0 chemistry (+ optional aux
features) to answer: does the GENESIS encoder add value over the most obvious
baselines? If XGBoost on raw inputs matches the encoder, the architecture
buys nothing.

Outputs match the schema of 07_classify_arsenic.py for easy comparison:
  results/classify_<param>_xgboost.json
  results/classify_<param>_logreg.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, confusion_matrix,
)
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from model.genesis_encoder import GENESIS_PARAMS, AUX_FEATURES, NUM_PARAMS
from model.temporal_dataset import _wide_pairs_to_arrays


WHO_THRESHOLDS = {
    'As': 10.0, 'F': 1.5, 'NO3': 50.0, 'PO4': 0.1, 'U': 30.0,
}


def _safe_auc(y, p):
    if len(y) < 2 or y.sum() == 0 or y.sum() == len(y):
        return float('nan')
    return float(roc_auc_score(y, p))


def _safe_ap(y, p):
    if y.sum() == 0:
        return float('nan')
    return float(average_precision_score(y, p))


def bootstrap_metrics(labels, probs, n_boot=1000, seed=0):
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


def evaluate(probs, labels, n_boot=1000, seed=0):
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
        out.update(bootstrap_metrics(labels, probs, n_boot=n_boot, seed=seed))
    else:
        out['note'] = 'single-class split — AUC undefined'
    return out


def build_features(t0, df, aux=None, use_aux=True):
    """Stack t0 chemistry (raw) + optional aux features. NaNs preserved for imputer."""
    feats = [t0]  # (N, 20)
    feature_names = list(GENESIS_PARAMS)
    if use_aux and aux is not None:
        feats.append(aux)
        feature_names += list(AUX_FEATURES)
    # Lat/lon from df
    if df is not None:
        if 'lat' in df.columns and 'lon' in df.columns:
            ll = np.stack([df['lat'].to_numpy(np.float32),
                           df['lon'].to_numpy(np.float32)], axis=1)
            feats.append(ll)
            feature_names += ['lat', 'lon']
    X = np.concatenate(feats, axis=1)
    return X.astype(np.float32), feature_names


def aggregate_seeds(per_seed, key):
    def _good(v):
        return v is not None and v == v  # drop None and NaN
    aucs = [r[key].get('auc') for r in per_seed if _good(r[key].get('auc'))]
    aps  = [r[key].get('ap')  for r in per_seed if _good(r[key].get('ap'))]
    f1s  = [r[key].get('f1_best') for r in per_seed
            if _good(r[key].get('f1_best'))]
    def _ms(v):
        return ([float('nan'), float('nan')] if not v
                else [float(np.mean(v)), float(np.std(v))])
    return {
        'n_seeds_used': len(aucs),
        'auc_mean_std': _ms(aucs),
        'ap_mean_std':  _ms(aps),
        'f1_best_mean_std': _ms(f1s),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_param', default='As')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--model', choices=['xgboost', 'logreg'], default='xgboost')
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap_n', type=int, default=1000)
    parser.add_argument('--no_aux', action='store_true',
                        help='Drop aux features; chemistry only')
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--out_tag', default=None)
    args = parser.parse_args()

    target = args.target_param
    threshold = (args.threshold if args.threshold is not None
                 else WHO_THRESHOLDS.get(target, 10.0))
    use_aux = not args.no_aux
    if args.out_tag is None:
        args.out_tag = args.model + ('' if use_aux else '_chemonly')

    project_root = Path(__file__).parent.parent
    data_dir = (Path(args.data_dir) if args.data_dir
                else project_root / 'data' / 'processed')
    out_dir = (Path(args.out_dir) if args.out_dir
               else project_root / 'results')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print(f'Baseline {args.model} — target {target} > {threshold}  '
          f'tag={args.out_tag}')
    print('=' * 70)

    # ---- Global pairs ----
    print(f'\nLoading global pairs')
    df = pd.read_parquet(data_dir / 'final_temporal_pairs.parquet')
    t0, t1, _gap = _wide_pairs_to_arrays(df)
    p_idx = GENESIS_PARAMS.index(target)
    raw_t1 = t1[:, p_idx]
    valid = ~np.isnan(raw_t1)
    labels = (raw_t1[valid] > threshold).astype(np.float32)
    df_v = df[valid].reset_index(drop=True)
    t0_v = t0[valid]
    # Global pairs have no aux .pt — pass aux=None
    X_global, feature_names = build_features(t0_v, df_v, aux=None, use_aux=False)
    print(f'  global: {X_global.shape}, {int(labels.sum())} pos / {len(labels)}')

    # ---- BD held-out ----
    print(f'\nLoading BD held-out')
    bd_bundle = torch.load(data_dir / 'genesis_held_out_bd_pairs.pt',
                           weights_only=True)
    bd_t0 = bd_bundle['t0'].numpy().astype(np.float32)
    bd_t1 = bd_bundle['t1'].numpy().astype(np.float32)
    bd_meta = pd.read_parquet(data_dir / 'genesis_held_out_bd_pairs_meta.parquet')
    bd_aux_path = data_dir / 'genesis_held_out_bd_pairs_aux.pt'
    bd_aux = (torch.load(bd_aux_path, weights_only=True).numpy().astype(np.float32)
              if (use_aux and bd_aux_path.exists()) else None)
    bd_raw_t1 = bd_t1[:, p_idx]
    bd_valid = ~np.isnan(bd_raw_t1)
    bd_labels = (bd_raw_t1[bd_valid] > threshold).astype(np.float32)
    bd_meta_v = bd_meta[bd_valid].reset_index(drop=True)
    bd_t0_v = bd_t0[bd_valid]
    bd_aux_v = bd_aux[bd_valid] if bd_aux is not None else None
    X_bd, _ = build_features(bd_t0_v, bd_meta_v, aux=bd_aux_v, use_aux=use_aux)

    # If global has no aux but BD does, slice BD aux columns off so feature
    # dims match (we use the chem-only feature set on both for fair comparison).
    if X_bd.shape[1] != X_global.shape[1]:
        X_bd = X_bd[:, :X_global.shape[1]]
    print(f'  bd: {X_bd.shape}, {int(bd_labels.sum())} pos / {len(bd_labels)}')

    n_total = len(labels)
    n_test = max(50, int(n_total * 0.15))
    n_val  = max(50, int(n_total * 0.15))
    n_train = n_total - n_val - n_test

    seeds = [args.seed + i for i in range(args.n_seeds)]
    per_seed = []
    for s in seeds:
        print(f'\n=== seed {s} ===')
        rng = np.random.default_rng(s)
        perm = rng.permutation(n_total)
        train_idx = perm[:n_train]
        val_idx   = perm[n_train:n_train + n_val]
        test_idx  = perm[n_train + n_val:]
        Xtr, ytr = X_global[train_idx], labels[train_idx]
        Xva, yva = X_global[val_idx],   labels[val_idx]
        Xte, yte = X_global[test_idx],  labels[test_idx]

        n_pos = int(ytr.sum())
        n_neg = len(ytr) - n_pos
        scale_pos_weight = (n_neg / max(1, n_pos))

        # Skip seeds with single-class train/test splits — common for rare
        # contaminants like U where positives can land entirely in val/test.
        if n_pos == 0 or n_neg == 0 or int(yte.sum()) == 0:
            print(f'  [skip] seed {s}: degenerate split '
                  f'(train_pos={n_pos}/{len(ytr)}, '
                  f'test_pos={int(yte.sum())}/{len(yte)})')
            per_seed.append({
                'seed': s,
                'global_test': {'note': 'degenerate split — skipped',
                                'n_samples': len(yte),
                                'n_positive': int(yte.sum())},
                'bangladesh_transfer': {'note': 'skipped due to global split',
                                        'n_samples': len(bd_labels),
                                        'n_positive': int(bd_labels.sum())},
                'elapsed_sec': 0.0,
            })
            continue

        t0_t = time.time()
        if args.model == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight,
                eval_metric='aucpr',
                early_stopping_rounds=30,
                random_state=s, n_jobs=4,
                tree_method='hist',
            )
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            te_probs = model.predict_proba(Xte)[:, 1]
            bd_probs = (model.predict_proba(X_bd)[:, 1]
                        if len(X_bd) else np.zeros(0))
        else:  # logreg
            model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(class_weight='balanced',
                                          max_iter=2000, random_state=s)),
            ])
            model.fit(Xtr, ytr)
            te_probs = model.predict_proba(Xte)[:, 1]
            bd_probs = (model.predict_proba(X_bd)[:, 1]
                        if len(X_bd) else np.zeros(0))
        elapsed = time.time() - t0_t

        test_metrics = evaluate(te_probs, yte, n_boot=args.bootstrap_n, seed=s)
        bd_metrics   = evaluate(bd_probs, bd_labels,
                                n_boot=args.bootstrap_n, seed=s)
        print(f'  Global test AUC: {test_metrics.get("auc", float("nan")):.4f}  '
              f'AP: {test_metrics.get("ap", float("nan")):.4f}')
        print(f'  BD          AUC: {bd_metrics.get("auc", float("nan")):.4f}  '
              f'AP: {bd_metrics.get("ap", float("nan")):.4f}')
        per_seed.append({
            'seed': s,
            'global_test': test_metrics,
            'bangladesh_transfer': bd_metrics,
            'elapsed_sec': float(elapsed),
        })

    out = {
        'target_param': target,
        'threshold': threshold,
        'tag': args.out_tag,
        'model': args.model,
        'use_aux': use_aux,
        'feature_names': feature_names,
        'seeds': seeds,
        'aggregate_global_test': aggregate_seeds(per_seed, 'global_test'),
        'aggregate_bangladesh':  aggregate_seeds(per_seed, 'bangladesh_transfer'),
        'per_seed': per_seed,
    }
    out_path = out_dir / f'classify_{target.lower()}_{args.out_tag}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print('\n' + '=' * 70)
    print(f'Summary — {args.model} on {target} > {threshold}')
    print('=' * 70)
    g = out['aggregate_global_test']; b = out['aggregate_bangladesh']
    print(f"  Global AUC: {g['auc_mean_std'][0]:.4f} ± {g['auc_mean_std'][1]:.4f}  "
          f"AP: {g['ap_mean_std'][0]:.4f} ± {g['ap_mean_std'][1]:.4f}")
    print(f"  BD     AUC: {b['auc_mean_std'][0]:.4f} ± {b['auc_mean_std'][1]:.4f}  "
          f"AP: {b['ap_mean_std'][0]:.4f} ± {b['ap_mean_std'][1]:.4f}")
    print(f'\n  Wrote: {out_path}')


if __name__ == '__main__':
    main()
