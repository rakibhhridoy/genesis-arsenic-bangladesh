"""
Paper 5 — LogReg coefficient + calibration analysis for the BD-As headline.

Re-trains the same LogReg pipeline used in 11_baseline_xgboost.py (median impute
+ standardize + class_weight=balanced) on the global pairs, then dumps:

  - standardized coefficients per feature, mean ± std across 3 seeds
  - BD held-out probabilities and labels (per seed)
  - F1-best threshold + confusion matrix at that threshold (per seed + mean)
  - reliability diagram (10 quantile bins) on BD held-out

Outputs:
  results/logreg_coefs.json
  results/calibration_logreg_as.json
  figures/fig4a_confusion_matrix.png
  figures/fig4b_reliability.png
  figures/fig_logreg_coefs.png

Usage:
  python src/13_logreg_analysis.py --target_param As
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    f1_score, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent))
from model.genesis_encoder import GENESIS_PARAMS
from model.temporal_dataset import _wide_pairs_to_arrays

WHO_THRESHOLDS = {'As': 10.0, 'F': 1.5, 'NO3': 50.0, 'U': 30.0}


def build_features(t0, df, use_latlon=True):
    X = t0.copy()
    feature_names = list(GENESIS_PARAMS)
    if use_latlon and 'lat' in df.columns and 'lon' in df.columns:
        lat = df['lat'].to_numpy(dtype=np.float32).reshape(-1, 1)
        lon = df['lon'].to_numpy(dtype=np.float32).reshape(-1, 1)
        X = np.concatenate([X, lat, lon], axis=1)
        feature_names += ['lat', 'lon']
    return X.astype(np.float32), feature_names


def confusion_at_threshold(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return {
        'threshold': float(thr),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'precision': float(tp / max(tp + fp, 1)),
        'recall':    float(tp / max(tp + fn, 1)),
        'specificity': float(tn / max(tn + fp, 1)),
        'f1': float(f1_score(y, yhat, zero_division=0)),
        'n_positive': n_pos, 'n_negative': n_neg,
    }


def reliability_diagram_quantile(y, p, n_bins=10):
    """Quantile-binned reliability: equal-count bins, returns bin centers,
    observed positive rate per bin, mean predicted prob per bin, bin counts."""
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    bin_edges = np.linspace(0, len(p), n_bins + 1).astype(int)
    pred_mean, obs_rate, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi <= lo:
            continue
        pred_mean.append(float(p_sorted[lo:hi].mean()))
        obs_rate.append(float(y_sorted[lo:hi].mean()))
        counts.append(int(hi - lo))
    return np.array(pred_mean), np.array(obs_rate), np.array(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_param', default='As')
    ap.add_argument('--n_seeds', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_latlon', action='store_true',
                    help='Drop lat/lon features; chemistry-only baseline.')
    ap.add_argument('--tag', default=None,
                    help='Filename suffix for outputs (default: chemonly if '
                         '--no_latlon else chemloc)')
    args = ap.parse_args()
    use_latlon = not args.no_latlon
    if args.tag is None:
        args.tag = 'chemloc' if use_latlon else 'chemonly'

    target = args.target_param
    threshold_who = WHO_THRESHOLDS.get(target, 10.0)
    proj = Path(__file__).resolve().parent.parent
    data_dir = proj / 'data' / 'processed'
    results_dir = proj / 'results'
    figures_dir = proj / 'figures'
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    print(f'Loading global pairs for {target} > {threshold_who}')
    df = pd.read_parquet(data_dir / 'final_temporal_pairs.parquet')
    t0, t1, _ = _wide_pairs_to_arrays(df)
    p_idx = GENESIS_PARAMS.index(target)
    valid = ~np.isnan(t1[:, p_idx])
    y_global = (t1[valid, p_idx] > threshold_who).astype(np.float32)
    df_v = df[valid].reset_index(drop=True)
    X_global, feature_names = build_features(t0[valid], df_v, use_latlon=use_latlon)
    print(f'  global: {X_global.shape}, pos={int(y_global.sum())}/{len(y_global)}')

    # Pre-fill columns that are fully NaN across the whole global set with 0,
    # so SimpleImputer doesn't silently drop them and produce varying feature
    # counts across seeds (happens with U in particular).
    all_nan_cols = np.where(np.all(np.isnan(X_global), axis=0))[0]
    if len(all_nan_cols):
        print(f'  fully-NaN columns in global: '
              f'{[feature_names[i] for i in all_nan_cols]} → filling with 0')
        X_global[:, all_nan_cols] = 0.0

    print('Loading BD held-out')
    bd_bundle = torch.load(data_dir / 'genesis_held_out_bd_pairs.pt',
                           weights_only=True)
    bd_t0 = bd_bundle['t0'].numpy().astype(np.float32)
    bd_t1 = bd_bundle['t1'].numpy().astype(np.float32)
    bd_meta = pd.read_parquet(data_dir / 'genesis_held_out_bd_pairs_meta.parquet')
    bd_valid = ~np.isnan(bd_t1[:, p_idx])
    y_bd = (bd_t1[bd_valid, p_idx] > threshold_who).astype(np.float32)
    X_bd, _ = build_features(bd_t0[bd_valid],
                             bd_meta[bd_valid].reset_index(drop=True),
                             use_latlon=use_latlon)
    if X_bd.shape[1] != X_global.shape[1]:
        X_bd = X_bd[:, :X_global.shape[1]]
    # Same fully-NaN guard on BD (defensive — keeps feature dims aligned)
    bd_all_nan = np.where(np.all(np.isnan(X_bd), axis=0))[0]
    if len(bd_all_nan):
        print(f'  fully-NaN columns in BD: '
              f'{[feature_names[i] for i in bd_all_nan]} → filling with 0')
        X_bd[:, bd_all_nan] = 0.0
    print(f'  bd: {X_bd.shape}, pos={int(y_bd.sum())}/{len(y_bd)}')

    n_total = len(y_global)
    n_test = max(50, int(n_total * 0.15))
    n_val = max(50, int(n_total * 0.15))
    n_train = n_total - n_val - n_test

    seeds = [args.seed + i for i in range(args.n_seeds)]
    per_seed_coefs = []
    per_seed_bd_probs = []
    per_seed_metrics = []

    for s in seeds:
        print(f'\n=== seed {s} ===')
        rng = np.random.default_rng(s)
        perm = rng.permutation(n_total)
        tr = perm[:n_train]
        te = perm[n_train + n_val:]
        Xtr, ytr = X_global[tr], y_global[tr]
        Xte, yte = X_global[te], y_global[te]

        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median',
                                      keep_empty_features=True)),
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(class_weight='balanced',
                                      max_iter=2000, random_state=s)),
        ])
        pipe.fit(Xtr, ytr)
        coefs = pipe.named_steps['lr'].coef_.ravel()
        intercept = float(pipe.named_steps['lr'].intercept_[0])

        te_probs = pipe.predict_proba(Xte)[:, 1]
        bd_probs = pipe.predict_proba(X_bd)[:, 1]

        # F1-best threshold from grid scan on BD probs
        ts = np.linspace(0.05, 0.95, 91)
        f1s = [f1_score(y_bd, bd_probs >= t, zero_division=0) for t in ts]
        best_i = int(np.argmax(f1s))
        f1_best_thr = float(ts[best_i])

        metrics = {
            'seed': s,
            'global_test': {
                'auc': float(roc_auc_score(yte, te_probs)),
                'ap':  float(average_precision_score(yte, te_probs)),
                'brier': float(brier_score_loss(yte, te_probs)),
            },
            'bd': {
                'auc': float(roc_auc_score(y_bd, bd_probs)),
                'ap':  float(average_precision_score(y_bd, bd_probs)),
                'brier': float(brier_score_loss(y_bd, bd_probs)),
                'f1_best_threshold': f1_best_thr,
                'confusion_at_f1_best': confusion_at_threshold(
                    y_bd, bd_probs, f1_best_thr),
                'confusion_at_0.5': confusion_at_threshold(y_bd, bd_probs, 0.5),
            },
            'intercept': intercept,
        }
        per_seed_metrics.append(metrics)
        per_seed_coefs.append(coefs)
        per_seed_bd_probs.append(bd_probs)
        print(f"  global AUC={metrics['global_test']['auc']:.3f}  "
              f"BD AUC={metrics['bd']['auc']:.3f}  "
              f"F1-best thr={f1_best_thr:.2f}  "
              f"recall={metrics['bd']['confusion_at_f1_best']['recall']:.2f}  "
              f"prec={metrics['bd']['confusion_at_f1_best']['precision']:.2f}")

    # ---- Aggregate coefficients ----
    coef_arr = np.stack(per_seed_coefs, axis=0)
    coef_mean = coef_arr.mean(axis=0)
    coef_std = coef_arr.std(axis=0)
    coef_records = sorted(
        [{'feature': f, 'coef_mean': float(m), 'coef_std': float(s),
          'abs_coef_mean': float(abs(m))}
         for f, m, s in zip(feature_names, coef_mean, coef_std)],
        key=lambda r: -r['abs_coef_mean']
    )

    coefs_out = {
        'target': target,
        'threshold': threshold_who,
        'feature_names': feature_names,
        'seeds': seeds,
        'coefficients_sorted_by_abs': coef_records,
        'intercept_mean': float(np.mean([m['intercept'] for m in per_seed_metrics])),
        'intercept_std':  float(np.std([m['intercept'] for m in per_seed_metrics])),
        'note': ('Coefficients are on standardized features (median-imputed, '
                 'StandardScaler). class_weight=balanced. Positive coef = higher '
                 'feature value increases P(exceedance).'),
    }
    coefs_path = results_dir / f'logreg_coefs_{args.tag}.json'
    with open(coefs_path, 'w') as f:
        json.dump(coefs_out, f, indent=2)
    print(f'\nWrote {coefs_path}')

    # ---- Coefficient figure ----
    top_n = min(15, len(coef_records))
    top = coef_records[:top_n][::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ys = np.arange(top_n)
    colors = ['#c0392b' if r['coef_mean'] > 0 else '#2c5f8a' for r in top]
    ax.barh(ys, [r['coef_mean'] for r in top],
            xerr=[r['coef_std'] for r in top],
            color=colors, edgecolor='black', linewidth=0.5,
            error_kw={'lw': 0.8, 'capsize': 2.5})
    ax.set_yticks(ys)
    ax.set_yticklabels([r['feature'] for r in top], fontsize=9)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('Standardized LogReg coefficient (mean ± std, 3 seeds)')
    ax.set_title(f'Predictive features for {target} > {threshold_who} '
                 f'µg/L exceedance (global LogReg)')
    fig.tight_layout()
    coefs_fig_path = figures_dir / f'fig_logreg_coefs_{target.lower()}_{args.tag}.png'
    fig.savefig(coefs_fig_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {coefs_fig_path}')

    # ---- Calibration / confusion outputs ----
    # Use mean BD probs across seeds for the figure (3-seed average)
    bd_probs_stack = np.stack(per_seed_bd_probs, axis=0)
    bd_probs_mean = bd_probs_stack.mean(axis=0)

    pred_centers, obs_rates, bin_counts = reliability_diagram_quantile(
        y_bd, bd_probs_mean, n_bins=10)

    # F1-best threshold from the averaged probs (for the headline figure)
    ts = np.linspace(0.05, 0.95, 91)
    f1s_avg = [f1_score(y_bd, bd_probs_mean >= t, zero_division=0) for t in ts]
    best_i = int(np.argmax(f1s_avg))
    f1_best_thr_avg = float(ts[best_i])
    cm_avg = confusion_at_threshold(y_bd, bd_probs_mean, f1_best_thr_avg)

    calib_out = {
        'target': target,
        'threshold_who': threshold_who,
        'method': 'LogReg (raw chem + lat/lon, median-imputed, standardized, '
                  'class_weight=balanced)',
        'seeds': seeds,
        'per_seed': per_seed_metrics,
        'bd_n_samples': int(len(y_bd)),
        'bd_n_positive': int(y_bd.sum()),
        'avg_f1_best_threshold': f1_best_thr_avg,
        'avg_confusion_at_f1_best': cm_avg,
        'reliability_quantile_10bin': {
            'bin_mean_predicted_prob': pred_centers.tolist(),
            'bin_observed_positive_rate': obs_rates.tolist(),
            'bin_counts': bin_counts.tolist(),
            'brier_avg_probs': float(brier_score_loss(y_bd, bd_probs_mean)),
        },
    }
    calib_path = results_dir / f'calibration_logreg_{target.lower()}_{args.tag}.json'
    with open(calib_path, 'w') as f:
        json.dump(calib_out, f, indent=2)
    print(f'Wrote {calib_path}')

    # ---- Confusion matrix figure ----
    fig, ax = plt.subplots(figsize=(4.4, 4))
    cm = np.array([[cm_avg['tn'], cm_avg['fp']],
                   [cm_avg['fn'], cm_avg['tp']]])
    im = ax.imshow(cm, cmap='Blues', vmin=0)
    for (i, j), v in np.ndenumerate(cm):
        color = 'white' if v > cm.max() * 0.5 else 'black'
        ax.text(j, i, str(v), ha='center', va='center', color=color,
                fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Safe', 'Unsafe'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Safe', 'Unsafe'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Observed')
    ax.set_title(f'BD held-out at F1-best thr={f1_best_thr_avg:.2f}\n'
                 f"recall={cm_avg['recall']:.2f}  "
                 f"precision={cm_avg['precision']:.2f}  "
                 f"specificity={cm_avg['specificity']:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    cm_fig_path = figures_dir / f'fig4a_confusion_matrix_{args.tag}.png'
    fig.savefig(cm_fig_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {cm_fig_path}')

    # ---- Reliability figure ----
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Perfect calibration')
    ax.plot(pred_centers, obs_rates, 'o-', color='#c0392b', lw=1.5,
            markersize=8, label='LogReg (BD held-out)')
    for x, y, c in zip(pred_centers, obs_rates, bin_counts):
        ax.annotate(f'n={c}', (x, y), textcoords='offset points',
                    xytext=(6, -4), fontsize=7, color='dimgray')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed exceedance rate')
    brier = calib_out['reliability_quantile_10bin']['brier_avg_probs']
    base_rate = y_bd.mean()
    ax.set_title(f'Reliability diagram — BD As (n={len(y_bd)}, '
                 f'base rate={base_rate:.2f})\n'
                 f'Brier={brier:.3f}, 10 quantile bins')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    rel_fig_path = figures_dir / f'fig4b_reliability_{args.tag}.png'
    fig.savefig(rel_fig_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {rel_fig_path}')

    # ---- Console summary ----
    print('\n' + '=' * 60)
    print(f'Top 10 features by |coef|:')
    for r in coef_records[:10]:
        sign = '+' if r['coef_mean'] > 0 else '-'
        print(f"  {sign} {r['feature']:<10}  "
              f"{r['coef_mean']:+.3f} ± {r['coef_std']:.3f}")
    print('=' * 60)
    print(f"BD operating point at F1-best thr={f1_best_thr_avg:.2f}:")
    print(f"  recall      = {cm_avg['recall']:.2%} of unsafe wells caught")
    print(f"  precision   = {cm_avg['precision']:.2%}")
    print(f"  specificity = {cm_avg['specificity']:.2%}")
    print(f"  TP={cm_avg['tp']}  FN={cm_avg['fn']}  FP={cm_avg['fp']}  "
          f"TN={cm_avg['tn']}")


if __name__ == '__main__':
    main()
