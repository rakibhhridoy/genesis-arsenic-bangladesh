"""
Paper 5 — Distribution-shift quantification (Figure 6 / §6.1).

Compares the 2.09M global pretrain corpus against the 2.79K BD samples on
shared chemistry features, using:

  - per-parameter 1-Wasserstein distance (standardized scale)
  - PCA fit on pretrain, project BD on top
  - sample-level energy distance on the 16-D shared subset

Drops F, U, SiO2, DOC (BD coverage <40% so per-param comparison is unreliable).
Pre-transform: log1p on concentrations (skip pH, Eh which are not concentrations).

Outputs:
  results/dist_shift.json
  figures/fig6_dist_shift_pca.png
  figures/fig6b_dist_shift_per_param.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).parent))
from model.genesis_encoder import GENESIS_PARAMS

NON_CONC = {'pH', 'Eh'}  # log1p not appropriate
BD_LOW_COVERAGE = {'F', 'U', 'SiO2', 'DOC'}  # BD missingness > 40%


def log_transform(x, name):
    if name in NON_CONC:
        return x
    # log1p with negative clamp (Eh is negative-allowed but excluded above)
    return np.log1p(np.clip(x, 0, None))


def per_dim_w1(a, b):
    """1-Wasserstein on standardized values (so distances are comparable
    across chem params with different units/scales)."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    mu, sd = a.mean(), a.std() + 1e-9
    return float(wasserstein_distance((a - mu) / sd, (b - mu) / sd))


def energy_distance_subsampled(X, Y, n_max=2000, seed=0):
    """E-distance: 2·E|X-Y| − E|X-X'| − E|Y-Y'|. Subsample both to n_max
    points so the O(n²) pairwise computation stays bounded."""
    rng = np.random.default_rng(seed)
    if len(X) > n_max:
        X = X[rng.choice(len(X), n_max, replace=False)]
    if len(Y) > n_max:
        Y = Y[rng.choice(len(Y), n_max, replace=False)]
    def mean_pdist(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=-1)
        return float(np.sqrt(d2).mean())
    xy = mean_pdist(X, Y)
    xx = mean_pdist(X, X)
    yy = mean_pdist(Y, Y)
    return float(2 * xy - xx - yy), int(len(X)), int(len(Y))


def main():
    proj = Path(__file__).resolve().parent.parent
    data_dir = proj / 'data' / 'processed'
    results_dir = proj / 'results'
    figures_dir = proj / 'figures'
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    print('Loading pretrain corpus')
    pre = torch.load(data_dir / 'genesis_pretrain.pt', weights_only=True).numpy()
    pre_meta = pd.read_parquet(data_dir / 'genesis_pretrain_meta.parquet')
    print(f'  pretrain: {pre.shape}')

    print('Loading BD samples')
    bd = torch.load(data_dir / 'genesis_held_out_bd.pt', weights_only=True).numpy()
    print(f'  BD: {bd.shape}')

    # Feature subset: chem params present in both with reasonable BD coverage
    kept = [p for p in GENESIS_PARAMS if p not in BD_LOW_COVERAGE]
    keep_idx = [GENESIS_PARAMS.index(p) for p in kept]
    print(f'  shared usable params ({len(kept)}): {kept}')
    print(f'  dropped (BD coverage low): {sorted(BD_LOW_COVERAGE)}')

    pre_X = pre[:, keep_idx]
    bd_X  = bd[:, keep_idx]

    # ---- Per-parameter Wasserstein-1 (on log + standardized scale) ----
    per_param = {}
    for i, name in enumerate(kept):
        a = log_transform(pre_X[:, i], name)
        b = log_transform(bd_X[:, i], name)
        w1 = per_dim_w1(a, b)
        per_param[name] = {
            'w1_standardized': w1,
            'pretrain_nonnan': int((~np.isnan(a)).sum()),
            'bd_nonnan': int((~np.isnan(b)).sum()),
            'pretrain_median': float(np.nanmedian(pre_X[:, i])),
            'bd_median': float(np.nanmedian(bd_X[:, i])),
        }

    # Build the joint matrix for PCA / energy distance:
    # log-transform, then for missingness use per-column median imputation
    # computed on the pretrain side only (BD inherits pretrain's median).
    pre_log = np.column_stack([log_transform(pre_X[:, i], n)
                               for i, n in enumerate(kept)])
    bd_log  = np.column_stack([log_transform(bd_X[:, i], n)
                               for i, n in enumerate(kept)])

    pre_medians = np.nanmedian(pre_log, axis=0)
    pre_filled = np.where(np.isnan(pre_log), pre_medians, pre_log)
    bd_filled  = np.where(np.isnan(bd_log),  pre_medians, bd_log)

    scaler = StandardScaler().fit(pre_filled)
    pre_z = scaler.transform(pre_filled)
    bd_z  = scaler.transform(bd_filled)

    # ---- PCA fit on pretrain, project BD ----
    pca = PCA(n_components=4).fit(pre_z)
    pre_pcs = pca.transform(pre_z)
    bd_pcs  = pca.transform(bd_z)
    print(f'  PCA explained var (first 4): '
          f'{pca.explained_variance_ratio_.round(3).tolist()}  '
          f'cum={pca.explained_variance_ratio_.sum():.3f}')

    # ---- Energy distance on PCA-reduced 4-D coords (fast & shift-sensitive) ----
    e_dist, n_pre_sub, n_bd_sub = energy_distance_subsampled(
        pre_pcs, bd_pcs, n_max=2000, seed=0)
    print(f'  energy distance (4-D PC space, n_pre={n_pre_sub}, '
          f'n_bd={n_bd_sub}): {e_dist:.4f}')

    # ---- Null distribution: random splits of the pretrain corpus ----
    # For each of n_null seeds, partition pretrain into two random halves
    # of size n_bd each (matching the BD scale) and compute the same energy
    # distance. The distribution of these values is the null against which
    # the actual pretrain-vs-BD distance should be compared.
    n_null = 20
    null_dists = []
    rng_null = np.random.default_rng(2026)
    n_bd_actual = len(bd_pcs)
    for k in range(n_null):
        idx = rng_null.permutation(len(pre_pcs))
        a = pre_pcs[idx[:n_bd_actual]]
        b = pre_pcs[idx[n_bd_actual:2*n_bd_actual]]
        d, _, _ = energy_distance_subsampled(a, b, n_max=2000, seed=k)
        null_dists.append(d)
    null_dists = np.array(null_dists)
    print(f'  random-split null (n={n_null}): '
          f'mean={null_dists.mean():.4f}  std={null_dists.std():.4f}  '
          f'max={null_dists.max():.4f}')
    print(f'  z-score of pretrain-vs-BD vs null: '
          f'{(e_dist - null_dists.mean())/(null_dists.std()+1e-9):.1f}')

    # ---- Save JSON ----
    out = {
        'n_pretrain': int(pre.shape[0]),
        'n_bd': int(bd.shape[0]),
        'params_used': kept,
        'params_dropped_bd_low_coverage': sorted(BD_LOW_COVERAGE),
        'per_param_w1_standardized': per_param,
        'pca': {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': float(pca.explained_variance_ratio_.sum()),
            'components_loadings': pca.components_.tolist(),
            'feature_order': kept,
        },
        'energy_distance_pca4d': {
            'value': e_dist,
            'n_pretrain_subsampled': n_pre_sub,
            'n_bd_subsampled': n_bd_sub,
            'random_split_null_n': int(n_null),
            'random_split_null_mean': float(null_dists.mean()),
            'random_split_null_std': float(null_dists.std()),
            'random_split_null_max': float(null_dists.max()),
            'z_vs_null': float((e_dist - null_dists.mean())
                               / (null_dists.std() + 1e-9)),
            'ratio_to_null_mean': float(e_dist / max(null_dists.mean(), 1e-9)),
            'note': ('Sample-level energy distance on first 4 PCs of '
                     'standardized log-chem features. 0 = same distribution, '
                     '>0 = shift. Subsampled for tractability. The null '
                     'distribution is computed by randomly splitting the '
                     'pretrain corpus into two halves of size n_bd each.'),
        },
        'pretrain_pc_mean': pre_pcs.mean(axis=0).tolist(),
        'pretrain_pc_std':  pre_pcs.std(axis=0).tolist(),
        'bd_pc_mean':       bd_pcs.mean(axis=0).tolist(),
        'bd_pc_std':        bd_pcs.std(axis=0).tolist(),
    }
    out_path = results_dir / 'dist_shift.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote {out_path}')

    # ---- PCA figure ----
    rng = np.random.default_rng(0)
    sub = rng.choice(len(pre_pcs), min(20000, len(pre_pcs)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    ax.scatter(pre_pcs[sub, 0], pre_pcs[sub, 1],
               s=2, alpha=0.10, color='#7f8c8d', label=f'Pretrain (n={len(pre_pcs):,})')
    ax.scatter(bd_pcs[:, 0], bd_pcs[:, 1],
               s=8, alpha=0.6, color='#c0392b', edgecolor='black',
               linewidth=0.2, label=f'BD (n={len(bd_pcs):,})')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PC1–PC2 of log-transformed shared chemistry')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.scatter(pre_pcs[sub, 2], pre_pcs[sub, 3],
               s=2, alpha=0.10, color='#7f8c8d')
    ax.scatter(bd_pcs[:, 2], bd_pcs[:, 3],
               s=8, alpha=0.6, color='#c0392b', edgecolor='black',
               linewidth=0.2)
    ax.set_xlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
    ax.set_ylabel(f'PC4 ({pca.explained_variance_ratio_[3]*100:.1f}%)')
    ax.set_title(f'PC3–PC4  (energy dist = {e_dist:.3f})')
    ax.grid(alpha=0.2)

    fig.suptitle(f'Distribution shift: global pretrain vs Bangladesh '
                 f'({len(kept)} shared chemistry features, log-transformed)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(figures_dir / 'fig6_dist_shift_pca.png',
                dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {figures_dir / "fig6_dist_shift_pca.png"}')

    # ---- Per-parameter shift figure ----
    sorted_params = sorted(kept, key=lambda p: -per_param[p]['w1_standardized']
                           if not np.isnan(per_param[p]['w1_standardized']) else 0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ys = np.arange(len(sorted_params))
    w1s = [per_param[p]['w1_standardized'] for p in sorted_params]
    bars = ax.barh(ys, w1s, color='#2c5f8a', edgecolor='black', linewidth=0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels(sorted_params, fontsize=9)
    ax.set_xlabel('1-Wasserstein distance (standardized scale)')
    ax.set_title(f'Per-parameter shift, BD vs global pretrain  '
                 f'({len(kept)} shared chemistry features)')
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')
    for i, (p, w) in enumerate(zip(sorted_params, w1s)):
        ax.text(w + 0.02, i, f'{w:.2f}', va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / 'fig6b_dist_shift_per_param.png',
                dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {figures_dir / "fig6b_dist_shift_per_param.png"}')

    # ---- Console summary ----
    print('\n' + '=' * 60)
    print(f'Per-parameter W1 distance (standardized), top movers:')
    for p in sorted_params[:8]:
        d = per_param[p]
        print(f"  {p:<6}  W1={d['w1_standardized']:.3f}  "
              f"pre_med={d['pretrain_median']:.3g}  "
              f"bd_med={d['bd_median']:.3g}  "
              f"(BD n={d['bd_nonnan']})")
    print('=' * 60)
    print(f'Multivariate energy distance (4-D PC space, subsampled): {e_dist:.3f}')
    print('Interpretation: 0 = identical distribution, larger = more shift.')


if __name__ == '__main__':
    main()
