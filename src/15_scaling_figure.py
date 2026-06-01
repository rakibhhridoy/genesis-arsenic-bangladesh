"""
Paper 5 — Figure 3: pretraining-scaling forest plot.

Reads BD-As AUC from the existing classify_*_*.json files for the frozen+MLP
configuration at each encoder size, and plots them against parameter count.
Adds a horizontal reference line for the chem-only LogReg BD AUC.

Output: figures/fig3_scaling.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


SIZES = {
    'small': {'params_M': 1.47, 'label': 'Small\n(1.47M)'},
    'base':  {'params_M': 8.99, 'label': 'Base\n(8.99M)'},
    'large': {'params_M': 48.4, 'label': 'Large\n(48.4M)'},
}


def auc_for(path):
    d = json.load(open(path))
    a, s = d['aggregate_bangladesh']['auc_mean_std']
    return float(a), float(s)


def main():
    proj = Path(__file__).resolve().parent.parent
    results = proj / 'results'
    figures = proj / 'figures'
    figures.mkdir(exist_ok=True)

    # File layout in results/:
    #   classify_as_small_mlp_frozen.json
    #   classify_as_mlp_frozen.json           (base, no size tag)
    #   classify_as_large_mlp_frozen.json
    file_map = {
        'small': 'classify_as_small_mlp_frozen.json',
        'base':  'classify_as_mlp_frozen.json',
        'large': 'classify_as_large_mlp_frozen.json',
    }
    nopretrain_map = {
        'small': 'classify_as_small_mlp_nopretrain.json',
        'base':  'classify_as_mlp_nopretrain.json',
        'large': 'classify_as_large_mlp_nopretrain.json',
    }

    pre_means, pre_stds = [], []
    np_means, np_stds = [], []
    params = []
    labels = []

    for size in ['small', 'base', 'large']:
        m, s = auc_for(results / file_map[size])
        nm, ns = auc_for(results / nopretrain_map[size])
        pre_means.append(m); pre_stds.append(s)
        np_means.append(nm); np_stds.append(ns)
        params.append(SIZES[size]['params_M'])
        labels.append(SIZES[size]['label'])

    # Reference: chem-only LogReg
    calib = json.load(open(results / 'calibration_logreg_as_chemonly.json'))
    aucs_lr = [s['bd']['auc'] for s in calib['per_seed']]
    lr_mean, lr_std = float(np.mean(aucs_lr)), float(np.std(aucs_lr))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ax.errorbar(params, pre_means, yerr=pre_stds, fmt='o-',
                color='#c0392b', lw=1.8, markersize=9, capsize=4,
                label='Pretrained encoder, frozen + MLP (BD-As)')
    ax.errorbar(params, np_means, yerr=np_stds, fmt='s--',
                color='#2c5f8a', lw=1.3, markersize=8, capsize=4,
                alpha=0.85, label='No-pretrain control, MLP (same size)')

    ax.axhline(lr_mean, color='black', lw=1.2, ls=':',
               label=f'LogReg (chem only): {lr_mean:.3f} $\\pm$ {lr_std:.3f}')
    ax.fill_between(
        [params[0] * 0.7, params[-1] * 1.4],
        lr_mean - lr_std, lr_mean + lr_std,
        color='gray', alpha=0.12, zorder=0,
    )

    ax.set_xscale('log')
    ax.set_xticks(params)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel('Pretrained encoder size (parameters)')
    ax.set_ylabel('Bangladesh-transfer AUC')
    ax.set_title('Pretraining benefit is non-monotonic and never positive\n'
                 'BD-As exceedance, 3 seeds, $\\pm$std error bars')
    ax.set_ylim(0.50, 0.78)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.95)

    # Annotate each point
    for x, y, s in zip(params, pre_means, pre_stds):
        ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points',
                    xytext=(0, 12), ha='center', fontsize=8, color='#c0392b')

    fig.tight_layout()
    out = figures / 'fig3_scaling.png'
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')

    # Console summary
    print('\nFrozen+MLP BD-As AUC by size:')
    for sz, m, s in zip(['small', 'base', 'large'], pre_means, pre_stds):
        print(f'  {sz:6s} ({SIZES[sz]["params_M"]:>5.2f}M): {m:.3f} ± {s:.3f}')
    print(f'\nNo-pretrain control:')
    for sz, m, s in zip(['small', 'base', 'large'], np_means, np_stds):
        print(f'  {sz:6s}: {m:.3f} ± {s:.3f}')
    print(f'\nChem-only LogReg ref: {lr_mean:.3f} ± {lr_std:.3f}')


if __name__ == '__main__':
    main()
