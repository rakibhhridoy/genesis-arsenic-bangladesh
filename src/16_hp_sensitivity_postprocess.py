"""
Paper 5 — Build supplementary table + figure for HP sensitivity sweep.

Reads results/classify_as_hp_*_mlp_frozen.json files produced by
run_hp_sensitivity.sh, builds a single table of frozen-encoder BD-As AUC
per config, and overlays them against the chemistry-only LogReg reference.

Outputs:
  results/hp_sensitivity_summary.json
  figures/figS3_hp_sensitivity.png
  manuscript_snippets/hp_sensitivity_table.tex
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


CONFIGS = [
    ('baseline',  'mask 0.20, lr 1e-4', '#7f8c8d', 'o'),
    ('low_mask',  'mask 0.10',          '#3498db', 'v'),
    ('high_mask', 'mask 0.30',          '#9b59b6', '^'),
    ('high_lr',   'lr 5e-4',            '#e67e22', 's'),
    ('low_lr',    'lr 3e-5',            '#27ae60', 'D'),
]


def auc_ms(path):
    d = json.load(open(path))
    a, s = d['aggregate_bangladesh']['auc_mean_std']
    g, gs = d['aggregate_global_test']['auc_mean_std']
    return float(a), float(s), float(g), float(gs)


def main():
    proj = Path(__file__).resolve().parent.parent
    results = proj / 'results'
    figures = proj / 'figures'
    snippets = proj / 'manuscript_snippets'
    figures.mkdir(exist_ok=True)
    snippets.mkdir(exist_ok=True)

    # Chem-only LogReg reference
    calib = json.load(open(results / 'calibration_logreg_as_chemonly.json'))
    aucs_lr = [s['bd']['auc'] for s in calib['per_seed']]
    lr_mean, lr_std = float(np.mean(aucs_lr)), float(np.std(aucs_lr))

    rows = []
    for cfg, desc, color, marker in CONFIGS:
        path = results / f'classify_as_hp_{cfg}_mlp_frozen.json'
        if not path.exists():
            print(f'[skip] {path.name} not found (sweep still running?)')
            continue
        m, s, gm, gs = auc_ms(path)
        rows.append({
            'config': cfg,
            'description': desc,
            'bd_auc_mean': m, 'bd_auc_std': s,
            'global_auc_mean': gm, 'global_auc_std': gs,
            'beats_logreg': bool(m - s > lr_mean + lr_std),
            'overlaps_logreg': bool(abs(m - lr_mean) < s + lr_std),
        })
        print(f'  {cfg:<12s} BD={m:.3f}±{s:.3f}  global={gm:.3f}±{gs:.3f}')

    if not rows:
        print('No HP configs ready yet. Re-run when sweep completes.')
        return

    summary = {
        'logreg_chemonly_bd_auc_mean': lr_mean,
        'logreg_chemonly_bd_auc_std': lr_std,
        'configs': rows,
    }
    out_json = results / 'hp_sensitivity_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {out_json}')

    # ---- Figure: BD AUC for each config, with LogReg reference band ----
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(rows))
    means = [r['bd_auc_mean'] for r in rows]
    stds = [r['bd_auc_std'] for r in rows]
    colors = [c for cfg, _, c, _ in CONFIGS if cfg in [r['config'] for r in rows]]
    markers = [m for cfg, _, _, m in CONFIGS if cfg in [r['config'] for r in rows]]

    for xi, m, s, c, mk in zip(x, means, stds, colors, markers):
        ax.errorbar(xi, m, yerr=s, fmt=mk, color=c, markersize=10,
                    capsize=4, lw=1.5, mec='black', mew=0.8)

    ax.axhline(lr_mean, color='black', lw=1.2, ls='--',
               label=f'LogReg chem-only: {lr_mean:.3f}')
    ax.fill_between([-0.5, len(rows) - 0.5],
                    lr_mean - lr_std, lr_mean + lr_std,
                    color='gray', alpha=0.12, zorder=0,
                    label=f'LogReg ±std band')

    ax.set_xticks(x)
    ax.set_xticklabels([r['description'] for r in rows],
                       rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Bangladesh-transfer AUC (As, frozen + MLP)')
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_ylim(0.50, 0.80)
    ax.set_title('HP sensitivity — Small encoder, 50K subsample, 20 epochs\n'
                 'No configuration produces a representation that beats LogReg')
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    out_fig = figures / 'figS3_hp_sensitivity.png'
    fig.savefig(out_fig, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_fig}')

    # ---- LaTeX snippet ----
    tex_lines = [
        '\\begin{table}[H]',
        '\\centering',
        '\\caption{Hyperparameter sensitivity sweep: BD-As frozen+MLP transfer AUC '
        'across five MGM pretraining configurations on the Small encoder, '
        '50K-sample subsample, 20 epochs each. Chemistry-only LogReg reference: '
        f'${lr_mean:.3f} \\pm {lr_std:.3f}$.}}',
        '\\label{tab:sm_hp_sensitivity}',
        '\\small',
        '\\begin{tabular}{lccc}',
        '\\toprule',
        'Configuration & BD AUC & Global AUC & Beats LogReg? \\\\',
        '\\midrule',
    ]
    for r in rows:
        beats = 'No' if not r['beats_logreg'] else 'Yes'
        tex_lines.append(
            f"{r['description']} & ${r['bd_auc_mean']:.3f} \\pm {r['bd_auc_std']:.3f}$ "
            f"& ${r['global_auc_mean']:.3f} \\pm {r['global_auc_std']:.3f}$ "
            f"& {beats} \\\\"
        )
    tex_lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    tex_path = snippets / 'hp_sensitivity_table.tex'
    with open(tex_path, 'w') as f:
        f.write('\n'.join(tex_lines) + '\n')
    print(f'Wrote {tex_path}')

    # ---- Console summary ----
    print('\n' + '='*60)
    print(f'LogReg chem-only reference:  {lr_mean:.3f} ± {lr_std:.3f}')
    print('HP configs:')
    for r in rows:
        verdict = 'BEATS LogReg' if r['beats_logreg'] else (
            'overlaps' if r['overlaps_logreg'] else 'LOSES to LogReg')
        print(f"  {r['description']:<28} BD={r['bd_auc_mean']:.3f}±{r['bd_auc_std']:.3f}  → {verdict}")


if __name__ == '__main__':
    main()
