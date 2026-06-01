#!/usr/bin/env python3
"""
Compile a single comparison table from all classify_*.json runs.

Reads results/classify_<param>_<tag>.json files produced by:
  - 10_classify_exceedance.py (encoder runs: frozen / unfrozen / nopretrain / linprobe)
  - 11_baseline_xgboost.py    (xgboost / logreg)

Prints two markdown tables (global test, Bangladesh held-out) showing
mean ± std AUC / AP / F1_best across seeds for every (target, condition)
cell, plus per-cell 95% CIs averaged across seeds. Also writes the same
to results/classify_summary.md.

Usage:
    python src/12_compile_classify_table.py
    python src/12_compile_classify_table.py --results_dir /custom/path
"""

import argparse
import json
from pathlib import Path

import numpy as np


# Display order — left-to-right in the printed table
CONDITION_ORDER = [
    'xgboost',
    'logreg',
    'small_mlp_frozen',
    'small_mlp_unfrozen',
    'small_mlp_nopretrain',
    'small_linprobe_frozen',
    'mlp_frozen',
    'mlp_unfrozen',
    'mlp_nopretrain',
    'linprobe_frozen',
    'large_mlp_frozen',
    'large_mlp_unfrozen',
    'large_mlp_nopretrain',
    'large_linprobe_frozen',
]

CONDITION_LABEL = {
    'xgboost': 'XGBoost (raw)',
    'logreg':  'LogReg (raw)',
    'small_mlp_frozen':     'Small frozen + MLP',
    'small_mlp_unfrozen':   'Small unfrozen + MLP',
    'small_mlp_nopretrain': 'Small no-pretrain + MLP',
    'small_linprobe_frozen': 'Small frozen + LinProbe',
    'mlp_frozen':     'Base frozen + MLP',
    'mlp_unfrozen':   'Base unfrozen + MLP',
    'mlp_nopretrain': 'Base no-pretrain + MLP',
    'linprobe_frozen': 'Base frozen + LinProbe',
    'large_mlp_frozen':     'Large frozen + MLP',
    'large_mlp_unfrozen':   'Large unfrozen + MLP',
    'large_mlp_nopretrain': 'Large no-pretrain + MLP',
    'large_linprobe_frozen': 'Large frozen + LinProbe',
}

TARGET_ORDER = ['As', 'F', 'NO3', 'U']


def load_runs(results_dir: Path):
    """Map (target, tag) → loaded JSON dict."""
    runs = {}
    for p in sorted(results_dir.glob('classify_*.json')):
        # Skip aggregated summaries
        if p.stem.endswith('_summary'):
            continue
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            print(f'  [skip] {p.name}: {e}')
            continue
        target = d.get('target_param')
        tag = d.get('tag')
        if not target or not tag:
            continue
        runs[(target, tag)] = d
    return runs


def _mean_ci_across_seeds(per_seed, split_key, metric):
    """Mean of per-seed bootstrap CI lower/upper across seeds."""
    los, his = [], []
    for s in per_seed:
        ci = s.get(split_key, {}).get(f'{metric}_ci95')
        if not ci:
            continue
        if ci[0] == ci[0] and ci[1] == ci[1]:
            los.append(ci[0]); his.append(ci[1])
    if not los:
        return None, None
    return float(np.mean(los)), float(np.mean(his))


def fmt_cell(d, split_key):
    """One markdown cell: 'AUC=0.70±0.03 (CI 0.65-0.75)'."""
    if d is None:
        return '—'
    agg = d.get(f'aggregate_{split_key}', {})
    # Map keys to file naming convention
    auc = agg.get('auc_mean_std', [float('nan'), float('nan')])
    if not (auc[0] == auc[0]):
        return 'NaN'
    per_seed = d.get('per_seed', [])
    seed_split_key = ('global_test' if split_key == 'global_test'
                      else 'bangladesh_transfer')
    ci_lo, ci_hi = _mean_ci_across_seeds(per_seed, seed_split_key, 'auc')
    cell = f'{auc[0]:.3f}±{auc[1]:.3f}'
    if ci_lo is not None:
        cell += f' (CI {ci_lo:.2f}–{ci_hi:.2f})'
    return cell


def fmt_cell_metric(d, split_key, metric):
    """Generic mean±std cell for any metric in aggregate dict."""
    if d is None:
        return '—'
    agg = d.get(f'aggregate_{split_key}', {})
    key = f'{metric}_mean_std'
    v = agg.get(key, [float('nan'), float('nan')])
    if not (v[0] == v[0]):
        return 'NaN'
    return f'{v[0]:.3f}±{v[1]:.3f}'


def build_markdown_table(runs, split_label, split_key, metric_label, fmt_fn):
    """Markdown table with rows = targets, cols = conditions."""
    header = '| Target | ' + ' | '.join(
        CONDITION_LABEL[c] for c in CONDITION_ORDER
    ) + ' |'
    sep = '|' + '|'.join(['---'] * (len(CONDITION_ORDER) + 1)) + '|'
    rows = [f'### {split_label} — {metric_label}', '', header, sep]
    for target in TARGET_ORDER:
        cells = [target]
        for cond in CONDITION_ORDER:
            d = runs.get((target, cond))
            cells.append(fmt_fn(d, split_key))
        rows.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def find_best_per_target(runs, split_key='bangladesh_transfer', metric='auc'):
    """Per target, return (best_condition, best_mean) ranked by metric."""
    best = {}
    for target in TARGET_ORDER:
        per_cond = []
        for cond in CONDITION_ORDER:
            d = runs.get((target, cond))
            if d is None:
                continue
            agg = d.get(f'aggregate_{split_key}', {})
            v = agg.get(f'{metric}_mean_std', [float('nan'), float('nan')])
            if v[0] == v[0]:
                per_cond.append((cond, v[0], v[1]))
        per_cond.sort(key=lambda x: x[1], reverse=True)
        best[target] = per_cond
    return best


def encoder_vs_xgboost(runs, target, split_key='bangladesh'):
    """Does any encoder condition beat XGBoost with non-overlapping CIs?"""
    xg = runs.get((target, 'xgboost'))
    if xg is None:
        return 'XGBoost run not found', []
    xg_agg = xg['aggregate_' + split_key]
    xg_mean = xg_agg['auc_mean_std'][0]
    xg_std = xg_agg['auc_mean_std'][1]
    xg_upper = xg_mean + 1.96 * xg_std
    lines = [f'  XGBoost AUC = {xg_mean:.3f}±{xg_std:.3f}  '
             f'(95% upper = {xg_upper:.3f})']
    encoder_conditions = ['mlp_frozen', 'mlp_unfrozen',
                          'linprobe_frozen', 'mlp_nopretrain']
    verdicts = []
    for cond in encoder_conditions:
        d = runs.get((target, cond))
        if d is None:
            continue
        agg = d['aggregate_' + split_key]
        mean = agg['auc_mean_std'][0]
        std = agg['auc_mean_std'][1]
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        if lower > xg_upper:
            verdict = 'BEATS XGBoost (non-overlapping CI)'
        elif upper < xg_mean - 1.96 * xg_std:
            verdict = 'LOSES to XGBoost (non-overlapping CI)'
        else:
            verdict = 'tie (overlapping CI)'
        lines.append(f'  {CONDITION_LABEL[cond]:<28s} '
                     f'AUC={mean:.3f}±{std:.3f}  → {verdict}')
        verdicts.append((cond, verdict))
    return '\n'.join(lines), verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default=None)
    ap.add_argument('--out', default=None,
                    help='Write markdown to this file (default results/classify_summary.md)')
    args = ap.parse_args()

    project_root = Path(__file__).parent.parent
    results_dir = (Path(args.results_dir) if args.results_dir
                   else project_root / 'results')
    out_path = (Path(args.out) if args.out
                else results_dir / 'classify_summary.md')

    runs = load_runs(results_dir)
    if not runs:
        print(f'No classify_*.json files found in {results_dir}')
        return

    available = sorted({k[1] for k in runs})
    targets_with_data = sorted({k[0] for k in runs})
    print(f'Loaded {len(runs)} runs')
    print(f'  conditions present: {available}')
    print(f'  targets present:    {targets_with_data}')

    blocks = ['# Classification sweep — comparison tables', '']
    blocks.append(build_markdown_table(
        runs, 'Bangladesh held-out (transfer)',
        'bangladesh', 'AUC ± std (95% CI)', fmt_cell))
    blocks.append('')
    blocks.append(build_markdown_table(
        runs, 'Bangladesh held-out (transfer)',
        'bangladesh', 'AP ± std',
        lambda d, k: fmt_cell_metric(d, k, 'ap')))
    blocks.append('')
    blocks.append(build_markdown_table(
        runs, 'Bangladesh held-out (transfer)',
        'bangladesh', 'F1_best ± std',
        lambda d, k: fmt_cell_metric(d, k, 'f1_best')))
    blocks.append('')
    blocks.append(build_markdown_table(
        runs, 'Global test (in-distribution)',
        'global_test', 'AUC ± std (95% CI)', fmt_cell))
    blocks.append('')

    blocks.append('## Per-target verdict — does any encoder condition beat XGBoost on BD AUC?')
    for target in TARGET_ORDER:
        blocks.append(f'\n### {target}')
        try:
            txt, _ = encoder_vs_xgboost(runs, target)
            blocks.append(txt)
        except Exception as e:
            blocks.append(f'  (error: {e})')

    blocks.append('')
    blocks.append('## Best condition per target (BD AUC ranking)')
    best = find_best_per_target(runs, 'bangladesh', 'auc')
    for target in TARGET_ORDER:
        if not best.get(target):
            continue
        blocks.append(f'\n### {target}')
        for i, (cond, mean, std) in enumerate(best[target], start=1):
            blocks.append(f'  {i}. {CONDITION_LABEL[cond]:<30s} '
                          f'AUC={mean:.3f}±{std:.3f}')

    text = '\n'.join(blocks)
    print('\n' + text + '\n')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(text)
    print(f'\nWrote: {out_path}')


if __name__ == '__main__':
    main()
