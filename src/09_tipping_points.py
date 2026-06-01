"""
GENESIS Paper 5 — Step 9: Tipping Point Detection
====================================================
Detects geochemical tipping points — stations where diffusion model
predictions show bimodal distributions or exploding uncertainty,
indicating potential irreversible state changes.

The finding "X% of safe aquifers are at a geochemical tipping point"
is the potential Science-level headline.

Runs on CPU/MPS (Mac) with downloaded checkpoints.

Usage:
  python 09_tipping_points.py --ckpt checkpoints/diffusion_base/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from model.genesis_encoder import (
    GENESISForMGM, GENESIS_PARAMS, PARAM_TO_ID, PAD_TOKEN, CLS_TOKEN,
)
from model.latent_diffusion import GENESISLatentDiffusion
from model.dataset import load_stats

# Reuse station dataset from projection script
from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location("proj", str(Path(__file__).parent / "08_global_projection.py"))
_proj = module_from_spec(_spec)
_spec.loader.exec_module(_proj)
StationDataset = _proj.StationDataset
station_collate = _proj.station_collate
load_model = _proj.load_model
WHO_THRESHOLDS = _proj.WHO_THRESHOLDS

DPI = 300
FONT_SIZE = 9


def _setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE + 1,
        'axes.titlesize': FONT_SIZE + 2,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
    })


def detect_bimodality(values, bw_factor=0.3):
    """
    Detect if a distribution is bimodal using KDE peak finding.

    Returns:
        n_modes: number of detected modes
        is_bimodal: True if 2+ modes
    """
    if len(values) < 10 or np.std(values) < 1e-6:
        return 1, False

    try:
        kde = gaussian_kde(values, bw_method=bw_factor)
        x_grid = np.linspace(np.min(values), np.max(values), 200)
        density = kde(x_grid)
        peaks, properties = find_peaks(density, height=np.max(density) * 0.1,
                                        distance=20)
        n_modes = len(peaks)
        return n_modes, n_modes >= 2
    except Exception:
        return 1, False


def generate_multi_horizon(model, batch, device, norm_stats, horizons, n_samples, ddim_steps):
    """Generate predictions at multiple time horizons for one batch.

    Denormalization mirrors training: undo z-score then expm1 for log-mask
    params, so WHO-threshold comparisons downstream are in raw units.
    """
    chem_mean = np.asarray(norm_stats['chem_mean'], dtype=np.float32)
    chem_std = np.asarray(norm_stats['chem_std'], dtype=np.float32)
    chem_log = np.asarray(norm_stats['chem_log_mask'], dtype=bool)
    B = batch['param_ids'].shape[0]

    current_state = {
        'param_ids': batch['param_ids'].to(device),
        'values': batch['values'].to(device),
        'padding_mask': batch['padding_mask'].to(device),
    }
    metadata = batch['metadata'].to(device)

    horizon_results = {}

    with torch.no_grad():
        for dt_val in horizons:
            dt = torch.full((B, 1), dt_val, dtype=torch.float32, device=device)
            preds_list = model.generate_fast(
                current_state, metadata, dt,
                n_samples=n_samples, ddim_steps=ddim_steps,
            )

            batch_preds = []
            for b in range(B):
                station_preds = {}
                for i, p in enumerate(GENESIS_PARAMS):
                    mean = float(chem_mean[i])
                    std = float(chem_std[i])
                    is_log = bool(chem_log[i])
                    raw_preds = []
                    for sample in preds_list:
                        if p in sample:
                            v = sample[p][b].item() * std + mean
                            if is_log:
                                v = float(np.expm1(v))
                            raw_preds.append(v)
                    if raw_preds:
                        station_preds[p] = np.array(raw_preds)
                batch_preds.append(station_preds)

            horizon_results[dt_val] = batch_preds

    return horizon_results


def classify_station(current_val, horizon_preds, threshold):
    """
    Classify station regime based on multi-horizon predictions.

    Returns:
        regime: one of STABLE_SAFE, STABLE_UNSAFE, TIPPING_DANGEROUS,
                TIPPING_RECOVERY, UNCERTAIN
        metrics: dict of analysis results
    """
    if pd.isna(current_val):
        return 'NO_DATA', {}

    currently_safe = current_val <= threshold
    horizons = sorted(horizon_preds.keys())

    p_exceeds = []
    pred_stds = []
    bimodal_flags = []

    for h in horizons:
        preds = horizon_preds[h]
        if preds is None or len(preds) == 0:
            continue
        p_exceed = np.mean(preds > threshold)
        p_exceeds.append(p_exceed)
        pred_stds.append(np.std(preds))
        n_modes, is_bimodal = detect_bimodality(preds)
        bimodal_flags.append(is_bimodal)

    if not p_exceeds:
        return 'NO_PRED', {}

    metrics = {
        'current_value': float(current_val),
        'currently_safe': currently_safe,
        'p_exceed_by_horizon': {h: float(p) for h, p in zip(horizons, p_exceeds)},
        'std_by_horizon': {h: float(s) for h, s in zip(horizons, pred_stds)},
        'bimodal_any': any(bimodal_flags),
        'bimodal_horizons': [h for h, b in zip(horizons, bimodal_flags) if b],
        'final_p_exceed': p_exceeds[-1],
    }

    # Uncertainty explosion: std grows >2x from first to last horizon
    if len(pred_stds) >= 2 and pred_stds[0] > 0:
        std_ratio = pred_stds[-1] / pred_stds[0]
        metrics['std_growth_ratio'] = float(std_ratio)
    else:
        std_ratio = 1.0
        metrics['std_growth_ratio'] = 1.0

    # Classification
    if any(bimodal_flags) or std_ratio > 3.0:
        regime = 'UNCERTAIN'
    elif currently_safe and p_exceeds[-1] > 0.5:
        regime = 'TIPPING_DANGEROUS'
    elif not currently_safe and p_exceeds[-1] < 0.5:
        regime = 'TIPPING_RECOVERY'
    elif currently_safe:
        regime = 'STABLE_SAFE'
    else:
        regime = 'STABLE_UNSAFE'

    # Override: if safe but rapid p_exceed jump
    if currently_safe and len(p_exceeds) >= 2:
        if p_exceeds[0] < 0.1 and p_exceeds[-1] > 0.3:
            regime = 'TIPPING_DANGEROUS'

    return regime, metrics


def run(args):
    print("=" * 60)
    print("GENESIS Tipping Point Detection")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    output_dir = project_root / "results" / "tipping_points"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, ckpt_stats = load_model(project_root / args.ckpt, device)

    # Prefer canonical array-format stats (same pattern as 06/08).
    canonical_stats_path = project_root / "data" / "processed" / "normalization_stats.json"
    if canonical_stats_path.exists():
        norm_stats = load_stats(canonical_stats_path)
    elif isinstance(ckpt_stats, dict) and 'chem_mean' in ckpt_stats:
        norm_stats = ckpt_stats
    else:
        raise FileNotFoundError(
            "No array-format normalization_stats.json found; ckpt stats are "
            "not in canonical format."
        )

    # Load groundwater stations
    data_path = project_root / args.data
    df = pd.read_parquet(data_path)
    if 'water_type' in df.columns:
        df = df[df['water_type'].str.contains('Groundwater', case=False, na=False)]
    if 'station_id' in df.columns and 'year' in df.columns:
        df = df.sort_values('year').groupby('station_id').last().reset_index()
    print(f"Groundwater stations: {len(df):,}")

    dataset = StationDataset(df, norm_stats)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        collate_fn=station_collate, shuffle=False
    )

    horizons = [5, 10, 15, 20]
    print(f"Time horizons: {horizons} years")
    print(f"Samples per station per horizon: {args.n_samples}")

    # Process all stations
    all_station_results = []
    station_idx = 0

    for batch_idx, batch in enumerate(loader):
        B = batch['param_ids'].shape[0]
        horizon_data = generate_multi_horizon(
            model, batch, device, norm_stats,
            horizons, args.n_samples, args.ddim_steps,
        )

        for b in range(B):
            if station_idx >= len(df):
                break
            row = df.iloc[station_idx]
            result = {
                'station_id': row.get('station_id', f'station_{station_idx}'),
                'lat': float(row.get('lat', 0)),
                'lon': float(row.get('lon', 0)),
                'country': row.get('country', 'Unknown'),
            }

            # Classify for each contaminant
            for param, threshold in WHO_THRESHOLDS.items():
                current_val = row.get(param, np.nan)
                h_preds = {}
                for h in horizons:
                    station_h_preds = horizon_data[h][b]
                    if param in station_h_preds:
                        h_preds[h] = station_h_preds[param]

                regime, metrics = classify_station(current_val, h_preds, threshold)
                result[f'{param}_regime'] = regime
                result[f'{param}_current'] = float(current_val) if pd.notna(current_val) else None
                if metrics:
                    result[f'{param}_final_p_exceed'] = metrics.get('final_p_exceed', None)
                    result[f'{param}_std_growth'] = metrics.get('std_growth_ratio', None)
                    result[f'{param}_bimodal'] = metrics.get('bimodal_any', False)

            all_station_results.append(result)
            station_idx += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {station_idx}/{len(df)} stations")

    results_df = pd.DataFrame(all_station_results)
    results_df.to_csv(output_dir / 'station_tipping_analysis.csv', index=False)

    # Summary
    print(f"\n{'='*60}")
    print("TIPPING POINT SUMMARY")
    print(f"{'='*60}")

    summary = {'total_stations': len(results_df)}

    for param in WHO_THRESHOLDS:
        col = f'{param}_regime'
        if col not in results_df.columns:
            continue
        counts = results_df[col].value_counts().to_dict()
        print(f"\n  {param} (WHO: {WHO_THRESHOLDS[param]}):")
        for regime in ['STABLE_SAFE', 'STABLE_UNSAFE', 'TIPPING_DANGEROUS',
                       'TIPPING_RECOVERY', 'UNCERTAIN', 'NO_DATA', 'NO_PRED']:
            n = counts.get(regime, 0)
            if n > 0:
                print(f"    {regime:<22s}: {n:>5d} ({n/len(results_df)*100:.1f}%)")
        summary[param] = counts

    # Key headline numbers
    print(f"\n{'='*60}")
    print("HEADLINE FINDINGS")
    print(f"{'='*60}")
    for param in ['As', 'F', 'NO3']:
        col = f'{param}_regime'
        if col in results_df.columns:
            n_tipping = (results_df[col] == 'TIPPING_DANGEROUS').sum()
            n_uncertain = (results_df[col] == 'UNCERTAIN').sum()
            n_safe = (results_df[col] == 'STABLE_SAFE').sum()
            total_risk = n_tipping + n_uncertain
            pct = total_risk / max(n_safe + n_tipping + n_uncertain, 1) * 100
            print(f"  {param}: {n_tipping} tipping + {n_uncertain} uncertain = "
                  f"{total_risk} at risk ({pct:.1f}%)")

    # Save summary
    with open(output_dir / 'summary_stats.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Figures
    print("\nGenerating figures...")
    _setup_style()

    # 1. World map by regime
    for param in ['As', 'F']:
        col = f'{param}_regime'
        if col not in results_df.columns:
            continue
        mask = results_df[col].isin(['STABLE_SAFE', 'STABLE_UNSAFE',
                                      'TIPPING_DANGEROUS', 'TIPPING_RECOVERY', 'UNCERTAIN'])
        plot_df = results_df[mask].copy()
        if len(plot_df) == 0:
            continue

        regime_colors = {
            'STABLE_SAFE': '#2ecc71',
            'STABLE_UNSAFE': '#95a5a6',
            'TIPPING_DANGEROUS': '#e74c3c',
            'TIPPING_RECOVERY': '#3498db',
            'UNCERTAIN': '#f39c12',
        }

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_facecolor('#f0f8ff')

        for regime, color in regime_colors.items():
            subset = plot_df[plot_df[col] == regime]
            if len(subset) > 0:
                ax.scatter(subset['lon'], subset['lat'], c=color, s=10,
                           alpha=0.7, label=f'{regime} ({len(subset)})',
                           edgecolors='none')

        ax.legend(fontsize=9, loc='lower left', frameon=True, markerscale=2)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'GENESIS {param} Regime Classification — Tipping Point Map',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        fig.savefig(output_dir / f'{param}_regime_map.png', dpi=DPI)
        fig.savefig(output_dir / f'{param}_regime_map.pdf', dpi=DPI)
        plt.close()
        print(f"  Saved {param}_regime_map.png/pdf")

    # 2. Country bar chart
    for param in ['As']:
        col = f'{param}_regime'
        if col not in results_df.columns:
            continue
        risky = results_df[results_df[col].isin(['TIPPING_DANGEROUS', 'UNCERTAIN'])]
        if len(risky) == 0:
            continue

        country_counts = risky['country'].value_counts().head(15)

        fig, ax = plt.subplots(figsize=(8, 6))
        country_counts.plot(kind='barh', ax=ax, color='#e74c3c', edgecolor='white')
        ax.set_xlabel('Number of At-Risk Stations')
        ax.set_title(f'Top Countries: {param} Tipping Point Stations',
                     fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(output_dir / f'{param}_country_risk.png', dpi=DPI)
        plt.close()
        print(f"  Saved {param}_country_risk.png")

    # 3. Scatter: current vs predicted
    for param in ['As']:
        curr_col = f'{param}_current'
        pred_col = f'{param}_final_p_exceed'
        regime_col = f'{param}_regime'
        if curr_col not in results_df.columns or pred_col not in results_df.columns:
            continue
        mask = results_df[curr_col].notna() & results_df[pred_col].notna()
        plot_df = results_df[mask].copy()
        if len(plot_df) == 0:
            continue

        regime_colors = {
            'STABLE_SAFE': '#2ecc71', 'STABLE_UNSAFE': '#95a5a6',
            'TIPPING_DANGEROUS': '#e74c3c', 'TIPPING_RECOVERY': '#3498db',
            'UNCERTAIN': '#f39c12',
        }

        fig, ax = plt.subplots(figsize=(8, 7))
        for regime, color in regime_colors.items():
            sub = plot_df[plot_df[regime_col] == regime]
            if len(sub) > 0:
                ax.scatter(sub[curr_col], sub[pred_col], c=color, s=15,
                           alpha=0.6, label=regime, edgecolors='none')

        ax.axvline(WHO_THRESHOLDS[param], color='red', linestyle='--',
                   label=f'WHO limit ({WHO_THRESHOLDS[param]})')
        ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5,
                   label='50% exceedance')
        ax.set_xlabel(f'Current {param} (µg/L)', fontsize=12)
        ax.set_ylabel(f'P({param} > threshold) in 2035', fontsize=12)
        ax.set_title(f'Current Concentration vs Future Risk: {param}',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, markerscale=2)
        ax.set_xscale('symlog', linthresh=1)
        plt.tight_layout()
        fig.savefig(output_dir / f'{param}_current_vs_risk.png', dpi=DPI)
        plt.close()
        print(f"  Saved {param}_current_vs_risk.png")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GENESIS Tipping Point Detection")
    parser.add_argument('--ckpt', type=str, default='checkpoints/diffusion_base/best.pt')
    parser.add_argument('--data', type=str,
                        default='data/processed/all_water_vectors_stage1.parquet')
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    run(args)
