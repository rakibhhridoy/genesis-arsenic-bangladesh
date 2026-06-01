"""
GENESIS Paper 5 — Step 8: Global Projection (2025 → 2035)
============================================================
Uses trained diffusion model to predict future geochemical states
for all groundwater stations globally. Identifies stations at risk
of crossing WHO safety thresholds.

Runs on CPU/MPS (Mac) with downloaded checkpoints.

Usage:
  python 08_global_projection.py --ckpt checkpoints/diffusion_base/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent))
from model.genesis_encoder import (
    GENESISForMGM, GENESIS_PARAMS, PARAM_TO_ID, PAD_TOKEN, CLS_TOKEN,
)
from model.latent_diffusion import GENESISLatentDiffusion
from model.dataset import load_stats

# WHO Guidelines / Safety Thresholds (in original measurement units)
WHO_THRESHOLDS = {
    'As': 10.0,       # µg/L
    'F': 1.5,         # mg/L
    'NO3': 50.0,      # mg/L as NO3
    'Fe': 0.3,        # mg/L (aesthetic)
    'Mn': 0.4,        # mg/L
    'U': 30.0,        # µg/L
}

# Style
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


class StationDataset(Dataset):
    """Wraps station data for batch prediction.

    Uses canonical array-format norm_stats (chem_mean / chem_std / chem_log_mask
    indexed by GENESIS_PARAMS position) and applies log1p before z-scoring for
    params flagged in chem_log_mask — matching GENESISDataset's training-time
    normalization exactly. Skipping log1p here would feed the diffusion model
    out-of-distribution inputs.
    """

    def __init__(self, df, norm_stats, max_seq_len=25):
        self.df = df.reset_index(drop=True)
        self.chem_mean = np.asarray(norm_stats['chem_mean'], dtype=np.float32)
        self.chem_std = np.asarray(norm_stats['chem_std'], dtype=np.float32)
        self.chem_log_mask = np.asarray(norm_stats['chem_log_mask'], dtype=bool)
        self.max_seq_len = max_seq_len
        # Remember param position within GENESIS_PARAMS so we index arrays correctly
        self.param_cols = [(i, p) for i, p in enumerate(GENESIS_PARAMS) if p in df.columns]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        param_ids = torch.full((self.max_seq_len,), PARAM_TO_ID[PAD_TOKEN], dtype=torch.long)
        values = torch.zeros(self.max_seq_len, dtype=torch.float32)
        padding_mask = torch.ones(self.max_seq_len, dtype=torch.bool)

        # CLS token
        param_ids[0] = PARAM_TO_ID[CLS_TOKEN]
        values[0] = 0.0
        padding_mask[0] = False

        pos = 1
        for (i, p) in self.param_cols:
            if pos >= self.max_seq_len:
                break
            val = row[p]
            if pd.notna(val):
                raw = float(val)
                if self.chem_log_mask[i]:
                    raw = float(np.log1p(max(raw, 0.0)))
                norm = (raw - float(self.chem_mean[i])) / float(self.chem_std[i])
                param_ids[pos] = PARAM_TO_ID[p]
                values[pos] = norm
                padding_mask[pos] = False
                pos += 1

        lat = float(row.get('lat', 0)) / 90.0
        lon = float(row.get('lon', 0)) / 180.0
        metadata = torch.tensor([lat, lon, 0.0, 0.0], dtype=torch.float32)

        return {
            'param_ids': param_ids,
            'values': values,
            'padding_mask': padding_mask,
            'metadata': metadata,
        }


def station_collate(batch):
    return {
        'param_ids': torch.stack([b['param_ids'] for b in batch]),
        'values': torch.stack([b['values'] for b in batch]),
        'padding_mask': torch.stack([b['padding_mask'] for b in batch]),
        'metadata': torch.stack([b['metadata'] for b in batch]),
    }


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    encoder = GENESISForMGM(ckpt['encoder_size'])
    model = GENESISLatentDiffusion(
        pretrained_encoder=encoder,
        d_latent=ckpt['d_latent'],
        diffusion_steps=ckpt['diffusion_steps'],
        n_denoising_layers=ckpt['n_layers'],
        n_heads=ckpt['n_heads'],
        freeze_encoder=True,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model, ckpt['norm_stats']


def generate_projections(model, dataloader, device, norm_stats, delta_t, n_samples, ddim_steps):
    """Generate multiple future samples for each station.

    Denormalization mirrors training-time normalization: first undo z-score,
    then expm1 for log-transformed params. The model emits normalized log-space
    values; WHO thresholds and downstream analysis are in raw units, so both
    steps are required.
    """
    chem_mean = np.asarray(norm_stats['chem_mean'], dtype=np.float32)
    chem_std = np.asarray(norm_stats['chem_std'], dtype=np.float32)
    chem_log = np.asarray(norm_stats['chem_log_mask'], dtype=bool)

    all_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            current_state = {
                'param_ids': batch['param_ids'].to(device),
                'values': batch['values'].to(device),
                'padding_mask': batch['padding_mask'].to(device),
            }
            metadata = batch['metadata'].to(device)
            B = metadata.shape[0]
            dt = torch.full((B, 1), delta_t, dtype=torch.float32, device=device)

            preds_list = model.generate_fast(
                current_state, metadata, dt,
                n_samples=n_samples, ddim_steps=ddim_steps,
            )

            for b in range(B):
                station_preds = {}
                for i, p in enumerate(GENESIS_PARAMS):
                    mean = float(chem_mean[i])
                    std = float(chem_std[i])
                    is_log = bool(chem_log[i])
                    raw_preds = []
                    for sample in preds_list:
                        if p in sample:
                            pred_norm = sample[p][b].item()
                            pred_raw = pred_norm * std + mean
                            if is_log:
                                pred_raw = float(np.expm1(pred_raw))
                            raw_preds.append(pred_raw)
                    if raw_preds:
                        arr = np.array(raw_preds)
                        station_preds[p] = {
                            'pred_mean': float(np.mean(arr)),
                            'pred_std': float(np.std(arr)),
                            'pred_median': float(np.median(arr)),
                            'pred_p10': float(np.percentile(arr, 10)),
                            'pred_p90': float(np.percentile(arr, 90)),
                            'all_preds': raw_preds,
                        }
                all_results.append(station_preds)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(dataloader)}")

    return all_results


def analyze_threshold_crossings(stations_df, predictions, norm_stats):
    """Identify stations crossing WHO thresholds.

    Raw-space comparison throughout: `stations_df` holds unnormalized mg/L
    (and µg/L for As, U) values, and `predictions` has already been
    denormalized by generate_projections(). WHO_THRESHOLDS are in the same
    raw units.
    """
    results = []

    for idx in range(len(stations_df)):
        row = stations_df.iloc[idx]
        preds = predictions[idx]
        station_result = {
            'station_id': row.get('station_id', f'station_{idx}'),
            'lat': float(row.get('lat', 0)),
            'lon': float(row.get('lon', 0)),
            'country': row.get('country', 'Unknown'),
        }

        for param, threshold in WHO_THRESHOLDS.items():
            current_val = row.get(param, np.nan)
            if param in preds:
                pred = preds[param]
                currently_safe = pd.notna(current_val) and current_val <= threshold
                currently_unsafe = pd.notna(current_val) and current_val > threshold

                # Probability of exceeding threshold
                all_preds = pred['all_preds']
                p_exceed = np.mean([p > threshold for p in all_preds])

                station_result[f'{param}_current'] = float(current_val) if pd.notna(current_val) else None
                station_result[f'{param}_pred_mean'] = pred['pred_mean']
                station_result[f'{param}_pred_std'] = pred['pred_std']
                station_result[f'{param}_p_exceed'] = float(p_exceed)

                if currently_safe and p_exceed > 0.5:
                    station_result[f'{param}_status'] = 'FLIP_TO_UNSAFE'
                elif currently_unsafe and p_exceed < 0.5:
                    station_result[f'{param}_status'] = 'FLIP_TO_SAFE'
                elif currently_safe:
                    station_result[f'{param}_status'] = 'STABLE_SAFE'
                elif currently_unsafe:
                    station_result[f'{param}_status'] = 'STABLE_UNSAFE'
                else:
                    station_result[f'{param}_status'] = 'NO_DATA'
            else:
                station_result[f'{param}_status'] = 'NO_PRED'

        results.append(station_result)

    return pd.DataFrame(results)


def plot_world_map_arsenic(results_df, output_dir):
    """World map of predicted arsenic exceedance probability."""
    _setup_style()

    mask = results_df['As_p_exceed'].notna()
    df = results_df[mask].copy()
    if len(df) == 0:
        print("  No arsenic predictions to plot")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    # Simple world outline
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_facecolor('#f0f8ff')

    # Scatter stations
    sc = ax.scatter(
        df['lon'], df['lat'],
        c=df['As_p_exceed'],
        cmap='RdYlGn_r',
        s=8, alpha=0.7,
        vmin=0, vmax=1,
        edgecolors='none',
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('P(As > 10 µg/L) by 2035', fontsize=11)

    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('GENESIS Projected Arsenic Exceedance Probability (2025 → 2035)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_dir / 'arsenic_world_map.png', dpi=DPI)
    fig.savefig(output_dir / 'arsenic_world_map.pdf', dpi=DPI)
    plt.close()
    print("  Saved arsenic_world_map.png/pdf")


def plot_flip_summary(results_df, output_dir):
    """Bar chart: stations flipping from safe to unsafe by parameter."""
    _setup_style()

    params = list(WHO_THRESHOLDS.keys())
    flip_counts = []
    for p in params:
        col = f'{p}_status'
        if col in results_df.columns:
            flip_counts.append((results_df[col] == 'FLIP_TO_UNSAFE').sum())
        else:
            flip_counts.append(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#9b59b6', '#1abc9c']
    bars = ax.bar(params, flip_counts, color=colors[:len(params)], edgecolor='white')

    for bar, count in zip(bars, flip_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Contaminant', fontsize=12)
    ax.set_ylabel('Number of Stations Flipping Safe → Unsafe', fontsize=11)
    ax.set_title('Stations Predicted to Cross WHO Thresholds by 2035',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_dir / 'threshold_flip_summary.png', dpi=DPI)
    plt.close()
    print("  Saved threshold_flip_summary.png")


def plot_arsenic_change_hist(results_df, output_dir):
    """Histogram of predicted arsenic change."""
    _setup_style()

    mask = results_df['As_pred_mean'].notna() & results_df['As_current'].notna()
    df = results_df[mask].copy()
    if len(df) == 0:
        print("  No arsenic data for histogram")
        return

    delta_as = df['As_pred_mean'] - df['As_current']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delta_as.clip(-50, 50), bins=50, color='#e74c3c', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(delta_as.median(), color='#2c3e50', linestyle='-', linewidth=2,
               label=f'Median: {delta_as.median():.2f} µg/L')

    ax.set_xlabel('Predicted Change in As (µg/L)', fontsize=12)
    ax.set_ylabel('Number of Stations', fontsize=11)
    ax.set_title('Predicted Arsenic Concentration Change (2025 → 2035)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / 'arsenic_change_histogram.png', dpi=DPI)
    plt.close()
    print("  Saved arsenic_change_histogram.png")


def plot_uncertainty_map(results_df, output_dir):
    """Stations colored by prediction uncertainty (std)."""
    _setup_style()

    mask = results_df['As_pred_std'].notna()
    df = results_df[mask].copy()
    if len(df) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_facecolor('#f0f8ff')

    sc = ax.scatter(
        df['lon'], df['lat'],
        c=df['As_pred_std'],
        cmap='magma',
        s=8, alpha=0.7,
        vmin=0,
        edgecolors='none',
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Prediction Uncertainty (σ, µg/L)', fontsize=11)

    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('GENESIS Arsenic Prediction Uncertainty — Potential Tipping Points',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_dir / 'uncertainty_map.png', dpi=DPI)
    fig.savefig(output_dir / 'uncertainty_map.pdf', dpi=DPI)
    plt.close()
    print("  Saved uncertainty_map.png/pdf")


def run(args):
    print("=" * 60)
    print("GENESIS Global Projection (2025 → 2035)")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    output_dir = project_root / "results" / "global_projection"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model: {args.ckpt}")
    model, ckpt_stats = load_model(project_root / args.ckpt, device)

    # Prefer canonical array-format stats; ckpt may embed a legacy dict stub.
    canonical_stats_path = project_root / "data" / "processed" / "normalization_stats.json"
    if canonical_stats_path.exists():
        norm_stats = load_stats(canonical_stats_path)
        print(f"  Using canonical norm stats: {canonical_stats_path}")
    elif isinstance(ckpt_stats, dict) and 'chem_mean' in ckpt_stats:
        norm_stats = ckpt_stats
    else:
        raise FileNotFoundError(
            "No array-format normalization_stats.json found; ckpt stats are "
            "not in canonical format. Run 04_pretrain_genesis.py first."
        )

    # Load groundwater stations
    data_path = project_root / args.data
    print(f"Loading data: {data_path}")
    df = pd.read_parquet(data_path)

    # Filter groundwater only
    if 'water_type' in df.columns:
        df = df[df['water_type'].str.contains('Groundwater', case=False, na=False)]
    print(f"  Groundwater stations: {len(df):,}")

    # Take most recent sample per station
    if 'station_id' in df.columns and 'year' in df.columns:
        df = df.sort_values('year').groupby('station_id').last().reset_index()
    print(f"  Unique stations (most recent): {len(df):,}")

    # Create dataset
    dataset = StationDataset(df, norm_stats)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        collate_fn=station_collate, shuffle=False)

    # Generate predictions
    print(f"\nGenerating {args.n_samples} future samples per station (delta_t={args.delta_t}y)...")
    predictions = generate_projections(
        model, loader, device, norm_stats,
        delta_t=args.delta_t, n_samples=args.n_samples, ddim_steps=args.ddim_steps,
    )

    # Analyze threshold crossings
    print("\nAnalyzing threshold crossings...")
    results_df = analyze_threshold_crossings(df, predictions, norm_stats)

    # Save results
    results_df.to_csv(output_dir / 'station_predictions.csv', index=False)
    print(f"  Saved station_predictions.csv ({len(results_df)} stations)")

    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY: Stations at Risk")
    print(f"{'='*60}")
    for param, threshold in WHO_THRESHOLDS.items():
        status_col = f'{param}_status'
        if status_col in results_df.columns:
            n_flip = (results_df[status_col] == 'FLIP_TO_UNSAFE').sum()
            n_safe = (results_df[status_col] == 'STABLE_SAFE').sum()
            n_unsafe = (results_df[status_col] == 'STABLE_UNSAFE').sum()
            total = n_safe + n_flip
            pct = n_flip / max(total, 1) * 100
            print(f"  {param:>4s} (>{threshold:g}): {n_flip} stations flip safe→unsafe "
                  f"({pct:.1f}% of currently safe)")

    # Generate figures
    print("\nGenerating figures...")
    plot_world_map_arsenic(results_df, output_dir)
    plot_flip_summary(results_df, output_dir)
    plot_arsenic_change_hist(results_df, output_dir)
    plot_uncertainty_map(results_df, output_dir)

    # Save summary JSON
    summary = {
        'total_stations': len(results_df),
        'delta_t_years': args.delta_t,
        'n_samples': args.n_samples,
    }
    for param in WHO_THRESHOLDS:
        col = f'{param}_status'
        if col in results_df.columns:
            summary[f'{param}_flip_to_unsafe'] = int((results_df[col] == 'FLIP_TO_UNSAFE').sum())
            summary[f'{param}_stable_safe'] = int((results_df[col] == 'STABLE_SAFE').sum())
            summary[f'{param}_stable_unsafe'] = int((results_df[col] == 'STABLE_UNSAFE').sum())

    with open(output_dir / 'projection_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GENESIS Global Projection")
    parser.add_argument('--ckpt', type=str, default='checkpoints/diffusion_base/best.pt')
    parser.add_argument('--data', type=str,
                        default='data/processed/all_water_vectors_stage1.parquet')
    parser.add_argument('--delta_t', type=float, default=10.0,
                        help='Projection horizon in years')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Future samples per station')
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    run(args)
