"""
GENESIS Paper 5 — Step 6: Zero-Shot Transfer Evaluation
=========================================================
Evaluates trained diffusion model on:
  1. Held-out test set (in-distribution)
  2. Bangladesh groundwater (out-of-distribution transfer)
  3. Per-parameter analysis (which elements predict best?)
  4. Uncertainty quantification (multi-sample generation)

Generates figures for the paper.

Usage:
  python 06_evaluate_transfer.py --ckpt checkpoints/diffusion_base/best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import erfinv
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Subset

import sys
sys.path.insert(0, str(Path(__file__).parent))

from model.genesis_encoder import GENESISForMGM, GENESIS_PARAMS, PARAM_TO_ID
from model.latent_diffusion import GENESISLatentDiffusion
from model.temporal_dataset import (
    TemporalPairDataset, temporal_collate_fn,
    load_temporal_pairs_dataset, load_bd_holdout_dataset,
)
from model.dataset import load_stats


def load_model(ckpt_path, device):
    """Load trained diffusion model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder_size = ckpt['encoder_size']
    d_latent = ckpt['d_latent']
    diffusion_steps = ckpt['diffusion_steps']
    n_layers = ckpt['n_layers']
    n_heads = ckpt['n_heads']
    norm_stats = ckpt['norm_stats']

    encoder = GENESISForMGM(encoder_size)
    model = GENESISLatentDiffusion(
        pretrained_encoder=encoder,
        d_latent=d_latent,
        diffusion_steps=diffusion_steps,
        n_denoising_layers=n_layers,
        n_heads=n_heads,
        freeze_encoder=True,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, norm_stats, encoder_size


def predict_and_compare(model, dataloader, device, norm_stats, n_samples=10, ddim_steps=50):
    """
    Generate multiple future samples and compare with actual future.

    Returns:
        results: list of dicts with per-sample metrics
    """
    chem_mean = np.asarray(norm_stats['chem_mean'], dtype=np.float32)
    chem_std = np.asarray(norm_stats['chem_std'], dtype=np.float32)
    chem_log = np.asarray(norm_stats['chem_log_mask'], dtype=bool)
    results = []

    with torch.no_grad():
        for batch in dataloader:
            current = {k: v.to(device) for k, v in batch['current'].items()}
            future = {k: v.to(device) for k, v in batch['future'].items()}
            metadata = batch['metadata'].to(device)
            delta_t = batch['delta_t'].to(device)
            batch_size = current['param_ids'].shape[0]

            # Generate multiple samples
            all_preds = model.generate_fast(
                current, metadata, delta_t,
                n_samples=n_samples, ddim_steps=ddim_steps
            )

            # For each sample in batch
            for b in range(batch_size):
                sample_result = {
                    'delta_t': delta_t[b, 0].item(),
                    'params': {},
                }

                for idx_p, p in enumerate(GENESIS_PARAMS):
                    pid = PARAM_TO_ID[p]
                    # Check if this param exists in future
                    future_mask = (future['param_ids'][b] == pid) & ~future['padding_mask'][b]
                    if not future_mask.any():
                        continue

                    # Get actual future value (denormalized)
                    target_norm = future['values'][b][future_mask][0].item()
                    mean = float(chem_mean[idx_p])
                    std = float(chem_std[idx_p])
                    target_raw = target_norm * std + mean
                    if chem_log[idx_p]:
                        target_raw = float(np.expm1(target_raw))

                    # Get all predicted values (denormalized)
                    pred_vals = []
                    for sample_preds in all_preds:
                        if p in sample_preds:
                            pred_norm = sample_preds[p][b].item()
                            pred_raw = pred_norm * std + mean
                            if chem_log[idx_p]:
                                pred_raw = float(np.expm1(pred_raw))
                            pred_vals.append(pred_raw)

                    if pred_vals:
                        pred_array = np.array(pred_vals)
                        # ddof=1 (sample std) to avoid ~5% under-estimation at
                        # n_samples=10 — matters for coverage/ECE calibration.
                        pstd = (float(np.std(pred_array, ddof=1))
                                if pred_array.size > 1 else 0.0)
                        sample_result['params'][p] = {
                            'target': target_raw,
                            'pred_mean': float(np.mean(pred_array)),
                            'pred_std': pstd,
                            'pred_median': float(np.median(pred_array)),
                            'mae': float(np.abs(np.mean(pred_array) - target_raw)),
                            'all_preds': pred_vals,
                        }

                if sample_result['params']:
                    results.append(sample_result)

    return results


# WHO / EPA / regulatory thresholds in raw units (μg/L or mg/L depending on param).
# Exceedance of these is the decision-relevant binary for groundwater safety — AUC
# on these is the metric regulators actually use, not raw MAE.
WHO_THRESHOLDS = {
    'As': 10.0,     # μg/L WHO drinking water limit
    'Fe': 300.0,    # μg/L WHO aesthetic
    'Mn': 100.0,    # μg/L WHO health
    'F':  1.5,      # mg/L WHO
    'U':  30.0,     # μg/L WHO
    'NO3': 50.0,    # mg/L WHO (as NO3)
    'PO4': 0.1,     # mg/L eutrophication threshold
    'pH': None,     # non-threshold; handled separately if needed
}


def _auc_binary(y_true_bin, y_score):
    """ROC-AUC via Mann-Whitney U with average ranks (handles ties correctly).

    Returns NaN if only one class is present or arrays are empty.
    Equivalent to sklearn.metrics.roc_auc_score; inlined to avoid sklearn dep.
    """
    y_true_bin = np.asarray(y_true_bin, dtype=bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true_bin.sum())
    n_neg = int((~y_true_bin).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = rankdata(y_score, method='average')
    sum_ranks_pos = float(ranks[y_true_bin].sum())
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _crps_ensemble(samples, target):
    """CRPS for a finite ensemble (1D samples array, scalar target).

    Uses the unbiased / "fair" estimator (Gneiting & Raftery 2007):
        CRPS = (1/n) Σ |x_i - y|  -  (1 / (2·n·(n-1))) Σ_{i≠j} |x_i - x_j|

    For small ensembles the bias between this and the 1/n² form is
    non-trivial (~10% at n=10), so use the fair form for reported metrics.
    """
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    if n < 2:
        # Degenerate: point estimate → CRPS reduces to |x - y|
        return float(np.abs(samples[0] - target)) if n == 1 else float('nan')
    term1 = float(np.mean(np.abs(samples - target)))
    # Full pairwise matrix (includes i=j zeros); subtract nothing since 0s
    # don't contribute. Divide by n*(n-1) (off-diagonal pair count), then /2.
    pair_sum = float(np.sum(np.abs(samples[:, None] - samples[None, :])))
    term2 = pair_sum / (2.0 * n * (n - 1))
    return term1 - term2


def _bootstrap_ci(values, stat_fn, n_boot=1000, alpha=0.05, rng=None):
    """Percentile-bootstrap CI. Returns (lo, hi); NaN if <5 samples."""
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) < 5:
        return (float('nan'), float('nan'))
    rng = rng if rng is not None else np.random.default_rng(42)
    n = len(values)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = stat_fn(values[idx])
    lo = float(np.percentile(stats, 100 * (alpha / 2)))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def _bootstrap_auc_ci(y_bin, p_exceed, n_boot=1000, alpha=0.05, rng=None):
    """Bootstrap CI for AUC. Resamples paired (label, score)."""
    y_bin = np.asarray(y_bin, dtype=bool)
    p_exceed = np.asarray(p_exceed, dtype=np.float64)
    valid = ~np.isnan(p_exceed)
    y_bin = y_bin[valid]; p_exceed = p_exceed[valid]
    if len(y_bin) < 5 or y_bin.all() or (~y_bin).all():
        return (float('nan'), float('nan'))
    rng = rng if rng is not None else np.random.default_rng(42)
    n = len(y_bin)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_bin[idx]; sc = p_exceed[idx]
        if yb.all() or (~yb).all():
            continue
        stats.append(_auc_binary(yb, sc))
    if len(stats) < 10:
        return (float('nan'), float('nan'))
    return (float(np.percentile(stats, 100 * (alpha / 2))),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


def compute_metrics(results, n_boot=1000):
    """Compute aggregate metrics from prediction results.

    Per-parameter metrics:
        - MAE, RMSE, R² (raw), R² (log1p space), sMAPE
        - coverage_50, coverage_90 (empirical from sample quantiles)
        - CRPS (proper scoring rule for probabilistic forecasts)
        - AUC on WHO-threshold exceedance (if target param has a threshold)
        - Reliability bins for calibration curve
        - Bootstrap 95% CIs on MAE and AUC (n_boot=1000 by default)
    """
    per_param = {p: {
        'targets': [], 'pred_means': [], 'pred_stds': [],
        'all_samples': [], 'abs_err': [],
    } for p in GENESIS_PARAMS}

    for r in results:
        for p, vals in r['params'].items():
            per_param[p]['targets'].append(vals['target'])
            per_param[p]['pred_means'].append(vals['pred_mean'])
            per_param[p]['pred_stds'].append(vals['pred_std'])
            per_param[p]['all_samples'].append(vals.get('all_preds', []))
            per_param[p]['abs_err'].append(vals['mae'])

    summary = {}
    for p in GENESIS_PARAMS:
        d = per_param[p]
        n = len(d['targets'])
        if n == 0:
            continue
        y = np.asarray(d['targets'], dtype=np.float64)
        yhat = np.asarray(d['pred_means'], dtype=np.float64)
        sd = np.asarray(d['pred_stds'], dtype=np.float64)

        abs_err = np.abs(yhat - y)
        sq_err = (yhat - y) ** 2
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(sq_err)))
        # Bootstrap CIs on MAE — reviewers expect these on small eval sets (n~224)
        rng = np.random.default_rng(42)
        mae_lo, mae_hi = _bootstrap_ci(abs_err, np.mean, n_boot=n_boot, rng=rng)

        # R² raw
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

        # R² in log1p space (heavy-tailed chem is better compared there)
        if np.all(y >= 0) and np.all(yhat >= 0):
            yl = np.log1p(y)
            yhl = np.log1p(yhat)
            ss_res_l = float(np.sum((yl - yhl) ** 2))
            ss_tot_l = float(np.sum((yl - yl.mean()) ** 2))
            r2_log = float(1.0 - ss_res_l / ss_tot_l) if ss_tot_l > 0 else float('nan')
        else:
            r2_log = float('nan')

        # sMAPE — scale-invariant, handles orders-of-magnitude range
        denom = (np.abs(y) + np.abs(yhat)) / 2.0
        smape_mask = denom > 1e-12
        smape = (float(np.mean(np.abs(y[smape_mask] - yhat[smape_mask]) / denom[smape_mask]) * 100.0)
                 if smape_mask.any() else float('nan'))

        # Gaussian CI coverage (analytical — assumes sample dist ~ Normal)
        lo90 = yhat - 1.645 * sd; hi90 = yhat + 1.645 * sd
        lo50 = yhat - 0.674 * sd; hi50 = yhat + 0.674 * sd
        cov90 = float(np.mean((y >= lo90) & (y <= hi90)))
        cov50 = float(np.mean((y >= lo50) & (y <= hi50)))

        # CRPS — proper scoring rule, uses the full sample ensemble
        crps_vals = []
        for samples, target in zip(d['all_samples'], y):
            if samples:
                crps_vals.append(_crps_ensemble(samples, target))
        crps = float(np.nanmean(crps_vals)) if crps_vals else float('nan')

        # AUC on WHO threshold exceedance — the decision-relevant metric.
        # Score = ensemble exceedance probability (fraction of diffusion samples
        # above the threshold). This uses the full probabilistic prediction,
        # not just the mean — which is the main advantage of a diffusion model
        # over a point-estimate baseline.
        thresh = WHO_THRESHOLDS.get(p)
        auc = float('nan')
        exceed_rate = float('nan')
        brier = float('nan')
        auc_ci = (float('nan'), float('nan'))
        if thresh is not None and thresh > 0:
            y_bin = y > thresh
            exceed_rate = float(y_bin.mean())
            p_exceed = np.full(n, np.nan, dtype=np.float64)
            for i, samples in enumerate(d['all_samples']):
                if samples:
                    arr = np.asarray(samples, dtype=np.float64)
                    p_exceed[i] = float(np.mean(arr > thresh))
            valid = ~np.isnan(p_exceed)
            if valid.sum() >= 2 and y_bin[valid].any() and (~y_bin[valid]).any():
                auc = _auc_binary(y_bin[valid], p_exceed[valid])
                brier = float(np.mean((p_exceed[valid] - y_bin[valid].astype(float)) ** 2))
                auc_ci = _bootstrap_auc_ci(y_bin[valid], p_exceed[valid], n_boot=n_boot, rng=rng)
            # Insufficient-data cases leave auc/brier as NaN — do not fall back
            # to predicted-mean AUC, which would silently mix metric definitions.

        # Calibration: expected vs observed coverage across nominal levels
        # (used for reliability diagrams; kept as raw arrays)
        nominal = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        observed = []
        for q in nominal:
            z = float(np.sqrt(2) * erfinv(q))  # half-width multiplier
            lo = yhat - z * sd
            hi = yhat + z * sd
            observed.append(float(np.mean((y >= lo) & (y <= hi))))
        ece = float(np.mean(np.abs(np.array(observed) - nominal)))  # expected calibration error

        summary[p] = {
            'mae': mae,
            'mae_ci95': [mae_lo, mae_hi],
            'rmse': rmse,
            'r2': r2,
            'r2_log': r2_log,
            'smape_pct': smape,
            'coverage_50': cov50,
            'coverage_90': cov90,
            'crps': crps,
            'auc_who': auc,
            'auc_who_ci95': [auc_ci[0], auc_ci[1]],
            'brier_who': brier,
            'who_threshold': thresh,
            'exceedance_rate': exceed_rate,
            'calibration': {
                'nominal': nominal.tolist(),
                'observed': observed,
                'ece': ece,
            },
            'n_samples': n,
        }

    return summary


# Published param counts per variant (matches configs in model/genesis_encoder.py).
# Ablation rows (keys ending _nopt) are excluded from the scaling fit.
MODEL_PARAM_COUNTS = {
    'small': 1.47e6,
    'base':  9.0e6,
    'large': 48.0e6,
}


def fit_scaling_law(summary_path, out_path, n_boot=1000):
    """Fit a power law MAE(N) = a · N^(-b) per-param over the Small/Base/Large
    variants present in scaling_summary.json, with bootstrap 95% CIs on the
    exponent. Writes scaling_law.json next to the summary.

    Fitting is done in log-log space via least squares; bootstrap resamples the
    variants with replacement (cheap — only 2–3 points).
    """
    try:
        summary = json.loads(Path(summary_path).read_text())
    except Exception:
        return
    variants = [v for v in ('small', 'base', 'large') if v in summary]
    if len(variants) < 2:
        print(f"[scaling] only {len(variants)} variant(s); skipping power-law fit")
        return

    N = np.array([MODEL_PARAM_COUNTS[v] for v in variants], dtype=np.float64)
    logN = np.log(N)
    fits = {}

    # Union of params present in the BD-transfer section across variants
    all_params = set()
    for v in variants:
        all_params.update(summary[v].get('bangladesh_transfer', {}).keys())

    rng = np.random.default_rng(42)
    for p in sorted(all_params):
        maes = []
        for v in variants:
            m = summary[v].get('bangladesh_transfer', {}).get(p, {}).get('mae')
            maes.append(float(m) if m is not None and not np.isnan(m) else np.nan)
        maes = np.asarray(maes, dtype=np.float64)
        valid = np.isfinite(maes) & (maes > 0)
        if valid.sum() < 2:
            continue
        logY = np.log(maes[valid])
        X = logN[valid]
        # Closed-form LS: logY = a - b·logN  →  fit slope m, intercept c
        slope, intercept = np.polyfit(X, logY, 1)
        b = float(-slope)
        a = float(np.exp(intercept))
        # R² of the log-log fit
        yhat = slope * X + intercept
        ss_res = float(np.sum((logY - yhat) ** 2))
        ss_tot = float(np.sum((logY - logY.mean()) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

        # Bootstrap CI on exponent — resample (N, MAE) pairs with replacement.
        # With only 2–3 points this is noisy; report but treat as illustrative.
        n_pts = int(valid.sum())
        if n_pts >= 3:
            b_samples = []
            for _ in range(n_boot):
                idx = rng.integers(0, n_pts, size=n_pts)
                if len(set(idx.tolist())) < 2:
                    continue
                s, _ = np.polyfit(X[idx], logY[idx], 1)
                b_samples.append(-s)
            if len(b_samples) >= 10:
                b_lo = float(np.percentile(b_samples, 2.5))
                b_hi = float(np.percentile(b_samples, 97.5))
            else:
                b_lo = b_hi = float('nan')
        else:
            b_lo = b_hi = float('nan')  # 2-point fit — no CI possible

        fits[p] = {
            'a': a, 'b': b, 'b_ci95': [b_lo, b_hi],
            'r2_loglog': r2,
            'n_variants': n_pts,
            'variants_used': [v for i, v in enumerate(variants) if valid[i]],
            'maes': maes.tolist(),
        }

    out = {
        'model': 'MAE(N) = a * N^(-b)',
        'model_param_counts': {v: MODEL_PARAM_COUNTS[v] for v in variants},
        'per_param': fits,
        'note': ('Power-law fit in log-log space over 2–3 GENESIS variants. '
                 'With only 3 points the exponent CI is illustrative, not inferential.'),
    }
    Path(out_path).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nScaling-law fit saved → {out_path}")
    # Terse table
    print(f"{'Param':>6s} | {'exponent b':>12s} | {'CI95':>22s} | {'R²':>6s} | n")
    print("-" * 70)
    for p, f in fits.items():
        ci = (f"[{f['b_ci95'][0]:+.2f},{f['b_ci95'][1]:+.2f}]"
              if not np.isnan(f['b_ci95'][0]) else "    (2-pt, no CI)   ")
        print(f"{p:>6s} | {f['b']:>+12.3f} | {ci:>22s} | {f['r2_loglog']:>6.3f} | {f['n_variants']}")


def run_evaluation(args):
    print("=" * 70)
    print("GENESIS Zero-Shot Transfer Evaluation")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model, norm_stats, encoder_size = load_model(args.ckpt, device)
    print(f"Loaded model: encoder={encoder_size}")

    data_dir = Path(__file__).parent.parent / "data" / "processed"
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize stats container — ckpts may embed a dict-format stub; prefer
    # the canonical array format from data/processed/normalization_stats.json.
    canonical_stats = data_dir / "normalization_stats.json"
    if canonical_stats.exists() and 'chem_mean' not in norm_stats:
        norm_stats = load_stats(canonical_stats)

    # Load full temporal dataset (wide parquet → tensors).
    # No aux .pt matches final_temporal_pairs.parquet; omit aux here.
    dataset, pairs_df = load_temporal_pairs_dataset(
        data_dir / "final_temporal_pairs.parquet", norm_stats,
    )

    # Split same as training
    n_test = max(100, int(len(dataset) * 0.15))
    n_val = max(100, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val - n_test
    _, _, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Held-out Bangladesh pairs (separate curated file); this is the true
    # zero-shot transfer set, not a country filter on the training pairs.
    bd_ds = None
    bd_pairs_path = data_dir / "genesis_held_out_bd_pairs.pt"
    if bd_pairs_path.exists():
        try:
            bd_ds = load_bd_holdout_dataset(data_dir, norm_stats)
            print(f"\nBangladesh held-out pairs: {len(bd_ds):,}")
        except Exception as e:
            print(f"  (skipping BD held-out: {e})")
    print(f"Total in-distribution pairs: {len(dataset):,}")

    # ---- In-distribution evaluation ----
    print(f"\n{'='*50}")
    print("1. In-Distribution Test Set")
    print(f"{'='*50}")

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=temporal_collate_fn
    )
    id_results = predict_and_compare(
        model, test_loader, device, norm_stats,
        n_samples=args.n_samples, ddim_steps=args.ddim_steps
    )
    id_metrics = compute_metrics(id_results)

    print(f"\nPer-parameter metrics (in-distribution, n={len(id_results)}):")
    for p in GENESIS_PARAMS:
        if p in id_metrics:
            m = id_metrics[p]
            auc_s = f"  AUC_WHO={m['auc_who']:.3f}" if not np.isnan(m.get('auc_who', float('nan'))) else ""
            print(f"  {p:>6s}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
                  f"R²={m['r2']:.3f}  R²log={m['r2_log']:.3f}  "
                  f"sMAPE={m['smape_pct']:.1f}%  "
                  f"CRPS={m['crps']:.4f}  90%CI={m['coverage_90']:.2%}  "
                  f"ECE={m['calibration']['ece']:.3f}{auc_s}  (n={m['n_samples']})")

    # ---- Bangladesh transfer evaluation ----
    bd_metrics = {}
    if bd_ds is not None:
        print(f"\n{'='*50}")
        print("2. Zero-Shot Transfer: Bangladesh")
        print(f"{'='*50}")

        bd_loader = DataLoader(
            bd_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=temporal_collate_fn
        )
        bd_results = predict_and_compare(
            model, bd_loader, device, norm_stats,
            n_samples=args.n_samples, ddim_steps=args.ddim_steps
        )
        bd_metrics = compute_metrics(bd_results)

        print(f"\nPer-parameter metrics (Bangladesh transfer, n={len(bd_results)}):")
        for p in GENESIS_PARAMS:
            if p in bd_metrics:
                m = bd_metrics[p]
                auc_s = f"  AUC_WHO={m['auc_who']:.3f}" if not np.isnan(m.get('auc_who', float('nan'))) else ""
                print(f"  {p:>6s}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
                      f"R²={m['r2']:.3f}  R²log={m['r2_log']:.3f}  "
                      f"sMAPE={m['smape_pct']:.1f}%  "
                      f"CRPS={m['crps']:.4f}  90%CI={m['coverage_90']:.2%}  "
                      f"ECE={m['calibration']['ece']:.3f}{auc_s}  (n={m['n_samples']})")

    # ---- Save results ----
    all_results = {
        'in_distribution': id_metrics,
        'bangladesh_transfer': bd_metrics,
        'config': {
            'encoder_size': encoder_size,
            'checkpoint': str(args.ckpt),
            'n_samples': args.n_samples,
            'ddim_steps': args.ddim_steps,
        }
    }
    # Per-variant file. Ablation runs (checkpoints/diffusion_<size>_nopt/) get
    # a distinct _nopt suffix so they don't overwrite the pretrained eval.
    ckpt_parent = Path(args.ckpt).parent.name
    variant_tag = encoder_size + ("_nopt" if ckpt_parent.endswith("_nopt") else "")
    results_path = output_dir / f"eval_{variant_tag}.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Consolidated scaling summary: merge this variant into the aggregate
    # so running eval for small/base/large produces a single file with all three.
    summary_path = output_dir / "scaling_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            summary = {}
    else:
        summary = {}
    summary[variant_tag] = all_results
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Scaling summary updated: {summary_path}")

    # Power-law fit on the scaling curve (runs from 2+ variants onward; no-op
    # if only one is present). Writes results/scaling_law.json.
    fit_scaling_law(summary_path, output_dir / "scaling_law.json")

    # ---- Summary table for paper ----
    print(f"\n{'='*70}")
    print("PAPER TABLE: Comparative Performance Summary")
    print(f"{'='*70}")
    print(f"{'Parameter':>10s} | {'ID MAE':>8s} | {'BD MAE':>8s} | {'Transfer':>10s}")
    print("-" * 45)
    for p in GENESIS_PARAMS:
        id_mae = id_metrics.get(p, {}).get('mae', float('nan'))
        bd_mae = bd_metrics.get(p, {}).get('mae', float('nan'))
        ratio = bd_mae / id_mae if id_mae > 0 and not np.isnan(bd_mae) else float('nan')
        if not np.isnan(id_mae):
            print(f"{p:>10s} | {id_mae:>8.4f} | "
                  f"{bd_mae:>8.4f} | {ratio:>8.2f}x" if not np.isnan(bd_mae)
                  else f"{p:>10s} | {id_mae:>8.4f} |      N/A |        N/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GENESIS Evaluation")
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to trained diffusion checkpoint')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of future samples to generate per input')
    parser.add_argument('--ddim_steps', type=int, default=50)
    args = parser.parse_args()

    run_evaluation(args)
