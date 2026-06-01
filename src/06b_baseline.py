"""
GENESIS Paper 5 — Step 6b: Baseline (pretrain-augmented, non-learned features)
==============================================================================
Journal-strength baseline that uses the SAME 2 M pretrain corpus as GENESIS,
but via hand-crafted kNN features instead of a learned transformer encoder.

For each (t0, t1) pair:
    features = [ chem_t0 (20, log1p where applicable),
                 gap_years,
                 aux (18, from BD meta or nearest pretrain site),
                 neighborhood chem mean/median/std (60, log1p where applicable)
                   — computed over the k nearest pretrain sites in (lat, lon) ]
    target   = delta (raw, per-param)

Trains one XGBoost regressor per chem parameter on the 3.5 K temporal pairs
(excluding Bangladesh); evaluates on the 224 Bangladesh held-out pairs.
Writes results/baseline_eval.json in the same schema as eval_<size>.json so
the head-to-head against GENESIS is a direct JSON diff.

Runs on Mac CPU; ~2–5 min on an M1. No GPU, no RunPod needed.

Usage:
    python src/06b_baseline.py
    python src/06b_baseline.py --k 100 --xgb_rounds 500
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from xgboost import XGBRegressor

GENESIS_PARAMS = ['As','Fe','Mn','PO4','F','U','NO3','pH','Eh','EC','TDS',
                  'Ca','Mg','Na','K','Cl','HCO3','SO4','SiO2','DOC']
AUX_COLS = ['elevation_m','slope_deg','sand_pct','clay_pct','soc_g_per_kg',
            'soil_ph','bulk_density_kg_m3','landcover_esa','population_density',
            'ndvi_mean','ndvi_std','precip_mm_yr','temp_mean_C','aet_mm_yr',
            'pet_mm_yr','aridity_index','twsa_mean_cm','twsa_trend_cm_yr']


def log1p_safe(x, log_mask):
    """Apply log1p to columns flagged in log_mask; clip negatives to 0."""
    out = x.copy()
    if log_mask.any():
        cols = np.where(log_mask)[0]
        sub = out[..., cols]
        sub = np.where(np.isnan(sub), np.nan, np.log1p(np.maximum(sub, 0.0)))
        out[..., cols] = sub
    return out


def build_knn_features(lat_q, lon_q, tree, chem, aux, log_mask, k=50):
    """Spatial kNN features from the 2 M corpus.

    Returns (n_q, 20*3 + 18) = (n_q, 78). Rows with non-finite query coords
    get all-zero features (should not occur for BD; guard for safety).
    """
    lat_q = np.asarray(lat_q, dtype=np.float64)
    lon_q = np.asarray(lon_q, dtype=np.float64)
    ok = np.isfinite(lat_q) & np.isfinite(lon_q)
    idx = np.zeros((len(lat_q), k), dtype=np.int64)
    if ok.any():
        _, idx_ok = tree.query(np.column_stack([lat_q[ok], lon_q[ok]]), k=k)
        idx[ok] = idx_ok
    neigh_chem = chem[idx]                       # (n_q, k, 20)
    neigh_chem = log1p_safe(neigh_chem, log_mask)
    # nanmean/median/std over the k neighbors; some cols may be fully-NaN
    with np.errstate(invalid='ignore'):
        chem_mean = np.nanmean(neigh_chem, axis=1)
        chem_med = np.nanmedian(neigh_chem, axis=1)
        chem_std = np.nanstd(neigh_chem, axis=1)
    aux_mean = np.nanmean(aux[idx], axis=1)      # (n_q, 18)
    feats = np.concatenate([chem_mean, chem_med, chem_std, aux_mean], axis=1)
    # Replace remaining NaNs (fully-missing neighbor cols) with 0 — XGB handles
    # missing natively, but np.nan in features works too; keep as-is.
    return feats.astype(np.float32)


def pair_features(chem_t0, gap_years, aux_per_pair, knn_feats, log_mask):
    """Concatenate per-pair features. chem_t0 is log1p-transformed where masked."""
    c0 = log1p_safe(chem_t0, log_mask).astype(np.float32)
    g = np.asarray(gap_years, dtype=np.float32).reshape(-1, 1)
    return np.concatenate([c0, g, aux_per_pair.astype(np.float32), knn_feats], axis=1)


def load_pretrain_corpus(data_dir):
    chem = torch.load(data_dir / "genesis_pretrain.pt", weights_only=False).numpy()
    aux = torch.load(data_dir / "genesis_pretrain_aux.pt", weights_only=False).numpy()
    meta = pd.read_parquet(data_dir / "genesis_pretrain_meta.parquet")
    return chem.astype(np.float32), aux.astype(np.float32), meta


def load_train_pairs(data_dir):
    """3.5 K global temporal pairs (wide format)."""
    df = pd.read_parquet(data_dir / "final_temporal_pairs.parquet")
    t0 = np.stack([df[f"{p}_t0"].to_numpy(np.float32) for p in GENESIS_PARAMS], axis=1)
    delta = np.stack([df[f"delta_{p}"].to_numpy(np.float32) for p in GENESIS_PARAMS], axis=1)
    gap = df['gap_years'].to_numpy(np.float32) if 'gap_years' in df.columns else \
          (df['year_t1'].to_numpy(np.float32) - df['year_t0'].to_numpy(np.float32))
    lat = df['lat'].to_numpy(np.float32)
    lon = df['lon'].to_numpy(np.float32)
    country = df['country'].astype(str).to_numpy() if 'country' in df.columns else np.array(['?']*len(df))
    return t0, delta, gap, lat, lon, country


def load_bd_pairs(data_dir):
    """224 Bangladesh held-out pairs."""
    bundle = torch.load(data_dir / "genesis_held_out_bd_pairs.pt", weights_only=True)
    t0 = bundle['t0'].numpy().astype(np.float32)
    delta = bundle['delta'].numpy().astype(np.float32)
    gap = bundle['gap_years']
    gap = (np.full(t0.shape[0], float(gap.item()), dtype=np.float32)
           if gap.numel() == 1 else gap.numpy().astype(np.float32))
    meta = pd.read_parquet(data_dir / "genesis_held_out_bd_pairs_meta.parquet")
    lat = meta['lat'].to_numpy(np.float32)
    lon = meta['lon'].to_numpy(np.float32)
    aux = torch.load(data_dir / "genesis_held_out_bd_pairs_aux.pt", weights_only=True).numpy().astype(np.float32)
    return t0, delta, gap, lat, lon, aux


def aux_from_nearest(lat_q, lon_q, tree, aux_corpus):
    """For training pairs where aux isn't stored — take aux of single nearest pretrain site."""
    _, idx = tree.query(np.column_stack([lat_q, lon_q]), k=1)
    return aux_corpus[idx]


def train_and_eval(args):
    print("=" * 70)
    print("Baseline B: pretrain-augmented kNN + XGBoost")
    print("=" * 70)

    data_dir = Path(__file__).parent.parent / "data" / "processed"
    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_stats = json.loads((data_dir / "normalization_stats.json").read_text())
    log_mask = np.asarray(norm_stats['chem_log_mask'], dtype=bool)
    assert log_mask.shape == (20,)

    print("\nLoading 2 M pretrain corpus ...")
    pt_chem, pt_aux, pt_meta = load_pretrain_corpus(data_dir)
    pt_lat = pt_meta['lat'].to_numpy(np.float32)
    pt_lon = pt_meta['lon'].to_numpy(np.float32)
    print(f"  chem: {pt_chem.shape}  aux: {pt_aux.shape}")

    # Drop any rows with NaN lat/lon — KDTree rejects non-finite coords
    finite = np.isfinite(pt_lat) & np.isfinite(pt_lon)
    if not finite.all():
        print(f"  dropping {(~finite).sum()} pretrain rows with non-finite lat/lon")
        pt_lat = pt_lat[finite]; pt_lon = pt_lon[finite]
        pt_chem = pt_chem[finite]; pt_aux = pt_aux[finite]

    print("Building spatial kNN index ...")
    tree = cKDTree(np.column_stack([pt_lat, pt_lon]))

    print("Loading training pairs ...")
    tr_t0, tr_delta, tr_gap, tr_lat, tr_lon, tr_country = load_train_pairs(data_dir)
    # Exclude Bangladesh from training so we don't leak the transfer set
    bd_mask = np.array([c.upper().startswith('BGD') or c.upper() == 'BANGLADESH'
                        or c.upper() == 'BD' for c in tr_country])
    if bd_mask.any():
        print(f"  dropping {bd_mask.sum()} Bangladesh rows from train split")
        keep = ~bd_mask
        tr_t0, tr_delta, tr_gap = tr_t0[keep], tr_delta[keep], tr_gap[keep]
        tr_lat, tr_lon = tr_lat[keep], tr_lon[keep]
    print(f"  train rows: {len(tr_t0)}")

    print("Loading BD held-out pairs ...")
    bd_t0, bd_delta, bd_gap, bd_lat, bd_lon, bd_aux = load_bd_pairs(data_dir)
    print(f"  BD rows: {len(bd_t0)}")

    print(f"\nComputing kNN features (k={args.k}) ...")
    tr_knn = build_knn_features(tr_lat, tr_lon, tree, pt_chem, pt_aux, log_mask, k=args.k)
    bd_knn = build_knn_features(bd_lat, bd_lon, tree, pt_chem, pt_aux, log_mask, k=args.k)

    tr_aux = aux_from_nearest(tr_lat, tr_lon, tree, pt_aux)
    print(f"  train feature dim: chem_t0=20 gap=1 aux=18 knn={tr_knn.shape[1]}")

    X_train = pair_features(tr_t0, tr_gap, tr_aux, tr_knn, log_mask)
    X_bd = pair_features(bd_t0, bd_gap, bd_aux, bd_knn, log_mask)
    print(f"  X_train: {X_train.shape}  X_bd: {X_bd.shape}")

    per_param = {}
    print("\nTraining + evaluating per-parameter XGBoost ...")
    for i, p in enumerate(GENESIS_PARAMS):
        y_tr = tr_delta[:, i]
        y_bd = bd_delta[:, i]
        tr_ok = ~np.isnan(y_tr)
        bd_ok = ~np.isnan(y_bd)
        if tr_ok.sum() < 50 or bd_ok.sum() < 5:
            print(f"  {p:>6s}: SKIP (train_n={tr_ok.sum()}, bd_n={bd_ok.sum()})")
            continue
        model = XGBRegressor(
            n_estimators=args.xgb_rounds, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method='hist',
            n_jobs=args.n_jobs, verbosity=0, objective='reg:squarederror',
        )
        model.fit(X_train[tr_ok], y_tr[tr_ok])
        y_hat = model.predict(X_bd[bd_ok])
        y_true = y_bd[bd_ok]
        mae = float(np.mean(np.abs(y_hat - y_true)))
        rmse = float(np.sqrt(np.mean((y_hat - y_true) ** 2)))
        per_param[p] = {
            'mae': mae, 'rmse': rmse,
            'coverage_90': None,  # XGB point-predict; no CI. Extend w/ quantile reg if needed.
            'n_samples': int(bd_ok.sum()),
            'n_train': int(tr_ok.sum()),
        }
        print(f"  {p:>6s}: MAE={mae:.4f}  RMSE={rmse:.4f}  "
              f"(train_n={tr_ok.sum()}, bd_n={bd_ok.sum()})")

    all_results = {
        'bangladesh_transfer': per_param,
        'in_distribution': {},  # not split for baseline; fill if desired
        'config': {
            'model': 'XGBoost per-param',
            'features': 'chem_t0 + gap + aux(nearest) + kNN chem mean/median/std (log1p)',
            'k': args.k,
            'xgb_rounds': args.xgb_rounds,
            'n_train_rows': int(len(tr_t0)),
            'n_bd_rows': int(len(bd_t0)),
        },
    }
    out_path = out_dir / "baseline_eval.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {out_path}")

    # Head-to-head if GENESIS eval files exist
    print(f"\n{'='*70}")
    print("HEAD-TO-HEAD (Bangladesh MAE): baseline vs GENESIS")
    print(f"{'='*70}")
    print(f"{'Param':>6s} | {'Baseline':>10s} | {'Small':>10s} | {'Base':>10s} | {'Large':>10s}")
    print("-" * 60)
    genesis = {}
    for sz in ('small', 'base', 'large'):
        fp = out_dir / f"eval_{sz}.json"
        if fp.exists():
            try:
                genesis[sz] = json.loads(fp.read_text()).get('bangladesh_transfer', {})
            except Exception:
                genesis[sz] = {}
    for p in GENESIS_PARAMS:
        row = [f"{per_param.get(p, {}).get('mae', np.nan):>10.4f}" if p in per_param else f"{'--':>10s}"]
        for sz in ('small', 'base', 'large'):
            v = genesis.get(sz, {}).get(p, {}).get('mae')
            row.append(f"{v:>10.4f}" if v is not None else f"{'--':>10s}")
        print(f"{p:>6s} | " + " | ".join(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=50, help='kNN neighbors from 2 M corpus')
    ap.add_argument('--xgb_rounds', type=int, default=300)
    ap.add_argument('--n_jobs', type=int, default=-1)
    args = ap.parse_args()
    train_and_eval(args)


if __name__ == '__main__':
    main()
