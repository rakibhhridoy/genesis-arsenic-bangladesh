#!/usr/bin/env python3
"""
GENESIS Paper 5 — Prepare Training Data
=========================================
Joins auxiliary features from GEE extraction to the GENESIS database,
computes normalization statistics, and exports final training-ready tensors.

Steps:
  1. Load aux_features.parquet (from GEE) and join to all vectors by lat/lon
  2. Compute normalization stats (chem + aux) from pretrain set only
  3. Export joined tensors: chem (N,20) + aux (N,18) + meta parquet
  4. Save normalization_stats.json

Usage:
  python src/06_prepare_training.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model.genesis_encoder import GENESIS_PARAMS, AUX_FEATURES, NUM_PARAMS, NUM_AUX
from model.dataset import compute_normalization_stats, save_stats

PAPER5 = Path(__file__).resolve().parent.parent
PROC_DIR = PAPER5 / "data" / "processed"

AUX_PATH = PROC_DIR / "aux_features.parquet"
STATS_PATH = PROC_DIR / "normalization_stats.json"


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def join_aux_to_meta(meta_path, aux_df):
    """Spatial join: match each vector's lat/lon to nearest aux feature row.

    Uses rounded lat/lon (6 decimal places) for exact matching first,
    then nearest-neighbor for remaining unmatched.
    """
    meta = pd.read_parquet(meta_path)
    n_total = len(meta)

    # Round for matching (same precision as unique_sites.csv export)
    meta['_lat_r'] = meta['lat'].round(6)
    meta['_lon_r'] = meta['lon'].round(6)
    aux_df['_lat_r'] = aux_df['lat'].round(6)
    aux_df['_lon_r'] = aux_df['lon'].round(6)

    # Exact merge on rounded lat/lon
    aux_cols = [c for c in AUX_FEATURES if c in aux_df.columns]
    aux_merge = aux_df[['_lat_r', '_lon_r'] + aux_cols].drop_duplicates(
        subset=['_lat_r', '_lon_r']
    )

    merged = meta.merge(aux_merge, on=['_lat_r', '_lon_r'], how='left')

    n_matched = merged[aux_cols[0]].notna().sum()
    _log(f"  Exact match: {n_matched:,} / {n_total:,} ({100*n_matched/n_total:.1f}%)")

    # For unmatched (NaN lat/lon or no GEE coverage), leave aux as NaN
    n_unmatched = n_total - n_matched
    if n_unmatched > 0:
        _log(f"  Unmatched: {n_unmatched:,} (will have NaN aux features)")

    # Clean up temp columns
    merged = merged.drop(columns=['_lat_r', '_lon_r'])

    return merged, aux_cols


def main():
    _log("GENESIS Training Data Preparation")
    _log("=" * 50)

    # Check aux features exist
    if not AUX_PATH.exists():
        _log(f"ERROR: {AUX_PATH} not found. Run 05_extract_aux_features_gee.py first.")
        return

    # Load aux features
    aux_df = pd.read_parquet(AUX_PATH)
    _log(f"Loaded aux features: {len(aux_df):,} sites, {len(aux_df.columns)} columns")

    # --- PRETRAIN ---
    _log("\n1. Joining aux features to pretrain vectors...")
    pretrain_meta_path = PROC_DIR / "genesis_pretrain_meta.parquet"
    pretrain_chem_path = PROC_DIR / "genesis_pretrain.pt"

    pretrain_meta, aux_cols = join_aux_to_meta(pretrain_meta_path, aux_df)
    pretrain_chem = torch.load(pretrain_chem_path, weights_only=True).numpy()
    _log(f"  Pretrain: {pretrain_chem.shape[0]:,} vectors × {pretrain_chem.shape[1]} chem params")

    # Extract aux array in correct column order
    pretrain_aux = np.full((len(pretrain_meta), NUM_AUX), np.nan, dtype=np.float32)
    for i, feat in enumerate(AUX_FEATURES):
        if feat in pretrain_meta.columns:
            pretrain_aux[:, i] = pretrain_meta[feat].values.astype(np.float32)

    # --- COMPUTE NORMALIZATION STATS (pretrain only) ---
    _log("\n2. Computing normalization statistics...")
    stats = compute_normalization_stats(pretrain_chem, pretrain_aux)
    save_stats(stats, STATS_PATH)
    _log(f"  Saved → {STATS_PATH}")

    _log("  Chemistry stats (log-transformed params marked):")
    for i, p in enumerate(GENESIS_PARAMS):
        log_flag = " (log)" if stats['chem_log_mask'][i] else ""
        _log(f"    {p:6s}: mean={stats['chem_mean'][i]:8.3f}, std={stats['chem_std'][i]:8.3f}{log_flag}")

    _log("  Auxiliary stats:")
    for i, p in enumerate(AUX_FEATURES):
        _log(f"    {p:25s}: mean={stats['aux_mean'][i]:10.3f}, std={stats['aux_std'][i]:10.3f}")

    # --- SAVE JOINED TENSORS ---
    _log("\n3. Saving training-ready tensors...")

    # Pretrain
    torch.save(torch.from_numpy(pretrain_aux), PROC_DIR / "genesis_pretrain_aux.pt")
    pretrain_meta.to_parquet(PROC_DIR / "genesis_pretrain_meta.parquet", index=False)
    _log(f"  genesis_pretrain_aux.pt: {pretrain_aux.shape}")

    # --- TEMPORAL PAIRS ---
    _log("\n4. Joining aux to temporal pairs...")
    pairs_meta_path = PROC_DIR / "genesis_temporal_pairs_meta.parquet"
    if pairs_meta_path.exists():
        pairs_meta, _ = join_aux_to_meta(pairs_meta_path, aux_df)
        pairs_aux = np.full((len(pairs_meta), NUM_AUX), np.nan, dtype=np.float32)
        for i, feat in enumerate(AUX_FEATURES):
            if feat in pairs_meta.columns:
                pairs_aux[:, i] = pairs_meta[feat].values.astype(np.float32)
        torch.save(torch.from_numpy(pairs_aux), PROC_DIR / "genesis_temporal_pairs_aux.pt")
        pairs_meta.to_parquet(pairs_meta_path, index=False)
        _log(f"  genesis_temporal_pairs_aux.pt: {pairs_aux.shape}")

    # --- HELD-OUT BD ---
    _log("\n5. Joining aux to Bangladesh held-out...")
    bd_meta_path = PROC_DIR / "genesis_held_out_bd_meta.parquet"
    if bd_meta_path.exists():
        bd_meta, _ = join_aux_to_meta(bd_meta_path, aux_df)
        bd_aux = np.full((len(bd_meta), NUM_AUX), np.nan, dtype=np.float32)
        for i, feat in enumerate(AUX_FEATURES):
            if feat in bd_meta.columns:
                bd_aux[:, i] = bd_meta[feat].values.astype(np.float32)
        torch.save(torch.from_numpy(bd_aux), PROC_DIR / "genesis_held_out_bd_aux.pt")
        bd_meta.to_parquet(bd_meta_path, index=False)
        _log(f"  genesis_held_out_bd_aux.pt: {bd_aux.shape}")

    # --- BD PAIRS ---
    bd_pairs_meta_path = PROC_DIR / "genesis_held_out_bd_pairs_meta.parquet"
    if bd_pairs_meta_path.exists():
        bd_pairs_meta, _ = join_aux_to_meta(bd_pairs_meta_path, aux_df)
        bd_pairs_aux = np.full((len(bd_pairs_meta), NUM_AUX), np.nan, dtype=np.float32)
        for i, feat in enumerate(AUX_FEATURES):
            if feat in bd_pairs_meta.columns:
                bd_pairs_aux[:, i] = bd_pairs_meta[feat].values.astype(np.float32)
        torch.save(torch.from_numpy(bd_pairs_aux), PROC_DIR / "genesis_held_out_bd_pairs_aux.pt")
        bd_pairs_meta.to_parquet(bd_pairs_meta_path, index=False)
        _log(f"  genesis_held_out_bd_pairs_aux.pt: {bd_pairs_aux.shape}")

    # --- COVERAGE REPORT ---
    _log("\n6. Aux feature coverage:")
    for i, feat in enumerate(AUX_FEATURES):
        n_valid = np.sum(~np.isnan(pretrain_aux[:, i]))
        pct = 100 * n_valid / len(pretrain_aux)
        _log(f"  {feat:25s}: {n_valid:>9,} / {len(pretrain_aux):,} ({pct:.1f}%)")

    _log("\nDone. Training data is ready.")
    _log(f"  Chem tensor:  {PROC_DIR / 'genesis_pretrain.pt'}")
    _log(f"  Aux tensor:   {PROC_DIR / 'genesis_pretrain_aux.pt'}")
    _log(f"  Metadata:     {PROC_DIR / 'genesis_pretrain_meta.parquet'}")
    _log(f"  Norm stats:   {STATS_PATH}")


if __name__ == "__main__":
    main()
