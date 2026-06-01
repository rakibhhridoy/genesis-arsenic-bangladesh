"""
Temporal Pair Dataset for GENESIS Latent Diffusion fine-tuning.
Converts temporal geochemical pairs (t0 → t1) into encoder-ready inputs.

Both t0 and t1 get the same aux features (site doesn't move) and wbt.
No masking — both states are fully visible for diffusion training.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .genesis_encoder import (
    GENESIS_PARAMS, AUX_FEATURES, PARAM_TO_ID,
    PAD_TOKEN, CLS_TOKEN, WBT_TO_ID,
    NUM_PARAMS, NUM_AUX,
)
from .dataset import MAX_SEQ_LEN


class TemporalPairDataset(Dataset):
    """
    Each sample yields:
      - current_state (t0): tokenized geochemical vector + aux + wbt
      - future_state  (t1): tokenized geochemical vector + aux + wbt
      - metadata: [lat, lon, 0, 0]  (depth/elevation not in pair data)
      - delta_t: time gap in years
    """

    def __init__(self, t0_values, t1_values, gap_years, norm_stats,
                 meta_df=None, aux_values=None):
        """
        Args:
            t0_values:  (N, 20) float32 — chemistry at time 0
            t1_values:  (N, 20) float32 — chemistry at time 1
            gap_years:  (N,) float32 — years between t0 and t1
            norm_stats: dict from compute_normalization_stats()
            meta_df:    DataFrame with 'lat', 'lon', 'water_body_type'
            aux_values: (N, 18) float32, or None
        """
        self.n_samples = t0_values.shape[0]
        self.gap_years = gap_years

        # Pre-normalize chemistry
        chem_mean = norm_stats['chem_mean']
        chem_std = norm_stats['chem_std']
        chem_log = norm_stats['chem_log_mask']

        def normalize_chem(raw):
            c = raw.copy()
            for i in range(NUM_PARAMS):
                if chem_log[i]:
                    c[:, i] = np.log1p(np.maximum(c[:, i], 0))
            return (c - chem_mean) / chem_std

        self.t0_norm = normalize_chem(t0_values)
        self.t1_norm = normalize_chem(t1_values)
        self.t0_avail = ~np.isnan(t0_values)
        self.t1_avail = ~np.isnan(t1_values)

        # Aux features
        if aux_values is not None and 'aux_mean' in norm_stats:
            aux = aux_values.copy()
            self.aux_norm = (aux - norm_stats['aux_mean']) / norm_stats['aux_std']
            self.aux_avail = ~np.isnan(aux_values)
            self.has_aux = True
        else:
            self.has_aux = False

        # Metadata
        if meta_df is not None:
            self.lat = np.nan_to_num(meta_df['lat'].values.astype(np.float32), nan=0.0) if 'lat' in meta_df.columns else np.zeros(self.n_samples, dtype=np.float32)
            self.lon = np.nan_to_num(meta_df['lon'].values.astype(np.float32), nan=0.0) if 'lon' in meta_df.columns else np.zeros(self.n_samples, dtype=np.float32)
            if 'water_body_type' in meta_df.columns:
                self.wbt_ids = meta_df['water_body_type'].map(
                    lambda x: WBT_TO_ID.get(x, WBT_TO_ID['unknown'])
                ).values.astype(np.int64)
                self.has_wbt = True
            else:
                self.has_wbt = False
        else:
            self.lat = np.zeros(self.n_samples, dtype=np.float32)
            self.lon = np.zeros(self.n_samples, dtype=np.float32)
            self.has_wbt = False

    def __len__(self):
        return self.n_samples

    def _tokenize(self, chem_norm, chem_avail, idx):
        """Build token sequence for one timepoint (no masking)."""
        param_ids = torch.full((MAX_SEQ_LEN,), PARAM_TO_ID[PAD_TOKEN], dtype=torch.long)
        values = torch.zeros(MAX_SEQ_LEN, dtype=torch.float32)
        padding_mask = torch.ones(MAX_SEQ_LEN, dtype=torch.bool)

        pos = 0

        # CLS
        param_ids[pos] = PARAM_TO_ID[CLS_TOKEN]
        padding_mask[pos] = False
        pos += 1

        # Chemistry tokens
        for i, p in enumerate(GENESIS_PARAMS):
            if pos >= MAX_SEQ_LEN:
                break
            if chem_avail[i]:
                param_ids[pos] = PARAM_TO_ID[p]
                values[pos] = float(chem_norm[i])
                padding_mask[pos] = False
                pos += 1

        # Aux tokens (same for t0 and t1 — site-level features)
        if self.has_aux:
            for i, p in enumerate(AUX_FEATURES):
                if pos >= MAX_SEQ_LEN:
                    break
                if self.aux_avail[idx, i]:
                    param_ids[pos] = PARAM_TO_ID[p]
                    values[pos] = float(self.aux_norm[idx, i])
                    padding_mask[pos] = False
                    pos += 1

        return {'param_ids': param_ids, 'values': values, 'padding_mask': padding_mask}

    def __getitem__(self, idx):
        t0 = self._tokenize(self.t0_norm[idx], self.t0_avail[idx], idx)
        t1 = self._tokenize(self.t1_norm[idx], self.t1_avail[idx], idx)

        metadata = torch.tensor([
            self.lat[idx] / 90.0,
            self.lon[idx] / 180.0,
            0.0,  # depth placeholder
            0.0,  # elevation placeholder
        ], dtype=torch.float32)

        delta_t = torch.tensor([self.gap_years[idx]], dtype=torch.float32)

        return {
            'current': t0,
            'future': t1,
            'metadata': metadata,
            'delta_t': delta_t,
        }


def temporal_collate_fn(batch):
    """Collate temporal pairs."""
    current = {k: torch.stack([b['current'][k] for b in batch]) for k in batch[0]['current']}
    future = {k: torch.stack([b['future'][k] for b in batch]) for k in batch[0]['future']}
    metadata = torch.stack([b['metadata'] for b in batch])
    delta_t = torch.stack([b['delta_t'] for b in batch])

    return {
        'current': current,
        'future': future,
        'metadata': metadata,
        'delta_t': delta_t,
    }


def _wide_pairs_to_arrays(df: pd.DataFrame):
    """Extract (t0, t1, gap_years) arrays from wide-format temporal pairs parquet.

    Expects columns like 'As_t0','As_t1', ..., 'gap_years' for each GENESIS_PARAMS entry.
    Missing parameters become NaN.
    """
    n = len(df)
    t0 = np.full((n, NUM_PARAMS), np.nan, dtype=np.float32)
    t1 = np.full((n, NUM_PARAMS), np.nan, dtype=np.float32)
    for i, p in enumerate(GENESIS_PARAMS):
        c0, c1 = f"{p}_t0", f"{p}_t1"
        if c0 in df.columns:
            t0[:, i] = df[c0].to_numpy(dtype=np.float32)
        if c1 in df.columns:
            t1[:, i] = df[c1].to_numpy(dtype=np.float32)
    gap = df['gap_years'].to_numpy(dtype=np.float32) if 'gap_years' in df.columns \
          else np.zeros(n, dtype=np.float32)
    return t0, t1, gap


def load_temporal_pairs_dataset(parquet_path, norm_stats, aux_path=None):
    """Build a TemporalPairDataset from a wide-format pair parquet.

    Returns the dataset plus the source DataFrame (so callers can subset by
    country/source columns for transfer eval splits).
    """
    df = pd.read_parquet(parquet_path)
    t0, t1, gap = _wide_pairs_to_arrays(df)
    aux = None
    if aux_path is not None and Path(aux_path).exists():
        aux = torch.load(aux_path, weights_only=True).numpy().astype(np.float32)
        if aux.shape[0] != len(df):
            raise ValueError(
                f"aux rows {aux.shape[0]} != pairs rows {len(df)} for {aux_path}"
            )
    ds = TemporalPairDataset(
        t0_values=t0, t1_values=t1, gap_years=gap,
        norm_stats=norm_stats, meta_df=df, aux_values=aux,
    )
    return ds, df


def load_bd_holdout_dataset(data_dir, norm_stats):
    """Build a TemporalPairDataset from the Bangladesh held-out .pt bundle.

    Layout (from data prep step):
      genesis_held_out_bd_pairs.pt        — dict: t0, t1, gap_years (scalar), ...
      genesis_held_out_bd_pairs_aux.pt    — (N, NUM_AUX) tensor
      genesis_held_out_bd_pairs_meta.parquet — lat/lon/depth + aux columns
    """
    data_dir = Path(data_dir)
    bundle = torch.load(data_dir / "genesis_held_out_bd_pairs.pt", weights_only=True)
    t0 = bundle['t0'].numpy().astype(np.float32)
    t1 = bundle['t1'].numpy().astype(np.float32)
    gap = bundle.get('gap_years')
    if gap is None or gap.numel() == 1:
        # Single scalar — broadcast to per-row
        scalar = float(gap.item()) if gap is not None else 0.0
        gap = np.full(t0.shape[0], scalar, dtype=np.float32)
    else:
        gap = gap.numpy().astype(np.float32)

    aux_path = data_dir / "genesis_held_out_bd_pairs_aux.pt"
    aux = (torch.load(aux_path, weights_only=True).numpy().astype(np.float32)
           if aux_path.exists() else None)

    meta_path = data_dir / "genesis_held_out_bd_pairs_meta.parquet"
    meta = pd.read_parquet(meta_path) if meta_path.exists() else None

    return TemporalPairDataset(
        t0_values=t0, t1_values=t1, gap_years=gap,
        norm_stats=norm_stats, meta_df=meta, aux_values=aux,
    )
