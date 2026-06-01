"""
GENESIS Dataset — Converts geochemical vectors + auxiliary features
to tokenized inputs with random masking for MGM pretraining.

Token sequence layout:
  [CLS] + [chem tokens (variable)] + [aux tokens (variable)] + [wbt] + [PAD...]

Only chemistry tokens are masked. Aux features, water body type, and CLS
are always visible — they serve as conditioning context.
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

from .genesis_encoder import (
    GENESIS_PARAMS, AUX_FEATURES, WATER_BODY_TYPES, WBT_TO_ID,
    PARAM_TO_ID, PAD_TOKEN, MASK_TOKEN, CLS_TOKEN,
    NUM_PARAMS, NUM_AUX,
)

# Max sequence: CLS(1) + chem(20) + aux(18) + wbt(1) = 40, pad to 44
MAX_SEQ_LEN = 44

# Parameters that should be log-transformed before normalization
# (heavy right-skew: concentrations spanning orders of magnitude)
LOG_TRANSFORM_PARAMS = {'As', 'Fe', 'Mn', 'PO4', 'F', 'U', 'NO3',
                        'Ca', 'Mg', 'Na', 'K', 'Cl', 'HCO3', 'SO4',
                        'SiO2', 'DOC', 'EC', 'TDS'}

# Small constant to avoid log(0)
LOG_EPS = 1e-6


def compute_normalization_stats(chem_tensor, aux_tensor=None):
    """Compute per-feature mean/std from pretrain tensors.

    Args:
        chem_tensor: (N, 20) chemistry values with NaN for missing
        aux_tensor:  (N, 18) aux feature values with NaN for missing, or None

    Returns:
        dict with 'chem_mean', 'chem_std', 'chem_log_mask',
                   'aux_mean', 'aux_std' (if aux provided)
    """
    stats = {}

    # Chemistry: log-transform skewed params, then z-score
    chem_means = np.zeros(NUM_PARAMS, dtype=np.float32)
    chem_stds = np.ones(NUM_PARAMS, dtype=np.float32)
    chem_log_mask = np.zeros(NUM_PARAMS, dtype=bool)

    for i, p in enumerate(GENESIS_PARAMS):
        col = chem_tensor[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue

        if p in LOG_TRANSFORM_PARAMS:
            valid = np.log1p(np.maximum(valid, 0))  # log(1+x), clamp negatives
            chem_log_mask[i] = True

        chem_means[i] = float(np.mean(valid))
        chem_stds[i] = max(float(np.std(valid)), 1e-8)

    stats['chem_mean'] = chem_means
    stats['chem_std'] = chem_stds
    stats['chem_log_mask'] = chem_log_mask

    # Auxiliary features: z-score (no log transform — already well-scaled)
    if aux_tensor is not None:
        aux_means = np.zeros(NUM_AUX, dtype=np.float32)
        aux_stds = np.ones(NUM_AUX, dtype=np.float32)

        for i in range(NUM_AUX):
            col = aux_tensor[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                continue
            aux_means[i] = float(np.mean(valid))
            aux_stds[i] = max(float(np.std(valid)), 1e-8)

        stats['aux_mean'] = aux_means
        stats['aux_std'] = aux_stds

    return stats


def save_stats(stats, path):
    """Save normalization stats to JSON."""
    out = {}
    for k, v in stats.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)


def load_stats(path):
    """Load normalization stats from JSON."""
    with open(path) as f:
        raw = json.load(f)
    stats = {}
    for k, v in raw.items():
        if isinstance(v, list):
            stats[k] = np.array(v, dtype=np.float32 if 'mask' not in k else bool)
        else:
            stats[k] = v
    return stats


class GENESISDataset(Dataset):
    """
    Dataset for GENESIS Masked Geochemical Modeling pretraining.

    Each sample is a variable-length token sequence:
      [CLS] + available_chem_tokens + available_aux_tokens + [wbt] + [PAD...]

    During training, 15-30% of chemistry tokens are randomly masked.
    Aux tokens and wbt are never masked.
    """

    def __init__(self, chem_values, norm_stats, meta_df=None,
                 aux_values=None, mask_ratio=0.2):
        """
        Args:
            chem_values:  (N, 20) float32 array, NaN = missing
            norm_stats:   dict from compute_normalization_stats()
            meta_df:      DataFrame with 'lat', 'lon', 'water_body_type' columns
            aux_values:   (N, 18) float32 array, NaN = missing. Or None.
            mask_ratio:   fraction of available chem params to mask
        """
        self.n_samples = chem_values.shape[0]
        self.mask_ratio = mask_ratio

        # Normalization stats (applied lazily in __getitem__ to save RAM).
        # On 8 GB Macs, materializing a full normalized copy of a 2M-row tensor
        # consumed ~160 MB and contributed to MPS OOM crashes.
        self.chem_mean = np.asarray(norm_stats['chem_mean'], dtype=np.float32)
        self.chem_std = np.asarray(norm_stats['chem_std'], dtype=np.float32)
        self.chem_log_mask = np.asarray(norm_stats['chem_log_mask'], dtype=bool)

        # Keep raw values + NaN mask only
        self.chem_raw = np.ascontiguousarray(chem_values, dtype=np.float32)
        self.chem_avail = ~np.isnan(self.chem_raw)

        # Auxiliary features (raw; normalized lazily)
        if aux_values is not None and 'aux_mean' in norm_stats:
            self.aux_raw = np.ascontiguousarray(aux_values, dtype=np.float32)
            self.aux_mean = np.asarray(norm_stats['aux_mean'], dtype=np.float32)
            self.aux_std = np.asarray(norm_stats['aux_std'], dtype=np.float32)
            self.aux_avail = ~np.isnan(self.aux_raw)
            self.has_aux = True
        else:
            self.has_aux = False

        # Water body type
        if meta_df is not None and 'water_body_type' in meta_df.columns:
            self.wbt_ids = meta_df['water_body_type'].map(
                lambda x: WBT_TO_ID.get(x, WBT_TO_ID['unknown'])
            ).values.astype(np.int64)
            self.has_wbt = True
        else:
            self.has_wbt = False

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """Build token sequence and apply random masking."""
        # Collect available chemistry tokens (normalize lazily per row)
        chem_tokens = []  # (position_in_param_list, param_id, normalized_value)
        chem_row = self.chem_raw[idx]
        for i, p in enumerate(GENESIS_PARAMS):
            if self.chem_avail[idx, i]:
                pid = PARAM_TO_ID[p]
                raw = chem_row[i]
                if self.chem_log_mask[i]:
                    raw = np.log1p(max(raw, 0.0))
                val = float((raw - self.chem_mean[i]) / self.chem_std[i])
                chem_tokens.append((i, pid, val))

        # Collect available aux tokens
        aux_tokens = []
        if self.has_aux:
            aux_row = self.aux_raw[idx]
            for i, p in enumerate(AUX_FEATURES):
                if self.aux_avail[idx, i]:
                    pid = PARAM_TO_ID[p]
                    val = float((aux_row[i] - self.aux_mean[i]) / self.aux_std[i])
                    aux_tokens.append((pid, val))

        # Total sequence: CLS + chem + aux + wbt
        n_chem = len(chem_tokens)
        n_aux = len(aux_tokens)
        n_extra = 1 + (1 if self.has_wbt else 0)  # CLS + optional wbt
        n_tokens = n_extra + n_chem + n_aux

        # Initialize tensors
        param_ids = torch.full((MAX_SEQ_LEN,), PARAM_TO_ID[PAD_TOKEN], dtype=torch.long)
        values = torch.zeros(MAX_SEQ_LEN, dtype=torch.float32)
        original_param_ids = torch.full((MAX_SEQ_LEN,), PARAM_TO_ID[PAD_TOKEN], dtype=torch.long)
        original_values = torch.zeros(MAX_SEQ_LEN, dtype=torch.float32)
        masked_positions = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        padding_mask = torch.ones(MAX_SEQ_LEN, dtype=torch.bool)

        pos = 0

        # CLS token
        param_ids[pos] = PARAM_TO_ID[CLS_TOKEN]
        original_param_ids[pos] = PARAM_TO_ID[CLS_TOKEN]
        padding_mask[pos] = False
        pos += 1

        # Chemistry tokens (these will be partially masked)
        chem_start = pos
        for (param_idx, pid, val) in chem_tokens:
            if pos >= MAX_SEQ_LEN:
                break
            original_param_ids[pos] = pid
            original_values[pos] = val
            param_ids[pos] = pid
            values[pos] = val
            padding_mask[pos] = False
            pos += 1
        chem_end = pos

        # Random masking of chemistry tokens only
        n_maskable = chem_end - chem_start
        if n_maskable > 0:
            n_mask = max(1, int(n_maskable * self.mask_ratio))
            n_mask = min(n_mask, n_maskable)
            mask_indices = np.random.choice(
                range(chem_start, chem_end), size=n_mask, replace=False
            )
            for mi in mask_indices:
                param_ids[mi] = PARAM_TO_ID[MASK_TOKEN]
                values[mi] = 0.0
                masked_positions[mi] = True

        # Auxiliary tokens (never masked)
        for (pid, val) in aux_tokens:
            if pos >= MAX_SEQ_LEN:
                break
            param_ids[pos] = pid
            original_param_ids[pos] = pid
            values[pos] = val
            original_values[pos] = val
            padding_mask[pos] = False
            pos += 1

        # Water body type token (never masked)
        if self.has_wbt and pos < MAX_SEQ_LEN:
            wbt_id = self.wbt_ids[idx]
            # Encode as a special value: param_id is not in vocab,
            # so we use CLS embedding + wbt embedding additively.
            # Simpler: store wbt_id as the value, use CLS as placeholder param.
            # Actually, wbt is handled via the embedding's wbt_emb.
            # For now, store as CLS token with wbt value for the forward pass
            # to pick up. We'll handle this properly in the collate.
            # TODO: Add WBT as a proper vocab token
            param_ids[pos] = PARAM_TO_ID[CLS_TOKEN]  # placeholder
            values[pos] = float(wbt_id)
            original_param_ids[pos] = PARAM_TO_ID[CLS_TOKEN]
            original_values[pos] = float(wbt_id)
            padding_mask[pos] = False
            pos += 1

        return {
            'param_ids': param_ids,
            'values': values,
            'padding_mask': padding_mask,
            'original_param_ids': original_param_ids,
            'masked_positions': masked_positions,
            'original_values': original_values,
        }


def collate_fn(batch):
    """Collate function for DataLoader — simple stack."""
    return {
        key: torch.stack([b[key] for b in batch])
        for key in batch[0].keys()
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    proc = Path(__file__).parent.parent.parent / "data" / "processed"
    pt_path = proc / "genesis_pretrain.pt"
    meta_path = proc / "genesis_pretrain_meta.parquet"

    if pt_path.exists():
        import pandas as pd

        # Load tensors
        chem = torch.load(pt_path, weights_only=True).numpy()
        meta = pd.read_parquet(meta_path)
        print(f"Loaded: {chem.shape[0]:,} vectors, {chem.shape[1]} params")

        # Compute stats
        stats = compute_normalization_stats(chem)
        print(f"\nNormalization stats (first 5 params):")
        for i, p in enumerate(GENESIS_PARAMS[:5]):
            log = " (log)" if stats['chem_log_mask'][i] else ""
            print(f"  {p:6s}: mean={stats['chem_mean'][i]:8.3f}, "
                  f"std={stats['chem_std'][i]:8.3f}{log}")

        # Save stats
        stats_path = proc / "normalization_stats.json"
        save_stats(stats, stats_path)
        print(f"\nSaved stats → {stats_path}")

        # Create dataset
        ds = GENESISDataset(chem, stats, meta_df=meta, mask_ratio=0.2)
        print(f"\nDataset: {len(ds):,} samples, max_seq_len={MAX_SEQ_LEN}")

        # Test a sample
        sample = ds[0]
        n_real = (~sample['padding_mask']).sum().item()
        n_masked = sample['masked_positions'].sum().item()
        print(f"Sample 0: {n_real} tokens, {n_masked} masked")

        # Test DataLoader
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
        batch = next(iter(loader))
        print(f"\nBatch shapes:")
        for k, v in batch.items():
            print(f"  {k}: {v.shape}")

        print("\nDataset test PASSED!")
    else:
        print(f"Data not found at {pt_path}")
