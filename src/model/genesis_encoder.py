"""
GENESIS Encoder — Flexible Variable-Length Geochemical Transformer
===================================================================
Each geochemical parameter is treated as a token:
  token = param_ID_embedding + value_embedding + metadata_embedding

The encoder processes variable-length sets of parameters using
self-attention, learning inter-element relationships (As-Fe-PO4
coupling, F-Ca-pH equilibria, etc.) from data.

Input token types:
  1. Chemistry tokens (20): As, Fe, Mn, ... DOC — variable-length, maskable
  2. Auxiliary feature tokens (18): elevation, slope, soil, climate, GRACE, NDVI
  3. Metadata tokens (4): lat, lon, depth, elevation (always present)
  4. Water body type token (1): categorical {groundwater, river, lake, unknown}
  5. CLS token (1): aggregation token for latent representation

Three model sizes for scaling law analysis:
  GENESIS-Small: 4 layers, 4 heads, d_model=128
  GENESIS-Base:  8 layers, 8 heads, d_model=256
  GENESIS-Large: 12 layers, 12 heads, d_model=512
"""

import torch
import torch.nn as nn
import math


# ============================================================
# GENESIS parameter vocabulary
# ============================================================

GENESIS_PARAMS = [
    'As', 'Fe', 'Mn', 'PO4', 'F', 'U', 'NO3',
    'pH', 'Eh', 'EC', 'TDS',
    'Ca', 'Mg', 'Na', 'K', 'Cl', 'HCO3', 'SO4',
    'SiO2', 'DOC',
]

# Auxiliary feature names (from GEE extraction, order matters for tensor columns)
AUX_FEATURES = [
    'elevation_m', 'slope_deg',
    'sand_pct', 'clay_pct', 'soc_g_per_kg', 'soil_ph', 'bulk_density_kg_m3',
    'landcover_esa', 'population_density',
    'ndvi_mean', 'ndvi_std',
    'precip_mm_yr', 'temp_mean_C', 'aet_mm_yr', 'pet_mm_yr', 'aridity_index',
    'twsa_mean_cm', 'twsa_trend_cm_yr',
]

# Water body type categories
WATER_BODY_TYPES = ['groundwater', 'river', 'lake', 'unknown']
WBT_TO_ID = {wbt: i for i, wbt in enumerate(WATER_BODY_TYPES)}

# Special tokens
PAD_TOKEN = '[PAD]'
MASK_TOKEN = '[MASK]'
CLS_TOKEN = '[CLS]'

# Full vocabulary: special + chemistry + auxiliary
VOCAB = [PAD_TOKEN, MASK_TOKEN, CLS_TOKEN] + GENESIS_PARAMS + AUX_FEATURES
PARAM_TO_ID = {p: i for i, p in enumerate(VOCAB)}
NUM_PARAMS = len(GENESIS_PARAMS)
NUM_AUX = len(AUX_FEATURES)
VOCAB_SIZE = len(VOCAB)

# Metadata tokens (appended as additional context)
METADATA_KEYS = ['depth', 'lat', 'lon', 'elevation']


# ============================================================
# Model configurations
# ============================================================

MODEL_CONFIGS = {
    'small': {
        'n_layers': 4,
        'n_heads': 4,
        'd_model': 128,
        'd_ff': 512,
        'dropout': 0.1,
    },
    'base': {
        'n_layers': 8,
        'n_heads': 8,
        'd_model': 256,
        'd_ff': 1024,
        'dropout': 0.1,
    },
    'large': {
        'n_layers': 12,
        'n_heads': 8,
        'd_model': 512,
        'd_ff': 2048,
        'dropout': 0.1,
    },
}


# ============================================================
# Token embeddings
# ============================================================

class ParameterTokenEmbedding(nn.Module):
    """
    Converts (param_id, value) pairs into token embeddings.

    Each token = param_type_embedding + value_projection

    The value_projection uses a per-parameter learned linear transform,
    so pH (range 0-14) and EC (range 0-50000) are handled naturally.
    Separate MLPs for chemistry params and auxiliary features.
    """

    def __init__(self, d_model, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.d_model = d_model

        # Learnable embedding for each parameter type (including special tokens)
        self.param_type_emb = nn.Embedding(vocab_size, d_model)

        # Per-parameter value projection: scalar value → d_model vector
        # Chemistry params: 2-layer MLP each (handles different scales)
        self.chem_value_proj = nn.ModuleDict({
            p: nn.Sequential(
                nn.Linear(1, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for p in GENESIS_PARAMS
        })

        # Auxiliary features: 2-layer MLP each
        self.aux_value_proj = nn.ModuleDict({
            p: nn.Sequential(
                nn.Linear(1, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for p in AUX_FEATURES
        })

        # Water body type embedding (categorical, 4 classes)
        self.wbt_emb = nn.Embedding(len(WATER_BODY_TYPES), d_model)

        # Metadata projection (lat, lon, depth, elevation → single token)
        self.metadata_proj = nn.Sequential(
            nn.Linear(len(METADATA_KEYS), d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, param_ids, values, mask=None):
        """
        Args:
            param_ids: (batch, seq_len) - parameter type IDs from VOCAB
            values:    (batch, seq_len) - parameter values (0 for special tokens)
            mask:      (batch, seq_len) - True for real tokens, False for padding

        Returns:
            embeddings: (batch, seq_len, d_model)
        """
        batch_size, seq_len = param_ids.shape

        # Parameter type embeddings
        type_emb = self.param_type_emb(param_ids)  # (B, S, D)

        # Value embeddings (per-parameter projection)
        value_emb = torch.zeros_like(type_emb)
        values_expanded = values.unsqueeze(-1)  # (B, S, 1)

        # Chemistry params
        for param_name, proj in self.chem_value_proj.items():
            pid = PARAM_TO_ID[param_name]
            param_mask = (param_ids == pid)
            if param_mask.any():
                param_values = values_expanded[param_mask]  # (N, 1)
                value_emb[param_mask] = proj(param_values).to(value_emb.dtype)

        # Auxiliary features
        for param_name, proj in self.aux_value_proj.items():
            pid = PARAM_TO_ID[param_name]
            param_mask = (param_ids == pid)
            if param_mask.any():
                param_values = values_expanded[param_mask]  # (N, 1)
                value_emb[param_mask] = proj(param_values).to(value_emb.dtype)

        # Combined embedding
        embeddings = type_emb + value_emb
        embeddings = self.norm(embeddings)

        return embeddings


# ============================================================
# Transformer Encoder
# ============================================================

class GENESISEncoder(nn.Module):
    """
    GENESIS Transformer encoder for geochemical data.

    Processes variable-length sets of tokens using multi-head self-attention.
    Token sequence: [CLS] + chem_tokens + aux_tokens + metadata_token + wbt_token
    All tokens share the same d_model space and attend to each other.
    """

    def __init__(self, config_name='base'):
        super().__init__()
        cfg = MODEL_CONFIGS[config_name]
        self.config_name = config_name
        self.d_model = cfg['d_model']
        self.n_layers = cfg['n_layers']
        self.n_heads = cfg['n_heads']

        # Token embedding (handles chem + aux via unified vocabulary)
        self.embedding = ParameterTokenEmbedding(self.d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg['d_model'],
            nhead=cfg['n_heads'],
            dim_feedforward=cfg['d_ff'],
            dropout=cfg['dropout'],
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg['n_layers'],
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(cfg['d_model'])

    def forward(self, param_ids, values, padding_mask=None):
        """
        Args:
            param_ids:    (batch, seq_len) - parameter type IDs from unified VOCAB
                          Sequence includes CLS + chem + aux + metadata + wbt tokens
            values:       (batch, seq_len) - parameter values
            padding_mask: (batch, seq_len) - True for PAD tokens

        Returns:
            hidden_states: (batch, seq_len, d_model)
        """
        # Embed all tokens through unified embedding
        x = self.embedding(param_ids, values)

        # Transformer expects: src_key_padding_mask where True = ignore
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.final_norm(x)

        return x

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Masked Geochemical Modeling (MGM) Head
# ============================================================

class MGMHead(nn.Module):
    """
    Prediction head for Masked Geochemical Modeling.

    For each masked parameter token, predicts the original value.
    Uses per-parameter output projections (inverse of input projections).
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Shared hidden layer
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Per-parameter output heads (predict scalar value)
        self.output_heads = nn.ModuleDict({
            p: nn.Linear(d_model, 1)
            for p in GENESIS_PARAMS
        })

    def forward(self, hidden_states, param_ids, masked_positions):
        """
        Args:
            hidden_states:   (batch, seq_len, d_model)
            param_ids:       (batch, seq_len) - original param IDs (before masking)
            masked_positions: (batch, seq_len) - True where tokens were masked

        Returns:
            predictions: dict of {param_name: (values, indices)}
        """
        h = self.shared(hidden_states)

        predictions = {}
        for param_name, head in self.output_heads.items():
            pid = PARAM_TO_ID[param_name]
            # Find positions where this param was masked
            param_masked = masked_positions & (param_ids == pid)
            if param_masked.any():
                masked_h = h[param_masked]  # (N, D)
                pred = head(masked_h).squeeze(-1)  # (N,)
                predictions[param_name] = pred

        return predictions


# ============================================================
# Full GENESIS Model for Pretraining
# ============================================================

class GENESISForMGM(nn.Module):
    """
    GENESIS model with Masked Geochemical Modeling head.

    Full pipeline:
    1. Build token sequence: [CLS] + chem_tokens + aux_tokens + metadata + wbt
    2. Mask random chemistry tokens (aux/metadata/wbt never masked)
    3. Encode with Transformer
    4. Predict masked chemistry parameter values

    The MGM objective only predicts chemistry params. Aux features, metadata,
    and water body type serve as conditioning context — they attend to
    chemistry tokens but are never prediction targets.
    """

    def __init__(self, config_name='base'):
        super().__init__()
        cfg = MODEL_CONFIGS[config_name]
        self.config_name = config_name
        self.encoder = GENESISEncoder(config_name)
        self.mgm_head = MGMHead(cfg['d_model'])

    def forward(self, param_ids, values, padding_mask=None,
                original_param_ids=None, masked_positions=None):
        """
        Args:
            param_ids:          (B, S) - param IDs with MASK tokens in chem positions
            values:             (B, S) - values (0 for masked tokens)
            padding_mask:       (B, S) - True for PAD tokens
            original_param_ids: (B, S) - original param IDs (before masking)
            masked_positions:   (B, S) - True where masking was applied

        Token sequence layout (variable length, padded to max):
            [CLS, chem_1, ..., chem_k, aux_1, ..., aux_m, meta, wbt, PAD...]
            where k ≤ 20 (available chem params), m ≤ 18 (available aux features)

        Returns:
            predictions: dict of {param_name: predicted_values} (chemistry only)
        """
        hidden = self.encoder(param_ids, values, padding_mask)
        predictions = self.mgm_head(hidden, original_param_ids, masked_positions)
        return predictions

    def get_latent(self, param_ids, values, padding_mask=None):
        """
        Get latent representation (CLS token output, or mean-pool).
        Used for downstream tasks and latent diffusion.

        Returns:
            latent: (batch, d_model)
        """
        hidden = self.encoder(param_ids, values, padding_mask)

        # Use CLS token (position 0) as the latent representation
        # CLS attends to all tokens (chem + aux + meta + wbt) via self-attention
        cls_id = PARAM_TO_ID[CLS_TOKEN]
        cls_mask = (param_ids == cls_id)

        if cls_mask.any():
            # CLS is always at position 0
            latent = hidden[:, 0, :]
        else:
            # Fallback: mean pool over non-padding tokens
            if padding_mask is not None:
                valid_mask = ~padding_mask
                valid_mask = valid_mask.unsqueeze(-1).float()
                latent = (hidden * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
            else:
                latent = hidden.mean(dim=1)

        return latent

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    print("GENESIS Model Architecture Test")
    print("=" * 50)
    print(f"Vocabulary: {VOCAB_SIZE} tokens "
          f"(3 special + {NUM_PARAMS} chem + {NUM_AUX} aux)")
    print(f"Water body types: {WATER_BODY_TYPES}")

    for size in ['small', 'base', 'large']:
        model = GENESISForMGM(size)
        n_params = model.get_num_params()
        cfg = MODEL_CONFIGS[size]
        mem_mb = n_params * 4 / 1024 / 1024
        print(f"\nGENESIS-{size.capitalize()}:")
        print(f"  Layers: {cfg['n_layers']}, Heads: {cfg['n_heads']}, "
              f"d_model: {cfg['d_model']}")
        print(f"  Parameters: {n_params:,} ({mem_mb:.1f} MB)")

        # Test forward pass with full token sequence:
        # [CLS] + 12 chem + 10 aux + PAD...
        batch_size = 4
        seq_len = 25  # CLS + up to 20 chem + up to 18 aux + wbt + pad

        # Build realistic token sequence
        param_ids = torch.full((batch_size, seq_len), PARAM_TO_ID[PAD_TOKEN],
                               dtype=torch.long)
        values = torch.zeros(batch_size, seq_len)
        padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        # Position 0: CLS
        param_ids[:, 0] = PARAM_TO_ID[CLS_TOKEN]
        padding_mask[:, 0] = False

        # Positions 1-12: chemistry tokens
        for j, p in enumerate(GENESIS_PARAMS[:12]):
            param_ids[:, 1 + j] = PARAM_TO_ID[p]
            values[:, 1 + j] = torch.randn(batch_size)
            padding_mask[:, 1 + j] = False

        # Positions 13-22: aux tokens
        for j, p in enumerate(AUX_FEATURES[:10]):
            param_ids[:, 13 + j] = PARAM_TO_ID[p]
            values[:, 13 + j] = torch.randn(batch_size)
            padding_mask[:, 13 + j] = False

        # Create masking (only on chemistry positions 1-12)
        original_ids = param_ids.clone()
        masked_pos = torch.zeros_like(param_ids, dtype=torch.bool)
        masked_pos[:, 1:4] = True  # mask first 3 chem tokens
        param_ids[masked_pos] = PARAM_TO_ID[MASK_TOKEN]
        values[masked_pos] = 0

        # Forward
        preds = model(param_ids, values, padding_mask, original_ids, masked_pos)
        print(f"  Forward pass OK. Predictions for {len(preds)} chem params.")

        # Latent (from CLS token)
        latent = model.get_latent(param_ids, values, padding_mask)
        print(f"  Latent shape: {latent.shape}")

    print("\nAll tests passed!")
