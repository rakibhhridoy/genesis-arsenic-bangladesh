"""
GENESIS Latent Diffusion Model — Probabilistic Future-State Generation
========================================================================
Given a current geochemical state encoded in latent space by the
pretrained GENESIS encoder, generates a distribution of plausible
future geochemical states via denoising diffusion.

Architecture:
  - Encoder: pretrained GENESIS encoder (frozen or fine-tuned)
  - Latent projection: d_model → d_latent (64)
  - Diffusion: 1000 timesteps, cosine noise schedule
  - Denoising network: 4-layer Transformer with cross-attention
    conditioning on current state + metadata (depth, geology, time)
  - Decoder: d_latent → per-parameter value predictions

Cross-attention conditioning (Stable Diffusion style):
  Context = [current geochemical embedding, metadata tokens]
  Injected via cross-attention at each denoising layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# ============================================================
# Noise schedule
# ============================================================

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class DiffusionSchedule:
    """Precomputed diffusion schedule constants."""

    def __init__(self, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(device)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)

    def to(self, device):
        """Move all tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: add noise to x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)

        return sqrt_alpha * x_0 + sqrt_one_minus * noise


# ============================================================
# Cross-Attention Denoising Block
# ============================================================

class CrossAttentionDenoisingBlock(nn.Module):
    """
    Single denoising block with:
    1. Self-attention over noisy latent
    2. Cross-attention conditioned on context (current state + metadata)
    3. Feed-forward network
    """

    def __init__(self, d_latent, d_context, n_heads=4, dropout=0.1):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_latent, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_latent)

        # Cross-attention (query = latent, key/value = context)
        self.cross_attn = nn.MultiheadAttention(
            d_latent, n_heads, dropout=dropout, batch_first=True
        )
        self.context_proj = nn.Linear(d_context, d_latent) if d_context != d_latent else nn.Identity()
        self.norm2 = nn.LayerNorm(d_latent)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_latent, d_latent * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent * 4, d_latent),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_latent)

    def forward(self, x, context, timestep_emb):
        """
        Args:
            x:           (B, 1, d_latent) - noisy latent vector (as single token)
            context:     (B, C, d_context) - conditioning context
            timestep_emb: (B, d_latent) - timestep embedding

        Returns:
            x: (B, 1, d_latent)
        """
        # Add timestep embedding
        x = x + timestep_emb.unsqueeze(1)

        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x)
        x = x + residual

        # Cross-attention with context
        residual = x
        x = self.norm2(x)
        ctx = self.context_proj(context)
        x, _ = self.cross_attn(x, ctx, ctx)
        x = x + residual

        # FFN
        residual = x
        x = self.norm3(x)
        x = self.ffn(x) + residual

        return x


# ============================================================
# Denoising Network
# ============================================================

class DenoisingNetwork(nn.Module):
    """
    4-layer Transformer denoiser with cross-attention conditioning.

    Takes noisy latent z_t, conditioning context, and timestep,
    predicts the noise ε added to the clean latent z_0.
    """

    def __init__(self, d_latent=64, d_context=256, n_layers=4, n_heads=4,
                 n_metadata=4, dropout=0.1):
        super().__init__()
        self.d_latent = d_latent
        self.d_context = d_context

        # Timestep embedding (sinusoidal + MLP)
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbedding(d_latent),
            nn.Linear(d_latent, d_latent * 4),
            nn.GELU(),
            nn.Linear(d_latent * 4, d_latent),
        )

        # Metadata embedding (depth, lat, lon, elevation → d_context)
        # Note: aux features are already encoded in the encoder's latent
        # via self-attention with aux tokens. The metadata here is for
        # additional conditioning in the diffusion denoiser.
        self.metadata_emb = nn.Sequential(
            nn.Linear(n_metadata, d_context),
            nn.GELU(),
            nn.Linear(d_context, d_context),
        )

        # Time horizon embedding: log1p + sinusoidal features → MLP → d_context.
        # delta_t in years ranges from months to decades (~0.1 to ~30) — a raw
        # scalar compresses that dynamic range to near-zero gradient for short
        # gaps. log1p flattens the range; sinusoidal features give the network
        # a rich multi-scale representation reviewers expect for time-aware
        # conditioning.
        self._dt_sin_dim = 32  # must be even
        # Frequencies span ~1/30 yr⁻¹ (decade scale) to ~10 yr⁻¹ (month scale).
        # Registered as a buffer so it moves with .to(device) / .cuda() and is
        # saved/restored atomically with the checkpoint.
        self.register_buffer(
            'dt_freqs',
            torch.exp(torch.linspace(math.log(1.0 / 30.0),
                                     math.log(10.0),
                                     self._dt_sin_dim // 2)),
            persistent=True,
        )
        self.time_horizon_emb = nn.Sequential(
            nn.Linear(1 + self._dt_sin_dim, d_context),
            nn.GELU(),
            nn.Linear(d_context, d_context),
        )

        # Input projection
        self.input_proj = nn.Linear(d_latent, d_latent)

        # Denoising blocks
        self.blocks = nn.ModuleList([
            CrossAttentionDenoisingBlock(d_latent, d_context, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Output projection (predict noise)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, d_latent),
        )

    def forward(self, z_noisy, t, context_emb, metadata=None, delta_t=None):
        """
        Args:
            z_noisy:     (B, d_latent) - noisy latent
            t:           (B,) - integer timesteps
            context_emb: (B, d_context) - encoded current geochemical state
            metadata:    (B, n_metadata) - optional [depth, lat, lon, elev]
            delta_t:     (B, 1) - time horizon in years

        Returns:
            noise_pred: (B, d_latent) - predicted noise
        """
        # Timestep embedding
        t_emb = self.time_emb(t)  # (B, d_latent)

        # Build context sequence for cross-attention
        # Context = [current_state, metadata, time_horizon]
        ctx_tokens = [context_emb.unsqueeze(1)]  # (B, 1, d_context)

        if metadata is not None:
            meta_emb = self.metadata_emb(metadata)  # (B, d_context)
            ctx_tokens.append(meta_emb.unsqueeze(1))

        if delta_t is not None:
            # delta_t: (B, 1) in years, positive. log1p → sinusoidal → concat.
            dt_log = torch.log1p(delta_t.clamp_min(0.0))  # (B, 1)
            freqs = self.dt_freqs.to(dt_log.dtype)        # cast to match AMP dtype
            angles = dt_log * freqs.unsqueeze(0)          # (B, half)
            dt_sin = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, sin_dim)
            dt_feat = torch.cat([dt_log, dt_sin], dim=-1)  # (B, 1 + sin_dim)
            dt_emb = self.time_horizon_emb(dt_feat)        # (B, d_context)
            ctx_tokens.append(dt_emb.unsqueeze(1))

        context = torch.cat(ctx_tokens, dim=1)  # (B, C, d_context)

        # Process noisy latent
        x = self.input_proj(z_noisy).unsqueeze(1)  # (B, 1, d_latent)

        for block in self.blocks:
            x = block(x, context, t_emb)

        x = x.squeeze(1)  # (B, d_latent)
        noise_pred = self.output_proj(x)

        return noise_pred


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# ============================================================
# Latent Space Projections
# ============================================================

class LatentProjector(nn.Module):
    """Projects encoder output to/from latent space."""

    def __init__(self, d_model, d_latent=64):
        super().__init__()
        self.encoder_to_latent = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_latent),
        )
        self.latent_to_decoder = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def encode(self, encoder_output):
        """d_model → d_latent"""
        return self.encoder_to_latent(encoder_output)

    def decode(self, latent):
        """d_latent → d_model"""
        return self.latent_to_decoder(latent)


# ============================================================
# Parameter Decoder
# ============================================================

class ParameterDecoder(nn.Module):
    """
    Decodes latent vector back to geochemical parameter values.
    Per-parameter output heads (mirrors encoder's per-parameter input).
    """

    def __init__(self, d_model, param_names):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.param_heads = nn.ModuleDict({
            p: nn.Linear(d_model, 1) for p in param_names
        })

    def forward(self, x):
        """
        Args:
            x: (B, d_model) - decoded latent

        Returns:
            predictions: dict of {param_name: (B,) values}
        """
        h = self.shared(x)
        return {p: head(h).squeeze(-1) for p, head in self.param_heads.items()}


# ============================================================
# Full Latent Diffusion Model
# ============================================================

class GENESISLatentDiffusion(nn.Module):
    """
    Complete GENESIS Latent Diffusion Model.

    Pipeline:
    1. Encode current geochemical state → latent z_current
    2. During training: encode future state → z_future, add noise, denoise
    3. During inference: sample z_future from noise via iterative denoising
    4. Decode z_future → predicted future geochemical parameters
    """

    def __init__(self, pretrained_encoder, d_latent=64, diffusion_steps=1000,
                 n_denoising_layers=4, n_heads=4, param_names=None,
                 freeze_encoder=True, recon_weight=1.0):
        super().__init__()

        d_model = pretrained_encoder.encoder.d_model
        self.d_model = d_model
        self.d_latent = d_latent
        self.diffusion_steps = diffusion_steps
        self.recon_weight = recon_weight

        if param_names is None:
            try:
                from .genesis_encoder import GENESIS_PARAMS
            except ImportError:
                from genesis_encoder import GENESIS_PARAMS
            param_names = GENESIS_PARAMS

        # Pretrained encoder (optionally frozen)
        self.encoder = pretrained_encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Latent projections
        self.projector = LatentProjector(d_model, d_latent)

        # Denoising network
        self.denoiser = DenoisingNetwork(
            d_latent=d_latent,
            d_context=d_model,  # context in encoder space
            n_layers=n_denoising_layers,
            n_heads=n_heads,
        )

        # Parameter decoder
        self.decoder = ParameterDecoder(d_model, param_names)

        # Diffusion schedule (created on first use)
        self._schedule = None

    def get_schedule(self, device):
        if self._schedule is None or self._schedule.betas.device != device:
            self._schedule = DiffusionSchedule(self.diffusion_steps, device)
        return self._schedule

    def encode_state(self, param_ids, values, padding_mask=None):
        """Encode a geochemical state to latent space."""
        with torch.no_grad() if not any(p.requires_grad for p in self.encoder.parameters()) else torch.enable_grad():
            encoder_emb = self.encoder.get_latent(param_ids, values, padding_mask)
        latent = self.projector.encode(encoder_emb)
        return latent, encoder_emb

    def training_step(self, current_state, future_state, metadata=None, delta_t=None):
        """
        Training forward pass.

        Two losses:
          1. Diffusion (MSE on predicted noise) — trains projector.encode + denoiser
          2. Reconstruction (MSE on decoded future chemistry) — trains projector.decode
             and ParameterDecoder. Without this term those modules never receive
             gradient and inference generates random-init outputs.

        Args:
            current_state: dict with param_ids, values, padding_mask (time t)
            future_state:  dict with param_ids, values, padding_mask (time t+dt)
            metadata:      (B, 4) - [depth, lat, lon, elevation]
            delta_t:       (B, 1) - time gap in years

        Returns:
            loss: combined diffusion + reconstruction loss
        """
        try:
            from .genesis_encoder import PARAM_TO_ID, GENESIS_PARAMS
        except ImportError:
            from genesis_encoder import PARAM_TO_ID, GENESIS_PARAMS

        device = current_state['param_ids'].device
        schedule = self.get_schedule(device)
        batch_size = current_state['param_ids'].shape[0]

        # Encode current state (context for cross-attention)
        z_current, current_emb = self.encode_state(
            current_state['param_ids'],
            current_state['values'],
            current_state.get('padding_mask')
        )

        # Encode future state (target for diffusion)
        z_future, _ = self.encode_state(
            future_state['param_ids'],
            future_state['values'],
            future_state.get('padding_mask')
        )

        # Sample random timesteps
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=device)

        # Add noise to future latent
        noise = torch.randn_like(z_future)
        z_noisy = schedule.q_sample(z_future, t, noise)

        # Predict noise (trains projector.encode via z_future→z_noisy path + denoiser)
        noise_pred = self.denoiser(
            z_noisy, t, current_emb, metadata, delta_t
        )
        diff_loss = F.mse_loss(noise_pred, noise)

        # Reconstruction: decode the clean future latent and regress to the
        # normalized chemistry values at each param position.
        future_decoded = self.projector.decode(z_future)  # (B, d_model)
        recon_preds = self.decoder(future_decoded)        # dict {p: (B,)}

        future_ids = future_state['param_ids']
        future_vals = future_state['values']
        future_pad = future_state.get('padding_mask')
        if future_pad is None:
            future_valid = torch.ones_like(future_ids, dtype=torch.bool)
        else:
            future_valid = ~future_pad

        recon_loss = torch.tensor(0.0, device=device)
        n_recon = 0
        for p, pred_vec in recon_preds.items():
            pid = PARAM_TO_ID[p]
            mask2d = (future_ids == pid) & future_valid  # (B, S), 0 or 1 True per row
            row_has = mask2d.any(dim=1)                  # (B,)
            if not row_has.any():
                continue
            # Each param appears at most once per row in tokenization → sum picks the
            # single value at the matching position; rows without the param sum to 0.
            target_per_row = (future_vals * mask2d.float()).sum(dim=1)  # (B,)
            target = target_per_row[row_has]
            pred = pred_vec[row_has]
            recon_loss = recon_loss + F.mse_loss(pred, target)
            n_recon += 1

        if n_recon > 0:
            recon_loss = recon_loss / n_recon

        return diff_loss + self.recon_weight * recon_loss

    @torch.no_grad()
    def generate(self, current_state, metadata=None, delta_t=None,
                 n_samples=1, guidance_scale=1.0):
        """
        Generate future geochemical states via iterative denoising.

        Args:
            current_state: dict with param_ids, values, padding_mask
            metadata:      (B, 4)
            delta_t:       (B, 1)
            n_samples:     number of future states to generate per input
            guidance_scale: classifier-free guidance (1.0 = no guidance)

        Returns:
            predictions: list of dicts, each {param_name: (B,) values}
        """
        device = current_state['param_ids'].device
        schedule = self.get_schedule(device)
        batch_size = current_state['param_ids'].shape[0]

        # Encode current state
        _, current_emb = self.encode_state(
            current_state['param_ids'],
            current_state['values'],
            current_state.get('padding_mask')
        )

        all_predictions = []

        for _ in range(n_samples):
            # Start from pure noise
            z = torch.randn(batch_size, self.d_latent, device=device)

            # Iterative denoising (reverse process)
            for t_idx in reversed(range(self.diffusion_steps)):
                t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)

                # Predict noise
                noise_pred = self.denoiser(z, t, current_emb, metadata, delta_t)

                # Denoise one step
                alpha = schedule.alphas[t_idx]
                alpha_cumprod = schedule.alphas_cumprod[t_idx]
                beta = schedule.betas[t_idx]

                # Mean of posterior
                z = schedule.sqrt_recip_alphas[t_idx] * (
                    z - beta / schedule.sqrt_one_minus_alphas_cumprod[t_idx] * noise_pred
                )

                # Add noise (except at t=0)
                if t_idx > 0:
                    noise = torch.randn_like(z)
                    z = z + torch.sqrt(schedule.posterior_variance[t_idx]) * noise

            # Decode latent to parameter values
            decoded = self.projector.decode(z)
            predictions = self.decoder(decoded)
            all_predictions.append(predictions)

        return all_predictions

    @torch.no_grad()
    def generate_fast(self, current_state, metadata=None, delta_t=None,
                      n_samples=1, ddim_steps=50):
        """
        Fast generation using DDIM sampling (fewer steps).

        Args:
            ddim_steps: number of denoising steps (default 50 vs 1000)
        """
        device = current_state['param_ids'].device
        schedule = self.get_schedule(device)
        batch_size = current_state['param_ids'].shape[0]

        _, current_emb = self.encode_state(
            current_state['param_ids'],
            current_state['values'],
            current_state.get('padding_mask')
        )

        # DDIM: select evenly spaced timesteps
        step_size = self.diffusion_steps // ddim_steps
        timesteps = list(range(0, self.diffusion_steps, step_size))[::-1]

        all_predictions = []

        for _ in range(n_samples):
            z = torch.randn(batch_size, self.d_latent, device=device)

            for i, t_idx in enumerate(timesteps):
                t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
                noise_pred = self.denoiser(z, t, current_emb, metadata, delta_t)

                # DDIM update
                alpha_cumprod = schedule.alphas_cumprod[t_idx]
                if i + 1 < len(timesteps):
                    alpha_cumprod_prev = schedule.alphas_cumprod[timesteps[i + 1]]
                else:
                    alpha_cumprod_prev = torch.tensor(1.0, device=device)

                # Predict x_0
                pred_x0 = (z - torch.sqrt(1 - alpha_cumprod) * noise_pred) / torch.sqrt(alpha_cumprod)

                # DDIM deterministic step
                z = (torch.sqrt(alpha_cumprod_prev) * pred_x0 +
                     torch.sqrt(1 - alpha_cumprod_prev) * noise_pred)

            decoded = self.projector.decode(z)
            predictions = self.decoder(decoded)
            all_predictions.append(predictions)

        return all_predictions

    def get_num_params(self, include_encoder=False):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if include_encoder:
            total += sum(p.numel() for p in self.encoder.parameters())
        return total


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    from model.genesis_encoder import GENESISForMGM, GENESIS_PARAMS

    print("GENESIS Latent Diffusion Model Test")
    print("=" * 50)

    # Create pretrained encoder (random weights for testing)
    encoder = GENESISForMGM('small')

    # Create diffusion model
    diffusion = GENESISLatentDiffusion(
        pretrained_encoder=encoder,
        d_latent=64,
        diffusion_steps=100,  # small for testing
        n_denoising_layers=4,
        n_heads=4,
        freeze_encoder=True,
    )

    n_params = diffusion.get_num_params()
    n_total = diffusion.get_num_params(include_encoder=True)
    print(f"Diffusion trainable params: {n_params:,}")
    print(f"Total params (incl encoder): {n_total:,}")

    # Test training step
    batch_size = 4
    seq_len = 12

    current_state = {
        'param_ids': torch.randint(3, 23, (batch_size, seq_len)),
        'values': torch.randn(batch_size, seq_len),
        'padding_mask': torch.zeros(batch_size, seq_len, dtype=torch.bool),
    }
    future_state = {
        'param_ids': torch.randint(3, 23, (batch_size, seq_len)),
        'values': torch.randn(batch_size, seq_len),
        'padding_mask': torch.zeros(batch_size, seq_len, dtype=torch.bool),
    }
    metadata = torch.randn(batch_size, 4)
    delta_t = torch.ones(batch_size, 1) * 8.0  # 8 year gap

    loss = diffusion.training_step(current_state, future_state, metadata, delta_t)
    print(f"\nTraining loss: {loss.item():.4f}")

    # Test generation (fast, 10 steps)
    preds = diffusion.generate_fast(current_state, metadata, delta_t,
                                     n_samples=3, ddim_steps=10)
    print(f"Generated {len(preds)} samples")
    for i, pred in enumerate(preds):
        print(f"  Sample {i+1}: {len(pred)} params, "
              f"As={pred['As'][:2].tolist()}")

    print("\nLatent Diffusion test PASSED!")
