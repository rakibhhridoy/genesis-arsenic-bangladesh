"""
GENESIS Paper 5 — Step 5: Latent Diffusion Fine-tuning
=======================================================
Fine-tunes the latent diffusion model on temporal geochemical pairs.
Learns to generate plausible future geochemical states conditioned on
current state + time horizon.

Usage:
  python 05_finetune_diffusion.py --encoder_size small --encoder_ckpt checkpoints/small_stage1/best.pt --epochs 200
  python 05_finetune_diffusion.py --encoder_size base --encoder_ckpt checkpoints/base_stage2/best.pt --epochs 300
"""

import argparse
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import sys
sys.path.insert(0, str(Path(__file__).parent))

# -------- signal-safe checkpoint state --------
_SHUTDOWN = {"requested": False, "signal": None}


def _handle_shutdown(signum, _frame):
    _SHUTDOWN["requested"] = True
    _SHUTDOWN["signal"] = signum
    print(f"\n[signal {signum}] shutdown requested — will save at next boundary",
          flush=True)


def _install_signal_handlers():
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _handle_shutdown)
        except (ValueError, OSError):
            pass


def _atomic_save(state: dict, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None),
    }


def _restore_rng(state):
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state.get("torch_cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception as e:
        print(f"[warn] RNG restore failed: {e}")

from model.genesis_encoder import GENESISForMGM, GENESIS_PARAMS, PARAM_TO_ID
from model.latent_diffusion import GENESISLatentDiffusion
from model.temporal_dataset import (
    TemporalPairDataset, temporal_collate_fn, load_temporal_pairs_dataset
)
from model.dataset import load_stats


def evaluate(model, dataloader, device, max_batches=None):
    """Evaluate diffusion model on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break

            current = {k: v.to(device) for k, v in batch['current'].items()}
            future = {k: v.to(device) for k, v in batch['future'].items()}
            metadata = batch['metadata'].to(device)
            delta_t = batch['delta_t'].to(device)

            loss = model.training_step(current, future, metadata, delta_t)
            total_loss += loss.item()
            n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def evaluate_predictions(model, dataloader, device, norm_stats, n_batches=20):
    """
    Generate predictions and compute per-parameter MAE in original scale.
    """
    model.eval()
    param_errors = {p: [] for p in GENESIS_PARAMS}
    chem_mean = np.asarray(norm_stats['chem_mean'], dtype=np.float32)
    chem_std = np.asarray(norm_stats['chem_std'], dtype=np.float32)
    chem_log = np.asarray(norm_stats['chem_log_mask'], dtype=bool)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break

            current = {k: v.to(device) for k, v in batch['current'].items()}
            future = {k: v.to(device) for k, v in batch['future'].items()}
            metadata = batch['metadata'].to(device)
            delta_t = batch['delta_t'].to(device)

            # Generate one sample per input (fast DDIM)
            preds_list = model.generate_fast(
                current, metadata, delta_t, n_samples=1, ddim_steps=50
            )
            preds = preds_list[0]

            # Compare with future state (denormalize). Each param appears at most
            # once per row in tokenization, so we gather the single target value
            # per row and align it with the per-row decoder prediction.
            future_ids = future['param_ids']
            future_vals = future['values']
            future_valid = ~future['padding_mask']

            for idx, p in enumerate(GENESIS_PARAMS):
                pid = PARAM_TO_ID[p]
                if p not in preds:
                    continue
                mask2d = (future_ids == pid) & future_valid  # (B, S)
                row_has = mask2d.any(dim=1)                  # (B,)
                if not row_has.any():
                    continue
                # One target per row (zero for rows lacking the param, then subset)
                target_per_row = (future_vals * mask2d.float()).sum(dim=1)
                target_norm = target_per_row[row_has]
                pred_norm = preds[p][row_has]

                mean = float(chem_mean[idx])
                std = float(chem_std[idx])
                target_raw = target_norm * std + mean
                pred_raw = pred_norm * std + mean
                if chem_log[idx]:
                    target_raw = torch.expm1(target_raw)
                    pred_raw = torch.expm1(pred_raw)
                # True MAE: mean of elementwise absolute errors (not |means|).
                mae = torch.abs(pred_raw - target_raw).mean().item()
                param_errors[p].append(mae)

    model.train()
    avg_errors = {p: np.mean(e) if e else float('nan') for p, e in param_errors.items()}
    return avg_errors


def train(args):
    print("=" * 70)
    print(f"GENESIS Latent Diffusion Fine-tuning")
    print(f"  Encoder: {args.encoder_size}, Checkpoint: {args.encoder_ckpt}")
    print("=" * 70)

    _install_signal_handlers()

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Paths. no_pretrain ablation writes to diffusion_{size}_nopt/ so it does
    # not clobber the pretrained run's checkpoints or results.
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    ckpt_tag = f"diffusion_{args.encoder_size}" + ("_nopt" if args.no_pretrain else "")
    ckpt_dir = Path(__file__).parent.parent / "checkpoints" / ckpt_tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load normalization stats. Canonical source is data/processed/normalization_stats.json
    # (array format expected by GENESISDataset/TemporalPairDataset). Falls back to a copy
    # written by pretrain into the encoder ckpt dir (not used in --no_pretrain).
    canonical_stats = data_dir / "normalization_stats.json"
    encoder_ckpt_dir = Path(args.encoder_ckpt).parent if args.encoder_ckpt else None
    fallback_stats = encoder_ckpt_dir / "norm_stats.json" if encoder_ckpt_dir else None
    if canonical_stats.exists():
        norm_stats = load_stats(canonical_stats)
        print(f"Loaded norm stats from {canonical_stats}")
    elif fallback_stats is not None and fallback_stats.exists():
        norm_stats = load_stats(fallback_stats)
        print(f"Loaded norm stats from {fallback_stats}")
    else:
        raise FileNotFoundError(
            f"normalization_stats.json not found in {data_dir}. "
            "Run pretraining first (it writes the canonical stats)."
        )

    # Load temporal pairs (wide-format parquet → tensors via factory).
    # final_temporal_pairs.parquet is the curated subset; no matching aux .pt
    # exists for it, so aux is omitted here. If you produce one, pass aux_path=.
    data_path = data_dir / "final_temporal_pairs.parquet"
    print(f"\nLoading temporal pairs from: {data_path}")
    dataset, _ = load_temporal_pairs_dataset(data_path, norm_stats)
    print(f"Dataset size: {len(dataset):,}")

    # Train/val/test split (70/15/15)
    n_test = max(100, int(len(dataset) * 0.15))
    n_val = max(100, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {n_train:,}, Val: {n_val:,}, Test: {n_test:,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=temporal_collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=temporal_collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=temporal_collate_fn,
        pin_memory=True,
    )

    # Load (or skip) pretrained encoder
    encoder = GENESISForMGM(args.encoder_size)
    if args.no_pretrain:
        print("\n[ABLATION] --no_pretrain: encoder initialized randomly, no MGM weights loaded.")
        # Encoder MUST be trainable in the ablation — otherwise diffusion
        # conditions on random features forever.
        freeze_encoder = False
    else:
        if not args.encoder_ckpt:
            raise ValueError("--encoder_ckpt required unless --no_pretrain is set")
        print(f"\nLoading pretrained encoder: {args.encoder_ckpt}")
        ckpt_data = torch.load(args.encoder_ckpt, map_location='cpu', weights_only=False)
        encoder.load_state_dict(ckpt_data['model_state_dict'])
        print(f"  Encoder loaded (val_loss={ckpt_data.get('val_loss', 'N/A')})")
        freeze_encoder = args.freeze_encoder

    # Create diffusion model
    model = GENESISLatentDiffusion(
        pretrained_encoder=encoder,
        d_latent=args.d_latent,
        diffusion_steps=args.diffusion_steps,
        n_denoising_layers=args.n_layers,
        n_heads=args.n_heads,
        freeze_encoder=freeze_encoder,
    ).to(device)

    n_params = model.get_num_params()
    n_total = model.get_num_params(include_encoder=True)
    print(f"\nDiffusion model:")
    print(f"  Trainable params: {n_params:,}")
    print(f"  Total params (incl encoder): {n_total:,}")
    print(f"  d_latent: {args.d_latent}, layers: {args.n_layers}, heads: {args.n_heads}")
    print(f"  Diffusion steps: {args.diffusion_steps}")
    print(f"  Encoder frozen: {args.freeze_encoder}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Mixed precision
    use_amp = args.fp16 and device.type in ('cuda', 'mps')
    scaler = torch.amp.GradScaler(enabled=use_amp and device.type == 'cuda')
    amp_dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    start_epoch = 0
    global_step = 0

    # -------- Resume from latest.pt / best.pt --------
    latest_path = ckpt_dir / "latest.pt"
    resume_path = None
    if args.resume:
        if args.resume == "auto":
            if latest_path.exists():
                resume_path = latest_path
            elif (ckpt_dir / "best.pt").exists():
                resume_path = ckpt_dir / "best.pt"
        else:
            cand = Path(args.resume)
            if cand.exists():
                resume_path = cand
            else:
                print(f"[warn] --resume {args.resume} not found; starting fresh")

    if resume_path is not None:
        print(f"\nResuming from {resume_path}")
        rckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(rckpt['model_state_dict'])
        if 'optimizer_state_dict' in rckpt:
            optimizer.load_state_dict(rckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in rckpt:
            scheduler.load_state_dict(rckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in rckpt and scaler is not None:
            try:
                scaler.load_state_dict(rckpt['scaler_state_dict'])
            except Exception as e:
                print(f"  [warn] scaler restore failed: {e}")
        _restore_rng(rckpt.get('rng_state'))
        start_epoch = int(rckpt.get('epoch', -1)) + 1
        best_val_loss = float(rckpt.get('best_val_loss', rckpt.get('val_loss', float('inf'))))
        patience_counter = int(rckpt.get('patience_counter', 0))
        history = rckpt.get('history', []) or []
        global_step = int(rckpt.get('global_step', 0))
        print(f"  start_epoch={start_epoch}  best_val_loss={best_val_loss:.4f}  "
              f"patience={patience_counter}  global_step={global_step}")

    def _full_state(epoch, val_loss):
        return {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'rng_state': _rng_state(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'history': history,
            'encoder_size': args.encoder_size,
            'd_latent': args.d_latent,
            'diffusion_steps': args.diffusion_steps,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads,
            'norm_stats': norm_stats,
        }

    print(f"\nTraining for {args.epochs} epochs (from epoch {start_epoch+1})...")
    print(f"  Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Ckpt every {args.ckpt_every_steps} steps → {latest_path}")
    print()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            current = {k: v.to(device) for k, v in batch['current'].items()}
            future = {k: v.to(device) for k, v in batch['future'].items()}
            metadata = batch['metadata'].to(device)
            delta_t = batch['delta_t'].to(device)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model.training_step(current, future, metadata, delta_t)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            # Periodic latest.pt (preemption-safe mid-epoch resume)
            if args.ckpt_every_steps > 0 and global_step % args.ckpt_every_steps == 0:
                _atomic_save(_full_state(epoch - 1, float('nan')), latest_path)

            if _SHUTDOWN["requested"]:
                print(f"[shutdown] saving latest.pt at step {global_step}")
                _atomic_save(_full_state(epoch - 1, float('nan')), latest_path)
                with open(ckpt_dir / "history.json", 'w') as f:
                    json.dump(history, f, indent=2)
                print("[shutdown] clean exit")
                return

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # Validation
        val_loss = evaluate(model, val_loader, device)

        record = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'time': elapsed,
        }
        history.append(record)

        # Print every 10 epochs or if best
        if (epoch + 1) % 10 == 0 or val_loss < best_val_loss:
            print(f"Epoch {epoch+1}/{args.epochs} — "
                  f"train={train_loss:.4f} val={val_loss:.4f} "
                  f"lr={record['lr']:.2e} time={elapsed:.1f}s")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            _atomic_save(_full_state(epoch, val_loss), ckpt_dir / "best.pt")
        else:
            patience_counter += 1

        # Always save latest.pt at epoch boundary for clean resume
        _atomic_save(_full_state(epoch, val_loss), latest_path)

        # Incremental history dump (survives preemption)
        with open(ckpt_dir / "history.json", 'w') as f:
            json.dump(history, f, indent=2)

        # Periodic epoch snapshot
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, ckpt_dir / f"epoch_{epoch+1}.pt")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        if _SHUTDOWN["requested"]:
            print(f"[shutdown] clean exit at epoch boundary {epoch+1}")
            return

    # Final evaluation on test set
    print(f"\n{'='*50}")
    print("Final evaluation on test set")
    print(f"{'='*50}")

    # Load best model if one was written; otherwise fall through with current weights.
    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
    else:
        print(f"  (no best.pt at {best_path}; evaluating with final weights)")

    test_loss = evaluate(model, test_loader, device)
    print(f"Test diffusion loss: {test_loss:.4f}")

    # Per-parameter prediction quality
    print("\nGenerating predictions (DDIM 50 steps)...")
    param_mae = evaluate_predictions(model, test_loader, device, norm_stats, n_batches=10)
    print("\nPer-parameter MAE (original scale):")
    for p in GENESIS_PARAMS:
        mae = param_mae.get(p, float('nan'))
        if not np.isnan(mae):
            print(f"  {p:>6s}: {mae:.4f}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GENESIS Diffusion Fine-tuning")
    parser.add_argument('--encoder_size', type=str, default='small',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--encoder_ckpt', type=str, default=None,
                        help='Path to pretrained encoder checkpoint (required unless --no_pretrain)')
    parser.add_argument('--no_pretrain', action='store_true',
                        help='Ablation: skip MGM pretrain, init encoder randomly and train end-to-end '
                             '(writes to checkpoints/diffusion_{size}_nopt/)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--d_latent', type=int, default=64)
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--freeze_encoder', action='store_true', default=True)
    parser.add_argument('--unfreeze_encoder', action='store_true')
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use mixed precision (fp16) training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume path, or "auto" to pick latest.pt / best.pt in ckpt_dir')
    parser.add_argument('--ckpt_every_steps', type=int, default=500,
                        help='Save latest.pt every N optimizer steps (0 to disable)')
    args = parser.parse_args()

    if args.unfreeze_encoder:
        args.freeze_encoder = False

    train(args)
