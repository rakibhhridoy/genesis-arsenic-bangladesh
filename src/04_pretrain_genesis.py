"""
GENESIS Paper 5 — Masked Geochemical Modeling Pretraining
==========================================================
Trains the GENESIS set transformer via masked prediction of chemistry
parameters, conditioned on auxiliary features and water body type.

Supports safe interruption (SIGINT/SIGTERM/SIGHUP): a step-level
`latest.pt` is written every --ckpt_every_steps and on signal,
containing full optimizer/scheduler/scaler/RNG state plus epoch+step,
so you can terminate and resume without losing progress.

Usage:
  # Mac smoke test (tiny subsample, safe on 8 GB MPS):
  python src/04_pretrain_genesis.py --size small --subsample 50000 \
      --epochs 2 --batch_size 128 --num_workers 0 --ckpt_every_steps 100

  # RunPod full run (Stage 1 pretraining on full 2M):
  python src/04_pretrain_genesis.py --size base --stage 1 --epochs 100 \
      --batch_size 512 --num_workers 4 --fp16

  # Resume from the latest step-level checkpoint:
  python src/04_pretrain_genesis.py --size base --stage 1 --resume auto
"""

import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

sys.path.insert(0, str(Path(__file__).parent))

from model.genesis_encoder import (
    GENESISForMGM, GENESIS_PARAMS, PARAM_TO_ID, MODEL_CONFIGS
)
from model.dataset import (
    GENESISDataset, collate_fn, load_stats, MAX_SEQ_LEN
)

PAPER5 = Path(__file__).resolve().parent.parent
PROC_DIR = PAPER5 / "data" / "processed"
CKPT_DIR = PAPER5 / "checkpoints"


# -------- signal-safe checkpoint state --------
_SHUTDOWN = {"requested": False, "signal": None}


def _handle_shutdown(signum, _frame):
    _SHUTDOWN["requested"] = True
    _SHUTDOWN["signal"] = signum
    print(f"\n[signal {signum}] shutdown requested — will save at next step boundary",
          flush=True)


def _install_signal_handlers():
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _handle_shutdown)
        except (ValueError, OSError):
            pass  # e.g. SIGHUP on Windows


def _atomic_save(state: dict, path: Path):
    """Write to .tmp then rename — a crash mid-write never corrupts latest.pt."""
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


def compute_mgm_loss(predictions, batch, criterion):
    """MSE loss on masked chemistry predictions."""
    total_loss = torch.tensor(0.0, requires_grad=True,
                              device=next(iter(predictions.values())).device
                              if predictions else 'cpu')
    per_param_loss = {}
    n_predictions = 0

    original_values = batch['original_values']
    original_param_ids = batch['original_param_ids']
    masked_positions = batch['masked_positions']

    for param_name, pred_values in predictions.items():
        pid = PARAM_TO_ID[param_name]
        param_masked = masked_positions & (original_param_ids == pid)
        target = original_values[param_masked]

        if len(target) > 0:
            loss = criterion(pred_values, target)
            total_loss = total_loss + loss * len(target)
            per_param_loss[param_name] = loss.item()
            n_predictions += len(target)

    if n_predictions > 0:
        total_loss = total_loss / n_predictions

    return total_loss, per_param_loss, n_predictions


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, max_batches=200):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_preds = 0
    param_losses = {p: [] for p in GENESIS_PARAMS}

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        predictions = model(
            batch['param_ids'], batch['values'], batch['padding_mask'],
            batch['original_param_ids'], batch['masked_positions']
        )
        loss, per_param, n_pred = compute_mgm_loss(predictions, batch, criterion)

        if n_pred > 0:
            total_loss += loss.item() * n_pred
            total_preds += n_pred
            for p, l in per_param.items():
                param_losses[p].append(l)

    avg_loss = total_loss / max(total_preds, 1)
    avg_param = {p: np.mean(ls) if ls else float('nan')
                 for p, ls in param_losses.items()}

    model.train()
    return avg_loss, avg_param, total_preds


def save_ckpt(path, *, epoch, step, model, optimizer, scheduler, scaler,
              val_loss, best_val_loss, args, extra=None):
    state = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "config": args.size,
        "stage": getattr(args, "stage", 1),
        "rng": _rng_state(),
    }
    if extra:
        state.update(extra)
    _atomic_save(state, Path(path))


def train(args):
    _install_signal_handlers()

    print("=" * 60)
    stage_tag = f" Stage {args.stage}" if args.stage else ""
    print(f"GENESIS MGM Pretraining — {args.size.upper()}{stage_tag}")
    print("=" * 60)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Checkpoint directory: always {size}_stage{stage} (matches existing
    # small_stage1/ layout and run_{small,base,large}.sh resume paths)
    ckpt_dir = CKPT_DIR / f"{args.size}_stage{args.stage}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"

    # Load data. Stage 2 uses a quality-filtered subset if a stage-specific
    # tensor is present (genesis_pretrain_stage2.pt + _stage2_meta + _stage2_aux);
    # otherwise it falls back to the Stage 1 corpus with a loud warning so we
    # never silently burn GPU time retraining on the same data.
    print(f"\nLoading data...")

    def _load_tensors(tag):
        chem_p = PROC_DIR / f"genesis_pretrain{tag}.pt"
        meta_p = PROC_DIR / f"genesis_pretrain{tag}_meta.parquet"
        aux_p = PROC_DIR / f"genesis_pretrain{tag}_aux.pt"
        if not chem_p.exists() or not meta_p.exists():
            return None
        chem = torch.load(chem_p, weights_only=True).numpy()
        meta = pd.read_parquet(meta_p)
        aux = torch.load(aux_p, weights_only=True).numpy() if aux_p.exists() else None
        return chem, meta, aux, chem_p

    loaded = None
    if args.stage == 2:
        loaded = _load_tensors("_stage2")
        if loaded is None:
            print("  [warn] Stage 2 requested but genesis_pretrain_stage2.pt not found; "
                  "falling back to Stage 1 corpus. Stage 2 will reuse the same data "
                  "with different hyperparams — re-run 06_prepare_training.py to "
                  "produce a quality-filtered stage-2 tensor if that matters.")
    if loaded is None:
        loaded = _load_tensors("")

    chem_tensor, meta_df, aux_tensor, chem_src = loaded
    print(f"  Loaded chem tensor from {chem_src.name}: {chem_tensor.shape}")
    if aux_tensor is not None:
        print(f"  Aux features loaded: {aux_tensor.shape}")
    else:
        print("  No aux features (run 06_prepare_training.py after GEE extraction)")

    # Subsample for smoke test (keep aligned rows)
    if args.subsample and args.subsample < chem_tensor.shape[0]:
        rng = np.random.default_rng(42)
        idx = rng.choice(chem_tensor.shape[0], size=args.subsample, replace=False)
        idx.sort()
        chem_tensor = chem_tensor[idx]
        if aux_tensor is not None:
            aux_tensor = aux_tensor[idx]
        meta_df = meta_df.iloc[idx].reset_index(drop=True)
        print(f"  Subsampled → {chem_tensor.shape[0]:,} rows (seed=42)")

    # Normalization stats
    stats_path = PROC_DIR / "normalization_stats.json"
    from model.dataset import compute_normalization_stats, save_stats
    if stats_path.exists():
        stats = load_stats(stats_path)
        print(f"  Loaded normalization stats from {stats_path}")
    else:
        stats = compute_normalization_stats(chem_tensor, aux_tensor)
        save_stats(stats, stats_path)
        print(f"  Computed and saved normalization stats")

    # Copy stats into ckpt_dir so downstream diffusion/eval can find a
    # self-contained norm_stats.json next to the encoder checkpoint.
    ckpt_stats_path = ckpt_dir / "norm_stats.json"
    save_stats(stats, ckpt_stats_path)

    print(f"  Pretrain: {chem_tensor.shape[0]:,} vectors × "
          f"{chem_tensor.shape[1]} chem params")

    dataset = GENESISDataset(
        chem_values=chem_tensor,
        norm_stats=stats,
        meta_df=meta_df,
        aux_values=aux_tensor,
        mask_ratio=args.mask_ratio,
    )

    # Train/val split
    n_val = min(50000, max(1, int(len(dataset) * 0.05)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  Train: {n_train:,}, Val: {n_val:,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'), drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )

    # Model
    model = GENESISForMGM(args.size).to(device)
    n_params = model.get_num_params()
    cfg = MODEL_CONFIGS[args.size]
    print(f"\nGENESIS-{args.size.capitalize()}:")
    print(f"  Layers={cfg['n_layers']}, Heads={cfg['n_heads']}, "
          f"d_model={cfg['d_model']}, d_ff={cfg['d_ff']}")
    print(f"  Parameters: {n_params:,} ({n_params*4/1024/1024:.1f} MB)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(2000, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.MSELoss()

    # AMP — fp16 on CUDA only; MPS does not support GradScaler
    use_amp = args.fp16 and device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Resume
    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    resume_path = None
    if args.resume:
        if args.resume == "auto":
            if latest_path.exists():
                resume_path = latest_path
            elif best_path.exists():
                resume_path = best_path
        else:
            resume_path = Path(args.resume)

    if resume_path and resume_path.exists():
        print(f"\nResuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

        ckpt_stage = ckpt.get('stage', 1)
        cross_stage = ckpt_stage != args.stage
        # Cross-stage resume (e.g. stage1 → stage2) = weights-only warm start:
        # fresh optimizer, scheduler, epoch counter, and best-val tracking.
        if cross_stage:
            print(f"  cross-stage warm start: ckpt stage={ckpt_stage} → "
                  f"stage={args.stage} (weights only)")
        else:
            if 'optimizer_state_dict' in ckpt and ckpt['optimizer_state_dict']:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if ckpt.get('scheduler_state_dict'):
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if use_amp and ckpt.get('scaler_state_dict'):
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            _restore_rng(ckpt.get('rng'))
            start_epoch = ckpt.get('epoch', 0)
            start_step_in_epoch = ckpt.get('step', 0)
            best_val_loss = ckpt.get('best_val_loss',
                                     ckpt.get('val_loss', float('inf')))
            global_step = start_epoch * len(train_loader) + start_step_in_epoch
            print(f"  epoch={start_epoch}, step_in_epoch={start_step_in_epoch}, "
                  f"best_val={best_val_loss:.4f}")

    print(f"\nTraining: {args.epochs} epochs, batch={args.batch_size}, "
          f"lr={args.lr}, mask={args.mask_ratio}, fp16={use_amp}")
    print(f"  Steps/epoch: {len(train_loader):,}, total: {total_steps:,}")
    print(f"  Warmup: {warmup_steps} steps | ckpt_every_steps: {args.ckpt_every_steps}")
    print()

    history = []
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_preds = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Skip already-completed steps when resuming inside an epoch
            if epoch == start_epoch and batch_idx < start_step_in_epoch:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                predictions = model(
                    batch['param_ids'], batch['values'], batch['padding_mask'],
                    batch['original_param_ids'], batch['masked_positions']
                )
                loss, _, n_pred = compute_mgm_loss(predictions, batch, criterion)

            if n_pred > 0:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()

                epoch_loss += loss.item() * n_pred
                epoch_preds += n_pred
                global_step += 1

            # Progress log
            if (batch_idx + 1) % 200 == 0:
                avg = epoch_loss / max(epoch_preds, 1)
                lr = optimizer.param_groups[0]['lr']
                speed = (batch_idx + 1 - start_step_in_epoch) * args.batch_size \
                        / max(time.time() - t0, 1e-6)
                print(f"  [{epoch+1}] {batch_idx+1}/{len(train_loader)} "
                      f"loss={avg:.4f} lr={lr:.2e} {speed:.0f} samples/s",
                      flush=True)

            # Step-level checkpoint
            if (args.ckpt_every_steps > 0
                    and (batch_idx + 1) % args.ckpt_every_steps == 0):
                save_ckpt(
                    latest_path, epoch=epoch, step=batch_idx + 1,
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler if use_amp else None,
                    val_loss=float('nan'), best_val_loss=best_val_loss,
                    args=args,
                )

            # Graceful shutdown
            if _SHUTDOWN["requested"]:
                print(f"[shutdown] saving latest at epoch={epoch} "
                      f"step={batch_idx+1}")
                save_ckpt(
                    latest_path, epoch=epoch, step=batch_idx + 1,
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler if use_amp else None,
                    val_loss=float('nan'), best_val_loss=best_val_loss,
                    args=args,
                )
                print(f"[shutdown] saved → {latest_path}")
                return

        # Epoch fully completed — reset resume counter
        start_step_in_epoch = 0

        train_loss = epoch_loss / max(epoch_preds, 1)
        elapsed = time.time() - t0

        val_loss, val_params, _ = evaluate(model, val_loader, criterion, device)

        record = {
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 6),
            'val_loss': round(val_loss, 6),
            'lr': optimizer.param_groups[0]['lr'],
            'time_s': round(elapsed, 1),
        }
        history.append(record)

        print(f"Epoch {epoch+1}/{args.epochs} — "
              f"train={train_loss:.4f} val={val_loss:.4f} "
              f"lr={record['lr']:.2e} ({elapsed:.0f}s)")

        valid_params = {p: l for p, l in val_params.items() if not np.isnan(l)}
        if valid_params:
            best3 = sorted(valid_params.items(), key=lambda x: x[1])[:3]
            worst3 = sorted(valid_params.items(), key=lambda x: x[1])[-3:]
            print(f"  Best:  {', '.join(f'{p}={l:.4f}' for p,l in best3)}")
            print(f"  Worst: {', '.join(f'{p}={l:.4f}' for p,l in worst3)}")

        # Best checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            save_ckpt(
                best_path, epoch=epoch + 1, step=0,
                model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler if use_amp else None,
                val_loss=val_loss, best_val_loss=best_val_loss, args=args,
            )
            print(f"  *** New best → {best_path}")
        else:
            patience_counter += 1

        # Always roll latest forward at epoch boundary
        save_ckpt(
            latest_path, epoch=epoch + 1, step=0,
            model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler if use_amp else None,
            val_loss=val_loss, best_val_loss=best_val_loss, args=args,
        )

        if (epoch + 1) % 10 == 0:
            save_ckpt(
                ckpt_dir / f"epoch_{epoch+1}.pt",
                epoch=epoch + 1, step=0,
                model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler if use_amp else None,
                val_loss=val_loss, best_val_loss=best_val_loss, args=args,
            )

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        if _SHUTDOWN["requested"]:
            print("[shutdown] requested between epochs — exiting")
            return

    with open(ckpt_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='base',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                        help='1=pretrain on full corpus, 2=fine-tune on quality-filtered subset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--mask_ratio', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--fp16', action='store_true',
                        help='Enable fp16 AMP (CUDA only; ignored on MPS/CPU)')
    parser.add_argument('--subsample', type=int, default=0,
                        help='Use only N rows (for Mac smoke test). 0 = full dataset.')
    parser.add_argument('--ckpt_every_steps', type=int, default=500,
                        help='Save latest.pt every N steps. 0 = only at epoch boundaries.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint path, or "auto" to pick latest.pt')
    args = parser.parse_args()

    train(args)
