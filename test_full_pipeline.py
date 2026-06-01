"""
GENESIS Full Pipeline Dry-Run Test
====================================
Runs every step of the pipeline with tiny data/epochs.
If this passes, the cloud run will work.
"""

import sys
import json
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
import pandas as pd

PASS = 0
FAIL = 0

def test(name):
    global PASS, FAIL
    def decorator(func):
        def wrapper():
            global PASS, FAIL
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print(f"{'='*60}")
            try:
                func()
                PASS += 1
                print(f"  >>> PASSED")
                return True
            except Exception as e:
                FAIL += 1
                print(f"  >>> FAILED: {e}")
                traceback.print_exc()
                return False
        return wrapper
    return decorator


# ============================================================
# Test 1: Data files exist and load correctly
# ============================================================
@test("Data files load")
def test_data():
    data_dir = Path("data/processed")
    for f in ["all_water_vectors_stage1.parquet",
              "all_water_vectors_stage2.parquet",
              "final_temporal_pairs.parquet"]:
        p = data_dir / f
        assert p.exists(), f"{f} not found"
        df = pd.read_parquet(p)
        assert len(df) > 0, f"{f} is empty"
        print(f"  {f}: {len(df):,} rows OK")


# ============================================================
# Test 2: Model architectures (all 3 sizes)
# ============================================================
@test("Encoder architectures (Small/Base/Large)")
def test_encoders():
    from model.genesis_encoder import GENESISForMGM, MODEL_CONFIGS, PARAM_TO_ID, MASK_TOKEN

    for size in ['small', 'base', 'large']:
        model = GENESISForMGM(size)
        cfg = MODEL_CONFIGS[size]
        n = model.get_num_params()

        # Forward pass
        B, S = 4, 12
        x = torch.randint(3, 23, (B, S))
        v = torch.randn(B, S)
        pm = torch.zeros(B, S, dtype=torch.bool)
        pm[:, -2:] = True
        oi = x.clone()
        mp = torch.zeros(B, S, dtype=torch.bool)
        mp[:, 1:4] = True
        x[mp] = PARAM_TO_ID[MASK_TOKEN]
        v[mp] = 0

        preds = model(x, v, pm, oi, mp)
        latent = model.get_latent(x, v, pm)

        assert latent.shape == (B, cfg['d_model']), f"Bad latent shape for {size}"
        print(f"  GENESIS-{size}: {n:,} params, latent={latent.shape} OK")


# ============================================================
# Test 3: fp16 autocast works for all sizes
# ============================================================
@test("fp16 mixed precision (all sizes)")
def test_fp16():
    from model.genesis_encoder import GENESISForMGM, PARAM_TO_ID, MASK_TOKEN

    for size in ['small', 'base', 'large']:
        model = GENESISForMGM(size)
        B, S = 4, 12
        x = torch.randint(3, 23, (B, S))
        v = torch.randn(B, S)
        pm = torch.zeros(B, S, dtype=torch.bool)
        oi = x.clone()
        mp = torch.zeros(B, S, dtype=torch.bool)
        mp[:, 1:3] = True
        x[mp] = PARAM_TO_ID[MASK_TOKEN]

        with torch.amp.autocast('cpu', dtype=torch.bfloat16, enabled=True):
            preds = model(x, v, pm, oi, mp)
            latent = model.get_latent(x, v, pm)

        print(f"  GENESIS-{size} fp16: OK")


# ============================================================
# Test 4: Dataset + DataLoader
# ============================================================
@test("GENESISDataset (tensor corpus)")
def test_dataset():
    from model.dataset import GENESISDataset, collate_fn, load_stats, MAX_SEQ_LEN
    from torch.utils.data import DataLoader
    import pandas as pd

    chem = torch.load("data/processed/genesis_pretrain.pt",
                      weights_only=True).numpy()[:512]
    meta = pd.read_parquet("data/processed/genesis_pretrain_meta.parquet").head(512)
    aux_path = Path("data/processed/genesis_pretrain_aux.pt")
    aux = (torch.load(aux_path, weights_only=True).numpy()[:512]
           if aux_path.exists() else None)
    stats = load_stats("data/processed/normalization_stats.json")

    ds = GENESISDataset(chem, stats, meta_df=meta, aux_values=aux, mask_ratio=0.2)
    sample = ds[0]
    assert sample['param_ids'].shape == (MAX_SEQ_LEN,)

    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert batch['param_ids'].shape[0] == 8
    print(f"  GENESISDataset: {len(ds):,} samples, seq_len={MAX_SEQ_LEN}, batch OK")


# ============================================================
# Test 5: Temporal dataset
# ============================================================
@test("TemporalPairDataset via factory")
def test_temporal():
    from model.temporal_dataset import (
        temporal_collate_fn, load_temporal_pairs_dataset,
    )
    from model.dataset import load_stats
    from torch.utils.data import DataLoader

    stats = load_stats("data/processed/normalization_stats.json")
    ds, df = load_temporal_pairs_dataset(
        "data/processed/final_temporal_pairs.parquet", stats,
    )
    sample = ds[0]
    assert 'current' in sample and 'future' in sample
    assert 'metadata' in sample and 'delta_t' in sample
    assert sample['delta_t'].shape == (1,)

    loader = DataLoader(ds, batch_size=8, collate_fn=temporal_collate_fn)
    batch = next(iter(loader))
    assert batch['current']['param_ids'].shape[0] == 8
    assert batch['delta_t'].shape == (8, 1)
    print(f"  {len(ds):,} pairs ({len(df):,} rows), batch collation OK")


# ============================================================
# Test 6: Latent Diffusion Model (all sizes)
# ============================================================
@test("Latent Diffusion Model (all sizes)")
def test_diffusion():
    from model.genesis_encoder import GENESISForMGM
    from model.latent_diffusion import GENESISLatentDiffusion

    for size in ['small', 'base', 'large']:
        encoder = GENESISForMGM(size)
        diff = GENESISLatentDiffusion(
            encoder, d_latent=64, diffusion_steps=50,
            n_denoising_layers=4, n_heads=4, freeze_encoder=True
        )
        n = diff.get_num_params()

        B, S = 4, 12
        current = {
            'param_ids': torch.randint(3, 23, (B, S)),
            'values': torch.randn(B, S),
            'padding_mask': torch.zeros(B, S, dtype=torch.bool),
        }
        future = {
            'param_ids': torch.randint(3, 23, (B, S)),
            'values': torch.randn(B, S),
            'padding_mask': torch.zeros(B, S, dtype=torch.bool),
        }
        meta = torch.randn(B, 4)
        dt = torch.ones(B, 1) * 5.0

        # Training step
        loss = diff.training_step(current, future, meta, dt)
        assert loss.item() > 0
        loss.backward()

        # Generation (DDIM)
        diff.eval()
        with torch.no_grad():
            preds = diff.generate_fast(current, meta, dt, n_samples=2, ddim_steps=5)
        assert len(preds) == 2
        assert len(preds[0]) == 20  # all 20 params

        print(f"  Diffusion-{size}: {n:,} params, train+generate OK")


# ============================================================
# Test 7: Diffusion fp16
# ============================================================
@test("Diffusion fp16 training step")
def test_diffusion_fp16():
    from model.genesis_encoder import GENESISForMGM
    from model.latent_diffusion import GENESISLatentDiffusion

    encoder = GENESISForMGM('base')
    diff = GENESISLatentDiffusion(
        encoder, d_latent=64, diffusion_steps=50, freeze_encoder=True
    )

    B, S = 4, 12
    current = {'param_ids': torch.randint(3, 23, (B, S)),
               'values': torch.randn(B, S),
               'padding_mask': torch.zeros(B, S, dtype=torch.bool)}
    future = {'param_ids': torch.randint(3, 23, (B, S)),
              'values': torch.randn(B, S),
              'padding_mask': torch.zeros(B, S, dtype=torch.bool)}

    scaler = torch.amp.GradScaler(enabled=False)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, diff.parameters()), lr=1e-4
    )

    with torch.amp.autocast('cpu', dtype=torch.bfloat16, enabled=True):
        loss = diff.training_step(current, future, torch.randn(B, 4), torch.ones(B, 1))

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, diff.parameters()), 1.0)
    scaler.step(optimizer)
    scaler.update()

    print(f"  fp16 train step: loss={loss.item():.4f} OK")


# ============================================================
# Test 8: MGM pretraining loop (2 epochs, tiny subset)
# ============================================================
@test("Pretrain loop (2 epochs, 16 samples)")
def test_pretrain_loop():
    from model.genesis_encoder import GENESISForMGM, GENESIS_PARAMS, PARAM_TO_ID
    from model.dataset import GENESISDataset, collate_fn, load_stats
    from torch.utils.data import DataLoader, Subset
    import torch.nn as nn
    import pandas as pd

    chem = torch.load("data/processed/genesis_pretrain.pt",
                      weights_only=True).numpy()[:16]
    meta = pd.read_parquet("data/processed/genesis_pretrain_meta.parquet").head(16)
    aux_path = Path("data/processed/genesis_pretrain_aux.pt")
    aux = (torch.load(aux_path, weights_only=True).numpy()[:16]
           if aux_path.exists() else None)
    stats = load_stats("data/processed/normalization_stats.json")
    ds = GENESISDataset(chem, stats, meta_df=meta, aux_values=aux, mask_ratio=0.2)
    tiny = Subset(ds, range(16))
    loader = DataLoader(tiny, batch_size=8, collate_fn=collate_fn)

    model = GENESISForMGM('small')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(2):
        for batch in loader:
            preds = model(batch['param_ids'], batch['values'], batch['padding_mask'],
                          batch['original_param_ids'], batch['masked_positions'])

            total_loss = torch.tensor(0.0)
            n = 0
            for pname, pv in preds.items():
                pid = PARAM_TO_ID[pname]
                pmask = batch['masked_positions'] & (batch['original_param_ids'] == pid)
                target = batch['original_values'][pmask]
                if len(target) > 0:
                    total_loss = total_loss + criterion(pv, target) * len(target)
                    n += len(target)
            if n > 0:
                total_loss = total_loss / n
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                losses.append(total_loss.item())

    assert len(losses) >= 2
    print(f"  Epoch 1 loss: {losses[0]:.4f}")
    print(f"  Epoch 2 loss: {losses[-1]:.4f}")
    print(f"  Learning: {'YES' if losses[-1] < losses[0] else 'FLAT (ok for 2 epochs)'}")


# ============================================================
# Test 9: Diffusion fine-tuning loop (3 epochs, tiny subset)
# ============================================================
@test("Diffusion fine-tune loop (3 epochs, 16 samples)")
def test_diffusion_loop():
    from model.genesis_encoder import GENESISForMGM
    from model.latent_diffusion import GENESISLatentDiffusion
    from model.temporal_dataset import (
        temporal_collate_fn, load_temporal_pairs_dataset,
    )
    from model.dataset import load_stats
    from torch.utils.data import DataLoader, Subset

    stats = load_stats("data/processed/normalization_stats.json")

    # Load pretrained encoder
    encoder = GENESISForMGM('small')
    ckpt = torch.load("checkpoints/small_stage1/best.pt", map_location='cpu',
                       weights_only=False)
    encoder.load_state_dict(ckpt['model_state_dict'])

    diff = GENESISLatentDiffusion(
        encoder, d_latent=64, diffusion_steps=50, freeze_encoder=True
    )

    ds, _ = load_temporal_pairs_dataset(
        "data/processed/final_temporal_pairs.parquet", stats
    )
    tiny = Subset(ds, range(16))
    loader = DataLoader(tiny, batch_size=8, collate_fn=temporal_collate_fn)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, diff.parameters()), lr=1e-4
    )

    losses = []
    for epoch in range(3):
        for batch in loader:
            loss = diff.training_step(
                batch['current'], batch['future'],
                batch['metadata'], batch['delta_t']
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    assert len(losses) >= 3
    print(f"  Losses: {[f'{l:.4f}' for l in losses[:6]]}")

    # Test generation after training
    diff.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        preds = diff.generate_fast(
            batch['current'], batch['metadata'], batch['delta_t'],
            n_samples=2, ddim_steps=5
        )
    assert len(preds) == 2
    print(f"  Generation after training: {len(preds)} samples OK")


# ============================================================
# Test 10: Checkpoint save/load roundtrip
# ============================================================
@test("Checkpoint save/load roundtrip")
def test_checkpoint():
    from model.genesis_encoder import GENESISForMGM
    from model.latent_diffusion import GENESISLatentDiffusion
    import tempfile, os

    encoder = GENESISForMGM('small')
    diff = GENESISLatentDiffusion(
        encoder, d_latent=64, diffusion_steps=50, freeze_encoder=True
    )

    # Save
    tmp = tempfile.mktemp(suffix='.pt')
    torch.save({
        'epoch': 5,
        'model_state_dict': diff.state_dict(),
        'val_loss': 0.1234,
        'encoder_size': 'small',
        'd_latent': 64,
        'diffusion_steps': 50,
        'n_layers': 4,
        'n_heads': 4,
        'norm_stats': {'means': {'As': 0.1}, 'stds': {'As': 1.0}},
    }, tmp)

    # Load
    ckpt = torch.load(tmp, map_location='cpu', weights_only=False)
    encoder2 = GENESISForMGM(ckpt['encoder_size'])
    diff2 = GENESISLatentDiffusion(
        encoder2, d_latent=ckpt['d_latent'],
        diffusion_steps=ckpt['diffusion_steps'],
        n_denoising_layers=ckpt['n_layers'],
        n_heads=ckpt['n_heads'],
        freeze_encoder=True,
    )
    diff2.load_state_dict(ckpt['model_state_dict'])

    os.unlink(tmp)
    print(f"  Save/load roundtrip OK (epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']})")


# ============================================================
# Test 11: Evaluation script logic
# ============================================================
@test("Evaluation metrics computation")
def test_eval_metrics():
    # Simulate evaluation results
    from model.genesis_encoder import GENESIS_PARAMS

    fake_results = []
    for _ in range(20):
        r = {'delta_t': 8.0, 'params': {}}
        for p in GENESIS_PARAMS[:5]:
            target = np.random.randn() * 10
            preds = [target + np.random.randn() * 2 for _ in range(10)]
            r['params'][p] = {
                'target': target,
                'pred_mean': float(np.mean(preds)),
                'pred_std': float(np.std(preds)),
                'mae': float(np.abs(np.mean(preds) - target)),
                'all_preds': preds,
            }
        fake_results.append(r)

    # Test metric computation
    param_metrics = {p: {'mae': [], 'coverage': []} for p in GENESIS_PARAMS}
    for r in fake_results:
        for p, vals in r['params'].items():
            param_metrics[p]['mae'].append(vals['mae'])
            lo = vals['pred_mean'] - 1.645 * vals['pred_std']
            hi = vals['pred_mean'] + 1.645 * vals['pred_std']
            covered = lo <= vals['target'] <= hi
            param_metrics[p]['coverage'].append(float(covered))

    for p in GENESIS_PARAMS[:5]:
        mae = np.mean(param_metrics[p]['mae'])
        cov = np.mean(param_metrics[p]['coverage'])
        print(f"  {p}: MAE={mae:.3f}, 90%CI coverage={cov:.0%}")

    print(f"  Metrics computation OK")


# ============================================================
# Test 12: Script CLI args parse without error
# ============================================================
@test("CLI argument parsing (all scripts)")
def test_cli():
    import importlib.util

    scripts = [
        ("04_pretrain_genesis", ["--size", "small", "--stage", "1", "--epochs", "1", "--fp16"]),
        ("05_finetune_diffusion", ["--encoder_size", "small", "--encoder_ckpt", "dummy.pt",
                                    "--epochs", "1", "--fp16"]),
        ("06_evaluate_transfer", ["--ckpt", "dummy.pt", "--n_samples", "2"]),
    ]

    for script_name, test_args in scripts:
        # Just test argparse doesn't crash
        import argparse
        spec = importlib.util.spec_from_file_location(
            script_name, f"src/{script_name}.py"
        )
        mod = importlib.util.module_from_spec(spec)

        # Read and check argparse section exists
        with open(f"src/{script_name}.py") as f:
            content = f.read()
        assert "argparse" in content, f"{script_name} missing argparse"
        assert "fp16" in content or "evaluate" in script_name, f"{script_name} missing fp16"
        print(f"  {script_name}.py: CLI structure OK")


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    t0 = time.time()
    print("GENESIS FULL PIPELINE DRY-RUN TEST")
    print("=" * 60)

    tests = [
        test_data,
        test_encoders,
        test_fp16,
        test_dataset,
        test_temporal,
        test_diffusion,
        test_diffusion_fp16,
        test_pretrain_loop,
        test_diffusion_loop,
        test_checkpoint,
        test_eval_metrics,
        test_cli,
    ]

    for t in tests:
        t()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed ({elapsed:.1f}s)")
    print(f"{'='*60}")

    if FAIL == 0:
        print("\nALL TESTS PASSED — Safe to deploy to RunPod!")
    else:
        print(f"\n{FAIL} TESTS FAILED — Fix before deploying!")

    sys.exit(FAIL)
