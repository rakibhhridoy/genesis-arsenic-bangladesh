#!/bin/bash
# ============================================================
# RunPod Instance Setup — Run this once after SSH into your pod
# ============================================================
# Template: RunPod PyTorch 2.x (has CUDA + PyTorch pre-installed)
# GPU: RTX A5000 (Phase 1) or RTX 4090 (Phase 2)
# Disk: 50GB (enough for data + checkpoints)
#
# Upload the pre-staged runpod_upload/ directory first:
#   rsync -avz --exclude='__pycache__' \
#     runpod_upload/ root@<pod-ip>:/workspace/Paper5/
# ============================================================

set -e
cd "$(dirname "$0")"

echo "=============================================="
echo "GENESIS — RunPod environment setup"
echo "=============================================="

# Install dependencies (most already in RunPod PyTorch image)
echo ""
echo ">>> [1/4] Installing deps"
pip install -q pandas pyarrow

# Verify GPU + fp16
echo ""
echo ">>> [2/4] Verifying GPU"
python - <<'PY'
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM: {vram:.1f} GB')
    assert vram > 20, f'GPU has only {vram:.1f} GB — Phase 1 needs >=24 GB'
else:
    raise SystemExit('No CUDA GPU detected — aborting.')
PY

# Verify required data files — strict gate: all 8 must exist with non-trivial size.
# Missing any of these will fail training somewhere between 2 minutes and 6 hours in.
# Catching it here saves at least that much spot-GPU time.
echo ""
echo ">>> [3/4] Verifying data/processed/ (8 required files)"
python - <<'PY'
import sys
from pathlib import Path

REQUIRED = {
    # name : (min_size_bytes, kind)
    'genesis_pretrain.pt':                    (100 * 1024 * 1024, 'Stage 1 chem tensor'),
    'genesis_pretrain_aux.pt':                (100 * 1024 * 1024, 'Stage 1 aux tensor'),
    'genesis_pretrain_meta.parquet':          (10  * 1024 * 1024, 'Stage 1 metadata'),
    'final_temporal_pairs.parquet':           (100 * 1024,        'Diffusion pairs'),
    'normalization_stats.json':               (500,               'Canonical norm stats'),
    'genesis_held_out_bd_pairs.pt':           (10  * 1024,        'BD held-out pairs'),
    'genesis_held_out_bd_pairs_aux.pt':       (5   * 1024,        'BD held-out aux'),
    'genesis_held_out_bd_pairs_meta.parquet': (5   * 1024,        'BD held-out meta'),
}

data_dir = Path('data/processed')
if not data_dir.exists():
    print(f'MISSING: {data_dir} does not exist')
    sys.exit(1)

bad = []
ok = 0
for name, (min_size, kind) in REQUIRED.items():
    p = data_dir / name
    if not p.exists():
        bad.append(f'  [MISS] {name} — {kind}')
        continue
    size = p.stat().st_size
    if size < min_size:
        bad.append(f'  [TINY] {name} — {size/1024:.0f} KB < expected {min_size/1024:.0f} KB ({kind})')
        continue
    print(f'  [OK]   {name:42s} {size/(1024*1024):7.1f} MB  {kind}')
    ok += 1

if bad:
    print()
    print(f'DATA CHECK FAILED ({ok}/{len(REQUIRED)} OK):')
    for b in bad:
        print(b)
    sys.exit(1)
print(f'\nAll {ok}/{len(REQUIRED)} data files present and non-truncated.')

# Quick schema validation: normalization_stats.json must be canonical array format
import json
stats = json.loads((data_dir / 'normalization_stats.json').read_text())
needed = {'chem_mean', 'chem_std', 'chem_log_mask'}
missing_keys = needed - stats.keys()
if missing_keys:
    print(f'SCHEMA ERROR: normalization_stats.json missing keys: {missing_keys}')
    sys.exit(1)
assert len(stats['chem_mean']) == 20, f"chem_mean has {len(stats['chem_mean'])} entries, expected 20"
print('normalization_stats.json: canonical array format OK (20 chem params)')
PY

# Quick model + encoder forward pass
echo ""
echo ">>> [4/4] Quick GPU forward-pass test"
python - <<'PY'
import sys; sys.path.insert(0, 'src')
from model.genesis_encoder import GENESISForMGM
from model.latent_diffusion import GENESISLatentDiffusion
import torch
encoder = GENESISForMGM('small').cuda()
x = torch.randint(3, 23, (4, 12)).cuda()
v = torch.randn(4, 12).cuda()
with torch.amp.autocast('cuda', dtype=torch.float16):
    out = encoder.encoder(x, v)
print(f'Encoder fp16 forward: {out.shape}')

diff = GENESISLatentDiffusion(
    pretrained_encoder=encoder, d_latent=64, diffusion_steps=1000,
    n_denoising_layers=2, n_heads=4, freeze_encoder=True,
).cuda()
n_train = sum(p.numel() for p in diff.parameters() if p.requires_grad)
print(f'Diffusion model built (trainable params: {n_train:,})')
PY

echo ""
echo "=============================================="
echo "Setup complete."
echo ""
echo "Three independent training runs — run in order:"
echo "  mkdir -p logs"
echo ""
echo "  # 1. Small (~6–8 h, A5000) — fast gate for pipeline bugs"
echo "  nohup ./run_small.sh > logs/small.log 2>&1 &"
echo "  tail -f logs/small.log"
echo ""
echo "  # 2. Base  (~30 h, A5000) — main scaling-law data point"
echo "  nohup ./run_base.sh  > logs/base.log  2>&1 &"
echo ""
echo "  # 3. Large (~50 h, 4090/A6000) — only if Small→Base shows scaling"
echo "  nohup ./run_large.sh > logs/large.log 2>&1 &"
echo ""
echo "Each script is standalone: eval writes results/eval_<size>.json and"
echo "merges into results/scaling_summary.json, so partial runs still produce"
echo "usable scaling data."
echo "=============================================="
