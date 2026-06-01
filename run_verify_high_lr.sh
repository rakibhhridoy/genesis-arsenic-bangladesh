#!/bin/bash
# ============================================================
# Paper 5 — Verify high_lr (lr=5e-4) finding at full corpus
# ============================================================
# Designed to run on RunPod A40 with the genesis_pretrain.pt
# corpus already in data/processed/. Mirrors the main-paper
# Small protocol but with lr=5e-4 instead of lr=1e-4.
#
# Why: the HP sensitivity sweep on M1 (50K subsample × 20 epochs)
# found that lr=5e-4 produced an encoder whose frozen+MLP head
# transfers to BD-As at AUC 0.732 — outside the chem-only LogReg
# CI (0.694 ± 0.009). The supplementary's intended message
# ("no HP variation beats LogReg") depends on whether this
# survives at the full-corpus, full-epoch scale.
#
# Defensive measures (per prior RunPod ops experience):
#   - setsid nohup for SIGHUP immunity
#   - PYTHONUNBUFFERED=1 for log streaming
#   - step-level checkpoints every 1000 steps
#   - --resume auto so interruption is recoverable
#
# Usage on RunPod (after corpus is in place):
#   chmod +x run_verify_high_lr.sh
#   setsid nohup ./run_verify_high_lr.sh > logs/verify_high_lr.log 2>&1 < /dev/null &
#   disown
#   tail -f logs/verify_high_lr.log
#
# Expected runtime on A40 spot: ~3 hours.
# ============================================================
set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
mkdir -p logs checkpoints results

echo "============================================================"
echo "high_lr verification run — Small × full corpus × lr=5e-4"
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU/MPS')"
echo "============================================================"

CKPT_DIR="checkpoints/small_stage1_high_lr"
CLASSIFY_TAG="verify_high_lr_mlp_frozen"

# ---- Step 1: Pretrain Small Stage 1 at lr=5e-4 ----
# The script writes to checkpoints/<size>_stage<stage>/ by default.
# We'll move the result after.
RESUME_FLAG=""
if [ -f "$CKPT_DIR/latest.pt" ]; then
    echo "Found existing $CKPT_DIR/latest.pt — will resume"
    mkdir -p checkpoints/small_stage1
    cp "$CKPT_DIR/latest.pt" checkpoints/small_stage1/latest.pt
    cp "$CKPT_DIR/best.pt" checkpoints/small_stage1/best.pt 2>/dev/null || true
    cp "$CKPT_DIR/norm_stats.json" checkpoints/small_stage1/ 2>/dev/null || true
    RESUME_FLAG="--resume auto"
fi

echo ""
echo ">>> [1/2] Small Stage 1 pretraining at lr=5e-4 (full corpus)"
python src/04_pretrain_genesis.py \
    --size small --stage 1 \
    --epochs 100 \
    --batch_size 256 --lr 5e-4 --mask_ratio 0.20 \
    --patience 20 --num_workers 3 --fp16 \
    --ckpt_every_steps 1000 \
    $RESUME_FLAG

# Move the produced checkpoint to the verification directory
mkdir -p "$CKPT_DIR"
cp checkpoints/small_stage1/best.pt "$CKPT_DIR/best.pt"
cp checkpoints/small_stage1/latest.pt "$CKPT_DIR/latest.pt"
cp checkpoints/small_stage1/history.json "$CKPT_DIR/history.json"
cp checkpoints/small_stage1/norm_stats.json "$CKPT_DIR/norm_stats.json"
echo "Saved Small high_lr checkpoint to $CKPT_DIR/"

# ---- Step 2: Classify BD-As frozen + MLP, 3 seeds ----
echo ""
echo ">>> [2/2] Classify BD-As, frozen + MLP, 3 seeds"
python src/10_classify_exceedance.py \
    --encoder_size small \
    --encoder_ckpt "$CKPT_DIR/best.pt" \
    --target_param As \
    --epochs 30 \
    --n_seeds 3 \
    --bootstrap_n 200 \
    --out_tag "$CLASSIFY_TAG" \
    --fp16

echo ""
echo "============================================================"
echo "Verification complete: $(date)"
echo "Result: results/classify_as_${CLASSIFY_TAG}.json"
echo ""
echo "Headline number to extract:"
python3 -c "
import json
d = json.load(open('results/classify_as_${CLASSIFY_TAG}.json'))
b = d['aggregate_bangladesh']['auc_mean_std']
g = d['aggregate_global_test']['auc_mean_std']
print(f'  BD AUC:     {b[0]:.3f} ± {b[1]:.3f}')
print(f'  Global AUC: {g[0]:.3f} ± {g[1]:.3f}')
print()
print('  Reference: LogReg chem-only = 0.694 ± 0.009')
if b[0] - b[1] > 0.703:
    print('  → VERIFIES the HP-sweep finding: high_lr genuinely beats LogReg.')
elif b[0] + b[1] < 0.685:
    print('  → REFUTES the HP-sweep finding: a subsample artifact.')
else:
    print('  → INCONCLUSIVE: overlaps with LogReg CI.')
"
echo "============================================================"
