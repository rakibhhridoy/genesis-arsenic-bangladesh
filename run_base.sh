#!/bin/bash
# ============================================================
# GENESIS — Base variant (end-to-end)
# ============================================================
# Stage 1 → Stage 2 → Diffusion → Eval for GENESIS-Base.
# Runs on A5000 24 GB or better. ~30 h, ~$7 at $0.22/hr spot.
# Run this AFTER ./run_small.sh completes and looks healthy —
# Base is independent of Small (no cross-size warm-starting),
# but you want Small's results for the scaling-law comparison
# and to catch pipeline bugs before burning Base-scale compute.
#
# Usage:
#   mkdir -p logs
#   nohup ./run_base.sh > logs/base.log 2>&1 &
#   tail -f logs/base.log
# ============================================================

set -e
cd "$(dirname "$0")"

echo "=============================================="
echo "GENESIS — Base variant"
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU/MPS')"
echo "=============================================="

pick_resume() {
    local stage_latest="$1"
    local warm_start="$2"
    if [ -f "$stage_latest" ]; then
        echo "$stage_latest"
    elif [ -n "$warm_start" ] && [ -f "$warm_start" ]; then
        echo "$warm_start"
    else
        echo ""
    fi
}

# ---- Step 1: Base Stage 1 (biggest single job in the Base run) ----
echo ""
echo ">>> [1/4] Base — Stage 1 Pretraining (fp16)"
BASE1_RESUME=$(pick_resume checkpoints/base_stage1/latest.pt "")
python src/04_pretrain_genesis.py \
    --size base --stage 1 --epochs 100 \
    --batch_size 512 --lr 1e-4 --mask_ratio 0.2 \
    --patience 20 --num_workers 4 --fp16 \
    ${BASE1_RESUME:+--resume "$BASE1_RESUME"}

# ---- Step 2: Base Stage 2 ----
echo ""
echo ">>> [2/4] Base — Stage 2 Fine-tuning (fp16)"
BASE2_RESUME=$(pick_resume checkpoints/base_stage2/latest.pt checkpoints/base_stage1/best.pt)
python src/04_pretrain_genesis.py \
    --size base --stage 2 --epochs 50 \
    --batch_size 256 --lr 5e-5 --mask_ratio 0.25 \
    --patience 15 --num_workers 4 --fp16 \
    --resume "$BASE2_RESUME"

# ---- Step 3: Diffusion on Base ----
echo ""
echo ">>> [3/4] Diffusion — Base encoder"
python src/05_finetune_diffusion.py \
    --encoder_size base \
    --encoder_ckpt checkpoints/base_stage2/best.pt \
    --epochs 300 --batch_size 64 --lr 5e-5 \
    --patience 30 --fp16

# ---- Step 4: Evaluate Base ----
echo ""
echo ">>> [4/4] Evaluation — Base"
python src/06_evaluate_transfer.py \
    --ckpt checkpoints/diffusion_base/best.pt \
    --n_samples 10 --ddim_steps 50

echo ""
echo "=============================================="
echo "Base variant complete: $(date)"
echo "Results: results/eval_base.json  (also merged into results/scaling_summary.json)"
echo "If Small → Base shows clear scaling, switch to 4090/A6000 pod and run ./run_large.sh"
echo "=============================================="
