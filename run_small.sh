#!/bin/bash
# ============================================================
# GENESIS — Small variant (end-to-end)
# ============================================================
# Stage 1 → Stage 2 → Diffusion → Eval for GENESIS-Small.
# Runs on A5000 24 GB or better. ~6–8 h, ~$2 at $0.22/hr spot.
# Small acts as the fast gate — if anything is broken in the
# pipeline, you'll find out here instead of 30 h into Base or Large.
#
# Usage:
#   mkdir -p logs
#   nohup ./run_small.sh > logs/small.log 2>&1 &
#   tail -f logs/small.log
# ============================================================

set -e
cd "$(dirname "$0")"

echo "=============================================="
echo "GENESIS — Small variant"
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU/MPS')"
echo "=============================================="

# Preemption-safe resume selector: prefer stage-local latest.pt (mid-run
# resume after preempt); else the cross-stage warm-start (first-time run
# of Stage 2); else empty (fresh Stage 1).
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

# ---- Step 1: Small Stage 1 ----
echo ""
echo ">>> [1/4] Small — Stage 1 Pretraining (fp16)"
SMALL1_RESUME=$(pick_resume checkpoints/small_stage1/latest.pt "")
python src/04_pretrain_genesis.py \
    --size small --stage 1 --epochs 100 \
    --batch_size 512 --lr 3e-4 --mask_ratio 0.2 \
    --patience 20 --num_workers 4 --fp16 \
    ${SMALL1_RESUME:+--resume "$SMALL1_RESUME"}

# ---- Step 2: Small Stage 2 ----
echo ""
echo ">>> [2/4] Small — Stage 2 Fine-tuning (fp16)"
SMALL2_RESUME=$(pick_resume checkpoints/small_stage2/latest.pt checkpoints/small_stage1/best.pt)
python src/04_pretrain_genesis.py \
    --size small --stage 2 --epochs 50 \
    --batch_size 512 --lr 5e-5 --mask_ratio 0.25 \
    --patience 15 --num_workers 4 --fp16 \
    --resume "$SMALL2_RESUME"

# ---- Step 3: Diffusion on Small ----
echo ""
echo ">>> [3/4] Diffusion — Small encoder"
python src/05_finetune_diffusion.py \
    --encoder_size small \
    --encoder_ckpt checkpoints/small_stage2/best.pt \
    --epochs 300 --batch_size 64 --lr 5e-5 \
    --patience 30 --fp16

# ---- Step 4: Evaluate Small ----
echo ""
echo ">>> [4/4] Evaluation — Small"
python src/06_evaluate_transfer.py \
    --ckpt checkpoints/diffusion_small/best.pt \
    --n_samples 10 --ddim_steps 50

echo ""
echo "=============================================="
echo "Small variant complete: $(date)"
echo "Results: results/eval_small.json  (also merged into results/scaling_summary.json)"
echo "Next step if Small looks good: ./run_base.sh"
echo "=============================================="
