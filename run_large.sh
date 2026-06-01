#!/bin/bash
# ============================================================
# GENESIS — Large variant (end-to-end)
# ============================================================
# Stage 1 → Stage 2 → Diffusion → Eval for GENESIS-Large.
# Only run this AFTER ./run_small.sh and ./run_base.sh — the
# Small→Base scaling trend in results/scaling_summary.json is
# how you decide whether Large is worth the spend.
#
# Recommended pod: RTX 4090 or A6000 spot (Large needs more VRAM
# and scales better on faster cores). A5000 works but is ~1.5× slower.
# Estimated: ~50 h @ ~$17 on 4090 spot.
#
# Defensive measures (per prior RunPod ops experience —
# fixes the "stuck at epoch 21" SIGHUP interruptions seen
# during the 2026-05-03 and 2026-05-12 Large runs):
#   - setsid nohup ... < /dev/null & disown  (SIGHUP immune)
#   - PYTHONUNBUFFERED=1                     (log streams immediately)
#   - --ckpt_every_steps 1000                (step-level latest.pt
#                                             ~every 4 min on Large)
#   - --resume auto                          (auto-pick latest.pt
#                                             if interrupted)
#   - --num_workers 3                        (4 crashes A40; 3 stable)
#
# Usage on RunPod (after corpus + setup):
#   mkdir -p logs
#   setsid nohup ./run_large.sh > logs/large.log 2>&1 < /dev/null &
#   disown
#   tail -f logs/large.log
# ============================================================

set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
mkdir -p logs checkpoints results

echo "=============================================="
echo "GENESIS — Large variant"
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

# ---- Step 1: Large Stage 1 ----
# --resume auto picks checkpoints/large_stage1/latest.pt if present,
# else best.pt, else cold start. Step-level ckpts every 1000 steps
# so a SIGHUP/preempt loses <4 min of progress, not 30+ min.
echo ""
echo ">>> [1/4] Large — Stage 1 Pretraining (fp16)"
python src/04_pretrain_genesis.py \
    --size large --stage 1 --epochs 100 \
    --batch_size 256 --lr 1e-4 --mask_ratio 0.2 \
    --patience 20 --num_workers 3 --fp16 \
    --ckpt_every_steps 1000 \
    --resume auto

# ---- Step 2: Large Stage 2 ----
# Stage 2 needs to warm-start from Stage 1's best.pt the first time,
# so we keep the pick_resume helper rather than use --resume auto
# (which only looks inside large_stage2/).
echo ""
echo ">>> [2/4] Large — Stage 2 Fine-tuning (fp16)"
LARGE2_RESUME=$(pick_resume checkpoints/large_stage2/latest.pt checkpoints/large_stage1/best.pt)
python src/04_pretrain_genesis.py \
    --size large --stage 2 --epochs 50 \
    --batch_size 128 --lr 5e-5 --mask_ratio 0.25 \
    --patience 15 --num_workers 3 --fp16 \
    --ckpt_every_steps 1000 \
    --resume "$LARGE2_RESUME"

# ---- Step 3: Diffusion on Large ----
# Same SIGHUP hardening as Stages 1-2. --resume auto looks in
# checkpoints/diffusion_large/ (the script's default out dir).
echo ""
echo ">>> [3/4] Diffusion — Large encoder"
python src/05_finetune_diffusion.py \
    --encoder_size large \
    --encoder_ckpt checkpoints/large_stage2/best.pt \
    --epochs 300 --batch_size 32 --lr 5e-5 \
    --patience 30 --fp16 --unfreeze_encoder \
    --ckpt_every_steps 1000 \
    --resume auto

# ---- Step 4: Evaluate Large ----
echo ""
echo ">>> [4/4] Evaluation — Large"
python src/06_evaluate_transfer.py \
    --ckpt checkpoints/diffusion_large/best.pt \
    --n_samples 10 --ddim_steps 50

echo ""
echo "=============================================="
echo "Large variant complete: $(date)"
echo "Results: results/eval_large.json"
echo "Full scaling law: results/scaling_summary.json (small/base/large merged)"
echo "=============================================="
