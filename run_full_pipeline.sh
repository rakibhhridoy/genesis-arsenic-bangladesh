#!/bin/bash
# ============================================================
# GENESIS Full Training Pipeline — Cloud GPU Script
# ============================================================
# Run on a machine with GPU (A100/V100/T4).
# Trains all 3 encoder sizes through both stages,
# then fine-tunes diffusion model on temporal pairs.
#
# Usage:
#   chmod +x run_full_pipeline.sh
#   nohup ./run_full_pipeline.sh > training.log 2>&1 &
# ============================================================

set -e
cd "$(dirname "$0")"

echo "=============================================="
echo "GENESIS Full Training Pipeline"
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No NVIDIA GPU / using MPS')"
echo "=============================================="

# ============================================================
# 1. Pretraining: Stage 1 (all water types, 943K samples)
# ============================================================
echo ""
echo ">>> STAGE 1: Pretraining on all water types"

for SIZE in small base large; do
    echo ""
    echo "--- GENESIS-${SIZE} Stage 1 ---"
    python src/04_pretrain_genesis.py \
        --size $SIZE \
        --stage 1 \
        --epochs 100 \
        --batch_size 512 \
        --lr 1e-4 \
        --mask_ratio 0.2 \
        --patience 20 \
        --num_workers 4
done

# ============================================================
# 2. Pretraining: Stage 2 (quality-filtered, 398K samples)
# ============================================================
echo ""
echo ">>> STAGE 2: Fine-tuning on quality-filtered data"

for SIZE in small base large; do
    echo ""
    echo "--- GENESIS-${SIZE} Stage 2 ---"
    python src/04_pretrain_genesis.py \
        --size $SIZE \
        --stage 2 \
        --epochs 50 \
        --batch_size 256 \
        --lr 5e-5 \
        --mask_ratio 0.25 \
        --patience 15 \
        --num_workers 4 \
        --resume checkpoints/${SIZE}_stage1/best.pt
done

# ============================================================
# 3. Latent Diffusion Fine-tuning (3,527 temporal pairs)
# ============================================================
echo ""
echo ">>> STAGE 3: Latent Diffusion Fine-tuning"

for SIZE in small base large; do
    echo ""
    echo "--- Diffusion with GENESIS-${SIZE} encoder ---"
    python src/05_finetune_diffusion.py \
        --encoder_size $SIZE \
        --encoder_ckpt checkpoints/${SIZE}_stage2/best.pt \
        --epochs 300 \
        --batch_size 32 \
        --lr 5e-5 \
        --d_latent 64 \
        --diffusion_steps 1000 \
        --n_layers 4 \
        --n_heads 4 \
        --patience 30
done

# ============================================================
# 4. Evaluation
# ============================================================
echo ""
echo ">>> STAGE 4: Evaluation & Zero-Shot Transfer"

for SIZE in small base large; do
    echo ""
    echo "--- Evaluating GENESIS-${SIZE} ---"
    python src/06_evaluate_transfer.py \
        --ckpt checkpoints/diffusion_${SIZE}/best.pt \
        --n_samples 10 \
        --ddim_steps 50
done

echo ""
echo "=============================================="
echo "Pipeline complete: $(date)"
echo "=============================================="
