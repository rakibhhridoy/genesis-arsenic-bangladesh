#!/bin/bash
# ============================================================
# GENESIS FULL SCIENTIFIC PIPELINE — All 3 Sizes
# ============================================================
# GPU:  RTX 4090 (spot or on-demand)
# Time: ~89 hours (~4 days)
# Cost: ~$39 (spot) / ~$53 (on-demand)
#
# Trains Small → Base → Large sequentially with fp16.
# Produces all data needed for:
#   - Scaling law analysis (3 model sizes)
#   - Latent diffusion temporal prediction
#   - Zero-shot Bangladesh transfer
#   - Uncertainty quantification
#
# Usage:
#   chmod +x run_full_science.sh
#   nohup ./run_full_science.sh > training.log 2>&1 &
#   tail -f training.log
# ============================================================

set -e
cd "$(dirname "$0")"

LOG_START=$(date +%s)

echo "=============================================="
echo "GENESIS Full Scientific Pipeline"
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No NVIDIA GPU')"
echo "=============================================="

# ============================================================
# STAGE 1: MGM Pretraining — All water types (943K samples)
# ============================================================

for SIZE in small base large; do
    echo ""
    echo ">>> Stage 1: GENESIS-${SIZE} pretraining (fp16)"
    echo "    Started: $(date)"

    # Adjust batch size by model size to maximize GPU utilization
    case $SIZE in
        small) BS=1024 ;;
        base)  BS=512  ;;
        large) BS=256  ;;
    esac

    python src/04_pretrain_genesis.py \
        --size $SIZE --stage 1 --epochs 100 \
        --batch_size $BS --lr 1e-4 --mask_ratio 0.2 \
        --patience 20 --num_workers 4 --fp16

    echo "    Finished: $(date)"
done

# ============================================================
# STAGE 2: Quality-filtered fine-tuning (398K samples)
# ============================================================

for SIZE in small base large; do
    echo ""
    echo ">>> Stage 2: GENESIS-${SIZE} fine-tuning (fp16)"
    echo "    Started: $(date)"

    case $SIZE in
        small) BS=1024 ;;
        base)  BS=256  ;;
        large) BS=128  ;;
    esac

    python src/04_pretrain_genesis.py \
        --size $SIZE --stage 2 --epochs 50 \
        --batch_size $BS --lr 5e-5 --mask_ratio 0.25 \
        --patience 15 --num_workers 4 --fp16 \
        --resume checkpoints/${SIZE}_stage1/best.pt

    echo "    Finished: $(date)"
done

# ============================================================
# STAGE 3: Latent Diffusion (3,527 temporal pairs)
# ============================================================

for SIZE in small base large; do
    echo ""
    echo ">>> Stage 3: Diffusion with GENESIS-${SIZE} encoder"
    echo "    Started: $(date)"

    python src/05_finetune_diffusion.py \
        --encoder_size $SIZE \
        --encoder_ckpt checkpoints/${SIZE}_stage2/best.pt \
        --epochs 300 --batch_size 64 --lr 5e-5 \
        --d_latent 64 --diffusion_steps 1000 \
        --n_layers 4 --n_heads 4 \
        --patience 30 --fp16

    echo "    Finished: $(date)"
done

# ============================================================
# STAGE 4: Evaluation — All sizes
# ============================================================

echo ""
echo ">>> Stage 4: Evaluation & Zero-Shot Transfer"

for SIZE in small base large; do
    echo ""
    echo "--- Evaluating GENESIS-${SIZE} ---"
    python src/06_evaluate_transfer.py \
        --ckpt checkpoints/diffusion_${SIZE}/best.pt \
        --n_samples 10 --ddim_steps 50
done

# ============================================================
# Summary
# ============================================================

LOG_END=$(date +%s)
ELAPSED=$(( (LOG_END - LOG_START) / 3600 ))

echo ""
echo "=============================================="
echo "PIPELINE COMPLETE"
echo "Total time: ${ELAPSED} hours"
echo "Finished: $(date)"
echo ""
echo "Outputs:"
echo "  checkpoints/small_stage{1,2}/best.pt"
echo "  checkpoints/base_stage{1,2}/best.pt"
echo "  checkpoints/large_stage{1,2}/best.pt"
echo "  checkpoints/diffusion_{small,base,large}/best.pt"
echo "  results/evaluation_results.json"
echo ""
echo "Download results:"
echo "  rsync -avz root@\$(hostname):/workspace/Paper5/checkpoints/ ./checkpoints/"
echo "  rsync -avz root@\$(hostname):/workspace/Paper5/results/ ./results/"
echo "=============================================="
