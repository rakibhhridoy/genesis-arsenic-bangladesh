#!/bin/bash
# ============================================================
# Paper 5 — Hyperparameter sensitivity on Small encoder
# ============================================================
# Tests robustness of the negative-transfer result to MGM
# pretraining hyperparameters. Each config:
#   1) Pretrain Small Stage 1 on 50K subsample, 20 epochs
#   2) Classify BD-As frozen+MLP, 3 seeds, 200 bootstrap
# Configs:
#   baseline:    mask=0.20  lr=1e-4
#   low_mask:    mask=0.10  lr=1e-4
#   high_mask:   mask=0.30  lr=1e-4
#   high_lr:     mask=0.20  lr=5e-4
#   low_lr:      mask=0.20  lr=3e-5
#
# Estimated: ~1-2h per config on M1 MPS → ~5-10h total.
# Output:
#   checkpoints/hp_<cfg>_stage1/best.pt
#   results/hp_<cfg>_classify_bd_as.json
# ============================================================
set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
mkdir -p logs checkpoints results

CONFIGS=(
    "baseline    0.20  1e-4"
    "low_mask    0.10  1e-4"
    "high_mask   0.30  1e-4"
    "high_lr     0.20  5e-4"
    "low_lr      0.20  3e-5"
)

SUBSAMPLE=50000
PRETRAIN_EPOCHS=20
CLASSIFY_EPOCHS=30
N_SEEDS=3

# Back up the existing local small_stage1 (Apr 13 smoke test) so it survives
# the sweep, which writes to that directory by default.
if [ -d checkpoints/small_stage1 ] && [ ! -d checkpoints/small_stage1.preHP_backup ]; then
    echo "Backing up existing checkpoints/small_stage1 → small_stage1.preHP_backup"
    cp -R checkpoints/small_stage1 checkpoints/small_stage1.preHP_backup
fi

for line in "${CONFIGS[@]}"; do
    read -r name mask lr <<< "$line"
    ckpt_dir="checkpoints/hp_${name}_stage1"
    ckpt="${ckpt_dir}/best.pt"
    log="logs/hp_${name}.log"

    echo "============================================================"
    echo "=== Config: $name  mask=$mask  lr=$lr  ($(date))"
    echo "============================================================"

    # ---- Pretrain (rename ckpt dir so 04_pretrain_genesis writes
    #      to the hp_* directory by overriding --size symlinks).
    # The script writes to checkpoints/<size>_stage<stage>/, so we
    # train then move.
    rm -rf "checkpoints/small_stage1_hp_tmp"
    python src/04_pretrain_genesis.py \
        --size small --stage 1 \
        --subsample "$SUBSAMPLE" \
        --epochs "$PRETRAIN_EPOCHS" \
        --batch_size 128 --lr "$lr" --mask_ratio "$mask" \
        --patience 15 --num_workers 0 \
        --ckpt_every_steps 0 \
        > "$log" 2>&1

    # Move the produced checkpoint to the hp_* directory.
    if [ -f checkpoints/small_stage1/best.pt ]; then
        mkdir -p "$ckpt_dir"
        cp checkpoints/small_stage1/best.pt "$ckpt"
        cp checkpoints/small_stage1/history.json "$ckpt_dir/history.json" 2>/dev/null || true
        cp checkpoints/small_stage1/norm_stats.json "$ckpt_dir/norm_stats.json" 2>/dev/null || true
        echo "Saved $ckpt"
    else
        echo "ERROR: pretrain did not produce checkpoint"; exit 1
    fi

    # ---- Classify BD-As frozen + MLP ----
    python src/10_classify_exceedance.py \
        --encoder_size small \
        --encoder_ckpt "$ckpt" \
        --target_param As \
        --epochs "$CLASSIFY_EPOCHS" \
        --n_seeds "$N_SEEDS" \
        --bootstrap_n 200 \
        --out_tag "hp_${name}_mlp_frozen" \
        >> "$log" 2>&1

    echo "Finished $name at $(date)"
done

echo "=== ALL HP CONFIGS DONE ($(date)) ==="

# Restore the pre-sweep small_stage1 backup
if [ -d checkpoints/small_stage1.preHP_backup ]; then
    echo "Restoring checkpoints/small_stage1 from backup"
    rm -rf checkpoints/small_stage1
    mv checkpoints/small_stage1.preHP_backup checkpoints/small_stage1
fi
echo "=== Sweep complete ==="
