#!/bin/bash
# Smoke test: run the same pipeline as run_hp_sensitivity.sh but tiny.
# Pretrain 5K samples × 2 epochs, classify 1 seed × 50 bootstrap.
# Total: ~3 min, just verifies the orchestration runs end-to-end.
set -e
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
mkdir -p logs checkpoints results

if [ -d checkpoints/small_stage1 ] && [ ! -d checkpoints/small_stage1.preHP_backup ]; then
    cp -R checkpoints/small_stage1 checkpoints/small_stage1.preHP_backup
fi

name="smoke"
mask=0.20
lr=1e-4
ckpt_dir="checkpoints/hp_${name}_stage1"
ckpt="${ckpt_dir}/best.pt"
log="logs/hp_${name}.log"

echo "=== Smoke test config: $name  mask=$mask  lr=$lr ==="

python src/04_pretrain_genesis.py \
    --size small --stage 1 \
    --subsample 5000 \
    --epochs 2 \
    --batch_size 128 --lr "$lr" --mask_ratio "$mask" \
    --patience 5 --num_workers 0 \
    --ckpt_every_steps 0 \
    > "$log" 2>&1

mkdir -p "$ckpt_dir"
cp checkpoints/small_stage1/best.pt "$ckpt"
cp checkpoints/small_stage1/norm_stats.json "$ckpt_dir/" 2>/dev/null || true
echo "Saved $ckpt"

python src/10_classify_exceedance.py \
    --encoder_size small \
    --encoder_ckpt "$ckpt" \
    --target_param As \
    --epochs 3 \
    --n_seeds 1 \
    --bootstrap_n 50 \
    --out_tag "hp_${name}_mlp_frozen" \
    >> "$log" 2>&1

echo "=== Smoke test done ==="
ls -la "results/classify_as_hp_${name}_mlp_frozen.json"
python3 -c "
import json
d = json.load(open('results/classify_as_hp_${name}_mlp_frozen.json'))
print('global AUC:', d['aggregate_global_test'].get('auc_mean_std'))
print('BD AUC:', d['aggregate_bangladesh'].get('auc_mean_std'))
"

# Restore
if [ -d checkpoints/small_stage1.preHP_backup ]; then
    rm -rf checkpoints/small_stage1
    mv checkpoints/small_stage1.preHP_backup checkpoints/small_stage1
fi
echo "=== Smoke complete ==="
