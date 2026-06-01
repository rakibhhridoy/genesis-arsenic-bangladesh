#!/bin/bash
# ============================================================
# Small-encoder variant of the classification sweep
# ============================================================
# Reuses 10_classify_exceedance.py + 11_baseline_xgboost.py with
# --encoder_size small. Output tags include "_small" so results
# don't collide with the Base sweep already in results/.
#
# Skips XGBoost / LogReg baselines — those are encoder-independent
# and already in results/classify_<param>_xgboost.json etc.
#
# Usage:
#   chmod +x run_classify_sweep_small.sh
#   nohup ./run_classify_sweep_small.sh > logs/classify_small.log 2>&1 &
#   tail -f logs/classify_small.log
# ============================================================

set -e
cd "$(dirname "$0")"
mkdir -p logs

export PYTHONUNBUFFERED=1

ENCODER_CKPT="${ENCODER_CKPT:-runpod_backup/checkpoints/small_stage2/best.pt}"
TARGETS="${TARGETS:-As F NO3 U}"
N_SEEDS="${N_SEEDS:-3}"
EPOCHS="${EPOCHS:-30}"
BOOTSTRAP_N="${BOOTSTRAP_N:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# Auto-pick device (M1 → mps, NVIDIA → cuda, else cpu)
if python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    DEVICE_ARG="--device mps"; DEVICE_LABEL="mps"
elif python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE_ARG="--device cuda"; DEVICE_LABEL="cuda"
else
    DEVICE_ARG="--device cpu"; DEVICE_LABEL="cpu"
fi
FP16_ARG=""

if [ ! -f "$ENCODER_CKPT" ]; then
    echo "ERROR: Small encoder ckpt not found at $ENCODER_CKPT"
    exit 1
fi

echo "=========================================================="
echo "Small-encoder classification sweep started: $(date)"
echo "Encoder ckpt: $ENCODER_CKPT  (size=small, 1.47M params)"
echo "Targets:      $TARGETS"
echo "Seeds:        $N_SEEDS  Epochs: $EPOCHS  Bootstrap: $BOOTSTRAP_N"
echo "Device:       $DEVICE_LABEL"
echo "=========================================================="

run_step() {
    local desc="$1"; shift
    local target="$1"; shift
    local tag="$1"; shift
    local target_lc
    target_lc=$(echo "$target" | tr '[:upper:]' '[:lower:]')
    local out_path="results/classify_${target_lc}_${tag}.json"
    if [ -f "$out_path" ]; then
        echo ""
        echo "[skip] $desc — $out_path already exists"
        return 0
    fi
    echo ""
    echo ">>> $desc — $(date '+%H:%M:%S')"
    echo "    cmd: $*"
    "$@"
}

for T in $TARGETS; do
    echo ""
    echo "##########################################################"
    echo "# Target: $T  (Small encoder)"
    echo "##########################################################"

    run_step "[$T] Small frozen + MLP" "$T" "small_mlp_frozen" \
        python3 src/10_classify_exceedance.py \
            --encoder_size small --encoder_ckpt "$ENCODER_CKPT" \
            --target_param "$T" --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --n_seeds "$N_SEEDS" --bootstrap_n "$BOOTSTRAP_N" \
            --out_tag small_mlp_frozen \
            $FP16_ARG $DEVICE_ARG

    run_step "[$T] Small unfrozen + MLP" "$T" "small_mlp_unfrozen" \
        python3 src/10_classify_exceedance.py \
            --encoder_size small --encoder_ckpt "$ENCODER_CKPT" \
            --target_param "$T" --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --n_seeds "$N_SEEDS" --bootstrap_n "$BOOTSTRAP_N" \
            --out_tag small_mlp_unfrozen \
            --unfreeze_encoder $FP16_ARG $DEVICE_ARG

    run_step "[$T] Small no-pretrain + MLP" "$T" "small_mlp_nopretrain" \
        python3 src/10_classify_exceedance.py \
            --encoder_size small \
            --target_param "$T" --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --n_seeds "$N_SEEDS" --bootstrap_n "$BOOTSTRAP_N" \
            --out_tag small_mlp_nopretrain \
            --no_pretrain $FP16_ARG $DEVICE_ARG

    run_step "[$T] Small frozen + LinProbe" "$T" "small_linprobe_frozen" \
        python3 src/10_classify_exceedance.py \
            --encoder_size small --encoder_ckpt "$ENCODER_CKPT" \
            --target_param "$T" --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --n_seeds "$N_SEEDS" --bootstrap_n "$BOOTSTRAP_N" \
            --out_tag small_linprobe_frozen \
            --linear_probe $FP16_ARG $DEVICE_ARG
done

echo ""
echo "=========================================================="
echo "Small-encoder sweep complete: $(date)"
echo "=========================================================="
