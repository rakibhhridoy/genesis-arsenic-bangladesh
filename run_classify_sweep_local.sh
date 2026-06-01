#!/bin/bash
# ============================================================
# M1 / local variant of the classification sweep
# ============================================================
# Same matrix as run_classify_sweep.sh, but configured for an M1 Mac
# (or any non-CUDA host):
#   - --device mps (or auto)
#   - no --fp16 (Apple Silicon doesn't have native fp16 acceleration
#     for transformer training, and runs of this size don't benefit)
#   - smaller default BOOTSTRAP_N to keep wall-clock reasonable
#   - same idempotent skip-if-JSON-exists logic, so it works alongside
#     pod runs (just don't run both at once)
#
# Encoder ckpt path defaults to base_full/checkpoints/base_stage2/best.pt
# (the artifact we pulled back from the pod after Base finished).
#
# Usage:
#   chmod +x run_classify_sweep_local.sh
#   ./run_classify_sweep_local.sh 2>&1 | tee logs/classify_local.log
#
# Or unattended overnight:
#   nohup ./run_classify_sweep_local.sh > logs/classify_local.log 2>&1 &
#   tail -f logs/classify_local.log
# ============================================================

set -e
cd "$(dirname "$0")"
mkdir -p logs

export PYTHONUNBUFFERED=1

ENCODER_CKPT="${ENCODER_CKPT:-base_full/checkpoints/base_stage2/best.pt}"
TARGETS="${TARGETS:-As F NO3 U}"
N_SEEDS="${N_SEEDS:-3}"
EPOCHS="${EPOCHS:-30}"
BOOTSTRAP_N="${BOOTSTRAP_N:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# Auto-pick device: mps if available, else cpu
if python3 -c "import torch; assert torch.backends.mps.is_available()" 2>/dev/null; then
    DEVICE_ARG="--device mps"
    DEVICE_LABEL="mps"
elif python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE_ARG="--device cuda"
    DEVICE_LABEL="cuda"
else
    DEVICE_ARG="--device cpu"
    DEVICE_LABEL="cpu"
fi

# No fp16 on M1 — Apple Silicon's MPS backend doesn't support it well.
FP16_ARG=""

if [ ! -f "$ENCODER_CKPT" ]; then
    echo "ERROR: encoder ckpt not found at $ENCODER_CKPT"
    echo "Set ENCODER_CKPT=<path> or extract base_full.tar.gz first."
    exit 1
fi

echo "=========================================================="
echo "Local classification sweep started: $(date)"
echo "Encoder ckpt: $ENCODER_CKPT"
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
    echo "# Target: $T"
    echo "##########################################################"

    # Baselines (CPU; fast)
    run_step "[$T] XGBoost baseline" "$T" "xgboost" \
        python3 src/11_baseline_xgboost.py \
            --target_param "$T" --model xgboost \
            --n_seeds "$N_SEEDS" --bootstrap_n "$BOOTSTRAP_N"

    run_step "[$T] LogReg baseline" "$T" "logreg" \
        python3 src/11_baseline_xgboost.py \
            --target_param "$T" --model logreg \
            --n_seeds "$N_SEEDS" --bootstrap_n "$BOOTSTRAP_N"

    # Encoder conditions (MPS / CPU)
    run_step "[$T] frozen MGM encoder + MLP" "$T" "mlp_frozen" \
        python3 src/10_classify_exceedance.py \
            --encoder_size base --encoder_ckpt "$ENCODER_CKPT" \
            --target_param "$T" --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --n_seeds "$N_SEEDS" --bootstrap_n "$BOOTSTRAP_N" \
            $FP16_ARG $DEVICE_ARG

    run_step "[$T] unfrozen MGM encoder + MLP" "$T" "mlp_unfrozen" \
        python3 src/10_classify_exceedance.py \
            --encoder_size base --encoder_ckpt "$ENCODER_CKPT" \
            --target_param "$T" --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --n_seeds "$N_SEEDS" --bootstrap_n "$BOOTSTRAP_N" \
            --unfreeze_encoder $FP16_ARG $DEVICE_ARG

    run_step "[$T] no-pretrain (random encoder) + MLP" "$T" "mlp_nopretrain" \
        python3 src/10_classify_exceedance.py \
            --encoder_size base \
            --target_param "$T" --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --n_seeds "$N_SEEDS" --bootstrap_n "$BOOTSTRAP_N" \
            --no_pretrain $FP16_ARG $DEVICE_ARG

    run_step "[$T] linear probe on frozen MGM encoder" "$T" "linprobe_frozen" \
        python3 src/10_classify_exceedance.py \
            --encoder_size base --encoder_ckpt "$ENCODER_CKPT" \
            --target_param "$T" --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --n_seeds "$N_SEEDS" --bootstrap_n "$BOOTSTRAP_N" \
            --linear_probe $FP16_ARG $DEVICE_ARG
done

echo ""
echo "=========================================================="
echo "Local sweep complete: $(date)"
echo "Results: results/classify_*.json"
echo "Compile table: python3 src/12_compile_classify_table.py"
echo "=========================================================="
