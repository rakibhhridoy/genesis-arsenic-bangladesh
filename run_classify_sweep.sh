#!/bin/bash
# ============================================================
# Full classification sweep — As / F / NO3 / U × 5 conditions × 3 seeds
# ============================================================
# Conditions:
#   1. xgboost            (non-neural baseline on raw chem + lat/lon)
#   2. logreg             (linear baseline)
#   3. encoder_frozen     (pretrained MGM encoder, MLP head)
#   4. encoder_unfrozen   (pretrained MGM, fine-tune end-to-end)
#   5. no_pretrain        (random encoder, MLP head)
#   6. linear_probe       (pretrained MGM, single Linear head — no MLP)
#
# Each condition × target uses 3 seeds, 1000-bootstrap CIs.
# Total runtime on A40: ~3 h (~$1.50 at $0.44/hr on-demand).
# ============================================================

set -e
cd "$(dirname "$0")"
mkdir -p logs

# Force unbuffered Python so logs appear in real time
export PYTHONUNBUFFERED=1

ENCODER_CKPT="${ENCODER_CKPT:-checkpoints/base_stage2/best.pt}"
TARGETS="${TARGETS:-As F NO3 U}"
N_SEEDS="${N_SEEDS:-3}"
EPOCHS="${EPOCHS:-30}"
BOOTSTRAP_N="${BOOTSTRAP_N:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE_ARG="${DEVICE_ARG:-}"   # e.g. "--device cuda" or empty for auto
FP16_ARG="${FP16_ARG:---fp16}"

if [ ! -f "$ENCODER_CKPT" ] && [ -z "$NO_PRETRAIN_ONLY" ]; then
    echo "ERROR: encoder ckpt not found at $ENCODER_CKPT"
    echo "Set ENCODER_CKPT=<path> or rerun pretraining."
    exit 1
fi

echo "=========================================================="
echo "Classification sweep started: $(date)"
echo "Encoder ckpt: $ENCODER_CKPT"
echo "Targets:      $TARGETS"
echo "Seeds:        $N_SEEDS  Epochs: $EPOCHS  Bootstrap: $BOOTSTRAP_N"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "=========================================================="

# Idempotent runner: skip a step if its output JSON already exists.
# Each python invocation writes results/classify_<lower(target)>_<tag>.json
# so we derive the expected output path from --target_param + --out_tag
# (or implicit tag).
run_step() {
    local desc="$1"; shift
    local target="$1"; shift
    local tag="$1"; shift
    local out_path="results/classify_${target,,}_${tag}.json"
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

    # --- Baselines (CPU-bound, fast) ---
    run_step "[$T] XGBoost baseline" "$T" "xgboost" \
        python src/11_baseline_xgboost.py \
            --target_param "$T" --model xgboost \
            --n_seeds "$N_SEEDS" --bootstrap_n "$BOOTSTRAP_N"

    run_step "[$T] LogReg baseline" "$T" "logreg" \
        python src/11_baseline_xgboost.py \
            --target_param "$T" --model logreg \
            --n_seeds "$N_SEEDS" --bootstrap_n "$BOOTSTRAP_N"

    # --- Encoder conditions (GPU) ---
    run_step "[$T] frozen MGM encoder + MLP" "$T" "mlp_frozen" \
        python src/10_classify_exceedance.py \
            --encoder_size base --encoder_ckpt "$ENCODER_CKPT" \
            --target_param "$T" --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --n_seeds "$N_SEEDS" --bootstrap_n "$BOOTSTRAP_N" \
            $FP16_ARG $DEVICE_ARG

    run_step "[$T] unfrozen MGM encoder + MLP" "$T" "mlp_unfrozen" \
        python src/10_classify_exceedance.py \
            --encoder_size base --encoder_ckpt "$ENCODER_CKPT" \
            --target_param "$T" --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --n_seeds "$N_SEEDS" --bootstrap_n "$BOOTSTRAP_N" \
            --unfreeze_encoder $FP16_ARG $DEVICE_ARG

    run_step "[$T] no-pretrain (random encoder) + MLP" "$T" "mlp_nopretrain" \
        python src/10_classify_exceedance.py \
            --encoder_size base \
            --target_param "$T" --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --n_seeds "$N_SEEDS" --bootstrap_n "$BOOTSTRAP_N" \
            --no_pretrain $FP16_ARG $DEVICE_ARG

    run_step "[$T] linear probe on frozen MGM encoder" "$T" "linprobe_frozen" \
        python src/10_classify_exceedance.py \
            --encoder_size base --encoder_ckpt "$ENCODER_CKPT" \
            --target_param "$T" --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --n_seeds "$N_SEEDS" --bootstrap_n "$BOOTSTRAP_N" \
            --linear_probe $FP16_ARG $DEVICE_ARG
done

echo ""
echo "=========================================================="
echo "Sweep complete: $(date)"
echo "Results: results/classify_*.json"
echo "=========================================================="
