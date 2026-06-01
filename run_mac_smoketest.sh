#!/bin/bash
# ============================================================
# Mac Smoke Test — GENESIS MGM pretraining
# ============================================================
# Goal: prove the pretrain pipeline runs end-to-end on an 8 GB Mac
#       without crashing, and that step-level checkpoints are saved
#       and resumable. NOT a real training run.
#
# Settings chosen for stability on 8 GB unified memory:
#   - size small           : ~1–2 M param model
#   - subsample 50,000     : ~50× smaller than full corpus
#   - batch 128            : fits MPS; raise only if RAM headroom
#   - num_workers 0        : avoids Mac spawn-multiprocessing OOM
#   - ckpt_every_steps 100 : frequent safety net
#   - 2 epochs             : enough to verify loss decreases + ckpt + resume
#
# To terminate cleanly at any time:  Ctrl-C once (SIGINT handler saves latest.pt)
# To resume from where it stopped:   add --resume auto
#
# Wall time expectation: ~30–90 min depending on MPS speed.
# ============================================================

set -e
cd "$(dirname "$0")"

LOG=logs/mac_smoketest.log
mkdir -p logs checkpoints

echo "=============================================="
echo "GENESIS Mac smoke test — $(date)"
echo "Writing log to: $LOG"
echo "=============================================="

# caffeinate -i  = prevent idle sleep while training
# nohup          = survives terminal close
# Remove `nohup` + `&` at the end if you want to watch it attached.
caffeinate -i nohup python src/04_pretrain_genesis.py \
    --size small \
    --stage 1 \
    --subsample 50000 \
    --epochs 2 \
    --batch_size 128 \
    --lr 3e-4 \
    --mask_ratio 0.2 \
    --patience 5 \
    --num_workers 0 \
    --ckpt_every_steps 100 \
    > "$LOG" 2>&1 &

PID=$!
echo "Started PID=$PID"
echo "Tail log with:   tail -f $LOG"
echo "Stop safely:     kill -INT $PID   (saves latest.pt before exit)"
echo "Resume:          bash run_mac_smoketest.sh  (reruns) or add --resume auto"
