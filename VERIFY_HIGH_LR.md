# High-lr Verification Run — RunPod Instructions

## Why this run

The HP sensitivity sweep on M1 (Small × 50K subsample × 20 epochs) found
that one configuration (lr=5e-4) produced a frozen+MLP encoder that
transfers to BD-As at AUC 0.732 ± 0.012 — outside the chem-only LogReg
confidence interval (0.694 ± 0.009). The main-paper headline claim ("no
pretrained-encoder configuration beats LogReg") depends on whether this
finding survives at the full-corpus 100-epoch scale.

This run trains Small at full corpus × 100 epochs × lr=5e-4 (otherwise
matching the main paper's Small protocol exactly), then classifies BD-As
frozen+MLP × 3 seeds. Expected runtime: ~3 h on A40 spot.

---

## Mac-side prep

```bash
cd "/Volumes/SSD Rx/Research/GroundWater/Paper5"

# Bundle includes run_verify_high_lr.sh
tar -czf /tmp/runpod_upload.tar.gz \
    --exclude='._*' --no-xattrs \
    -C runpod_upload .

# Ship to pod (note: requires runpodctl on Mac; brew install runpod/runpodctl/runpodctl)
runpodctl send /tmp/runpod_upload.tar.gz
# Take note of the receive code
```

---

## Pod-side run

Spin up an **A40 spot** pod with the RunPod PyTorch 2.x template, 50 GB
container + 50 GB volume. Open the web terminal.

```bash
# Install runpodctl if not already
curl -sSL https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 \
    -o /usr/local/bin/runpodctl && chmod +x /usr/local/bin/runpodctl

# Receive + extract
cd /workspace
runpodctl receive <code-from-mac>
tar xzf runpod_upload.tar.gz --no-same-owner
rm runpod_upload.tar.gz

# One-time environment setup (installs sklearn, xgboost, scipy)
bash runpod_setup.sh

# Launch the verification run with SIGHUP-immune setsid+nohup
# (this is the pattern that fixed the prior 'stuck at epoch 21' interruptions)
mkdir -p logs
setsid nohup ./run_verify_high_lr.sh > logs/verify_high_lr.log 2>&1 < /dev/null &
disown

# Monitor
tail -f logs/verify_high_lr.log
```

If the terminal disconnects, the run continues. Reconnect with
`tail -f logs/verify_high_lr.log`.

---

## When complete

The pod will print a verdict line at the bottom of the log:
- `→ VERIFIES the HP-sweep finding` (BD AUC lower CI > 0.703) — the
  effect is real at full corpus; the paper's headline claim needs a
  footnote and we may have a positive finding for the encoder.
- `→ REFUTES the HP-sweep finding` (BD AUC upper CI < 0.685) — the
  HP-sweep result was a subsample artifact; clean negative survives.
- `→ INCONCLUSIVE` — overlaps with LogReg CI; reportable as
  "marginally above LogReg, not significantly different".

Ship the result back:

```bash
# On the pod:
tar -czf /tmp/verify_high_lr_result.tar.gz \
    results/classify_as_verify_high_lr_mlp_frozen.json \
    checkpoints/small_stage1_high_lr/history.json \
    logs/verify_high_lr.log

runpodctl send /tmp/verify_high_lr_result.tar.gz
# Note the code

# On the Mac:
cd "/Volumes/SSD Rx/Research/GroundWater/Paper5"
runpodctl receive <code>
tar xzf verify_high_lr_result.tar.gz
```

Then run `python3 src/16_hp_sensitivity_postprocess.py` locally to update
the supplementary figure (post-processor already handles the verify_*
config if you rename the JSON to `classify_as_hp_high_lr_full_mlp_frozen.json`
— or I can wire that in when the result lands).

---

## Defensive measures already baked in

- `setsid nohup ... < /dev/null & disown` — SIGHUP immune
- `PYTHONUNBUFFERED=1` — log streams immediately, no buffering
- `--ckpt_every_steps 1000` — step-level checkpoint every ~4 minutes
- `--resume auto` if `checkpoints/small_stage1_high_lr/latest.pt` exists —
  the script will auto-resume if interrupted
- Bundle ships with `--no-xattrs --no-same-owner` to avoid AppleDouble issues
- `--num_workers 3` (not 4; 4 crashed A40 historically)

If the pod terminates mid-run, just restart with the same command — it
will pick up from the latest step-level checkpoint.

---

## Expected outputs

- `checkpoints/small_stage1_high_lr/best.pt` (~17 MB, encoder weights)
- `checkpoints/small_stage1_high_lr/history.json` (100-epoch loss curve)
- `results/classify_as_verify_high_lr_mlp_frozen.json` (3-seed metrics)
- `logs/verify_high_lr.log` (full training log)

Combined tarball <50 MB → safe for runpodctl send back to Mac.
