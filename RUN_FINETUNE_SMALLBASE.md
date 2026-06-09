# Cross-size fine-tune (Small + Base) on RunPod

**Goal:** replicate the redox-coupled fine-tune win across encoder sizes. The
headline result (fine-tuning beats Random Forest on As/Fe/Mn/PO$_4$, p=0.0096)
was established on **Large only** (`results/loro_finetune_large_multiseed.json`).
Running Small + Base lets the paper say the advantage *replicates across the 33×
parameter range* — the same robustness language used for the attention/uranium
findings.

## Bundle
`loro_finetune_smallbase_bundle.tar.gz` (120 MB) is **self-contained** — unlike the
Large run, both checkpoints fit inside it:
- `src/29_loro_finetune.py` + imported deps (`10`, `19`, `27`, `28`, `model/`)
- `data/processed/` — all 7 tensors the script needs
- `results/region_transfer_tree_baselines.json` — the RF/XGB head-to-head column
- `checkpoints/small_stage2/best.pt` (17 M) + `checkpoints/base_stage2/best.pt` (103 M)

Pre-verified locally: both checkpoints load into `GENESISForMGM(size)` with
**0 missing / 0 unexpected keys** (254/254 small, 302/302 base) — no silent
random-weight load.

## 1. Send to the pod
From this machine (pod must be running; get its id with `runpodctl get pod`):
```bash
runpodctl send loro_finetune_smallbase_bundle.tar.gz
# -> copy the one-time code, then on the pod: runpodctl receive <code>
```

## 2. Unpack (at /workspace — the script uses relative paths data/, checkpoints/, results/)
```bash
cd /workspace
tar -xzf loro_finetune_smallbase_bundle.tar.gz
pip install xgboost scikit-learn pandas pyarrow scipy   # if not already present
export PYTHONUNBUFFERED=1
df -h /                # sanity: container root overlay is the real disk limit (see RUNPOD ops lesson)
```

## 3. Run both sizes (3 seeds × 47 cells each, resumable, detached)
Smaller encoders are faster per cell than Large (~6 h); expect ~2–4 h each.
Chained so Base starts automatically after Small:
```bash
cd /workspace
setsid nohup bash -c '
python src/29_loro_finetune.py --encoder_size small \
    --ckpt checkpoints/small_stage2/best.pt --seeds 42,43,44 \
    --out results/loro_finetune_small_multiseed.json &&
python src/29_loro_finetune.py --encoder_size base \
    --ckpt checkpoints/base_stage2/best.pt --seeds 42,43,44 \
    --out results/loro_finetune_base_multiseed.json
' > /workspace/ft_smallbase.log 2>&1 < /dev/null &
disown
```
Verify it launched (do NOT use the `ENCODER_CKPT=… setsid` form — it silently no-ops):
```bash
sleep 8 && tail -8 /workspace/ft_smallbase.log && pgrep -af 29_loro
```
Watch progress: `tail -f /workspace/ft_smallbase.log`. Each (cell, seed) is skipped
if already in `--out`, so a disconnect never loses completed work — just re-run the
same command. The end of each size prints a seed-averaged paired Wilcoxon FT-vs-RF
+ per-target breakdown.

## 4. Pull results back
```bash
runpodctl send results/loro_finetune_small_multiseed.json
runpodctl send results/loro_finetune_base_multiseed.json
# receive both on the Mac into results/
```

## 5. Analyze (back on the Mac) — does the redox win replicate?
Re-run the headline stats per size (src/30 now takes `--ft`/`--enc`/`--out`; it
groups As/Fe/Mn/PO$_4$ vs RF and runs the paired Wilcoxon). RF baselines are
encoder-independent, so this is FT(size) vs the same RF:
```bash
python src/30_redox_dissociation.py \
    --ft results/loro_finetune_small_multiseed.json \
    --enc results/region_transfer_encoder_small.json \
    --out results/redox_dissociation_small.json

python src/30_redox_dissociation.py \
    --ft results/loro_finetune_base_multiseed.json \
    --enc results/region_transfer_encoder.json \
    --out results/redox_dissociation_base.json
```
(`region_transfer_encoder.json` is the Base frozen+RF file; `_small` is Small.)
If Small and Base both show a positive redox-suite Δ(FT−RF), the manuscript line
becomes "replicates across all three sizes." If not, that itself is an honest,
informative size-dependence finding worth reporting.

**Cost estimate:** ~$3–4 total for both at A40 spot pricing. Stop the pod when the
two result JSONs are pulled.
