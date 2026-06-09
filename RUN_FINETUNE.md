# LORO fine-tuning on RunPod — run instructions

The decisive test: does an **end-to-end fine-tuned** GENESIS encoder beat Random
Forest on the leave-one-region-out benchmark? (Frozen does not — see
`results/tournament_summary_large.json`.)

## What's in this bundle
- `src/29_loro_finetune.py` — the fine-tune script (incremental/resumable writes)
- `src/27_loro_tree_baselines.py`, `src/28_encoder_vs_rf_tournament.py` — the baseline/tournament scripts (for re-running if needed)
- `src/10_classify_exceedance.py`, `src/19_region_transfer_encoder.py`, `src/model/` — imported dependencies
- `data/processed/` — all tensors the script needs (~28M)
- `results/region_transfer_tree_baselines.json` — RF/XGB baselines for the head-to-head column

## NOT in this bundle (must already be on the pod)
- The Large Stage-2 checkpoint (553M). It should already be on the pod from the
  Large run. Confirm the path and pass it via `--ckpt`. If missing, re-pull it
  (e.g. `pull_large_stage2.sh`) before running.

## Setup on the pod
```bash
# unpack into the repo root (where src/, data/, results/ live)
cd /workspace/Paper5
tar -xzf loro_finetune_bundle.tar.gz
pip install xgboost scikit-learn pandas pyarrow   # if not already present
export PYTHONUNBUFFERED=1
```

## MULTI-SEED RUN (the robustness test — 2026-06-08)
Single-seed full run is done (`results/loro_finetune_large.json`): fine-tuned mean
0.756 vs RF 0.735, but a statistical TIE (Wilcoxon p=0.11) and the only live
positive is a target-specific As/PO4 edge that single-seed cannot validate. The
distance-conditional law is dead vs RF (rho=0.00). So the decisive remaining test
is multi-seed.

**Recommended cheap test — 5 seeds on the 18 As+PO4 cells (~2 h, ~$1.50):**
```bash
cd /workspace
export PYTHONUNBUFFERED=1
setsid nohup python src/29_loro_finetune.py --encoder_size large \
    --ckpt checkpoints/large_stage2/best.pt \
    --seeds 42,43,44,45,46 \
    --cells "PO4/France,PO4/USA,PO4/EU,PO4/GEMStat:Italy,PO4/GEMStat:India,PO4/GEMStat:Lithuania,PO4/GEMStat:Poland,PO4/GEMStat:Mexico,PO4/GEMStat:Greece,PO4/GEMStat:Netherlands (-the ),PO4/Bangladesh,As/France,As/USA,As/EU,As/GEMStat:Italy,As/GEMStat:Mexico,As/GEMStat:Poland,As/Bangladesh" \
    --out results/loro_ft_aspo4_5seed.json \
    > /workspace/multiseed.log 2>&1 < /dev/null &
disown
```
**Full 3-seed benchmark (all 47 cells, ~6 h, ~$3) — only if you want the complete table:**
```bash
setsid nohup python src/29_loro_finetune.py --encoder_size large \
    --ckpt checkpoints/large_stage2/best.pt --seeds 42,43,44 \
    --out results/loro_finetune_large_multiseed.json \
    > /workspace/multiseed.log 2>&1 < /dev/null &
disown
```
Resumable: each (cell,seed) is skipped if already in `--out`. The run prints a
seed-averaged paired Wilcoxon FT-vs-RF + per-target breakdown at the end. Watch:
`tail -f /workspace/multiseed.log`. Verify it launched: `sleep 5 && tail -5 /workspace/multiseed.log && pgrep -af 29_loro`.

## 1. (single-seed) PILOT — the decisive 6 cells
```bash
python src/29_loro_finetune.py --encoder_size large \
    --ckpt checkpoints/large_stage2/best.pt \
    --cells "U/USA,PO4/GEMStat:Italy,PO4/GEMStat:India,As/Bangladesh,Fe/Bangladesh,U/GEMStat:Canada" \
    --out results/loro_finetune_large_pilot.json
```
Each cell prints: `ft=<fine-tuned AUC>  rf=<RF AUC>  ... val=<val AUC>  <elapsed>s  [FT BEATS RF]`.

**DECISION GATE — read the pilot before paying for the full run:**
- The **first cell's `elapsed_sec` × 47** ≈ your full-run time.
- If fine-tuning does **not** beat RF on **U/USA** (the cell where pretraining
  looked strongest), stop — the full sweep won't change the verdict, and the
  honest paper is the controlled negative.
- If it **does** beat RF on U/USA → run the full benchmark below.

## 2. FULL benchmark (only if pilot justifies it, ~2–5 h, ~$1–2.50)
```bash
python src/29_loro_finetune.py --encoder_size large \
    --ckpt checkpoints/large_stage2/best.pt \
    --out results/loro_finetune_large.json
```
Resumable: re-running skips finished cells (survives a disconnect). Writes after
every cell, so partial runs still produce usable output.

## Knobs
- `--lr 2e-5` (default) is a gentle fine-tune for a pretrained encoder; try `1e-4`
  if val AUC is flat / underfitting.
- `--epochs 20 --patience 4` — early-stops on held-in val AUC.
- `--batch_size 128` — raise on a big GPU to speed it up.

## Send the results back
```bash
# on the pod
COPYFILE_DISABLE=1 tar --no-xattrs --exclude='._*' -czf loro_ft_results.tar.gz results/loro_finetune_large*.json
runpodctl send loro_ft_results.tar.gz
# then `runpodctl receive <code>` locally
```
