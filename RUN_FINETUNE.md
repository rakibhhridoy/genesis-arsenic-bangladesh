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

## 1. PILOT FIRST (~15–40 min, ~$0.20) — the decisive 6 cells
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
