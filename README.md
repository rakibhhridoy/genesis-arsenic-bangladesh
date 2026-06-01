# GENESIS — When foundation models fail

**Pretraining a 9-million-parameter transformer on 2 million global geochemistry samples does not improve Bangladesh arsenic-risk classification over logistic regression.**

This repository contains the code, per-seed result JSONs, figures, and manuscript
for the GENESIS study. The central finding is **negative**: a self-supervised
Transformer encoder (GENESIS) pretrained with Masked Geochemical Modeling on
2,087,970 harmonized global water-chemistry vectors fails to outperform L2-regularized
logistic regression on Bangladesh arsenic-exceedance transfer (WHO 10 µg/L), at any
of three model sizes (Small 1.47M, Base 8.99M, Large 48.4M) under any adaptation
configuration. We trace the failure to a redox-regime mismatch between the oxic
pretraining corpus (median Eh +245 mV) and the strongly reducing Bangladesh aquifers
(median Eh −35 mV).

## Headline result (BD-As exceedance AUC, frozen-encoder + MLP head)

| Method | BD AUC |
|---|---|
| **LogReg, chemistry-only** (baseline) | **0.694 ± 0.009** |
| Large (48.4M) frozen + MLP | 0.690 ± 0.022 |
| Small (1.47M) frozen + MLP | 0.632 ± 0.004 |
| Base (8.99M) frozen + MLP | 0.559 ± 0.019 |

No pretrained configuration's 95% CI lies entirely above the chemistry-only LogReg
interval; pretraining benefit is non-monotonic in capacity.

## Repository layout

```
src/                 Numbered pipeline 01–16 (curation → pretrain → classify → figures)
  model/             GENESIS encoder, datasets, latent diffusion
run_*.sh             Orchestration scripts (RunPod + local)
results/             Per-seed metric JSONs (verify exact paper numbers here)
figures/             Generated paper figures
manuscript/          LaTeX manuscript + supplementary
manuscript_snippets/ Auto-generated LaTeX table fragments
RUNPOD.md            RunPod GPU pretraining playbook
VERIFY_HIGH_LR.md    Protocol for the lr=5e-4 verification run
requirements.txt     Pinned Python environment
DATA.md              How to obtain the data + checkpoints (released separately)
```

## Pipeline (`src/`)

| Stage | Scripts |
|---|---|
| Data curation | `01_*`, `02_*`, `03_*`, `04_build_genesis_db.py` |
| Auxiliary GEE features | `05_extract_aux_features_gee.py` |
| Pretraining (MGM) | `04_pretrain_genesis.py` (Stage 1 + Stage 2) |
| Diffusion / regression (abandoned, R²<0) | `05_finetune_diffusion.py`, `06_evaluate_transfer.py` |
| Classification (headline) | `10_classify_exceedance.py`, `11_baseline_xgboost.py`, `12_compile_classify_table.py` |
| Analysis & figures | `13_logreg_analysis.py`, `14_distribution_shift.py`, `15_scaling_figure.py`, `16_hp_sensitivity_postprocess.py` |

## Reproducing the results

The data tensors and encoder checkpoints are **not** in this repo (size); see
**[DATA.md](DATA.md)**. Once `data/processed/` and `checkpoints/` are in place:

```bash
pip install -r requirements.txt

# Classification sweep (produces the BD-As AUC table) — from a fixed encoder checkpoint
bash run_classify_sweep.sh           # or run_classify_sweep_small.sh / _large.sh
python src/12_compile_classify_table.py

# Baselines, mechanism, figures
python src/11_baseline_xgboost.py
python src/13_logreg_analysis.py --no_latlon     # chemistry-only headline
python src/14_distribution_shift.py
python src/15_scaling_figure.py
```

GPU pretraining from scratch is documented in [RUNPOD.md](RUNPOD.md).

## Reproducibility caveats

- **Encoder pretraining is not bitwise deterministic** — it uses fp16 AMP on CUDA
  without deterministic kernels. A fresh re-train yields a slightly different encoder.
  All reported numbers reproduce from the **released checkpoints**, not from re-training.
- **Classification / analysis is seed-controlled** (seeds 42–44; `random_state` and
  bootstrap seeded), so the statistical results reproduce exactly from a fixed checkpoint.
- **Some raw data is license/auth-gated** (GEMStat, Google Earth Engine, live USGS NWIS);
  start from the released harmonized `data/processed/` tensors. See [DATA.md](DATA.md).

## Citation

Citation details will be added on publication. See `manuscript/manuscript.tex`.

## License

Code is released under the MIT License (see `LICENSE`). Data sources retain their
original licenses — see [DATA.md](DATA.md).
