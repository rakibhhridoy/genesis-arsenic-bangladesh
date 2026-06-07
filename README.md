# GENESIS — A geochemical foundation model for groundwater

**When does self-supervised pretraining help? A conditional transfer law: pretraining gains on uranium but not arsenic under extreme distribution shift.**

This repository contains the code, per-seed result JSONs, figures, and manuscript
for the GENESIS study. We pretrain a Transformer encoder with Masked Geochemical
Modeling (MGM) on **2,087,970** harmonized global water-chemistry vectors, at three
sizes (Small 1.47M, Base 8.99M, Large 48.4M), and ask — on a leave-one-region-out
benchmark of **7 regions × 7 contaminant targets (47 cells)** — when the frozen
representation beats strong, honest baselines (L2 logistic regression, XGBoost) on
identical zero-shot splits.

The value of pretraining is **conditional**, and the conditions are predictable:

- **Pretraining wins for nonlinear contaminants under modest shift.** The clearest
  case is **uranium**: chemistry-only logistic regression transfers *worse than
  random* to the U.S. (AUC **0.340**) because uranium–major-ion associations reverse
  sign across regions, while the encoder recovers genuine signal (**0.671–0.678**,
  replicating across all three sizes). Phosphate and iron show the same pattern.
- **Pretraining fails under extreme distribution shift.** Bangladesh's strongly
  reducing aquifers are the extreme corner of the benchmark (distribution distance
  **2.7**, more than double any other region). There **no** pretrained configuration
  at any size beats logistic regression, and arsenic transfer collapses toward random.

We tie both poles to a **single mechanism, read directly from inside the model**: its
attention and masked reconstruction encode the oxic major-ion couplings the corpus
contains (median Eh **+245 mV**) but never the reducing-aquifer As–Fe–Eh–PO₄ coupling
that governs Bangladesh arsenic (median Eh **−35 mV**) — and a 33× increase in
parameters does not change this. A controlled synthetic experiment separates the
cause: **covariate shift** (inputs move) leaves transfer intact, whereas **concept
shift** (the input→target relationship changes) collapses it.

## Headline: the conditional transfer law (leave-one-region-out)

Frozen-encoder linear probe vs. chemistry-only logistic regression on identical
zero-shot splits (`src/19_region_transfer_encoder.py`):

| Cell | chem-only LogReg | GENESIS encoder | Encoder wins? |
|---|---|---|---|
| **U / USA** (nonlinear, modest shift) | 0.340 *(worse than random)* | **0.671–0.678** (all 3 sizes) | ✅ |
| **PO₄ / Italy** | 0.575 | 0.762 | ✅ (all sizes) |
| **As / Bangladesh** (extreme shift) | 0.565 | 0.402–0.599 | ❌ (no size) |

Across the 47 cells the encoder's advantage is largest exactly where the linear
baseline is weakest (Spearman −0.61), and it rescues 5 of the 6 worse-than-random
cells — the lone exception being an arsenic cell. The encoder **never** beats logistic
regression on arsenic and **never** helps any Bangladesh target, at any size.

## Repository layout

```
src/                 Numbered pipeline (curation → pretrain → transfer → mechanism → figures)
  model/             GENESIS encoder, datasets, latent diffusion
run_*.sh             Orchestration scripts (RunPod + local)
results/             Per-seed metric JSONs (verify exact paper numbers here)
figures/             Generated paper figures
manuscript/          LaTeX manuscript + supplementary
manuscript_snippets/ Auto-generated LaTeX table fragments
RUNPOD.md            RunPod GPU pretraining playbook
DATA.md              How to obtain the data + checkpoints (released separately)
```

## Pipeline (`src/`)

| Stage | Scripts |
|---|---|
| Data curation & integration | `01_*`, `02_*`, `03_*`, `04_build_genesis_db.py` |
| Auxiliary GEE features | `05_extract_aux_features_gee.py` |
| Pretraining (MGM, Stage 1 + 2) | `04_pretrain_genesis.py` |
| Bangladesh-As classification | `10_classify_exceedance.py`, `11_baseline_xgboost.py`, `12_compile_classify_table.py` |
| **Leave-one-region-out transfer law** | `18_region_transfer.py` (LogReg), `19_region_transfer_encoder.py` (encoder) |
| **Mechanism (model-internal)** | `07_attention_analysis.py` + `26_fig7_attention_scaling.py` (attention), `22_reconstruction_probe.py` (MGM reconstruction / OOD) |
| **Controlled experiment** | `23_synthetic_law.py` (covariate vs. concept shift) |
| Distribution shift & figures | `13_logreg_analysis.py`, `14_distribution_shift.py`, `15_scaling_figure.py`, `16_hp_sensitivity_postprocess.py` |
| Second-domain robustness check | `25_superconductor_transfer.py` (UCI, not in manuscript) |

## Reproducing the results

The data tensors and encoder checkpoints are **not** in this repo (size); see
**[DATA.md](DATA.md)**. Once `data/processed/` and `checkpoints/` are in place:

```bash
pip install -r requirements.txt

# The conditional transfer law (47-cell LORO benchmark)
python src/18_region_transfer.py            # chemistry-only LogReg baseline
python src/19_region_transfer_encoder.py    # frozen-encoder linear probe
#   --encoder_size {small,base,large} --ckpt <stage2 best.pt>

# Mechanism (reads the released attention matrices)
python src/26_fig7_attention_scaling.py     # Fig 7: couplings vs. encoder scale
python src/22_reconstruction_probe.py       # MGM reconstruction hierarchy + OOD

# Controlled experiment, distribution shift, scaling
python src/23_synthetic_law.py
python src/14_distribution_shift.py
python src/15_scaling_figure.py
```

GPU pretraining from scratch is documented in [RUNPOD.md](RUNPOD.md).

## Reproducibility caveats

- **Encoder pretraining is not bitwise deterministic** — it uses fp16 AMP on CUDA
  without deterministic kernels. A fresh re-train yields a slightly different encoder.
  All reported numbers reproduce from the **released checkpoints**, not from re-training.
- **Transfer / analysis is seed-controlled** (seeds 42–44; `random_state` and bootstrap
  seeded), so the statistical results reproduce exactly from a fixed checkpoint.
- **Encoder embedding on Apple Silicon (MPS)**: `src/19` releases the MPS allocator
  pool per batch (`torch.mps.empty_cache()`) and uses a small batch size — a full
  47-cell embed otherwise pins enough unified memory on an 8 GB machine to stall the
  desktop. On CUDA/CPU this is a no-op.
- **Some raw data is license/auth-gated** (GEMStat, Google Earth Engine, live USGS
  NWIS); start from the released harmonized `data/processed/` tensors. See [DATA.md](DATA.md).

## Citation

Citation details will be added on publication. See `manuscript/manuscript.tex`.

## License

Code is released under the MIT License (see `LICENSE`). Data sources retain their
original licenses — see [DATA.md](DATA.md).
