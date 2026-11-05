# Groundwater chemistry inference, transferability and physical consistency evaluation with machine learning

Code and results for the preprint
**"Groundwater chemistry inference, transferability and physical consistency evaluation with machine
learning across global monitoring records"** by Md Rakib Hasan, Shoumik Zubyer and Anwar Zahid.
Preprint on Research Square, https://doi.org/10.21203/rs.3.rs-11578171/v1 (not peer reviewed).

Nine methods are compared on identical splits drawn from 2.1 million harmonized water-chemistry
samples, ranging from logistic regression, spatial interpolation and a raster-only hazard model to
random forests, gradient boosting, the tabular foundation model TabPFN, a groundwater-pretrained
Transformer (GENESIS) and a simulated quantum kernel. Each method is scored on four axes.

| Axis | Question | Principal result |
|---|---|---|
| A, inferability | Which unmeasured parameters can routine data recover? | Major ions in unseen wells reach R² 0.83–0.95, arsenic at most 0.53 |
| B, transfer | Does skill survive in a region the model never saw? (47 leave-one-region-out cells) | Measured chemistry reaches AUC 0.71–0.76, environmental rasters alone 0.58 |
| C, calibration | Does nominal 90% exceedance coverage hold out of region? | Every method falls short, and the best-ranking model misses the most exceedances (35%) |
| D, physical consistency | Do reconstructed waters obey mineral equilibria and charge balance? | Accurate models reproduce both on familiar water and fail together in reducing Bangladesh aquifers |

The simulated quantum kernel matches its classical RBF counterpart (out-of-region AUC 0.708 for both).

## Evaluation scripts

| Scripts | Purpose |
|---|---|
| `src/52`–`55` | Raster-only hazard transfer, within-source transfer and detection-limit controls |
| `src/56`, `src/57` | Inferability ladder and minimum measurement set (axis A) |
| `src/58` | Thermodynamic consistency of reconstructed waters (axis D) |
| `src/59`, `src/60` | Per-sample leave-one-region-out predictions and Mondrian conformal coverage (axes B, C) |
| `src/61` | Baseline regressors for axes A and D |
| `src/65`–`68` | Simulated quantum kernel, its controls, summary, figure and regression runs |
| `src/69` | Exports the fixed partitions and checks them against every saved prediction |
| `src/62`, `src/63`, `src/67` | Figures 2–5, Figure 1 and Figure 6 |
| `src/64` | Supplementary tables |

Summary results are in `results/` and figures are in `figures/fig_eval_*`. Per-sample predictions and
data tensors are too large for git and are archived on Zenodo, see [DATA.md](DATA.md).

## Reproducing the results

Regenerating the scores and figures from the archived predictions needs no GPU.

```bash
pip install -r requirements-evaluation.txt
# unzip the Zenodo evaluation archive at the repository root, then
python src/60_conformal_coverage.py
python src/66_quantum_kernel_summary.py
python src/62_evaluation_tables_figures.py
python src/67_fig_quantum_kernel.py
```

Rerunning the models also needs `data/processed/` and `checkpoints/` from the main Zenodo archive.
The fine-tuned Transformer and TabPFN need a GPU in practice.

```bash
python src/59_loro_conformal_preds.py --method rf --seeds 42,43,44
python src/59_loro_conformal_preds.py --method finetune --encoder_size base \
    --ckpt checkpoints/base_stage2/best.pt --seeds 42,43,44
python src/65_quantum_kernel_loro.py --method qkernel --seeds 42,43,44
python src/68_quantum_kernel_regression.py --mode both
```

Partitions are seeded from target, region and seed, so they regenerate from the code, and
`results/evaluation_splits.npz` in the evaluation archive stores them as row indices. Encoder
pretraining uses fp16 AMP on CUDA and is not bitwise deterministic, so reported numbers reproduce
from the released checkpoints rather than from a fresh re-train.

## Earlier work in this repository

The remaining scripts (`src/01`–`51`), `run_*.sh`, `RUNPOD.md` and the June 2026 results cover data
curation, GENESIS pretraining and the earlier foundation-model study. Curation is summarized in
[DATA.md](DATA.md) and pretraining in [RUNPOD.md](RUNPOD.md). The June README and its claims are
preserved at tag `v1-foundation-model`. They predate this evaluation and are superseded by it.
A later comparison showed that the fine-tuned Transformer ties TabPFN, a tabular model without
domain pretraining.

## Citation

Hasan, M. R., Zubyer, S. and Zahid, A. Groundwater chemistry inference, transferability and physical
consistency evaluation with machine learning across global monitoring records. Preprint at Research
Square, https://doi.org/10.21203/rs.3.rs-11578171/v1 (2026).

## License

Code is released under the MIT License (see `LICENSE`). Data sources retain their original licenses,
see [DATA.md](DATA.md).
