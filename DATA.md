# Data & checkpoints

Large artifacts (harmonized data tensors, encoder checkpoints, raw sources) are
**not stored in git** because of size. They are released separately.

> **Zenodo archive DOI:** https://doi.org/10.5281/zenodo.20631197
> Archive file: `zenodo_genesis_arsenic_bangladesh_v3.zip` (~1.0 GB compressed).

## What's in the separate release (v3)

The release is **license-clean and derived-data-only**: it contains the harmonized
tensors and the encoders/results needed to reproduce every number in the
manuscript, but **no raw GEMStat** and **no `genesis.duckdb`** (the DB contains
raw GEMStat rows that cannot be redistributed).

v3 adds, over v2: the **Large (48.4M) Stage-2 encoder**, the **cross-size
end-to-end fine-tune** results (`loro_finetune_{small,base,large}_multiseed.json`)
and their headline-stat aggregations (`redox_dissociation{,_small,_base}.json`),
the **random-forest / gradient-boosted-tree baselines and tournament**
(`region_transfer_tree_baselines.json`, `tournament_summary_large.json`,
`encoder_vs_rf_tournament_large.json`), the figures `fig13_finetune_vs_rf.png`
and `fig14_crosssize_replication.png`, scripts `src/27`–`src/32`, and the
Bangladesh primary field data.

| Artifact | Approx. size | Reproduces |
|---|---|---|
| `data/processed/` — harmonized tensors (`genesis_pretrain*.pt`, `genesis_temporal_pairs*.{pt,parquet}`, `genesis_held_out_bd*.{pt,parquet}`, `normalization_stats.json`) | ~365 MB | distribution shift, LORO, reconstruction/attention probes, fine-tune |
| `data/processed/bangladesh_all_samples.parquet`, `bangladesh_temporal_pairs.parquet` | ~0.3 MB | the two-campaign Bangladesh primary field dataset (held-out evaluation set) |
| `checkpoints/{small,base,large}_stage2/best.pt` | 17 MB, 103 MB, 553 MB | attention, reconstruction, aux-reliance, layer-wise, encoder-LORO, and end-to-end fine-tune at all three sizes |
| `checkpoints/classify_*` (per-seed heads, all targets) | ~0.4 GB | headline BD-As classification table + per-target in-distribution tables |
| `results/` — all per-seed JSONs incl. `region_transfer*.json`, `loro_finetune_*_multiseed.json`, `redox_dissociation*.json`, tournament/baseline JSONs, `reconstruction_probe.json`, `aux_reliance.json`, `layerwise_decodability.json`, `dist_shift.json`, `calibration_*` | ~23 MB | every figure and reported number |
| `figures/` — all main and Extended Data figures | ~5 MB | the published figures |

**Not in the release** (reproducible only with provider access or the pod):
- `genesis.duckdb` and raw source dumps — excluded for license/size (see below).
  Exploration scripts `src/20`, `src/21` read the DB and therefore will not run
  from the release; they are not needed for any manuscript number.
- The Stage-1 (pre-fine-tune) and diffusion checkpoints — not needed for any
  reported number; the Stage-2 encoders above suffice.

To reproduce the manuscript numbers, unzip at the repository root so the
`Path(__file__).parent.parent` references in `src/` resolve, then
`pip install -r requirements.txt` and run the scripts in README order. The LORO
benchmark (`src/18`, `src/19`) and all probes read only `data/processed/` tensors
— no database required.

## Raw data provenance & access constraints

Curation (`src/01_*`–`src/04_build_genesis_db.py`) is **not fully push-button** for a
third party because several upstream sources are license- or auth-gated:

- **GEMStat** (UN GEMS/Water) — requires a data request/agreement; cannot be redistributed.
  Obtain directly from https://gemstat.org.
- **Google Earth Engine auxiliary features** (`src/05_extract_aux_features_gee.py`) —
  requires a Google Earth Engine account and `earthengine authenticate`.
- **USGS NWIS** (`src/01c_curate_usgs_nwis.py`) — pulled from live water-services APIs;
  results may drift over time as the database is updated.
- **Podgorski et al.** global arsenic dataset, **BRGM ADES**, **EEA WISE**, **EA Water
  Quality**, **GROW** — public; download scripts in `src/02_*`/`src/03_*`.
- **Bangladesh** held-out set — the authors' own two-campaign primary field
  dataset (a 2012–2013 BWDB-station monitoring leg and a 2020–2021 ICP-MS primary
  campaign), released here as `data/processed/bangladesh_all_samples.parquet` and
  `bangladesh_temporal_pairs.parquet`, together with the de-identified processed
  held-out tensors used for evaluation.

Because of GEMStat redistribution terms, the **raw** global corpus is not republished. The
**harmonized** `data/processed/` tensors (aggregated, non-redistributable rows removed
where required) are provided in the Zenodo release so the modeling pipeline is runnable.
The Bangladesh primary field data are the authors' own and are released openly with this
archive.
