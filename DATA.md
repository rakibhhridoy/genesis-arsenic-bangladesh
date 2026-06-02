# Data & checkpoints

Large artifacts (harmonized data tensors, encoder checkpoints, raw sources) are
**not stored in git** because of size. They are released separately.

> **Zenodo / archive DOI:** _TODO — add the DOI/URL once the artifact archive is uploaded._
> Archive file: `zenodo_genesis_arsenic_bangladesh_v2.zip` (~540 MB compressed).

## What's in the separate release (v2)

The release is **license-clean and derived-data-only**: it contains the harmonized
tensors and the encoders/results needed to reproduce every number in the
manuscript, but **no raw GEMStat** and **no `genesis.duckdb`** (the DB contains
raw GEMStat rows that cannot be redistributed).

| Artifact | Approx. size | Reproduces |
|---|---|---|
| `data/processed/` — harmonized tensors (`genesis_pretrain*.pt`, `genesis_temporal_pairs*.{pt,parquet}`, `genesis_held_out_bd*.pt`, `normalization_stats.json`) | ~565 MB | distribution shift, LORO, reconstruction/attention probes |
| `checkpoints/small_stage2/best.pt`, `checkpoints/base_stage2/best.pt` | 17 MB, 103 MB | attention, reconstruction, aux-reliance, layer-wise, encoder-LORO (Small/Base) |
| `checkpoints/classify_as*` (per-seed heads) | ~0.5 GB | headline BD-As classification table |
| `results/` — all per-seed JSONs incl. `region_transfer*.json`, `reconstruction_probe.json`, `aux_reliance.json`, `layerwise_decodability.json`, `dist_shift.json`, `calibration_*` | ~30 MB | every figure and reported number |

**Not in the release** (reproducible only with provider access or the pod):
- Large Stage-1 encoder (552 MB) — pull from the training pod; only needed to
  re-derive the Large column of the attention / encoder-LORO results (the
  shipped `results/*_large.json` already contain those numbers).
- `genesis.duckdb` and raw source dumps — excluded for license/size (see below).
  Exploration scripts `src/20`, `src/21` read the DB and therefore will not run
  from the release; they are not needed for any manuscript number.

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
- **Bangladesh / BWDB** held-out set — derived from BWDB monitoring stations
  (Shamsuddha et al. 2022, Bengal Water Machine).

Because of GEMStat redistribution terms, the **raw** corpus is not republished. The
**harmonized** `data/processed/` tensors (aggregated, non-redistributable rows removed
where required) are provided in the Zenodo release so the modeling pipeline is runnable.
