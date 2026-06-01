# Data & checkpoints

Large artifacts (harmonized data tensors, encoder checkpoints, raw sources) are
**not stored in git** because of size. They are released separately.

> **Zenodo / archive DOI:** _TODO — add the DOI/URL once the artifact archive is uploaded._

## What's in the separate release

| Artifact | Approx. size | Needed for |
|---|---|---|
| `data/processed/` (harmonized tensors: `genesis_pretrain*.pt`, `genesis_held_out_bd*.pt`, parquet vectors) | ~565 MB | Pretraining + classification |
| `checkpoints/{small,base,large}_stage{1,2}/best.pt` | ~17 MB – 553 MB each | Classification from a fixed encoder |
| Raw source dumps (`data/GEMStat`, `data/Podgorski`, `data/raw/*`) | ~3.2 GB | Re-running curation from scratch |

To reproduce the headline numbers you only need `data/processed/` and the relevant
`checkpoints/`. Place them at the repository root so the `Path(__file__).parent.parent`
references in `src/` resolve.

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
