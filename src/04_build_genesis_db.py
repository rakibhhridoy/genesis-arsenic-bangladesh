#!/usr/bin/env python3
"""
GENESIS Paper 5 — DuckDB Data Pipeline
========================================
Ingests all raw sources into a single DuckDB database, harmonizes units
and schema, creates pretrain / temporal / held-out tables, and exports
memory-mapped PyTorch tensors for training.

Design principles:
  - Streaming SQL: never loads full CSVs into RAM (safe on 8 GB M1 Pro)
  - Float64 throughout SQL processing for maximum precision
  - Float32 only at final .pt export (7 significant digits — more than
    any lab instrument provides)
  - All unit conversions are explicit and documented
  - Bangladesh data is NEVER mixed into pretrain tables

Usage:
  python src/04_build_genesis_db.py                     # full pipeline
  python src/04_build_genesis_db.py --step ingest       # only ingest raw
  python src/04_build_genesis_db.py --step harmonize    # only harmonize
  python src/04_build_genesis_db.py --step export       # only export .pt
  python src/04_build_genesis_db.py --step report       # only print report

Outputs:
  data/genesis.duckdb                         — unified database
  data/processed/genesis_pretrain.pt          — pretrain vectors (mmap)
  data/processed/genesis_pretrain_meta.pt     — metadata (site, date, etc.)
  data/processed/genesis_temporal_pairs.pt    — temporal pair tensors
  data/processed/genesis_held_out_bd.pt       — Bangladesh held-out
  data/processed/genesis_held_out_bd_pairs.pt — Bangladesh temporal pairs
  data/processed/genesis_db_report.txt        — full statistics report
"""

import argparse
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np

# ============================================================
# PATHS
# ============================================================

PAPER5 = Path(__file__).resolve().parent.parent
PAPER3 = PAPER5.parent / "Paper3"
DATA_DIR = PAPER5 / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "genesis.duckdb"
LOG_DIR = PAPER5 / "logs"
LOG_DIR.mkdir(exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# GENESIS SCHEMA
# ============================================================

# The 20 core geochemical parameters in canonical order.
# This order is fixed and used for all tensor exports.
GENESIS_PARAMS = [
    "As", "Fe", "Mn", "PO4", "F", "U", "NO3",
    "pH", "Eh", "EC", "TDS",
    "Ca", "Mg", "Na", "K", "Cl", "HCO3", "SO4",
    "SiO2", "DOC",
]

# Standard units for each parameter.
# All values in the database are stored in these units.
GENESIS_UNITS = {
    "As": "µg/L", "Fe": "mg/L", "Mn": "mg/L", "PO4": "mg/L",
    "F": "mg/L", "U": "µg/L", "NO3": "mg/L",
    "pH": "unitless", "Eh": "mV", "EC": "µS/cm", "TDS": "mg/L",
    "Ca": "mg/L", "Mg": "mg/L", "Na": "mg/L", "K": "mg/L",
    "Cl": "mg/L", "HCO3": "mg/L", "SO4": "mg/L",
    "SiO2": "mg/L", "DOC": "mg/L",
}

# Physical validity ranges (values outside are set to NULL).
VALID_RANGES = {
    "As": (0.0, 10000.0),     # µg/L — up to extreme contamination
    "Fe": (0.0, 500.0),       # mg/L
    "Mn": (0.0, 100.0),       # mg/L
    "PO4": (0.0, 50.0),       # mg/L
    "F": (0.0, 50.0),         # mg/L
    "U": (0.0, 10000.0),      # µg/L
    "NO3": (0.0, 1000.0),     # mg/L as NO3
    "pH": (2.0, 12.0),        # unitless
    "Eh": (-500.0, 1000.0),   # mV
    "EC": (1.0, 200000.0),    # µS/cm
    "TDS": (1.0, 200000.0),   # mg/L
    "Ca": (0.0, 5000.0),      # mg/L
    "Mg": (0.0, 5000.0),      # mg/L
    "Na": (0.0, 50000.0),     # mg/L
    "K": (0.0, 5000.0),       # mg/L
    "Cl": (0.0, 100000.0),    # mg/L
    "HCO3": (0.0, 10000.0),   # mg/L
    "SO4": (0.0, 50000.0),    # mg/L
    "SiO2": (0.0, 500.0),     # mg/L
    "DOC": (0.0, 500.0),      # mg/L
}

MIN_PARAMS = 3            # minimum non-null params for a valid vector
TEMPORAL_GAP_YEARS = 5    # minimum gap for temporal pairs

# Water body type categories (canonical)
WBT_GROUNDWATER = "groundwater"
WBT_RIVER = "river"
WBT_LAKE = "lake"
WBT_UNKNOWN = "unknown"

# EEA parameter name → GENESIS parameter mapping
EEA_PARAM_MAP = {
    "Arsenic and its compounds": ("As", 1.0),        # µg/L → µg/L
    "Iron and its compounds": ("Fe", 0.001),          # µg/L → mg/L
    "Manganese and its compounds": ("Mn", 0.001),     # µg/L → mg/L
    "Phosphate": ("PO4", None),                       # mg{P}/L → needs ×3.066 → mg PO4/L
    "Fluoride": ("F", 0.001),                         # µg/L → mg/L
    "Uranium": ("U", 1.0),                            # µg/L → µg/L
    "Nitrate": ("NO3", 1.0),                          # mg{NO3}/L → mg/L
    "pH": ("pH", 1.0),                                # unitless
    "Electrical conductivity": ("EC", 1.0),           # µS/cm → µS/cm
    "Calcium": ("Ca", 1.0),                           # mg/L → mg/L
    "Magnesium": ("Mg", 1.0),                         # mg/L → mg/L
    "Sodium": ("Na", 1.0),                            # mg/L → mg/L
    "Potassium": ("K", 1.0),                          # mg/L → mg/L
    "Chloride": ("Cl", 1.0),                          # mg/L → mg/L
    "Hydrogen Carbonate (Bicarbonate) HCO3": ("HCO3", 1.0),  # mg/L → mg/L
    "Sulphate": ("SO4", 1.0),                         # mg/L → mg/L
    "Silicate": ("SiO2", None),                       # mg{Si}/L → needs ×2.139 → mg SiO2/L
    "Total organic carbon (TOC)": ("DOC", 1.0),       # mg{C}/L → mg/L (proxy)
    "Dissolved organic carbon (DOC)": ("DOC", 1.0),   # mg{C}/L → mg/L (preferred)
}

# EEA water body category code → canonical water body type
EEA_WBT_MAP = {
    "RW": WBT_RIVER,
    "LW": WBT_LAKE,
    "GW": WBT_GROUNDWATER,
    "TW": WBT_RIVER,       # transitional → treat as river
}

# GEMStat water_type → canonical water body type
GEMSTAT_WBT_MAP = {
    "River station": WBT_RIVER,
    "Lake station": WBT_LAKE,
    "Groundwater station": WBT_GROUNDWATER,
    "Reservoir station": WBT_LAKE,
    "Wetland station": WBT_RIVER,
}

# ============================================================
# LOGGING
# ============================================================

_LOG_FILE = None

def _log(msg: str):
    """Print to stdout and append to log file."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _LOG_FILE:
        _LOG_FILE.write(line + "\n")
        _LOG_FILE.flush()


# ============================================================
# STEP 1: INGEST RAW SOURCES
# ============================================================

def _range_filter_sql(param: str) -> str:
    """SQL CASE expression that NULLs out-of-range values for one param."""
    lo, hi = VALID_RANGES[param]
    return (
        f'CASE WHEN "{param}" >= {lo} AND "{param}" <= {hi} '
        f'THEN "{param}" ELSE NULL END AS "{param}"'
    )


def ingest_nwis(con: duckdb.DuckDBPyConnection):
    """Ingest USGS NWIS from existing processed parquet."""
    path = PROC_DIR / "usgs_nwis_vectors.parquet"
    if not path.exists():
        _log("  SKIP: usgs_nwis_vectors.parquet not found")
        return

    _log("  Ingesting USGS NWIS...")

    # NWIS is already in standard units:
    #   As, U = µg/L; all others = mg/L; pH unitless; Eh mV; EC µS/cm
    param_cols = ", ".join([_range_filter_sql(p) for p in GENESIS_PARAMS])

    con.execute(f"""
        CREATE OR REPLACE TABLE raw_nwis AS
        SELECT
            site_id AS site_id,
            date AS sample_date,
            CAST(EXTRACT(YEAR FROM date) AS INTEGER) AS year,
            CAST(lat AS DOUBLE) AS lat,
            CAST(lon AS DOUBLE) AS lon,
            'USA' AS country,
            'USGS_NWIS' AS source,
            '{WBT_GROUNDWATER}' AS water_body_type,
            {param_cols}
        FROM read_parquet('{path}')
    """)

    n = con.execute("SELECT COUNT(*) FROM raw_nwis").fetchone()[0]
    _log(f"    raw_nwis: {n:,} rows")


def ingest_gemstat(con: duckdb.DuckDBPyConnection):
    """Ingest GEMStat from existing processed parquet.

    UNIT CONVERSION:
      - GEMStat As is in mg/L → multiply by 1000 → µg/L
      - GEMStat U  is in mg/L → multiply by 1000 → µg/L
      - GEMStat Eh is stored as object (empty) → cast to DOUBLE (will be NULL)
      - All other params already in standard units
    """
    path = PROC_DIR / "all_water_vectors_stage1.parquet"
    if not path.exists():
        _log("  SKIP: all_water_vectors_stage1.parquet not found")
        return

    _log("  Ingesting GEMStat...")

    # Build column list with unit conversions + range filters
    col_exprs = []
    for p in GENESIS_PARAMS:
        lo, hi = VALID_RANGES[p]
        if p == "As":
            # mg/L → µg/L
            expr = f"CASE WHEN (CAST(\"As\" AS DOUBLE) * 1000.0) >= {lo} AND (CAST(\"As\" AS DOUBLE) * 1000.0) <= {hi} THEN CAST(\"As\" AS DOUBLE) * 1000.0 ELSE NULL END AS \"As\""
        elif p == "U":
            # mg/L → µg/L
            expr = f"CASE WHEN (CAST(\"U\" AS DOUBLE) * 1000.0) >= {lo} AND (CAST(\"U\" AS DOUBLE) * 1000.0) <= {hi} THEN CAST(\"U\" AS DOUBLE) * 1000.0 ELSE NULL END AS \"U\""
        elif p == "Eh":
            # Object column → cast to DOUBLE (will produce NULLs for non-numeric)
            expr = f"CASE WHEN TRY_CAST(\"Eh\" AS DOUBLE) >= {lo} AND TRY_CAST(\"Eh\" AS DOUBLE) <= {hi} THEN TRY_CAST(\"Eh\" AS DOUBLE) ELSE NULL END AS \"Eh\""
        else:
            expr = (
                f"CASE WHEN CAST(\"{p}\" AS DOUBLE) >= {lo} AND CAST(\"{p}\" AS DOUBLE) <= {hi} "
                f"THEN CAST(\"{p}\" AS DOUBLE) ELSE NULL END AS \"{p}\""
            )
        col_exprs.append(expr)

    param_sql = ",\n            ".join(col_exprs)

    # Build water_body_type CASE expression from GEMStat water_type column
    wbt_cases = " ".join(
        [f"WHEN water_type = '{k}' THEN '{v}'" for k, v in GEMSTAT_WBT_MAP.items()]
    )
    wbt_expr = f"CASE {wbt_cases} ELSE '{WBT_UNKNOWN}' END"

    con.execute(f"""
        CREATE OR REPLACE TABLE raw_gemstat AS
        SELECT
            station_id AS site_id,
            sample_date,
            CAST(year AS INTEGER) AS year,
            CAST(lat AS DOUBLE) AS lat,
            CAST(lon AS DOUBLE) AS lon,
            country,
            'GEMStat' AS source,
            {wbt_expr} AS water_body_type,
            {param_sql}
        FROM read_parquet('{path}')
    """)

    n = con.execute("SELECT COUNT(*) FROM raw_gemstat").fetchone()[0]
    _log(f"    raw_gemstat: {n:,} rows")


def ingest_ades(con: duckdb.DuckDBPyConnection):
    """Ingest BRGM ADES from checkpointed long-format parquet chunks.

    ADES chunks are in long format: (site_id, date, param, value, lat, lon, country)
    where param is already mapped to GENESIS names (As, Fe, pH, etc.)

    UNITS from Hub'Eau API (verified 2026-04-13):
      - As = µg(As)/L → µg/L (no conversion)
      - Fe = µg(Fe)/L → mg/L (÷1000)  ← API returns µg, GENESIS wants mg
      - Mn = µg(Mn)/L → mg/L (÷1000)  ← API returns µg, GENESIS wants mg
      - U  = mg(U)/L → µg/L (×1000)   ← API returns mg, GENESIS wants µg
      - All others already in standard GENESIS units
    NOTE: Old chunks (before fix) may contain wrong params:
      - 1301 was mapped as Eh but is actually Temperature → filter out
      - 1337 was mapped as K but is actually Cl
      - 1367 was mapped as U but is actually K
      - 1305 was mapped as TDS but is actually Suspended Matter
      - 1395 was mapped as Cl but is actually Molybdenum
    """
    chunks_dir = RAW_DIR / "brgm_ades" / "chunks"
    if not chunks_dir.exists():
        _log("  SKIP: brgm_ades/chunks/ not found")
        return

    # Find all valid parquet chunks (skip macOS ._ files)
    # Separate OLD chunks (mislabeled codes) from NEW chunks (codes 1330, 1361, 1307)
    NEW_CODES = {"1330", "1361", "1307"}  # Eh, U, TDS — downloaded with correct labels
    old_chunks = []
    new_chunks = []
    for f in sorted(os.listdir(chunks_dir)):
        if not f.endswith(".parquet") or f.startswith("."):
            continue
        fpath = str(chunks_dir / f)
        # Filename format: {dept}_{code}.parquet — extract code
        code = f.rsplit("_", 1)[-1].replace(".parquet", "")
        if code in NEW_CODES:
            new_chunks.append(fpath)
        else:
            old_chunks.append(fpath)

    if not old_chunks and not new_chunks:
        _log("  SKIP: no ADES chunk files found")
        return

    _log(f"  Ingesting ADES ({len(old_chunks)} old + {len(new_chunks)} new chunks)...")

    # Build SQL for OLD chunks (mislabeled param codes, need remapping):
    #   "Eh" in old chunks = Temperature (code 1301) → DISCARD
    #   "K"  in old chunks = Chloride (code 1337)    → remap to "Cl"
    #   "U"  in old chunks = Potassium (code 1367)   → remap to "K", undo ×1000
    #   "TDS" in old chunks = Suspended Matter (1305) → DISCARD
    #   "Cl" in old chunks = Molybdenum (code 1395)  → DISCARD
    sql_parts = []
    if old_chunks:
        old_union = " UNION ALL ".join(
            [f"SELECT * FROM read_parquet('{f}')" for f in old_chunks]
        )
        sql_parts.append(f"""
            SELECT
                site_id, CAST(date AS DATE) AS sample_date,
                CASE
                    WHEN param = 'K'  THEN 'Cl'
                    WHEN param = 'U'  THEN 'K'
                    ELSE param
                END AS param,
                CASE
                    WHEN param = 'U'  THEN CAST(value AS DOUBLE) / 1000.0
                    ELSE CAST(value AS DOUBLE)
                END AS value,
                CAST(lat AS DOUBLE) AS lat, CAST(lon AS DOUBLE) AS lon, 'FR' AS country
            FROM ({old_union})
            WHERE param IS NOT NULL AND value IS NOT NULL
              AND param NOT IN ('Eh', 'TDS', 'Cl')
        """)

    # Build SQL for NEW chunks (correct labels: Eh, U, TDS — no remapping needed)
    # U: download script 04b already converted mg→µg (×1000), so value is µg/L
    if new_chunks:
        new_union = " UNION ALL ".join(
            [f"SELECT * FROM read_parquet('{f}')" for f in new_chunks]
        )
        sql_parts.append(f"""
            SELECT
                site_id, CAST(date AS DATE) AS sample_date,
                param,
                CAST(value AS DOUBLE) AS value,
                CAST(lat AS DOUBLE) AS lat, CAST(lon AS DOUBLE) AS lon, 'FR' AS country
            FROM ({new_union})
            WHERE param IS NOT NULL AND value IS NOT NULL
        """)

    combined_sql = " UNION ALL ".join(sql_parts)
    con.execute(f"""
        CREATE OR REPLACE TABLE ades_long AS {combined_sql}
    """)

    n_long = con.execute("SELECT COUNT(*) FROM ades_long").fetchone()[0]
    _log(f"    ades_long: {n_long:,} measurements")

    # Step 2: Pivot long → wide using conditional aggregation
    # This is the memory-efficient way to pivot in DuckDB
    pivot_exprs = []
    for p in GENESIS_PARAMS:
        lo, hi = VALID_RANGES[p]
        pivot_exprs.append(
            f"CASE WHEN median_val >= {lo} AND median_val <= {hi} "
            f"THEN median_val ELSE NULL END AS \"{p}\""
        )

    # First compute medians per (site, date, param)
    # Then pivot via subqueries
    pivot_select = ",\n            ".join([
        f"""(SELECT CASE WHEN v >= {VALID_RANGES[p][0]} AND v <= {VALID_RANGES[p][1]}
                    THEN v ELSE NULL END
             FROM (SELECT MEDIAN(value) AS v FROM ades_long
                   WHERE ades_long.site_id = g.site_id
                   AND ades_long.sample_date = g.sample_date
                   AND ades_long.param = '{p}') sub
            ) AS \"{p}\""""
        for p in GENESIS_PARAMS
    ])

    # More efficient: use PIVOT or conditional aggregation
    agg_exprs = []
    for p in GENESIS_PARAMS:
        lo, hi = VALID_RANGES[p]
        agg_exprs.append(
            f"""CASE
                WHEN MEDIAN(CASE WHEN param = '{p}' THEN value END) >= {lo}
                 AND MEDIAN(CASE WHEN param = '{p}' THEN value END) <= {hi}
                THEN MEDIAN(CASE WHEN param = '{p}' THEN value END)
                ELSE NULL
            END AS \"{p}\""""
        )
    agg_sql = ",\n            ".join(agg_exprs)

    con.execute(f"""
        CREATE OR REPLACE TABLE raw_ades AS
        SELECT
            site_id,
            sample_date,
            CAST(EXTRACT(YEAR FROM sample_date) AS INTEGER) AS year,
            MIN(lat) AS lat,
            MIN(lon) AS lon,
            'FR' AS country,
            'BRGM_ADES' AS source,
            '{WBT_GROUNDWATER}' AS water_body_type,
            {agg_sql}
        FROM ades_long
        GROUP BY site_id, sample_date
    """)

    # Drop intermediate table
    con.execute("DROP TABLE IF EXISTS ades_long")

    n = con.execute("SELECT COUNT(*) FROM raw_ades").fetchone()[0]
    _log(f"    raw_ades: {n:,} wide vectors")


def ingest_eea(con: duckdb.DuckDBPyConnection):
    """Ingest EEA WISE-6 from ZIP containing aggregated CSV + spatial CSV.

    Uses the AggregatedData CSV (~2 GB, long format: one row per param per
    site per year) and the SpatialObject CSV for lat/lon and water body type.

    UNIT CONVERSIONS (EEA → GENESIS standard):
      - As: µg/L → µg/L (×1)
      - Fe, Mn: µg/L → mg/L (×0.001)
      - F: µg/L → mg/L (×0.001)
      - U: µg/L → µg/L (×1)
      - PO4: mg{P}/L → mg PO4/L (×3.066)
      - SiO2: mg{Si}/L → mg SiO2/L (×2.139)
      - TOC/DOC: mg{C}/L → mg/L (×1, proxy)
      - All others: mg/L or native units (×1)
    """
    eea_dir = RAW_DIR / "eea_wise"
    zip_file = eea_dir / "eea_wise6.zip"

    if not zip_file.exists():
        _log("  SKIP: eea_wise6.zip not found")
        return

    _log("  Ingesting EEA WISE-6 from ZIP...")

    # Paths inside the ZIP
    zip_prefix = "eea_t_waterbase-water-quality-icm-3_p_1900-2024_v01_r00"
    agg_csv = f"{zip_prefix}/Waterbase_v2024_1_T_WISE6_AggregatedData.csv"
    spatial_csv = f"{zip_prefix}/Waterbase_v2024_1_S_WISE6_SpatialObject_DerivedData.csv"

    # Step 1: Extract only the two CSVs we need (not the full 54 GB)
    import zipfile
    extracted_agg = eea_dir / "AggregatedData.csv"
    extracted_spatial = eea_dir / "SpatialObject.csv"

    if not extracted_agg.exists():
        _log("    Extracting AggregatedData CSV (~2 GB)...")
        with zipfile.ZipFile(zip_file, "r") as zf:
            with zf.open(agg_csv) as src, open(extracted_agg, "wb") as dst:
                import shutil
                shutil.copyfileobj(src, dst)
        _log(f"    Extracted: {extracted_agg.stat().st_size / 1e9:.2f} GB")

    if not extracted_spatial.exists():
        _log("    Extracting SpatialObject CSV...")
        with zipfile.ZipFile(zip_file, "r") as zf:
            with zf.open(spatial_csv) as src, open(extracted_spatial, "wb") as dst:
                import shutil
                shutil.copyfileobj(src, dst)
        _log(f"    Extracted: {extracted_spatial.stat().st_size / 1e6:.1f} MB")

    # Step 2: Load spatial table for lat/lon and water body type
    _log("    Loading spatial reference...")
    # Read header to handle BOM-prefixed column names
    import pandas as pd
    spatial_cols = list(pd.read_csv(extracted_spatial, nrows=0).columns)
    # Find actual column names (may have BOM prefix)
    country_col = [c for c in spatial_cols if "countryCode" in c][0]
    site_col = "monitoringSiteIdentifier"

    con.execute(f"""
        CREATE OR REPLACE TABLE eea_spatial AS
        SELECT
            "{site_col}" AS site_id,
            CAST(lat AS DOUBLE) AS lat,
            CAST(lon AS DOUBLE) AS lon,
            "{country_col}" AS country,
            specialisedZoneType AS zone_type
        FROM read_csv('{extracted_spatial}', auto_detect=true)
        WHERE "{site_col}" IS NOT NULL
           AND "{site_col}" != ''
    """)
    n_spatial = con.execute("SELECT COUNT(*) FROM eea_spatial").fetchone()[0]
    _log(f"    eea_spatial: {n_spatial:,} sites")

    # Step 3: Load aggregated data, filtering only GENESIS-relevant params
    _log("    Loading aggregated data (filtering to GENESIS params)...")
    eea_param_names = "', '".join(EEA_PARAM_MAP.keys())

    # Handle BOM in aggregated CSV header too
    agg_cols = list(pd.read_csv(extracted_agg, nrows=0).columns)
    agg_country_col = [c for c in agg_cols if "countryCode" in c][0]

    con.execute(f"""
        CREATE OR REPLACE TABLE eea_long AS
        SELECT
            monitoringSiteIdentifier AS site_id,
            CAST(phenomenonTimeReferenceYear AS INTEGER) AS year,
            observedPropertyDeterminandLabel AS param_label,
            parameterWaterBodyCategory AS wbt_code,
            TRY_CAST(resultMeanValue AS DOUBLE) AS mean_value,
            resultUom AS unit
        FROM read_csv('{extracted_agg}', auto_detect=true)
        WHERE observedPropertyDeterminandLabel IN ('{eea_param_names}')
          AND TRY_CAST(resultMeanValue AS DOUBLE) IS NOT NULL
    """)

    n_long = con.execute("SELECT COUNT(*) FROM eea_long").fetchone()[0]
    _log(f"    eea_long: {n_long:,} measurements (GENESIS params only)")

    # Step 4: Pivot long → wide with unit conversions
    agg_exprs = []
    for eea_name, (genesis_name, factor) in EEA_PARAM_MAP.items():
        lo, hi = VALID_RANGES[genesis_name]
        if factor is None:
            # Special unit conversions
            if genesis_name == "PO4":
                conv = f"MEDIAN(CASE WHEN param_label = '{eea_name}' THEN mean_value * 3.066 END)"
            elif genesis_name == "SiO2":
                conv = f"MEDIAN(CASE WHEN param_label = '{eea_name}' THEN mean_value * 2.139 END)"
            else:
                conv = f"MEDIAN(CASE WHEN param_label = '{eea_name}' THEN mean_value END)"
        elif factor != 1.0:
            conv = f"MEDIAN(CASE WHEN param_label = '{eea_name}' THEN mean_value * {factor} END)"
        else:
            conv = f"MEDIAN(CASE WHEN param_label = '{eea_name}' THEN mean_value END)"

        agg_exprs.append(
            f"""CASE WHEN {conv} >= {lo} AND {conv} <= {hi}
                THEN {conv} ELSE NULL END AS \"{genesis_name}\""""
        )

    # DOC has two possible sources — prefer DOC over TOC
    # Already handled: both map to "DOC", MEDIAN picks from both

    agg_sql = ",\n            ".join(agg_exprs)

    # Water body type mapping
    wbt_cases = " ".join(
        [f"WHEN MIN(wbt_code) = '{k}' THEN '{v}'" for k, v in EEA_WBT_MAP.items()]
    )
    wbt_expr = f"CASE {wbt_cases} ELSE '{WBT_UNKNOWN}' END"

    con.execute(f"""
        CREATE OR REPLACE TABLE eea_wide AS
        SELECT
            site_id,
            year,
            {wbt_expr} AS water_body_type,
            MIN(wbt_code) AS wbt_code,
            {agg_sql}
        FROM eea_long
        GROUP BY site_id, year
    """)

    n_wide = con.execute("SELECT COUNT(*) FROM eea_wide").fetchone()[0]
    _log(f"    eea_wide: {n_wide:,} site-year vectors")

    # Step 5: Join with spatial for lat/lon, create final raw_eea
    # Determine which GENESIS params actually exist in eea_wide
    eea_wide_cols = set(r[0] for r in con.execute("DESCRIBE eea_wide").fetchall())
    param_col_exprs = []
    n_param_parts = []
    for p in GENESIS_PARAMS:
        if p in eea_wide_cols:
            param_col_exprs.append(f'w."{p}"')
            n_param_parts.append(f'CASE WHEN w."{p}" IS NOT NULL THEN 1 ELSE 0 END')
        else:
            param_col_exprs.append(f'CAST(NULL AS DOUBLE) AS "{p}"')

    param_cols = ", ".join(param_col_exprs)
    n_params_expr = " + ".join(n_param_parts) if n_param_parts else "0"

    con.execute(f"""
        CREATE OR REPLACE TABLE raw_eea AS
        SELECT
            w.site_id,
            MAKE_DATE(w.year, 7, 1) AS sample_date,
            w.year,
            COALESCE(s.lat, 0.0) AS lat,
            COALESCE(s.lon, 0.0) AS lon,
            COALESCE(s.country, SUBSTRING(w.site_id, 1, 2)) AS country,
            'EEA_WISE' AS source,
            w.water_body_type,
            {param_cols}
        FROM eea_wide w
        LEFT JOIN eea_spatial s ON w.site_id = s.site_id
        WHERE ({n_params_expr}) >= {MIN_PARAMS}
    """)

    n = con.execute("SELECT COUNT(*) FROM raw_eea").fetchone()[0]
    _log(f"    raw_eea: {n:,} vectors (≥{MIN_PARAMS} params)")

    # Water body type breakdown
    wbt_stats = con.execute("""
        SELECT water_body_type, COUNT(*) AS n
        FROM raw_eea GROUP BY water_body_type ORDER BY n DESC
    """).fetchall()
    for wbt, cnt in wbt_stats:
        _log(f"      {wbt}: {cnt:,}")

    # Cleanup intermediate tables
    con.execute("DROP TABLE IF EXISTS eea_long")
    con.execute("DROP TABLE IF EXISTS eea_wide")
    con.execute("DROP TABLE IF EXISTS eea_spatial")


def ingest_bangladesh(con: duckdb.DuckDBPyConnection):
    """Ingest Bangladesh held-out data from Paper3 harmonized CSVs.

    These are the author's primary field data and must NEVER be included
    in pretraining. They are stored in separate tables.

    UNITS: Already harmonized in Paper3:
      - As = µg/L (confirmed: raw was 'As (µg/l)')
      - Fe, Mn, PO4, Ca, Mg, Na, K, Cl, HCO3, SO4, NO3, SiO2, F, DOC = mg/L
      - pH = unitless, Eh (ORP) = mV, EC = µS/cm, TDS = mg/L
    """
    old_path = PAPER3 / "analysis" / "output" / "tables" / "old_harmonized.csv"
    new_path = PAPER3 / "analysis" / "output" / "tables" / "new_harmonized.csv"
    matched_path = PAPER3 / "analysis" / "output" / "tables" / "matched_wells.csv"

    if not old_path.exists() or not new_path.exists():
        _log("  SKIP: Bangladesh harmonized CSVs not found")
        return

    _log("  Ingesting Bangladesh held-out data...")

    # Read column names to know which params exist in each file
    import pandas as pd
    old_cols = set(pd.read_csv(old_path, nrows=0).columns)
    new_cols = set(pd.read_csv(new_path, nrows=0).columns)
    matched_cols = set(pd.read_csv(matched_path, nrows=0).columns) if matched_path.exists() else set()

    # --- Old (2012-2013) ---
    param_cols_old = []
    for p in GENESIS_PARAMS:
        lo, hi = VALID_RANGES[p]
        if p in old_cols:
            param_cols_old.append(
                f'CASE WHEN TRY_CAST("{p}" AS DOUBLE) >= {lo} AND TRY_CAST("{p}" AS DOUBLE) <= {hi} '
                f'THEN TRY_CAST("{p}" AS DOUBLE) ELSE NULL END AS "{p}"'
            )
        else:
            param_cols_old.append(f'CAST(NULL AS DOUBLE) AS "{p}"')
    param_sql_old = ", ".join(param_cols_old)

    con.execute(f"""
        CREATE OR REPLACE TABLE held_out_bd_old AS
        SELECT
            "Sample_ID" AS site_id,
            TRY_CAST(
                CASE
                    WHEN "Date" LIKE '%.%.%'
                    THEN SUBSTR("Date", 7, 4) || '-' || SUBSTR("Date", 4, 2) || '-' || SUBSTR("Date", 1, 2)
                    ELSE "Date"
                END AS DATE
            ) AS sample_date,
            CAST(EXTRACT(YEAR FROM TRY_CAST(
                CASE
                    WHEN "Date" LIKE '%.%.%'
                    THEN SUBSTR("Date", 7, 4) || '-' || SUBSTR("Date", 4, 2) || '-' || SUBSTR("Date", 1, 2)
                    ELSE "Date"
                END AS DATE
            )) AS INTEGER) AS year,
            CAST("Latitude" AS DOUBLE) AS lat,
            CAST("Longitude" AS DOUBLE) AS lon,
            'BGD' AS country,
            'BD_PRIMARY_OLD' AS source,
            '{WBT_GROUNDWATER}' AS water_body_type,
            CAST("Depth" AS DOUBLE) AS depth,
            "District" AS district,
            "Season" AS season,
            "ID_norm" AS id_norm,
            "Depth_bin" AS depth_bin,
            CAST("CBE" AS DOUBLE) AS cbe,
            {param_sql_old}
        FROM read_csv('{old_path}', auto_detect=true)
    """)
    n_old = con.execute("SELECT COUNT(*) FROM held_out_bd_old").fetchone()[0]
    _log(f"    held_out_bd_old: {n_old:,} rows (2012-2013)")

    # --- New (2020-2021) ---
    param_cols_new = []
    for p in GENESIS_PARAMS:
        lo, hi = VALID_RANGES[p]
        if p in new_cols:
            param_cols_new.append(
                f'CASE WHEN TRY_CAST("{p}" AS DOUBLE) >= {lo} AND TRY_CAST("{p}" AS DOUBLE) <= {hi} '
                f'THEN TRY_CAST("{p}" AS DOUBLE) ELSE NULL END AS "{p}"'
            )
        else:
            param_cols_new.append(f'CAST(NULL AS DOUBLE) AS "{p}"')
    param_sql_new = ", ".join(param_cols_new)

    con.execute(f"""
        CREATE OR REPLACE TABLE held_out_bd_new AS
        SELECT
            "Sample_ID" AS site_id,
            TRY_CAST(
                CASE
                    WHEN "Date" LIKE '%.%.%'
                    THEN SUBSTR("Date", 7, 4) || '-' || SUBSTR("Date", 4, 2) || '-' || SUBSTR("Date", 1, 2)
                    ELSE "Date"
                END AS DATE
            ) AS sample_date,
            CAST(EXTRACT(YEAR FROM TRY_CAST(
                CASE
                    WHEN "Date" LIKE '%.%.%'
                    THEN SUBSTR("Date", 7, 4) || '-' || SUBSTR("Date", 4, 2) || '-' || SUBSTR("Date", 1, 2)
                    ELSE "Date"
                END AS DATE
            )) AS INTEGER) AS year,
            CAST("Latitude" AS DOUBLE) AS lat,
            CAST("Longitude" AS DOUBLE) AS lon,
            'BGD' AS country,
            'BD_PRIMARY_NEW' AS source,
            '{WBT_GROUNDWATER}' AS water_body_type,
            CAST("Depth" AS DOUBLE) AS depth,
            "District" AS district,
            "Season" AS season,
            "ID_norm" AS id_norm,
            "Depth_bin" AS depth_bin,
            CAST("CBE" AS DOUBLE) AS cbe,
            {param_sql_new}
        FROM read_csv('{new_path}', auto_detect=true)
    """)
    n_new = con.execute("SELECT COUNT(*) FROM held_out_bd_new").fetchone()[0]
    _log(f"    held_out_bd_new: {n_new:,} rows (2020-2021)")

    # --- Matched temporal pairs ---
    if matched_path.exists():
        param_cols_m = []
        for p in GENESIS_PARAMS:
            lo, hi = VALID_RANGES[p]
            if p in matched_cols:
                param_cols_m.append(
                    f'CASE WHEN TRY_CAST("{p}" AS DOUBLE) >= {lo} AND TRY_CAST("{p}" AS DOUBLE) <= {hi} '
                    f'THEN TRY_CAST("{p}" AS DOUBLE) ELSE NULL END AS "{p}"'
                )
            else:
                param_cols_m.append(f'CAST(NULL AS DOUBLE) AS "{p}"')
        param_sql_m = ", ".join(param_cols_m)

        con.execute(f"""
            CREATE OR REPLACE TABLE held_out_bd_matched AS
            SELECT
                "ID_norm" AS id_norm,
                "Depth_bin" AS depth_bin,
                "pair_key" AS pair_key,
                "Period" AS period,
                "District" AS district,
                "Upazila" AS upazila,
                CAST("Latitude" AS DOUBLE) AS lat,
                CAST("Longitude" AS DOUBLE) AS lon,
                CAST("Depth" AS DOUBLE) AS depth,
                CAST("CBE" AS DOUBLE) AS cbe,
                "CBE_flag" AS cbe_flag,
                {param_sql_m}
            FROM read_csv('{matched_path}', auto_detect=true)
        """)
        n_m = con.execute("SELECT COUNT(*) FROM held_out_bd_matched").fetchone()[0]
        _log(f"    held_out_bd_matched: {n_m:,} rows ({n_m // 2} pairs)")


def step_ingest(con: duckdb.DuckDBPyConnection):
    """Run all ingest functions."""
    _log("STEP 1: INGEST RAW SOURCES")
    _log("=" * 60)
    ingest_nwis(con)
    ingest_gemstat(con)
    ingest_ades(con)
    ingest_eea(con)
    ingest_bangladesh(con)
    _log("")


# ============================================================
# STEP 2: HARMONIZE INTO UNIFIED PRETRAIN TABLE
# ============================================================

def step_harmonize(con: duckdb.DuckDBPyConnection):
    """Merge all non-Bangladesh sources into a unified pretrain table."""
    _log("STEP 2: HARMONIZE")
    _log("=" * 60)

    # Check which raw tables exist
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()]

    source_tables = [t for t in tables if t.startswith("raw_")]
    _log(f"  Source tables found: {source_tables}")

    if not source_tables:
        _log("  ERROR: no raw tables to harmonize")
        return

    # Build UNION ALL across all raw source tables
    param_list = ", ".join([f'CAST("{p}" AS DOUBLE) AS "{p}"' for p in GENESIS_PARAMS])

    union_parts = []
    for t in source_tables:
        # Get columns available in this table
        cols = [r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()]

        select_parts = []
        select_parts.append("CAST(site_id AS VARCHAR) AS site_id")
        select_parts.append("CAST(sample_date AS DATE) AS sample_date")
        select_parts.append("CAST(year AS INTEGER) AS year")
        select_parts.append("CAST(lat AS DOUBLE) AS lat")
        select_parts.append("CAST(lon AS DOUBLE) AS lon")
        select_parts.append("CAST(country AS VARCHAR) AS country")
        select_parts.append("CAST(source AS VARCHAR) AS source")

        # Water body type — use column if present, else 'unknown'
        if "water_body_type" in cols:
            select_parts.append("CAST(water_body_type AS VARCHAR) AS water_body_type")
        else:
            select_parts.append(f"'{WBT_UNKNOWN}' AS water_body_type")

        for p in GENESIS_PARAMS:
            if p in cols:
                select_parts.append(f'CAST("{p}" AS DOUBLE) AS "{p}"')
            else:
                select_parts.append(f"CAST(NULL AS DOUBLE) AS \"{p}\"")

        union_parts.append(
            f"SELECT {', '.join(select_parts)} FROM {t}"
        )

    union_sql = "\n    UNION ALL\n    ".join(union_parts)

    # Create unified table with n_params count
    n_params_expr = " + ".join(
        [f'CASE WHEN "{p}" IS NOT NULL THEN 1 ELSE 0 END' for p in GENESIS_PARAMS]
    )

    con.execute(f"""
        CREATE OR REPLACE TABLE genesis_all_vectors AS
        SELECT *,
            ({n_params_expr}) AS n_params
        FROM (
            {union_sql}
        )
        WHERE ({n_params_expr}) >= {MIN_PARAMS}
    """)

    n_all = con.execute("SELECT COUNT(*) FROM genesis_all_vectors").fetchone()[0]
    _log(f"  genesis_all_vectors: {n_all:,} rows (≥{MIN_PARAMS} params)")

    # Per-source breakdown
    breakdown = con.execute("""
        SELECT source, COUNT(*) AS n,
               ROUND(AVG(n_params), 1) AS avg_params,
               MIN(year) AS min_year, MAX(year) AS max_year
        FROM genesis_all_vectors
        GROUP BY source ORDER BY n DESC
    """).fetchall()
    for src, n, avg_p, y0, y1 in breakdown:
        _log(f"    {src}: {n:,} vectors, avg {avg_p} params, {y0}-{y1}")

    # --- Create pretrain table (excludes Bangladesh) ---
    con.execute("""
        CREATE OR REPLACE TABLE genesis_pretrain AS
        SELECT * FROM genesis_all_vectors
        WHERE country != 'BGD'
          AND source NOT LIKE 'BD_PRIMARY%'
    """)
    n_pt = con.execute("SELECT COUNT(*) FROM genesis_pretrain").fetchone()[0]
    _log(f"  genesis_pretrain: {n_pt:,} rows (Bangladesh excluded)")

    # --- Build temporal pairs from pretrain data ---
    _log("  Building temporal pairs...")
    _build_temporal_pairs(con, "genesis_pretrain", "genesis_temporal_pairs")

    n_pairs = con.execute("SELECT COUNT(*) FROM genesis_temporal_pairs").fetchone()[0]
    _log(f"  genesis_temporal_pairs: {n_pairs:,} pairs")

    # Per-source pair breakdown
    pair_breakdown = con.execute("""
        SELECT source, COUNT(*) AS n
        FROM genesis_temporal_pairs
        GROUP BY source ORDER BY n DESC
    """).fetchall()
    for src, n in pair_breakdown:
        _log(f"    {src}: {n:,} pairs")

    # --- Per-parameter coverage in pretrain ---
    _log("  Parameter coverage in pretrain:")
    for p in GENESIS_PARAMS:
        n_nonnull = con.execute(
            f'SELECT COUNT("{p}") FROM genesis_pretrain WHERE "{p}" IS NOT NULL'
        ).fetchone()[0]
        pct = n_nonnull / n_pt * 100 if n_pt > 0 else 0
        _log(f"    {p:6s}: {n_nonnull:>10,} ({pct:5.1f}%)")

    _log("")


def _build_temporal_pairs(con: duckdb.DuckDBPyConnection, source_table: str,
                          output_table: str):
    """Build temporal pairs from a vectors table.

    A temporal pair is (earliest, latest) sample at the same site
    with a gap >= TEMPORAL_GAP_YEARS and at least MIN_PARAMS shared
    non-null parameters.
    """
    # For each site, get the earliest and latest sample
    param_t0 = ", ".join([f'first_row."{p}" AS "{p}_t0"' for p in GENESIS_PARAMS])
    param_t1 = ", ".join([f'last_row."{p}" AS "{p}_t1"' for p in GENESIS_PARAMS])
    delta_cols = ", ".join([
        f'CASE WHEN last_row."{p}" IS NOT NULL AND first_row."{p}" IS NOT NULL '
        f'THEN last_row."{p}" - first_row."{p}" ELSE NULL END AS "delta_{p}"'
        for p in GENESIS_PARAMS
    ])
    n_paired_expr = " + ".join([
        f'CASE WHEN last_row."{p}" IS NOT NULL AND first_row."{p}" IS NOT NULL '
        f"THEN 1 ELSE 0 END"
        for p in GENESIS_PARAMS
    ])

    con.execute(f"""
        CREATE OR REPLACE TABLE {output_table} AS
        WITH ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY site_id ORDER BY sample_date ASC) AS rn_first,
                ROW_NUMBER() OVER (PARTITION BY site_id ORDER BY sample_date DESC) AS rn_last,
                COUNT(*) OVER (PARTITION BY site_id) AS site_count
            FROM {source_table}
            WHERE sample_date IS NOT NULL
        ),
        first_samples AS (
            SELECT * FROM ranked WHERE rn_first = 1 AND site_count >= 2
        ),
        last_samples AS (
            SELECT * FROM ranked WHERE rn_last = 1 AND site_count >= 2
        )
        SELECT
            first_row.site_id AS station_id,
            first_row.sample_date AS date_t0,
            last_row.sample_date AS date_t1,
            CAST(first_row.year AS INTEGER) AS year_t0,
            CAST(last_row.year AS INTEGER) AS year_t1,
            ROUND(CAST(last_row.sample_date - first_row.sample_date AS DOUBLE) / 365.25, 2) AS gap_years,
            first_row.country AS country,
            first_row.lat AS lat,
            first_row.lon AS lon,
            first_row.source AS source,
            first_row.water_body_type AS water_body_type,
            ({n_paired_expr}) AS n_paired_params,
            {param_t0},
            {param_t1},
            {delta_cols}
        FROM first_samples first_row
        JOIN last_samples last_row
            ON first_row.site_id = last_row.site_id
        WHERE CAST(last_row.sample_date - first_row.sample_date AS DOUBLE) / 365.25 >= {TEMPORAL_GAP_YEARS}
          AND ({n_paired_expr}) >= {MIN_PARAMS}
    """)


# ============================================================
# STEP 3: EXPORT TO PYTORCH TENSORS
# ============================================================

def step_export(con: duckdb.DuckDBPyConnection):
    """Export database tables to memory-mapped PyTorch tensors."""
    import torch

    _log("STEP 3: EXPORT TO PYTORCH TENSORS")
    _log("=" * 60)

    # --- Pretrain vectors ---
    _export_vectors(con, "genesis_pretrain", "genesis_pretrain", torch)

    # --- Temporal pairs ---
    _export_temporal_pairs(con, "genesis_temporal_pairs", "genesis_temporal_pairs", torch)

    # --- Bangladesh held-out (all vectors: old + new) ---
    # Check if tables exist
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()]

    if "held_out_bd_old" in tables and "held_out_bd_new" in tables:
        # Create combined BD vectors view
        param_list = ", ".join([f'CAST("{p}" AS DOUBLE) AS "{p}"' for p in GENESIS_PARAMS])
        n_params_expr = " + ".join(
            [f'CASE WHEN "{p}" IS NOT NULL THEN 1 ELSE 0 END' for p in GENESIS_PARAMS]
        )

        con.execute(f"""
            CREATE OR REPLACE VIEW held_out_bd_all AS
            SELECT site_id, sample_date, year, lat, lon, country, source,
                   depth, district, season, id_norm, depth_bin, cbe,
                   {param_list},
                   ({n_params_expr}) AS n_params
            FROM held_out_bd_old
            UNION ALL
            SELECT site_id, sample_date, year, lat, lon, country, source,
                   depth, district, season, id_norm, depth_bin, cbe,
                   {param_list},
                   ({n_params_expr}) AS n_params
            FROM held_out_bd_new
        """)
        _export_vectors(con, "held_out_bd_all", "genesis_held_out_bd", torch)

    if "held_out_bd_matched" in tables:
        _export_bd_pairs(con, "held_out_bd_matched", "genesis_held_out_bd_pairs", torch)

    _log("")


def _export_vectors(con: duckdb.DuckDBPyConnection, table: str, prefix: str, torch):
    """Export a vectors table to .pt files (chemistry + metadata)."""
    _log(f"  Exporting {table}...")

    # Chemistry tensor: N × 20 (float64 in DB, float32 in tensor)
    # IMPORTANT: DuckDB fetchnumpy() converts NULL → 0.0 by default.
    # We must use fetchdf() and convert via pandas to preserve NaN for missing data.
    param_select = ", ".join([f'CAST("{p}" AS DOUBLE) AS "{p}"' for p in GENESIS_PARAMS])
    df = con.execute(f"SELECT {param_select} FROM {table}").fetchdf()

    # pandas preserves NULL as NaN; convert to numpy float64 then float32
    chem_array = df[GENESIS_PARAMS].values  # float64 with NaN

    # float32 has 7 significant digits — sufficient for all geochemical measurements
    # NaN is preserved in float32
    chem_tensor = torch.from_numpy(chem_array.astype(np.float32))

    out_path = PROC_DIR / f"{prefix}.pt"
    torch.save(chem_tensor, out_path)
    _log(f"    {prefix}.pt: shape {list(chem_tensor.shape)}, "
         f"{out_path.stat().st_size / 1e6:.1f} MB")

    # Metadata: site_id, date, lat, lon, country, source, water_body_type
    # Check if water_body_type column exists in this table
    table_cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
    wbt_col = ", water_body_type" if "water_body_type" in table_cols else ", 'unknown' AS water_body_type"
    meta = con.execute(f"""
        SELECT site_id, CAST(sample_date AS VARCHAR) AS date,
               lat, lon, country, source{wbt_col}
        FROM {table}
    """).fetchdf()
    meta_path = PROC_DIR / f"{prefix}_meta.parquet"
    meta.to_parquet(meta_path, engine="pyarrow", compression="snappy")
    _log(f"    {prefix}_meta.parquet: {len(meta):,} rows, "
         f"{meta_path.stat().st_size / 1e6:.1f} MB")

    # Verify: count NaN pattern
    n_total = chem_tensor.shape[0]
    n_complete = int((~torch.isnan(chem_tensor)).all(dim=1).sum())
    n_nan_per_param = {p: int(torch.isnan(chem_tensor[:, i]).sum())
                       for i, p in enumerate(GENESIS_PARAMS)}
    _log(f"    Fully complete rows: {n_complete:,} / {n_total:,}")


def _export_temporal_pairs(con: duckdb.DuckDBPyConnection, table: str,
                           prefix: str, torch):
    """Export temporal pairs: t0 tensor, t1 tensor, delta tensor, metadata."""
    _log(f"  Exporting {table}...")

    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if n == 0:
        _log(f"    {prefix}: 0 pairs, skipping")
        return

    # t0 chemistry — use fetchdf() to preserve NULL → NaN
    t0_cols = [f"{p}_t0" for p in GENESIS_PARAMS]
    t0_select = ", ".join([f'CAST("{c}" AS DOUBLE) AS "{c}"' for c in t0_cols])
    t0_df = con.execute(f"SELECT {t0_select} FROM {table}").fetchdf()
    t0_array = t0_df[t0_cols].values

    # t1 chemistry
    t1_cols = [f"{p}_t1" for p in GENESIS_PARAMS]
    t1_select = ", ".join([f'CAST("{c}" AS DOUBLE) AS "{c}"' for c in t1_cols])
    t1_df = con.execute(f"SELECT {t1_select} FROM {table}").fetchdf()
    t1_array = t1_df[t1_cols].values

    # delta
    delta_cols = [f"delta_{p}" for p in GENESIS_PARAMS]
    delta_select = ", ".join([f'CAST("{c}" AS DOUBLE) AS "{c}"' for c in delta_cols])
    delta_df = con.execute(f"SELECT {delta_select} FROM {table}").fetchdf()
    delta_array = delta_df[delta_cols].values

    # Gap years as conditioning input
    gap_df = con.execute(f'SELECT CAST(gap_years AS DOUBLE) AS gap_years FROM {table}').fetchdf()
    gap_array = gap_df["gap_years"].values.astype(np.float32)

    # Save as dict tensor
    pairs_dict = {
        "t0": torch.from_numpy(t0_array.astype(np.float32)),
        "t1": torch.from_numpy(t1_array.astype(np.float32)),
        "delta": torch.from_numpy(delta_array.astype(np.float32)),
        "gap_years": torch.from_numpy(gap_array),
        "params": GENESIS_PARAMS,
    }
    out_path = PROC_DIR / f"{prefix}.pt"
    torch.save(pairs_dict, out_path)
    _log(f"    {prefix}.pt: {n:,} pairs, shape {list(pairs_dict['t0'].shape)}, "
         f"{out_path.stat().st_size / 1e6:.1f} MB")

    # Metadata
    table_cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
    wbt_col = ", water_body_type" if "water_body_type" in table_cols else ", 'unknown' AS water_body_type"
    meta = con.execute(f"""
        SELECT station_id, CAST(date_t0 AS VARCHAR) AS date_t0,
               CAST(date_t1 AS VARCHAR) AS date_t1,
               year_t0, year_t1, gap_years, country, lat, lon, source,
               n_paired_params{wbt_col}
        FROM {table}
    """).fetchdf()
    meta_path = PROC_DIR / f"{prefix}_meta.parquet"
    meta.to_parquet(meta_path, engine="pyarrow", compression="snappy")
    _log(f"    {prefix}_meta.parquet: {len(meta):,} rows")


def _export_bd_pairs(con: duckdb.DuckDBPyConnection, table: str,
                     prefix: str, torch):
    """Export Bangladesh matched pairs: pivot period column into t0/t1 tensors."""
    _log(f"  Exporting Bangladesh matched pairs...")

    # Separate by period
    n_params_expr = " + ".join(
        [f'CASE WHEN "{p}" IS NOT NULL THEN 1 ELSE 0 END' for p in GENESIS_PARAMS]
    )

    for period_label, period_alias in [("2012-2013", "t0"), ("2020-2021", "t1")]:
        param_select = ", ".join([f'CAST("{p}" AS DOUBLE) AS "{p}"' for p in GENESIS_PARAMS])
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE bd_pairs_{period_alias} AS
            SELECT pair_key, {param_select},
                   CAST(lat AS DOUBLE) AS lat, CAST(lon AS DOUBLE) AS lon,
                   CAST(depth AS DOUBLE) AS depth, district,
                   CAST(cbe AS DOUBLE) AS cbe
            FROM {table}
            WHERE period = '{period_label}'
            ORDER BY pair_key
        """)

    # Join on pair_key to ensure alignment
    t0_select = ", ".join([f'CAST(t0."{p}" AS DOUBLE) AS "{p}_t0"' for p in GENESIS_PARAMS])
    t1_select = ", ".join([f'CAST(t1."{p}" AS DOUBLE) AS "{p}_t1"' for p in GENESIS_PARAMS])
    delta_select = ", ".join([
        f'CASE WHEN t1."{p}" IS NOT NULL AND t0."{p}" IS NOT NULL '
        f'THEN t1."{p}" - t0."{p}" ELSE NULL END AS "delta_{p}"'
        for p in GENESIS_PARAMS
    ])
    n_paired_expr = " + ".join([
        f'CASE WHEN t1."{p}" IS NOT NULL AND t0."{p}" IS NOT NULL THEN 1 ELSE 0 END'
        for p in GENESIS_PARAMS
    ])

    con.execute(f"""
        CREATE OR REPLACE TABLE bd_pairs_aligned AS
        SELECT
            t0.pair_key,
            t0.lat, t0.lon, t0.depth, t0.district,
            t0.cbe AS cbe_t0, t1.cbe AS cbe_t1,
            ({n_paired_expr}) AS n_paired_params,
            {t0_select},
            {t1_select},
            {delta_select}
        FROM bd_pairs_t0 t0
        JOIN bd_pairs_t1 t1 ON t0.pair_key = t1.pair_key
        WHERE ({n_paired_expr}) >= {MIN_PARAMS}
    """)

    n = con.execute("SELECT COUNT(*) FROM bd_pairs_aligned").fetchone()[0]
    _log(f"    Aligned BD pairs: {n:,}")

    if n == 0:
        return

    # Export tensors
    t0_cols = ", ".join([f'CAST("{p}_t0" AS DOUBLE)' for p in GENESIS_PARAMS])
    t1_cols = ", ".join([f'CAST("{p}_t1" AS DOUBLE)' for p in GENESIS_PARAMS])
    delta_cols = ", ".join([f'CAST("delta_{p}" AS DOUBLE)' for p in GENESIS_PARAMS])

    t0_cols = [f"{p}_t0" for p in GENESIS_PARAMS]
    t0_select = ", ".join([f'CAST("{c}" AS DOUBLE) AS "{c}"' for c in t0_cols])
    t0_df = con.execute(f"SELECT {t0_select} FROM bd_pairs_aligned").fetchdf()
    t0_np = t0_df[t0_cols].values

    t1_cols = [f"{p}_t1" for p in GENESIS_PARAMS]
    t1_select = ", ".join([f'CAST("{c}" AS DOUBLE) AS "{c}"' for c in t1_cols])
    t1_df = con.execute(f"SELECT {t1_select} FROM bd_pairs_aligned").fetchdf()
    t1_np = t1_df[t1_cols].values

    delta_cols = [f"delta_{p}" for p in GENESIS_PARAMS]
    delta_select = ", ".join([f'CAST("{c}" AS DOUBLE) AS "{c}"' for c in delta_cols])
    delta_df = con.execute(f"SELECT {delta_select} FROM bd_pairs_aligned").fetchdf()
    delta_np = delta_df[delta_cols].values

    pairs_dict = {
        "t0": torch.from_numpy(t0_np.astype(np.float32)),
        "t1": torch.from_numpy(t1_np.astype(np.float32)),
        "delta": torch.from_numpy(delta_np.astype(np.float32)),
        "gap_years": torch.tensor(9.0),  # ~9 year gap for all pairs
        "params": GENESIS_PARAMS,
        "n_pairs": n,
    }
    out_path = PROC_DIR / f"{prefix}.pt"
    torch.save(pairs_dict, out_path)
    _log(f"    {prefix}.pt: {n:,} pairs, shape {list(pairs_dict['t0'].shape)}, "
         f"{out_path.stat().st_size / 1e6:.1f} MB")

    # Metadata
    meta = con.execute("""
        SELECT pair_key, lat, lon, depth, district,
               cbe_t0, cbe_t1, n_paired_params
        FROM bd_pairs_aligned
    """).fetchdf()
    meta_path = PROC_DIR / f"{prefix}_meta.parquet"
    meta.to_parquet(meta_path, engine="pyarrow", compression="snappy")
    _log(f"    {prefix}_meta.parquet: {len(meta):,} rows")

    # Cleanup temp tables
    con.execute("DROP TABLE IF EXISTS bd_pairs_t0")
    con.execute("DROP TABLE IF EXISTS bd_pairs_t1")
    con.execute("DROP TABLE IF EXISTS bd_pairs_aligned")


# ============================================================
# STEP 4: REPORT
# ============================================================

def step_report(con: duckdb.DuckDBPyConnection):
    """Generate comprehensive statistics report."""
    _log("STEP 4: REPORT")
    _log("=" * 60)

    report_lines = []
    def rlog(msg):
        _log(msg)
        report_lines.append(msg)

    rlog("GENESIS Database Report")
    rlog(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")
    rlog(f"Database: {DB_PATH}")
    rlog(f"Database size: {DB_PATH.stat().st_size / 1e6:.1f} MB")
    rlog("")

    # Tables
    tables = con.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema='main' AND table_type='BASE TABLE'
        ORDER BY table_name
    """).fetchall()
    rlog("Tables:")
    for (t,) in tables:
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        rlog(f"  {t}: {n:,} rows")
    rlog("")

    # Pretrain summary
    if con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='genesis_pretrain'").fetchone()[0]:
        rlog("--- PRETRAIN CORPUS ---")
        n = con.execute("SELECT COUNT(*) FROM genesis_pretrain").fetchone()[0]
        rlog(f"Total vectors: {n:,}")

        breakdown = con.execute("""
            SELECT source, COUNT(*) AS n,
                   ROUND(AVG(n_params), 1) AS avg_p,
                   COUNT(DISTINCT country) AS n_countries,
                   COUNT(DISTINCT site_id) AS n_sites,
                   MIN(year) AS y0, MAX(year) AS y1
            FROM genesis_pretrain GROUP BY source ORDER BY n DESC
        """).fetchall()
        rlog(f"{'Source':<15} {'Vectors':>10} {'AvgP':>6} {'Countries':>10} {'Sites':>10} {'Years':>12}")
        for src, cnt, ap, nc, ns, y0, y1 in breakdown:
            rlog(f"{src:<15} {cnt:>10,} {ap:>6.1f} {nc:>10,} {ns:>10,} {y0:>5}-{y1:<5}")
        rlog("")

        # Water body type breakdown
        wbt_stats = con.execute("""
            SELECT water_body_type, COUNT(*) AS n
            FROM genesis_pretrain GROUP BY water_body_type ORDER BY n DESC
        """).fetchall()
        rlog("Water body type breakdown:")
        for wbt, cnt in wbt_stats:
            rlog(f"  {wbt:<15s}: {cnt:>10,} ({cnt/n*100:5.1f}%)")
        rlog("")

        rlog("Parameter coverage:")
        for p in GENESIS_PARAMS:
            nn = con.execute(f'SELECT COUNT(*) FROM genesis_pretrain WHERE "{p}" IS NOT NULL').fetchone()[0]
            rlog(f"  {p:6s}: {nn:>10,} ({nn/n*100:5.1f}%)  [{GENESIS_UNITS[p]}]")
        rlog("")

    # Temporal pairs
    if con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='genesis_temporal_pairs'").fetchone()[0]:
        n_pairs = con.execute("SELECT COUNT(*) FROM genesis_temporal_pairs").fetchone()[0]
        rlog(f"--- TEMPORAL PAIRS (pretrain) ---")
        rlog(f"Total pairs: {n_pairs:,}")
        pair_stats = con.execute("""
            SELECT source, COUNT(*) AS n,
                   ROUND(AVG(gap_years), 1) AS avg_gap,
                   ROUND(AVG(n_paired_params), 1) AS avg_paired
            FROM genesis_temporal_pairs GROUP BY source ORDER BY n DESC
        """).fetchall()
        for src, cnt, ag, ap in pair_stats:
            rlog(f"  {src}: {cnt:,} pairs, avg gap {ag}y, avg {ap} paired params")
        rlog("")

    # Bangladesh held-out
    for tbl, label in [("held_out_bd_old", "BD 2012-13"), ("held_out_bd_new", "BD 2020-21")]:
        if con.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name='{tbl}'").fetchone()[0]:
            n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            rlog(f"--- HELD OUT: {label} ---")
            rlog(f"Samples: {n:,}")
            for p in GENESIS_PARAMS:
                nn = con.execute(f'SELECT COUNT(*) FROM {tbl} WHERE "{p}" IS NOT NULL').fetchone()[0]
                rlog(f"  {p:6s}: {nn:>6,} ({nn/n*100:5.1f}%)")
            rlog("")

    if con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='held_out_bd_matched'").fetchone()[0]:
        n = con.execute("SELECT COUNT(*) FROM held_out_bd_matched").fetchone()[0]
        rlog(f"--- HELD OUT: BD Matched Pairs ---")
        rlog(f"Total rows: {n:,} ({n//2} pairs)")
        rlog("")

    # Exported files
    rlog("--- EXPORTED FILES ---")
    for f in sorted(PROC_DIR.glob("genesis_*.pt")) + sorted(PROC_DIR.glob("genesis_*_meta.parquet")):
        rlog(f"  {f.name}: {f.stat().st_size / 1e6:.2f} MB")
    rlog("")

    # Unit documentation
    rlog("--- UNIT CONVENTIONS ---")
    for p in GENESIS_PARAMS:
        lo, hi = VALID_RANGES[p]
        rlog(f"  {p:6s}: {GENESIS_UNITS[p]:8s}  valid range [{lo}, {hi}]")

    # Write report to file
    report_path = PROC_DIR / "genesis_db_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    _log(f"\nReport saved → {report_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    global _LOG_FILE

    parser = argparse.ArgumentParser(description="GENESIS DuckDB Pipeline")
    parser.add_argument("--step", choices=["ingest", "harmonize", "export", "report"],
                        help="Run only one step (default: all)")
    args = parser.parse_args()

    _LOG_FILE = open(LOG_DIR / "genesis_db_build.log", "a")

    _log(f"\n{'=' * 60}")
    _log(f"GENESIS DuckDB Pipeline — {time.strftime('%Y-%m-%d %H:%M')}")
    _log(f"{'=' * 60}")
    _log(f"Database: {DB_PATH}")
    _log("")

    # Open (or create) persistent DuckDB database
    con = duckdb.connect(str(DB_PATH))

    # Set memory limit for 8GB M1 Pro safety
    con.execute("SET memory_limit='3GB'")
    con.execute("SET threads=4")

    steps = {
        "ingest": step_ingest,
        "harmonize": step_harmonize,
        "export": step_export,
        "report": step_report,
    }

    if args.step:
        steps[args.step](con)
    else:
        for name, fn in steps.items():
            fn(con)

    con.close()
    _log("Done.")
    _LOG_FILE.close()


if __name__ == "__main__":
    main()
