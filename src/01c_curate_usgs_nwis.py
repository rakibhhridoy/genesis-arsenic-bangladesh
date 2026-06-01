"""
GENESIS Paper 5 — Step 1c: USGS NWIS Data Curation (via dataretrieval)
======================================================================
Downloads US groundwater quality data using USGS's `dataretrieval` package,
querying the Water Quality Portal state-by-state to avoid server timeouts.

Builds geochemical vectors per site-date and extracts temporal pairs (≥5 yr gap)
to augment the existing 3,527 GEMStat pairs.

Output:
  data/raw/usgs_nwis/<STATE>_results.parquet  (cached per-state results)
  data/processed/usgs_nwis_vectors.parquet
  data/processed/usgs_nwis_temporal_pairs.parquet
  data/processed/final_temporal_pairs_enriched.parquet  (merged with GEMStat)
  data/processed/usgs_nwis_curation_report.txt

Usage:
  pip install dataretrieval pandas pyarrow
  python src/01c_curate_usgs_nwis.py
  python src/01c_curate_usgs_nwis.py --states CA,TX,FL   # subset of states
  python src/01c_curate_usgs_nwis.py --merge_only         # skip download
"""

import argparse
import socket
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# Prevent hung WQP HTTP calls from blocking indefinitely. If the server
# accepts a TCP connection but never responds, recv() will raise after
# this timeout and the retry logic will catch it.
socket.setdefaulttimeout(120)  # seconds

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw" / "usgs_nwis"
CHUNKS_DIR = RAW_DIR / "chunks"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

GENESIS_PARAMS = ['As', 'Fe', 'Mn', 'PO4', 'F', 'U', 'NO3',
                  'pH', 'Eh', 'EC', 'TDS',
                  'Ca', 'Mg', 'Na', 'K', 'Cl', 'HCO3', 'SO4',
                  'SiO2', 'DOC']

# WQP CharacteristicName(s) → GENESIS param
CHAR_TO_GENESIS = {
    "Arsenic": "As",
    "Iron": "Fe",
    "Manganese": "Mn",
    "Phosphorus": "PO4", "Orthophosphate": "PO4",
    "Phosphate-phosphorus": "PO4", "Orthophosphate as phosphorus": "PO4",
    "Fluoride": "F",
    "Uranium": "U",
    "Nitrate": "NO3", "Nitrate as nitrogen": "NO3", "Nitrate as N": "NO3",
    "pH": "pH", "pH, water, unfiltered, field": "pH",
    "Oxidation reduction potential (ORP)": "Eh",
    "Specific conductance": "EC",
    "Total dissolved solids": "TDS",
    "Calcium": "Ca",
    "Magnesium": "Mg",
    "Sodium": "Na",
    "Potassium": "K",
    "Chloride": "Cl",
    "Bicarbonate": "HCO3", "Alkalinity, bicarbonate": "HCO3",
    "Sulfate": "SO4", "Sulfate as SO4": "SO4",
    "Silica": "SiO2", "Silicon": "SiO2",
    "Organic carbon": "DOC",
}

# All WQP characteristicNames to query
WQP_CHAR_NAMES = [
    "Arsenic", "Iron", "Manganese", "Phosphate-phosphorus", "Orthophosphate",
    "Fluoride", "Uranium", "Nitrate",
    "pH", "Oxidation reduction potential (ORP)",
    "Specific conductance", "Total dissolved solids",
    "Calcium", "Magnesium", "Sodium", "Potassium", "Chloride",
    "Bicarbonate", "Sulfate", "Silica", "Organic carbon",
]

# Unit conversions: source_unit_lower → (target_unit, factor)
UNIT_CONVERT = {
    "ug/l":    {"mg/l": 0.001, "ug/l": 1.0},
    "mg/l":    {"mg/l": 1.0, "ug/l": 1000.0},
    "mg/l as n": {"mg/l": 1.0},
    "mg/l as p": {"mg/l": 1.0},
    "mg/l as caco3": {"mg/l": 1.22},  # CaCO3 → HCO3 factor
    "mg/l as sio2": {"mg/l": 1.0},
    "us/cm":   {"us/cm": 1.0},
    "umho/cm": {"us/cm": 1.0},
    "ms/cm":   {"us/cm": 1000.0},
    "mv":      {"mv": 1.0},
    "none":    {"std": 1.0},
    "std units": {"std": 1.0},
}

# Target units per GENESIS param
TARGET_UNITS = {
    "As": "ug/l", "Fe": "mg/l", "Mn": "mg/l", "PO4": "mg/l",
    "F": "mg/l", "U": "ug/l", "NO3": "mg/l",
    "pH": "std", "Eh": "mv", "EC": "us/cm", "TDS": "mg/l",
    "Ca": "mg/l", "Mg": "mg/l", "Na": "mg/l", "K": "mg/l",
    "Cl": "mg/l", "HCO3": "mg/l", "SO4": "mg/l",
    "SiO2": "mg/l", "DOC": "mg/l",
}

# Plausibility ranges
VALID_RANGES = {
    "As": (0, 10000), "Fe": (0, 500), "Mn": (0, 100), "PO4": (0, 50),
    "F": (0, 50), "U": (0, 10000), "NO3": (0, 1000),
    "pH": (2.0, 12.0), "Eh": (-500, 1000), "EC": (10, 200000),
    "TDS": (1, 200000), "Ca": (0, 5000), "Mg": (0, 5000),
    "Na": (0, 50000), "K": (0, 5000), "Cl": (0, 100000),
    "HCO3": (0, 5000), "SO4": (0, 50000), "SiO2": (0, 500), "DOC": (0, 500),
}

MIN_PARAMS = 3
TEMPORAL_GAP_YEARS = 5

# US state FIPS codes (all 50 + DC + territories)
US_STATE_FIPS = {
    "AL": "US:01", "AK": "US:02", "AZ": "US:04", "AR": "US:05",
    "CA": "US:06", "CO": "US:08", "CT": "US:09", "DE": "US:10",
    "DC": "US:11", "FL": "US:12", "GA": "US:13", "HI": "US:15",
    "ID": "US:16", "IL": "US:17", "IN": "US:18", "IA": "US:19",
    "KS": "US:20", "KY": "US:21", "LA": "US:22", "ME": "US:23",
    "MD": "US:24", "MA": "US:25", "MI": "US:26", "MN": "US:27",
    "MS": "US:28", "MO": "US:29", "MT": "US:30", "NE": "US:31",
    "NV": "US:32", "NH": "US:33", "NJ": "US:34", "NM": "US:35",
    "NY": "US:36", "NC": "US:37", "ND": "US:38", "OH": "US:39",
    "OK": "US:40", "OR": "US:41", "PA": "US:42", "RI": "US:44",
    "SC": "US:45", "SD": "US:46", "TN": "US:47", "TX": "US:48",
    "UT": "US:49", "VT": "US:50", "VA": "US:51", "WA": "US:53",
    "WV": "US:54", "WI": "US:55", "WY": "US:56",
}


# ============================================================
# DOWNLOAD — state by state using dataretrieval
# ============================================================

KEEP_COLS = [
    "MonitoringLocationIdentifier",
    "ActivityStartDate",
    "CharacteristicName",
    "ResultSampleFractionText",
    "ResultMeasureValue",
    "ResultMeasure/MeasureUnitCode",
    "ResultStatusIdentifier",
]

# Empty-chunk marker files (0-byte sentinel) indicate a chunk returned no data
# and should not be re-queried on resume.
EMPTY_MARKER = ".empty"


def _chunk_path(state_abbr: str, char_name: str, y0: int, y1: int) -> Path:
    safe = char_name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    return CHUNKS_DIR / f"{state_abbr}__{safe}__{y0}_{y1}.parquet"


def download_state(state_abbr: str, fips: str, start: str, end: str,
                   retries: int = 2, chunk_years: int = 5) -> pd.DataFrame:
    """Download all GENESIS-relevant characteristics for one state.

    Checkpointed: each (state, char, year-window) query result is saved as
    its own parquet file in CHUNKS_DIR. On resume, existing chunks are
    skipped. Empty results create a zero-byte `.empty` marker so they are
    not re-queried. The final per-state parquet is a merged view."""
    state_cache = RAW_DIR / f"{state_abbr}_results.parquet"
    if state_cache.exists():
        print(f"  [cached] {state_abbr}", flush=True)
        return pd.read_parquet(state_cache)

    import dataretrieval.wqp as wqp

    y_start, y_end = int(start), int(end)
    all_frames = []
    fail_count = 0
    ok_count = 0
    skip_count = 0

    for char_name in WQP_CHAR_NAMES:
        y = y_start
        while y <= y_end:
            y1 = min(y + chunk_years - 1, y_end)
            chunk_file = _chunk_path(state_abbr, char_name, y, y1)
            empty_marker = chunk_file.with_suffix(chunk_file.suffix + EMPTY_MARKER)

            # Resume: skip if chunk already fetched
            if chunk_file.exists():
                try:
                    all_frames.append(pd.read_parquet(chunk_file))
                    skip_count += 1
                except Exception:
                    chunk_file.unlink()  # corrupt — retry
                y = y1 + 1
                continue
            if empty_marker.exists():
                skip_count += 1
                y = y1 + 1
                continue

            for attempt in range(retries):
                try:
                    df, _ = wqp.get_results(
                        statecode=fips,
                        characteristicName=char_name,
                        siteType="Well",
                        providers="NWIS",
                        startDateLo=f"01-01-{y}",
                        startDateHi=f"12-31-{y1}",
                    )
                    if df.empty:
                        empty_marker.touch()
                    else:
                        cols = [c for c in KEEP_COLS if c in df.columns]
                        df = df[cols]
                        df.to_parquet(chunk_file, index=False)
                        all_frames.append(df)
                    ok_count += 1
                    print(f"    {state_abbr} {char_name[:18]:18s} {y}-{y1}: "
                          f"{len(df):>6,} rows", flush=True)
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        time.sleep(5)
                    else:
                        fail_count += 1
                        print(f"    {state_abbr} {char_name[:18]:18s} {y}-{y1}: FAIL",
                              flush=True)
            y = y1 + 1

    if not all_frames:
        print(f"  {state_abbr}: no data ({fail_count} fails, {skip_count} skipped)",
              flush=True)
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    result.to_parquet(state_cache, index=False)
    print(f"  {state_abbr}: {len(result):,} rows "
          f"({ok_count} ok, {fail_count} fail, {skip_count} resumed)",
          flush=True)
    return result


def download_all(states: list[str], start: str, end: str,
                 workers: int = 4) -> pd.DataFrame:
    """Download all states in parallel, concatenate."""
    print(f"\nDownloading from WQP via dataretrieval ({len(states)} states, {workers} workers)...")
    frames = []

    tasks = [(st, US_STATE_FIPS[st]) for st in states if st in US_STATE_FIPS]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(download_state, st, fips, start, end): st
            for st, fips in tasks
        }
        done = 0
        for fut in as_completed(futures):
            st = futures[fut]
            done += 1
            try:
                df = fut.result()
                if not df.empty:
                    frames.append(df)
                print(f"  [{done:>2}/{len(tasks)}] {st} ✓", flush=True)
            except Exception as e:
                print(f"  [{done:>2}/{len(tasks)}] {st} FAILED: {e}", flush=True)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ============================================================
# PARSE & NORMALIZE
# ============================================================

def _clean_unit(raw_unit: str) -> str:
    """Normalize WQP unit strings to our canonical form."""
    u = str(raw_unit).strip().lower()
    # Strip temperature qualifiers: "us/cm @25c" → "us/cm"
    if "@" in u:
        u = u.split("@")[0].strip()
    # Common aliases
    aliases = {
        "ug/l": "ug/l", "mg/l": "mg/l", "mg/l as n": "mg/l",
        "mg/l as p": "mg/l", "mg/l as caco3": "mg/l as caco3",
        "mg/l as sio2": "mg/l", "mg/l asnacl": "mg/l",
        "us/cm": "us/cm", "umho/cm": "us/cm", "ms/cm": "ms/cm",
        "mv": "mv", "none": "none", "std units": "std units",
        "pci/l": "pci/l", "deg c": "deg c",
    }
    return aliases.get(u, u)


def normalize_value(row, genesis_param: str) -> float:
    """Convert a WQP result row to the GENESIS target unit."""
    value = row.get("raw_value")
    if pd.isna(value):
        return np.nan

    try:
        value = float(value)
    except (ValueError, TypeError):
        return np.nan

    src_unit = _clean_unit(row.get("unit", ""))
    tgt_unit = TARGET_UNITS[genesis_param]

    if not src_unit or src_unit == "nan" or src_unit == "none":
        if genesis_param in ("pH", "Eh"):
            return value
        return np.nan

    if src_unit == tgt_unit:
        return value

    # Direct conversion table
    conv = UNIT_CONVERT.get(src_unit)
    if conv and tgt_unit in conv:
        return value * conv[tgt_unit]

    # mg↔ug fallback
    if src_unit == "ug/l" and tgt_unit == "mg/l":
        return value * 0.001
    if src_unit == "mg/l" and tgt_unit == "ug/l":
        return value * 1000.0
    # std units = pH
    if src_unit == "std units" and tgt_unit == "std":
        return value

    return np.nan


def process_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw WQP results → long format (site_id, date, param, value)."""
    if raw.empty:
        return pd.DataFrame()

    raw = raw.rename(columns={
        "MonitoringLocationIdentifier": "site_id",
        "ActivityStartDate": "date",
        "CharacteristicName": "char_name",
        "ResultSampleFractionText": "fraction",
        "ResultMeasureValue": "raw_value",
        "ResultMeasure/MeasureUnitCode": "unit",
        "ResultStatusIdentifier": "status",
    })

    # Drop rejected
    if "status" in raw.columns:
        bad = raw["status"].fillna("").str.lower().str.contains("rejected")
        raw = raw[~bad]

    # Map characteristic → GENESIS param
    raw["param"] = raw["char_name"].map(CHAR_TO_GENESIS)
    raw = raw.dropna(subset=["param"])

    # Convert values
    records = []
    for param in raw["param"].unique():
        sub = raw[raw["param"] == param].copy()
        sub["value"] = sub.apply(lambda r: normalize_value(r, param), axis=1)
        sub = sub.dropna(subset=["value"])
        lo, hi = VALID_RANGES.get(param, (None, None))
        if lo is not None:
            sub = sub[(sub["value"] >= lo) & (sub["value"] <= hi)]
        records.append(sub[["site_id", "date", "param", "fraction", "value"]])

    if not records:
        return pd.DataFrame()

    long = pd.concat(records, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long = long.dropna(subset=["date"])
    return long


def aggregate_to_vectors(long: pd.DataFrame) -> pd.DataFrame:
    """Long → wide: one row per (site_id, date), columns = GENESIS params.
    Prefer Dissolved > Total > other fractions.
    """
    if long.empty:
        return pd.DataFrame()

    frac_rank = {"Dissolved": 1, "Total": 2}
    long["frank"] = long["fraction"].fillna("").map(lambda f: frac_rank.get(f, 3))

    best = (
        long.sort_values(["site_id", "date", "param", "frank"])
        .groupby(["site_id", "date", "param"], as_index=False)
        .agg(value=("value", "median"))
    )

    wide = best.pivot_table(
        index=["site_id", "date"],
        columns="param",
        values="value",
        aggfunc="median",
    ).reset_index()

    for p in GENESIS_PARAMS:
        if p not in wide.columns:
            wide[p] = np.nan

    wide["n_params"] = wide[GENESIS_PARAMS].notna().sum(axis=1)
    wide = wide[wide["n_params"] >= MIN_PARAMS]
    return wide


# ============================================================
# STATION METADATA (lat/lon from NWIS site service)
# ============================================================

def attach_metadata(vectors: pd.DataFrame) -> pd.DataFrame:
    """Get lat/lon by parsing USGS site IDs (format encodes lat/lon for many sites)
    or via WQP Station endpoint as fallback. NWIS site service skipped (DNS issues)."""
    cache = RAW_DIR / "site_metadata.parquet"
    if cache.exists():
        meta = pd.read_parquet(cache)
    else:
        # Try WQP Station endpoint (same host as Result endpoint, works reliably)
        import dataretrieval.wqp as wqp
        site_ids = vectors["site_id"].unique().tolist()
        print(f"\nFetching metadata for {len(site_ids):,} sites via WQP...")
        meta_frames = []
        # Query in chunks of 200 sites
        for i in range(0, len(site_ids), 200):
            chunk = site_ids[i:i+200]
            try:
                df, _ = wqp.get_stations(siteid=";".join(chunk), providers="NWIS")
                if not df.empty:
                    meta_frames.append(df)
            except Exception:
                pass  # silent — metadata is best-effort

        if meta_frames:
            meta = pd.concat(meta_frames, ignore_index=True)
            keep = ["MonitoringLocationIdentifier", "LatitudeMeasure", "LongitudeMeasure"]
            keep = [c for c in keep if c in meta.columns]
            meta = meta[keep].drop_duplicates("MonitoringLocationIdentifier")
            meta = meta.rename(columns={
                "MonitoringLocationIdentifier": "site_no",
                "LatitudeMeasure": "dec_lat_va",
                "LongitudeMeasure": "dec_long_va",
            })
            meta.to_parquet(cache, index=False)
            print(f"  got metadata for {len(meta):,} sites")
        else:
            meta = pd.DataFrame()

    if meta.empty:
        vectors["lat"] = np.nan
        vectors["lon"] = np.nan
        vectors["country"] = "USA"
        return vectors

    # site_no is already the full WQP identifier like "USGS-12345678"
    meta["site_id"] = meta["site_no"].astype(str)
    meta = meta.rename(columns={"dec_lat_va": "lat", "dec_long_va": "lon"})
    meta["lat"] = pd.to_numeric(meta["lat"], errors="coerce")
    meta["lon"] = pd.to_numeric(meta["lon"], errors="coerce")
    meta["country"] = "USA"

    return vectors.merge(meta[["site_id", "lat", "lon", "country"]],
                         on="site_id", how="left")


# ============================================================
# TEMPORAL PAIRS
# ============================================================

def build_temporal_pairs(vectors: pd.DataFrame) -> pd.DataFrame:
    """For each site with ≥2 samples spanning ≥5 years, build t0→t1 pair."""
    if vectors.empty:
        return pd.DataFrame()

    vectors = vectors.sort_values(["site_id", "date"])
    pairs = []

    for site_id, grp in vectors.groupby("site_id", sort=False):
        if len(grp) < 2:
            continue
        t0 = grp.iloc[0]
        t1 = grp.iloc[-1]
        gap_years = (t1["date"] - t0["date"]).days / 365.25
        if gap_years < TEMPORAL_GAP_YEARS:
            continue

        n_paired = sum(
            1 for p in GENESIS_PARAMS
            if pd.notna(t0.get(p)) and pd.notna(t1.get(p))
        )
        if n_paired < MIN_PARAMS:
            continue

        row = {
            "station_id": f"USGS_{site_id}",
            "date_t0": t0["date"].strftime("%Y-%m-%d"),
            "date_t1": t1["date"].strftime("%Y-%m-%d"),
            "year_t0": int(t0["date"].year),
            "year_t1": int(t1["date"].year),
            "gap_years": round(gap_years, 2),
            "country": t0.get("country", "USA") or "USA",
            "lat": t0.get("lat"),
            "lon": t0.get("lon"),
            "n_paired_params": int(n_paired),
            "source": "USGS_NWIS",
        }
        for p in GENESIS_PARAMS:
            v0 = t0.get(p)
            v1 = t1.get(p)
            row[f"{p}_t0"] = v0
            row[f"{p}_t1"] = v1
            row[f"delta_{p}"] = (v1 - v0) if (pd.notna(v0) and pd.notna(v1)) else np.nan
        pairs.append(row)

    return pd.DataFrame(pairs)


# ============================================================
# MERGE WITH EXISTING
# ============================================================

def merge_with_gemstat(usgs_pairs: pd.DataFrame) -> pd.DataFrame:
    existing_path = PROCESSED_DIR / "final_temporal_pairs.parquet"
    if not existing_path.exists():
        print(f"  {existing_path.name} missing — USGS pairs standalone", flush=True)
        return usgs_pairs

    existing = pd.read_parquet(existing_path)
    print(f"\nExisting GEMStat pairs: {len(existing):,}")
    print(f"New USGS NWIS pairs:    {len(usgs_pairs):,}")

    all_cols = list(existing.columns)
    for c in all_cols:
        if c not in usgs_pairs.columns:
            usgs_pairs[c] = np.nan
    usgs_pairs = usgs_pairs[all_cols]

    merged = pd.concat([existing, usgs_pairs], ignore_index=True)
    merged = merged.drop_duplicates(subset=["station_id", "date_t0", "date_t1"])
    print(f"Merged total:           {len(merged):,}")
    return merged


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="1990")
    parser.add_argument("--end", default="2023")
    parser.add_argument("--states", default=None,
                        help="Comma-separated state codes, e.g. CA,TX,FL (default: all 50+DC)")
    parser.add_argument("--merge_only", action="store_true")
    args = parser.parse_args()

    if args.states:
        states = [s.strip().upper() for s in args.states.split(",")]
    else:
        states = sorted(US_STATE_FIPS.keys())

    print("=" * 60)
    print("GENESIS Paper 5 — USGS NWIS Data Enrichment")
    print(f"States: {len(states)}  |  Range: {args.start}–{args.end}")
    print("=" * 60)

    if not args.merge_only:
        raw = download_all(states, args.start, args.end)
    else:
        # Load cached per-state parquets AND any partial chunks
        state_files = sorted(RAW_DIR.glob("*_results.parquet"))
        chunk_files = sorted(CHUNKS_DIR.glob("*.parquet"))

        # Only use chunks for states that don't have a merged state file
        covered_states = {p.stem.split("_")[0] for p in state_files}
        partial_chunks = [
            p for p in chunk_files
            if p.stem.split("__")[0] not in covered_states
        ]

        frames = [pd.read_parquet(p) for p in state_files]
        if partial_chunks:
            frames.extend(pd.read_parquet(p) for p in partial_chunks)

        if not frames:
            print("No cached data. Run without --merge_only first.")
            return
        raw = pd.concat(frames, ignore_index=True)
        print(f"\nLoaded {len(raw):,} rows "
              f"({len(state_files)} full states + {len(partial_chunks)} partial chunks).")

    if raw.empty:
        print("No data retrieved.")
        return

    print(f"\nProcessing {len(raw):,} raw result rows...")
    long = process_raw(raw)
    print(f"  Clean measurements: {len(long):,}")

    print("\nAggregating to per (site, date) vectors...")
    vectors = aggregate_to_vectors(long)
    if vectors.empty or "site_id" not in vectors.columns:
        print("  No vectors created. Check parameter mapping / units.")
        return
    print(f"  Vectors: {len(vectors):,}")
    print(f"  Unique sites: {vectors['site_id'].nunique():,}")

    vectors = attach_metadata(vectors)

    vec_out = PROCESSED_DIR / "usgs_nwis_vectors.parquet"
    vectors.to_parquet(vec_out, index=False)
    print(f"  → {vec_out.name}")

    print("\nBuilding temporal pairs (≥5 yr gap)...")
    pairs = build_temporal_pairs(vectors)
    print(f"  Pairs: {len(pairs):,}")

    if not pairs.empty:
        pairs_out = PROCESSED_DIR / "usgs_nwis_temporal_pairs.parquet"
        pairs.to_parquet(pairs_out, index=False)
        print(f"  → {pairs_out.name}")

    print("\nMerging with GEMStat...")
    merged = merge_with_gemstat(pairs)
    merged_out = PROCESSED_DIR / "final_temporal_pairs_enriched.parquet"
    merged.to_parquet(merged_out, index=False)
    print(f"  → {merged_out.name}")

    # Report
    report = PROCESSED_DIR / "usgs_nwis_curation_report.txt"
    with open(report, "w") as f:
        f.write("GENESIS Paper 5 — USGS NWIS Curation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date range:          {args.start}–{args.end}\n")
        f.write(f"States:              {len(states)}\n")
        f.write(f"Raw results:         {len(raw):,}\n")
        f.write(f"Clean measurements:  {len(long):,}\n")
        f.write(f"Unique sites:        {vectors['site_id'].nunique():,}\n")
        f.write(f"Site-date vectors:   {len(vectors):,}\n")
        f.write(f"Temporal pairs:      {len(pairs):,}\n")
        f.write(f"Merged total pairs:  {len(merged):,}\n\n")
        f.write("Per-parameter coverage in vectors:\n")
        for p in GENESIS_PARAMS:
            if p in vectors.columns:
                n = vectors[p].notna().sum()
                pct = 100 * n / len(vectors) if len(vectors) > 0 else 0
                f.write(f"  {p:5s}: {n:>8,} ({pct:5.1f}%)\n")
    print(f"\n→ {report.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
