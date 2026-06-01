#!/usr/bin/env python3
"""
GENESIS Paper 5 — Global Groundwater Chemistry Download & Harmonize
=====================================================================
Downloads from public groundwater chemistry sources, harmonizes into
the same 20-parameter GENESIS schema, and saves checkpointed parquets.

Sources:
  1. BRGM ADES / Hub'Eau   (France, REST API — fully automated)
  2. EEA Waterbase WISE-6  (EU-27, manual download → auto-process)
  3. EA Water Quality       (England, manual download → auto-process)

Usage:
  python src/02_download_global_sources.py                # all sources
  python src/02_download_global_sources.py --only ades    # single source
  python src/02_download_global_sources.py --merge_only   # just merge cached

Checkpoint: each (department × parameter) chunk is saved as a parquet.
Restart picks up where it left off.
"""

import argparse
import json
import os
import socket
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ============================================================
# PATHS
# ============================================================

PAPER5 = Path(__file__).resolve().parent.parent
RAW_DIR = PAPER5 / "data" / "raw"
PROCESSED_DIR = PAPER5 / "data" / "processed"
LOG_DIR = PAPER5 / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============================================================
# GENESIS parameter vocabulary (must match genesis_encoder.py)
# ============================================================

GENESIS_PARAMS = [
    'As', 'Fe', 'Mn', 'PO4', 'F', 'U', 'NO3',
    'pH', 'Eh', 'EC', 'TDS',
    'Ca', 'Mg', 'Na', 'K', 'Cl', 'HCO3', 'SO4',
    'SiO2', 'DOC',
]

MIN_PARAMS = 3
TEMPORAL_GAP_YEARS = 5

VALID_RANGES = {
    "As": (0, 10000), "Fe": (0, 500), "Mn": (0, 100), "PO4": (0, 50),
    "F": (0, 50), "U": (0, 10000), "NO3": (0, 1000),
    "pH": (2.0, 12.0), "Eh": (-500, 1000), "EC": (10, 200000),
    "TDS": (1, 200000), "Ca": (0, 5000), "Mg": (0, 5000),
    "Na": (0, 50000), "K": (0, 5000), "Cl": (0, 100000),
    "HCO3": (0, 5000), "SO4": (0, 50000), "SiO2": (0, 500), "DOC": (0, 500),
}

socket.setdefaulttimeout(180)


# ============================================================
# LOGGING
# ============================================================

_LOGFILE = None

def _log(msg: str):
    line = f"    {msg}"
    print(line, flush=True)
    if _LOGFILE:
        _LOGFILE.write(line + "\n")
        _LOGFILE.flush()


# ============================================================
# SHARED: clean + pivot + temporal pairs
# ============================================================

def clean_and_pivot(long_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Long (site_id, date, param, value, lat, lon, country) → wide GENESIS."""
    if long_df.empty:
        return pd.DataFrame()

    long_df = long_df.copy()
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value", "param", "site_id", "date"])
    long_df = long_df[long_df["param"].isin(GENESIS_PARAMS)]

    # Range filter (vectorized)
    mask = pd.Series(True, index=long_df.index)
    for p, (lo, hi) in VALID_RANGES.items():
        pmask = long_df["param"] == p
        mask &= ~(pmask & ((long_df["value"] < lo) | (long_df["value"] > hi)))
    long_df = long_df[mask]

    if long_df.empty:
        return pd.DataFrame()

    agg = (long_df.groupby(["site_id", "date", "param"], as_index=False)
           .agg(value=("value", "median"),
                lat=("lat", "first"), lon=("lon", "first"),
                country=("country", "first")))

    wide = agg.pivot_table(
        index=["site_id", "date", "lat", "lon", "country"],
        columns="param", values="value", aggfunc="median",
    ).reset_index()

    for p in GENESIS_PARAMS:
        if p not in wide.columns:
            wide[p] = np.nan

    wide["n_params"] = wide[GENESIS_PARAMS].notna().sum(axis=1)
    wide = wide[wide["n_params"] >= MIN_PARAMS]
    wide["source"] = source_name
    return wide


def build_temporal_pairs(vectors: pd.DataFrame, source_tag: str) -> pd.DataFrame:
    if vectors.empty:
        return pd.DataFrame()
    vectors = vectors.copy()
    vectors["date"] = pd.to_datetime(vectors["date"], errors="coerce")
    vectors = vectors.dropna(subset=["date"]).sort_values(["site_id", "date"])

    pairs = []
    for site_id, grp in vectors.groupby("site_id", sort=False):
        if len(grp) < 2:
            continue
        t0, t1 = grp.iloc[0], grp.iloc[-1]
        gap = (t1["date"] - t0["date"]).days / 365.25
        if gap < TEMPORAL_GAP_YEARS:
            continue
        n_paired = sum(1 for p in GENESIS_PARAMS
                       if pd.notna(t0.get(p)) and pd.notna(t1.get(p)))
        if n_paired < MIN_PARAMS:
            continue
        row = {
            "station_id": f"{source_tag}_{site_id}",
            "date_t0": t0["date"].strftime("%Y-%m-%d"),
            "date_t1": t1["date"].strftime("%Y-%m-%d"),
            "year_t0": int(t0["date"].year),
            "year_t1": int(t1["date"].year),
            "gap_years": round(gap, 2),
            "country": t0.get("country", ""),
            "lat": t0.get("lat"), "lon": t0.get("lon"),
            "n_paired_params": int(n_paired), "source": source_tag,
        }
        for p in GENESIS_PARAMS:
            row[f"{p}_t0"] = t0.get(p)
            row[f"{p}_t1"] = t1.get(p)
            row[f"delta_{p}"] = (
                (t1.get(p) - t0.get(p))
                if pd.notna(t0.get(p)) and pd.notna(t1.get(p)) else np.nan)
        pairs.append(row)
    return pd.DataFrame(pairs)


# ############################################################
# SOURCE 1: BRGM ADES / Hub'Eau (France)
# ############################################################

# Hub'Eau code → GENESIS param
HUBEAU_PARAM_MAP = {
    # Verified against Hub'Eau API 2026-04-13
    "1369": "As",    # Arsenic, µg(As)/L → µg/L (no conversion)
    "1393": "Fe",    # Fer, µg(Fe)/L → mg/L (÷1000 in curation)
    "1394": "Mn",    # Manganèse, µg(Mn)/L → mg/L (÷1000 in curation)
    "1433": "PO4",   # Orthophosphates, mg(PO4)/L → mg/L (no conversion)
    "1391": "F",     # Fluor, mg(F)/L → mg/L (no conversion)
    "1361": "U",     # Uranium, mg(U)/L → µg/L (×1000 in curation)
    "1340": "NO3",   # Nitrates, mg(NO3)/L → mg/L (no conversion)
    "1302": "pH",    # pH, unité pH (no conversion)
    "1330": "Eh",    # Potentiel REDOX, mV (no conversion)
    "1303": "EC",    # Conductivité à 25°C, µS/cm (no conversion)
    "1307": "TDS",   # Matière sèche à 105°C (≈TDS), mg/L (no conversion)
    "1374": "Ca",    # Calcium, mg(Ca)/L → mg/L (no conversion)
    "1372": "Mg",    # Magnésium, mg(Mg)/L → mg/L (no conversion)
    "1375": "Na",    # Sodium, mg(Na)/L → mg/L (no conversion)
    "1367": "K",     # Potassium, mg(K)/L → mg/L (no conversion)
    "1337": "Cl",    # Chlorures, mg(Cl)/L → mg/L (no conversion)
    "1327": "HCO3",  # Hydrogénocarbonates, mg(HCO3)/L → mg/L (no conversion)
    "1338": "SO4",   # Sulfates, mg(SO4)/L → mg/L (no conversion)
    "1348": "SiO2",  # Silice, mg(SiO2)/L → mg/L (no conversion)
    "1841": "DOC",   # Carbone Organique, mg(C)/L → mg/L (proxy)
    # REMOVED: 1301 (Temperature °C, NOT Eh)
    # REMOVED: 1305 (Suspended Matter, NOT TDS)
    # REMOVED: 1395 (Molybdenum, NOT Chloride)
}

HUBEAU_CODES = sorted(set(HUBEAU_PARAM_MAP.keys()))

# French departments (mainland + DOM-TOM)
FR_DEPTS = [f"{i:02d}" for i in range(1, 96)] + ["2A", "2B"] + \
           [f"{i:03d}" for i in range(971, 977)]


def _hubeau_fetch_page(dept: str, code: str, page: int = 1) -> tuple:
    """Single paginated fetch. Returns (records, has_more)."""
    url = "https://hubeau.eaufrance.fr/api/v1/qualite_nappes/analyses"
    params = {
        "num_departement": dept,
        "code_param": code,
        "date_debut_prelevement": "1990-01-01",
        "date_fin_prelevement": "2024-12-31",
        "size": 20000,
        "page": page,
        "format": "json",
    }
    resp = requests.get(url, params=params, timeout=120)
    if resp.status_code not in (200, 206):
        return [], False
    data = resp.json()
    records = data.get("data", [])
    has_more = (resp.status_code == 206 and len(records) == 20000)
    return records, has_more


def _hubeau_fetch_all(dept: str, code: str) -> list:
    """Paginated fetch, all pages."""
    all_recs = []
    page = 1
    while page <= 50:  # safety cap
        recs, more = _hubeau_fetch_page(dept, code, page)
        if not recs:
            break
        all_recs.extend(recs)
        if not more:
            break
        page += 1
    return all_recs


def _hubeau_to_long(records: list, param_name: str) -> pd.DataFrame:
    """Convert raw Hub'Eau records to long-format DataFrame."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    long = pd.DataFrame({
        "site_id": df.get("bss_id", df.get("code_bss", "")).astype(str),
        "date": pd.to_datetime(df.get("date_debut_prelevement"), errors="coerce"),
        "param": param_name,
        "value": pd.to_numeric(df.get("resultat"), errors="coerce"),
        "lat": pd.to_numeric(df.get("latitude"), errors="coerce"),
        "lon": pd.to_numeric(df.get("longitude"), errors="coerce"),
        "country": "FR",
    })
    # Unit: Hub'Eau returns mg/L for most, µg/L for trace metals
    # As and U targets are µg/L; if the source gives mg/L we multiply by 1000
    if "symbole_unite" in df.columns:
        units = df["symbole_unite"].fillna("").str.lower()
        if param_name in ("As", "U"):
            mg_mask = units.str.contains("mg", na=False) & ~units.str.contains("µg|ug", na=False)
            if mg_mask.any():
                long.loc[mg_mask.values, "value"] *= 1000.0
        elif param_name not in ("pH", "Eh", "EC"):
            ug_mask = units.str.contains("µg|ug", na=False)
            if ug_mask.any():
                long.loc[ug_mask.values, "value"] *= 0.001

    return long.dropna(subset=["date", "value", "site_id"])


def download_ades() -> pd.DataFrame:
    """Download France groundwater chemistry via Hub'Eau (checkpointed)."""
    src_dir = RAW_DIR / "brgm_ades"
    src_dir.mkdir(parents=True, exist_ok=True)
    vectors_cache = src_dir / "ades_vectors.parquet"

    if vectors_cache.exists():
        _log("[ADES] Loading cached vectors")
        return pd.read_parquet(vectors_cache)

    chunk_dir = src_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)

    _log(f"[ADES] Downloading via Hub'Eau API "
         f"({len(FR_DEPTS)} depts × {len(HUBEAU_CODES)} params) ...")

    all_long = []
    total_chunks = len(FR_DEPTS) * len(HUBEAU_CODES)
    done = 0
    skipped = 0

    for dept in FR_DEPTS:
        for code in HUBEAU_CODES:
            done += 1
            param_name = HUBEAU_PARAM_MAP[code]
            chunk_file = chunk_dir / f"{dept}_{code}.parquet"
            empty_file = chunk_dir / f"{dept}_{code}.empty"

            if chunk_file.exists():
                df = pd.read_parquet(chunk_file)
                if not df.empty:
                    all_long.append(df)
                skipped += 1
                continue
            if empty_file.exists():
                skipped += 1
                continue

            try:
                records = _hubeau_fetch_all(dept, code)
            except Exception as e:
                _log(f"[ADES] {dept}/{param_name}: FAIL ({e})")
                continue

            if not records:
                empty_file.touch()
                continue

            long = _hubeau_to_long(records, param_name)
            if long.empty:
                empty_file.touch()
                continue

            long.to_parquet(chunk_file, index=False)
            all_long.append(long)

            if done % 50 == 0 or len(long) > 5000:
                _log(f"[ADES] {done}/{total_chunks}  {dept}/{param_name}: "
                     f"{len(long):,} rows  (total so far: "
                     f"{sum(len(d) for d in all_long):,})")

    _log(f"[ADES] Download complete: {done} chunks, {skipped} cached, "
         f"{len(all_long)} with data")

    if not all_long:
        _log("[ADES] No data")
        return pd.DataFrame()

    combined = pd.concat(all_long, ignore_index=True)
    _log(f"[ADES] Total measurements: {len(combined):,}")

    vectors = clean_and_pivot(combined, "BRGM_ADES")
    _log(f"[ADES] Vectors (≥{MIN_PARAMS} params): {len(vectors):,}")

    if not vectors.empty:
        vectors.to_parquet(vectors_cache, index=False)
    return vectors


# ############################################################
# SOURCE 2: EEA Waterbase WISE-6 (EU-27) — manual download
# ############################################################

# EEA determinand codes → GENESIS param
EEA_PARAM_MAP = {
    "EEA_3103-01-3": "As", "CAS_7440-38-2": "As",
    "EEA_3151-01-3": "Fe", "CAS_7439-89-6": "Fe",
    "EEA_3157-01-3": "Mn", "CAS_7439-96-5": "Mn",
    "EEA_3153-01-3": "PO4", "CAS_14265-44-2": "PO4",
    "EEA_3129-01-3": "F", "CAS_16984-48-8": "F",
    "EEA_3195-01-3": "U", "CAS_7440-61-1": "U",
    "EEA_3161-01-3": "NO3", "CAS_14797-55-8": "NO3",
    "EEA_3177-01-3": "pH",
    "EEA_3121-01-3": "EC",
    "EEA_3109-01-3": "Ca", "CAS_7440-70-2": "Ca",
    "EEA_3155-01-3": "Mg", "CAS_7439-95-4": "Mg",
    "EEA_3185-01-3": "Na", "CAS_7440-23-5": "Na",
    "EEA_3181-01-3": "K", "CAS_7440-09-7": "K",
    "EEA_3113-01-3": "Cl", "CAS_16887-00-6": "Cl",
    "EEA_3105-01-3": "HCO3",
    "EEA_3187-01-3": "SO4", "CAS_14808-79-8": "SO4",
    "EEA_3183-01-3": "SiO2",
    "EEA_3119-01-3": "DOC",
    "EEA_3189-01-3": "TDS",
    "EEA_3123-01-3": "Eh",
}

EEA_UNIT_FACTORS = {
    ("mg/l", "mg/l"): 1.0, ("ug/l", "ug/l"): 1.0,
    ("ug/l", "mg/l"): 0.001, ("mg/l", "ug/l"): 1000.0,
    ("ng/l", "ug/l"): 0.001,
    ("us/cm", "us/cm"): 1.0, ("µs/cm", "us/cm"): 1.0,
    ("ms/cm", "us/cm"): 1000.0, ("ms/m", "us/cm"): 10.0,
    ("mv", "mv"): 1.0, ("v", "mv"): 1000.0,
}


def process_eea() -> pd.DataFrame:
    """Process a manually-downloaded EEA Waterbase CSV/ZIP."""
    src_dir = RAW_DIR / "eea_wise"
    src_dir.mkdir(parents=True, exist_ok=True)
    vectors_cache = src_dir / "eea_vectors.parquet"

    if vectors_cache.exists():
        _log("[EEA] Loading cached vectors")
        return pd.read_parquet(vectors_cache)

    # Look for manually-placed files
    candidates = list(src_dir.glob("*.csv")) + list(src_dir.glob("*.zip"))
    if not candidates:
        _log("[EEA] ⚠ No CSV/ZIP found in data/raw/eea_wise/")
        _log("[EEA] To add EEA data:")
        _log("[EEA]   1. Visit https://www.eea.europa.eu/en/datahub/datahubitem-view/"
             "4c5a8252-361a-4da2-a3f2-90e43d4fbc2e")
        _log("[EEA]   2. Download the 'Disaggregated data' CSV/ZIP")
        _log("[EEA]   3. Place it in data/raw/eea_wise/")
        _log("[EEA]   4. Re-run this script")
        return pd.DataFrame()

    csv_path = None
    for f in candidates:
        if f.suffix == ".zip":
            _log(f"[EEA] Extracting {f.name} ...")
            with zipfile.ZipFile(f) as zf:
                csvs = [n for n in zf.namelist() if n.endswith(".csv")]
                if csvs:
                    biggest = max(csvs, key=lambda n: zf.getinfo(n).file_size)
                    zf.extract(biggest, src_dir)
                    csv_path = src_dir / biggest
            break
        elif f.suffix == ".csv":
            csv_path = f
            break

    if csv_path is None or not csv_path.exists():
        _log("[EEA] No usable CSV found after extraction")
        return pd.DataFrame()

    _log(f"[EEA] Parsing {csv_path.name} in chunks ...")
    chunk_dir = src_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)

    all_long = []
    chunk_n = 0

    # Try comma first, fall back to tab
    for sep in [",", "\t", ";"]:
        try:
            test = pd.read_csv(csv_path, sep=sep, nrows=5)
            if len(test.columns) > 3:
                break
        except Exception:
            continue

    for chunk_df in pd.read_csv(csv_path, sep=sep, low_memory=False,
                                chunksize=500_000):
        chunk_n += 1
        chunk_file = chunk_dir / f"eea_chunk_{chunk_n:04d}.parquet"

        if chunk_file.exists():
            df = pd.read_parquet(chunk_file)
            if not df.empty:
                all_long.append(df)
            _log(f"[EEA]   chunk {chunk_n}: cached ({len(df):,} rows)")
            continue

        # Filter for groundwater
        gw_col = None
        for col in chunk_df.columns:
            if "waterbody" in col.lower() and "category" in col.lower():
                gw_col = col
                break
            if "matrix" in col.lower():
                gw_col = col
                break

        if gw_col:
            gw = chunk_df[chunk_df[gw_col].astype(str).str.upper().str.contains("GW")].copy()
        else:
            gw = chunk_df.copy()

        if gw.empty:
            _log(f"[EEA]   chunk {chunk_n}: 0 GW rows")
            continue

        # Find columns by partial name match
        def _find_col(df, *keywords):
            for col in df.columns:
                cl = col.lower()
                if all(kw in cl for kw in keywords):
                    return col
            return None

        det_col = _find_col(gw, "determinand") or _find_col(gw, "observed", "property")
        val_col = _find_col(gw, "result", "value") or _find_col(gw, "result", "observed")
        unit_col = _find_col(gw, "result", "uom") or _find_col(gw, "unit")
        date_col = _find_col(gw, "date") or _find_col(gw, "time", "sampling")
        site_col = _find_col(gw, "site", "identifier") or _find_col(gw, "station")
        country_col = _find_col(gw, "country")
        lat_col = _find_col(gw, "lat")
        lon_col = _find_col(gw, "lon")

        if not det_col or not val_col:
            _log(f"[EEA]   chunk {chunk_n}: missing det/val columns. "
                 f"Cols: {list(gw.columns)[:10]}")
            continue

        gw["param"] = gw[det_col].map(EEA_PARAM_MAP)
        gw = gw.dropna(subset=["param"])

        if gw.empty:
            _log(f"[EEA]   chunk {chunk_n}: 0 mapped params")
            continue

        # Unit conversion
        gw["value"] = pd.to_numeric(gw[val_col], errors="coerce")
        if unit_col:
            for idx in gw.index:
                p = gw.at[idx, "param"]
                src_u = str(gw.at[idx, unit_col]).strip().lower()
                tgt_u = {"As": "ug/l", "U": "ug/l", "EC": "us/cm",
                         "pH": "std", "Eh": "mv"}.get(p, "mg/l")
                factor = EEA_UNIT_FACTORS.get((src_u, tgt_u))
                if factor and factor != 1.0:
                    gw.at[idx, "value"] *= factor

        long = pd.DataFrame({
            "site_id": gw[site_col].astype(str) if site_col else "unknown",
            "date": pd.to_datetime(gw[date_col], errors="coerce") if date_col else pd.NaT,
            "param": gw["param"],
            "value": gw["value"],
            "lat": pd.to_numeric(gw[lat_col], errors="coerce") if lat_col else np.nan,
            "lon": pd.to_numeric(gw[lon_col], errors="coerce") if lon_col else np.nan,
            "country": gw[country_col].astype(str) if country_col else "EU",
        })
        long = long.dropna(subset=["date", "value"])
        long.to_parquet(chunk_file, index=False)
        all_long.append(long)
        _log(f"[EEA]   chunk {chunk_n}: {len(long):,} GW measurements")

    if not all_long:
        _log("[EEA] No data extracted")
        return pd.DataFrame()

    combined = pd.concat(all_long, ignore_index=True)
    _log(f"[EEA] Total GW measurements: {len(combined):,}")

    vectors = clean_and_pivot(combined, "EEA_WISE")
    _log(f"[EEA] Vectors (≥{MIN_PARAMS} params): {len(vectors):,}")

    if not vectors.empty:
        vectors.to_parquet(vectors_cache, index=False)
    return vectors


# ############################################################
# SOURCE 3: EA Water Quality (England) — manual download
# ############################################################

EA_PARAM_MAP = {
    "arsenic": "As", "arsenic - dissolved": "As", "arsenic, dissolved": "As",
    "iron": "Fe", "iron - dissolved": "Fe", "iron, dissolved": "Fe",
    "manganese": "Mn", "manganese - dissolved": "Mn",
    "orthophosphate": "PO4", "phosphate": "PO4",
    "fluoride": "F", "uranium": "U",
    "nitrate": "NO3", "nitrate as no3": "NO3",
    "ph": "pH",
    "oxidation reduction potential": "Eh", "redox potential": "Eh",
    "conductivity": "EC", "electrical conductivity": "EC",
    "specific electrical conductance": "EC",
    "total dissolved solids": "TDS",
    "calcium": "Ca", "magnesium": "Mg", "sodium": "Na", "potassium": "K",
    "chloride": "Cl",
    "alkalinity to ph 4.5 as hco3": "HCO3", "bicarbonate": "HCO3",
    "sulphate": "SO4", "sulfate": "SO4",
    "silica": "SiO2", "reactive silica": "SiO2",
    "dissolved organic carbon": "DOC",
}


def process_ea() -> pd.DataFrame:
    """Process a manually-downloaded EA Water Quality CSV."""
    src_dir = RAW_DIR / "ea_water_quality"
    src_dir.mkdir(parents=True, exist_ok=True)
    vectors_cache = src_dir / "ea_vectors.parquet"

    if vectors_cache.exists():
        _log("[EA] Loading cached vectors")
        return pd.read_parquet(vectors_cache)

    candidates = list(src_dir.glob("*.csv")) + list(src_dir.glob("*.zip"))
    if not candidates:
        _log("[EA] ⚠ No CSV/ZIP found in data/raw/ea_water_quality/")
        _log("[EA] To add EA data:")
        _log("[EA]   1. Visit https://environment.data.gov.uk/water-quality/view/download")
        _log("[EA]   2. Select: Area=all, Year=all, Type=Groundwater")
        _log("[EA]   3. Download the CSV")
        _log("[EA]   4. Place it in data/raw/ea_water_quality/")
        _log("[EA]   5. Re-run this script")
        return pd.DataFrame()

    csv_path = None
    for f in candidates:
        if f.suffix == ".zip":
            _log(f"[EA] Extracting {f.name} ...")
            with zipfile.ZipFile(f) as zf:
                csvs = [n for n in zf.namelist() if n.endswith(".csv")]
                if csvs:
                    biggest = max(csvs, key=lambda n: zf.getinfo(n).file_size)
                    zf.extract(biggest, src_dir)
                    csv_path = src_dir / biggest
            break
        elif f.suffix == ".csv":
            csv_path = f
            break

    if csv_path is None or not csv_path.exists():
        _log("[EA] No usable CSV found")
        return pd.DataFrame()

    _log(f"[EA] Parsing {csv_path.name} ...")

    all_long = []
    for chunk_df in pd.read_csv(csv_path, low_memory=False, chunksize=500_000):
        # Flexible column detection
        def _find_col(df, *keywords):
            for col in df.columns:
                cl = col.lower()
                if all(kw in cl for kw in keywords):
                    return col
            return None

        det_col = _find_col(chunk_df, "determinand") or _find_col(chunk_df, "parameter")
        val_col = _find_col(chunk_df, "result") or _find_col(chunk_df, "value")
        date_col = _find_col(chunk_df, "date") or _find_col(chunk_df, "time")
        site_col = _find_col(chunk_df, "sampling", "point") or _find_col(chunk_df, "site")
        lat_col = _find_col(chunk_df, "lat")
        lon_col = _find_col(chunk_df, "lon") or _find_col(chunk_df, "long")
        unit_col = _find_col(chunk_df, "unit")

        if not det_col or not val_col:
            _log(f"[EA] Can't find det/val columns. Cols: {list(chunk_df.columns)[:15]}")
            continue

        chunk_df["param"] = chunk_df[det_col].astype(str).str.lower().str.strip().map(EA_PARAM_MAP)
        chunk_df = chunk_df.dropna(subset=["param"])

        if chunk_df.empty:
            continue

        chunk_df["value"] = pd.to_numeric(chunk_df[val_col], errors="coerce")
        # Basic unit handling
        if unit_col:
            units = chunk_df[unit_col].fillna("").str.lower()
            for p_target in ["As", "U"]:
                mg_mask = (chunk_df["param"] == p_target) & units.str.contains("mg", na=False)
                chunk_df.loc[mg_mask, "value"] *= 1000.0
            for p_target in [p for p in GENESIS_PARAMS if p not in ("As", "U", "pH", "Eh", "EC")]:
                ug_mask = (chunk_df["param"] == p_target) & units.str.contains("ug|µg", na=False)
                chunk_df.loc[ug_mask, "value"] *= 0.001

        long = pd.DataFrame({
            "site_id": chunk_df[site_col].astype(str) if site_col else "unknown",
            "date": pd.to_datetime(chunk_df[date_col], errors="coerce") if date_col else pd.NaT,
            "param": chunk_df["param"],
            "value": chunk_df["value"],
            "lat": pd.to_numeric(chunk_df[lat_col], errors="coerce") if lat_col else np.nan,
            "lon": pd.to_numeric(chunk_df[lon_col], errors="coerce") if lon_col else np.nan,
            "country": "GB",
        })
        all_long.append(long.dropna(subset=["date", "value"]))

    if not all_long:
        _log("[EA] No data extracted")
        return pd.DataFrame()

    combined = pd.concat(all_long, ignore_index=True)
    _log(f"[EA] Total measurements: {len(combined):,}")

    vectors = clean_and_pivot(combined, "EA_UK")
    _log(f"[EA] Vectors (≥{MIN_PARAMS} params): {len(vectors):,}")

    if not vectors.empty:
        vectors.to_parquet(vectors_cache, index=False)
    return vectors


# ############################################################
# MERGE + REPORT
# ############################################################

def merge_all(source_vectors: dict):
    _log("\n" + "=" * 60)
    _log("MERGING ALL SOURCES")
    _log("=" * 60)

    new_frames = []
    for name, vdf in source_vectors.items():
        if vdf is not None and not vdf.empty:
            _log(f"  {name}: {len(vdf):,} vectors")
            new_frames.append(vdf)

    if not new_frames:
        _log("  No new vectors to merge")
        return

    new_vectors = pd.concat(new_frames, ignore_index=True)
    _log(f"  New vectors total: {len(new_vectors):,}")

    # Temporal pairs
    pairs_frames = []
    for name, vdf in source_vectors.items():
        if vdf is not None and not vdf.empty:
            pairs = build_temporal_pairs(vdf, name)
            if not pairs.empty:
                pairs_frames.append(pairs)
                _log(f"  {name} temporal pairs: {len(pairs):,}")

    new_pairs = pd.concat(pairs_frames, ignore_index=True) if pairs_frames else pd.DataFrame()
    _log(f"  New temporal pairs total: {len(new_pairs):,}")

    # Save per-source outputs
    new_vectors.to_parquet(PROCESSED_DIR / "global_sources_vectors.parquet", index=False)
    if not new_pairs.empty:
        new_pairs.to_parquet(PROCESSED_DIR / "global_sources_temporal_pairs.parquet", index=False)

    # Merge with existing
    for label, existing_name, new_df in [
        ("vectors", "all_water_vectors_stage1.parquet", new_vectors),
        ("pairs", "final_temporal_pairs_enriched.parquet", new_pairs),
    ]:
        existing_path = PROCESSED_DIR / existing_name
        if existing_path.exists() and not new_df.empty:
            existing = pd.read_parquet(existing_path)
            _log(f"\n  Existing {label}: {len(existing):,}")

            for c in new_df.columns:
                if c not in existing.columns:
                    existing[c] = np.nan
            for c in existing.columns:
                if c not in new_df.columns:
                    new_df[c] = np.nan

            merged = pd.concat([existing, new_df], ignore_index=True)
            if "station_id" in merged.columns and "date_t0" in merged.columns:
                merged = merged.drop_duplicates(subset=["station_id", "date_t0", "date_t1"])

            out_name = existing_name.replace(".parquet", "_global.parquet")
            merged.to_parquet(PROCESSED_DIR / out_name, index=False)
            _log(f"  + new: {len(new_df):,}  = merged: {len(merged):,}")
            _log(f"  → {out_name}")

    # Report
    lines = ["GENESIS Paper 5 — Global Sources Curation Report",
             "=" * 50, ""]
    total_v = total_p = 0
    for name, vdf in sorted(source_vectors.items()):
        nv = len(vdf) if vdf is not None and not vdf.empty else 0
        total_v += nv
        lines.append(f"Source: {name}")
        lines.append(f"  Vectors: {nv:>10,}")
        if vdf is not None and not vdf.empty:
            pairs = build_temporal_pairs(vdf, name)
            np_ = len(pairs)
            total_p += np_
            lines.append(f"  Temporal pairs: {np_:>10,}")
            lines.append(f"  Per-param coverage:")
            for p in GENESIS_PARAMS:
                if p in vdf.columns:
                    n = int(vdf[p].notna().sum())
                    pct = 100 * n / nv if nv else 0
                    lines.append(f"    {p:>5s}: {n:>10,} ({pct:5.1f}%)")
        lines.append("")

    lines.append(f"TOTAL new vectors:        {total_v:>10,}")
    lines.append(f"TOTAL new temporal pairs: {total_p:>10,}")

    report = "\n".join(lines) + "\n"
    report_path = PROCESSED_DIR / "global_sources_curation_report.txt"
    report_path.write_text(report)
    _log(f"\nReport → {report_path}")
    print("\n" + report)


# ############################################################
# MAIN
# ############################################################

def main():
    global _LOGFILE

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["ades", "eea", "ea"],
                        help="Process only one source")
    parser.add_argument("--merge_only", action="store_true")
    args = parser.parse_args()

    log_path = LOG_DIR / "global_sources_download.log"
    _LOGFILE = open(log_path, "a")

    _log(f"\n{'=' * 60}")
    _log(f"GENESIS Global Sources — {time.strftime('%Y-%m-%d %H:%M')}")
    _log(f"{'=' * 60}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Source registry
    sources = {
        "ades": ("BRGM_ADES", download_ades),
        "eea": ("EEA_WISE", process_eea),
        "ea": ("EA_UK", process_ea),
    }

    if args.only:
        sources = {args.only: sources[args.only]}

    source_vectors = {}

    if args.merge_only:
        cache_map = {
            "BRGM_ADES": RAW_DIR / "brgm_ades" / "ades_vectors.parquet",
            "EEA_WISE": RAW_DIR / "eea_wise" / "eea_vectors.parquet",
            "EA_UK": RAW_DIR / "ea_water_quality" / "ea_vectors.parquet",
        }
        for name, path in cache_map.items():
            if path.exists():
                source_vectors[name] = pd.read_parquet(path)
                _log(f"  Loaded {name}: {len(source_vectors[name]):,}")
    else:
        # ADES runs via API (automated). EEA and EA process local files.
        # Run ADES in a thread, process EEA/EA on main thread.
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for key, (name, fn) in sources.items():
                futures[executor.submit(fn)] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    source_vectors[name] = result
                    n = len(result) if result is not None and not result.empty else 0
                    _log(f"\n✓ {name}: {n:,} vectors")
                except Exception as e:
                    _log(f"\n✗ {name} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    source_vectors[name] = pd.DataFrame()

    merge_all(source_vectors)
    _LOGFILE.close()
    print(f"\nLog → {log_path}")
    print("Done.")


if __name__ == "__main__":
    main()
