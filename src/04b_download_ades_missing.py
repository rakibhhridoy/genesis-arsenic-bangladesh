#!/usr/bin/env python3
"""
Download the 3 ADES param codes that were missing from the original download.

Codes:
  1330 = Potentiel REDOX (Eh), mV         ~238K records
  1361 = Uranium, mg(U)/L                 ~51K records
  1307 = Matière sèche (TDS), mg/L        ~14K records

Uses the same Hub'Eau API + checkpointing as 02_download_global_sources.py.
New chunks land in the same chunks/ dir — 04_build_genesis_db.py picks them up.

Usage:
  python src/04b_download_ades_missing.py
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PAPER5 = Path(__file__).resolve().parent.parent
RAW_DIR = PAPER5 / "data" / "raw"
LOG_DIR = PAPER5 / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_PATH = LOG_DIR / "ades_missing_download.log"

# The 3 missing codes and their GENESIS param names
MISSING_CODES = {
    "1330": "Eh",   # Potentiel REDOX, mV
    "1361": "U",    # Uranium, mg(U)/L → needs ×1000 → µg/L (done in pipeline)
    "1307": "TDS",  # Matière sèche à 105°C, mg/L
}

# French departments
FR_DEPTS = [f"{i:02d}" for i in range(1, 96)] + ["2A", "2B"] + \
           [f"{i:03d}" for i in range(971, 977)]


def _log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _hubeau_fetch_page(dept, code, page=1):
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


def _hubeau_fetch_all(dept, code):
    all_recs = []
    page = 1
    while page <= 50:
        recs, more = _hubeau_fetch_page(dept, code, page)
        if not recs:
            break
        all_recs.extend(recs)
        if not more:
            break
        page += 1
    return all_recs


def _hubeau_to_long(records, param_name):
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
    # Unit conversions matching the download script logic:
    # U (1361): source is mg(U)/L → GENESIS wants µg/L → ×1000
    # Eh (1330): source is mV → already standard
    # TDS (1307): source is mg/L → already standard
    if "symbole_unite" in df.columns:
        units = df["symbole_unite"].fillna("").str.lower()
        if param_name == "U":
            # Hub'Eau returns mg(U)/L, GENESIS wants µg/L
            mg_mask = units.str.contains("mg", na=False) & ~units.str.contains("µg|ug", na=False)
            if mg_mask.any():
                long.loc[mg_mask.values, "value"] *= 1000.0
        elif param_name not in ("pH", "Eh", "EC"):
            # Convert µg/L → mg/L for any unexpected µg unit
            ug_mask = units.str.contains("µg|ug", na=False)
            if ug_mask.any():
                long.loc[ug_mask.values, "value"] *= 0.001

    return long.dropna(subset=["date", "value", "site_id"])


def main():
    chunk_dir = RAW_DIR / "brgm_ades" / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    total = len(FR_DEPTS) * len(MISSING_CODES)
    done = 0
    skipped = 0
    total_rows = 0

    _log(f"Downloading 3 missing ADES param codes: {list(MISSING_CODES.values())}")
    _log(f"  {len(FR_DEPTS)} departments × {len(MISSING_CODES)} codes = {total} chunks")

    for dept in FR_DEPTS:
        for code, param_name in MISSING_CODES.items():
            done += 1
            chunk_file = chunk_dir / f"{dept}_{code}.parquet"
            empty_file = chunk_dir / f"{dept}_{code}.empty"

            if chunk_file.exists() or empty_file.exists():
                skipped += 1
                continue

            try:
                records = _hubeau_fetch_all(dept, code)
            except Exception as e:
                _log(f"  {done}/{total} {dept}/{param_name}: FAIL ({e})")
                continue

            if not records:
                empty_file.touch()
                continue

            long = _hubeau_to_long(records, param_name)
            if long.empty:
                empty_file.touch()
                continue

            long.to_parquet(chunk_file, index=False)
            total_rows += len(long)

            if done % 30 == 0 or len(long) > 5000:
                _log(f"  {done}/{total} {dept}/{param_name}: "
                     f"{len(long):,} rows (total: {total_rows:,})")

    _log(f"\nDone. {total_rows:,} new rows across {done - skipped} chunks.")
    _log(f"Skipped {skipped} (already cached).")
    _log("Re-run 04_build_genesis_db.py to pick up new data.")


if __name__ == "__main__":
    main()
