#!/usr/bin/env python3
"""
GENESIS Paper 5 — Download Auxiliary Data Sources
===================================================
Downloads GROW (Zenodo), USGS NGS, and EEA WISE-6 (large ZIP).

Usage:
  python src/03_download_auxiliary_sources.py                # all sources
  python src/03_download_auxiliary_sources.py --only grow    # single source
  python src/03_download_auxiliary_sources.py --only ngs
  python src/03_download_auxiliary_sources.py --only eea
"""

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

PAPER5 = Path(__file__).resolve().parent.parent
RAW_DIR = PAPER5 / "data" / "raw"
LOG_DIR = PAPER5 / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_PATH = LOG_DIR / "auxiliary_downloads.log"


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def download_file(url, dest_path, chunk_size=8 * 1024 * 1024):
    """Stream-download with resume support and progress."""
    dest_path = Path(dest_path)
    partial = dest_path.with_suffix(dest_path.suffix + ".partial")

    # Resume from partial download
    start_byte = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={start_byte}-"} if start_byte > 0 else {}

    if start_byte > 0:
        _log(f"  Resuming from {start_byte / 1e9:.2f} GB")

    resp = requests.get(url, headers=headers, stream=True, timeout=300)

    # If server doesn't support range, restart
    if resp.status_code == 200 and start_byte > 0:
        _log("  Server doesn't support resume, restarting")
        start_byte = 0
        partial.unlink(missing_ok=True)
    elif resp.status_code not in (200, 206):
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.reason}")

    total = resp.headers.get("content-length")
    total_bytes = int(total) + start_byte if total else None

    mode = "ab" if start_byte > 0 else "wb"
    downloaded = start_byte
    last_report = time.time()

    with open(partial, mode) as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if time.time() - last_report > 30:
                if total_bytes:
                    pct = downloaded / total_bytes * 100
                    _log(f"  {downloaded / 1e9:.2f} / {total_bytes / 1e9:.2f} GB ({pct:.1f}%)")
                else:
                    _log(f"  {downloaded / 1e9:.2f} GB downloaded")
                last_report = time.time()

    partial.rename(dest_path)
    _log(f"  Saved: {dest_path} ({downloaded / 1e9:.2f} GB)")
    return dest_path


# ============================================================
# GROW Dataset (Zenodo 15149480)
# ============================================================

def download_grow():
    """Download GROW dataset from Zenodo — ~3.6 GB ZIP with auxiliary attributes."""
    out_dir = RAW_DIR / "grow_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / "grow_dataset.zip"
    if zip_path.exists() and zip_path.stat().st_size > 1e9:
        _log("GROW ZIP already downloaded, skipping")
    else:
        _log("Downloading GROW dataset from Zenodo...")
        # Zenodo direct download URL for record 15149480
        url = "https://zenodo.org/records/15149480/files/data.zip?download=1"
        download_file(url, zip_path)

    # Extract if not already done
    extract_marker = out_dir / ".extracted"
    if not extract_marker.exists():
        _log("Extracting GROW ZIP...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        extract_marker.touch()
        _log(f"  Extracted to {out_dir}")
    else:
        _log("GROW already extracted")

    # List what we got
    files = list(out_dir.rglob("*.csv")) + list(out_dir.rglob("*.parquet")) + list(out_dir.rglob("*.shp"))
    _log(f"  GROW files: {len(files)}")
    for f in files[:10]:
        _log(f"    {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    return out_dir


# ============================================================
# USGS NGS (National Geochemical Survey)
# ============================================================

def download_ngs():
    """Download USGS NGS shapefile — ~6.4 MB."""
    out_dir = RAW_DIR / "usgs_ngs"
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / "ngs_data.zip"
    if zip_path.exists() and zip_path.stat().st_size > 1e6:
        _log("USGS NGS already downloaded, skipping")
    else:
        _log("Downloading USGS NGS shapefile...")
        # Full 287-attribute CSV (77k records, 13 MB)
        url = "https://mrdata.usgs.gov/geochem/geochem.zip"
        download_file(url, zip_path)

    extract_marker = out_dir / ".extracted"
    if not extract_marker.exists():
        _log("Extracting NGS ZIP...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        extract_marker.touch()
        _log(f"  Extracted to {out_dir}")
    else:
        _log("NGS already extracted")

    files = list(out_dir.rglob("*.shp")) + list(out_dir.rglob("*.dbf"))
    _log(f"  NGS files: {len(files)}")
    for f in files[:10]:
        _log(f"    {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    return out_dir


# ============================================================
# EEA WISE-6 Waterbase (EU-27) — 50 GB ZIP
# ============================================================

def download_eea():
    """Download EEA WISE-6 bulk ZIP — ~50 GB. Streams with resume."""
    out_dir = RAW_DIR / "eea_wise"
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / "eea_wise6.zip"
    if zip_path.exists() and zip_path.stat().st_size > 40e9:
        _log("EEA WISE-6 ZIP already downloaded, skipping")
    else:
        _log("Downloading EEA WISE-6 (EU-27) — ~50 GB, this will take hours...")
        url = "https://sdi.eea.europa.eu/datashare/s/3JiTia3qePyGxyA/download"
        download_file(url, zip_path)

    _log(f"EEA ZIP size: {zip_path.stat().st_size / 1e9:.2f} GB")
    _log("EEA ZIP downloaded. Run 02_download_global_sources.py --only eea to process.")
    return out_dir


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["grow", "ngs", "eea"],
                        help="Download only one source")
    args = parser.parse_args()

    _log(f"\n{'=' * 60}")
    _log(f"GENESIS Auxiliary Downloads — {time.strftime('%Y-%m-%d %H:%M')}")
    _log(f"{'=' * 60}")

    tasks = {
        "grow": ("GROW (Zenodo)", download_grow),
        "ngs": ("USGS NGS", download_ngs),
        "eea": ("EEA WISE-6", download_eea),
    }

    if args.only:
        tasks = {args.only: tasks[args.only]}

    for key, (name, fn) in tasks.items():
        _log(f"\n--- {name} ---")
        try:
            fn()
            _log(f"  {name} done.")
        except Exception as e:
            _log(f"  {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    _log("\nAll downloads complete.")


if __name__ == "__main__":
    main()
