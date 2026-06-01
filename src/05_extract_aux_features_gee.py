#!/usr/bin/env python3
"""
GENESIS Paper 5 — Extract Auxiliary Features via Google Earth Engine
=====================================================================
Extracts ~20 scalar auxiliary features at all unique site locations using
GEE server-side computation. No large raster downloads needed.

Features extracted (19 total):
  TOPOGRAPHIC:
    1. elevation_m          — NASADEM 30m elevation
    2. slope_deg            — NASADEM-derived slope

  SOIL (OpenLandMap / SoilGrids, 0-30cm mean):
    3. sand_pct             — Sand weight fraction (%)
    4. clay_pct             — Clay weight fraction (%)
    5. soc_g_per_kg         — Soil organic carbon (g/kg, ×5 scale)
    6. soil_ph              — Soil pH in H₂O (÷10 scale)
    7. bulk_density_kg_m3   — Bulk density fine earth (kg/m³, ×10 scale)

  LAND SURFACE:
    8. landcover_esa        — ESA WorldCover 2021 class (categorical)
    9. population_density   — WorldPop 2020 (people/100m pixel)
   10. ndvi_mean            — MODIS 2000-2023 mean annual max NDVI
   11. ndvi_std             — MODIS inter-annual NDVI std

  CLIMATE NORMALS (1991-2020):
   12. precip_mm_yr         — TerraClimate mean annual precipitation
   13. temp_mean_C          — TerraClimate mean annual temperature (°C)
   14. aet_mm_yr            — TerraClimate mean actual evapotranspiration
   15. pet_mm_yr            — TerraClimate mean potential ET
   16. aridity_index        — precip / pet

  HYDROLOGICAL (GRACE 2002-2023):
   17. twsa_mean_cm         — Mean terrestrial water storage anomaly
   18. twsa_trend_cm_yr     — Linear trend in TWSA

  GROUNDWATER:
   19. wtd_mean_m           — Fan et al. (2013) water table depth (m)

Usage:
  python src/05_extract_aux_features_gee.py

Requires:
  - earthengine-api authenticated
  - GEE project: ee-arsenicbd
  - Input: data/processed/unique_sites.csv (lat, lon)
  - Output: data/processed/aux_features.csv
"""

import os
import sys
import time
import math
from pathlib import Path

import ee
import pandas as pd
import numpy as np

PAPER5 = Path(__file__).resolve().parent.parent
PROC_DIR = PAPER5 / "data" / "processed"
SITES_CSV = PROC_DIR / "unique_sites.csv"
OUTPUT_CSV = PROC_DIR / "aux_features.csv"
CHECKPOINT_DIR = PROC_DIR / "aux_chunks"
CHECKPOINT_DIR.mkdir(exist_ok=True)

PROJECT_ID = "ee-arsenicbd"

# GEE can handle ~5000 features per reduceRegions call reliably
BATCH_SIZE = 4000


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def init_gee():
    """Initialize and authenticate GEE."""
    try:
        ee.Initialize(project=PROJECT_ID)
        _log(f"GEE initialized (project: {PROJECT_ID})")
    except Exception:
        _log("Authenticating with GEE...")
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
        _log("GEE authenticated and initialized")


def make_feature_collection(df_chunk):
    """Convert a DataFrame chunk (lat, lon) to a GEE FeatureCollection."""
    features = []
    for _, row in df_chunk.iterrows():
        pt = ee.Geometry.Point([float(row["lon"]), float(row["lat"])])
        features.append(ee.Feature(pt, {
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
        }))
    return ee.FeatureCollection(features)


def build_extraction_image():
    """Build a single multi-band image with all auxiliary features.

    All computation is server-side — no data downloaded until we sample.
    All asset paths verified against GEE catalog 2026-04-13.
    """
    bands = []

    # --- 1-2. NASADEM elevation + slope ---
    nasadem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")
    bands.append(nasadem.rename("elevation_m").toFloat())
    bands.append(ee.Terrain.slope(nasadem).rename("slope_deg").toFloat())

    # --- 3-7. Soil properties (OpenLandMap, 0-30cm mean of b0, b10, b30) ---
    def olm_030_mean(asset_id, name):
        img = ee.Image(asset_id).select(["b0", "b10", "b30"])
        return img.reduce(ee.Reducer.mean()).rename(name).toFloat()

    bands.append(olm_030_mean(
        "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02", "sand_pct"))
    bands.append(olm_030_mean(
        "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02", "clay_pct"))
    bands.append(olm_030_mean(
        "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02", "soc_g_per_kg"))
    bands.append(olm_030_mean(
        "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02", "soil_ph"))
    bands.append(olm_030_mean(
        "OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02", "bulk_density_kg_m3"))

    # --- 8. ESA WorldCover 2021 ---
    bands.append(
        ee.Image("ESA/WorldCover/v200/2021")
        .select("Map").rename("landcover_esa").toFloat()
    )

    # --- 9. WorldPop 2020 ---
    bands.append(
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("year", 2020))
        .mosaic().rename("population_density").toFloat()
    )

    # --- 10-11. MODIS NDVI (2000-2023 annual max → mean + std) ---
    modis_ndvi = ee.ImageCollection("MODIS/061/MOD13A2") \
        .filterDate("2000-01-01", "2024-01-01") \
        .select("NDVI")

    def annual_max_ndvi(year):
        year = ee.Number(year)
        return modis_ndvi.filter(
            ee.Filter.calendarRange(year, year, "year")
        ).max().set("year", year)

    years_list = ee.List.sequence(2000, 2023)
    annual_ndvi = ee.ImageCollection(years_list.map(annual_max_ndvi))
    # MODIS NDVI scale factor = 0.0001
    bands.append(annual_ndvi.mean().multiply(0.0001).rename("ndvi_mean").toFloat())
    bands.append(annual_ndvi.reduce(ee.Reducer.stdDev()).multiply(0.0001).rename("ndvi_std").toFloat())

    # --- 12-16. TerraClimate normals (1991-2020) ---
    tc = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE") \
        .filterDate("1991-01-01", "2021-01-01")

    def annual_tc(year):
        year = ee.Number(year)
        yr_imgs = tc.filter(ee.Filter.calendarRange(year, year, "year"))
        return ee.Image([
            yr_imgs.select("pr").sum().rename("pr"),
            yr_imgs.select("aet").sum().rename("aet"),
            yr_imgs.select("pet").sum().rename("pet"),
            yr_imgs.select("tmmx").mean()
                .add(yr_imgs.select("tmmn").mean())
                .divide(2).rename("tmean"),
        ]).set("year", year)

    tc_years = ee.List.sequence(1991, 2020)
    tc_annual = ee.ImageCollection(tc_years.map(annual_tc))

    precip = tc_annual.select("pr").mean().rename("precip_mm_yr").toFloat()
    aet = tc_annual.select("aet").mean().multiply(0.1).rename("aet_mm_yr").toFloat()
    pet = tc_annual.select("pet").mean().multiply(0.1).rename("pet_mm_yr").toFloat()
    temp = tc_annual.select("tmean").mean().multiply(0.1).rename("temp_mean_C").toFloat()

    bands.extend([precip, temp, aet, pet])
    bands.append(precip.divide(pet.max(ee.Image.constant(1)))
                 .rename("aridity_index").toFloat())

    # --- 17-18. GRACE TWS anomaly (V04, 2002-2023) ---
    grace = ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI") \
        .filterDate("2002-04-01", "2024-01-01") \
        .select("lwe_thickness")

    bands.append(grace.mean().rename("twsa_mean_cm").toFloat())

    # Linear trend via linearFit
    def grace_with_time(img):
        t = img.date().difference(ee.Date("2002-04-01"), "year")
        return img.addBands(ee.Image.constant(t).float().rename("t"))

    grace_t = grace.map(grace_with_time)
    trend = grace_t.select(["t", "lwe_thickness"]).reduce(ee.Reducer.linearFit())
    bands.append(trend.select("scale").rename("twsa_trend_cm_yr").toFloat())

    # Note: Water table depth (Fan et al. 2013) not available on GEE.
    # Can be added later from GROW dataset spatial join if needed.

    return ee.Image(bands)


def process_results(sampled_fc):
    """Convert GEE FeatureCollection result to pandas DataFrame."""
    features = sampled_fc.getInfo()["features"]
    rows = []
    for f in features:
        props = f["properties"]
        rows.append(props)
    return pd.DataFrame(rows)


def main():
    init_gee()

    # Load sites
    sites = pd.read_csv(SITES_CSV)
    _log(f"Loaded {len(sites):,} unique sites")

    # Check for existing checkpoints
    existing = set()
    for f in CHECKPOINT_DIR.iterdir():
        if f.suffix == ".csv" and f.stem.startswith("batch_"):
            existing.add(int(f.stem.split("_")[1]))

    n_batches = math.ceil(len(sites) / BATCH_SIZE)
    _log(f"Processing {n_batches} batches of {BATCH_SIZE} sites each")
    if existing:
        _log(f"  {len(existing)} batches already cached, resuming...")

    # Build the extraction image (server-side, instant)
    _log("Building multi-band extraction image...")
    image = build_extraction_image()
    band_names = image.bandNames().getInfo()
    _log(f"  {len(band_names)} bands: {band_names}")

    total_done = len(existing)
    for i in range(n_batches):
        if i in existing:
            continue

        chunk = sites.iloc[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        _log(f"  Batch {i+1}/{n_batches} ({len(chunk)} sites)...")

        t0 = time.time()
        fc = make_feature_collection(chunk)

        sampled = image.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.first(),
            scale=250,
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result_df = process_results(sampled)
                result_df.to_csv(CHECKPOINT_DIR / f"batch_{i}.csv", index=False)
                total_done += 1
                elapsed = time.time() - t0
                _log(f"    {len(result_df)} rows, {elapsed:.0f}s "
                     f"({total_done}/{n_batches} done)")
                break
            except Exception as e:
                err = str(e).lower()
                if attempt < max_retries - 1 and ("deadline" in err or "quota" in err or "timeout" in err or "memory" in err):
                    wait = 30 * (attempt + 1)
                    _log(f"    Attempt {attempt+1} failed: {e}")
                    _log(f"    Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    _log(f"    FAILED (attempt {attempt+1}/{max_retries}): {e}")
                    break

        # Rate limit
        time.sleep(1)

    # Merge all checkpoint CSVs
    _log("\nMerging checkpoint files...")
    all_dfs = []
    for f in sorted(CHECKPOINT_DIR.glob("batch_*.csv"),
                    key=lambda p: int(p.stem.split("_")[1])):
        all_dfs.append(pd.read_csv(f))

    if not all_dfs:
        _log("ERROR: No results to merge")
        return

    merged = pd.concat(all_dfs, ignore_index=True)

    # Post-processing: apply OpenLandMap scale factors
    # SOC: stored as ×5 → divide by 5 to get g/kg
    if "soc_g_per_kg" in merged.columns:
        merged["soc_g_per_kg"] = merged["soc_g_per_kg"] / 5.0
    # pH: stored as ×10 → divide by 10
    if "soil_ph" in merged.columns:
        merged["soil_ph"] = merged["soil_ph"] / 10.0
    # Bulk density: stored as ×10 → divide by 10 to get kg/m³
    if "bulk_density_kg_m3" in merged.columns:
        merged["bulk_density_kg_m3"] = merged["bulk_density_kg_m3"] / 10.0

    # Keep only the columns we need (drop GEE internal columns)
    keep_cols = [
        "lat", "lon",
        "elevation_m", "slope_deg",
        "sand_pct", "clay_pct", "soc_g_per_kg", "soil_ph", "bulk_density_kg_m3",
        "landcover_esa", "population_density",
        "ndvi_mean", "ndvi_std",
        "precip_mm_yr", "temp_mean_C", "aet_mm_yr", "pet_mm_yr", "aridity_index",
        "twsa_mean_cm", "twsa_trend_cm_yr",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    merged = merged[keep_cols]

    # Drop exact duplicates
    merged = merged.drop_duplicates(subset=["lat", "lon"])

    merged.to_csv(OUTPUT_CSV, index=False)
    # Also save as parquet for efficient joining
    merged.to_parquet(OUTPUT_CSV.with_suffix(".parquet"), index=False)

    _log(f"\nDone. {len(merged):,} sites with aux features")
    _log(f"  → {OUTPUT_CSV}")
    _log(f"  → {OUTPUT_CSV.with_suffix('.parquet')}")
    _log(f"Columns ({len(keep_cols)}): {keep_cols}")

    # Coverage report
    _log("\nFeature coverage:")
    for col in keep_cols:
        if col in ("lat", "lon"):
            continue
        n = merged[col].notna().sum()
        _log(f"  {col:25s}: {n:>7,} / {len(merged):,} ({100*n/len(merged):.1f}%)")


if __name__ == "__main__":
    main()
