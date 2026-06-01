"""
GENESIS Paper 5 — Step 1b: Curate ALL Water Types from GEMStat
================================================================
Processes ALL 50M+ records (river, lake, groundwater, reservoir, wetland)
for foundation model pretraining. Then separates GW and surface water
for domain-specific fine-tuning.

Output:
  - data/processed/all_water_vectors_stage1.parquet   (ALL water, ≥3 params)
  - data/processed/all_water_vectors_stage2.parquet   (ALL water, ≥8 params)
  - data/processed/all_water_temporal_pairs.parquet   (ALL water temporal pairs)
  - data/processed/surface_water_temporal_pairs.parquet (rivers+lakes only)
  - data/processed/all_water_curation_report.txt
"""

import os
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

GEMSTAT_DIR = Path(__file__).parent.parent / "data" / "GEMStat" / "GFQA_v3"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Same parameter mapping as 01_curate_gemstat.py
PARAM_MAP = {
    'As-Dis': 'As', 'As-Tot': 'As',
    'Fe-Dis': 'Fe', 'Fe-Tot': 'Fe', 'Fe2-Dis': 'Fe',
    'Mn-Dis': 'Mn', 'Mn-Tot': 'Mn',
    'PO4-P-Dis': 'PO4', 'PO4-P-Tot': 'PO4', 'P-Tot': 'PO4',
    'o-PO4-Dis': 'PO4', 'o-PO4-Tot': 'PO4',
    'DIP': 'PO4', 'DRP': 'PO4', 'TDP': 'PO4', 'TIP': 'PO4', 'TP': 'PO4',
    'F-Dis': 'F', 'F-Tot': 'F',
    'U-Dis': 'U', 'U-Tot': 'U',
    'NO3-N-Dis': 'NO3', 'NO3-Dis': 'NO3', 'NO3-N': 'NO3',
    'NO3N': 'NO3', 'NOxN': 'NO3',
    'pH': 'pH', 'pH-Lab': 'pH',
    'Eh': 'Eh', 'ORP': 'Eh',
    'EC-Ins': 'EC', 'EC-Lab': 'EC', 'EC': 'EC',
    'TDS-Dis': 'TDS', 'TDS': 'TDS',
    'Ca-Dis': 'Ca', 'Ca-Tot': 'Ca',
    'Mg-Dis': 'Mg', 'Mg-Tot': 'Mg',
    'Na-Dis': 'Na', 'Na-Tot': 'Na',
    'K-Dis': 'K', 'K-Tot': 'K',
    'Cl-Dis': 'Cl', 'Cl-Tot': 'Cl',
    'HCO3-Dis': 'HCO3', 'Alk-HCO3': 'HCO3', 'HCO3': 'HCO3',
    'Alk-Tot': 'HCO3',
    'SO4-Dis': 'SO4', 'SO4-Tot': 'SO4',
    'Si-Dis': 'SiO2', 'SiO2-Dis': 'SiO2', 'Si-Tot': 'SiO2',
    'SiO2-Tot': 'SiO2', 'SiO2-Rea': 'SiO2', 'Si-Ext': 'SiO2',
    'DOC': 'DOC', 'DOC-Dis': 'DOC', 'DC': 'DOC', 'TOC': 'DOC',
}

FRACTION_PRIORITY = {'Dis': 1, 'Tot': 2, 'Lab': 3, 'Ins': 1, '': 4}

GENESIS_PARAMS = ['As', 'Fe', 'Mn', 'PO4', 'F', 'U', 'NO3',
                  'pH', 'Eh', 'EC', 'TDS',
                  'Ca', 'Mg', 'Na', 'K', 'Cl', 'HCO3', 'SO4',
                  'SiO2', 'DOC']

GEMSTAT_FILES = [
    'Arsenic.csv', 'Iron.csv', 'Manganese.csv', 'Phosphorus.csv',
    'Fluoride.csv', 'Uranium.csv', 'Sulfur.csv', 'Oxidized_Nitrogen.csv',
    'Other_Nitrogen.csv', 'pH.csv', 'Electrical_Conductance.csv',
    'Calcium.csv', 'Magnesium.csv', 'Sodium.csv', 'Potassium.csv',
    'Chloride.csv', 'Bicarbonate.csv', 'Silicon.csv',
    'Carbon.csv', 'Dissolved_Gas.csv', 'Water.csv', 'Alkalinity.csv',
]

QUALITY_ACCEPT = {'Good', 'Fair', 'Unknown', ''}
MIN_PARAMS_STAGE1 = 3
MIN_PARAMS_STAGE2 = 8
TEMPORAL_GAP_YEARS = 5


def load_all_stations():
    """Load ALL station metadata (all water types)."""
    print("Loading ALL station metadata...")
    stations = {}
    water_type_counts = defaultdict(int)
    meta_file = GEMSTAT_DIR / "GEMStat_station_metadata.csv"
    with open(meta_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stn_id = row['GEMS Station Number']
            wtype = row.get('Water Type', 'Unknown')
            water_type_counts[wtype] += 1
            stations[stn_id] = {
                'station_id': stn_id,
                'country': row.get('Country Name', ''),
                'water_type': wtype,
                'lat': float(row['Latitude']) if row.get('Latitude') else None,
                'lon': float(row['Longitude']) if row.get('Longitude') else None,
                'elevation': float(row['Elevation']) if row.get('Elevation') else None,
                'depth_casing': float(row['Depth of Impermeable Lining']) if row.get('Depth of Impermeable Lining') else None,
                'basin': row.get('Main Basin', ''),
                'station_name': row.get('Station Identifier', ''),
            }
    print(f"  Total stations: {len(stations)}")
    for wt, count in sorted(water_type_counts.items(), key=lambda x: -x[1]):
        print(f"    {wt}: {count}")
    return stations


def get_fraction_priority(param_code):
    for frac, pri in FRACTION_PRIORITY.items():
        if frac and frac in param_code:
            return pri
    return 4


def parse_value(value_str, flag_str):
    if not value_str or value_str.strip() == '':
        return None
    try:
        val = float(value_str.replace(',', '.'))
    except (ValueError, TypeError):
        return None
    if flag_str and '<' in flag_str:
        val = val / 2.0
    return val


def standardize_unit(value, unit, param_name):
    if param_name in ('pH', 'Eh'):
        return value
    if param_name == 'EC':
        if unit and 'ms' in unit.lower():
            return value * 1000
        return value
    if unit:
        unit_lower = unit.lower().strip()
        if 'ug' in unit_lower or 'µg' in unit_lower or 'ppb' in unit_lower:
            return value / 1000.0
        elif 'ng' in unit_lower:
            return value / 1e6
        elif 'meq' in unit_lower or 'mval' in unit_lower:
            return None
    return value


def process_all_gemstat_files(all_stations):
    """Read ALL GEMStat CSVs — no groundwater filter."""
    records = defaultdict(dict)
    total_read = 0
    matched = 0

    for fname in GEMSTAT_FILES:
        fpath = GEMSTAT_DIR / fname
        if not fpath.exists():
            print(f"  SKIP: {fname}")
            continue

        print(f"  Processing {fname}...", end='', flush=True)
        file_count = 0
        file_matched = 0

        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_count += 1
                stn = row.get('GEMS Station Number', '')
                if stn not in all_stations:
                    continue

                file_matched += 1
                param_code = row.get('Parameter Code', '')
                genesis_param = PARAM_MAP.get(param_code)
                if genesis_param is None:
                    continue

                quality = row.get('Data Quality', '')
                if quality not in QUALITY_ACCEPT:
                    continue

                value = parse_value(row.get('Value', ''), row.get('Value Flags', ''))
                if value is None:
                    continue

                unit = row.get('Unit', '')
                value = standardize_unit(value, unit, genesis_param)
                if value is None:
                    continue

                date = row.get('Sample Date', '')
                if not date or len(date) < 4:
                    continue

                key = (stn, date)
                priority = get_fraction_priority(param_code)

                if genesis_param in records[key]:
                    if priority >= records[key][genesis_param][1]:
                        continue

                records[key][genesis_param] = (value, priority)

        total_read += file_count
        matched += file_matched
        print(f" {file_count:,} total, {file_matched:,} matched")

    print(f"\n  Total rows read: {total_read:,}")
    print(f"  Matched rows: {matched:,}")
    print(f"  Unique (station, date) samples: {len(records):,}")
    return records


def build_vectors(records, all_stations):
    """Build geochemical vectors for ALL water types."""
    print("\nBuilding geochemical vectors (all water types)...")

    rows = []
    for (stn, date), params in records.items():
        meta = all_stations.get(stn, {})
        row = {
            'station_id': stn,
            'sample_date': date,
            'year': int(date[:4]) if date[:4].isdigit() else None,
            'lat': meta.get('lat'),
            'lon': meta.get('lon'),
            'country': meta.get('country', ''),
            'water_type': meta.get('water_type', ''),
            'elevation': meta.get('elevation'),
            'depth_casing': meta.get('depth_casing'),
            'basin': meta.get('basin', ''),
        }

        n_params = 0
        for param in GENESIS_PARAMS:
            if param in params:
                row[param] = params[param][0]
                n_params += 1
            else:
                row[param] = None
        row['n_params'] = n_params
        rows.append(row)

    df = pd.DataFrame(rows)
    df['sample_date'] = pd.to_datetime(df['sample_date'], errors='coerce')
    df = df.dropna(subset=['year', 'lat', 'lon'])

    print(f"  Total samples: {len(df):,}")
    print(f"  Unique stations: {df['station_id'].nunique():,}")
    print(f"  Year range: {df['year'].min():.0f} - {df['year'].max():.0f}")
    print(f"  Countries: {df['country'].nunique()}")

    print(f"\n  By water type:")
    for wt, count in df['water_type'].value_counts().items():
        print(f"    {wt}: {count:,}")

    print(f"\n  Parameter coverage:")
    for param in GENESIS_PARAMS:
        n = df[param].notna().sum()
        print(f"    {param:<6}: {n:>8,} ({n/len(df)*100:5.1f}%)")

    return df


def apply_quality_filters(df):
    """Apply quality filters."""
    print("\nApplying quality filters...")
    n_before = len(df)

    conc_params = [p for p in GENESIS_PARAMS if p not in ('pH', 'Eh', 'EC', 'TDS')]
    for param in conc_params:
        mask = df[param].notna() & (df[param] < 0)
        if mask.sum() > 0:
            df.loc[mask, param] = None

    mask = df['pH'].notna() & ((df['pH'] < 0) | (df['pH'] > 14))
    df.loc[mask, 'pH'] = None

    mask = df['Eh'].notna() & ((df['Eh'] < -1000) | (df['Eh'] > 1500))
    df.loc[mask, 'Eh'] = None

    df['n_params'] = df[GENESIS_PARAMS].notna().sum(axis=1)

    df_s1 = df[df['n_params'] >= MIN_PARAMS_STAGE1].copy()
    df_s2 = df[df['n_params'] >= MIN_PARAMS_STAGE2].copy()

    print(f"  Stage 1 (≥{MIN_PARAMS_STAGE1} params): {len(df_s1):,} ({len(df_s1)/n_before*100:.1f}%)")
    print(f"  Stage 2 (≥{MIN_PARAMS_STAGE2} params): {len(df_s2):,} ({len(df_s2)/n_before*100:.1f}%)")

    print(f"\n  Stage 1 by water type:")
    for wt, count in df_s1['water_type'].value_counts().items():
        print(f"    {wt}: {count:,}")

    return df, df_s1, df_s2


def find_temporal_pairs(df, water_type_filter=None, label="all"):
    """Find temporal pairs, optionally filtered by water type."""
    if water_type_filter:
        df_filtered = df[df['water_type'].isin(water_type_filter)]
        print(f"\nFinding temporal pairs [{label}] (≥{TEMPORAL_GAP_YEARS}yr gap, "
              f"water types: {water_type_filter})...")
    else:
        df_filtered = df
        print(f"\nFinding temporal pairs [{label}] (≥{TEMPORAL_GAP_YEARS}yr gap)...")

    pairs = []
    grouped = df_filtered.groupby('station_id')

    for stn_id, group in grouped:
        if len(group) < 2:
            continue

        dates = group.sort_values('sample_date')
        date_list = dates['sample_date'].dropna().unique()
        if len(date_list) < 2:
            continue

        date_list = sorted(date_list)
        first_ts = pd.Timestamp(date_list[0])
        last_ts = pd.Timestamp(date_list[-1])
        gap_years = (last_ts - first_ts).days / 365.25
        if gap_years < TEMPORAL_GAP_YEARS:
            continue

        earliest = dates.iloc[0]
        latest = dates.iloc[-1]

        pair = {
            'station_id': stn_id,
            'date_t0': str(earliest['sample_date']),
            'date_t1': str(latest['sample_date']),
            'year_t0': earliest['year'],
            'year_t1': latest['year'],
            'gap_years': gap_years,
            'country': earliest['country'],
            'water_type': earliest['water_type'],
            'lat': earliest['lat'],
            'lon': earliest['lon'],
        }

        n_paired = 0
        for param in GENESIS_PARAMS:
            v0 = earliest[param]
            v1 = latest[param]
            pair[f'{param}_t0'] = v0 if pd.notna(v0) else None
            pair[f'{param}_t1'] = v1 if pd.notna(v1) else None
            if pd.notna(v0) and pd.notna(v1):
                pair[f'delta_{param}'] = v1 - v0
                n_paired += 1
            else:
                pair[f'delta_{param}'] = None
        pair['n_paired_params'] = n_paired

        if n_paired >= 3:
            pairs.append(pair)

    df_pairs = pd.DataFrame(pairs)

    if len(df_pairs) > 0:
        print(f"  Pairs found: {len(df_pairs):,}")
        print(f"  Stations: {df_pairs['station_id'].nunique():,}")
        print(f"  Countries: {df_pairs['country'].nunique()}")
        print(f"  Mean paired params: {df_pairs['n_paired_params'].mean():.1f}")
        if 'water_type' in df_pairs.columns:
            print(f"  By water type:")
            for wt, count in df_pairs['water_type'].value_counts().items():
                print(f"    {wt}: {count:,}")
    else:
        print("  No pairs found.")

    return df_pairs


def main():
    print("=" * 70)
    print("GENESIS — ALL Water Types Curation Pipeline")
    print("=" * 70)

    # 1. Load ALL stations
    all_stations = load_all_stations()

    # 2. Process all files (no GW filter)
    print("\nProcessing ALL GEMStat files (no water type filter)...")
    records = process_all_gemstat_files(all_stations)

    # 3. Build vectors
    df_all = build_vectors(records, all_stations)

    # 4. Quality filters
    df_all, df_s1, df_s2 = apply_quality_filters(df_all)

    # 5. Find temporal pairs — ALL water types
    pairs_all = find_temporal_pairs(df_s1, water_type_filter=None, label="all")

    # 6. Find temporal pairs — surface water only (rivers + lakes)
    pairs_surface = find_temporal_pairs(
        df_s1,
        water_type_filter=['River station', 'Lake station', 'Reservoir station'],
        label="surface"
    )

    # 7. Find temporal pairs — groundwater only
    pairs_gw = find_temporal_pairs(
        df_s1,
        water_type_filter=['Groundwater station'],
        label="groundwater"
    )

    # 8. Save outputs
    print("\nSaving outputs...")

    df_s1.to_parquet(OUTPUT_DIR / "all_water_vectors_stage1.parquet", index=False)
    print(f"  Stage 1 (all water): {len(df_s1):,}")

    df_s2.to_parquet(OUTPUT_DIR / "all_water_vectors_stage2.parquet", index=False)
    print(f"  Stage 2 (all water): {len(df_s2):,}")

    if len(pairs_all) > 0:
        pairs_all.to_parquet(OUTPUT_DIR / "all_water_temporal_pairs.parquet", index=False)
        print(f"  Temporal pairs (all): {len(pairs_all):,}")

    if len(pairs_surface) > 0:
        pairs_surface.to_parquet(OUTPUT_DIR / "surface_water_temporal_pairs.parquet", index=False)
        print(f"  Temporal pairs (surface): {len(pairs_surface):,}")

    if len(pairs_gw) > 0:
        # Overwrite the GW-only pairs with this consistent extraction
        pairs_gw.to_parquet(OUTPUT_DIR / "gw_temporal_pairs_v2.parquet", index=False)
        print(f"  Temporal pairs (GW): {len(pairs_gw):,}")

    # 9. Report
    report_path = OUTPUT_DIR / "all_water_curation_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GENESIS — All Water Types Curation Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total samples (all water): {len(df_all):,}\n")
        f.write(f"Stage 1 (≥3 params): {len(df_s1):,}\n")
        f.write(f"Stage 2 (≥8 params): {len(df_s2):,}\n\n")
        f.write("By water type (Stage 1):\n")
        for wt, count in df_s1['water_type'].value_counts().items():
            f.write(f"  {wt}: {count:,}\n")
        f.write(f"\nTemporal pairs (all): {len(pairs_all):,}\n")
        f.write(f"Temporal pairs (surface): {len(pairs_surface):,}\n")
        f.write(f"Temporal pairs (GW): {len(pairs_gw):,}\n")
    print(f"\n  Report: {report_path}")

    # 10. Summary
    print("\n" + "=" * 70)
    print("GENESIS PRETRAINING DATA — ALL WATER TYPES")
    print("=" * 70)
    print(f"\nPretraining Stage 1:  {len(df_s1):,} samples")
    print(f"Pretraining Stage 2:  {len(df_s2):,} samples")
    print(f"Temporal pairs (all): {len(pairs_all):,}")
    print(f"  - Groundwater:      {len(pairs_gw):,}")
    print(f"  - Surface water:    {len(pairs_surface):,}")
    print(f"Countries:            {df_s1['country'].nunique()}")
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
