"""
GENESIS Paper 5 — Step 1: GEMStat Data Curation Pipeline
=========================================================
Reads GEMStat v3 CSVs, filters to groundwater stations, builds
geochemical vectors per station-date, applies quality filters,
and identifies temporal pairs with 5+ year gaps.

Output:
  - data/processed/gw_geochemical_vectors.parquet  (all GW samples as vectors)
  - data/processed/gw_temporal_pairs.parquet        (paired temporal observations)
  - data/processed/gw_station_metadata.parquet      (station info with lat/lon)
  - data/processed/curation_report.txt              (summary statistics)
"""

import os
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================

GEMSTAT_DIR = Path(__file__).parent.parent / "data" / "GEMStat" / "GFQA_v3"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameter mapping: GEMStat parameter codes → GENESIS standard names
# We map dissolved and total fractions; prefer dissolved where available
PARAM_MAP = {
    # Arsenic
    'As-Dis': 'As', 'As-Tot': 'As',
    # Iron
    'Fe-Dis': 'Fe', 'Fe-Tot': 'Fe', 'Fe2-Dis': 'Fe',
    # Manganese
    'Mn-Dis': 'Mn', 'Mn-Tot': 'Mn',
    # Phosphorus → PO4
    'PO4-P-Dis': 'PO4', 'PO4-P-Tot': 'PO4', 'P-Tot': 'PO4',
    'o-PO4-Dis': 'PO4', 'o-PO4-Tot': 'PO4',
    'DIP': 'PO4', 'DRP': 'PO4', 'TDP': 'PO4', 'TIP': 'PO4', 'TP': 'PO4',
    # Fluoride
    'F-Dis': 'F', 'F-Tot': 'F',
    # Uranium
    'U-Dis': 'U', 'U-Tot': 'U',
    # Nitrate
    'NO3-N-Dis': 'NO3', 'NO3-Dis': 'NO3', 'NO3-N': 'NO3',
    'NO3N': 'NO3', 'NOxN': 'NO3',
    # pH
    'pH': 'pH', 'pH-Lab': 'pH',
    # Eh / ORP
    'Eh': 'Eh', 'ORP': 'Eh',
    # Electrical conductivity
    'EC-Ins': 'EC', 'EC-Lab': 'EC', 'EC': 'EC',
    # TDS
    'TDS-Dis': 'TDS', 'TDS': 'TDS',
    # Calcium
    'Ca-Dis': 'Ca', 'Ca-Tot': 'Ca',
    # Magnesium
    'Mg-Dis': 'Mg', 'Mg-Tot': 'Mg',
    # Sodium
    'Na-Dis': 'Na', 'Na-Tot': 'Na',
    # Potassium
    'K-Dis': 'K', 'K-Tot': 'K',
    # Chloride
    'Cl-Dis': 'Cl', 'Cl-Tot': 'Cl',
    # Bicarbonate
    'HCO3-Dis': 'HCO3', 'Alk-HCO3': 'HCO3', 'HCO3': 'HCO3',
    'Alk-Tot': 'HCO3',  # Total alkalinity as proxy for HCO3
    # Sulphate
    'SO4-Dis': 'SO4', 'SO4-Tot': 'SO4',
    # Silica
    'Si-Dis': 'SiO2', 'SiO2-Dis': 'SiO2', 'Si-Tot': 'SiO2',
    'SiO2-Tot': 'SiO2', 'SiO2-Rea': 'SiO2', 'Si-Ext': 'SiO2',
    # DOC
    'DOC': 'DOC', 'DOC-Dis': 'DOC', 'DC': 'DOC', 'TOC': 'DOC',
}

# Priority: dissolved > total (lower number = higher priority)
FRACTION_PRIORITY = {
    'Dis': 1, 'Tot': 2, 'Lab': 3, 'Ins': 1, '': 4
}

# GENESIS target parameters
GENESIS_PARAMS = ['As', 'Fe', 'Mn', 'PO4', 'F', 'U', 'NO3',
                  'pH', 'Eh', 'EC', 'TDS',
                  'Ca', 'Mg', 'Na', 'K', 'Cl', 'HCO3', 'SO4',
                  'SiO2', 'DOC']

# GEMStat CSV files to process (parameter group → file name)
GEMSTAT_FILES = [
    'Arsenic.csv', 'Iron.csv', 'Manganese.csv', 'Phosphorus.csv',
    'Fluoride.csv', 'Uranium.csv', 'Sulfur.csv', 'Oxidized_Nitrogen.csv',
    'Other_Nitrogen.csv', 'pH.csv', 'Electrical_Conductance.csv',
    'Calcium.csv', 'Magnesium.csv', 'Sodium.csv', 'Potassium.csv',
    'Chloride.csv', 'Bicarbonate.csv', 'Sulphate.csv', 'Silicon.csv',
    'Carbon.csv', 'Dissolved_Gas.csv', 'Water.csv',
    'Alkalinity.csv',
]

# Quality filters
QUALITY_ACCEPT = {'Good', 'Fair', 'Unknown', ''}
MIN_PARAMS_STAGE1 = 3   # minimum params per sample for Stage 1 pretraining
MIN_PARAMS_STAGE2 = 8   # minimum params per sample for Stage 2 refinement
TEMPORAL_GAP_YEARS = 5   # minimum gap for temporal pairs


def load_groundwater_stations():
    """Load station metadata, filter to groundwater only."""
    print("Loading station metadata...")
    stations = {}
    meta_file = GEMSTAT_DIR / "GEMStat_station_metadata.csv"
    with open(meta_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'ground' in row.get('Water Type', '').lower():
                stn_id = row['GEMS Station Number']
                stations[stn_id] = {
                    'station_id': stn_id,
                    'country': row.get('Country Name', ''),
                    'water_type': row.get('Water Type', ''),
                    'lat': float(row['Latitude']) if row.get('Latitude') else None,
                    'lon': float(row['Longitude']) if row.get('Longitude') else None,
                    'elevation': float(row['Elevation']) if row.get('Elevation') else None,
                    'depth_casing': float(row['Depth of Impermeable Lining']) if row.get('Depth of Impermeable Lining') else None,
                    'production_zone': float(row['Production Zone']) if row.get('Production Zone') else None,
                    'basin': row.get('Main Basin', ''),
                    'station_name': row.get('Station Identifier', ''),
                }
    print(f"  Found {len(stations)} groundwater stations")
    return stations


def load_parameter_metadata():
    """Load parameter code descriptions."""
    params = {}
    meta_file = GEMSTAT_DIR / "GEMStat_parameter_metadata.csv"
    with open(meta_file, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get('Parameter Code', '')
            params[code] = {
                'name': row.get('Parameter Name', ''),
                'long_name': row.get('Parameter Long Name', ''),
                'group': row.get('Parameter Group', ''),
            }
    return params


def get_fraction_priority(param_code):
    """Get priority score for a parameter code (lower = preferred)."""
    for frac, pri in FRACTION_PRIORITY.items():
        if frac and frac in param_code:
            return pri
    return 4


def parse_value(value_str, flag_str):
    """Parse a measurement value, handling flags."""
    if not value_str or value_str.strip() == '':
        return None
    try:
        val = float(value_str.replace(',', '.'))
    except (ValueError, TypeError):
        return None
    # Handle below-detection-limit: use half the value
    if flag_str and '<' in flag_str:
        val = val / 2.0
    # Handle above-detection-limit: keep as-is
    return val


def standardize_unit(value, unit, param_name):
    """Convert to standard units: mg/L for concentrations, native for pH/Eh/EC."""
    if param_name in ('pH', 'Eh'):
        return value
    if param_name == 'EC':
        # Standardize to µS/cm
        if unit and 'ms' in unit.lower():
            return value * 1000  # mS/cm → µS/cm
        return value
    # Concentration parameters → mg/L
    if unit:
        unit_lower = unit.lower().strip()
        if 'ug' in unit_lower or 'µg' in unit_lower or 'ppb' in unit_lower:
            return value / 1000.0  # µg/L → mg/L
        elif 'ng' in unit_lower:
            return value / 1e6  # ng/L → mg/L
        elif 'meq' in unit_lower or 'mval' in unit_lower:
            return None  # skip meq/L, needs molar mass
    return value


def process_gemstat_files(gw_stations):
    """
    Read all GEMStat CSV files, extract groundwater measurements,
    map to GENESIS parameters.
    Returns dict: {(station_id, date): {param: (value, priority)}}
    """
    # Store: (station, date) → {genesis_param: (value, priority, unit)}
    records = defaultdict(dict)
    unmapped_codes = defaultdict(int)
    total_read = 0
    gw_matched = 0

    for fname in GEMSTAT_FILES:
        fpath = GEMSTAT_DIR / fname
        if not fpath.exists():
            print(f"  SKIP (not found): {fname}")
            continue

        print(f"  Processing {fname}...", end='', flush=True)
        file_count = 0
        file_gw = 0

        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_count += 1
                stn = row.get('GEMS Station Number', '')
                if stn not in gw_stations:
                    continue

                file_gw += 1
                param_code = row.get('Parameter Code', '')
                genesis_param = PARAM_MAP.get(param_code)

                if genesis_param is None:
                    unmapped_codes[param_code] += 1
                    continue

                # Quality check
                quality = row.get('Data Quality', '')
                if quality not in QUALITY_ACCEPT:
                    continue

                # Parse value
                value = parse_value(row.get('Value', ''), row.get('Value Flags', ''))
                if value is None:
                    continue

                # Standardize units
                unit = row.get('Unit', '')
                value = standardize_unit(value, unit, genesis_param)
                if value is None:
                    continue

                # Get date
                date = row.get('Sample Date', '')
                if not date or len(date) < 4:
                    continue

                key = (stn, date)
                priority = get_fraction_priority(param_code)

                # Keep highest priority (dissolved > total)
                if genesis_param in records[key]:
                    existing_priority = records[key][genesis_param][1]
                    if priority >= existing_priority:
                        continue

                records[key][genesis_param] = (value, priority)

        total_read += file_count
        gw_matched += file_gw
        print(f" {file_count:,} rows, {file_gw:,} GW")

    print(f"\n  Total rows read: {total_read:,}")
    print(f"  GW-matched rows: {gw_matched:,}")
    print(f"  Unique (station, date) samples: {len(records):,}")

    # Report unmapped codes
    if unmapped_codes:
        top_unmapped = sorted(unmapped_codes.items(), key=lambda x: -x[1])[:20]
        print(f"\n  Top unmapped parameter codes (GW only):")
        for code, count in top_unmapped:
            print(f"    {code}: {count:,}")

    return records


def build_geochemical_vectors(records, gw_stations):
    """Convert records dict to DataFrame of geochemical vectors."""
    print("\nBuilding geochemical vectors...")

    rows = []
    for (stn, date), params in records.items():
        row = {
            'station_id': stn,
            'sample_date': date,
            'year': int(date[:4]) if date[:4].isdigit() else None,
        }
        # Add station metadata
        meta = gw_stations.get(stn, {})
        row['lat'] = meta.get('lat')
        row['lon'] = meta.get('lon')
        row['country'] = meta.get('country', '')
        row['elevation'] = meta.get('elevation')
        row['depth_casing'] = meta.get('depth_casing')
        row['basin'] = meta.get('basin', '')

        # Add geochemical parameters
        n_params = 0
        for param in GENESIS_PARAMS:
            if param in params:
                row[param] = params[param][0]  # value only
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

    # Parameter coverage
    print(f"\n  Parameter coverage (% of samples with value):")
    for param in GENESIS_PARAMS:
        coverage = df[param].notna().sum() / len(df) * 100
        print(f"    {param:<6}: {coverage:5.1f}%  ({df[param].notna().sum():,} samples)")

    return df


def apply_quality_filters(df):
    """Apply quality filters: remove outliers, suspicious values."""
    print("\nApplying quality filters...")
    n_before = len(df)

    # Remove negative concentrations (except Eh which can be negative)
    conc_params = [p for p in GENESIS_PARAMS if p not in ('pH', 'Eh', 'EC', 'TDS')]
    for param in conc_params:
        mask = df[param].notna() & (df[param] < 0)
        n_neg = mask.sum()
        if n_neg > 0:
            df.loc[mask, param] = None
            print(f"  Removed {n_neg} negative values for {param}")

    # pH bounds: 0-14
    mask = df['pH'].notna() & ((df['pH'] < 0) | (df['pH'] > 14))
    df.loc[mask, 'pH'] = None
    print(f"  Removed {mask.sum()} out-of-range pH values")

    # Eh bounds: -1000 to +1500 mV
    mask = df['Eh'].notna() & ((df['Eh'] < -1000) | (df['Eh'] > 1500))
    df.loc[mask, 'Eh'] = None
    print(f"  Removed {mask.sum()} out-of-range Eh values")

    # Recount params after cleaning
    df['n_params'] = df[GENESIS_PARAMS].notna().sum(axis=1)

    # Stage 1 filter: at least MIN_PARAMS_STAGE1 parameters
    df_stage1 = df[df['n_params'] >= MIN_PARAMS_STAGE1].copy()
    print(f"\n  Stage 1 (≥{MIN_PARAMS_STAGE1} params): {len(df_stage1):,} samples "
          f"({len(df_stage1)/n_before*100:.1f}% of total)")

    # Stage 2 filter: at least MIN_PARAMS_STAGE2 parameters
    df_stage2 = df[df['n_params'] >= MIN_PARAMS_STAGE2].copy()
    print(f"  Stage 2 (≥{MIN_PARAMS_STAGE2} params): {len(df_stage2):,} samples "
          f"({len(df_stage2)/n_before*100:.1f}% of total)")

    return df, df_stage1, df_stage2


def find_temporal_pairs(df):
    """Find station-level temporal pairs with 5+ year gap."""
    print(f"\nFinding temporal pairs (≥{TEMPORAL_GAP_YEARS} year gap)...")

    pairs = []
    grouped = df.groupby('station_id')

    for stn_id, group in grouped:
        if len(group) < 2:
            continue

        # Get unique dates sorted
        dates = group.sort_values('sample_date')

        # Find pairs with sufficient gap
        date_list = dates['sample_date'].dropna().unique()
        if len(date_list) < 2:
            continue

        date_list = sorted(date_list)
        first = pd.Timestamp(date_list[0])
        last = pd.Timestamp(date_list[-1])

        gap_years = (last - first).days / 365.25
        if gap_years < TEMPORAL_GAP_YEARS:
            continue

        # Get earliest and latest sample for this station
        earliest = dates.iloc[0]
        latest = dates.iloc[-1]

        pair = {
            'station_id': stn_id,
            'date_t0': earliest['sample_date'],
            'date_t1': latest['sample_date'],
            'year_t0': earliest['year'],
            'year_t1': latest['year'],
            'gap_years': gap_years,
            'country': earliest['country'],
            'lat': earliest['lat'],
            'lon': earliest['lon'],
        }

        # Add delta values for each parameter
        for param in GENESIS_PARAMS:
            v0 = earliest[param]
            v1 = latest[param]
            pair[f'{param}_t0'] = v0
            pair[f'{param}_t1'] = v1
            if pd.notna(v0) and pd.notna(v1):
                pair[f'delta_{param}'] = v1 - v0
            else:
                pair[f'delta_{param}'] = None

        # Count how many parameters have both t0 and t1
        n_paired = sum(1 for p in GENESIS_PARAMS
                       if pd.notna(pair.get(f'{p}_t0')) and pd.notna(pair.get(f'{p}_t1')))
        pair['n_paired_params'] = n_paired

        if n_paired >= 3:  # At least 3 parameters with temporal pair
            pairs.append(pair)

    df_pairs = pd.DataFrame(pairs)

    if len(df_pairs) > 0:
        print(f"  Temporal pairs found: {len(df_pairs):,}")
        print(f"  Unique stations: {df_pairs['station_id'].nunique():,}")
        print(f"  Countries: {df_pairs['country'].nunique()}")
        print(f"  Gap range: {df_pairs['gap_years'].min():.1f} - {df_pairs['gap_years'].max():.1f} years")
        print(f"  Mean paired params per pair: {df_pairs['n_paired_params'].mean():.1f}")

        print(f"\n  Temporal pairs by country:")
        for country, count in df_pairs['country'].value_counts().head(15).items():
            print(f"    {country}: {count}")

        print(f"\n  Parameter coverage in temporal pairs:")
        for param in GENESIS_PARAMS:
            n = df_pairs[f'delta_{param}'].notna().sum()
            print(f"    {param:<6}: {n:,} pairs ({n/len(df_pairs)*100:.1f}%)")

    return df_pairs


def write_curation_report(df_all, df_s1, df_s2, df_pairs, gw_stations):
    """Write summary report."""
    report_path = OUTPUT_DIR / "curation_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GENESIS — GEMStat Data Curation Report\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Source: GEMStat GFQA v3 (Zenodo record 18459694)\n")
        f.write(f"Groundwater stations: {len(gw_stations):,}\n\n")

        f.write("--- ALL GROUNDWATER SAMPLES ---\n")
        f.write(f"Total samples: {len(df_all):,}\n")
        f.write(f"Unique stations: {df_all['station_id'].nunique():,}\n")
        f.write(f"Countries: {df_all['country'].nunique()}\n")
        f.write(f"Year range: {df_all['year'].min():.0f}-{df_all['year'].max():.0f}\n\n")

        f.write(f"--- STAGE 1 PRETRAINING (≥{MIN_PARAMS_STAGE1} params) ---\n")
        f.write(f"Samples: {len(df_s1):,}\n")
        f.write(f"Stations: {df_s1['station_id'].nunique():,}\n\n")

        f.write(f"--- STAGE 2 REFINEMENT (≥{MIN_PARAMS_STAGE2} params) ---\n")
        f.write(f"Samples: {len(df_s2):,}\n")
        f.write(f"Stations: {df_s2['station_id'].nunique():,}\n\n")

        f.write(f"--- TEMPORAL PAIRS (≥{TEMPORAL_GAP_YEARS} year gap, ≥3 shared params) ---\n")
        if len(df_pairs) > 0:
            f.write(f"Pairs: {len(df_pairs):,}\n")
            f.write(f"Stations: {df_pairs['station_id'].nunique():,}\n")
            f.write(f"Countries: {df_pairs['country'].nunique()}\n")
        else:
            f.write("No temporal pairs found.\n")

        f.write("\n--- PARAMETER COVERAGE (Stage 1) ---\n")
        for param in GENESIS_PARAMS:
            cov = df_s1[param].notna().sum() / len(df_s1) * 100 if len(df_s1) > 0 else 0
            f.write(f"  {param:<6}: {cov:5.1f}%\n")

    print(f"\nReport saved to: {report_path}")


def main():
    print("=" * 70)
    print("GENESIS — GEMStat Data Curation Pipeline")
    print("=" * 70)

    # 1. Load groundwater stations
    gw_stations = load_groundwater_stations()

    # 2. Load parameter metadata
    param_meta = load_parameter_metadata()
    print(f"  Parameter codes in metadata: {len(param_meta)}")

    # Check which of our mapped codes exist in metadata
    mapped_in_meta = sum(1 for code in PARAM_MAP if code in param_meta)
    print(f"  Mapped codes found in metadata: {mapped_in_meta}/{len(PARAM_MAP)}")

    # 3. Process all GEMStat files
    print("\nProcessing GEMStat files...")
    records = process_gemstat_files(gw_stations)

    # 4. Build geochemical vectors
    df_all = build_geochemical_vectors(records, gw_stations)

    # 5. Quality filters
    df_all, df_stage1, df_stage2 = apply_quality_filters(df_all)

    # 6. Find temporal pairs
    df_pairs = find_temporal_pairs(df_stage1)

    # 7. Save outputs
    print("\nSaving outputs...")

    # Station metadata
    stn_df = pd.DataFrame(list(gw_stations.values()))
    stn_df.to_parquet(OUTPUT_DIR / "gw_station_metadata.parquet", index=False)
    print(f"  Station metadata: {len(stn_df):,} stations")

    # All vectors (Stage 1)
    df_stage1.to_parquet(OUTPUT_DIR / "gw_geochemical_vectors_stage1.parquet", index=False)
    print(f"  Stage 1 vectors: {len(df_stage1):,} samples")

    # Refined vectors (Stage 2)
    df_stage2.to_parquet(OUTPUT_DIR / "gw_geochemical_vectors_stage2.parquet", index=False)
    print(f"  Stage 2 vectors: {len(df_stage2):,} samples")

    # Temporal pairs
    if len(df_pairs) > 0:
        df_pairs.to_parquet(OUTPUT_DIR / "gw_temporal_pairs.parquet", index=False)
        print(f"  Temporal pairs: {len(df_pairs):,} pairs")

    # Curation report
    write_curation_report(df_all, df_stage1, df_stage2, df_pairs, gw_stations)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
