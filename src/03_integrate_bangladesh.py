"""
GENESIS Paper 5 — Step 3: Integrate Bangladesh Temporal Data
==============================================================
Reads the 705 paired well-depth units (2012-2013 vs 2020-2021),
formats as GENESIS temporal pairs, and produces the fine-tuning dataset.

Also reads the full 2020-2021 campaign (1,807 samples from Paper 1)
for additional pretraining samples.

Output:
  - data/processed/bangladesh_temporal_pairs.parquet  (705 fine-tuning pairs)
  - data/processed/bangladesh_all_samples.parquet     (all BD samples for pretraining)
  - data/processed/final_pretraining_stage1.parquet   (GEMStat + Podgorski + BD)
  - data/processed/final_temporal_pairs.parquet       (GEMStat + BD temporal pairs)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
MATCHED_WELLS = Path("/Users/rakibhhridoy/AsGW/GroundWater/Paper3/analysis/output/tables/matched_wells.csv")
FULL_DATA = Path("/Users/rakibhhridoy/AsGW/GroundWater/Paper1/data/raw/Data.csv")
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

GENESIS_PARAMS = ['As', 'Fe', 'Mn', 'PO4', 'F', 'U', 'NO3',
                  'pH', 'Eh', 'EC', 'TDS',
                  'Ca', 'Mg', 'Na', 'K', 'Cl', 'HCO3', 'SO4',
                  'SiO2', 'DOC']


def load_matched_wells():
    """Load 705 paired wells from Paper 3."""
    print("Loading matched wells (705 pairs)...")
    df = pd.read_csv(MATCHED_WELLS)
    print(f"  Shape: {df.shape}")
    print(f"  Periods: {df['Period'].value_counts().to_dict()}")
    print(f"  Unique pairs: {df['pair_key'].nunique()}")
    print(f"  Districts: {df['District'].nunique()}")
    return df


def load_full_campaign():
    """Load full 2020-2021 campaign from Paper 1."""
    print("\nLoading full 2020-2021 campaign...")
    df = pd.read_csv(FULL_DATA)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)[:15]}...")

    # Rename columns to match GENESIS params
    col_map = {
        'As': 'As', 'Fe2+': 'Fe', 'Mn2+': 'Mn', 'PO43-': 'PO4',
        'pH': 'pH', 'ORP': 'Eh', 'EC': 'EC', 'TDS': 'TDS',
        'Ca2+': 'Ca', 'Mg2+': 'Mg', 'Na+': 'Na', 'K+': 'K',
        'Cl-': 'Cl', 'HCO3-': 'HCO3', 'SO42-': 'SO4', 'NO3-': 'NO3',
    }
    df = df.rename(columns=col_map)
    print(f"  Samples: {len(df):,}")
    return df


def format_matched_temporal_pairs(df):
    """Convert matched wells to GENESIS temporal pair format."""
    print("\nFormatting temporal pairs...")

    # Column mapping for matched wells
    col_map = {
        'As': 'As', 'Fe': 'Fe', 'Mn': 'Mn', 'PO4': 'PO4',
        'pH': 'pH', 'Eh': 'Eh', 'EC': 'EC', 'TDS': 'TDS',
        'Ca': 'Ca', 'Mg': 'Mg', 'Na': 'Na', 'K': 'K',
        'Cl': 'Cl', 'HCO3': 'HCO3', 'SO4': 'SO4', 'NO3': 'NO3',
    }

    # Split by period
    old = df[df['Period'] == '2012-2013'].copy()
    new = df[df['Period'] == '2020-2021'].copy()

    print(f"  2012-2013 samples: {len(old)}")
    print(f"  2020-2021 samples: {len(new)}")

    # Merge on pair_key
    pairs = old.merge(new, on='pair_key', suffixes=('_old', '_new'))
    print(f"  Matched pairs: {len(pairs)}")

    # Build temporal pair records
    records = []
    for _, row in pairs.iterrows():
        rec = {
            'station_id': f"BD_{row['pair_key']}",
            'date_t0': '2012-07-01',  # approximate mid-point
            'date_t1': '2020-07-01',
            'year_t0': 2012,
            'year_t1': 2020,
            'gap_years': 8.0,
            'country': 'Bangladesh',
            'lat': row.get('Latitude_new') or row.get('Latitude_old'),
            'lon': row.get('Longitude_new') or row.get('Longitude_old'),
            'depth': row.get('Depth_new') or row.get('Depth_old'),
            'depth_bin': row.get('Depth_bin_new') or row.get('Depth_bin_old'),
            'district': row.get('District_new') or row.get('District_old'),
            'source': 'bangladesh',
        }

        n_paired = 0
        for genesis_p in GENESIS_PARAMS:
            mapped = col_map.get(genesis_p)
            if mapped:
                v0 = row.get(f'{mapped}_old')
                v1 = row.get(f'{mapped}_new')
            else:
                v0, v1 = None, None

            rec[f'{genesis_p}_t0'] = v0 if pd.notna(v0) else None
            rec[f'{genesis_p}_t1'] = v1 if pd.notna(v1) else None

            if pd.notna(v0) and pd.notna(v1):
                rec[f'delta_{genesis_p}'] = v1 - v0
                n_paired += 1
            else:
                rec[f'delta_{genesis_p}'] = None

        rec['n_paired_params'] = n_paired
        records.append(rec)

    df_pairs = pd.DataFrame(records)
    print(f"  Temporal pairs created: {len(df_pairs)}")
    print(f"  Mean paired params: {df_pairs['n_paired_params'].mean():.1f}")

    print(f"\n  Parameter coverage in BD temporal pairs:")
    for p in GENESIS_PARAMS:
        n = df_pairs[f'delta_{p}'].notna().sum()
        pct = n / len(df_pairs) * 100 if len(df_pairs) > 0 else 0
        print(f"    {p:<6}: {n:>5} pairs ({pct:5.1f}%)")

    return df_pairs


def format_full_campaign_vectors(df):
    """Format full 2020-2021 campaign as GENESIS vectors for pretraining."""
    print("\nFormatting full campaign as pretraining vectors...")

    vectors = []
    for _, row in df.iterrows():
        rec = {
            'station_id': f"BD_full_{row.get('Sample ID', '')}",
            'sample_date': pd.NaT,
            'year': 2021,
            'lat': row.get('Latitude'),
            'lon': row.get('Longitude'),
            'country': 'Bangladesh',
            'elevation': np.nan,
            'depth_casing': row.get('Depth'),
            'basin': '',
            'source': 'bangladesh_full',
        }

        n_params = 0
        for p in GENESIS_PARAMS:
            val = row.get(p)
            if pd.notna(val):
                try:
                    rec[p] = float(val)
                    n_params += 1
                except (ValueError, TypeError):
                    rec[p] = None
            else:
                rec[p] = None

        rec['n_params'] = n_params
        vectors.append(rec)

    df_vec = pd.DataFrame(vectors)
    df_vec = df_vec.dropna(subset=['lat', 'lon'])
    print(f"  Vectors: {len(df_vec):,}")
    print(f"  Mean params per sample: {df_vec['n_params'].mean():.1f}")
    return df_vec


def combine_all_pretraining(bd_vectors):
    """Combine GEMStat + Podgorski + Bangladesh for final pretraining set."""
    print("\nCombining all pretraining data...")

    combined_s1 = pd.read_parquet(PROCESSED_DIR / "combined_pretraining_stage1.parquet")

    # Align columns
    shared_cols = ['station_id', 'sample_date', 'year', 'lat', 'lon',
                   'country', 'elevation', 'depth_casing', 'basin',
                   'source', 'n_params'] + GENESIS_PARAMS

    for col in shared_cols:
        if col not in combined_s1.columns:
            combined_s1[col] = np.nan
        if col not in bd_vectors.columns:
            bd_vectors[col] = np.nan

    final = pd.concat([combined_s1[shared_cols], bd_vectors[shared_cols]], ignore_index=True)

    print(f"\n  Final pretraining dataset:")
    by_source = final['source'].value_counts()
    for src, n in by_source.items():
        print(f"    {src}: {n:,}")
    print(f"    TOTAL: {len(final):,}")

    return final


def combine_all_temporal_pairs(bd_pairs):
    """Combine GEMStat + Bangladesh temporal pairs."""
    print("\nCombining all temporal pairs...")

    gemstat_pairs = pd.read_parquet(PROCESSED_DIR / "gw_temporal_pairs.parquet")
    gemstat_pairs['source'] = 'gemstat'

    # Align columns
    shared_cols = ['station_id', 'date_t0', 'date_t1', 'year_t0', 'year_t1',
                   'gap_years', 'country', 'lat', 'lon', 'n_paired_params', 'source']
    for p in GENESIS_PARAMS:
        shared_cols.extend([f'{p}_t0', f'{p}_t1', f'delta_{p}'])

    for col in shared_cols:
        if col not in gemstat_pairs.columns:
            gemstat_pairs[col] = np.nan
        if col not in bd_pairs.columns:
            bd_pairs[col] = np.nan

    # Ensure date columns are strings for compatibility
    for col in ['date_t0', 'date_t1']:
        gemstat_pairs[col] = gemstat_pairs[col].astype(str)
        bd_pairs[col] = bd_pairs[col].astype(str)

    final_pairs = pd.concat([gemstat_pairs[shared_cols], bd_pairs[shared_cols]], ignore_index=True)

    print(f"\n  Final temporal pairs:")
    by_source = final_pairs['source'].value_counts()
    for src, n in by_source.items():
        print(f"    {src}: {n:,}")
    print(f"    TOTAL: {len(final_pairs):,}")
    print(f"    Countries: {final_pairs['country'].nunique()}")
    print(f"    Mean paired params: {final_pairs['n_paired_params'].mean():.1f}")

    return final_pairs


def main():
    print("=" * 70)
    print("GENESIS — Bangladesh Data Integration")
    print("=" * 70)

    # 1. Load matched wells (705 pairs)
    matched = load_matched_wells()

    # 2. Format as temporal pairs
    bd_pairs = format_matched_temporal_pairs(matched)
    bd_pairs.to_parquet(PROCESSED_DIR / "bangladesh_temporal_pairs.parquet", index=False)
    print(f"\n  Saved BD temporal pairs: {len(bd_pairs)}")

    # 3. Load and format full 2020-2021 campaign
    full_campaign = load_full_campaign()
    bd_vectors = format_full_campaign_vectors(full_campaign)
    bd_vectors.to_parquet(PROCESSED_DIR / "bangladesh_all_samples.parquet", index=False)
    print(f"  Saved BD all samples: {len(bd_vectors)}")

    # 4. Run Podgorski integration first if not already done
    podgorski_path = PROCESSED_DIR / "combined_pretraining_stage1.parquet"
    if not podgorski_path.exists():
        print("\n  Running Podgorski integration first...")
        import subprocess
        subprocess.run(["python3", str(Path(__file__).parent / "02_integrate_podgorski.py")])

    # 5. Combine all pretraining data
    final_pretrain = combine_all_pretraining(bd_vectors)
    final_pretrain.to_parquet(PROCESSED_DIR / "final_pretraining_stage1.parquet", index=False)
    print(f"\n  Saved final pretraining Stage 1: {len(final_pretrain):,}")

    # 6. Combine all temporal pairs
    final_pairs = combine_all_temporal_pairs(bd_pairs)
    final_pairs.to_parquet(PROCESSED_DIR / "final_temporal_pairs.parquet", index=False)
    print(f"  Saved final temporal pairs: {len(final_pairs):,}")

    # 7. Summary
    print("\n" + "=" * 70)
    print("FINAL GENESIS DATASET SUMMARY")
    print("=" * 70)
    print(f"\nPretraining (Stage 1): {len(final_pretrain):,} samples")
    print(f"  - GEMStat groundwater: {(final_pretrain['source']=='gemstat').sum():,}")
    print(f"  - Podgorski arsenic:   {(final_pretrain['source']=='podgorski_as').sum():,}")
    print(f"  - Bangladesh full:     {(final_pretrain['source']=='bangladesh_full').sum():,}")
    print(f"\nTemporal pairs: {len(final_pairs):,} pairs")
    print(f"  - GEMStat (17 countries): {(final_pairs['source']=='gemstat').sum():,}")
    print(f"  - Bangladesh (705 wells): {(final_pairs['source']=='bangladesh').sum():,}")
    print(f"\nTotal countries with temporal data: {final_pairs['country'].nunique()}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
