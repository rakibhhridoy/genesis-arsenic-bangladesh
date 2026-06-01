"""
GENESIS Paper 5 — Step 2: Integrate Podgorski Arsenic Data
============================================================
Reads Podgorski & Berg (2020) arsenic point data (56K points with
predictor variables), formats as GENESIS geochemical vectors, and
merges with GEMStat curated data.

Output:
  - data/processed/podgorski_arsenic_vectors.parquet
  - data/processed/combined_pretraining_stage1.parquet
  - data/processed/integration_report.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path

PODGORSKI_CSV = Path(__file__).parent.parent / "data" / "Podgorski" / "arsenic_data" / "data" / "arsenic_concentrations_predictor_data.csv"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
GEMSTAT_S1 = PROCESSED_DIR / "gw_geochemical_vectors_stage1.parquet"

GENESIS_PARAMS = ['As', 'Fe', 'Mn', 'PO4', 'F', 'U', 'NO3',
                  'pH', 'Eh', 'EC', 'TDS',
                  'Ca', 'Mg', 'Na', 'K', 'Cl', 'HCO3', 'SO4',
                  'SiO2', 'DOC']


def load_podgorski():
    """Load Podgorski arsenic dataset."""
    print("Loading Podgorski arsenic data...")
    df = pd.read_csv(PODGORSKI_CSV, on_bad_lines='skip')
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns[:10])}...")
    print(f"  Lat range: {df['Latitude'].min():.2f} to {df['Latitude'].max():.2f}")
    print(f"  Lon range: {df['Longitude'].min():.2f} to {df['Longitude'].max():.2f}")
    print(f"  As range: {df['As_ppb'].min():.1f} to {df['As_ppb'].max():.1f} µg/L")
    print(f"  As median: {df['As_ppb'].median():.1f} µg/L")
    return df


def format_podgorski_vectors(df):
    """
    Format Podgorski data as GENESIS geochemical vectors.
    Only As concentration is available as a geochemical measurement.
    Predictor variables (climate, soil, geology) are kept as covariates.
    """
    print("\nFormatting as GENESIS vectors...")

    vectors = pd.DataFrame({
        'station_id': [f'POD_{i:06d}' for i in range(len(df))],
        'sample_date': pd.NaT,  # No date info in Podgorski
        'year': np.nan,
        'lat': df['Latitude'].values,
        'lon': df['Longitude'].values,
        'country': '',  # Not in dataset, could infer from coordinates
        'elevation': np.nan,
        'depth_casing': np.nan,
        'basin': '',
        'source': 'podgorski_as',
    })

    # Only As is a direct geochemical measurement (convert ppb → mg/L)
    for param in GENESIS_PARAMS:
        if param == 'As':
            vectors[param] = df['As_ppb'].values / 1000.0  # µg/L → mg/L
        else:
            vectors[param] = np.nan

    vectors['n_params'] = 1  # Only As available

    # Add predictor variables as covariates (for conditioning)
    covariate_cols = [
        'pet', 'precip', 'aridity', 'temp', 'aet',
        'water_table_depth', 'slope', 'twi',
        'soil_and_sedimentary_deposit_thickness',
        'CEC_subsoil', 'clay_topsoil', 'clay_subsoil',
        'organic_carbon', 'pH_subsoil',
        'sand_topsoil', 'sand_subsoil',
        'quaternary', 'allSedimentary', 'allVolcanics', 'allIgneous',
        'fluvisols', 'gleysols',
    ]

    for col in covariate_cols:
        if col in df.columns:
            vectors[f'cov_{col}'] = df[col].values

    print(f"  Vectors created: {len(vectors):,}")
    print(f"  With As values: {vectors['As'].notna().sum():,}")
    print(f"  Covariates added: {sum(1 for c in vectors.columns if c.startswith('cov_'))}")

    return vectors


def combine_with_gemstat(podgorski_vectors):
    """Combine Podgorski vectors with GEMStat Stage 1 data."""
    print("\nCombining with GEMStat data...")

    gemstat = pd.read_parquet(GEMSTAT_S1)
    gemstat['source'] = 'gemstat'

    # Align columns — keep only shared columns for combination
    shared_cols = ['station_id', 'sample_date', 'year', 'lat', 'lon',
                   'country', 'elevation', 'depth_casing', 'basin',
                   'source', 'n_params'] + GENESIS_PARAMS

    # Ensure both have the same columns
    for col in shared_cols:
        if col not in gemstat.columns:
            gemstat[col] = np.nan
        if col not in podgorski_vectors.columns:
            podgorski_vectors[col] = np.nan

    gemstat_subset = gemstat[shared_cols].copy()
    podgorski_subset = podgorski_vectors[shared_cols].copy()

    combined = pd.concat([gemstat_subset, podgorski_subset], ignore_index=True)

    print(f"  GEMStat Stage 1: {len(gemstat_subset):,} samples")
    print(f"  Podgorski As:    {len(podgorski_subset):,} samples")
    print(f"  Combined:        {len(combined):,} samples")

    print(f"\n  Combined parameter coverage:")
    for param in GENESIS_PARAMS:
        n = combined[param].notna().sum()
        pct = n / len(combined) * 100
        print(f"    {param:<6}: {n:>8,} ({pct:5.1f}%)")

    return combined


def write_report(podgorski_df, podgorski_vectors, combined):
    """Write integration report."""
    report_path = PROCESSED_DIR / "integration_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GENESIS — Data Integration Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("--- PODGORSKI ARSENIC DATA ---\n")
        f.write(f"Source: Podgorski & Berg (2020), Science\n")
        f.write(f"Points: {len(podgorski_df):,}\n")
        f.write(f"As range: {podgorski_df['As_ppb'].min():.1f} - "
                f"{podgorski_df['As_ppb'].max():.1f} µg/L\n")
        f.write(f"As median: {podgorski_df['As_ppb'].median():.1f} µg/L\n")
        f.write(f"As > 10 µg/L: {(podgorski_df['As_ppb'] > 10).sum():,} "
                f"({(podgorski_df['As_ppb'] > 10).mean()*100:.1f}%)\n\n")

        f.write("--- COMBINED PRETRAINING DATA ---\n")
        f.write(f"Total samples: {len(combined):,}\n")
        by_source = combined['source'].value_counts()
        for src, n in by_source.items():
            f.write(f"  {src}: {n:,}\n")
        f.write(f"\nParameter coverage:\n")
        for param in GENESIS_PARAMS:
            n = combined[param].notna().sum()
            f.write(f"  {param:<6}: {n:>8,} ({n/len(combined)*100:5.1f}%)\n")

    print(f"\nReport saved to: {report_path}")


def main():
    print("=" * 70)
    print("GENESIS — Podgorski Data Integration")
    print("=" * 70)

    # 1. Load Podgorski data
    podgorski_df = load_podgorski()

    # 2. Format as GENESIS vectors
    podgorski_vectors = format_podgorski_vectors(podgorski_df)

    # 3. Save Podgorski vectors separately
    podgorski_vectors.to_parquet(PROCESSED_DIR / "podgorski_arsenic_vectors.parquet", index=False)
    print(f"\n  Saved Podgorski vectors: {len(podgorski_vectors):,}")

    # 4. Combine with GEMStat
    combined = combine_with_gemstat(podgorski_vectors)

    # 5. Save combined pretraining dataset
    combined.to_parquet(PROCESSED_DIR / "combined_pretraining_stage1.parquet", index=False)
    print(f"\n  Saved combined Stage 1: {len(combined):,}")

    # 6. Report
    write_report(podgorski_df, podgorski_vectors, combined)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
