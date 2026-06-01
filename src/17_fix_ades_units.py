"""
GENESIS Paper 5 — Step 17: ADES (France) unit/label correction for the
temporal-pairs artifact.

ROOT CAUSE
----------
The pretraining-DB build (src/04_build_genesis_db.py) applies a remediation for
the ADES "old chunk" code-mislabeling problem, but the temporal-pairs file
(data/processed/final_temporal_pairs_enriched_global.parquet) was assembled
WITHOUT that remediation. As a result, ADES (France) rows carry corrupted values
for several parameters, while the pretraining corpus itself is clean.

Empirically verified against USA/NWIS (which is correct):
  - U   : uniformly ~1000x too high (median 1300 ug/L vs ~1.3 realistic).
          -> scalar fix: divide by 1000  (matches USA median 1.14 ug/L).
  - Eh  : values are ~8-17, i.e. groundwater TEMPERATURE (deg C), not redox
          potential (old-chunk SANDRE 1301 = temperature). -> set to NaN.
  - Cl  : median ~0.001 mg/L, physically impossible (mislabel). -> set to NaN.
  - TDS : median ~2 mg/L, physically impossible (mislabel). -> set to NaN.

This script is IDEMPOTENT: it only acts on ADES/BRGM rows, and only if the U
median still looks inflated (>100 ug/L), so re-running is safe.

The proper long-term fix is to rebuild the temporal-pairs file from the
corrected GENESIS DB; this script is a documented, reproducible stop-gap so the
evaluation artifact matches the (clean) pretraining corpus.

Usage:
  python src/17_fix_ades_units.py
"""
import os
import shutil
import pandas as pd
import numpy as np

PAIRS = "data/processed/final_temporal_pairs_enriched_global.parquet"
ADES_PAT = r"ADES|BRGM"


def main():
    if not os.path.exists(PAIRS):
        raise SystemExit(f"Not found: {PAIRS}")

    df = pd.read_parquet(PAIRS)
    src = df["source"].astype(str)
    ades = src.str.contains(ADES_PAT, case=False, na=False)
    n_ades = int(ades.sum())
    print(f"Loaded {len(df):,} pairs; {n_ades:,} ADES/France rows")

    u_med = df.loc[ades, "U_t1"].dropna().median()
    print(f"ADES U_t1 median before: {u_med:.3f} ug/L")
    if not (u_med > 100):
        print("U median not inflated -> already corrected; nothing to do (idempotent).")
        return

    # one-time backup
    bak = PAIRS + ".prefix17.bak"
    if not os.path.exists(bak):
        shutil.copy2(PAIRS, bak)
        print(f"Backed up original -> {bak}")

    # --- U: divide by 1000 (uniform inflation), recompute delta ---
    for leg in ("U_t0", "U_t1"):
        if leg in df.columns:
            df.loc[ades, leg] = df.loc[ades, leg] / 1000.0
    if {"U_t0", "U_t1", "delta_U"}.issubset(df.columns):
        df.loc[ades, "delta_U"] = df.loc[ades, "U_t1"] - df.loc[ades, "U_t0"]

    # --- Eh, Cl, TDS: wrong-parameter mislabels -> NaN (matches DB-build discard) ---
    for p in ("Eh", "Cl", "TDS"):
        for leg in (f"{p}_t0", f"{p}_t1", f"delta_{p}"):
            if leg in df.columns:
                df.loc[ades, leg] = np.nan

    df.to_parquet(PAIRS, index=False)

    # --- report ---
    print("\nAfter correction (ADES/France):")
    print(f"  U_t1   median: {df.loc[ades, 'U_t1'].dropna().median():.3f} ug/L "
          f"(USA ref ~1.14)")
    for p in ("Eh", "Cl", "TDS"):
        kept = int(df.loc[ades, f"{p}_t1"].notna().sum())
        print(f"  {p}_t1  non-null ADES rows now: {kept} (nulled mislabels)")
    print(f"\nSaved corrected file -> {PAIRS}")


if __name__ == "__main__":
    main()
