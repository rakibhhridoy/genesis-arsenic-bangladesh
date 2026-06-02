"""
GENESIS Paper 5 — Step 20: Discovery-angle exploration (2026-06-02).

Records the grounded exploration of four candidate discovery angles beyond the
transfer law. Run against data/genesis.duckdb (clean) + temporal-pairs meta.

SUMMARY OF FINDINGS (see commit / memory for detail):

ANGLE A — As-U redox antithesis: REFUTED in simple form.
  As-U positively correlated (rho +0.44, n=2323), not anti-correlated. U IS
  oxidizing-favoured (U-Eh +0.236; U median 0.58 reducing -> 6-37 oxic). As-Eh
  weak (-0.035, poor Eh coverage).

ANGLE B — uranium = carbonate complexation: REFUTED as clean mechanism.
  U-HCO3 +0.42 raw but partial corr | (pH,Ca,TDS) = +0.008 -> confounded by bulk
  mineralization. U-TDS +0.56. Honest version: U exceedance tracks MULTIVARIATE
  oxidizing high-mineralization context; individual major-ion correlation signs
  vary by region -> a multivariate nonlinear representation beats a linear
  sign-rule (this supports the transfer-law uranium win, confounder-robust).

ANGLE C — decade-scale change: ROBUST, standalone (second-paper seed).
  58,722 pairs, median 14-yr gap. Wilcoxon all p<<0.001:
    As: France median -1.75 (declining), GEMStat +2.61 (RISING, p=5e-204),
        USA/EU slightly declining.
    NO3: France +0.25 (p=1e-138), USA +0.06 (p=7e-24) rising (agricultural);
         EU declining.
  France nuance: bulk As halved (5.0->2.0) BUT dangerous tail grew
    (>10 ugL wells 666->695; 212 recovered, 119 newly-unsafe).
  Drivers weak: aridity strongest single correlate of dAs (rho -0.17);
    population/GRACE-twsa/precip all <0.08.

ANGLE D — depletion x contamination (GRACE twsa_trend): WEAK globally
  (rho 0.03-0.07). Possibly regional; not a headline.

Usage:
  python src/20_discovery_exploration.py
"""
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from numpy.linalg import lstsq

DB = "data/genesis.duckdb"
META = "data/processed/genesis_temporal_pairs_meta.parquet"


def region_of(s):
    s = str(s)
    if "NWIS" in s:
        return "USA"
    if "ADES" in s:
        return "France"
    if "EEA" in s:
        return "EU"
    if "GEMS" in s.upper():
        return "GEMStat"
    return s


def partial_spearman(y, x, Z):
    """Spearman of y,x controlling for columns of Z (rank-linear residuals)."""
    def resid(v):
        R = np.column_stack([pd.Series(Z[:, i]).rank().values for i in range(Z.shape[1])]
                            + [np.ones(len(v))])
        vr = pd.Series(v).rank().values
        b, *_ = lstsq(R, vr, rcond=None)
        return vr - R @ b
    return spearmanr(resid(y), resid(x)).correlation


def main():
    con = duckdb.connect(DB, read_only=True)
    df = con.execute("SELECT * FROM genesis_temporal_pairs").df()
    con.close()
    df["region"] = df["source"].map(region_of)

    print("ANGLE C — decade change (Wilcoxon on paired delta):")
    for p in ("As", "NO3"):
        for rg, g in df.groupby("region"):
            d = (g[f"{p}_t1"] - g[f"{p}_t0"]).dropna()
            if len(d) > 100:
                try:
                    _, pv = wilcoxon(d)
                except ValueError:
                    pv = np.nan
                print(f"  {p:4s} {rg:8s} n={len(d):6d} median={d.median():+7.3f} p={pv:.1e}")

    print("\nANGLE A/B — uranium chemistry:")
    s = df[["U_t0", "Eh_t0"]].dropna()
    print(f"  U-Eh   rho={spearmanr(s.U_t0, s.Eh_t0).correlation:+.3f} (n={len(s)})")
    s = df[["As_t0", "U_t0"]].dropna()
    print(f"  As-U   rho={spearmanr(s.As_t0, s.U_t0).correlation:+.3f} (n={len(s)})")
    s = df[["U_t0", "HCO3_t0", "pH_t0", "Ca_t0", "TDS_t0"]].dropna()
    raw = spearmanr(s.U_t0, s.HCO3_t0).correlation
    par = partial_spearman(s.U_t0.values, s.HCO3_t0.values,
                           s[["pH_t0", "Ca_t0", "TDS_t0"]].values)
    print(f"  U-HCO3 raw={raw:+.3f}  partial|(pH,Ca,TDS)={par:+.3f} (n={len(s)})  -> confounded")


if __name__ == "__main__":
    main()
