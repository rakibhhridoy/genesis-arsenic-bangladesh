#!/usr/bin/env python3
"""Control for src/52: is the transfer gradient geochemistry, or a data-source artifact?

In src/52 leave-one-region-out skill tracked the learnability gradient (rho +0.82 over 20
parameters), but every region there is also a separate data source (USGS NWIS, BRGM ADES,
EEA WISE, GEMStat). Sources differ in filtration, Eh method and detection limits, which
hit the redox species hardest. That alone could produce the same ordering.

This removes the source boundary. Inside one source (one protocol) it holds out large
geographic blocks (USA: 10 degree blocks; France: 2 degree blocks, since France spans
about 10 degrees) and asks whether out-of-block skill still falls with reconstruction R2.

  python3 src/53_within_source_transfer.py --seeds 5 --n 3000
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, GroupKFold

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("h52", ROOT/"src/52_hazard_transfer_gradient.py")
h52 = importlib.util.module_from_spec(spec); spec.loader.exec_module(h52)
SOURCES = {"USA": 10.0, "France": 2.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default=str(ROOT/"results/hazard_transfer_within_source.json"))
    a = ap.parse_args()

    s, _ = h52.load_sites()
    r2 = {k: v["r2"] for k, v in json.load(open(ROOT/"results/reconstruction_probe.json"))["corpus"].items()}
    rows = []
    for src, bdeg in SOURCES.items():
        d0 = s[s.region == src]
        for p in h52.GENESIS_PARAMS:
            d1 = d0[d0[p].notna()]
            if len(d1) < h52.MIN_SITES:
                continue
            thr = float(np.nanpercentile(d1[p], 80))
            for seed in range(a.seeds):
                rng = np.random.default_rng(seed)
                d = d1.iloc[rng.choice(len(d1), min(a.n, len(d1)), replace=False)]
                X = d[h52.AUX_FEATURES].to_numpy(np.float64)
                X = np.where(np.isfinite(X), X, np.nanmedian(X, 0))
                y = (d[p].to_numpy() > thr).astype(int)
                g = (np.floor(d.lat.values / bdeg).astype(int) * 10000
                     + np.floor(d.lon.values / bdeg).astype(int))
                rnd = h52.cv_auc(X, y, StratifiedKFold(5, shuffle=True, random_state=seed), None, seed)
                blk = h52.cv_auc(X, y, GroupKFold(min(5, len(np.unique(g)))), g, seed)
                rows.append(dict(source=src, param=p, seed=seed, n=int(len(y)), r2=r2.get(p),
                                 random=rnd, block=blk, gap=rnd - blk))
                print(f"  {src:6s} {p:5s} seed{seed} n={len(y):>5,} random={rnd:.3f} "
                      f"block{bdeg:g}={blk:.3f}", flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)

    t = pd.DataFrame(rows).groupby(["source", "param"]).mean(numeric_only=True).reset_index()
    for src in SOURCES:
        q = t[t.source == src].dropna()
        for k in ["random", "block", "gap"]:
            rho, pv = spearmanr(q.r2, q[k])
            print(f"{src:6s} {k:6s} vs R2: rho={rho:+.2f} p={pv:.3f} n={len(q)}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
