#!/usr/bin/env python3
"""Control for src/52: is the cross-source transfer gradient a detection-limit artifact?

Sources censor trace species at different reporting limits (a "<1 ug/L" in one network is
a "<0.1" in another). If an 80th-percentile threshold falls near those limits, the label
means different things in different sources, and transfer across sources fails more for
trace (mostly redox) species for purely reporting reasons.

Fix: estimate each source's reporting floor per parameter from the raw samples (the most
frequent value in its lowest decile, which is where censored values pile up), take the
highest floor across sources (Bangladesh included), and require the label threshold to
sit above it. A value at or below every source's floor is then negative everywhere, so
the label is comparable across networks. Filtration and Eh-method differences are NOT
addressed by this and remain a stated limitation.

  python3 src/55_detection_limit_control.py --seeds 5 --n 3000
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("h52", ROOT/"src/52_hazard_transfer_gradient.py")
h52 = importlib.util.module_from_spec(spec); spec.loader.exec_module(h52)
P = h52.GENESIS_PARAMS


def floor_of(v):
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 200:
        return np.nan
    low = v[v <= np.quantile(v, 0.10)]
    vals, cnt = np.unique(np.round(low, 6), return_counts=True)
    return float(vals[cnt.argmax()])


def floors():
    X = torch.load(ROOT/"data/processed/genesis_pretrain.pt", map_location="cpu",
                   weights_only=False).numpy()
    m = pd.read_parquet(ROOT/"data/processed/genesis_pretrain_meta.parquet",
                        columns=["source", "country", "water_body_type"])
    gw = (m.water_body_type == "groundwater").values
    reg = np.array([h52.region_of(a, b) for a, b in zip(m.source.values, m.country.values)])
    bx = torch.load(ROOT/"data/processed/genesis_held_out_bd.pt", map_location="cpu",
                    weights_only=False).numpy()
    out = {}
    for i, p in enumerate(P):
        f = {R: floor_of(X[gw & (reg == R), i]) for R in pd.unique(reg[gw])}
        f["Bangladesh"] = floor_of(bx[:, i])
        out[p] = {k: v for k, v in f.items() if np.isfinite(v)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default=str(ROOT/"results/hazard_transfer_dlcontrol.json"))
    a = ap.parse_args()

    fl = floors()
    s, bd = h52.load_sites()
    r2 = {k: v["r2"] for k, v in json.load(open(ROOT/"results/reconstruction_probe.json"))["corpus"].items()}
    rows = []
    print(f"{'param':5s} {'p80':>9s} {'max floor':>10s} {'thr':>9s} {'prev':>5s}  floors by source")
    for p in P:
        if s[p].notna().sum() < h52.MIN_SITES:
            continue
        p80 = float(np.nanpercentile(s[p], 80))
        mf = max(fl[p].values()) if fl[p] else 0.0
        thr = max(p80, mf)
        prev = float((s[p].dropna() > thr).mean())
        print(f"{p:5s} {p80:9.4g} {mf:10.4g} {thr:9.4g} {prev:5.2f}  "
              + ", ".join(f"{k}={v:.3g}" for k, v in sorted(fl[p].items())), flush=True)
        for seed in range(a.seeds):
            o = h52.evaluate(s, bd, p, thr, a.n, seed)
            if o is None:
                print(f"    {p} seed{seed}: too few positives above the floor", flush=True)
                continue
            o.update(mode="dl", thr=thr, p80=p80, floor=mf, r2=r2.get(p))
            rows.append(o)
            print(f"    seed{seed} prev={o['prev']:.2f} random={o['random']:.3f} "
                  f"b10={o['block10']:.3f} loro={o['loro_mean']:.3f} ({len(o['loro'])} regions)",
                  flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)
    json.dump({"floors": fl}, open(ROOT/"results/source_reporting_floors.json", "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
