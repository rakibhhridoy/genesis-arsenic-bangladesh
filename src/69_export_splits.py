#!/usr/bin/env python3
"""Export the fixed evaluation partitions as index arrays and check them against the predictions.

  site-disjoint split  src/56 and src/61, row indices into data/processed/genesis_pretrain.pt
  LORO partitions      src/59, row indices into the concatenated genesis_temporal_pairs.pt and
                       genesis_held_out_bd_pairs.pt, one set per (target, region, seed)

The partitions are regenerated with the functions that made them, then every saved prediction
file is compared with them on held-out indices and on calibration, in-distribution and
held-out labels. Any disagreement stops the run before the archive is written.

Writes results/evaluation_splits.npz.

  python src/69_export_splits.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = load_mod("p59", "src/59_loro_conformal_preds.py")
SEEDS = [42, 43, 44]
ROOTS = ["results/conformal_preds", "results/conformal_preds_quantum"]


def site_disjoint():
    """The train/test construction of src/61 load(), without reading the chemistry tensors."""
    import torch
    n = len(torch.load(ROOT / "data/processed/genesis_pretrain.pt", map_location="cpu", weights_only=False))
    meta = pd.read_parquet(ROOT / "data/processed/genesis_pretrain_meta.parquet",
                           columns=["site_id", "water_body_type"])
    n_val = min(50000, max(1, int(n * 0.05)))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42)).numpy()
    tr, va = perm[:n - n_val], perm[n - n_val:]
    sites = set(meta.site_id.values[tr])
    gw = meta.water_body_type.values == "groundwater"
    test = np.array([i for i in va if gw[i] and meta.site_id.values[i] not in sites])
    return tr[gw[tr]], test


def main():
    out, checked = {}, 0
    out["site_disjoint__train_groundwater"], out["site_disjoint__test"] = site_disjoint()
    _, _, _, _, t1, region = P.load()
    dirs = [d for r in ROOTS for d in sorted((ROOT / r).iterdir())
            if d.is_dir() and not d.name.startswith(".") and d.name.rsplit("_s", 1)[-1].isdigit()]
    n_cells = 0
    for target, R, y, tr_all, te in P.cells(t1, region):
        n_cells += 1
        name = f"{target}__{R.replace(':', '-').replace('/', '-')}"
        out[f"{name}__test_out"] = te
        for seed in SEEDS:
            tr, va, cal, tin = P.partition(tr_all, target, R, seed)
            for k, v in zip(("train", "val", "cal", "test_in"), (tr, va, cal, tin)):
                out[f"{name}__s{seed}__{k}"] = v
            for d in dirs:
                if not d.name.endswith(f"_s{seed}"):
                    continue
                f = d / f"{name}.npz"
                if not f.exists():
                    raise SystemExit(f"missing prediction file {f}")
                z = np.load(f)
                same = (np.array_equal(z["idx_out"], te) and np.array_equal(z["y_out"], y[te])
                        and np.array_equal(z["y_cal"], y[cal]) and np.array_equal(z["y_in"], y[tin]))
                if not same:
                    raise SystemExit(f"partition disagrees with {f}")
                checked += 1
    np.savez_compressed(ROOT / "results/evaluation_splits.npz", **out)
    print(f"{n_cells} cells, {len(dirs)} method-seed runs, {checked} prediction files agree; "
          f"site-disjoint train {len(out['site_disjoint__train_groundwater']):,}, "
          f"test {len(out['site_disjoint__test']):,}")


if __name__ == "__main__":
    main()
