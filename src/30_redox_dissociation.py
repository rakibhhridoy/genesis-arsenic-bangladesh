"""
GENESIS Paper 5 — Step 30: the redox-coupling dissociation (paper headline stats).

Computes, deterministically from the released result JSONs, every central number
of the reframed paper:
  - benchmark-wide fine-tuned vs Random Forest (seed-averaged, paired Wilcoxon)
  - the mechanism-defined group tests: REDOX-coupled (As,Fe,Mn,PO4) vs RF, and
    CONSERVATIVE (NO3,F) vs RF
  - frozen vs fine-tuned for each group (shows fine-tuning UNLOCKS the redox win)
  - uranium counter-example
  - per-target advantage table

The redox vs conservative grouping is PRE-SPECIFIED by the attention/reconstruction
mechanism (the encoder encodes the As-Fe-Eh-PO4 redox coupling; conservative ions
are linearly trivial), NOT chosen post-hoc to maximize significance.

Inputs (all released, no model/GPU needed):
  results/loro_finetune_large_multiseed.json   (3-seed fine-tune, src/29)
  results/region_transfer_encoder_large.json   (frozen encoder + RF/XGB cols, src/19+27)

Usage:  python src/30_redox_dissociation.py
Writes: results/redox_dissociation.json  (machine-readable headline stats)
"""
import json
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

REDOX = {"As", "Fe", "Mn", "PO4"}          # redox-coupled (mechanism-defined)
CONSERVATIVE = {"NO3", "F"}                 # conservatively-behaved major/minor ions
URANIUM = {"U"}                             # confounded counter-example
FT = "results/loro_finetune_large_multiseed.json"
ENC = "results/region_transfer_encoder_large.json"


def load_rows(path, key=None):
    d = json.load(open(path))
    if isinstance(d, list):
        return d
    for k in (key, "per_seed_rows", "results", "rows"):
        if k and k in d and isinstance(d[k], list):
            return d[k]
    raise ValueError(f"no row list in {path}")


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 1:
        return dict(n=0)
    out = dict(n=len(a), mean_a=float(a.mean()), mean_b=float(b.mean()),
               delta=float((a - b).mean()), wins=int((a > b).sum()))
    try:
        _, p = wilcoxon(a, b)
        out["wilcoxon_p"] = float(p)
        out["significant"] = bool(p < 0.05)
    except ValueError:
        out["wilcoxon_p"] = None
        out["significant"] = False
    return out


def main():
    # fine-tuned: seed-average per cell
    ft_rows = load_rows(FT)
    by = defaultdict(list)
    for r in ft_rows:
        by[(r["target"], r["held_out_region"])].append(r)
    ft = {k: float(np.mean([x["finetune_auc"] for x in v])) for k, v in by.items()}
    ft_std = {k: float(np.std([x["finetune_auc"] for x in v])) for k, v in by.items()}
    seeds = sorted({r["seed"] for r in ft_rows})

    # frozen encoder + RF baseline
    enc = {(c["target"], c["held_out_region"]): c for c in load_rows(ENC)}

    # assemble matched cells (need rf_auc + fine-tune + frozen)
    cells = []
    for k, c in enc.items():
        if c.get("rf_auc") is None or k not in ft:
            continue
        cells.append(dict(target=k[0], region=k[1], rf=c["rf_auc"],
                          frozen=c["encoder_auc"], ft=ft[k], ft_std=ft_std[k],
                          xgb=c.get("xgb_auc")))
    print(f"matched {len(cells)} cells | seeds {seeds} | "
          f"median fine-tune seed std {np.median([c['ft_std'] for c in cells]):.3f}")

    def sub(sel):
        return [c for c in cells if c["target"] in sel]

    results = {"n_cells": len(cells), "seeds": seeds,
               "median_seed_std": float(np.median([c["ft_std"] for c in cells]))}

    # ---- benchmark-wide + mechanism groups, fine-tuned vs RF ----
    print("\n=== fine-tuned vs Random Forest ===")
    for name, sel in [("ALL_47", None), ("REDOX_AsFeMnPO4", REDOX),
                      ("CONSERVATIVE_NO3F", CONSERVATIVE), ("URANIUM", URANIUM),
                      ("ALL_non_uranium", REDOX | CONSERVATIVE | {"Mn"})]:
        cc = cells if sel is None else sub(sel)
        st = paired([c["ft"] for c in cc], [c["rf"] for c in cc])
        results[f"ft_vs_rf__{name}"] = st
        if st["n"]:
            print(f"  {name:20s} n={st['n']:2d}  FT {st['mean_a']:.3f} RF {st['mean_b']:.3f}  "
                  f"Δ{st['delta']:+.3f}  win {st['wins']}/{st['n']}  "
                  f"p={st['wilcoxon_p']:.4f}  {'SIG' if st['significant'] else 'ns'}")

    # ---- frozen vs RF for the same groups (shows frozen does NOT win) ----
    print("\n=== frozen encoder vs Random Forest (frozen ties RF; fine-tuning unlocks) ===")
    for name, sel in [("REDOX_AsFeMnPO4", REDOX), ("CONSERVATIVE_NO3F", CONSERVATIVE)]:
        cc = sub(sel)
        st = paired([c["frozen"] for c in cc], [c["rf"] for c in cc])
        results[f"frozen_vs_rf__{name}"] = st
        print(f"  {name:20s} n={st['n']:2d}  frozen {st['mean_a']:.3f} RF {st['mean_b']:.3f}  "
              f"Δ{st['delta']:+.3f}  p={st['wilcoxon_p']:.4f}  {'SIG' if st['significant'] else 'ns'}")

    # ---- fine-tuned vs frozen (fine-tuning genuinely helps) ----
    st = paired([c["ft"] for c in cells], [c["frozen"] for c in cells])
    results["ft_vs_frozen__ALL"] = st
    print(f"\n=== fine-tuned vs frozen (ALL 47): Δ{st['delta']:+.3f} "
          f"p={st['wilcoxon_p']:.4f} {'SIG' if st['significant'] else 'ns'} ===")

    # ---- per-target advantage table ----
    print("\n=== per-target Δ(fine-tuned − RF) ===")
    per_t = {}
    for t in ["PO4", "As", "Fe", "Mn", "F", "NO3", "U"]:
        cc = sub({t})
        if not cc:
            continue
        st = paired([c["ft"] for c in cc], [c["rf"] for c in cc])
        per_t[t] = st
        print(f"  {t:4s} n={st['n']:2d}: FT {st['mean_a']:.3f} RF {st['mean_b']:.3f} "
              f"Δ{st['delta']:+.3f}  win {st['wins']}/{st['n']}  p={st['wilcoxon_p']:.3f}")
    results["per_target"] = per_t

    json.dump(results, open("results/redox_dissociation.json", "w"), indent=2)
    print("\nSaved results/redox_dissociation.json")


if __name__ == "__main__":
    main()
