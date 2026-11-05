#!/usr/bin/env python3
"""Which few measurements must be added to field data to infer each contaminant?

Greedy forward selection with the masked model. Start from the field set (pH, EC); at each
step add the candidate parameter that most raises reconstruction R2 of the target; stop
at k = 4 additions. This yields a per-contaminant measurement rule.

Selection and scoring use DIFFERENT samples, so the reported R2 is not inflated by the
search:
  selection set - validation samples from sites that ALSO appear in training (plentiful,
                  but leaky, so used only to choose)
  scoring set   - the site-disjoint groundwater validation samples of src/56 (clean)
The matched gradient-boosting baseline is refit for each final set and scored on the same
clean samples, so the rule is not an artefact of the masked model.

  python3 src/57_minimum_measurement_set.py
"""
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("l56", ROOT/"src/56_inferability_ladder.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
TARGETS = ["As", "Fe", "Mn", "U", "F", "NO3", "PO4"]
BASE = ["pH", "EC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n_select", type=int, default=800)
    ap.add_argument("--n_train", type=int, default=150000)
    ap.add_argument("--out", default=str(ROOT/"results/minimum_measurement_set.json"))
    a = ap.parse_args()
    torch.set_num_threads(8)

    ns = L.load_stats(str(ROOT/"data/processed/normalization_stats.json"))
    chem = torch.load(ROOT/"data/processed/genesis_pretrain.pt", map_location="cpu", weights_only=False).numpy()
    aux = torch.load(ROOT/"data/processed/genesis_pretrain_aux.pt", map_location="cpu", weights_only=False)
    aux = (aux["aux"] if isinstance(aux, dict) else aux)
    aux = aux.numpy() if hasattr(aux, "numpy") else aux
    meta = pd.read_parquet(ROOT/"data/processed/genesis_pretrain_meta.parquet",
                           columns=["site_id", "lat", "lon", "water_body_type"])
    N = len(chem); n_val = min(50000, int(N * 0.05))
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(42)).numpy()
    tr_idx, va_idx = perm[:N - n_val], perm[N - n_val:]
    train_sites = set(meta.site_id.values[tr_idx])
    gw = meta.water_body_type.values == "groundwater"
    in_tr = np.array([meta.site_id.values[i] in train_sites for i in va_idx])
    clean = va_idx[gw[va_idx] & ~in_tr]
    leaky = va_idx[gw[va_idx] & in_tr]
    print(f"selection pool {len(leaky):,} (leaky) | scoring pool {len(clean):,} (site-disjoint)", flush=True)

    ds = L.GENESISDataset(chem, ns, meta_df=meta, aux_values=aux, mask_ratio=0.0)
    model = L.GENESISForMGM("base")
    ck = torch.load(ROOT/"base_full/checkpoints/base_stage2/best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck), strict=False); model.eval()
    Z, AZ = L.znorm(chem, ns), L.aux_z(aux, ns)
    col = {p: i for i, p in enumerate(L.GENESIS_PARAMS)}
    req = [col[p] for p in L.REQUIRED]
    rng = np.random.default_rng(0)
    gw_tr = tr_idx[gw[tr_idx]]

    out = []
    for t in TARGETS:
        ti = col[t]
        full = lambda idx: idx[np.isfinite(chem[idx, ti]) & np.isfinite(chem[idx][:, req]).all(1)]
        sel, sco = full(leaky), full(clean)
        if len(sel) > a.n_select:
            sel = rng.choice(sel, a.n_select, replace=False)
        cand = [p for p in L.GENESIS_PARAMS if p not in BASE + [t]]
        chosen, path = list(BASE), []
        for step in range(a.k):
            best = max(cand, key=lambda c: L.r2(Z[sel, ti], L.fm_predict(model, ds, sel, t, chosen + [c])))
            chosen.append(best); cand.remove(best)
            fm = L.r2(Z[sco, ti], L.fm_predict(model, ds, sco, t, chosen))
            tr = gw_tr[np.isfinite(chem[gw_tr, ti])]
            tr = rng.choice(tr, min(a.n_train, len(tr)), replace=False)
            feats = [col[p] for p in chosen]
            gb = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(
                np.c_[Z[tr][:, feats], AZ[tr]], Z[tr, ti])
            gbr = L.r2(Z[sco, ti], gb.predict(np.c_[Z[sco][:, feats], AZ[sco]]))
            path.append(dict(step=step + 1, added=best, fm_r2=fm, gb_r2=gbr))
            print(f"  {t:4s} +{best:5s} -> clean FM R2 {fm:+.3f}  GB R2 {gbr:+.3f}  (n={len(sco)})", flush=True)
        base_fm = L.r2(Z[sco, ti], L.fm_predict(model, ds, sco, t, BASE))
        out.append(dict(target=t, n_select=int(len(sel)), n_score=int(len(sco)),
                        base_fm_r2=base_fm, path=path))
        json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
