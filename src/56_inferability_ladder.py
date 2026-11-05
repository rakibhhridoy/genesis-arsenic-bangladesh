#!/usr/bin/env python3
"""What can routine monitoring infer? An inferability ladder for all 20 parameters.

For every target parameter, the masked model and a matched gradient-boosting baseline
predict it from progressively richer observed sets:

  L0  rasters only (what a global hazard map sees)
  L1  + field measurements: pH, EC
  L2  + major ions: Ca, Mg, Na, K, Cl, HCO3, SO4, TDS
  L3  + routine lab: NO3, F, SiO2, DOC
  L4  every other parameter (the leave-one-out probe of src/22)

Unobserved parameters are DROPPED (padded out), not masked, so the model is never asked to
predict them. That matches how the corpus is naturally incomplete.

Test data must be unseen. Pretraining used random_split(seed 42, 50,000 validation samples)
on the full corpus at SAMPLE level, so the same site can sit in both halves. The test set
here is the validation samples whose site has NO training sample (site-disjoint). The
Bangladesh held-out set, never trained on, is a second test set. Within a target, every
level is scored on the same samples: those where the target and all L2 majors are present.
Scores are R2 in the model's own normalised space (log1p where flagged, then z-scored).

  python3 src/56_inferability_ladder.py --max_test 3000
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model.genesis_encoder import (GENESISForMGM, GENESIS_PARAMS, AUX_FEATURES,
                                   PARAM_TO_ID, MASK_TOKEN, PAD_TOKEN)
from model.dataset import GENESISDataset, load_stats

MASK_ID, PAD_ID = PARAM_TO_ID[MASK_TOKEN], PARAM_TO_ID[PAD_TOKEN]
FIELD = ["pH", "EC"]
MAJORS = ["Ca", "Mg", "Na", "K", "Cl", "HCO3", "SO4", "TDS"]
ROUTINE = ["NO3", "F", "SiO2", "DOC"]
LEVELS = {"L0": [], "L1": FIELD, "L2": FIELD + MAJORS, "L3": FIELD + MAJORS + ROUTINE,
          "L4": GENESIS_PARAMS}
REQUIRED = ["Ca", "Mg", "Na", "Cl", "HCO3", "SO4"]   # complete-majors condition for test rows


def r2(y, p):
    return float(1 - np.mean((y - p) ** 2) / np.var(y)) if np.var(y) > 1e-9 else np.nan


def boot(y, pf, pg, n=500, seed=0):
    """95% bootstrap intervals for FM R2, GB R2 and their difference (paired resampling)."""
    rng = np.random.default_rng(seed)
    f, g = [], []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        f.append(r2(y[s], pf[s])); g.append(r2(y[s], pg[s]))
    f, g = np.array(f), np.array(g)
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return q(f), q(g), q(f - g)


def znorm(chem, ns):
    z = chem.astype(np.float64).copy()
    lm = np.asarray(ns["chem_log_mask"], bool)
    z[:, lm] = np.log1p(np.clip(z[:, lm], 0, None))
    return (z - ns["chem_mean"]) / ns["chem_std"]


def aux_z(aux, ns):
    return (aux - ns["aux_mean"]) / ns["aux_std"]


@torch.no_grad()
def fm_predict(model, ds, rows, target, allowed, bs=256):
    tid, keep = PARAM_TO_ID[target], {PARAM_TO_ID[p] for p in allowed if p != target}
    chem_ids = {PARAM_TO_ID[p] for p in GENESIS_PARAMS}
    out = []
    for b in range(0, len(rows), bs):
        P, V, PAD, O, M = [], [], [], [], []
        for i in rows[b:b + bs]:
            s = ds[int(i)]
            o, ov, pad = s["original_param_ids"].clone(), s["original_values"].clone(), s["padding_mask"].clone()
            pid, val, mp = o.clone(), ov.clone(), torch.zeros_like(pad)
            for k in range(len(o)):
                t = int(o[k])
                if t == tid:
                    pid[k], val[k], mp[k] = MASK_ID, 0.0, True
                elif t in chem_ids and t not in keep:
                    pid[k], val[k], pad[k], o[k] = PAD_ID, 0.0, True, PAD_ID
            P.append(pid); V.append(val); PAD.append(pad); O.append(o); M.append(mp)
        pr = model(torch.stack(P), torch.stack(V), torch.stack(PAD), torch.stack(O), torch.stack(M))
        out.append(pr[target].detach().cpu().numpy().ravel())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT/"base_full/checkpoints/base_stage2/best.pt"))
    ap.add_argument("--size", default="base")
    ap.add_argument("--max_test", type=int, default=3000)
    ap.add_argument("--n_train", type=int, default=150000)
    ap.add_argument("--out", default=str(ROOT/"results/inferability_ladder.json"))
    a = ap.parse_args()
    torch.set_num_threads(8)

    ns = load_stats(str(ROOT/"data/processed/normalization_stats.json"))
    chem = torch.load(ROOT/"data/processed/genesis_pretrain.pt", map_location="cpu", weights_only=False).numpy()
    aux = torch.load(ROOT/"data/processed/genesis_pretrain_aux.pt", map_location="cpu", weights_only=False)
    aux = (aux["aux"] if isinstance(aux, dict) else aux)
    aux = aux.numpy() if hasattr(aux, "numpy") else aux
    meta = pd.read_parquet(ROOT/"data/processed/genesis_pretrain_meta.parquet",
                           columns=["site_id", "lat", "lon", "water_body_type"])
    N = len(chem)
    n_val = min(50000, max(1, int(N * 0.05)))
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(42)).numpy()
    tr_idx, va_idx = perm[:N - n_val], perm[N - n_val:]
    train_sites = set(meta.site_id.values[tr_idx])
    gw = (meta.water_body_type.values == "groundwater")
    test = np.array([i for i in va_idx if gw[i] and meta.site_id.values[i] not in train_sites])
    print(f"corpus {N:,}; val {len(va_idx):,}; site-disjoint groundwater val {len(test):,}", flush=True)

    ds = GENESISDataset(chem, ns, meta_df=meta, aux_values=aux, mask_ratio=0.0)
    bchem = torch.load(ROOT/"data/processed/genesis_held_out_bd.pt", map_location="cpu", weights_only=False).numpy()
    baux = torch.load(ROOT/"data/processed/genesis_held_out_bd_aux.pt", map_location="cpu", weights_only=False)
    baux = baux.numpy() if hasattr(baux, "numpy") else baux
    bmeta = pd.read_parquet(ROOT/"data/processed/genesis_held_out_bd_meta.parquet")
    bds = GENESISDataset(bchem, ns, meta_df=bmeta, aux_values=baux, mask_ratio=0.0)

    model = GENESISForMGM(a.size)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    model.eval()

    Z, AZ = znorm(chem, ns), aux_z(aux, ns)
    BZ, BAZ = znorm(bchem, ns), aux_z(baux, ns)
    col = {p: i for i, p in enumerate(GENESIS_PARAMS)}
    req = [col[p] for p in REQUIRED]
    rng = np.random.default_rng(0)
    gw_tr = tr_idx[gw[tr_idx]]

    rows = []
    for t in GENESIS_PARAMS:
        ti = col[t]
        tr = gw_tr[np.isfinite(chem[gw_tr, ti])]
        if len(tr) > a.n_train:
            tr = rng.choice(tr, a.n_train, replace=False)
        sets = {}
        for name, Zs, AZs, cs, dset, idx in [
                ("site_disjoint", Z, AZ, chem, ds, test),
                ("bangladesh", BZ, BAZ, bchem, bds, np.arange(len(bchem)))]:
            ok = idx[np.isfinite(cs[idx, ti]) & np.isfinite(cs[idx][:, req]).all(1)]
            if len(ok) > a.max_test:
                ok = rng.choice(ok, a.max_test, replace=False)
            sets[name] = (Zs, AZs, dset, ok)
        for lv, obs in LEVELS.items():
            allowed = [p for p in obs if p != t]
            feats = [col[p] for p in allowed]
            Xtr = np.c_[Z[tr][:, feats], AZ[tr]] if feats else AZ[tr]
            hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1,
                                                random_state=0).fit(Xtr, Z[tr, ti])
            for name, (Zs, AZs, dset, ok) in sets.items():
                if len(ok) < 50:
                    continue
                y = Zs[ok, ti]
                Xte = np.c_[Zs[ok][:, feats], AZs[ok]] if feats else AZs[ok]
                pg = hgb.predict(Xte)
                pf = fm_predict(model, dset, ok, t, allowed)
                cf, cg, cd = boot(y, pf, pg)
                rows.append(dict(target=t, level=lv, test=name, n=int(len(ok)),
                                 fm_r2=r2(y, pf), gb_r2=r2(y, pg),
                                 fm_ci=cf, gb_ci=cg, diff_ci=cd))
                print(f"  {t:5s} {lv} {name:13s} n={len(ok):>5,} FM R2={rows[-1]['fm_r2']:+.3f}  "
                      f"GB R2={rows[-1]['gb_r2']:+.3f}", flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
