#!/usr/bin/env python3
"""Has the masked model internalised equilibrium thermodynamics, and where does it stop?

Each major ion (and pH, F) is reconstructed with that one parameter masked and all others
observed (the L4 setting of src/56). The reconstructed composition is then checked against
two thermodynamic constraints the model was never told about:
  - electroneutrality: charge-balance error (CBE, meq) of the reconstructed water
  - mineral equilibrium: saturation indices of calcite, gypsum and fluorite
Same formulas for observed and reconstructed water, so approximation error cancels in the
comparison: Davies activities at 25 C, free-ion speciation without ion pairs, CO3 from
HCO3 and pH (pK2 = 10.33), log Ksp calcite -8.48, gypsum -4.58, fluorite -10.6.

Samples are split oxic vs reducing with a reduced McMahon & Chapelle (2008) rule (no DO in
the vocabulary): reducing if Fe >= 0.1 mg/L or Mn >= 0.05 mg/L. Bangladesh is reported
separately. Null: reconstructions shuffled across samples within the same set.
Test rows: site-disjoint groundwater validation samples (as src/56) with complete majors.

  python3 src/58_thermodynamic_consistency.py
"""
import importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("l56", ROOT/"src/56_inferability_ladder.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
IONS = {"Ca": (40.08, 2), "Mg": (24.31, 2), "Na": (22.99, 1), "K": (39.10, 1),
        "Cl": (35.45, -1), "HCO3": (61.02, -1), "SO4": (96.06, -2), "F": (19.00, -1)}
RECON = ["Ca", "Mg", "Na", "K", "Cl", "HCO3", "SO4", "F", "pH"]
LOGK = {"calcite": -8.48, "gypsum": -4.58, "fluorite": -10.6}


def inv(z, i, ns):
    v = z * ns["chem_std"][i] + ns["chem_mean"][i]
    return np.expm1(v) if ns["chem_log_mask"][i] else v


def thermo(d):
    """d: dict of arrays in mg/L (pH unitless). Returns CBE and SIs."""
    mol = {k: d[k] / (IONS[k][0] * 1000) for k in IONS}
    f_mol = mol["F"]
    mol["F"] = np.nan_to_num(f_mol)          # minor ion: absent F counts as zero in I and CBE
    I = 0.5 * sum(mol[k] * IONS[k][1] ** 2 for k in IONS)
    s = np.sqrt(np.clip(I, 0, None))
    lg = lambda z: -0.509 * z ** 2 * (s / (1 + s) - 0.3 * I)
    la = {k: np.log10(np.clip(mol[k], 1e-12, None)) + lg(IONS[k][1]) for k in IONS}
    la["F"] = np.log10(np.clip(f_mol, 1e-12, None)) + lg(1)   # NaN where F unmeasured
    la_co3 = la["HCO3"] + d["pH"] - 10.33 + lg(2) - lg(1)
    cat = sum(mol[k] * IONS[k][1] for k in IONS if IONS[k][1] > 0) * 1000
    an = -sum(mol[k] * IONS[k][1] for k in IONS if IONS[k][1] < 0) * 1000
    return {"CBE": 100 * (cat - an) / (cat + an),
            "SI_calcite": la["Ca"] + la_co3 - LOGK["calcite"],
            "SI_gypsum": la["Ca"] + la["SO4"] - LOGK["gypsum"],
            "SI_fluorite": la["Ca"] + 2 * la["F"] - LOGK["fluorite"]}


def evaluate(name, chem, Z, ds, idx, model, ns, col, rng):
    obs = {p: chem[idx, col[p]].astype(float) for p in RECON}
    rec = {}
    for p in RECON:
        have = np.isfinite(chem[idx, col[p]])
        rec[p] = np.full(len(idx), np.nan)
        rec[p][have] = inv(L.fm_predict(model, ds, idx[have], p, L.GENESIS_PARAMS), col[p], ns)
    T_obs, T_rec = thermo(obs), thermo(rec)
    sh = rng.permutation(len(idx))
    T_null = thermo({p: rec[p][sh] for p in RECON})
    fe, mn = chem[idx, col["Fe"]], chem[idx, col["Mn"]]
    red = (np.nan_to_num(fe) >= 0.1) | (np.nan_to_num(mn) >= 0.05)
    res = {"set": name, "n": int(len(idx)), "n_reducing": int(red.sum())}
    for cls, m in [("all", np.ones(len(idx), bool)), ("oxic", ~red), ("reducing", red)]:
        if m.sum() < 30:
            continue
        r = {"n": int(m.sum()),
             "abs_CBE_obs": float(np.nanmedian(np.abs(T_obs["CBE"][m]))),
             "abs_CBE_rec": float(np.nanmedian(np.abs(T_rec["CBE"][m]))),
             "abs_CBE_null": float(np.nanmedian(np.abs(T_null["CBE"][m])))}
        for k in ["SI_calcite", "SI_gypsum", "SI_fluorite"]:
            o, p, nl = T_obs[k][m], T_rec[k][m], T_null[k][m]
            ok = np.isfinite(o) & np.isfinite(p)
            if ok.sum() < 30:
                continue
            r[k] = {"r2": L.r2(o[ok], p[ok]), "r2_null": L.r2(o[ok], nl[ok]),
                    "mae": float(np.mean(np.abs(o[ok] - p[ok]))),
                    "sat_agree": float(np.mean((o[ok] > 0) == (p[ok] > 0))), "n": int(ok.sum())}
        res[cls] = r
    return res


def main():
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
    clean = np.array([i for i in va_idx if gw[i] and meta.site_id.values[i] not in train_sites])
    col = {p: i for i, p in enumerate(L.GENESIS_PARAMS)}
    need = [col[p] for p in RECON if p != "F"]
    clean = clean[np.isfinite(chem[clean][:, need]).all(1)]

    model = L.GENESISForMGM("base")
    ck = torch.load(ROOT/"base_full/checkpoints/base_stage2/best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck), strict=False); model.eval()
    ds = L.GENESISDataset(chem, ns, meta_df=meta, aux_values=aux, mask_ratio=0.0)
    bchem = torch.load(ROOT/"data/processed/genesis_held_out_bd.pt", map_location="cpu", weights_only=False).numpy()
    baux = torch.load(ROOT/"data/processed/genesis_held_out_bd_aux.pt", map_location="cpu", weights_only=False)
    baux = baux.numpy() if hasattr(baux, "numpy") else baux
    bmeta = pd.read_parquet(ROOT/"data/processed/genesis_held_out_bd_meta.parquet")
    bds = L.GENESISDataset(bchem, ns, meta_df=bmeta, aux_values=baux, mask_ratio=0.0)
    bidx = np.where(np.isfinite(bchem[:, need]).all(1))[0]
    rng = np.random.default_rng(0)
    if len(bidx) > 2000:
        bidx = rng.choice(bidx, 2000, replace=False)

    out = [evaluate("site_disjoint", chem, None, ds, clean, model, ns, col, rng),
           evaluate("bangladesh", bchem, None, bds, bidx, model, ns, col, rng)]
    json.dump(out, open(ROOT/"results/thermodynamic_consistency.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
