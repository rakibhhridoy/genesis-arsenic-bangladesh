"""Precompute data for the redesigned Fig 4 panels (a) and (d), into a small JSON
so the composite generator (src/41) stays fast (no 2 GB tensor reload).

(a) Sample-level PCA scatter: replicate the src/14 distribution-shift recipe
    (16 shared params, log1p except pH/Eh, pretrain-median impute, pretrain
    standardize, PCA(4) fit on pretrain, project both), subsample corpus for
    plotting, and store PC1/PC2 coordinates for corpus + Bangladesh.
(d) Predicted-probability histogram for the chemistry-only LogReg on the BD
    transfer set, reconstructed from the released 10 quantile-bin calibration
    summary into fixed-width bins (predictions pile up near 1.0).

Writes: results/fig4_panels.json
"""
import json, sys
from pathlib import Path
import numpy as np, torch, pandas as pd
sys.path.insert(0, "src")
from model.genesis_encoder import GENESIS_PARAMS

BD_LOW_COVERAGE = {"F", "U", "SiO2", "DOC"}
NON_CONC = {"pH", "Eh"}
N_PLOT_CORPUS = 4000


def log_t(x, name):
    if name in NON_CONC:
        return x
    return np.log1p(np.clip(x, 0, None))


def main():
    dd = Path("data/processed")
    pre = torch.load(dd / "genesis_pretrain.pt", weights_only=True).numpy()
    bd = torch.load(dd / "genesis_held_out_bd.pt", weights_only=True).numpy()
    kept = [p for p in GENESIS_PARAMS if p not in BD_LOW_COVERAGE]
    idx = [GENESIS_PARAMS.index(p) for p in kept]
    pre_X, bd_X = pre[:, idx], bd[:, idx]

    pre_log = np.column_stack([log_t(pre_X[:, i], n) for i, n in enumerate(kept)])
    bd_log = np.column_stack([log_t(bd_X[:, i], n) for i, n in enumerate(kept)])
    med = np.nanmedian(pre_log, axis=0)
    pre_f = np.where(np.isnan(pre_log), med, pre_log)
    bd_f = np.where(np.isnan(bd_log), med, bd_log)
    mu, sd = pre_f.mean(0), pre_f.std(0) + 1e-9
    pre_z, bd_z = (pre_f - mu) / sd, (bd_f - mu) / sd

    # PCA fit on pretrain
    Xc = pre_z - pre_z.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:4]
    pre_pc = (pre_z - pre_z.mean(0)) @ comps.T
    bd_pc = (bd_z - pre_z.mean(0)) @ comps.T

    rng = np.random.default_rng(0)
    sub = rng.choice(len(pre_pc), min(N_PLOT_CORPUS, len(pre_pc)), replace=False)
    out = {
        "pca_scatter": {
            "corpus_pc1": pre_pc[sub, 0].round(3).tolist(),
            "corpus_pc2": pre_pc[sub, 1].round(3).tolist(),
            "bd_pc1": bd_pc[:, 0].round(3).tolist(),
            "bd_pc2": bd_pc[:, 1].round(3).tolist(),
        }
    }

    # (d) predicted-prob histogram from the quantile-bin calibration summary
    cal = json.load(open("results/calibration_logreg_as_chemonly.json"))
    rb = cal["reliability_quantile_10bin"]
    means = np.array(rb["bin_mean_predicted_prob"])
    counts = np.array(rb["bin_counts"])
    edges = np.linspace(0, 1, 11)
    hist = np.zeros(10)
    for m, c in zip(means, counts):
        b = min(int(m * 10), 9)
        hist[b] += c
    out["pred_hist"] = {"edges": edges.round(2).tolist(), "counts": hist.astype(int).tolist(),
                        "n": int(counts.sum())}
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/fig4_panels.json", "w"), indent=1)
    print("wrote results/fig4_panels.json")
    print(f"  corpus scatter pts {len(sub)}, BD pts {len(bd_pc)}")
    print(f"  pred-prob hist (fixed 0.1 bins): {hist.astype(int).tolist()}")


if __name__ == "__main__":
    main()
