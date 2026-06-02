"""
GENESIS Paper 5 — Step 25: second-domain ROBUSTNESS CHECK (materials science).

NOT IN THE MANUSCRIPT. Kept as a documented, reproducible robustness check.

We tested whether the conditional-pretraining finding generalizes BEYOND water
geochemistry on a different scientific-tabular domain: the UCI Superconductivity
dataset (21,263 compounds; target = critical temperature). `number_of_elements`
gives a natural distribution-shift axis (materials analogue of leave-one-region-
out): high-Tc rate ranges 3% (2-element) to 88% (6-element).

HONEST OUTCOME (2026-06-03): the small pretrained encoder NEVER beats logistic
regression here, in either representation:
  - engineered features (81): baseline AUC 0.81-0.96, encoder mean advantage
    -0.001 (ties; the engineered features already linearize the problem).
  - raw element fractions (86, 95% sparse): encoder mean advantage -0.046 (worse).
This CONFIRMS the negative pole of the law ("in the <=50M-param / ~2M-sample
regime, simple baselines are strong and small-scale pretraining rarely beats
them") in a second real domain, but does NOT reproduce the positive (uranium)
pole, because this domain at ~20k samples lacks a weak-baseline/nonlinear regime
that pretraining can exploit. We therefore did not claim full second-domain
replication in the paper; the controlled synthetic experiment (src/23) carries
the generality evidence. We report both representations below for transparency.

Data: download UCI #464 to /tmp/superconduct/{train,unique_m}.csv.
Usage: python src/25_superconductor_transfer.py
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

CSV = "/tmp/superconduct/train.csv"
OUT = Path("results/superconductor_transfer.json")
THR = 20.0          # critical_temp threshold (K), dataset median
torch.manual_seed(0); np.random.seed(0)


class MaskedAE(nn.Module):
    def __init__(self, d, latent=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, latent), nn.GELU())
        self.dec = nn.Sequential(nn.Linear(latent, 128), nn.GELU(), nn.Linear(128, d))

    def forward(self, x):
        return self.dec(self.enc(x))

    def latent(self, x):
        return self.enc(x)


def pretrain(Xpool, d, epochs=60, mask=0.2):
    m = MaskedAE(d)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    X = torch.tensor(Xpool, dtype=torch.float32)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 512):
            b = X[perm[i:i + 512]]
            msk = (torch.rand_like(b) > mask).float()
            opt.zero_grad()
            loss = ((m(b * msk) - b) ** 2).mean()
            loss.backward(); opt.step()
    m.eval()
    return m


def energy_distance(A, B, max_n=400, rng=None):
    rng = rng or np.random.default_rng(0)
    def sub(M): return M[rng.choice(len(M), min(len(M), max_n), replace=False)]
    A, B = sub(A), sub(B)
    def mpd(X, Y): return np.sqrt(((X[:, None] - Y[None]) ** 2).sum(-1)).mean()
    return float(2 * mpd(A, B) - mpd(A, A) - mpd(B, B))


def run_raw_fractions():
    """Second representation: raw per-element fractions (unique_m.csv, 86 features,
    ~95% sparse) — the weak-baseline regime. Element count = number of nonzero
    fractions. Outcome: encoder mean advantage -0.046 (never wins)."""
    raw = "/tmp/superconduct/unique_m.csv"
    if not Path(raw).exists():
        print("(raw-fraction variant skipped: unique_m.csv not found)")
        return []
    df = pd.read_csv(raw)
    feat = [c for c in df.columns if c not in ("material", "critical_temp")]
    X = df[feat].to_numpy(np.float64)
    y = (df["critical_temp"].to_numpy() > THR).astype(int)
    nel = (X > 0).sum(1)
    out = []
    print("\n[raw element-fraction representation]")
    print(f"{'held-out':10s}{'n_test':>8s}{'base':>8s}{'enc':>8s}{'adv':>8s}")
    for g in sorted(np.unique(nel)):
        te = nel == g; tr = ~te
        if te.sum() < 200 or len(np.unique(y[te])) < 2 or y[te].sum() < 20 or (len(y[te]) - y[te].sum()) < 20:
            continue
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        enc = pretrain(Xtr, len(feat))
        with torch.no_grad():
            Ztr = enc.latent(torch.tensor(Xtr, dtype=torch.float32)).numpy()
            Zte = enc.latent(torch.tensor(Xte, dtype=torch.float32)).numpy()
        ec = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Ztr, y[tr])
        ae = roc_auc_score(y[te], ec.predict_proba(Zte)[:, 1])
        ab = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        ab = roc_auc_score(y[te], ab)
        out.append({"held_out_n_elements": int(g), "n_test": int(te.sum()),
                    "auc_base": float(ab), "auc_enc": float(ae), "advantage": float(ae - ab)})
        print(f"{int(g):>2d}-elem  {int(te.sum()):>8d}{ab:>8.3f}{ae:>8.3f}{ae - ab:>+8.3f}")
    if out:
        adv = np.array([r["advantage"] for r in out])
        print(f"raw-fraction: encoder beats baseline {(adv > 0).sum()}/{len(adv)}; mean advantage {adv.mean():+.3f}")
    return out


def main():
    if not Path(CSV).exists():
        raise SystemExit(f"Missing {CSV} — download UCI #464 superconductivty data first.")
    df = pd.read_csv(CSV)
    feat = [c for c in df.columns if c not in ("critical_temp",)]
    y = (df["critical_temp"].to_numpy() > THR).astype(int)
    groups = df["number_of_elements"].to_numpy()

    results = []
    for g in sorted(np.unique(groups)):
        te = groups == g
        tr = ~te
        yte, ytr = y[te], y[tr]
        if te.sum() < 200 or len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
            continue
        if yte.sum() < 20 or (len(yte) - yte.sum()) < 20:
            continue
        Xtr_raw = df.loc[tr, feat].to_numpy(np.float64)
        Xte_raw = df.loc[te, feat].to_numpy(np.float64)
        sc = StandardScaler().fit(Xtr_raw)
        Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)

        # self-supervised pretraining on the in-distribution pool only
        enc = pretrain(Xtr, len(feat))
        with torch.no_grad():
            Ztr = enc.latent(torch.tensor(Xtr, dtype=torch.float32)).numpy()
            Zte = enc.latent(torch.tensor(Xte, dtype=torch.float32)).numpy()

        # encoder (frozen latent + logreg) vs raw-feature logreg, same split
        ec = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Ztr, ytr)
        auc_enc = roc_auc_score(yte, ec.predict_proba(Zte)[:, 1])
        bc = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        auc_base = roc_auc_score(yte, bc.predict_proba(Xte)[:, 1])
        ed = energy_distance(Xte, Xtr)
        results.append({"held_out_n_elements": int(g), "n_test": int(te.sum()),
                        "pos_rate": float(yte.mean()), "auc_base": float(auc_base),
                        "auc_enc": float(auc_enc), "advantage": float(auc_enc - auc_base),
                        "energy_dist": ed})

    raw_results = run_raw_fractions()
    OUT.parent.mkdir(exist_ok=True)
    json.dump({"engineered_features": results, "raw_fractions": raw_results,
               "note": "Robustness check, NOT in manuscript. Encoder never beats "
                       "logistic regression in either representation; confirms the "
                       "negative pole of the law, does not reproduce the uranium pole."},
              open(OUT, "w"), indent=2)
    print(f"\n[engineered-feature representation]")
    print(f"{'held-out':10s}{'n_test':>8s}{'pos':>7s}{'base':>8s}{'enc':>8s}{'adv':>8s}{'dist':>8s}")
    for r in results:
        print(f"{r['held_out_n_elements']:>2d}-elem  {r['n_test']:>8d}{r['pos_rate']:>7.2f}"
              f"{r['auc_base']:>8.3f}{r['auc_enc']:>8.3f}{r['advantage']:>+8.3f}{r['energy_dist']:>8.2f}")
    adv = np.array([r["advantage"] for r in results])
    print(f"\nencoder beats baseline in {(adv > 0).sum()}/{len(adv)} held-out groups; mean advantage {adv.mean():+.3f}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
