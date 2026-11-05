"""Supplementary tables for the evaluation reframe, written as LaTeX from the result JSONs.

Every table is generated, never typed, so the SI cannot drift from the results. Outputs go to
RewriteGenesisRewrite/Evaluation/latex/si_tables/<name>.tex and are pulled into the SI by
build_si.py through {{TABLE:<name>}} markers in draft_si.md.

  inferability_site    R2 at L2 and L4 per target, four regressors, site-disjoint wells
  inferability_bd      R2 at L2 and L4 per target, four regressors, Bangladesh
  measurement_sets     greedy selection path per contaminant, GENESIS and gradient boosting R2
  loro_auc             out-of-region AUC per cell, eight methods (longtable)
  loro_coverage        out-of-region exceedance coverage per cell, eight methods (longtable)
  hazard_audit         raster-only RF AUC under random, block and LORO validation per parameter
  hazard_controls      within-region and within-source Spearman controls
  thermo               saturation-index R2 and |CBE| per regressor, set and redox class
  recalibration        coverage and ambiguity after local recalibration with k labels
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "RewriteGenesisRewrite/Evaluation/latex/si_tables"
OUT.mkdir(parents=True, exist_ok=True)

ENT = [("finetune_base", "GEN-FT"), ("frozen_base", "GEN-Fr"), ("tabpfn", "TabPFN"),
       ("xgb", "XGB"), ("rf", "RF"), ("logreg", "LogReg"), ("rf_aux", "Raster"),
       ("knn_space", "kNN")]
REG = [("fm", "GENESIS"), ("gb", "GB"), ("rf", "RF"), ("tabpfn", "TabPFN"), ("qkernel", "QK")]
CHEMTEX = {"NO3": "NO$_3$", "SO4": "SO$_4$", "HCO3": "HCO$_3$", "PO4": "PO$_4$", "SiO2": "SiO$_2$"}


def chem(p):
    return CHEMTEX.get(p, p)


def region(r):
    return r.replace("GEMStat:", "G:").replace(" (-the )", "").replace("EU", "EEA")


def f(x, d=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def write(name, tex):
    (OUT / f"{name}.tex").write_text(tex)
    print("  ->", f"si_tables/{name}.tex")


def tabular(cols, header, rows, long=False):
    env = "longtable" if long else "tabular"
    head = " & ".join(header) + r" \\" + "\n\\midrule"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    if long:
        return (f"\\begin{{longtable}}{{{cols}}}\n\\toprule\n{head}\n\\endhead\n"
                f"{body}\n\\bottomrule\n\\end{{longtable}}")
    return f"\\begin{{{env}}}{{{cols}}}\n\\toprule\n{head}\n{body}\n\\bottomrule\n\\end{{{env}}}"


# ---------------------------------------------------------------- inferability
lad = json.load(open(RES / "inferability_ladder.json"))
ladb = json.load(open(RES / "inferability_ladder_baselines.json"))["rows"]
R2, N = {}, {}
for r in lad:
    R2[("fm", r["level"], r["test"], r["target"])] = r["fm_r2"]
    N[(r["test"], r["target"])] = r["n"]
for r in ladb:
    R2[(r["method"], r["level"], r["test"], r["target"])] = r["r2"]
QREG = json.load(open(RES / "inferability_ladder_quantum.json"))["rows"]   # src/68
for r in QREG:
    R2[(r["method"], r["level"], r["test"], r["target"])] = r["r2"]
ORDER = ["Ca", "Mg", "Na", "K", "Cl", "HCO3", "SO4", "EC", "TDS", "pH", "SiO2", "F", "NO3",
         "PO4", "DOC", "Fe", "Mn", "U", "As", "Eh"]
for test, name in (("site_disjoint", "inferability_site"), ("bangladesh", "inferability_bd")):
    rows = []
    for t in ORDER:
        if ("fm", "L4", test, t) not in R2:
            continue
        vals = []
        for lv in ("L2", "L4"):
            v = {m: R2.get((m, lv, test, t)) for m, _ in REG}
            best = max(x for x in v.values() if x is not None)
            vals += [(f"\\textbf{{{f(v[m])}}}" if v[m] == best else f(v[m])) for m, _ in REG]
        rows.append([chem(t), str(N[(test, t)])] + vals)
    hdr = ["Target", "$n$"] + [f"{lab} L2" for _, lab in REG] + [f"{lab} L4" for _, lab in REG]
    write(name, "\\small\n" + tabular("lr" + "r" * (2 * len(REG)), hdr, rows))

ms = json.load(open(RES / "minimum_measurement_set.json"))
rows = []
for r in ms:
    path = r["path"]
    rows.append([chem(r["target"]), str(r["n_score"]), f(r["base_fm_r2"]),
                 " $\\to$ ".join(chem(p["added"]) for p in path),
                 " / ".join(f(p["fm_r2"]) for p in path),
                 " / ".join(f(p["gb_r2"]) for p in path)])
write("measurement_sets", "\\small\n" + tabular(
    "lrrp{3.4cm}p{3.2cm}p{3.2cm}",
    ["Target", "$n$", "pH+EC", "Added in order", "GENESIS $R^2$ per step", "GB $R^2$ per step"],
    rows))

# ---------------------------------------------------------------- LORO cells
conf = json.load(open(RES / "conformal_coverage.json"))["methods"]
cells = {k: {(c["target"], c["region"]): c for c in conf[k]["cells"]} for k, _ in ENT}
keys = sorted(cells["rf"], key=lambda k: (["As", "Fe", "Mn", "PO4", "NO3", "F", "U"].index(k[0]),
                                          k[1]))
for metric, name in (("auc", "loro_auc"), ("cov", "loro_coverage")):
    rows = []
    for k in keys:
        c0 = cells["rf"][k]
        vals = []
        for m, _ in ENT:
            c = cells[m].get(k)
            vals.append(f(c["auc_out"] if metric == "auc" else c["out"]["cov_pos"]) if c else "--")
        rows.append([chem(k[0]), region(k[1]), str(c0["n_out"]), str(c0["n_pos_out"])] + vals)
    mean = ["\\multicolumn{4}{l}{Mean of 47 cells}"] + [
        f(np.mean([(c["auc_out"] if metric == "auc" else c["out"]["cov_pos"])
                   for c in cells[m].values()])) for m, _ in ENT]
    rows.append(mean)
    write(name, "\\scriptsize\n" + tabular("llrr" + "r" * 8,
                                            ["Target", "Region", "$n$", "Exc."] +
                                            [lab for _, lab in ENT], rows, long=True))

rows = []
for m, lab in ENT:
    S = conf[m]["summary"]["all"]
    rows.append([lab, f(S["cov_pos_out"])] + [f"{f(S[f'local{k}_cov_pos'])} ({f(S[f'local{k}_size'] - 1)})"
                                              for k in (25, 50, 100, 200)])
write("recalibration", "\\small\n" + tabular(
    "lrrrrr", ["Method", "No recal.", "$k=25$", "$k=50$", "$k=100$", "$k=200$"], rows))

# ---------------------------------------------------------------- hazard audit
haz = [r for r in json.load(open(RES / "hazard_transfer_gradient.json")) if r["mode"] == "p80"]
P = defaultdict(lambda: defaultdict(list))
for r in haz:
    P[r["param"]]["r2"] = r["r2"]
    for k in ("random", "block1", "block5", "block10", "loro_mean"):
        if r[k] is not None and r[k] == r[k]:
            P[r["param"]][k].append(r[k])
rows = []
for p in sorted(P, key=lambda p: P[p]["r2"]):
    d = P[p]
    m = {k: np.mean(d[k]) for k in ("random", "block1", "block5", "block10", "loro_mean")}
    rows.append([chem(p), f(d["r2"]), f(m["random"]), f(m["block1"]), f(m["block5"]),
                 f(m["block10"]), f(m["loro_mean"]), f(m["random"] - m["loro_mean"])])
write("hazard_audit", "\\small\n" + tabular(
    "lrrrrrrr", ["Parameter", "Recon. $R^2$", "Random", "Block 1°", "Block 5°", "Block 10°",
                 "LORO", "Gap"], rows))

ctl = json.load(open(RES / "transfer_gradient_controls.json"))
rows = [["Across parameters, random CV", "all regions", f(ctl["across_params"]["random_all"]["rho"]),
         f(ctl["across_params"]["random_all"]["perm_p"], 4), "20"],
        ["Across parameters, LORO", "all regions", f(ctl["across_params"]["loro_mean_all"]["rho"]),
         f(ctl["across_params"]["loro_mean_all"]["perm_p"], 4), "20"]]
for R, v in sorted(ctl["within_region"].items()):
    rows.append(["Within held-out region", region(R), f(v["rho"]), f(v["p"], 3), str(v["n_params"])])
for src in ("France", "USA"):
    v = ctl["within_source"][f"{src}_block_all"]
    rows.append(["Within one source, block hold-out", src, f(v["rho"]), f(v["perm_p"], 4), str(v["n"])])
write("hazard_controls", "\\small\n" + tabular(
    "llrrr", ["Control", "Scope", "Spearman $\\rho$", "$p$", "Parameters"], rows))

# ---------------------------------------------------------------- thermodynamics
fm = {r["set"]: r for r in json.load(open(RES / "thermodynamic_consistency.json"))}
tb = json.load(open(RES / "thermodynamic_consistency_baselines.json"))
TH = {"fm": fm, **{k: {r["set"]: r for r in v} for k, v in tb.items()}}
TQ = json.load(open(RES / "thermodynamic_consistency_quantum.json"))      # src/68
TH.update({k: {r["set"]: r for r in v} for k, v in TQ.items()})
rows = []
for st, cls, lab in (("site_disjoint", "all", "Unseen sites, all"),
                     ("site_disjoint", "oxic", "Unseen sites, oxic"),
                     ("site_disjoint", "reducing", "Unseen sites, reducing"),
                     ("bangladesh", "all", "Bangladesh")):
    for m, ml in REG:
        d = TH[m][st][cls]
        rows.append([lab if m == "fm" else "", ml, str(d["n"]), f(d["SI_calcite"]["r2"]),
                     f(d["SI_gypsum"]["r2"]), f(d["SI_fluorite"]["r2"]),
                     f(d["SI_calcite"]["sat_agree"]), f(d["abs_CBE_rec"], 1), f(d["abs_CBE_obs"], 1)])
write("thermo", "\\small\n" + tabular(
    "llrrrrrrr", ["Test set", "Method", "$n$", "Calcite", "Gypsum", "Fluorite", "Cal. state",
                  "|CBE| rec.", "|CBE| obs."], rows))

# ---------------------------------------------------------------- quantum kernel (src/65, src/66)
def fp(x):
    return "$<$0.0001" if x < 1e-4 else f(x, 4)


qk = json.load(open(RES / "quantum_kernel_summary.json"))
QLAB = dict(ENT + [("rbf_svm", "RBF kernel"), ("qkernel_prod", "No entanglement")])
rows = []
for m in ["rbf_svm", "qkernel_prod", "finetune_base", "tabpfn", "xgb", "rf", "frozen_base", "logreg",
          "rf_aux", "knn_space"]:
    for g, gl in (("all", "All"), ("redox", "Redox"), ("conservative", "Conservative")):
        r = qk["pairs"][f"qkernel_vs_{m}_{g}"]
        rows.append([QLAB[m] if g == "all" else "", gl, str(r["n"]), f(r["auc"]["a"], 3),
                     f(r["auc"]["b"], 3), str(r["auc"]["a_wins"]), fp(r["auc"]["p"]),
                     f(r["cov_pos"]["a"], 3), f(r["cov_pos"]["b"], 3), fp(r["cov_pos"]["p"])])
write("quantum", "\\small\n" + tabular(
    "llrrrrrrrr", ["Compared with", "Cells", "$n$", "AUC QK", "AUC other", "QK higher", "$p$",
                   "Cov. QK", "Cov. other", "$p$"], rows))


# ---------------------------------------------------------------- quantum kernel, regression (src/68)
from scipy.stats import wilcoxon
QM = [("qkernel", "QK"), ("qkernel_prod", "No ent."), ("rbf_svm", "RBF")]
rows = []
for test, tl in (("site_disjoint", "Unseen sites"), ("bangladesh", "Bangladesh")):
    for lv in ("L0", "L1", "L2", "L3", "L4"):
        ts = sorted({t for (m, l, te, t) in R2 if m == "qkernel" and l == lv and te == test})
        v = {m: np.array([R2[(m, lv, test, t)] for t in ts]) for m, _ in QM}
        g = np.array([R2.get(("gb", lv, test, t), np.nan) for t in ts])
        ps = [wilcoxon(v["qkernel"], v[m]).pvalue if np.any(v["qkernel"] != v[m]) else 1.0
              for m in ("rbf_svm", "qkernel_prod")]
        rows.append([tl if lv == "L0" else "", lv, str(len(ts))] +
                    [f(float(np.median(v[m]))) for m, _ in QM] + [f(float(np.nanmedian(g)))] +
                    [f(float(p), 3) for p in ps])
write("quantum_regression", "\\small\n" + tabular(
    "llrrrrrrr", ["Test set", "Level", "Targets", "QK", "No ent.", "RBF", "GB", "$p$ QK--RBF",
                  "$p$ QK--No ent."], rows))
rows = []
for st, lab in (("site_disjoint", "Unseen sites"), ("bangladesh", "Bangladesh")):
    for m, ml in QM:
        d = TQ[m][[r["set"] for r in TQ[m]].index(st)]["all"]
        rows.append([lab if m == "qkernel" else "", ml, str(d["n"]), f(d["SI_calcite"]["r2"]),
                     f(d["SI_gypsum"]["r2"]), f(d["SI_fluorite"]["r2"]),
                     f(d["SI_calcite"]["sat_agree"]), f(d["abs_CBE_rec"], 1), f(d["abs_CBE_obs"], 1)])
write("quantum_thermo", "\\small\n" + tabular(
    "llrrrrrrr", ["Test set", "Kernel", "$n$", "Calcite", "Gypsum", "Fluorite", "Cal. state",
                  "|CBE| rec.", "|CBE| obs."], rows))
