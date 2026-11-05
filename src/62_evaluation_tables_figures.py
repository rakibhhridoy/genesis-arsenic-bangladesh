"""Common results table and one figure per axis for the evaluation reframe.

Reads only finished result JSONs (no model runs):
  A inferability  results/inferability_ladder.json (GENESIS + stored GB)
                  results/inferability_ladder_baselines.json (GB, RF, TabPFN)
  B transfer      results/conformal_coverage.json (8 entrants, AUC out of region)
                  results/hazard_transfer_gradient.json (raster-RF LORO vs R2)
  C calibration   results/conformal_coverage.json (coverage, ambiguity)
  D physics       results/thermodynamic_consistency.json (GENESIS)
                  results/thermodynamic_consistency_baselines.json (GB, RF, TabPFN)

Outputs
  results/evaluation_summary.json, results/evaluation_summary.csv
  figures/fig_eval_A_inferability.{pdf,png} ... fig_eval_D_physics.{pdf,png}
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

import evalstyle as es
INK2 = "#52514e"

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
rng = np.random.default_rng(0)

# method key -> (label, colour, marker), one fixed mapping shared by every figure
ENT = {k: es.METHOD[k] for k in es.CLASSIFIERS + ["qkernel"]}   # qkernel = simulated (src/65), last
REG = {k: es.METHOD[k] for k in es.REGRESSORS + ["qkernel"]}   # qkernel = kernel ridge (src/68), last
GROUP = {"As": "redox", "Fe": "redox", "Mn": "redox", "PO4": "redox", "Eh": "redox",
         "NO3": "major", "U": "uranium", "DOC": "redox",
         "Ca": "major", "Mg": "major", "Na": "major", "K": "major", "Cl": "major",
         "HCO3": "major", "SO4": "major", "SiO2": "major", "F": "major",
         "pH": "bulk", "EC": "bulk", "TDS": "bulk"}
LEVELS = ["L0", "L1", "L2", "L3", "L4"]
LEVEL_LAB = ["rasters", "field", "majors", "lab", "all"]


def boot_ci(x, n=5000):
    x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def save(fig, name):
    es.save(fig, FIG, name)
    plt.close(fig)


# ----------------------------------------------------------------- load
lad = json.load(open(RES / "inferability_ladder.json"))
ladb = json.load(open(RES / "inferability_ladder_baselines.json"))["rows"]
conf = json.load(open(RES / "conformal_coverage.json"))["methods"]
# the simulated quantum kernel lives in its own folder (src/65), scored by the src/60 functions
_m60 = __import__("60_conformal_coverage")
_qrng = np.random.default_rng(0)
_qcells = _m60.seed_mean([_m60.cell_metrics(f, _qrng) for f in sorted(
    (ROOT / "results/conformal_preds_quantum").glob("qkernel_s*/*.npz")) if not f.name.startswith("._")])
conf["qkernel"] = {"cells": _qcells, "summary": _m60.summarise(_qcells)}
haz = json.load(open(RES / "hazard_transfer_gradient.json"))
th_fm = json.load(open(RES / "thermodynamic_consistency.json"))
th_b = json.load(open(RES / "thermodynamic_consistency_baselines.json"))

# R2[(method, level, test)] = {target: r2}
R2 = {}
for r in lad:
    R2.setdefault(("fm", r["level"], r["test"]), {})[r["target"]] = r["fm_r2"]
for r in ladb:
    R2.setdefault((r["method"], r["level"], r["test"]), {})[r["target"]] = r["r2"]
for r in json.load(open(RES / "inferability_ladder_quantum.json"))["rows"]:
    if r["method"] == "qkernel":
        R2.setdefault(("qkernel", r["level"], r["test"]), {})[r["target"]] = r["r2"]


def med(m, lv, test):
    return float(np.median(list(R2[(m, lv, test)].values())))


THERMO = {"fm": {r["set"]: r for r in th_fm}}
for m in ("gb", "rf", "tabpfn"):
    THERMO[m] = {r["set"]: r for r in th_b[m]}
THERMO["qkernel"] = {r["set"]: r for r in
                     json.load(open(RES / "thermodynamic_consistency_quantum.json"))["qkernel"]}

# ----------------------------------------------------------------- table
summary = {"classifiers": {}, "regressors": {}}
for k in ENT:
    cells = conf[k]["cells"]
    auc = [c["auc_out"] for c in cells]
    cov = [c["out"]["cov_pos"] for c in cells]
    amb = [c["out"]["both"] for c in cells]
    red = [c["auc_out"] for c in cells if c["group"] == "redox"]
    summary["classifiers"][k] = {
        "label": ENT[k][0], "n_cells": len(cells),
        "auc_out": float(np.mean(auc)), "auc_out_ci": boot_ci(auc),
        "auc_out_redox": float(np.mean(red)),
        "cov_pos_in": float(np.mean([c["in"]["cov_pos"] for c in cells])),
        "cov_pos_out": float(np.mean(cov)), "cov_pos_out_ci": boot_ci(cov),
        "miss_out": 1 - float(np.mean(cov)),
        "ambiguous_out": float(np.mean(amb)),
        "frac_cells_below_0.85": float(np.mean(np.array(cov) < 0.85)),
    }
for m in REG:
    sd, bd = THERMO[m]["site_disjoint"]["all"], THERMO[m]["bangladesh"]["all"]
    summary["regressors"][m] = {
        "label": REG[m][0],
        **{f"median_r2_{lv}_site": med(m, lv, "site_disjoint") for lv in LEVELS},
        "median_r2_L4_bangladesh": med(m, "L4", "bangladesh"),
        "si_calcite_r2_site": sd["SI_calcite"]["r2"], "si_gypsum_r2_site": sd["SI_gypsum"]["r2"],
        "abs_cbe_site": sd["abs_CBE_rec"], "abs_cbe_obs_site": sd["abs_CBE_obs"],
        "si_calcite_r2_bd": bd["SI_calcite"]["r2"], "si_gypsum_r2_bd": bd["SI_gypsum"]["r2"],
        "abs_cbe_bd": bd["abs_CBE_rec"], "abs_cbe_obs_bd": bd["abs_CBE_obs"],
    }
# paired two-sided Wilcoxon on seed-averaged cell AUC, every pair of entrants
from itertools import combinations
from scipy.stats import wilcoxon
summary["auc_wilcoxon"] = {}
for a_, b_ in combinations(ENT, 2):
    ca = {(c["target"], c["region"]): c for c in conf[a_]["cells"]}
    cb = {(c["target"], c["region"]): c for c in conf[b_]["cells"]}
    for grp in ("all", "redox"):
        ks = [k for k in ca if k in cb and (grp == "all" or ca[k]["group"] == "redox")]
        da = np.array([ca[k]["auc_out"] for k in ks])
        db = np.array([cb[k]["auc_out"] for k in ks])
        summary["auc_wilcoxon"][f"{a_}_vs_{b_}_{grp}"] = {
            "n": len(ks), "a": float(da.mean()), "b": float(db.mean()),
            "p": float(wilcoxon(da, db).pvalue) if np.any(da != db) else 1.0}
# axis A: paired Wilcoxon on per-target R2, GENESIS against each regressor
summary["r2_wilcoxon"] = {}
for lv in LEVELS:
    for test in ("site_disjoint", "bangladesh"):
        if ("fm", lv, test) not in R2:
            continue
        for m in ("gb", "rf", "tabpfn", "qkernel"):
            ts = sorted(set(R2[("fm", lv, test)]) & set(R2[(m, lv, test)]))
            a_ = np.array([R2[("fm", lv, test)][t] for t in ts])
            b_ = np.array([R2[(m, lv, test)][t] for t in ts])
            ok = ~(np.isnan(a_) | np.isnan(b_))
            summary["r2_wilcoxon"][f"fm_vs_{m}_{lv}_{test}"] = {
                "n": int(ok.sum()), "fm_median": float(np.median(a_[ok])),
                "other_median": float(np.median(b_[ok])),
                "p": float(wilcoxon(a_[ok], b_[ok]).pvalue)}
json.dump(summary, open(RES / "evaluation_summary.json", "w"), indent=1)

# one flat CSV row per entrant, blank where the axis does not apply
CSV_MAP = {"finetune_base": "fm", "tabpfn": "tabpfn", "rf": "rf", "xgb": None}
with open(RES / "evaluation_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["entrant", "A_L2_site", "A_L4_site", "A_L4_bangladesh",
                "B_auc_out", "B_auc_out_redox", "C_miss_out", "C_ambiguous_out",
                "D_si_calcite_site", "D_si_gypsum_site", "D_cbe_site", "D_cbe_bd"])
    rows = [(k, CSV_MAP.get(k)) for k in ENT] + [("gb (regressor)", "gb")]
    for k, rk in rows:
        c = summary["classifiers"].get(k)
        r = summary["regressors"].get(rk) if rk else None
        g = lambda d, key: "" if d is None else f"{d[key]:.3f}"
        w.writerow([ENT[k][0] if k in ENT else "Gradient boosting (regressor)",
                    g(r, "median_r2_L2_site"), g(r, "median_r2_L4_site"),
                    g(r, "median_r2_L4_bangladesh"),
                    g(c, "auc_out"), g(c, "auc_out_redox"), g(c, "miss_out"),
                    g(c, "ambiguous_out"), g(r, "si_calcite_r2_site"),
                    g(r, "si_gypsum_r2_site"), g(r, "abs_cbe_site"), g(r, "abs_cbe_bd")])
print("  -> results/evaluation_summary.json/.csv")

es.apply()
GCOL = ["redox", "conservative", "uranium"]


def heat(ax, M, rows, cols, cmap, vmin, vmax, fmt="{:.2f}"):
    """Annotated method x group table; yellow = better in both uses (cividis)."""
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            lum = (v - vmin) / (vmax - vmin)
            lum = lum if cmap == "cividis" else 1 - lum
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6,
                    color="black" if lum > 0.55 else "white")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def group_matrix(metric):
    order = sorted(ENT, key=lambda k: -summary["classifiers"][k]["auc_out"])
    M = np.array([[(conf[k]["summary"][g]["auc_out"] if metric == "auc"
                    else 1 - conf[k]["summary"][g]["cov_pos_out"]) for g in GCOL] for k in order])
    ncell = [conf["rf"]["summary"][g]["n_cells"] for g in GCOL]
    short = {"redox": "Redox", "conservative": "Cons.", "uranium": "Uranium"}
    cols = [f"{short[g]}\n({n})" for g, n in zip(GCOL, ncell)]
    return M, [ENT[k][0] for k in order], cols


# ----------------------------------------------------------------- Fig A
fig = plt.figure(figsize=(es.DOUBLE, 3.25))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.32)
a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
x = np.arange(len(LEVELS))
xb = x[-1] + 1.0
for m in REG:
    lab, col, mk = REG[m]
    a1.plot(x, [med(m, lv, "site_disjoint") for lv in LEVELS], marker=mk, color=col,
            lw=1.6 if m == "fm" else 1.0, ms=6 if mk == "*" else 4, zorder=3 if m == "fm" else 2,
            markeredgecolor=es.edge(col) if mk == "*" else col, markeredgewidth=0.5)
    a1.plot([xb], [med(m, "L4", "bangladesh")], marker=mk, color=col, ms=6.5 if mk == "*" else 4.5,
            lw=0, markeredgecolor=es.edge(col), markeredgewidth=0.4 if mk != "*" else 0.5, zorder=3)
a1.axhline(0, color=es.INK, lw=0.5)
a1.axvspan(x[-1] + 0.5, xb + 0.5, color=es.LIGHT, lw=0, zorder=0)
a1.set_xlim(-0.4, xb + 0.5)
a1.set_xticks(list(x) + [xb])
a1.set_xticklabels([f"{l}\n{t}" for l, t in zip(LEVELS, LEVEL_LAB)] + ["L4\nBangl."])
a1.set_ylabel("Median $R^2$ over targets")
a1.set_ylim(-0.32, 0.9)
a1.set_xlabel("Input level")
es.label(a1, "a", x=-0.16)

L4 = R2[("fm", "L4", "site_disjoint")]
tg = sorted(L4, key=lambda t: L4[t])
yy = np.arange(len(tg))
for i, t in enumerate(tg):
    if GROUP[t] in ("redox", "uranium"):
        a2.axhspan(i - 0.5, i + 0.5, color=es.GROUP_FILL["redox"], lw=0, zorder=0)
    best = max(("gb", "rf", "tabpfn"), key=lambda m: R2[(m, "L4", "site_disjoint")].get(t, -9))
    bv, fv = R2[(best, "L4", "site_disjoint")][t], L4[t]
    a2.plot([bv, fv], [i, i], color="#B0B0B0", lw=0.9, zorder=1)
    a2.plot(bv, i, marker=REG[best][2], color=REG[best][1], ms=4.2, lw=0,
            markeredgecolor="white", markeredgewidth=0.4, zorder=2)
    a2.plot(fv, i, marker="o", color=REG["fm"][1], ms=4.6, lw=0,
            markeredgecolor="white", markeredgewidth=0.4, zorder=3)
    qv = R2[("qkernel", "L4", "site_disjoint")].get(t, np.nan)
    if qv == qv:
        a2.plot(max(qv, -1), i, marker="*", color=es.YELLOW, ms=5.5, lw=0,
                markeredgecolor=es.INK, markeredgewidth=0.4, zorder=2.5)
    bd = R2[("fm", "L4", "bangladesh")].get(t, np.nan)
    if bd == bd:
        a2.plot(max(bd, -1), i, marker="<" if bd < -1 else "x", color=es.INK, ms=3.6,
                lw=0, markeredgewidth=0.9, zorder=3)
a2.set_yticks(yy)
a2.set_yticklabels([es.chem(t) for t in tg])
a2.axvline(0, color=es.INK, lw=0.5)
a2.set_xlim(-1.06, 1.04)
a2.set_ylim(-0.6, len(tg) - 0.4)
a2.set_xlabel("$R^2$ with every other parameter supplied (L4)")
es.label(a2, "b", x=-0.13)
hs = es.handles(["fm", "gb", "rf", "tabpfn"], kind="line") + es.handles(["qkernel"]) + [
    Line2D([], [], marker="x", color=es.INK, lw=0, ms=4, markeredgewidth=0.9,
           label="GENESIS in Bangladesh (clipped at −1)"),
    Patch(facecolor=es.GROUP_FILL["redox"], label="redox-sensitive target")]
es.shared_legend(fig, hs, y=0.99)
save(fig, "fig_eval_A_inferability")

# ----------------------------------------------------------------- Fig B
P = {}
for r in haz:
    if r["mode"] != "p80" or r["loro_mean"] is None or r["loro_mean"] != r["loro_mean"]:
        continue
    P.setdefault(r["param"], {"r2": r["r2"], "loro": []})["loro"].append(r["loro_mean"])
ps = sorted(P)
xr = np.array([P[p]["r2"] for p in ps])
yr = np.array([np.mean(P[p]["loro"]) for p in ps])
rho, pv = spearmanr(xr, yr)
summary["hazard_gradient"] = {"rho": float(rho), "p": float(pv), "n_params": len(ps)}

fig = plt.figure(figsize=(es.DOUBLE, 3.0))
gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 0.62, 0.34, 1.1], wspace=0.08)
b1, b2, b3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[3])
order = sorted(ENT, key=lambda k: summary["classifiers"][k]["auc_out"])
for i, k in enumerate(order):
    s = summary["classifiers"][k]
    lab, col, mk = ENT[k]
    v = [c["auc_out"] for c in conf[k]["cells"]]
    b1.scatter(v, np.full(len(v), i) + rng.uniform(-0.2, 0.2, len(v)), s=4, color=col,
               alpha=0.35, lw=0, zorder=1)
    b1.errorbar(s["auc_out"], i, xerr=[[s["auc_out"] - s["auc_out_ci"][0]],
                [s["auc_out_ci"][1] - s["auc_out"]]], fmt=mk, color=col,
                mec=es.edge(col), mew=0.5, capsize=2, elinewidth=0.9, zorder=3,
                ms=6.5 if mk == "*" else 4.8)
b1.set_yticks(range(len(order)))
b1.set_yticklabels([ENT[k][0] for k in order])
b1.axvline(0.5, color=es.INK, lw=0.5, ls=":")
b1.set_xlim(0.05, 1.0)
b1.set_xlabel("Exceedance AUC, held-out region")
es.label(b1, "a", x=-0.62)

M, rows, cols = group_matrix("auc")
heat(b2, M, rows, cols, "cividis", 0.5, 0.9)
b2.set_yticklabels([])
es.label(b2, "b", x=-0.16)

NUDGE = {"Mn": (-9, -6.5), "PO4": (3, -5), "Fe": (-7, 3), "F": (3, -5), "pH": (-9, 3),
         "Mg": (-9, -6), "HCO3": (3, 1), "TDS": (3, 2), "EC": (4, -2.5), "Ca": (3, -5.5),
         "Na": (3, 1.5), "SO4": (-14, 2.5), "Cl": (3, 1)}
for p, xv, yv in zip(ps, xr, yr):
    red = GROUP[p] in ("redox", "uranium")
    b3.scatter(xv, yv, marker="^" if red else "o", s=18,
               facecolor=es.GROUP_FILL["redox"] if red else "#D9D9D9",
               edgecolor=es.INK, lw=0.5, zorder=3)
    b3.annotate(es.chem(p), (xv, yv), xytext=NUDGE.get(p, (2.5, 1.5)),
                textcoords="offset points", fontsize=5.5, color="#333333")
b3.axhline(0.5, color=es.INK, lw=0.5, ls=":")
b3.set_xlabel("Masked-reconstruction $R^2$")
b3.set_ylabel("Raster-only RF AUC, held-out region")
es.label(b3, "c", x=-0.2)
hs = es.handles(order[::-1]) + [
    Line2D([], [], marker="^", lw=0, ms=4, markerfacecolor=es.GROUP_FILL["redox"],
           markeredgecolor=es.INK, markeredgewidth=0.5, label="redox-sensitive parameter"),
    Line2D([], [], marker="o", lw=0, ms=4, markerfacecolor="#D9D9D9",
           markeredgecolor=es.INK, markeredgewidth=0.5, label="other parameter")]
es.shared_legend(fig, hs, ncol=6, y=0.99)
save(fig, "fig_eval_B_transfer")

# ----------------------------------------------------------------- Fig C
fig = plt.figure(figsize=(es.DOUBLE, 3.0))
gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 0.62, 0.72], wspace=0.32)
c1, c2, c3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[3])
for k, (lab, col, mk) in ENT.items():
    s = summary["classifiers"][k]
    c1.plot(s["ambiguous_out"], s["miss_out"], marker=mk, color=col, ms=7.5 if mk == "*" else 5.5, lw=0,
            markeredgecolor=es.edge(col), markeredgewidth=0.5, zorder=3)
c1.axhline(0.10, color=es.INK, lw=0.6, ls="--")
c1.text(0.59, 0.107, "nominal 0.10", fontsize=5.8, ha="right", va="bottom")
c1.set_xlim(0.05, 0.6)
c1.set_ylim(0.0, 0.4)
c1.set_xlabel("Ambiguous prediction sets, held-out region")
c1.set_ylabel("Miss rate on exceedances")
es.label(c1, "a", x=-0.26)

show = ["rf", "tabpfn", "finetune_base"]
for j, k in enumerate(show):
    col = ENT[k][1]
    for side, off in (("in", -0.19), ("out", 0.19)):
        v = [c[side]["cov_pos"] for c in conf[k]["cells"]]
        bp = c2.boxplot(v, positions=[j + off], widths=0.3, patch_artist=True,
                        showfliers=False, medianprops=dict(color=es.INK, lw=0.9),
                        whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
        bp["boxes"][0].set(facecolor=col if side == "out" else "white", edgecolor=col, lw=0.9)
        c2.scatter(np.full(len(v), j + off) + rng.uniform(-0.07, 0.07, len(v)), v, s=2.5,
                   color=es.INK, alpha=0.4, lw=0, zorder=3)
c2.axhline(0.90, color=es.INK, lw=0.6, ls="--")
c2.set_xticks(range(len(show)))
c2.set_xticklabels(["Random\nforest", "TabPFN", "GENESIS\nfine-tuned"])
c2.set_ylabel("Exceedance coverage per cell")
c2.set_ylim(0, 1.03)
es.label(c2, "b", x=-0.24)

M, rows, cols = group_matrix("miss")
heat(c3, M, rows, cols, "cividis_r", 0.0, 0.6)
es.label(c3, "c", x=-1.25)
hs = es.handles(sorted(ENT, key=lambda k: -summary["classifiers"][k]["auc_out"]))
es.shared_legend(fig, hs, ncol=5, y=0.99)
save(fig, "fig_eval_C_calibration")

# ----------------------------------------------------------------- Fig D
# dumbbell per regressor: open marker = unseen sites, filled = Bangladesh, so the loss of
# consistency under shift is read along one line per method
fig, axs = plt.subplots(1, 4, figsize=(es.DOUBLE, 2.5), sharey=True,
                        gridspec_kw={"wspace": 0.12, "width_ratios": [1, 1, 1, 1.15]})
mets = [("SI_calcite", "$R^2$, calcite SI"), ("SI_gypsum", "$R^2$, gypsum SI"),
        ("SI_fluorite", "$R^2$, fluorite SI"), ("abs_CBE_rec", "|Charge-balance error| (%)")]
order = ["fm", "tabpfn", "rf", "gb", "qkernel"]
yy = np.arange(len(order))[::-1]
D = {}
for n, (ax, (key, xl)) in enumerate(zip(axs, mets)):
    for yv, m in zip(yy, order):
        v = [THERMO[m][st]["all"][key] if key == "abs_CBE_rec"
             else THERMO[m][st]["all"][key]["r2"] for st in ("site_disjoint", "bangladesh")]
        D[(m, key)] = v
        col, mk = REG[m][1], REG[m][2]
        ax.plot(v, [yv, yv], color="#BDBDBD", lw=1.4, zorder=1, solid_capstyle="round")
        star = mk == "*"
        ax.plot(v[0], yv, marker=mk, ms=7 if star else 5.2, mfc="white", mec=es.INK if star else col,
                mew=0.8 if star else 1.2, lw=0, zorder=3)
        ax.plot(v[1], yv, marker=mk, ms=7 if star else 5.2, color=col, mec=es.edge(col),
                mew=0.5 if star else 0.4, lw=0, zorder=3)
        fmt = "{:.1f}" if key == "abs_CBE_rec" else "{:.2f}"
        lo, hi = sorted(v)
        width = {"abs_CBE_rec": 34, "SI_fluorite": 3.08}.get(key, 1.83)
        near = (hi - lo) / width < 0.14         # close pair: push the two labels apart
        box = dict(fc="white", ec="none", pad=0.4)
        ax.text(hi, yv + 0.28, fmt.format(hi), fontsize=5.4, ha="left" if near else "center",
                color=INK2, bbox=box, zorder=4)
        ax.text(lo, yv + 0.28, fmt.format(lo), fontsize=5.4, ha="right" if near else "center",
                color=INK2, bbox=box, zorder=4)
    if key == "abs_CBE_rec":
        for st, ls in (("site_disjoint", ":"), ("bangladesh", "--")):
            ob = THERMO["fm"][st]["all"]["abs_CBE_obs"]
            ax.axvline(ob, color=es.INK, lw=0.7, ls=ls, zorder=0)
        ax.set_xlim(0, 34)
    else:
        ax.axvline(0, color=es.INK, lw=0.5, zorder=0)
        ax.set_xlim(-2.0 if key == "SI_fluorite" else -0.75, 1.08)
    ax.set_xlabel(xl)
    ax.set_ylim(-0.6, len(order) - 0.35)
    ax.tick_params(axis="y", length=0)
    es.label(ax, "abcd"[n], x=-0.08 if n else -0.62, y=1.02)
axs[0].set_yticks(yy)
axs[0].set_yticklabels([REG[m][0] for m in order])
for ax in axs[1:]:
    ax.spines["left"].set_visible(False)
hs = [Line2D([], [], marker="o", lw=0, ms=4.5, mfc="white", mec=INK2, mew=1.1,
             label="unseen sites (site-disjoint)"),
      Line2D([], [], marker="o", lw=0, ms=4.5, color=INK2, label="Bangladesh"),
      Line2D([], [], color=es.INK, lw=0.7, ls=":", label="observed waters, unseen sites"),
      Line2D([], [], color=es.INK, lw=0.7, ls="--", label="observed waters, Bangladesh")]
es.shared_legend(fig, hs, y=0.99)
summary["thermo_dumbbell"] = {f"{m}_{k}": v for (m, k), v in D.items()}
save(fig, "fig_eval_D_physics")

json.dump(summary, open(RES / "evaluation_summary.json", "w"), indent=1)
