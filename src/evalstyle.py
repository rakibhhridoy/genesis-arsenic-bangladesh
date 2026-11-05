"""Figure style for the evaluation-reframe figures (Journal of Hydrology).

Elsevier sizes (90 mm single column, 190 mm double), Arial at 7-8 pt (Helvetica Neue's
bold face is not loadable by matplotlib from its .ttc), and the
Okabe-Ito colour-blind-safe palette with ONE fixed colour and marker per method, reused in
every figure. Legends go in a single shared row above the panels (shared_legend).

    import evalstyle as es
    es.apply()
    fig = plt.figure(figsize=(es.DOUBLE, 3.0))
    es.shared_legend(fig, es.handles(["finetune_base", "rf"]))
"""
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MM = 1 / 25.4
SINGLE, DOUBLE = 90 * MM, 190 * MM

# Okabe & Ito (2008) palette
BLACK, ORANGE, SKY, GREEN = "#000000", "#E69F00", "#56B4E9", "#009E73"
YELLOW, BLUE, VERMILLION, PURPLE = "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
GREY, LIGHT = "#8C8C8C", "#EDEDED"
INK = "#1A1A1A"

# one entry per method: label, colour, marker. Regressor keys share colours with their
# classifier counterparts so a method looks the same in every figure.
METHOD = {
    "finetune_base": ("GENESIS fine-tuned", VERMILLION, "o"),
    "frozen_base": ("GENESIS frozen", PURPLE, "o"),
    "tabpfn": ("TabPFN", ORANGE, "D"),
    "xgb": ("XGBoost", GREEN, "s"),
    "rf": ("Random forest", BLUE, "s"),
    "logreg": ("Logistic regression", SKY, "^"),
    "rf_aux": ("Raster-only RF", GREY, "v"),
    "knn_space": ("Spatial kNN", BLACK, "v"),
    # simulated quantum-kernel entrant and its two matched controls (src/65)
    "qkernel": ("Quantum kernel", YELLOW, "*"),
    "qkernel_prod": ("Circuit, no entanglement", "#FFFFFF", "D"),
    "rbf_svm": ("Classical RBF kernel", "#FFFFFF", "o"),
    # regressors (axes A and D)
    "fm": ("GENESIS", VERMILLION, "o"),
    "gb": ("Gradient boosting", GREEN, "s"),
}
CLASSIFIERS = ["finetune_base", "frozen_base", "tabpfn", "xgb", "rf", "logreg", "rf_aux",
               "knn_space"]
REGRESSORS = ["fm", "gb", "rf", "tabpfn"]

# chemical groups (axis B and C cells, axis A targets)
GROUP_LABEL = {"redox": "Redox-sensitive", "conservative": "Conservative", "uranium": "Uranium"}
GROUP_FILL = {"redox": "#FBE3D6", "conservative": "#DCEBF5", "uranium": "#EFEFEF"}

CHEMTEX = {"NO3": "NO$_3$", "SO4": "SO$_4$", "HCO3": "HCO$_3$", "PO4": "PO$_4$",
           "SiO2": "SiO$_2$"}


def chem(p):
    return CHEMTEX.get(p, p)


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.grid": False, "lines.linewidth": 1.1, "lines.markersize": 4,
        "legend.frameon": False, "legend.handletextpad": 0.4, "legend.columnspacing": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def label(ax, letter, x=-0.12, y=1.04):
    """Panel label (a), (b) ... outside the top-left corner. Figures carry no panel titles."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left")


def edge(col):
    """Marker edge colour: white on filled markers, ink on white or yellow ones."""
    return INK if col in ("#FFFFFF", YELLOW) else "white"


def handles(keys, kind="marker"):
    out = []
    for k in keys:
        lab, col, mk = METHOD[k]
        if kind == "line":
            out.append(Line2D([], [], color=col, marker=mk, ms=4, lw=1.1, label=lab))
        elif kind == "patch":
            out.append(Patch(facecolor=col, edgecolor="none", label=lab))
        else:
            out.append(Line2D([], [], color=col, marker=mk, ms=6.5 if mk == "*" else 4.5, lw=0,
                              markeredgecolor=edge(col), markeredgewidth=0.4 if edge(col) == "white" else 0.6,
                              label=lab))
    return out


def shared_legend(fig, hs, ncol=None, y=1.0):
    """One legend row centred above all panels."""
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, y),
               ncol=ncol or len(hs), frameon=False)


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", facecolor="white")
    print("  ->", f"figures/{name}.pdf/.png")
