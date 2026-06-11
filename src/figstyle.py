"""Shared figure design system for the GENESIS manuscript figures.

Vibrant 5-colour palette (ordered blue->red), fixed category->colour and
category->hatch mappings reused across every figure, and a clean sans-serif
theme. Colour and hatch are REDUNDANT (every category gets both) for
colourblind- and black-and-white-safe figures.

Usage:
    import figstyle as fs
    fs.apply_theme()
    ax.bar(x, y, color=fs.color(cat), hatch=fs.hatch(cat), **fs.BAR)
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- the palette (ordered: deep blue -> teal -> yellow -> orange -> red) ----
PALETTE = ["#00496F", "#0F85A0", "#EDD746", "#ED8B00", "#DD4124"]
NAVY, TEAL, YELLOW, ORANGE, RED = PALETTE
GREY = "#9aa0a6"

# ---- fixed category -> colour (reused in every figure) ----
COLORS = {
    # encoder sizes (ordered small->large)
    "Small": NAVY, "Base": TEAL, "Large": ORANGE,
    # mechanism groups
    "redox": RED, "conservative": NAVY, "uranium": YELLOW,
    # individual redox-coupled species
    "As": RED, "Fe": ORANGE, "Mn": YELLOW, "PO4": TEAL,
    # conservative / other contaminants
    "NO3": NAVY, "F": "#5BA3C2", "U": GREY,
    # models / methods
    "finetuned": RED, "frozen": TEAL, "rf": NAVY, "logreg": GREY,
    "histgb": ORANGE, "xgb": YELLOW,
    # generic redox-vs-conservative class (per-parameter figures)
    "redox_active": RED, "conservative_ion": NAVY, "bulk": TEAL,
}

# ---- fixed category -> hatch (group-wise; NO plain/solid bars) ----
# every category gets a non-empty pattern: dotted, crossed, horizontal,
# vertical, line-crossed, diagonal
HATCH = {
    "Small": "////", "Base": "....", "Large": "xxxx",
    "redox": "////", "conservative": "....", "uranium": "xxxx",
    "As": "////", "Fe": "....", "Mn": "xxxx", "PO4": "++++",
    "NO3": "----", "F": "||||", "U": "oooo",
    "finetuned": "////", "frozen": "....", "rf": "xxxx", "logreg": "++++",
    "histgb": "----", "xgb": "||||",
    "redox_active": "////", "conservative_ion": "....", "bulk": "xxxx",
}

# ordered hatch cycle (for arbitrary N categories); no plain/solid entry
HATCH_CYCLE = ["////", "....", "xxxx", "----", "||||", "++++", "\\\\\\\\", "oooo"]

# bar styling: vibrant fill, white edge so the hatch reads as a clean texture
BAR = dict(edgecolor="white", linewidth=0.8)


def color(cat, default=GREY):
    return COLORS.get(cat, default)


def hatch(cat, default=""):
    return HATCH.get(cat, default)


def apply_theme():
    """Clean, vibrant, publication theme (sans-serif, minimal axes, 300 dpi)."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.labelcolor": "#222222",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#C9C9C9",
        "grid.alpha": 0.3,          # very faint, barely visible
        "grid.linewidth": 0.5,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#CCCCCC",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "hatch.linewidth": 0.8,
        "hatch.color": "white",
    })


def legend_handles(cats, labels=None):
    """Patch handles (colour+hatch) for a manual legend over categories."""
    from matplotlib.patches import Patch
    labels = labels or cats
    return [Patch(facecolor=color(c), hatch=hatch(c), edgecolor="white", label=l)
            for c, l in zip(cats, labels)]
