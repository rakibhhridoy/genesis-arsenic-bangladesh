"""
GENESIS Paper 5 — Step 7: Attention Weight Analysis
=====================================================
Extracts and visualises the parameter-to-parameter attention patterns
learned by the GENESIS encoder during Masked Geochemical Modeling.

Key outputs (saved to results/attention_analysis/):
  1. 20×20 parameter attention heatmap
  2. Network graph of top-K strongest couplings
  3. Per-layer attention evolution across transformer depth
  4. Comparison table of learned vs. known geochemical couplings

Usage:
  python 07_attention_analysis.py --ckpt checkpoints/small_stage1/best.pt \
      --size small --n_samples 5000
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Headless matplotlib
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
import seaborn as sns
import networkx as nx

# ── project imports ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from model.genesis_encoder import (
    GENESISForMGM, GENESISEncoder, GENESIS_PARAMS, PARAM_TO_ID,
    MODEL_CONFIGS, PAD_TOKEN, CLS_TOKEN,
)
from model.dataset import GENESISDataset, collate_fn, load_stats


# ============================================================
# Known geochemical couplings (ground truth for validation)
# ============================================================

KNOWN_COUPLINGS = {
    # Reductive dissolution of As
    ('As', 'Fe'):   'Reductive dissolution',
    ('As', 'PO4'):  'Competitive adsorption',
    ('As', 'Eh'):   'Redox control',
    # Fluorite equilibrium
    ('F', 'Ca'):    'Fluorite solubility',
    ('F', 'pH'):    'Fluoride mobilisation',
    ('F', 'HCO3'): 'Ion exchange / fluorite',
    # Redox indicators
    ('NO3', 'Eh'):  'Redox indicator',
    ('NO3', 'DOC'): 'Denitrification substrate',
    # Carbonate system
    ('Ca', 'Mg'):   'Carbonate weathering',
    ('Ca', 'HCO3'): 'Carbonate dissolution',
    ('Mg', 'HCO3'): 'Dolomite dissolution',
    # Halite / marine
    ('Na', 'Cl'):   'Halite / marine influence',
    # Definitional
    ('EC', 'TDS'):  'Definitional relationship',
}

# Make lookup symmetric
_sym = {}
for (a, b), v in KNOWN_COUPLINGS.items():
    _sym[(a, b)] = v
    _sym[(b, a)] = v
KNOWN_COUPLINGS_SYM = _sym


# ============================================================
# Attention hook utilities
# ============================================================

class AttentionCapture:
    """
    Register forward hooks on every nn.MultiheadAttention inside a
    nn.TransformerEncoder to capture attention weights.

    PyTorch's nn.TransformerEncoder does not expose attention weights,
    so we must hook into the self_attn sub-modules directly and
    re-invoke them with need_weights=True to get the attention matrix.

    A re-entrancy guard prevents infinite recursion when the hook
    itself calls the module.
    """

    def __init__(self, encoder: GENESISEncoder):
        self.encoder = encoder
        self.hooks = []
        self.attention_weights = []  # list[layer_idx] → (B, n_heads, S, S)
        self._inside_hook = False  # re-entrancy guard
        self._register()

    def _register(self):
        """Walk the TransformerEncoder and hook each self_attn module."""
        for layer_idx, layer in enumerate(self.encoder.transformer.layers):
            mha = layer.self_attn
            hook = mha.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        """
        Create a hook that intercepts MultiheadAttention.forward and
        re-runs it with need_weights=True to get the attention matrix.
        """
        capture = self

        def hook_fn(module, args, output):
            # Guard against infinite recursion: the re-invocation below
            # will trigger this hook again; skip the nested call.
            if capture._inside_hook:
                return

            with torch.no_grad():
                q, k, v = args[0], args[1], args[2]
                kwargs = {}
                if len(args) > 3 and args[3] is not None:
                    kwargs['key_padding_mask'] = args[3]
                if len(args) > 5 and args[5] is not None:
                    kwargs['attn_mask'] = args[5]

                capture._inside_hook = True
                try:
                    _, attn_w = module(
                        q, k, v,
                        need_weights=True,
                        average_attn_weights=False,  # per-head weights
                        **kwargs,
                    )
                finally:
                    capture._inside_hook = False

                # attn_w shape: (B, n_heads, S, S)
                while len(capture.attention_weights) <= layer_idx:
                    capture.attention_weights.append(None)
                capture.attention_weights[layer_idx] = attn_w.detach().cpu()

        return hook_fn

    def clear(self):
        self.attention_weights = []

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ============================================================
# Build parameter-to-parameter attention matrix
# ============================================================

def build_param_attention_matrices(
    model: GENESISForMGM,
    dataloader: DataLoader,
    device: torch.device,
    n_samples: int,
):
    """
    Run forward passes and accumulate per-layer 20×20 parameter
    attention matrices.

    Returns:
        per_layer: np.ndarray (n_layers, 20, 20)
        mean_all:  np.ndarray (20, 20) — averaged across layers
        count_mat: np.ndarray (20, 20) — number of samples contributing
    """
    model.eval()
    encoder = model.encoder
    n_layers = encoder.n_layers
    n_params = len(GENESIS_PARAMS)

    # Accumulators
    attn_sum = np.zeros((n_layers, n_params, n_params), dtype=np.float64)
    attn_count = np.zeros((n_params, n_params), dtype=np.float64)

    capture = AttentionCapture(encoder)

    processed = 0
    with torch.no_grad():
        for batch in dataloader:
            if processed >= n_samples:
                break

            param_ids = batch['param_ids'].to(device)
            values = batch['values'].to(device)
            padding_mask = batch['padding_mask'].to(device)

            capture.clear()
            # Forward pass — hooks fire automatically
            _ = encoder(param_ids, values, padding_mask)

            B = param_ids.size(0)
            param_ids_np = param_ids.cpu().numpy()

            for layer_idx in range(n_layers):
                attn_w = capture.attention_weights[layer_idx]  # (B, H, S, S)
                # Average over heads
                attn_w = attn_w.mean(dim=1).numpy()  # (B, S, S)

                for b in range(B):
                    if processed + b >= n_samples:
                        break
                    ids = param_ids_np[b]
                    for i_seq in range(len(ids)):
                        pid_i = ids[i_seq]
                        # Map to GENESIS_PARAMS index (skip special tokens)
                        if pid_i < 3:  # PAD=0, MASK=1, CLS=2
                            continue
                        gi = pid_i - 3
                        if gi >= n_params:
                            continue
                        for j_seq in range(len(ids)):
                            pid_j = ids[j_seq]
                            if pid_j < 3:
                                continue
                            gj = pid_j - 3
                            if gj >= n_params:
                                continue
                            attn_sum[layer_idx, gi, gj] += attn_w[b, i_seq, j_seq]
                            if layer_idx == 0:
                                attn_count[gi, gj] += 1

            processed += B
            if processed % 1000 == 0:
                print(f"  Processed {processed:,} / {n_samples:,} samples")

    capture.remove_hooks()

    # Normalize
    per_layer = np.zeros_like(attn_sum)
    # attn_count is same across layers (same samples), scale per layer
    safe_count = np.where(attn_count > 0, attn_count, 1.0)
    for li in range(n_layers):
        per_layer[li] = attn_sum[li] / safe_count

    mean_all = per_layer.mean(axis=0)
    return per_layer, mean_all, attn_count


# ============================================================
# Identify top couplings
# ============================================================

def get_top_couplings(attn_matrix, k=30):
    """
    Return top-K strongest off-diagonal couplings (undirected).
    Uses (A[i,j] + A[j,i]) / 2 for symmetric strength.
    """
    n = attn_matrix.shape[0]
    sym = (attn_matrix + attn_matrix.T) / 2.0
    np.fill_diagonal(sym, 0)

    couplings = []
    seen = set()
    for _ in range(k):
        idx = np.unravel_index(np.argmax(sym), sym.shape)
        i, j = idx
        if i == j:
            break
        strength = sym[i, j]
        if strength <= 0:
            break
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            p_i = GENESIS_PARAMS[i]
            p_j = GENESIS_PARAMS[j]
            known = KNOWN_COUPLINGS_SYM.get((p_i, p_j), '')
            couplings.append({
                'param_i': p_i,
                'param_j': p_j,
                'strength': float(strength),
                'known_mechanism': known,
            })
        sym[i, j] = 0
        sym[j, i] = 0

    return couplings


# ============================================================
# Plotting — Publication quality
# ============================================================

# Style constants
FONT_SIZE = 9
TITLE_SIZE = 11
LABEL_PAD = 4
DPI = 300


def _setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE + 1,
        'axes.titlesize': TITLE_SIZE,
        'xtick.labelsize': FONT_SIZE - 1,
        'ytick.labelsize': FONT_SIZE - 1,
        'legend.fontsize': FONT_SIZE - 1,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
    })


def plot_attention_heatmap(attn_matrix, out_path, title='GENESIS Parameter Attention Matrix'):
    """20×20 heatmap with seaborn."""
    _setup_style()
    sym = (attn_matrix + attn_matrix.T) / 2.0

    fig, ax = plt.subplots(figsize=(8, 7))
    mask_diag = np.eye(len(GENESIS_PARAMS), dtype=bool)

    sns.heatmap(
        sym,
        xticklabels=GENESIS_PARAMS,
        yticklabels=GENESIS_PARAMS,
        cmap='YlOrRd',
        mask=mask_diag,
        linewidths=0.3,
        linecolor='white',
        square=True,
        cbar_kws={'label': 'Mean Attention Weight', 'shrink': 0.75},
        ax=ax,
    )
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=10)
    ax.set_xlabel('Attended Parameter (Key)', labelpad=LABEL_PAD)
    ax.set_ylabel('Attending Parameter (Query)', labelpad=LABEL_PAD)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved heatmap → {out_path}")


def plot_coupling_network(couplings, out_path, top_k=20,
                          title='Learned Geochemical Coupling Network'):
    """Network graph of strongest couplings (networkx)."""
    _setup_style()

    G = nx.Graph()
    for p in GENESIS_PARAMS:
        G.add_node(p)

    top = couplings[:top_k]
    strengths = [c['strength'] for c in top]
    if not strengths:
        print("  No couplings to plot — skipping network graph.")
        return
    max_s = max(strengths)
    min_s = min(strengths) if len(strengths) > 1 else 0

    for c in top:
        G.add_edge(
            c['param_i'], c['param_j'],
            weight=c['strength'],
            known=bool(c['known_mechanism']),
        )

    fig, ax = plt.subplots(figsize=(9, 8))
    pos = nx.spring_layout(G, seed=42, k=1.8, iterations=80)

    # Draw edges
    known_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('known')]
    novel_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('known')]

    edge_widths_known = [
        1.0 + 4.0 * (G[u][v]['weight'] - min_s) / (max_s - min_s + 1e-9)
        for u, v in known_edges
    ]
    edge_widths_novel = [
        1.0 + 4.0 * (G[u][v]['weight'] - min_s) / (max_s - min_s + 1e-9)
        for u, v in novel_edges
    ]

    if known_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=known_edges, width=edge_widths_known,
            edge_color='#2166ac', alpha=0.8, ax=ax,
        )
    if novel_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=novel_edges, width=edge_widths_novel,
            edge_color='#b2182b', style='dashed', alpha=0.7, ax=ax,
        )

    # Draw nodes
    node_sizes = []
    for p in G.nodes():
        deg = sum(G[p][nb]['weight'] for nb in G.neighbors(p)) if G.degree(p) > 0 else 0
        node_sizes.append(300 + 600 * deg / (max_s * top_k + 1e-9))

    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color='#f7f7f7',
        edgecolors='#333333', linewidths=1.0, ax=ax,
    )
    nx.draw_networkx_labels(
        G, pos, font_size=FONT_SIZE, font_weight='bold', ax=ax,
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2166ac', linewidth=2, label='Known coupling'),
        Line2D([0], [0], color='#b2182b', linewidth=2, linestyle='--', label='Novel coupling'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', frameon=True,
              fancybox=False, edgecolor='#cccccc')

    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=10)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved network → {out_path}")


def plot_layer_evolution(per_layer, couplings, out_path, top_k=8,
                         title='Coupling Strength Across Transformer Layers'):
    """Line plot showing how the top-K coupling strengths evolve layer-by-layer."""
    _setup_style()
    n_layers = per_layer.shape[0]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.colormaps.get_cmap('tab10')

    for rank, c in enumerate(couplings[:top_k]):
        i = GENESIS_PARAMS.index(c['param_i'])
        j = GENESIS_PARAMS.index(c['param_j'])
        # Symmetric average per layer
        strengths = [(per_layer[l, i, j] + per_layer[l, j, i]) / 2.0
                     for l in range(n_layers)]
        label = f"{c['param_i']}–{c['param_j']}"
        if c['known_mechanism']:
            label += ' *'
        color = cmap(rank / max(top_k - 1, 1))
        ax.plot(range(1, n_layers + 1), strengths, marker='o', markersize=4,
                linewidth=1.5, color=color, label=label)

    ax.set_xlabel('Transformer Layer', labelpad=LABEL_PAD)
    ax.set_ylabel('Mean Attention Weight', labelpad=LABEL_PAD)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='#cccccc',
              ncol=2 if top_k > 5 else 1, fontsize=FONT_SIZE - 1)
    ax.grid(True, alpha=0.25, linewidth=0.4)

    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved layer evolution → {out_path}")


def plot_known_vs_learned(attn_matrix, out_path,
                          title='Known Geochemical Couplings: Learned Attention Strength'):
    """Bar chart comparing attention strength for known couplings."""
    _setup_style()
    sym = (attn_matrix + attn_matrix.T) / 2.0

    labels = []
    strengths = []
    mechanisms = []
    seen = set()
    for (a, b), mech in KNOWN_COUPLINGS.items():
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        i = GENESIS_PARAMS.index(a)
        j = GENESIS_PARAMS.index(b)
        labels.append(f"{a}–{b}")
        strengths.append(sym[i, j])
        mechanisms.append(mech)

    # Sort by strength
    order = np.argsort(strengths)[::-1]
    labels = [labels[o] for o in order]
    strengths = [strengths[o] for o in order]
    mechanisms = [mechanisms[o] for o in order]

    # Background median for reference
    off_diag = sym[~np.eye(sym.shape[0], dtype=bool)]
    median_bg = np.median(off_diag)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(range(len(labels)), strengths, color='#4393c3', edgecolor='white',
                   height=0.7)
    ax.axvline(median_bg, color='#999999', linestyle='--', linewidth=0.8,
               label=f'Median off-diag ({median_bg:.4f})')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Mean Attention Weight', labelpad=LABEL_PAD)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=10)
    ax.invert_yaxis()
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#cccccc')
    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved known-vs-learned → {out_path}")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='GENESIS Attention Weight Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--ckpt', type=str,
                        default='checkpoints/small_stage1/best.pt',
                        help='Path to encoder checkpoint')
    parser.add_argument('--size', type=str, default='small',
                        choices=['small', 'base', 'large'],
                        help='Model size')
    parser.add_argument('--data', type=str,
                        default='data/processed/genesis_pretrain.pt',
                        help='Path to chemistry tensor (.pt) used by GENESISDataset')
    parser.add_argument('--meta', type=str,
                        default='data/processed/genesis_pretrain_meta.parquet',
                        help='Path to metadata parquet matching the chem tensor')
    parser.add_argument('--aux', type=str,
                        default='data/processed/genesis_pretrain_aux.pt',
                        help='Path to aux-feature tensor (optional, .pt)')
    parser.add_argument('--norm_stats', type=str,
                        default='data/processed/normalization_stats.json',
                        help='Path to normalisation statistics JSON')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Number of samples to process')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for forward passes')
    parser.add_argument('--top_k', type=int, default=25,
                        help='Number of top couplings to report')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto / cpu / mps / cuda)')
    parser.add_argument('--outdir', type=str,
                        default='results/attention_analysis',
                        help='Output directory')
    return parser.parse_args()


def resolve_device(requested):
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(requested)


def main():
    args = parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    ckpt_path = project_root / args.ckpt
    data_path = project_root / args.data
    meta_path = project_root / args.meta
    aux_path = project_root / args.aux
    norm_path = project_root / args.norm_stats
    out_dir = project_root / args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"GENESIS Attention Analysis")
    print(f"{'=' * 55}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Model size : {args.size}")
    print(f"  Data       : {data_path}")
    print(f"  Device     : {device}")
    print(f"  Samples    : {args.n_samples:,}")
    print(f"  Output     : {out_dir}")
    print()

    # ── Load model ───────────────────────────────────────────
    print("Loading model...")
    model = GENESISForMGM(args.size)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    cfg = MODEL_CONFIGS[args.size]
    print(f"  Loaded GENESIS-{args.size.capitalize()} "
          f"({cfg['n_layers']}L, {cfg['n_heads']}H, d={cfg['d_model']})")

    # ── Load data ────────────────────────────────────────────
    print("Loading data...")
    norm_stats = load_stats(norm_path)

    chem_tensor = torch.load(data_path, weights_only=True).numpy()
    meta_df = pd.read_parquet(meta_path) if meta_path.exists() else None
    aux_tensor = (torch.load(aux_path, weights_only=True).numpy()
                  if aux_path.exists() else None)
    print(f"  chem={chem_tensor.shape}, "
          f"aux={aux_tensor.shape if aux_tensor is not None else None}, "
          f"meta={'yes' if meta_df is not None else 'no'}")

    dataset = GENESISDataset(
        chem_values=chem_tensor,
        norm_stats=norm_stats,
        meta_df=meta_df,
        aux_values=aux_tensor,
        mask_ratio=0.0,  # clean attention patterns (no masking)
    )
    # Subsample
    n = min(args.n_samples, len(dataset))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(dataset), size=n, replace=False)
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(subset, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"  {n:,} samples in {len(loader)} batches")

    # ── Extract attention weights ────────────────────────────
    print("\nExtracting attention weights...")
    per_layer, mean_all, count_mat = build_param_attention_matrices(
        model, loader, device, n_samples=n,
    )
    print(f"  Done. Attention matrix shape: {mean_all.shape}")

    # ── Top couplings ────────────────────────────────────────
    couplings = get_top_couplings(mean_all, k=args.top_k)
    print(f"\nTop-{len(couplings)} Learned Couplings:")
    print(f"  {'Rank':<5} {'Coupling':<14} {'Strength':>10}  {'Known Mechanism'}")
    print(f"  {'-'*5} {'-'*14} {'-'*10}  {'-'*30}")
    for rank, c in enumerate(couplings, 1):
        tag = c['known_mechanism'] if c['known_mechanism'] else '(novel)'
        print(f"  {rank:<5} {c['param_i']:>5}–{c['param_j']:<6} {c['strength']:10.6f}  {tag}")

    # Count how many known couplings appear in top-K
    n_known_in_top = sum(1 for c in couplings if c['known_mechanism'])
    n_known_total = len(set(
        tuple(sorted(k)) for k in KNOWN_COUPLINGS.keys()
    ))
    print(f"\n  Known couplings recovered in top-{len(couplings)}: "
          f"{n_known_in_top}/{n_known_total}")

    # ── Generate figures ─────────────────────────────────────
    print("\nGenerating figures...")

    plot_attention_heatmap(
        mean_all,
        str(out_dir / 'attention_heatmap.png'),
        title=f'GENESIS-{args.size.capitalize()} Parameter Attention Matrix',
    )

    plot_coupling_network(
        couplings,
        str(out_dir / 'coupling_network.png'),
        top_k=min(20, len(couplings)),
        title=f'GENESIS-{args.size.capitalize()} Learned Coupling Network',
    )

    plot_layer_evolution(
        per_layer, couplings,
        str(out_dir / 'layer_evolution.png'),
        top_k=min(8, len(couplings)),
        title=f'GENESIS-{args.size.capitalize()} Coupling Strength Across Layers',
    )

    plot_known_vs_learned(
        mean_all,
        str(out_dir / 'known_vs_learned.png'),
        title=f'GENESIS-{args.size.capitalize()} Known Coupling Attention Strengths',
    )

    # ── Save numerical results ───────────────────────────────
    print("\nSaving numerical results...")

    # Attention matrices as .npy
    np.save(str(out_dir / 'attention_per_layer.npy'), per_layer)
    np.save(str(out_dir / 'attention_mean.npy'), mean_all)
    np.save(str(out_dir / 'attention_count.npy'), count_mat)
    print(f"  Saved attention matrices (.npy)")

    # Couplings table
    df_couplings = pd.DataFrame(couplings)
    df_couplings.to_csv(str(out_dir / 'top_couplings.csv'), index=False)
    print(f"  Saved top_couplings.csv")

    # Parameter labels for reference
    with open(str(out_dir / 'param_labels.json'), 'w') as f:
        json.dump(GENESIS_PARAMS, f, indent=2)

    # Summary stats
    off_diag = mean_all[~np.eye(20, dtype=bool)]
    summary = {
        'model_size': args.size,
        'n_samples': n,
        'n_layers': cfg['n_layers'],
        'n_heads': cfg['n_heads'],
        'attention_mean_offdiag': float(off_diag.mean()),
        'attention_median_offdiag': float(np.median(off_diag)),
        'attention_std_offdiag': float(off_diag.std()),
        'n_known_recovered': n_known_in_top,
        'n_known_total': n_known_total,
        'top_coupling': f"{couplings[0]['param_i']}-{couplings[0]['param_j']}" if couplings else '',
    }
    with open(str(out_dir / 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary.json")

    print(f"\nAttention analysis complete. Results in: {out_dir}")


if __name__ == '__main__':
    main()
