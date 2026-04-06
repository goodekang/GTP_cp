"""
Compact Spatial Motif Panel for Multi-Figure Assembly
======================================================
Nature/Cell-grade compact visualization of spatial cell-cell interactions.

Design principles:
- Single column width (89mm / 3.5 inches)
- High information density
- Clean, minimal aesthetic
- Publication-ready for composite figures

Reference styles:
- Schürch et al., Cell 2020 (Figure 3)
- Squidpy, Nature Methods 2022 (Figure 2)
- Keren et al., Cell 2018 (Figure 6)

Usage:
    python fig4_motif_panel.py --output_dir ../../figures/fig4_mechanism
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch, ConnectionPatch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

import sys
sys.path.insert(0, str(Path(__file__).parent))

from plot_config import (
    setup_nature_style, PALETTE, FONT_CONFIG, save_figure, add_panel_label
)


# ============================================================
# Cell Type Colors (Compact palette)
# ============================================================

CELL_COLORS = {
    'CD8T': '#2A9D8F',        # Teal
    'CD4T': '#48CAE4',        # Light blue  
    'Tumor': '#E63946',       # Red
    'Macrophages': '#F4A261', # Orange
    'DC': '#9B5DE5',          # Purple
    'NK': '#0A9396',          # Dark teal
    'B': '#4361EE',           # Blue
    'Tregs': '#90BE6D',       # Green
    'Neutrophils': '#ADB5BD', # Gray
    'Endothelial': '#E9C46A', # Gold
    'Mesenchymal': '#BC6C25', # Brown
    'DC/Mono': '#7B2CBF',     # Dark purple
    'Keratin+Tumor': '#D62828', # Dark red
    'Other': '#6C757D',       # Dark gray
}

def get_color(cell_type):
    """Get color with fallback."""
    return CELL_COLORS.get(cell_type, PALETTE['grid'])


# ============================================================
# Panel A: Top Spatial Motifs Network
# ============================================================

def plot_motif_network_compact(ax, motifs, n_show=3):
    """
    Compact network showing top spatial motifs.
    
    Style: Horizontal layout with node pairs connected by edges.
    Similar to Schürch et al. Cell 2020 Figure 3.
    """
    n_show = min(n_show, len(motifs))
    
    # Vertical spacing for each motif
    y_positions = np.linspace(0.85, 0.25, n_show)
    
    for i, motif in enumerate(motifs[:n_show]):
        y = y_positions[i]
        
        # Get cell types
        ct1 = motif['nodes'][0]['type']
        ct2 = motif['nodes'][1]['type']
        zscore = motif.get('zscore', -3.0)
        hr = motif.get('hazard_ratio', 0.5)
        
        # Node positions (horizontal pair)
        x1, x2 = 0.18, 0.42
        node_r = 0.055
        
        # Draw edge (thickness based on z-score strength)
        edge_width = 2.0 + abs(zscore) / 3
        edge_alpha = min(0.8, 0.3 + abs(zscore) / 10)
        
        ax.plot([x1 + node_r, x2 - node_r], [y, y], 
                color=PALETTE['spatial_main'], 
                linewidth=edge_width, 
                alpha=edge_alpha,
                solid_capstyle='round',
                zorder=1)
        
        # Draw nodes
        for x, ct in [(x1, ct1), (x2, ct2)]:
            color = get_color(ct)
            
            # Node circle
            circle = Circle((x, y), node_r, 
                           facecolor=color, 
                           edgecolor='white',
                           linewidth=1.5, 
                           zorder=3)
            ax.add_patch(circle)
            
            # Abbreviate label
            label = ct[:4] if len(ct) > 4 else ct
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=5.5, fontweight='bold', color='white',
                   zorder=4)
        
        # Motif name (right side)
        name = f"{ct1}–{ct2}"
        ax.text(0.52, y, name, ha='left', va='center',
               fontsize=FONT_CONFIG['size_annotation'],
               color=PALETTE['text'])
        
        # Z-score
        ax.text(0.78, y, f"z={zscore:.1f}",
               ha='center', va='center',
               fontsize=FONT_CONFIG['size_annotation'],
               color=PALETTE['text_secondary'])
        
        # HR with color coding
        hr_color = PALETTE['low_risk'] if hr < 1 else PALETTE['high_risk']
        ax.text(0.92, y, f"{hr:.2f}",
               ha='center', va='center',
               fontsize=FONT_CONFIG['size_annotation'],
               fontweight='bold',
               color=hr_color)
    
    # Column headers
    ax.text(0.30, 0.98, 'Motif', ha='center', va='bottom',
           fontsize=FONT_CONFIG['size_axis_title'], fontweight='bold')
    ax.text(0.78, 0.98, 'z-score', ha='center', va='bottom',
           fontsize=FONT_CONFIG['size_axis_title'], fontweight='bold')
    ax.text(0.92, 0.98, 'HR', ha='center', va='bottom',
           fontsize=FONT_CONFIG['size_axis_title'], fontweight='bold')
    
    # Separator line
    ax.axhline(y=0.95, xmin=0.05, xmax=0.98, color=PALETTE['grid'], 
              linewidth=0.5, alpha=0.5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')


# ============================================================
# Panel B: Neighborhood Enrichment Mini-Heatmap
# ============================================================

def plot_interaction_heatmap_compact(ax, data_path):
    """
    Compact heatmap of cell-cell interactions.
    
    Style: Small square heatmap with key cell types.
    Similar to Squidpy Nature Methods 2022 Figure 2.
    """
    import seaborn as sns
    
    # Load interaction data
    interaction_file = data_path / 'cell_type_interactions_real.csv'
    if not interaction_file.exists():
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.axis('off')
        return
    
    df = pd.read_csv(interaction_file)
    
    # Pivot to matrix
    pivot = df.pivot(index='cell_type_1', columns='cell_type_2', values='zscore')
    
    # Select key cell types (most variable/interesting)
    key_types = ['CD8T', 'CD4T', 'Tumor', 'Macrophages', 'DC', 'NK', 'B', 'Tregs']
    available = [ct for ct in key_types if ct in pivot.index and ct in pivot.columns]
    
    if len(available) < 3:
        available = list(pivot.index[:8])
    
    pivot_sub = pivot.loc[available, available]
    
    # Custom colormap: blue (attraction) - white - red (avoidance)
    cmap = LinearSegmentedColormap.from_list('interaction',
        [PALETTE['spatial_main'], '#FFFFFF', PALETTE['high_risk']])
    
    # Plot heatmap
    vmax = min(10, np.abs(pivot_sub.values).max())
    
    im = ax.imshow(pivot_sub.values, cmap=cmap, aspect='equal',
                   vmin=-vmax, vmax=vmax)
    
    # Labels
    n = len(available)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    
    # Abbreviated labels
    abbrev = lambda s: s[:3] if len(s) > 3 else s
    ax.set_xticklabels([abbrev(ct) for ct in available], 
                       rotation=45, ha='right',
                       fontsize=5)
    ax.set_yticklabels([abbrev(ct) for ct in available],
                       fontsize=5)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label('z-score', fontsize=6)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.tick_params(length=0)


# ============================================================
# Panel C: Summary Statistics
# ============================================================

def plot_summary_stats(ax, motifs):
    """
    Compact summary of motif statistics.
    
    Shows: number of motifs, prognosis distribution, key finding.
    """
    # Count statistics
    n_total = len(motifs)
    n_good = sum(1 for m in motifs if m.get('prognosis') == 'good')
    n_poor = n_total - n_good
    
    # Mean z-score of top motifs
    mean_z = np.mean([m.get('zscore', 0) for m in motifs[:5]])
    
    # Bar chart: good vs poor prognosis
    bar_width = 0.35
    x = [0.3, 0.7]
    heights = [n_good, n_poor]
    colors = [PALETTE['low_risk'], PALETTE['high_risk']]
    labels = ['Good\nprognosis', 'Poor\nprognosis']
    
    bars = ax.bar(x, heights, width=bar_width, color=colors, 
                  edgecolor='white', linewidth=1)
    
    # Value labels on bars
    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
               str(h), ha='center', va='bottom',
               fontsize=FONT_CONFIG['size_annotation'],
               fontweight='bold')
    
    # X labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(heights) * 1.3)
    
    ax.set_ylabel('Motifs', fontsize=FONT_CONFIG['size_axis_title'])
    
    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', length=0)
    
    # Summary text
    ax.text(0.5, 0.95, f'n={n_total} motifs', 
           ha='center', va='top', transform=ax.transAxes,
           fontsize=FONT_CONFIG['size_annotation'],
           color=PALETTE['text_secondary'])


# ============================================================
# Main Composite Figure
# ============================================================

def create_compact_motif_figure(data_path, output_dir):
    """
    Create compact composite figure for multi-panel assembly.
    
    Layout:
    +------------------+
    |   A: Top Motifs  |
    +--------+---------+
    | B:Heat | C:Stats |
    +--------+---------+
    
    Size: 89mm (3.5 in) x 100mm (~4 in)
    """
    setup_nature_style()
    
    # Load motifs
    motifs_file = data_path / 'spatial_motifs_real.json'
    with open(motifs_file, 'r') as f:
        motifs = json.load(f)
    
    # Create figure
    fig = plt.figure(figsize=(3.5, 3.8))
    
    # GridSpec layout
    gs = fig.add_gridspec(2, 2, 
                          height_ratios=[1.2, 1],
                          width_ratios=[1.2, 1],
                          hspace=0.35, wspace=0.3,
                          left=0.08, right=0.95, 
                          top=0.92, bottom=0.08)
    
    # Panel A: Top motifs (spans both columns)
    ax_a = fig.add_subplot(gs[0, :])
    plot_motif_network_compact(ax_a, motifs, n_show=4)
    ax_a.set_title('Spatial Cell-Cell Interactions', 
                  fontsize=FONT_CONFIG['size_figure_title'],
                  fontweight='bold', pad=8)
    add_panel_label(ax_a, 'a', x=-0.02, y=1.08)
    
    # Panel B: Heatmap
    ax_b = fig.add_subplot(gs[1, 0])
    plot_interaction_heatmap_compact(ax_b, data_path)
    ax_b.set_title('Neighborhood\nEnrichment', 
                  fontsize=FONT_CONFIG['size_axis_title'],
                  fontweight='bold', pad=3)
    add_panel_label(ax_b, 'b', x=-0.15, y=1.12)
    
    # Panel C: Summary stats
    ax_c = fig.add_subplot(gs[1, 1])
    plot_summary_stats(ax_c, motifs)
    ax_c.set_title('Prognosis\nAssociation', 
                  fontsize=FONT_CONFIG['size_axis_title'],
                  fontweight='bold', pad=3)
    add_panel_label(ax_c, 'c', x=-0.15, y=1.12)
    
    # Data source annotation
    fig.text(0.5, 0.01, 'MIBI-TOF Breast Cancer (Keren et al., Cell 2018)',
            ha='center', fontsize=5, color=PALETTE['text_secondary'],
            style='italic')
    
    # Save
    save_figure(fig, 'fig4_panel_motifs_compact', output_dir)
    plt.close(fig)
    
    print("  ✓ Compact motif panel complete")


# ============================================================
# Alternative: Ultra-Compact Single Row
# ============================================================

def create_single_row_motifs(data_path, output_dir):
    """
    Ultra-compact single-row motif visualization.
    
    For insertion as a sub-panel in larger figures.
    Size: 180mm (7 in) x 25mm (1 in)
    """
    setup_nature_style()
    
    # Load motifs
    motifs_file = data_path / 'spatial_motifs_real.json'
    with open(motifs_file, 'r') as f:
        motifs = json.load(f)
    
    n_show = min(4, len(motifs))
    
    # Create figure - very wide and short
    fig, ax = plt.subplots(figsize=(7, 1.2))
    
    # Horizontal layout
    x_positions = np.linspace(0.12, 0.88, n_show)
    
    for i, motif in enumerate(motifs[:n_show]):
        x_center = x_positions[i]
        
        ct1 = motif['nodes'][0]['type']
        ct2 = motif['nodes'][1]['type']
        zscore = motif.get('zscore', -3.0)
        hr = motif.get('hazard_ratio', 0.5)
        
        # Vertical node pair
        y1, y2 = 0.7, 0.3
        node_r = 0.06
        
        # Edge
        edge_width = 1.5 + abs(zscore) / 4
        ax.plot([x_center, x_center], [y1 - node_r, y2 + node_r],
               color=PALETTE['spatial_main'],
               linewidth=edge_width,
               alpha=0.7,
               solid_capstyle='round')
        
        # Nodes
        for y, ct in [(y1, ct1), (y2, ct2)]:
            color = get_color(ct)
            circle = Circle((x_center, y), node_r,
                           facecolor=color,
                           edgecolor='white',
                           linewidth=1.2,
                           zorder=3)
            ax.add_patch(circle)
            
            # Label inside
            label = ct[:4] if len(ct) > 4 else ct
            ax.text(x_center, y, label, ha='center', va='center',
                   fontsize=5, fontweight='bold', color='white',
                   zorder=4)
        
        # Motif name below
        name = f"{ct1[:3]}–{ct2[:3]}"
        ax.text(x_center, 0.08, name, ha='center', va='bottom',
               fontsize=6, fontweight='bold')
        
        # HR below name
        hr_color = PALETTE['low_risk'] if hr < 1 else PALETTE['high_risk']
        ax.text(x_center, -0.05, f'HR={hr:.2f}',
               ha='center', va='top',
               fontsize=5.5, fontweight='bold',
               color=hr_color)
        
        # Z-score above
        ax.text(x_center, 0.92, f'z={zscore:.1f}',
               ha='center', va='bottom',
               fontsize=5, color=PALETTE['text_secondary'])
    
    # Title
    ax.set_title('Key Spatial Motifs (Real Data)', 
                fontsize=FONT_CONFIG['size_figure_title'],
                fontweight='bold', pad=5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.15, 1.05)
    ax.axis('off')
    
    plt.tight_layout()
    save_figure(fig, 'fig4_panel_motifs_row', output_dir)
    plt.close(fig)
    
    print("  ✓ Single-row motif panel complete")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate compact spatial motif panel'
    )
    parser.add_argument('--data_dir', type=str, 
                       default='../../data/processed')
    parser.add_argument('--output_dir', type=str,
                       default='../../figures/fig4_mechanism')
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 55)
    print("  Compact Spatial Motif Panel (Nature/Cell Style)")
    print("=" * 55)
    
    # Check data
    if not (data_path / 'spatial_motifs_real.json').exists():
        print("❌ Data not found. Run download_and_analyze_spatial.py first.")
        return
    
    print(f"\n📁 Output: {output_dir.resolve()}")
    
    # Generate panels
    print("\n🎨 Generating compact composite panel...")
    create_compact_motif_figure(data_path, output_dir)
    
    print("\n🎨 Generating single-row panel...")
    create_single_row_motifs(data_path, output_dir)
    
    print("\n" + "=" * 55)
    print("  ✅ Complete!")
    print("=" * 55)
    
    print("\n📄 Generated files:")
    for f in sorted(output_dir.glob("fig4_panel_motifs*.pdf")):
        print(f"   - {f.name}")


if __name__ == "__main__":
    main()





