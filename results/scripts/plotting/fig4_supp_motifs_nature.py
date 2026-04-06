"""
Nature Biotechnology-Grade Spatial Motif Visualization
=======================================================
Creates publication-quality spatial motif diagrams from MIBI-TOF breast cancer data.

Visual Design Principles:
- Minimalist network topology with semantic cell type colors
- Clear hazard ratio and z-score annotations
- Professional edge styling based on interaction strength
- Cohesive color scheme following Nature standards

Usage:
    python fig4_supp_motifs_nature.py --data_dir ../../data/processed --output_dir ../../figures/fig4_mechanism
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, ConnectionPatch
from matplotlib.patches import Arc, Wedge
import matplotlib.patheffects as path_effects
from matplotlib.colors import to_rgba

import sys
sys.path.insert(0, str(Path(__file__).parent))

from plot_config import (
    setup_nature_style, PALETTE, FONT_CONFIG, save_figure
)


# ============================================================
# Cell Type Color Mapping (MIBI-TOF specific)
# ============================================================

CELL_TYPE_COLORS = {
    # Immune cells - cool tones
    'NK': '#0A9396',              # Teal - spatial_main
    'CD8+ T': '#2A9D8F',          # Seafoam
    'CD4+ T': '#48B89D',          # Light teal
    'B cell': '#4169E1',          # Royal blue
    'Neutrophils': '#A8DADC',     # Light blue-gray
    'Monocytes': '#6D9DC5',       # Steel blue
    'DC': '#7B68EE',              # Medium slate blue
    'Tregs': '#66CDAA',           # Aquamarine
    
    # Myeloid cells - warm tones
    'Macrophage': '#EE9B00',      # Amber - clinical_main
    'M1': '#FFB347',              # Light orange
    'M2': '#CA6702',              # Burnt sienna
    'DC/Mono': '#D4A574',         # Tan
    
    # Structural cells
    'Tumor': '#E63946',           # Crimson - intervention_main
    'Epithelial': '#FF6B6B',      # Coral
    'CAF': '#8B4513',             # Saddle brown
    'Fibroblast': '#A0522D',      # Sienna
    'Endothelial': '#9B5DE5',     # Violet - causal_main
    
    # Other
    'Keratin': '#DEB887',         # Burlywood
    'Mesenchymal': '#BC8F8F',     # Rosy brown
    'Unidentified': '#ADB5BD',    # Gray
    'Other': '#6C757D',           # Dark gray
}


def get_cell_color(cell_type):
    """Get color for a cell type, with fallback."""
    # Try exact match
    if cell_type in CELL_TYPE_COLORS:
        return CELL_TYPE_COLORS[cell_type]
    
    # Try partial match
    for key, color in CELL_TYPE_COLORS.items():
        if key.lower() in cell_type.lower() or cell_type.lower() in key.lower():
            return color
    
    # Fallback
    return PALETTE['grid']


def draw_cell_node(ax, x, y, cell_type, radius=0.08, label_below=True, 
                   fontsize=None, show_label=True):
    """
    Draw a cell type node with professional styling.
    
    Features:
    - Soft shadow for depth
    - White border for contrast
    - Abbreviated label
    """
    if fontsize is None:
        fontsize = FONT_CONFIG['size_annotation']
    
    color = get_cell_color(cell_type)
    
    # Shadow (subtle)
    shadow = Circle((x + 0.008, y - 0.008), radius, 
                    facecolor='#00000015', edgecolor='none', zorder=1)
    ax.add_patch(shadow)
    
    # Main node
    node = Circle((x, y), radius, 
                  facecolor=color, 
                  edgecolor='white', 
                  linewidth=2.5, 
                  zorder=3)
    ax.add_patch(node)
    
    # Abbreviate long names
    abbrev_map = {
        'Macrophage': 'Macr',
        'Neutrophils': 'Neut',
        'CD8+ T': 'CD8+',
        'CD4+ T': 'CD4+',
        'Endothelial': 'Endo',
        'Fibroblast': 'Fibro',
        'Epithelial': 'Epi',
        'Mesenchymal': 'Mesen',
    }
    
    short_name = abbrev_map.get(cell_type, cell_type[:5] if len(cell_type) > 5 else cell_type)
    
    # Inner label (on node)
    ax.text(x, y, short_name, ha='center', va='center',
            fontsize=fontsize - 0.5, fontweight='bold', color='white',
            zorder=4)
    
    return radius


def draw_interaction_edge(ax, x1, y1, x2, y2, zscore, width=2.0):
    """
    Draw an interaction edge with styling based on z-score.
    
    Negative z-score (attraction): solid, thicker
    Positive z-score (repulsion): dashed, thinner
    """
    # Calculate edge properties based on z-score
    if zscore < -2:
        # Strong attraction - solid thick line
        linestyle = '-'
        alpha = min(0.9, 0.4 + abs(zscore) / 10)
        lw = width * (1 + abs(zscore) / 5)
        color = PALETTE['spatial_main']
    elif zscore < 0:
        # Weak attraction
        linestyle = '-'
        alpha = 0.5
        lw = width * 0.8
        color = PALETTE['spatial_light']
    else:
        # Repulsion (not typically shown in motifs)
        linestyle = '--'
        alpha = 0.3
        lw = width * 0.5
        color = PALETTE['grid']
    
    ax.plot([x1, x2], [y1, y2], 
            color=color, linewidth=lw, 
            linestyle=linestyle, alpha=alpha,
            zorder=2, solid_capstyle='round')


def plot_single_motif(ax, motif, position, width=0.25):
    """
    Plot a single spatial motif as a network diagram.
    
    Args:
        ax: matplotlib axes
        motif: motif dictionary
        position: (x, y) center position
        width: total width of the motif diagram
    """
    cx, cy = position
    nodes = motif['nodes']
    edges = motif['edges']
    
    n_nodes = len(nodes)
    
    # Layout nodes
    if n_nodes == 2:
        # Two nodes - vertical arrangement
        node_positions = {
            nodes[0]['id']: (cx, cy + 0.12),
            nodes[1]['id']: (cx, cy - 0.12)
        }
    else:
        # Multiple nodes - arrange in circle
        angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, n_nodes, endpoint=False)
        radius = width * 0.35
        node_positions = {}
        for i, node in enumerate(nodes):
            nx = cx + radius * np.cos(angles[i])
            ny = cy + radius * np.sin(angles[i])
            node_positions[node['id']] = (nx, ny)
    
    # Draw edges first
    for edge in edges:
        src_pos = node_positions[edge['source']]
        tgt_pos = node_positions[edge['target']]
        
        zscore = motif.get('zscore', -3.0)
        draw_interaction_edge(ax, 
                             src_pos[0], src_pos[1],
                             tgt_pos[0], tgt_pos[1],
                             zscore)
    
    # Draw nodes
    for node in nodes:
        pos = node_positions[node['id']]
        draw_cell_node(ax, pos[0], pos[1], node['type'], radius=0.065)
    
    # Add motif name below
    name_parts = motif['name'].replace(' Interaction', '').split('-')
    if len(name_parts) == 2:
        display_name = f"{name_parts[0]}-{name_parts[1]}"
    else:
        display_name = motif['name'][:15]
    
    ax.text(cx, cy - 0.32, display_name, 
            ha='center', va='top',
            fontsize=FONT_CONFIG['size_axis_title'],
            fontweight='bold', color=PALETTE['text'])
    
    # Add HR annotation
    hr = motif.get('hazard_ratio', 1.0)
    hr_color = PALETTE['low_risk'] if hr < 1 else PALETTE['high_risk']
    
    ax.text(cx, cy - 0.42, f'HR={hr:.2f}',
            ha='center', va='top',
            fontsize=FONT_CONFIG['size_stats_box'],
            fontweight='bold', color=hr_color)
    
    # Add z-score
    zscore = motif.get('zscore', 0)
    ax.text(cx, cy - 0.50, f'z={zscore:.1f}',
            ha='center', va='top',
            fontsize=FONT_CONFIG['size_annotation'],
            color=PALETTE['text_secondary'])


def create_motif_legend(ax, cell_types):
    """Create a legend for cell type colors."""
    unique_types = list(set(cell_types))
    n_types = len(unique_types)
    
    # Layout in horizontal row
    x_start = 0.1
    x_end = 0.9
    y_pos = 0.08
    spacing = (x_end - x_start) / max(n_types - 1, 1)
    
    for i, cell_type in enumerate(unique_types):
        x = x_start + i * spacing
        color = get_cell_color(cell_type)
        
        # Color swatch
        rect = FancyBboxPatch((x - 0.03, y_pos - 0.015), 0.06, 0.03,
                              boxstyle="round,pad=0.005",
                              facecolor=color, edgecolor='white',
                              linewidth=1, zorder=2)
        ax.add_patch(rect)
        
        # Label
        ax.text(x, y_pos - 0.04, cell_type, 
                ha='center', va='top',
                fontsize=FONT_CONFIG['size_annotation'],
                color=PALETTE['text'])


def create_spatial_motifs_figure(motifs, output_dir, max_motifs=5):
    """
    Create the main spatial motifs figure.
    
    Layout: Horizontal arrangement of top motifs with legend below.
    """
    setup_nature_style()
    
    n_motifs = min(len(motifs), max_motifs)
    
    # Figure dimensions
    fig_width = 2.0 * n_motifs + 1
    fig_height = 4.0
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Title
    ax.text(0.5, 0.95, 'Key Spatial Motifs (Real Data)', 
            ha='center', va='top', transform=ax.transAxes,
            fontsize=FONT_CONFIG['size_figure_title'] + 2,
            fontweight='bold', color=PALETTE['text'])
    
    # Subtitle
    ax.text(0.5, 0.88, 'MIBI-TOF Breast Cancer Cohort | Neighborhood Enrichment Analysis',
            ha='center', va='top', transform=ax.transAxes,
            fontsize=FONT_CONFIG['size_annotation'],
            color=PALETTE['text_secondary'], style='italic')
    
    # Plot each motif
    positions = np.linspace(0.15, 0.85, n_motifs)
    
    all_cell_types = []
    for i, motif in enumerate(motifs[:n_motifs]):
        plot_single_motif(ax, motif, (positions[i], 0.55), width=0.2)
        
        # Collect cell types for legend
        for node in motif['nodes']:
            all_cell_types.append(node['type'])
    
    # Legend
    create_motif_legend(ax, all_cell_types)
    
    # Clean up axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    save_figure(fig, 'fig4_supp_motifs', output_dir)
    plt.close(fig)


def create_detailed_motif_panel(motif, output_dir, panel_name):
    """
    Create a detailed single-motif panel for supplementary figures.
    
    Features:
    - Larger node representation
    - Ligand-receptor annotation
    - Clinical relevance text
    """
    setup_nature_style()
    
    fig, ax = plt.subplots(figsize=(4, 4.5))
    
    # Title
    ax.text(0.5, 0.95, motif['name'], 
            ha='center', va='top', transform=ax.transAxes,
            fontsize=FONT_CONFIG['size_figure_title'],
            fontweight='bold', color=PALETTE['text'])
    
    # Plot motif (centered, larger)
    plot_single_motif(ax, motif, (0.5, 0.55), width=0.4)
    
    # Description
    desc = motif.get('description', '')
    if len(desc) > 50:
        desc = desc[:47] + '...'
    
    ax.text(0.5, 0.18, desc,
            ha='center', va='top', transform=ax.transAxes,
            fontsize=FONT_CONFIG['size_annotation'],
            color=PALETTE['text_secondary'],
            wrap=True)
    
    # Clinical relevance
    relevance = motif.get('clinical_relevance', '')
    if relevance:
        ax.text(0.5, 0.08, f'Clinical: {relevance[:40]}',
                ha='center', va='top', transform=ax.transAxes,
                fontsize=FONT_CONFIG['size_annotation'] - 0.5,
                color=PALETTE['text_secondary'], style='italic')
    
    # Source
    source = motif.get('source', 'MIBI-TOF')
    ax.text(0.5, 0.02, f'Source: {source}',
            ha='center', va='bottom', transform=ax.transAxes,
            fontsize=5, color=PALETTE['grid'])
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    save_figure(fig, panel_name, output_dir)
    plt.close(fig)


def create_interaction_heatmap(data_path, output_dir):
    """
    Create cell-cell interaction heatmap from real data.
    """
    import pandas as pd
    import seaborn as sns
    
    setup_nature_style()
    
    # Try to load interaction data
    interaction_file = data_path / 'cell_type_interactions_real.csv'
    if not interaction_file.exists():
        print(f"⚠️ Interaction data not found: {interaction_file}")
        return
    
    df = pd.read_csv(interaction_file)
    
    # Pivot to matrix
    pivot = df.pivot(index='cell_type_1', columns='cell_type_2', values='zscore')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Custom colormap: blue (attraction) - white - red (repulsion)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('interaction',
        [PALETTE['spatial_main'], 'white', PALETTE['high_risk']])
    
    # Heatmap
    vmax = np.abs(pivot.values).max()
    sns.heatmap(pivot, cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
                square=True, ax=ax, 
                cbar_kws={'label': 'Z-score\n(negative = co-localization)',
                          'shrink': 0.6},
                linewidths=0.5, linecolor='white')
    
    ax.set_title('Cell-Cell Spatial Interactions\nMIBI-TOF Breast Cancer',
                 fontsize=FONT_CONFIG['size_figure_title'],
                 fontweight='bold', pad=15)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.xticks(rotation=45, ha='right', fontsize=FONT_CONFIG['size_axis_tick'])
    plt.yticks(rotation=0, fontsize=FONT_CONFIG['size_axis_tick'])
    
    plt.tight_layout()
    save_figure(fig, 'fig4_supp_interaction_heatmap', output_dir)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Generate Nature-quality spatial motif visualizations'
    )
    parser.add_argument(
        '--data_dir', type=str, 
        default='../../data/processed',
        help='Path to processed data directory'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='../../figures/fig4_mechanism',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--max_motifs', type=int, default=3,
        help='Maximum number of motifs to display'
    )
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 65)
    print("  Spatial Motif Visualization (Nature Biotechnology Standard)")
    print("=" * 65)
    
    # Load motifs
    motifs_file = data_path / 'spatial_motifs_real.json'
    if not motifs_file.exists():
        print(f"❌ Motifs file not found: {motifs_file}")
        print("   Please run download_and_analyze_spatial.py first.")
        return
    
    with open(motifs_file, 'r') as f:
        motifs = json.load(f)
    
    print(f"\n📊 Loaded {len(motifs)} spatial motifs")
    print(f"📁 Output directory: {output_dir.resolve()}")
    
    # Generate main figure
    print("\n🎨 Generating main motifs figure...")
    create_spatial_motifs_figure(motifs, output_dir, max_motifs=args.max_motifs)
    print("  ✓ Main figure complete")
    
    # Generate individual panels
    print("\n🎨 Generating individual motif panels...")
    for i, motif in enumerate(motifs[:3]):
        create_detailed_motif_panel(
            motif, output_dir / 'panels',
            f'motif_{motif["motif_id"]}'
        )
    print("  ✓ Individual panels complete")
    
    # Generate interaction heatmap
    print("\n🎨 Generating interaction heatmap...")
    create_interaction_heatmap(data_path, output_dir)
    
    print("\n" + "=" * 65)
    print("  ✅ Visualization Complete!")
    print("=" * 65)
    
    print("\n📄 Generated files:")
    for f in sorted(output_dir.glob("*.pdf")):
        print(f"   - {f.name}")


if __name__ == "__main__":
    main()





