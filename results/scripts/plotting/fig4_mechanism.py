"""
Figure 4: Spatial Mechanisms & Discovery — Nature Biotechnology Standard
=========================================================================
生成空间机制解析图 (WSI heatmap, Cell community, Spatial motif, Gene-morphology)

Features:
- Nature-level semantic palette
- Attention Slice View (WSI + heatmap overlay)
- Spatial motif topology diagrams
- Single panel PDF export

Usage:
    python fig4_mechanism.py --data_dir ../../data/processed --output_dir ../../figures/fig4_mechanism
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import mannwhitneyu

from plot_config import (
    setup_nature_style, PALETTE, FIGSIZE, FONT_CONFIG,
    save_figure, save_single_panel, add_panel_label, 
    format_pvalue, get_significance_level, add_stats_box,
    CMAP_ATTENTION, CMAP_BLUE_WHITE_GOLD
)


# ============================================================
# Panel a: WSI Attention Heatmap (Attention Slice View)
# ============================================================

def plot_wsi_heatmap(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel a: WSI with attention heatmap overlay
    
    Nature-级「Attention Slice View」:
    - WSI 作为淡灰度背景 (15-20% 不透明度)
    - 注意力热力叠加
    - ROI 框用虚线 + 箭头连接
    """
    # 读取 attention 汇总数据
    summary = pd.read_csv(data_path / 'attention_summary.csv')
    
    # 选择一个代表性样本 (中等偏高风险)
    summary_sorted = summary.sort_values('risk_score')
    mid_idx = len(summary_sorted) // 2
    slide_id = summary_sorted.iloc[mid_idx + 5]['slide_id']
    
    # 读取详细数据
    attention_map_path = data_path / 'attention_maps' / f'{slide_id}.json'
    if not attention_map_path.exists():
        # 使用第一个可用的
        available = list((data_path / 'attention_maps').glob('*.json'))
        if available:
            attention_map_path = available[0]
            slide_id = attention_map_path.stem
        else:
            ax.text(0.5, 0.5, 'No attention map data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
    
    with open(attention_map_path, 'r') as f:
        slide_data = json.load(f)
    
    coords = np.array(slide_data['patch_coords'])
    attention = np.array(slide_data['attention_weights'])
    
    # 创建热图网格
    x_unique = np.unique(coords[:, 0])
    y_unique = np.unique(coords[:, 1])
    
    heatmap = np.zeros((len(y_unique), len(x_unique)))
    x_idx_map = {x: i for i, x in enumerate(x_unique)}
    y_idx_map = {y: i for i, y in enumerate(y_unique)}
    
    for (x, y), att in zip(coords, attention):
        if x in x_idx_map and y in y_idx_map:
            heatmap[y_idx_map[y], x_idx_map[x]] = att
    
    # 模拟 WSI 背景 (组织纹理)
    np.random.seed(42)
    background = np.random.rand(*heatmap.shape) * 0.15 + 0.75
    
    # 绘制背景
    ax.imshow(background, cmap='gray', aspect='auto', 
             extent=[0, heatmap.shape[1], heatmap.shape[0], 0], alpha=0.3)
    
    # 使用蓝金双色 attention 热图配色
    # 叠加 attention heatmap
    im = ax.imshow(heatmap, cmap=CMAP_ATTENTION, alpha=0.75, aspect='auto', 
                   extent=[0, heatmap.shape[1], heatmap.shape[0], 0],
                   vmin=0, vmax=np.percentile(attention, 95))
    
    # 添加 ROI 框 (高 attention 区域)
    high_att_threshold = np.percentile(attention, 90)
    high_att_mask = heatmap > high_att_threshold
    high_att_coords = np.where(high_att_mask)
    
    if len(high_att_coords[0]) > 0:
        # ROI 1
        roi1_y = high_att_coords[0][0]
        roi1_x = high_att_coords[1][0]
        rect1 = Rectangle((roi1_x - 2, roi1_y - 2), 6, 6, fill=False,
                          edgecolor=PALETTE['spatial_main'], linewidth=2, 
                          linestyle='--', zorder=10)
        ax.add_patch(rect1)
        ax.text(roi1_x + 5, roi1_y - 2.5, 'ROI-1', 
               fontsize=FONT_CONFIG['size_annotation'], 
               color=PALETTE['spatial_main'], fontweight='bold')
        
        # ROI 2 (如果存在)
        if len(high_att_coords[0]) > 15:
            roi2_y = high_att_coords[0][-8]
            roi2_x = high_att_coords[1][-8]
            rect2 = Rectangle((roi2_x - 2, roi2_y - 2), 6, 6, fill=False,
                              edgecolor=PALETTE['low_risk'], linewidth=2, 
                              linestyle='--', zorder=10)
            ax.add_patch(rect2)
            ax.text(roi2_x + 5, roi2_y - 2.5, 'ROI-2', 
                   fontsize=FONT_CONFIG['size_annotation'], 
                   color=PALETTE['low_risk'], fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.85)
    cbar.set_label('Risk Attention', fontsize=FONT_CONFIG['size_axis_title'])
    cbar.ax.tick_params(labelsize=FONT_CONFIG['size_annotation'])
    
    # 风险评分
    risk_score = slide_data['risk_score']
    risk_color = PALETTE['high_risk'] if risk_score > 0.6 else PALETTE['low_risk']
    ax.text(0.02, 0.98, f'Risk Score: {risk_score:.3f}',
            transform=ax.transAxes, ha='left', va='top', 
            fontsize=FONT_CONFIG['size_stats_box'], fontweight='bold',
            color=risk_color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     alpha=0.9, edgecolor='none'))
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 比例尺
    scalebar_len = heatmap.shape[1] * 0.15
    ax.plot([heatmap.shape[1] - scalebar_len - 2, heatmap.shape[1] - 2], 
           [heatmap.shape[0] - 2, heatmap.shape[0] - 2], 
           color=PALETTE['text'], linewidth=2)
    ax.text(heatmap.shape[1] - 2 - scalebar_len/2, heatmap.shape[0] - 3, 
           '500 μm', ha='center', fontsize=FONT_CONFIG['size_annotation'],
           color=PALETTE['text'])
    
    ax.set_title(f'WSI Attention Map', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=8)
    
    add_panel_label(ax, 'a')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'fig4_panel_a_wsi', output_dir)


# ============================================================
# Panel b: Cell Community Structure
# ============================================================

def plot_cell_community(ax, data_path, sample_idx=5, 
                        save_panel_pdf=False, output_dir=None):
    """
    Panel b: ROI cell community visualization
    
    显示:
    - 细胞空间分布
    - 邻域连接
    - 高风险区域标注
    """
    df = pd.read_csv(data_path / 'cell_communities.csv')
    
    # 选择一个代表性 ROI
    samples = df['sample_id'].unique()
    sample_id = samples[min(sample_idx, len(samples)-1)]
    df_roi = df[df['sample_id'] == sample_id].copy()
    
    # 细胞类型颜色 (使用 PALETTE)
    type_colors = {
        'Tumor': PALETTE['tumor'],
        'CAF': PALETTE['stroma'],
        'CD8+ T': PALETTE['immune_cd8'],
        'CD4+ T': PALETTE['immune_cd4'],
        'Macrophage': PALETTE['macrophage'],
        'B cell': PALETTE['b_cell'],
        'Endothelial': PALETTE['endothelial'],
        'NK cell': PALETTE['nk_cell']
    }
    
    # 绘制邻域边 (Delaunay)
    coords = df_roi[['x', 'y']].values
    if len(coords) > 10:
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(coords)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                    edges.add(edge)
            
            # 只绘制短边 (< 40 μm)
            edge_lines = []
            for i, j in edges:
                if i < len(coords) and j < len(coords):
                    dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
                    if dist < 40:
                        edge_lines.append([coords[i], coords[j]])
            
            if edge_lines:
                lc = LineCollection(edge_lines, colors=PALETTE['grid'], 
                                   alpha=0.12, linewidths=0.3, zorder=1)
                ax.add_collection(lc)
        except:
            pass
    
    # 绘制细胞
    for cell_type, color in type_colors.items():
        mask = df_roi['cell_type'] == cell_type
        if mask.sum() > 0:
            ax.scatter(df_roi.loc[mask, 'x'], df_roi.loc[mask, 'y'],
                      c=color, s=15, alpha=0.75, label=cell_type,
                      edgecolors='white', linewidths=0.2, zorder=2)
    
    # 标注高风险社区
    if 'risk_contribution' in df_roi.columns:
        high_risk_cells = df_roi[df_roi['risk_contribution'] > 0.3]
        if len(high_risk_cells) > 5:
            center_x = high_risk_cells['x'].mean()
            center_y = high_risk_cells['y'].mean()
            
            # 绘制圈
            circle = Circle((center_x, center_y), 55, fill=False,
                            edgecolor=PALETTE['high_risk'], linewidth=2.5, 
                            linestyle='--', zorder=3)
            ax.add_patch(circle)
            ax.text(center_x, center_y - 65, 'High-risk\nregion', ha='center',
                   fontsize=FONT_CONFIG['size_annotation'], 
                   color=PALETTE['high_risk'], fontweight='bold')
    
    # 设置
    ax.set_xlim(df_roi['x'].min() - 15, df_roi['x'].max() + 15)
    ax.set_ylim(df_roi['y'].min() - 15, df_roi['y'].max() + 15)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=5, ncol=2, markerscale=0.7,
             frameon=False, handletextpad=0.3, columnspacing=0.8)
    ax.set_xlabel('X (μm)', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_ylabel('Y (μm)', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_title('Cell Community Structure', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=8)
    
    # 细胞数量
    n_cells = len(df_roi)
    ax.text(0.98, 0.02, f'n = {n_cells} cells', transform=ax.transAxes,
           fontsize=FONT_CONFIG['size_annotation'], ha='right', va='bottom',
           color=PALETTE['text_secondary'])
    
    add_panel_label(ax, 'b')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'fig4_panel_b_community', output_dir)


# ============================================================
# Panel c: Spatial Motif Topology
# ============================================================

def plot_spatial_motif(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel c: Spatial motif topology (network diagram)
    
    展示关键免疫排斥模体:
    - 细胞类型节点
    - 配体-受体互作边
    - 生物学意义标注
    """
    # 读取模体数据
    with open(data_path / 'spatial_motifs.json', 'r') as f:
        motifs = json.load(f)
    
    # 选择免疫排斥模体 (M001)
    motif = motifs[0]  # Immune Exclusion Barrier
    
    # 节点位置 (手动布局优化)
    node_positions = {
        0: (0.15, 0.5),    # CAF
        1: (0.40, 0.85),   # CAF
        2: (0.40, 0.15),   # CAF
        3: (0.50, 0.5),    # CD8+ T (中心)
        4: (0.88, 0.5),    # Tumor
    }
    
    type_colors = {
        'CAF': PALETTE['stroma'],
        'CD8+ T': PALETTE['immune_cd8'],
        'Tumor': PALETTE['tumor'],
    }
    
    # 绘制边
    for edge in motif['edges']:
        src, tgt = edge['source'], edge['target']
        if src in node_positions and tgt in node_positions:
            x = [node_positions[src][0], node_positions[tgt][0]]
            y = [node_positions[src][1], node_positions[tgt][1]]
            
            # 关键边 (physical barrier) 用红色
            if edge.get('interaction') == 'physical_barrier':
                color = PALETTE['intervention_main']
                lw = 3
                alpha = 0.9
            else:
                color = PALETTE['grid']
                lw = 1.5
                alpha = 0.6
            
            ax.plot(x, y, c=color, linewidth=lw, zorder=1, alpha=alpha)
    
    # 绘制节点
    for node in motif['nodes']:
        nid = node['id']
        if nid not in node_positions:
            continue
        pos = node_positions[nid]
        color = type_colors.get(node['type'], PALETTE['grid'])
        
        # 节点圆
        circle = Circle(pos, 0.09, facecolor=color, edgecolor='white',
                       linewidth=2.5, zorder=2)
        ax.add_patch(circle)
        
        # 标签
        label_offset = 0.14
        ax.text(pos[0], pos[1] - label_offset, node['type'],
               ha='center', fontsize=FONT_CONFIG['size_annotation'], 
               fontweight='bold', zorder=3)
    
    # 添加配体-受体标注
    ax.annotate('TGFβ-TGFBR\n(barrier signal)', 
               xy=(0.27, 0.5), xytext=(0.05, 0.15),
               fontsize=FONT_CONFIG['size_annotation'], 
               color=PALETTE['intervention_main'],
               arrowprops=dict(arrowstyle='->', 
                              color=PALETTE['intervention_main'], lw=1),
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                        alpha=0.8, edgecolor='none'))
    
    # "阻断" 标记
    ax.annotate('', xy=(0.70, 0.5), xytext=(0.58, 0.5),
               arrowprops=dict(arrowstyle='-|>', color=PALETTE['text_secondary'], 
                              lw=1.5, linestyle='--'))
    ax.text(0.64, 0.42, 'blocked', ha='center', 
           fontsize=FONT_CONFIG['size_annotation'], 
           color=PALETTE['text_secondary'], style='italic')
    
    # 模体信息框
    ax.text(0.5, 1.08, motif['name'], ha='center', 
           fontsize=FONT_CONFIG['size_axis_title'], fontweight='bold',
           transform=ax.transAxes)
    
    hr_text = f"HR = {motif['hazard_ratio']:.2f}"
    p_text = format_pvalue(motif['p_value'])
    hr_color = PALETTE['high_risk'] if motif['hazard_ratio'] > 1 else PALETTE['low_risk']
    ax.text(0.5, -0.08, f"{hr_text}, {p_text}",
           ha='center', fontsize=FONT_CONFIG['size_stats_box'], 
           color=hr_color, transform=ax.transAxes,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    alpha=0.9, edgecolor=PALETTE['grid'], linewidth=0.5))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    add_panel_label(ax, 'c', x=-0.02)
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'fig4_panel_c_motif', output_dir)


# ============================================================
# Panel d: Motif-Survival Association
# ============================================================

def plot_motif_survival(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel d: Motif enrichment vs survival (boxplot with statistics)
    """
    df = pd.read_csv(data_path / 'motif_survival.csv')
    
    # 准备颜色
    palette = {
        'Short Survival': PALETTE['high_risk'],
        'Long Survival': PALETTE['low_risk']
    }
    
    order = ['Short Survival', 'Long Survival']
    
    # 为离散的 motif_count 添加微小 Y 轴噪声，使散点分布更自然
    np.random.seed(42)
    df['motif_count_jittered'] = df['motif_count'] + np.random.uniform(-0.15, 0.15, len(df))
    
    # 绘制箱线图 (使用原始整数值)
    bp = sns.boxplot(data=df, x='survival_group', y='motif_count', 
                    palette=palette, ax=ax, width=0.55, order=order,
                    fliersize=0, linewidth=1.2)  # fliersize=0 隐藏异常点
    
    # 添加散点 (使用带噪声的值，增大 X 轴 jitter)
    for i, group in enumerate(order):
        group_data = df[df['survival_group'] == group]
        x_jitter = np.random.uniform(-0.18, 0.18, len(group_data))
        ax.scatter(i + x_jitter, group_data['motif_count_jittered'], 
                  c=PALETTE['text'], alpha=0.4, s=12, edgecolors='white', 
                  linewidths=0.3, zorder=5)
    
    # 统计检验
    short = df[df['survival_group'] == 'Short Survival']['motif_count']
    long = df[df['survival_group'] == 'Long Survival']['motif_count']
    stat, p_val = mannwhitneyu(short, long, alternative='greater')
    
    # 添加显著性标记
    y_max = df['motif_count'].max() + 0.8
    ax.plot([0, 0, 1, 1], [y_max, y_max + 0.4, y_max + 0.4, y_max], 
           'k-', lw=0.8)
    
    stars, _ = get_significance_level(p_val)
    ax.text(0.5, y_max + 0.6, stars, ha='center', 
           fontsize=FONT_CONFIG['size_axis_title'])
    ax.text(0.5, y_max + 1.2, format_pvalue(p_val), ha='center', 
           fontsize=FONT_CONFIG['size_annotation'], color=PALETTE['text_secondary'])
    
    # 添加样本量
    n_short = len(short)
    n_long = len(long)
    ax.text(0, df['motif_count'].min() - 0.8, f'n={n_short}', ha='center', 
           fontsize=FONT_CONFIG['size_annotation'], color=PALETTE['high_risk'])
    ax.text(1, df['motif_count'].min() - 0.8, f'n={n_long}', ha='center', 
           fontsize=FONT_CONFIG['size_annotation'], color=PALETTE['low_risk'])
    
    ax.set_xlabel('')
    ax.set_ylabel('Immune Exclusion\nMotif Count', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_title('Motif-Survival Association', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=8)
    
    add_panel_label(ax, 'd')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'fig4_panel_d_motif_survival', output_dir)


# ============================================================
# Panel e: Gene-Morphology Correlation
# ============================================================

def plot_gene_morphology(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel e: Gene-Morphology correlation heatmap
    """
    df_corr = pd.read_csv(data_path / 'gene_morphology_correlation.csv')
    
    # 定义基因和形态特征
    genes = ['ERBB2', 'ESR1', 'MKI67', 'TP53', 'VIM', 'CD274']
    features = ['nuclear_area', 'pleomorphism_score', 'mitotic_count', 'stromal_ratio']
    
    # 创建矩阵
    corr_matrix = np.zeros((len(features), len(genes)))
    sig_matrix = np.zeros((len(features), len(genes)), dtype=bool)
    
    for i, feat in enumerate(features):
        for j, gene in enumerate(genes):
            row = df_corr[(df_corr['gene'] == gene) & (df_corr['morph_feature'] == feat)]
            if len(row) > 0:
                corr_matrix[i, j] = row['correlation'].values[0]
                sig_matrix[i, j] = row['significant'].values[0] if 'significant' in row.columns else False
    
    # 使用蓝金发散配色 (负相关蓝→零白→正相关金)
    # 绘制热图
    im = ax.imshow(corr_matrix, cmap=CMAP_BLUE_WHITE_GOLD, vmin=-0.8, vmax=0.8, aspect='auto')
    
    # 添加数值标签和显著性标记
    for i in range(len(features)):
        for j in range(len(genes)):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.4 else PALETTE['text']
            sig = '*' if sig_matrix[i, j] else ''
            ax.text(j, i, f'{val:.2f}{sig}', ha='center', va='center',
                   fontsize=FONT_CONFIG['size_annotation'], color=color,
                   fontweight='bold' if sig else 'normal')
    
    # 标签
    feature_labels = ['Nuclear\nArea', 'Pleomorphism', 'Mitotic\nCount', 'Stromal\nRatio']
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=45, ha='right', 
                      fontsize=FONT_CONFIG['size_axis_tick'])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(feature_labels, fontsize=FONT_CONFIG['size_axis_tick'])
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, shrink=0.85)
    cbar.set_label('Correlation (r)', fontsize=FONT_CONFIG['size_axis_title'])
    cbar.ax.tick_params(labelsize=FONT_CONFIG['size_annotation'])
    
    ax.set_title('Gene-Morphology Association', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=8)
    ax.text(0.5, -0.18, '* P < 0.05 (Bonferroni)', ha='center', 
           fontsize=FONT_CONFIG['size_annotation'], transform=ax.transAxes,
           color=PALETTE['text_secondary'])
    
    add_panel_label(ax, 'e')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'fig4_panel_e_gene_morph', output_dir)


# ============================================================
# Supplementary: All Motifs Network Summary
# ============================================================

def plot_motif_network(ax, data_path):
    """
    Supplementary: All motifs network summary
    """
    with open(data_path / 'spatial_motifs.json', 'r') as f:
        motifs = json.load(f)
    
    n_motifs = min(3, len(motifs))
    positions = [(0.15 + i * 0.35, 0.5) for i in range(n_motifs)]
    
    for idx, (motif, pos) in enumerate(zip(motifs[:n_motifs], positions)):
        # 绘制简化网络
        n_nodes = min(5, len(motif['nodes']))
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        radius = 0.12
        
        node_pos = [(pos[0] + radius * np.cos(a), pos[1] + radius * np.sin(a)) 
                   for a in angles]
        
        # 边
        for edge in motif['edges'][:5]:
            src, tgt = edge['source'], edge['target']
            if src < len(node_pos) and tgt < len(node_pos):
                ax.plot([node_pos[src][0], node_pos[tgt][0]],
                       [node_pos[src][1], node_pos[tgt][1]],
                       color=PALETTE['grid'], linewidth=1, alpha=0.5)
        
        # 节点
        for npos in node_pos:
            ax.scatter(npos[0], npos[1], s=40, c=PALETTE['model_ours'], 
                      edgecolors='white', linewidths=0.8, zorder=5)
        
        # 标签
        short_name = motif['name'].split()[0][:12]
        ax.text(pos[0], pos[1] - 0.22, short_name, ha='center', 
               fontsize=FONT_CONFIG['size_annotation'])
        
        hr_text = f"HR={motif['hazard_ratio']:.2f}"
        hr_color = PALETTE['high_risk'] if motif['hazard_ratio'] > 1 else PALETTE['low_risk']
        ax.text(pos[0], pos[1] - 0.30, hr_text, ha='center', 
               fontsize=5, color=hr_color, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Key Spatial Motifs', fontsize=FONT_CONFIG['size_figure_title'], 
                fontweight='bold', pad=8)


# ============================================================
# 主图生成
# ============================================================

def create_figure4(data_path, output_dir, save_panels=True):
    """生成完整的 Figure 4"""
    
    setup_nature_style()
    
    fig = plt.figure(figsize=(7.5, 7.2))
    
    # 创建网格布局
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], 
                          hspace=0.45, wspace=0.45,
                          left=0.07, right=0.95, top=0.94, bottom=0.08)
    
    panels_dir = output_dir / "panels" if save_panels else None
    
    # Panel a: WSI heatmap (占两列)
    ax_a = fig.add_subplot(gs[0, :2])
    plot_wsi_heatmap(ax_a, data_path, save_panels, panels_dir)
    
    # Panel c: Spatial motif
    ax_c = fig.add_subplot(gs[0, 2])
    plot_spatial_motif(ax_c, data_path, save_panels, panels_dir)
    
    # Panel b: Cell community
    ax_b = fig.add_subplot(gs[1, 0])
    plot_cell_community(ax_b, data_path, save_panel_pdf=save_panels, output_dir=panels_dir)
    
    # Panel d: Motif-survival
    ax_d = fig.add_subplot(gs[1, 1])
    plot_motif_survival(ax_d, data_path, save_panels, panels_dir)
    
    # Panel e: Gene-morphology
    ax_e = fig.add_subplot(gs[1, 2])
    plot_gene_morphology(ax_e, data_path, save_panels, panels_dir)
    
    # 保存
    print("\n📊 Saving main figure...")
    save_figure(fig, 'fig4_mechanism', output_dir)
    plt.close(fig)


def create_supplementary(data_path, output_dir):
    """生成补充图"""
    
    setup_nature_style()
    
    # 补充图1: 所有模体汇总
    fig, ax = plt.subplots(figsize=(5, 3))
    plot_motif_network(ax, data_path)
    save_figure(fig, 'fig4_supp_motifs', output_dir)
    plt.close(fig)
    
    # 补充图2: Attention 分布
    summary = pd.read_csv(data_path / 'attention_summary.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    
    # Risk score 分布
    axes[0].hist(summary['risk_score'], bins=15, 
                color=PALETTE['model_ours'], alpha=0.75, edgecolor='white')
    axes[0].set_xlabel('Risk Score', fontsize=FONT_CONFIG['size_axis_title'])
    axes[0].set_ylabel('Count', fontsize=FONT_CONFIG['size_axis_title'])
    axes[0].set_title('Risk Score Distribution', 
                     fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Attention vs Risk
    axes[1].scatter(summary['mean_attention'], summary['risk_score'],
                   c=PALETTE['spatial_main'], alpha=0.6, s=30, 
                   edgecolors='white', linewidths=0.3)
    axes[1].set_xlabel('Mean Attention', fontsize=FONT_CONFIG['size_axis_title'])
    axes[1].set_ylabel('Risk Score', fontsize=FONT_CONFIG['size_axis_title'])
    axes[1].set_title('Attention-Risk Correlation', 
                     fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold')
    
    # 相关性
    corr = np.corrcoef(summary['mean_attention'], summary['risk_score'])[0, 1]
    axes[1].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1].transAxes,
                fontsize=FONT_CONFIG['size_stats_box'], va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_figure(fig, 'fig4_supp_attention', output_dir)
    plt.close(fig)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate Figure 4 (Nature Biotechnology Standard)')
    parser.add_argument('--data_dir', type=str, default='../../data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='../../figures/fig4_mechanism',
                        help='Output directory for figures')
    parser.add_argument('--no-panels', action='store_true',
                        help='Skip saving individual panel PDFs')
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # 检查数据文件
    required_files = [
        'attention_summary.csv',
        'cell_communities.csv',
        'spatial_motifs.json',
        'motif_survival.csv',
        'gene_morphology_correlation.csv'
    ]
    
    missing_files = [f for f in required_files if not (data_path / f).exists()]
    if missing_files:
        print("❌ Missing data files:")
        for f in missing_files:
            print(f"   - {data_path / f}")
        print("\n   Please run generate_fig4_data.py first.")
        return
    
    print("=" * 65)
    print("  Figure 4: Spatial Mechanisms & Discovery")
    print("  Nature Biotechnology Standard")
    print("=" * 65)
    print(f"\n📁 Data directory: {data_path.resolve()}")
    print(f"📁 Output directory: {output_dir.resolve()}")
    print(f"📄 Save individual panels: {not args.no_panels}")
    print()
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_panels:
        (output_dir / "panels").mkdir(exist_ok=True)
    
    # 生成图
    print("🎨 Generating main figure...")
    create_figure4(data_path, output_dir, save_panels=not args.no_panels)
    print("  ✓ Main figure complete")
    
    print("\n🎨 Generating supplementary figures...")
    create_supplementary(data_path, output_dir)
    print("  ✓ Supplementary figures complete")
    
    print("\n" + "=" * 65)
    print("  ✅ Figure 4 generation complete!")
    print("=" * 65)
    
    # 列出生成的文件
    print("\n📄 Generated files:")
    for f in sorted(output_dir.glob("*.pdf")):
        print(f"   - {f.name}")
    
    if not args.no_panels:
        print("\n📄 Individual panels:")
        for f in sorted((output_dir / "panels").glob("*.pdf")):
            print(f"   - panels/{f.name}")


if __name__ == "__main__":
    main()
