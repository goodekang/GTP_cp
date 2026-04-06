"""
Extended Data Figure 1: QC & Graph Construction — Nature Biotechnology Standard
================================================================================
生成数据质量控制和图构建相关的补充图

Panels:
- a: QC metrics violin plots by cohort
- b: Neighborhood threshold sensitivity  
- c: Hierarchical graph compression statistics

Usage:
    python ed_fig1_qc.py --data_dir ../../data/processed --output_dir ../../extended_data/ed_fig1_qc
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from plot_config import (
    setup_nature_style, PALETTE, FIGSIZE, FONT_CONFIG,
    save_figure, save_single_panel, add_panel_label, add_stats_box
)


# ============================================================
# Panel a: QC Metrics Violin Plots
# ============================================================

def plot_qc_violins(axes, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel a: QC metrics violin plots by cohort
    
    展示各队列的数据质量分布:
    - n_genes, n_cells, mito_pct, complexity
    """
    df = pd.read_csv(data_path / 'qc_metrics.csv')
    
    metrics = [
        ('n_genes', 'Number of Genes', 'Genes detected'),
        ('n_cells', 'Number of Cells', 'Cells per sample'),
        ('mito_pct', 'Mitochondrial %', 'Mito. fraction (%)'),
        ('complexity', 'Library Complexity', 'Complexity score'),
    ]
    
    # 队列颜色
    cohort_order = ['HTAN-BRCA', 'HTAN-CRC', '10x-Visium', '10x-Xenium', 
                   'TCGA-BRCA', 'TCGA-LUAD', 'TCGA-COAD', 'ICGC-LIRI']
    
    # 颜色: 空间数据用蓝绿系，临床数据用琥珀系
    cohort_colors = {
        'HTAN-BRCA': PALETTE['spatial_main'],
        'HTAN-CRC': PALETTE['spatial_light'],
        '10x-Visium': PALETTE['spatial_dark'],
        '10x-Xenium': PALETTE['causal_main'],
        'TCGA-BRCA': PALETTE['clinical_main'],
        'TCGA-LUAD': PALETTE['clinical_light'],
        'TCGA-COAD': PALETTE['clinical_dark'],
        'ICGC-LIRI': PALETTE['baseline_3'],
    }
    
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        # 准备数据
        plot_data = df[df['cohort'].isin(cohort_order)].copy()
        
        # 绘制小提琴图
        parts = ax.violinplot(
            [plot_data[plot_data['cohort'] == c][metric].dropna().values 
             for c in cohort_order if c in plot_data['cohort'].values],
            positions=range(len([c for c in cohort_order if c in plot_data['cohort'].values])),
            showmeans=True, showmedians=False, showextrema=False
        )
        
        # 设置颜色
        available_cohorts = [c for c in cohort_order if c in plot_data['cohort'].values]
        for i, (pc, cohort) in enumerate(zip(parts['bodies'], available_cohorts)):
            pc.set_facecolor(cohort_colors[cohort])
            pc.set_alpha(0.7)
            pc.set_edgecolor('white')
            pc.set_linewidth(0.5)
        
        # 均值线
        parts['cmeans'].set_color(PALETTE['text'])
        parts['cmeans'].set_linewidth(1)
        
        # 添加散点 (jittered)
        for i, cohort in enumerate(available_cohorts):
            cohort_data = plot_data[plot_data['cohort'] == cohort][metric].values
            jitter = np.random.normal(0, 0.08, len(cohort_data))
            ax.scatter(np.full(len(cohort_data), i) + jitter, cohort_data,
                      s=3, alpha=0.3, c=cohort_colors[cohort], edgecolors='none')
        
        # 设置
        ax.set_xticks(range(len(available_cohorts)))
        ax.set_xticklabels([c.split('-')[-1] if 'TCGA' in c or 'ICGC' in c else c 
                          for c in available_cohorts], 
                         rotation=45, ha='right', fontsize=FONT_CONFIG['size_annotation'])
        ax.set_ylabel(ylabel, fontsize=FONT_CONFIG['size_axis_title'])
        ax.set_title(title, fontsize=FONT_CONFIG['size_axis_title'], fontweight='bold')
        
        # QC 阈值线 (如适用)
        if metric == 'mito_pct':
            ax.axhline(y=20, color=PALETTE['intervention_main'], linestyle='--', 
                      linewidth=1, alpha=0.7, label='QC threshold')
        elif metric == 'complexity':
            ax.axhline(y=0.5, color=PALETTE['intervention_main'], linestyle='--', 
                      linewidth=1, alpha=0.7)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 添加总体标签
    axes[0].text(-0.15, 1.15, 'a', transform=axes[0].transAxes,
                fontsize=FONT_CONFIG['size_panel_label'], fontweight='bold')
    
    if save_panel_pdf and output_dir:
        # 保存整个 panel a
        fig_temp = plt.figure(figsize=(7, 5))
        gs_temp = fig_temp.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
        # 这里需要重新绑图，简化处理
        pass


# ============================================================
# Panel b: Neighborhood Threshold Sensitivity
# ============================================================

def plot_neighbor_sensitivity(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel b: Neighborhood threshold sensitivity analysis
    
    双Y轴显示:
    - C-index vs threshold
    - Memory usage vs threshold
    """
    df = pd.read_csv(data_path / 'neighbor_sensitivity.csv')
    
    ax2 = ax.twinx()
    
    # C-index (左轴) - 主数据
    line1 = ax.fill_between(df['threshold_um'], df['cindex_lower'], df['cindex_upper'],
                           alpha=0.2, color=PALETTE['model_ours'])
    line1_main, = ax.plot(df['threshold_um'], df['cindex_mean'], 'o-',
                         color=PALETTE['model_ours'], linewidth=2, markersize=6,
                         markeredgecolor='white', markeredgewidth=0.5, label='C-index')
    
    # 标记最优点
    optimal = df[df['optimal']]
    if len(optimal) > 0:
        ax.scatter(optimal['threshold_um'], optimal['cindex_mean'],
                  s=100, c=PALETTE['intervention_main'], marker='D', 
                  zorder=10, edgecolors='white', linewidths=1, label='Optimal')
        ax.annotate(f"Optimal\n({optimal['threshold_um'].values[0]} μm)",
                   xy=(optimal['threshold_um'].values[0], optimal['cindex_mean'].values[0]),
                   xytext=(optimal['threshold_um'].values[0] + 30, 
                          optimal['cindex_mean'].values[0] + 0.02),
                   fontsize=FONT_CONFIG['size_annotation'],
                   arrowprops=dict(arrowstyle='->', color=PALETTE['text_secondary'], lw=0.8))
    
    # Memory (右轴)
    line2, = ax2.plot(df['threshold_um'], df['memory_gb'], 's--',
                     color=PALETTE['clinical_main'], linewidth=1.5, markersize=5,
                     alpha=0.7, label='GPU Memory')
    
    # 设置左轴
    ax.set_xlabel('Neighborhood Threshold (μm)', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_ylabel('C-index', color=PALETTE['model_ours'], fontsize=FONT_CONFIG['size_axis_title'])
    ax.tick_params(axis='y', labelcolor=PALETTE['model_ours'])
    ax.set_ylim(0.68, 0.76)
    
    # 设置右轴
    ax2.set_ylabel('GPU Memory (GB)', color=PALETTE['clinical_main'], 
                  fontsize=FONT_CONFIG['size_axis_title'])
    ax2.tick_params(axis='y', labelcolor=PALETTE['clinical_main'])
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(PALETTE['clinical_main'])
    ax2.spines['top'].set_visible(False)
    
    # 图例
    lines = [line1_main, line2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='lower right', frameon=False, 
             fontsize=FONT_CONFIG['size_legend'])
    
    ax.set_title('Neighborhood Threshold Sensitivity', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=10)
    
    add_panel_label(ax, 'b')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'ed_fig1_panel_b_neighbor', output_dir)


# ============================================================
# Panel c: Hierarchical Graph Compression
# ============================================================

def plot_compression_stats(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel c: Hierarchical graph compression statistics
    
    堆叠条形图显示各层级压缩率
    """
    df = pd.read_csv(data_path / 'compression_stats.csv')
    
    levels = df['level'].tolist()
    x_pos = range(len(levels))
    
    # 颜色
    level_colors = {
        'Gene': PALETTE['level_gene'],
        'Cell': PALETTE['level_cell'],
        'Patch': PALETTE['level_patch'],
        'Region': PALETTE['clinical_light'],
        'Slide': PALETTE['clinical_main'],
    }
    
    # 绘制原始节点数 (浅色)
    bars1 = ax.barh(x_pos, np.log10(df['n_nodes_original']), 
                   color=[level_colors[l] for l in levels],
                   alpha=0.3, height=0.6, label='Original')
    
    # 绘制压缩后节点数 (深色)
    bars2 = ax.barh(x_pos, np.log10(df['n_nodes_compressed']), 
                   color=[level_colors[l] for l in levels],
                   alpha=0.9, height=0.6, label='Compressed')
    
    # 添加数值标签
    for i, row in df.iterrows():
        # 原始数
        ax.text(np.log10(row['n_nodes_original']) + 0.1, i, 
               f"{row['n_nodes_original']:,}", va='center',
               fontsize=FONT_CONFIG['size_annotation'], color=PALETTE['text_secondary'])
        # 压缩后
        ax.text(np.log10(row['n_nodes_compressed']) - 0.1, i, 
               f"{row['n_nodes_compressed']:,}", va='center', ha='right',
               fontsize=FONT_CONFIG['size_annotation'], color='white', fontweight='bold')
        # 压缩率
        ax.text(5.5, i, f"{row['compression_ratio']:.0f}x", va='center',
               fontsize=FONT_CONFIG['size_annotation'], color=PALETTE['model_ours'],
               fontweight='bold')
    
    # 设置
    ax.set_yticks(x_pos)
    ax.set_yticklabels(levels, fontsize=FONT_CONFIG['size_axis_tick'])
    ax.set_xlabel(r'Number of Nodes ($\log_{10}$)', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_xlim(0, 6)
    
    # 添加信息保留率
    ax.text(0.98, 0.02, 'Info retained shown by intensity',
           transform=ax.transAxes, ha='right', va='bottom',
           fontsize=FONT_CONFIG['size_annotation'], color=PALETTE['text_secondary'],
           style='italic')
    
    ax.set_title('Hierarchical Graph Compression', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=10)
    ax.legend(loc='lower right', frameon=False, fontsize=FONT_CONFIG['size_legend'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    add_panel_label(ax, 'c')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'ed_fig1_panel_c_compression', output_dir)


# ============================================================
# 主图生成
# ============================================================

def create_ed_figure1(data_path, output_dir, save_panels=True):
    """生成 Extended Data Figure 1"""
    
    setup_nature_style()
    
    fig = plt.figure(figsize=(7.5, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.8, 0.8], 
                          hspace=0.45, wspace=0.40,
                          left=0.10, right=0.95, top=0.95, bottom=0.06)
    
    panels_dir = output_dir / "panels" if save_panels else None
    if panels_dir:
        panels_dir.mkdir(exist_ok=True)
    
    # Panel a: QC violins (占两列)
    gs_a = gs[0, :].subgridspec(2, 2, hspace=0.35, wspace=0.30)
    axes_a = [fig.add_subplot(gs_a[i, j]) for i in range(2) for j in range(2)]
    plot_qc_violins(axes_a, data_path, save_panels, panels_dir)
    
    # Panel b: Neighbor sensitivity
    ax_b = fig.add_subplot(gs[1, 0])
    plot_neighbor_sensitivity(ax_b, data_path, save_panels, panels_dir)
    
    # Panel c: Compression stats
    ax_c = fig.add_subplot(gs[1, 1])
    plot_compression_stats(ax_c, data_path, save_panels, panels_dir)
    
    # 保存
    print("\n📊 Saving ED Figure 1...")
    save_figure(fig, 'ed_fig1_qc', output_dir)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate ED Figure 1')
    parser.add_argument('--data_dir', type=str, default='../../data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='../../extended_data/ed_fig1_qc',
                        help='Output directory for figures')
    parser.add_argument('--no-panels', action='store_true',
                        help='Skip saving individual panel PDFs')
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    required_files = ['qc_metrics.csv', 'neighbor_sensitivity.csv', 'compression_stats.csv']
    missing = [f for f in required_files if not (data_path / f).exists()]
    if missing:
        print(f"❌ Missing: {missing}")
        print("   Run generate_extended_data.py first.")
        return
    
    print("=" * 65)
    print("  ED Figure 1: QC & Graph Construction")
    print("=" * 65)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    create_ed_figure1(data_path, output_dir, not args.no_panels)
    
    print("\n✅ ED Figure 1 complete!")


if __name__ == "__main__":
    main()

