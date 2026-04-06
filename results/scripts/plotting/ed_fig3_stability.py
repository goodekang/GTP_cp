"""
Extended Data Figure 3: Interpretation Stability — Nature Biotechnology Standard
=================================================================================
生成模型解释稳定性分析的补充图

Panels:
- a: Attention entropy distribution by model variant
- b: Random perturbation stability analysis

Usage:
    python ed_fig3_stability.py --data_dir ../../data/processed --output_dir ../../extended_data/ed_fig3_stability
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
    save_figure, save_single_panel, add_panel_label, add_stats_box,
    format_pvalue, get_significance_level, CMAP_BLUE_GOLD
)


# ============================================================
# Panel a: Attention Entropy Distribution
# ============================================================

def plot_attention_entropy(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel a: Attention entropy distribution by model variant
    
    小提琴图 + 箱线图组合,显示:
    - 各模型变体的注意力熵分布
    - 更低的熵 = 更聚焦的注意力
    """
    df = pd.read_csv(data_path / 'attention_entropy.csv')
    
    # 只取 layer 3 (中间层)
    df_layer3 = df[df['layer'] == 3].copy()
    
    # 模型顺序 (按熵从低到高)
    variant_order = ['Causal-STFM', 'w/o Causal Mask', 'w/o Spatial', 'Random Init']
    
    # 颜色
    variant_colors = {
        'Causal-STFM': PALETTE['model_ours'],
        'w/o Causal Mask': PALETTE['baseline_2'],
        'w/o Spatial': PALETTE['baseline_3'],
        'Random Init': PALETTE['intervention_light'],
    }
    
    # 绘制小提琴图
    parts = ax.violinplot(
        [df_layer3[df_layer3['model_variant'] == v]['entropy'].values for v in variant_order],
        positions=range(len(variant_order)),
        showmeans=False, showmedians=False, showextrema=False
    )
    
    # 设置颜色
    for i, (pc, variant) in enumerate(zip(parts['bodies'], variant_order)):
        pc.set_facecolor(variant_colors[variant])
        pc.set_alpha(0.6)
        pc.set_edgecolor('white')
        pc.set_linewidth(0.5)
    
    # 添加箱线图
    bp = ax.boxplot(
        [df_layer3[df_layer3['model_variant'] == v]['entropy'].values for v in variant_order],
        positions=range(len(variant_order)),
        widths=0.15, patch_artist=True, showfliers=False
    )
    
    for i, (patch, variant) in enumerate(zip(bp['boxes'], variant_order)):
        patch.set_facecolor('white')
        patch.set_edgecolor(variant_colors[variant])
        patch.set_linewidth(1.5)
    
    for element in ['whiskers', 'caps']:
        for line in bp[element]:
            line.set_color(PALETTE['text_secondary'])
            line.set_linewidth(1)
    for median in bp['medians']:
        median.set_color(PALETTE['text'])
        median.set_linewidth(1.5)
    
    # 添加统计显著性 (与 Causal-STFM 比较)
    from scipy.stats import mannwhitneyu
    
    baseline = df_layer3[df_layer3['model_variant'] == 'Causal-STFM']['entropy']
    y_max = df_layer3['entropy'].max() + 0.08
    
    for i, variant in enumerate(variant_order[1:], 1):
        variant_data = df_layer3[df_layer3['model_variant'] == variant]['entropy']
        _, p_val = mannwhitneyu(baseline, variant_data, alternative='less')
        
        stars, _ = get_significance_level(p_val)
        ax.text(i, y_max, stars, ha='center', fontsize=FONT_CONFIG['size_axis_title'])
    
    # 设置
    ax.set_xticks(range(len(variant_order)))
    ax.set_xticklabels([v.replace('w/o ', '- ') for v in variant_order], 
                      fontsize=FONT_CONFIG['size_axis_tick'], rotation=15, ha='right')
    ax.set_ylabel('Attention Entropy', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_ylim(0.4, y_max + 0.1)
    
    # 标注 "Lower = More focused"
    ax.annotate('', xy=(0, 0.5), xytext=(0, 0.7),
               arrowprops=dict(arrowstyle='<-', color=PALETTE['low_risk'], lw=2))
    ax.text(-0.4, 0.6, 'More\nfocused', fontsize=FONT_CONFIG['size_annotation'],
           color=PALETTE['low_risk'], ha='center', fontweight='bold')
    
    ax.set_title('Attention Entropy by Model Variant', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 图例
    ax.text(0.98, 0.98, 'Layer 3 (middle)', transform=ax.transAxes,
           ha='right', va='top', fontsize=FONT_CONFIG['size_annotation'],
           color=PALETTE['text_secondary'], style='italic')
    
    add_panel_label(ax, 'a')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'ed_fig3_panel_a_entropy', output_dir)


# ============================================================
# Panel b: Perturbation Stability
# ============================================================

def plot_perturbation_stability(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel b: Random perturbation stability analysis
    
    热图显示:
    - 行: 扰动类型
    - 列: 扰动强度
    - 颜色: Rank correlation (越高越稳定)
    """
    df = pd.read_csv(data_path / 'perturbation_stability.csv')
    
    # Pivot for heatmap
    pivot = df.pivot(index='perturbation_type', columns='perturbation_level', 
                    values='rank_correlation')
    
    # 使用蓝金双色热图配色 (低稳定性→高稳定性)
    # 绘制热图
    im = ax.imshow(pivot.values, cmap=CMAP_BLUE_GOLD, aspect='auto', vmin=0.6, vmax=1.0)
    
    # 添加数值和稳定性标记
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            
            # 检查是否稳定
            pert_type = pivot.index[i]
            pert_level = pivot.columns[j]
            stable_query = df[(df['perturbation_type'] == pert_type) & 
                             (df['perturbation_level'] == pert_level)]
            is_stable = stable_query['stable'].values[0] if len(stable_query) > 0 else False
            
            color = 'white' if val < 0.8 else PALETTE['text']
            text = f'{val:.2f}'
            if is_stable:
                text = f'{val:.2f}*'
            
            ax.text(j, i, text, ha='center', va='center',
                   fontsize=FONT_CONFIG['size_annotation'], color=color)
    
    # 设置
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{l*100:.0f}%' for l in pivot.columns], 
                      fontsize=FONT_CONFIG['size_axis_tick'])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=FONT_CONFIG['size_axis_tick'])
    ax.set_xlabel('Perturbation Level', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_ylabel('Perturbation Type', fontsize=FONT_CONFIG['size_axis_title'])
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.9)
    cbar.set_label('Rank Correlation', fontsize=FONT_CONFIG['size_axis_title'])
    cbar.ax.tick_params(labelsize=FONT_CONFIG['size_annotation'])
    
    # 稳定性统计
    stable_rate = df['stable'].mean() * 100
    ax.text(0.02, 0.98, f'Stable conditions: {stable_rate:.0f}%',
           transform=ax.transAxes, ha='left', va='top',
           fontsize=FONT_CONFIG['size_annotation'],
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    ax.set_title('Perturbation Robustness Analysis', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=10)
    
    add_panel_label(ax, 'b')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'ed_fig3_panel_b_perturbation', output_dir)


# ============================================================
# Panel c: Entropy by Layer (补充)
# ============================================================

def plot_entropy_by_layer(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel c: Attention entropy across layers
    """
    df = pd.read_csv(data_path / 'attention_entropy.csv')
    
    # 计算各模型各层的平均熵
    summary = df.groupby(['model_variant', 'layer'])['entropy'].agg(['mean', 'std']).reset_index()
    
    variant_order = ['Causal-STFM', 'w/o Causal Mask', 'w/o Spatial', 'Random Init']
    variant_colors = {
        'Causal-STFM': PALETTE['model_ours'],
        'w/o Causal Mask': PALETTE['baseline_2'],
        'w/o Spatial': PALETTE['baseline_3'],
        'Random Init': PALETTE['intervention_light'],
    }
    
    for variant in variant_order:
        data = summary[summary['model_variant'] == variant]
        ax.errorbar(data['layer'], data['mean'], yerr=data['std'],
                   fmt='o-', color=variant_colors[variant], linewidth=1.5,
                   markersize=6, capsize=3, label=variant.replace('w/o ', '- '),
                   markeredgecolor='white', markeredgewidth=0.5)
    
    ax.set_xlabel('Layer', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_ylabel('Attention Entropy', fontsize=FONT_CONFIG['size_axis_title'])
    ax.legend(loc='upper left', frameon=False, fontsize=FONT_CONFIG['size_annotation'])
    ax.set_title('Entropy Across Layers', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    add_panel_label(ax, 'c')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'ed_fig3_panel_c_layer', output_dir)


# ============================================================
# 主图生成
# ============================================================

def create_ed_figure3(data_path, output_dir, save_panels=True):
    """生成 Extended Data Figure 3"""
    
    setup_nature_style()
    
    fig = plt.figure(figsize=(7.5, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], 
                          hspace=0.45, wspace=0.45,
                          left=0.10, right=0.95, top=0.95, bottom=0.08)
    
    panels_dir = output_dir / "panels" if save_panels else None
    if panels_dir:
        panels_dir.mkdir(exist_ok=True)
    
    # Panel a: Attention entropy
    ax_a = fig.add_subplot(gs[0, 0])
    plot_attention_entropy(ax_a, data_path, save_panels, panels_dir)
    
    # Panel b: Perturbation stability
    ax_b = fig.add_subplot(gs[0, 1])
    plot_perturbation_stability(ax_b, data_path, save_panels, panels_dir)
    
    # Panel c: Entropy by layer
    ax_c = fig.add_subplot(gs[1, :])
    plot_entropy_by_layer(ax_c, data_path, save_panels, panels_dir)
    
    # 保存
    print("\n📊 Saving ED Figure 3...")
    save_figure(fig, 'ed_fig3_stability', output_dir)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate ED Figure 3')
    parser.add_argument('--data_dir', type=str, default='../../data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='../../extended_data/ed_fig3_stability',
                        help='Output directory for figures')
    parser.add_argument('--no-panels', action='store_true')
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    required_files = ['attention_entropy.csv', 'perturbation_stability.csv']
    missing = [f for f in required_files if not (data_path / f).exists()]
    if missing:
        print(f"❌ Missing: {missing}")
        print("   Run generate_extended_data.py first.")
        return
    
    print("=" * 65)
    print("  ED Figure 3: Interpretation Stability")
    print("=" * 65)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    create_ed_figure3(data_path, output_dir, not args.no_panels)
    
    print("\n✅ ED Figure 3 complete!")


if __name__ == "__main__":
    main()

