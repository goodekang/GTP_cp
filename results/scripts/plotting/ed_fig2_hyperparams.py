"""
Extended Data Figure 2: Hyperparameters & Ablation — Nature Biotechnology Standard
===================================================================================
生成超参数搜索和消融实验的补充图

Panels:
- a: Backbone comparison (forest plot)
- b: Head × Layer grid search (heatmap)
- c: MTM masking ratio sweep

Usage:
    python ed_fig2_hyperparams.py --data_dir ../../data/processed --output_dir ../../extended_data/ed_fig2_hyperparams
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
    CMAP_BLUE_GOLD
)


# ============================================================
# Panel a: Backbone Comparison (Forest Plot)
# ============================================================

def plot_backbone_comparison(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel a: Backbone feature extractor comparison
    
    森林图显示各 backbone 的 C-index ± 95% CI
    """
    df = pd.read_csv(data_path / 'backbone_ablation.csv')
    df = df.sort_values('cindex_mean', ascending=True)
    
    backbones = df['backbone'].tolist()
    y_pos = range(len(backbones))
    
    for i, (_, row) in enumerate(df.iterrows()):
        is_best = row['backbone'] == 'ViT-H/14 (CONCH)'
        color = PALETTE['model_ours'] if is_best else PALETTE['baseline_2']
        
        # CI 线
        ax.plot([row['cindex_lower'], row['cindex_upper']], [i, i],
               color=color, linewidth=2.5 if is_best else 1.5, alpha=0.8)
        
        # 点
        marker = 'D' if is_best else 'o'
        ms = 9 if is_best else 6
        ax.scatter(row['cindex_mean'], i, c=color, s=ms**2, marker=marker,
                  zorder=5, edgecolors='white', linewidths=1)
        
        # 数值
        ax.text(row['cindex_upper'] + 0.008, i, 
               f"{row['cindex_mean']:.3f}", va='center',
               fontsize=FONT_CONFIG['size_annotation'],
               fontweight='bold' if is_best else 'normal',
               color=color)
        
        # 参数量和时间
        ax.text(0.53, i, f"{row['n_params_M']}M", va='center',
               fontsize=FONT_CONFIG['size_annotation'], color=PALETTE['text_secondary'])
    
    # 设置
    ax.set_yticks(y_pos)
    ax.set_yticklabels(backbones, fontsize=FONT_CONFIG['size_axis_tick'])
    ax.set_xlabel('C-index (mean ± 95% CI)', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_xlim(0.52, 0.82)
    
    # 随机基线
    ax.axvline(x=0.5, color=PALETTE['grid'], linestyle='--', linewidth=0.8, alpha=0.5)
    
    # 参数列标题
    ax.text(0.53, len(backbones) - 0.3, 'Params', fontsize=FONT_CONFIG['size_annotation'],
           color=PALETTE['text_secondary'], fontweight='bold')
    
    ax.set_title('Feature Backbone Comparison', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    add_panel_label(ax, 'a')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'ed_fig2_panel_a_backbone', output_dir)


# ============================================================
# Panel b: Head × Layer Grid Search
# ============================================================

def plot_head_layer_grid(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel b: Head × Layer grid search heatmap
    """
    df = pd.read_csv(data_path / 'head_layer_grid.csv')
    
    # Pivot for heatmap
    pivot = df.pivot(index='n_layers', columns='n_heads', values='cindex')
    pivot = pivot.sort_index(ascending=False)  # 让高层数在上
    
    # 使用蓝金双色热图配色
    # 绘制热图
    im = ax.imshow(pivot.values, cmap=CMAP_BLUE_GOLD, aspect='auto',
                  vmin=pivot.values.min() - 0.01, vmax=pivot.values.max() + 0.01)
    
    # 添加数值
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            # 检查是否是最优
            is_optimal = df[(df['n_layers'] == pivot.index[i]) & 
                           (df['n_heads'] == pivot.columns[j])]['optimal'].values[0]
            
            color = 'white' if val > (pivot.values.max() + pivot.values.min()) / 2 else PALETTE['text']
            fontweight = 'bold' if is_optimal else 'normal'
            
            text = f'{val:.3f}'
            if is_optimal:
                # 使用边框标记最优值 (更学术化)
                text = f'[{val:.3f}]'
            
            ax.text(j, i, text, ha='center', va='center',
                   fontsize=FONT_CONFIG['size_annotation'], color=color,
                   fontweight=fontweight)
    
    # 设置
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=FONT_CONFIG['size_axis_tick'])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=FONT_CONFIG['size_axis_tick'])
    ax.set_xlabel('Number of Heads', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_ylabel('Number of Layers', fontsize=FONT_CONFIG['size_axis_title'])
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.9)
    cbar.set_label('C-index', fontsize=FONT_CONFIG['size_axis_title'])
    cbar.ax.tick_params(labelsize=FONT_CONFIG['size_annotation'])
    
    ax.set_title('Transformer Architecture Search', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=10)
    
    add_panel_label(ax, 'b')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'ed_fig2_panel_b_grid', output_dir)


# ============================================================
# Panel c: MTM Masking Ratio Sweep
# ============================================================

def plot_masking_ratio(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel c: Masked Token Modeling masking ratio sweep
    
    双Y轴:
    - Downstream C-index (左)
    - Reconstruction loss (右)
    """
    df = pd.read_csv(data_path / 'masking_ratio.csv')
    
    ax2 = ax.twinx()
    
    # 标记最优区间
    optimal_mask = df['in_optimal_range']
    ax.axvspan(0.3, 0.5, alpha=0.15, color=PALETTE['low_risk'], 
               label='Optimal range')
    
    # Downstream C-index (左轴)
    line1, = ax.plot(df['masking_ratio'], df['downstream_cindex'], 'o-',
                    color=PALETTE['model_ours'], linewidth=2, markersize=7,
                    markeredgecolor='white', markeredgewidth=0.5, 
                    label='Downstream C-index')
    
    # 高亮最优区间的点
    optimal_df = df[optimal_mask]
    ax.scatter(optimal_df['masking_ratio'], optimal_df['downstream_cindex'],
              s=80, c=PALETTE['intervention_main'], marker='D', zorder=10,
              edgecolors='white', linewidths=0.8)
    
    # Reconstruction loss (右轴)
    line2, = ax2.plot(df['masking_ratio'], df['reconstruction_loss'], 's--',
                     color=PALETTE['clinical_main'], linewidth=1.5, markersize=5,
                     alpha=0.7, label='Recon. Loss')
    
    # 左轴设置
    ax.set_xlabel('Masking Ratio', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_ylabel('C-index', color=PALETTE['model_ours'], 
                 fontsize=FONT_CONFIG['size_axis_title'])
    ax.tick_params(axis='y', labelcolor=PALETTE['model_ours'])
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.68, 0.76)
    
    # 右轴设置
    ax2.set_ylabel('Reconstruction Loss', color=PALETTE['clinical_main'],
                  fontsize=FONT_CONFIG['size_axis_title'])
    ax2.tick_params(axis='y', labelcolor=PALETTE['clinical_main'])
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(PALETTE['clinical_main'])
    ax2.spines['top'].set_visible(False)
    
    # 图例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='lower left', frameon=False,
             fontsize=FONT_CONFIG['size_legend'])
    
    # 标注最佳点
    best_idx = df['downstream_cindex'].idxmax()
    best_ratio = df.loc[best_idx, 'masking_ratio']
    best_cindex = df.loc[best_idx, 'downstream_cindex']
    ax.annotate(f'Best: {best_ratio*100:.0f}%',
               xy=(best_ratio, best_cindex),
               xytext=(best_ratio + 0.15, best_cindex + 0.015),
               fontsize=FONT_CONFIG['size_annotation'],
               arrowprops=dict(arrowstyle='->', color=PALETTE['text_secondary'], lw=0.8))
    
    ax.set_title('MTM Masking Ratio Optimization', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=10)
    
    add_panel_label(ax, 'c')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'ed_fig2_panel_c_masking', output_dir)


# ============================================================
# 主图生成
# ============================================================

def create_ed_figure2(data_path, output_dir, save_panels=True):
    """生成 Extended Data Figure 2"""
    
    setup_nature_style()
    
    fig = plt.figure(figsize=(7.5, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], 
                          hspace=0.45, wspace=0.45,
                          left=0.10, right=0.95, top=0.95, bottom=0.08)
    
    panels_dir = output_dir / "panels" if save_panels else None
    if panels_dir:
        panels_dir.mkdir(exist_ok=True)
    
    # Panel a: Backbone comparison (左上)
    ax_a = fig.add_subplot(gs[0, 0])
    plot_backbone_comparison(ax_a, data_path, save_panels, panels_dir)
    
    # Panel b: Head × Layer grid (右上)
    ax_b = fig.add_subplot(gs[0, 1])
    plot_head_layer_grid(ax_b, data_path, save_panels, panels_dir)
    
    # Panel c: Masking ratio (下方两列)
    ax_c = fig.add_subplot(gs[1, :])
    plot_masking_ratio(ax_c, data_path, save_panels, panels_dir)
    
    # 保存
    print("\n📊 Saving ED Figure 2...")
    save_figure(fig, 'ed_fig2_hyperparams', output_dir)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate ED Figure 2')
    parser.add_argument('--data_dir', type=str, default='../../data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='../../extended_data/ed_fig2_hyperparams',
                        help='Output directory for figures')
    parser.add_argument('--no-panels', action='store_true')
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    required_files = ['backbone_ablation.csv', 'head_layer_grid.csv', 'masking_ratio.csv']
    missing = [f for f in required_files if not (data_path / f).exists()]
    if missing:
        print(f"❌ Missing: {missing}")
        print("   Run generate_extended_data.py first.")
        return
    
    print("=" * 65)
    print("  ED Figure 2: Hyperparameters & Ablation")
    print("=" * 65)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    create_ed_figure2(data_path, output_dir, not args.no_panels)
    
    print("\n✅ ED Figure 2 complete!")


if __name__ == "__main__":
    main()

