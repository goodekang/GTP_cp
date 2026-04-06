"""
Extended Data Figure 4: Pan-cancer & Subgroup Analysis — Nature Biotechnology Standard
=======================================================================================
生成泛癌和亚组分析的补充图

Panels:
- a: Subgroup forest plot (HR + 95% CI)
- b: Cohort demographics table
- c: Cross-validation detailed results

Usage:
    python ed_fig4_pancancer.py --data_dir ../../data/processed --output_dir ../../extended_data/ed_fig4_pancancer
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
    format_pvalue, format_hr_ci
)


# ============================================================
# Panel a: Subgroup Forest Plot
# ============================================================

def plot_subgroup_forest(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel a: Subgroup forest plot
    
    显示各亚组的 HR + 95% CI
    """
    df = pd.read_csv(data_path / 'subgroup_hr.csv')
    
    # 按类别分组
    categories = df['subgroup_category'].unique()
    
    y_pos = 0
    y_positions = []
    y_labels = []
    category_positions = {}
    
    # 计算位置
    for cat in categories:
        cat_data = df[df['subgroup_category'] == cat]
        category_positions[cat] = y_pos + len(cat_data) / 2 - 0.5
        
        for _, row in cat_data.iterrows():
            y_positions.append(y_pos)
            y_labels.append(row['subgroup'])
            y_pos += 1
        y_pos += 0.5  # 类别间间隔
    
    # 绘制
    df_plot = df.copy()
    df_plot['y_pos'] = y_positions[:len(df)]
    
    # 参考线 (HR = 1)
    ax.axvline(x=1, color=PALETTE['grid'], linestyle='-', linewidth=1, zorder=1)
    
    # HR > 1 区域 (高风险)
    ax.axvspan(1, 4, alpha=0.05, color=PALETTE['high_risk'])
    
    for i, (_, row) in enumerate(df_plot.iterrows()):
        y = row['y_pos']
        
        # 确定颜色
        if row['subgroup'] == 'Overall':
            color = PALETTE['model_ours']
            lw = 3
            ms = 10
        elif row['significant']:
            color = PALETTE['high_risk'] if row['hr'] > 1 else PALETTE['low_risk']
            lw = 2
            ms = 7
        else:
            color = PALETTE['text_secondary']
            lw = 1.5
            ms = 6
        
        # CI 线
        ax.plot([row['hr_lower'], row['hr_upper']], [y, y],
               color=color, linewidth=lw, alpha=0.8)
        
        # 点
        marker = 'D' if row['subgroup'] == 'Overall' else 'o'
        ax.scatter(row['hr'], y, c=color, s=ms**2, marker=marker,
                  zorder=5, edgecolors='white', linewidths=1)
        
        # HR 数值
        ax.text(3.8, y, f"{row['hr']:.2f} ({row['hr_lower']:.2f}-{row['hr_upper']:.2f})",
               va='center', fontsize=FONT_CONFIG['size_annotation'],
               fontweight='bold' if row['subgroup'] == 'Overall' else 'normal')
        
        # N 值
        ax.text(0.35, y, f"n={row['n']}", va='center',
               fontsize=FONT_CONFIG['size_annotation'], color=PALETTE['text_secondary'])
    
    # 设置
    ax.set_yticks(y_positions[:len(df)])
    ax.set_yticklabels(y_labels, fontsize=FONT_CONFIG['size_axis_tick'])
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_xlim(0.3, 4.5)
    ax.set_xscale('log')
    ax.set_xticks([0.5, 1, 2, 4])
    ax.set_xticklabels(['0.5', '1', '2', '4'])
    ax.invert_yaxis()
    
    # 类别标签
    for cat, y_mid in category_positions.items():
        if cat != 'All':
            ax.text(-0.05, y_mid, cat, transform=ax.get_yaxis_transform(),
                   fontsize=FONT_CONFIG['size_annotation'], fontweight='bold',
                   color=PALETTE['text_secondary'], ha='right', va='center')
    
    # 列标题
    ax.text(3.8, -1, 'HR (95% CI)', fontsize=FONT_CONFIG['size_annotation'],
           fontweight='bold', va='center')
    
    ax.set_title('Subgroup Analysis', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    add_panel_label(ax, 'a')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'ed_fig4_panel_a_forest', output_dir)


# ============================================================
# Panel b: Cohort Demographics
# ============================================================

def plot_cohort_demographics(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel b: Cohort demographics table
    """
    df = pd.read_csv(data_path / 'cohort_demographics.csv')
    
    # 准备表格数据
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['cohort'],
            row['split'],
            f"{row['n']:,}",
            f"{row['age_mean']:.1f} ± {row['age_std']:.1f}",
            f"{row['female_pct']:.0f}%",
            f"{row['follow_up_median_months']:.0f}",
            f"{row['event_rate']*100:.0f}%"
        ])
    
    columns = ['Cohort', 'Split', 'N', 'Age', 'Female', 'F/U (mo)', 'Events']
    
    ax.axis('off')
    
    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=[PALETTE['model_ours']] * len(columns),
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_CONFIG['size_annotation'])
    table.scale(1.1, 1.5)
    
    # 设置样式
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor(PALETTE['model_ours'])
        else:
            # 根据 split 着色
            split_val = table_data[row-1][1]
            if split_val == 'external':
                cell.set_facecolor('#FFF3E0')  # 浅橙
            elif split_val == 'test':
                cell.set_facecolor('#E3F2FD')  # 浅蓝
            elif split_val == 'pretrain':
                cell.set_facecolor('#E8F5E9')  # 浅绿
            else:
                cell.set_facecolor('white')
        
        cell.set_edgecolor(PALETTE['grid'])
        cell.set_linewidth(0.5)
    
    ax.set_title('Cohort Demographics', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=15)
    
    add_panel_label(ax, 'b', x=-0.02)
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'ed_fig4_panel_b_demographics', output_dir)


# ============================================================
# Panel c: Cross-validation Results
# ============================================================

def plot_cv_results(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel c: 5-fold cross-validation detailed results
    """
    df = pd.read_csv(data_path / 'cv_detailed_results.csv')
    
    methods = ['Causal-STFM', 'MCAT', 'Porpoise', 'TransMIL', 'ABMIL']
    method_colors = {
        'Causal-STFM': PALETTE['model_ours'],
        'MCAT': PALETTE['baseline_1'],
        'Porpoise': PALETTE['baseline_2'],
        'TransMIL': PALETTE['baseline_3'],
        'ABMIL': PALETTE['baseline_4'],
    }
    
    # 计算每个方法的统计
    positions = range(len(methods))
    
    for i, method in enumerate(methods):
        method_data = df[df['method'] == method]['cindex']
        
        # 箱线图数据
        bp = ax.boxplot([method_data], positions=[i], widths=0.5, 
                       patch_artist=True, showfliers=True)
        
        # 设置颜色
        bp['boxes'][0].set_facecolor(method_colors[method])
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][0].set_edgecolor(method_colors[method])
        bp['boxes'][0].set_linewidth(1.5)
        
        for whisker in bp['whiskers']:
            whisker.set_color(method_colors[method])
        for cap in bp['caps']:
            cap.set_color(method_colors[method])
        bp['medians'][0].set_color('white')
        bp['medians'][0].set_linewidth(2)
        
        # 添加散点 (各 fold)
        jitter = np.random.normal(0, 0.08, len(method_data))
        ax.scatter(np.full(len(method_data), i) + jitter, method_data,
                  s=15, alpha=0.6, c=method_colors[method], 
                  edgecolors='white', linewidths=0.3, zorder=5)
        
        # 添加均值 ± std 标签
        mean = method_data.mean()
        std = method_data.std()
        ax.text(i, mean + 0.06, f'{mean:.3f}±{std:.3f}', 
               ha='center', fontsize=FONT_CONFIG['size_annotation'],
               color=method_colors[method], fontweight='bold')
    
    # 设置
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, fontsize=FONT_CONFIG['size_axis_tick'], rotation=15, ha='right')
    ax.set_ylabel('C-index', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_ylim(0.58, 0.85)
    
    # 添加折数标注
    ax.text(0.98, 0.02, '5-fold CV, 5 cancer types', transform=ax.transAxes,
           ha='right', va='bottom', fontsize=FONT_CONFIG['size_annotation'],
           color=PALETTE['text_secondary'], style='italic')
    
    ax.set_title('Cross-validation Performance', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    add_panel_label(ax, 'c')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'ed_fig4_panel_c_cv', output_dir)


# ============================================================
# 主图生成
# ============================================================

def create_ed_figure4(data_path, output_dir, save_panels=True):
    """生成 Extended Data Figure 4"""
    
    setup_nature_style()
    
    fig = plt.figure(figsize=(7.5, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], 
                          hspace=0.40, wspace=0.40,
                          left=0.12, right=0.95, top=0.95, bottom=0.06)
    
    panels_dir = output_dir / "panels" if save_panels else None
    if panels_dir:
        panels_dir.mkdir(exist_ok=True)
    
    # Panel a: Subgroup forest (左上)
    ax_a = fig.add_subplot(gs[0, 0])
    plot_subgroup_forest(ax_a, data_path, save_panels, panels_dir)
    
    # Panel b: Demographics (右上)
    ax_b = fig.add_subplot(gs[0, 1])
    plot_cohort_demographics(ax_b, data_path, save_panels, panels_dir)
    
    # Panel c: CV results (下方)
    ax_c = fig.add_subplot(gs[1, :])
    plot_cv_results(ax_c, data_path, save_panels, panels_dir)
    
    # 保存
    print("\n📊 Saving ED Figure 4...")
    save_figure(fig, 'ed_fig4_pancancer', output_dir)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Generate ED Figure 4')
    parser.add_argument('--data_dir', type=str, default='../../data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='../../extended_data/ed_fig4_pancancer',
                        help='Output directory for figures')
    parser.add_argument('--no-panels', action='store_true')
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    required_files = ['subgroup_hr.csv', 'cohort_demographics.csv', 'cv_detailed_results.csv']
    missing = [f for f in required_files if not (data_path / f).exists()]
    if missing:
        print(f"❌ Missing: {missing}")
        print("   Run generate_extended_data.py first.")
        return
    
    print("=" * 65)
    print("  ED Figure 4: Pan-cancer & Subgroup Analysis")
    print("=" * 65)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    create_ed_figure4(data_path, output_dir, not args.no_panels)
    
    print("\n✅ ED Figure 4 complete!")


if __name__ == "__main__":
    main()


