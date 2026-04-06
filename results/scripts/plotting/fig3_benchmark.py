"""
Figure 3: Comprehensive Benchmarking — Nature Biotechnology Standard
=====================================================================
生成基准测试结果图 (C-index, KM, Ablation, Efficiency)

Features:
- Nature-level semantic palette
- Complete statistical annotations (N, HR, CI, P)
- Risk tables for KM curves
- Single panel PDF export

Usage:
    python fig3_benchmark.py --data_dir ../../data/processed --output_dir ../../figures/fig3_benchmark
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# 导入 Nature 标准配置
from plot_config import (
    setup_nature_style, PALETTE, METHOD_PALETTE, FIGSIZE, FONT_CONFIG,
    save_figure, save_single_panel, add_panel_label, 
    format_pvalue, format_hr_ci, format_stats_complete,
    get_significance_level, add_stats_box, add_risk_table, set_axis_style,
    create_beeswarm_forest, create_confidence_waterfall, CMAP_BLUE_GOLD
)


# ============================================================
# Panel a: Pan-cancer C-index (Beeswarm Forest Plot)
# ============================================================

def plot_pancancer_cindex(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel a: Pan-cancer C-index comparison
    
    使用蜂群森林图同时显示:
    - 每个癌种的点 (beeswarm)
    - 均值与 95% CI
    - 我们的方法高亮
    """
    df = pd.read_csv(data_path / "cindex_pancancer.csv")
    
    # 分离汇总数据和分癌种数据
    df_mean = df[df['cancer_type'] == 'Mean'].copy()
    df_cancers = df[df['cancer_type'] != 'Mean'].copy()
    
    df_mean = df_mean.sort_values('cindex', ascending=True)
    methods = df_mean['method'].tolist()
    y_pos = range(len(methods))
    
    # 绘制每个方法
    for i, method in enumerate(methods):
        is_ours = (method == 'Causal-STFM')
        color = PALETTE['model_ours'] if is_ours else PALETTE['baseline_2']
        
        # 获取该方法在各癌种的数据
        method_data = df_cancers[df_cancers['method'] == method]['cindex'].values
        
        # 蜂群散点 (各癌种)
        jitter = np.random.normal(0, 0.08, len(method_data))
        alpha = 0.7 if is_ours else 0.4
        s = 20 if is_ours else 12
        ax.scatter(method_data, [i] * len(method_data) + jitter,
                  c=color, s=s, alpha=alpha, 
                  edgecolors='white', linewidths=0.3, zorder=2)
        
        # 获取汇总统计
        row = df_mean[df_mean['method'] == method].iloc[0]
        
        # 95% CI 线
        ci_color = PALETTE['model_ours'] if is_ours else PALETTE['text_secondary']
        ax.plot([row['ci_lower'], row['ci_upper']], [i, i], 
               color=ci_color, linewidth=2.5 if is_ours else 1.5, 
               alpha=0.9, zorder=3)
        
        # 均值菱形
        marker = 'D' if is_ours else 'o'
        ms = 8 if is_ours else 5
        ax.scatter(row['cindex'], i, c=color, s=ms**2, marker=marker, 
                  zorder=4, edgecolors='white', linewidths=1.5)
        
        # 数值标签
        label_color = PALETTE['model_ours'] if is_ours else PALETTE['text_secondary']
        fontweight = 'bold' if is_ours else 'normal'
        # 增加偏移量防止遮挡 (0.012 -> 0.025)
        ax.text(row['ci_upper'] + 0.025, i, f"{row['cindex']:.3f}", 
                va='center', fontsize=FONT_CONFIG['size_annotation'],
                color=label_color, fontweight=fontweight)
    
    # 设置坐标轴
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=FONT_CONFIG['size_axis_tick'])
    ax.set_xlabel('C-index (mean ± 95% CI)', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_xlim(0.52, 0.85)  # 稍微扩大范围以容纳标签
    ax.set_ylim(-0.8, len(methods) - 0.2) # 调整Y轴范围，给顶部留空间

    # 参考线 (随机 = 0.5)
    ax.axvline(x=0.5, color=PALETTE['grid'], linestyle='--', 
               alpha=0.5, linewidth=0.8, zorder=1)
    # 调整 Random 标签位置，避免被切断
    ax.text(0.505, len(methods) - 0.8, 'Random', fontsize=5, 
           color=PALETTE['text_secondary'], alpha=0.7)
    
    # 标题
    ax.set_title('Pan-cancer Performance', fontsize=FONT_CONFIG['size_figure_title'], 
                fontweight='bold', pad=10)
    
    # 添加癌种数量注释
    n_cancers = df_cancers['cancer_type'].nunique()
    ax.text(0.98, 0.02, f'n = {n_cancers} cancer types', 
           transform=ax.transAxes, fontsize=FONT_CONFIG['size_annotation'],
           ha='right', va='bottom', color=PALETTE['text_secondary'])
    
    add_panel_label(ax, 'a')
    
    # 保存单独 panel
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'fig3_panel_a_cindex', output_dir)


# ============================================================
# Panel b: KM Survival Curve (主队列)
# ============================================================

def plot_km_curve(ax, data_path, cohort='TCGA-BRCA', 
                  save_panel_pdf=False, output_dir=None):
    """
    Panel b: Kaplan-Meier survival curves with complete statistics
    
    包含:
    - KM曲线 + 95% CI
    - Censoring marks
    - Risk table
    - HR, CI, P-value
    """
    df = pd.read_csv(data_path / "survival_external.csv")
    df_cohort = df[df['cohort'] == cohort].copy()
    
    kmf = KaplanMeierFitter()
    
    # 低风险组
    low_risk = df_cohort[df_cohort['risk_group'] == 'Low']
    kmf.fit(low_risk['time'], event_observed=low_risk['event'], label='Low Risk')
    kmf.plot_survival_function(ax=ax, ci_show=True, 
                               color=PALETTE['low_risk'], linewidth=2,
                               ci_alpha=0.15)
    
    # 添加 censoring marks
    censored_low = low_risk[low_risk['event'] == 0]
    if len(censored_low) > 0:
        # 获取KM曲线在censoring时间点的值
        surv_at_censor = []
        for t in censored_low['time']:
            idx = np.searchsorted(kmf.survival_function_.index, t)
            if idx > 0:
                surv_at_censor.append(kmf.survival_function_.iloc[min(idx, len(kmf.survival_function_)-1)].values[0])
        if surv_at_censor:
            ax.scatter(censored_low['time'].values[:len(surv_at_censor)], 
                      surv_at_censor, marker='+', s=15, 
                      color=PALETTE['low_risk'], alpha=0.5, linewidths=0.5)
    
    # 高风险组
    high_risk = df_cohort[df_cohort['risk_group'] == 'High']
    kmf.fit(high_risk['time'], event_observed=high_risk['event'], label='High Risk')
    kmf.plot_survival_function(ax=ax, ci_show=True,
                               color=PALETTE['high_risk'], linewidth=2,
                               ci_alpha=0.15)
    
    # 添加 censoring marks
    censored_high = high_risk[high_risk['event'] == 0]
    if len(censored_high) > 0:
        surv_at_censor = []
        for t in censored_high['time']:
            idx = np.searchsorted(kmf.survival_function_.index, t)
            if idx > 0:
                surv_at_censor.append(kmf.survival_function_.iloc[min(idx, len(kmf.survival_function_)-1)].values[0])
        if surv_at_censor:
            ax.scatter(censored_high['time'].values[:len(surv_at_censor)], 
                      surv_at_censor, marker='+', s=15, 
                      color=PALETTE['high_risk'], alpha=0.5, linewidths=0.5)
    
    # Log-rank test
    result = logrank_test(
        high_risk['time'], low_risk['time'],
        high_risk['event'], low_risk['event']
    )
    p_value = result.p_value
    
    # 计算 Hazard Ratio
    hr = (high_risk['event'].sum() / high_risk['time'].sum()) / \
         (low_risk['event'].sum() / low_risk['time'].sum() + 1e-10)
    
    # 估算 CI (简化)
    se_log_hr = np.sqrt(1/high_risk['event'].sum() + 1/low_risk['event'].sum())
    hr_lower = hr * np.exp(-1.96 * se_log_hr)
    hr_upper = hr * np.exp(1.96 * se_log_hr)
    
    # 统计信息框
    n_total = len(df_cohort)
    n_events = df_cohort['event'].sum()
    stats_text = (f'N = {n_total}\n'
                 f'Events = {n_events}\n'
                 f'{format_hr_ci(hr, hr_lower, hr_upper)}\n'
                 f'{format_pvalue(p_value)}')
    
    add_stats_box(ax, stats_text, loc='upper right')
    
    # 坐标轴设置
    ax.set_xlabel('Time (months)', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_ylabel('Survival Probability', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left', frameon=False, fontsize=FONT_CONFIG['size_legend'])
    
    # 标题
    cohort_display = cohort.replace('-', ' ').replace('TCGA ', '')
    ax.set_title(f'Survival Analysis ({cohort_display})', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=10)
    
    add_panel_label(ax, 'b')
    
    # 保存单独 panel
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'fig3_panel_b_km', output_dir)


# ============================================================
# Panel c: External Validation KM
# ============================================================

def plot_external_km(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel c: External cohort KM curves (多队列对比)
    """
    df = pd.read_csv(data_path / "survival_external.csv")
    
    cohorts = ['External-Hospital-A', 'External-Hospital-B']
    cohort_colors = [PALETTE['spatial_main'], PALETTE['clinical_main']]
    cohort_labels = ['Hospital A', 'Hospital B']
    
    total_n = 0
    
    for cohort, color, label in zip(cohorts, cohort_colors, cohort_labels):
        df_cohort = df[df['cohort'] == cohort]
        if len(df_cohort) == 0:
            continue
            
        total_n += len(df_cohort)
        kmf = KaplanMeierFitter()
        
        # 高风险
        high = df_cohort[df_cohort['risk_group'] == 'High']
        if len(high) > 0:
            kmf.fit(high['time'], event_observed=high['event'], 
                   label=f'{label} High')
            kmf.plot_survival_function(ax=ax, ci_show=False, 
                                       color=color, linewidth=1.8, linestyle='-')
        
        # 低风险
        low = df_cohort[df_cohort['risk_group'] == 'Low']
        if len(low) > 0:
            kmf.fit(low['time'], event_observed=low['event'], 
                   label=f'{label} Low')
            kmf.plot_survival_function(ax=ax, ci_show=False,
                                       color=color, linewidth=1.8, linestyle='--')
    
    # 设置
    ax.set_xlabel('Time (months)', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_ylabel('Survival Probability', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left', frameon=False, fontsize=FONT_CONFIG['size_annotation'], ncol=2)
    ax.set_title('External Validation', fontsize=FONT_CONFIG['size_figure_title'], 
                fontweight='bold', pad=10)
    
    # 总样本量
    ax.text(0.98, 0.02, f'N = {total_n}', transform=ax.transAxes,
           fontsize=FONT_CONFIG['size_annotation'], ha='right', va='bottom',
           color=PALETTE['text_secondary'])
    
    add_panel_label(ax, 'c')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'fig3_panel_c_external_km', output_dir)


# ============================================================
# Panel d: Spatial Expression Prediction
# ============================================================

def plot_spatial_correlation(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel d: Spatial expression prediction correlation
    """
    df = pd.read_csv(data_path / "spatial_pred_vs_true.csv")
    
    # 采样避免过绘
    n_sample = min(3000, len(df))
    df_sample = df.sample(n=n_sample, random_state=42)
    
    # 计算相关性
    corr = np.corrcoef(df_sample['pred_expr'], df_sample['true_expr'])[0, 1]
    r_squared = corr ** 2
    
    # 密度散点图 (使用 alpha 渐变)
    ax.scatter(df_sample['true_expr'], df_sample['pred_expr'], 
               alpha=0.25, s=4, c=PALETTE['spatial_main'], edgecolors='none')
    
    # 拟合线
    z = np.polyfit(df_sample['true_expr'], df_sample['pred_expr'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_sample['true_expr'].min(), df_sample['true_expr'].max(), 100)
    ax.plot(x_line, p(x_line), color=PALETTE['intervention_main'], 
           linewidth=1.5, label=f'R² = {r_squared:.3f}')
    
    # 对角线 (完美预测)
    lims = [0, max(df_sample['true_expr'].max(), df_sample['pred_expr'].max())]
    ax.plot(lims, lims, color=PALETTE['text_secondary'], linestyle='--', 
           alpha=0.5, linewidth=0.8, label='y = x')
    
    # 设置
    ax.set_xlabel('True Expression', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_ylabel('Predicted Expression', fontsize=FONT_CONFIG['size_axis_title'])
    ax.legend(loc='upper left', frameon=False, fontsize=FONT_CONFIG['size_legend'])
    ax.set_title('Spatial Expression Prediction', fontsize=FONT_CONFIG['size_figure_title'], 
                fontweight='bold', pad=10)
    
    # 样本数
    ax.text(0.98, 0.02, f'n = {len(df)} spots', transform=ax.transAxes,
           fontsize=FONT_CONFIG['size_annotation'], ha='right', va='bottom',
           color=PALETTE['text_secondary'])
    
    add_panel_label(ax, 'd')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'fig3_panel_d_spatial', output_dir)


# ============================================================
# Panel e: Ablation Study (Lollipop Chart)
# ============================================================

def plot_ablation(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel e: Ablation study (棒棒糖图)
    
    清晰展示各组件的贡献
    """
    df = pd.read_csv(data_path / "ablation_results.csv")
    
    # 只取 Mean 汇总
    df_mean = df[df['cancer_type'] == 'Mean'].copy()
    df_mean = df_mean.sort_values('delta_cindex', ascending=False)
    
    variants = df_mean['variant'].tolist()
    deltas = df_mean['delta_cindex'].tolist()
    
    y_pos = range(len(variants))
    
    # 绘制
    for i, (variant, delta) in enumerate(zip(variants, deltas)):
        # 颜色和标记
        if 'Full Model' in variant:
            color = PALETTE['model_ours']
            marker = 'D'
            ms = 9
            lw = 2.5
        elif 'Random' in variant:
            color = PALETTE['intervention_dark']
            marker = 'X'
            ms = 8
            lw = 1.5
        else:
            color = PALETTE['baseline_3']
            marker = 'o'
            ms = 7
            lw = 1.5
        
        # 线
        ax.hlines(i, 0, delta, colors=color, linewidth=lw, alpha=0.8)
        # 点
        ax.scatter(delta, i, c=color, s=ms**2, marker=marker, zorder=5,
                   edgecolors='white' if marker != 'X' else color, linewidths=1.2)
        
        # 数值标签
        # 优化标签位置逻辑，避免遮挡
        if delta < 0:
            offset = -0.015  # 负值向左偏移
            ha = 'right'
        else:
            offset = 0.015   # 正值向右偏移
            ha = 'left'
            
        ax.text(delta + offset, i, f'{delta:+.3f}', 
               va='center', ha=ha, fontsize=FONT_CONFIG['size_annotation'],
               color=PALETTE['text_secondary'])
    
    # 设置
    ax.set_yticks(y_pos)
    variant_labels = [v.replace(' (Causal-STFM)', '').replace('w/o ', '- ') for v in variants]
    ax.set_yticklabels(variant_labels, fontsize=FONT_CONFIG['size_axis_tick'])
    ax.set_xlabel('ΔC-index (vs Full Model)', fontsize=FONT_CONFIG['size_axis_title'])
    
    # 零线
    ax.axvline(x=0, color=PALETTE['text'], linewidth=1, zorder=1)
    
    # 设置范围 (扩大左侧以容纳标签)
    ax.set_xlim(-0.25, 0.05)
    
    ax.set_title('Ablation Study', fontsize=FONT_CONFIG['size_figure_title'], 
                fontweight='bold', pad=10)
    
    add_panel_label(ax, 'e')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'fig3_panel_e_ablation', output_dir)


# ============================================================
# Panel f: Computational Efficiency
# ============================================================

def plot_efficiency(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel f: Computational efficiency (双Y轴)
    """
    df = pd.read_csv(data_path / "efficiency.csv")
    
    ax2 = ax.twinx()
    
    # GPU memory (左轴)
    line1, = ax.plot(df['n_nodes'], df['gpu_mem_gb'], 'o-', 
                     color=PALETTE['spatial_main'], linewidth=1.8, markersize=5,
                     label='GPU Memory', markeredgecolor='white', markeredgewidth=0.5)
    ax.set_ylabel('GPU Memory (GB)', color=PALETTE['spatial_main'], 
                 fontsize=FONT_CONFIG['size_axis_title'])
    ax.tick_params(axis='y', labelcolor=PALETTE['spatial_main'])
    ax.set_ylim(0, max(df['gpu_mem_gb']) * 1.25)
    
    # Throughput (右轴)
    line2, = ax2.plot(df['n_nodes'], df['throughput_wsi_per_min'], 's--', 
                      color=PALETTE['clinical_main'], linewidth=1.8, markersize=5,
                      label='Throughput', markeredgecolor='white', markeredgewidth=0.5)
    ax2.set_ylabel('Throughput (WSI/min)', color=PALETTE['clinical_main'],
                  fontsize=FONT_CONFIG['size_axis_title'])
    ax2.tick_params(axis='y', labelcolor=PALETTE['clinical_main'])
    ax2.set_ylim(0, max(df['throughput_wsi_per_min']) * 1.25)
    
    # 保持右侧 spine
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(PALETTE['clinical_main'])
    ax2.spines['top'].set_visible(False)
    
    # X轴
    ax.set_xlabel('Number of Graph Nodes', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_xscale('log')
    
    # 合并图例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    # 移动图例到左侧中间，避免遮挡右侧交叉点
    ax.legend(lines, labels, loc='center left', frameon=False, 
             fontsize=FONT_CONFIG['size_legend'], bbox_to_anchor=(0.02, 0.5))
    
    # 硬件信息 (稍微下移)
    gpu_info = df['gpu_model'].iloc[0]
    ax.text(0.02, 0.98, gpu_info, transform=ax.transAxes, 
            ha='left', va='top', fontsize=FONT_CONFIG['size_annotation'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor=PALETTE['clinical_light'], 
                     alpha=0.3, edgecolor='none'))
    
    # 标注推荐点 (5000节点)
    if 5000 in df['n_nodes'].values:
        idx_5k = df[df['n_nodes'] == 5000].index[0]
        mem_5k = df.loc[idx_5k, 'gpu_mem_gb']
        ax.annotate(f"Recommended\n(<24GB)", 
                    xy=(5000, mem_5k),
                    xytext=(8000, mem_5k + 8), fontsize=FONT_CONFIG['size_annotation'],
                    arrowprops=dict(arrowstyle='->', color=PALETTE['text_secondary'], lw=0.8),
                    color=PALETTE['text_secondary'])
    
    ax.set_title('Computational Efficiency', fontsize=FONT_CONFIG['size_figure_title'], 
                fontweight='bold', pad=10)
    add_panel_label(ax, 'f')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'fig3_panel_f_efficiency', output_dir)


# ============================================================
# Supplementary: C-index Heatmap
# ============================================================

def plot_cindex_heatmap(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Supplementary: C-index heatmap across cancer types
    """
    df = pd.read_csv(data_path / "cindex_pancancer.csv")
    df = df[df['cancer_type'] != 'Mean']
    
    # Pivot 表
    pivot = df.pivot(index='method', columns='cancer_type', values='cindex')
    
    # 排序 (按平均性能)
    method_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[method_order]
    
    # 使用蓝金双色热图配色
    # 绘制热图
    im = ax.imshow(pivot.values, cmap=CMAP_BLUE_GOLD, aspect='auto', vmin=0.50, vmax=0.80)
    
    # 标签
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', 
                      fontsize=FONT_CONFIG['size_axis_tick'])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=FONT_CONFIG['size_axis_tick'])
    
    # 数值标注
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            color = 'white' if val > 0.70 or val < 0.55 else PALETTE['text']
            fontweight = 'bold' if pivot.index[i] == 'Causal-STFM' else 'normal'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   fontsize=5, color=color, fontweight=fontweight)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label('C-index', fontsize=FONT_CONFIG['size_axis_title'])
    cbar.ax.tick_params(labelsize=FONT_CONFIG['size_annotation'])
    
    ax.set_title('Performance by Cancer Type', fontsize=FONT_CONFIG['size_figure_title'], 
                fontweight='bold', pad=10)
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'fig3_supp_heatmap', output_dir)


# ============================================================
# 主图生成
# ============================================================

def create_figure3(data_path, output_dir, save_panels=True):
    """生成完整的 Figure 3 (主图)"""
    
    setup_nature_style()
    
    # 创建 2x3 布局
    fig = plt.figure(figsize=(7.5, 6.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], 
                          hspace=0.45, wspace=0.50,
                          left=0.08, right=0.95, top=0.94, bottom=0.08)
    
    panels_dir = output_dir / "panels" if save_panels else None
    
    # Panel a: C-index
    ax_a = fig.add_subplot(gs[0, 0])
    plot_pancancer_cindex(ax_a, data_path, save_panels, panels_dir)
    
    # Panel b: KM (TCGA主队列)
    ax_b = fig.add_subplot(gs[0, 1])
    plot_km_curve(ax_b, data_path, cohort='TCGA-BRCA', 
                  save_panel_pdf=save_panels, output_dir=panels_dir)
    
    # Panel c: External KM
    ax_c = fig.add_subplot(gs[0, 2])
    plot_external_km(ax_c, data_path, save_panels, panels_dir)
    
    # Panel d: Spatial correlation
    ax_d = fig.add_subplot(gs[1, 0])
    plot_spatial_correlation(ax_d, data_path, save_panels, panels_dir)
    
    # Panel e: Ablation
    ax_e = fig.add_subplot(gs[1, 1])
    plot_ablation(ax_e, data_path, save_panels, panels_dir)
    
    # Panel f: Efficiency
    ax_f = fig.add_subplot(gs[1, 2])
    plot_efficiency(ax_f, data_path, save_panels, panels_dir)
    
    # 保存完整图
    print("\n📊 Saving main figure...")
    save_figure(fig, 'fig3_benchmark', output_dir)
    
    # 保存增强版 (带更多边距)
    fig.savefig(output_dir / 'fig3_benchmark_enhanced.pdf', 
               format='pdf', bbox_inches='tight', pad_inches=0.1,
               facecolor='white', edgecolor='none')
    fig.savefig(output_dir / 'fig3_benchmark_enhanced.png', 
               format='png', dpi=300, bbox_inches='tight', pad_inches=0.1,
               facecolor='white', edgecolor='none')
    print(f"  ✓ Saved: {output_dir / 'fig3_benchmark_enhanced.pdf'}")
    
    plt.close(fig)


def create_supplementary_heatmap(data_path, output_dir):
    """生成补充图: C-index 热图"""
    
    setup_nature_style()
    
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    plot_cindex_heatmap(ax, data_path, save_panel_pdf=True, 
                       output_dir=output_dir / "panels")
    
    save_figure(fig, 'fig3_supp_cindex_heatmap', output_dir)
    plt.close(fig)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate Figure 3 (Nature Biotechnology Standard)')
    parser.add_argument('--data_dir', type=str, default='../../data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='../../figures/fig3_benchmark',
                        help='Output directory for figures')
    parser.add_argument('--no-panels', action='store_true',
                        help='Skip saving individual panel PDFs')
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # 检查数据文件
    required_files = [
        'cindex_pancancer.csv',
        'survival_external.csv', 
        'spatial_pred_vs_true.csv',
        'ablation_results.csv',
        'efficiency.csv'
    ]
    
    missing_files = [f for f in required_files if not (data_path / f).exists()]
    if missing_files:
        print("❌ Missing data files:")
        for f in missing_files:
            print(f"   - {data_path / f}")
        print("\n   Please run generate_fig3_data.py first.")
        return
    
    print("=" * 65)
    print("  Figure 3: Comprehensive Benchmarking")
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
    create_figure3(data_path, output_dir, save_panels=not args.no_panels)
    print("  ✓ Main figure complete")
    
    print("\n🎨 Generating supplementary heatmap...")
    create_supplementary_heatmap(data_path, output_dir)
    print("  ✓ Supplementary heatmap complete")
    
    print("\n" + "=" * 65)
    print("  ✅ Figure 3 generation complete!")
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
