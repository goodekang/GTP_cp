"""
Figure 5: Virtual Clinical Trials & Drug Targets — Nature Biotechnology Standard
=================================================================================
生成虚拟临床试验结果图 (Waterfall, Counterfactual KM, UMAP trajectory, External validation)

Features:
- Nature-level semantic palette
- Confidence waterfall with statistical annotations
- Counterfactual KM curves with ΔMedian
- Single panel PDF export

Usage:
    python fig5_virtual_trial.py --data_dir ../../data/processed --output_dir ../../figures/fig5_virtual_trial
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

from plot_config import (
    setup_nature_style, PALETTE, FIGSIZE, FONT_CONFIG,
    save_figure, save_single_panel, add_panel_label, 
    format_pvalue, format_hr_ci, get_significance_level, add_stats_box
)


# ============================================================
# Panel a: In Silico Screening Waterfall
# ============================================================

def plot_screening_waterfall(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel a: In silico screening waterfall plot
    使用蓝金学术配色系统
    
    Nature 级瀑布图:
    - 按 ΔRisk 排序
    - Top 10 高亮 + 标注
    - 置信区间误差棒
    - 统计显著性标记
    """
    df = pd.read_csv(data_path / 'perturbation_effects.csv')
    
    # 取 Top 50 展示
    df_top = df.head(50).copy()
    
    # 颜色分层 — 蓝金渐变配色系统
    # Top 排名使用金色系，较低排名使用蓝色系
    colors = []
    for i, row in df_top.iterrows():
        rank = row['rank']
        if rank <= 5:
            colors.append(PALETTE['clinical_dark'])      # Top 5: 深金 #CA6702
        elif rank <= 10:
            colors.append(PALETTE['clinical_main'])      # Top 6-10: 琥珀 #EE9B00
        elif rank <= 20:
            colors.append(PALETTE['baseline_3'])         # Top 11-20: 中蓝 #457B9D
        else:
            colors.append(PALETTE['baseline_1'])         # 其余: 浅蓝 #A8DADC
    
    # 绘制条形图
    bars = ax.bar(range(len(df_top)), df_top['delta_risk'], color=colors,
                  edgecolor='white', linewidth=0.3, width=0.85, alpha=0.9)
    
    # 添加误差线 (Top 10)
    for i, (_, row) in enumerate(df_top.head(10).iterrows()):
        ax.errorbar(i, row['delta_risk'],
                   yerr=[[row['delta_risk'] - row['ci_lower']], 
                         [row['ci_upper'] - row['delta_risk']]],
                   fmt='none', color=PALETTE['text'], capsize=2, 
                   capthick=0.8, elinewidth=0.8)
    
    # 标注 Top 5 交互 — 使用深金色
    for i, (_, row) in enumerate(df_top.head(5).iterrows()):
        label = f"{row['source_gene']}-{row['target_gene']}"
        y_offset = row['delta_risk'] + 0.006
        ax.text(i, y_offset, label,
               rotation=55, ha='left', va='bottom', 
               fontsize=FONT_CONFIG['size_annotation'],
               color=PALETTE['clinical_dark'], fontweight='bold')
    
    # Top-10 阈值线 — 使用深金色
    if len(df_top) >= 10:
        threshold = df_top.iloc[9]['delta_risk']
        ax.axhline(y=threshold, color=PALETTE['clinical_dark'], linewidth=1.2,
                   linestyle='--', alpha=0.7, zorder=1)
        ax.text(48, threshold + 0.003, 'Top-10', fontsize=FONT_CONFIG['size_annotation'],
               ha='right', color=PALETTE['clinical_dark'], fontweight='bold')
    
    # 零线
    ax.axhline(y=0, color=PALETTE['text'], linewidth=0.8, zorder=1)
    
    # 设置
    ax.set_xlabel('Interactions (ranked by predicted benefit)', 
                 fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_ylabel('ΔRisk (survival benefit)', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_xlim(-1, 51)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    
    # Y轴范围 (留空间给标注)
    y_max = df_top['delta_risk'].max() * 1.35
    ax.set_ylim(-0.008, y_max)
    
    # 统计信息框
    n_validated = df_top.head(10)['literature_validated'].sum()
    validation_rate = n_validated / 10 * 100
    stats_text = f'Top-10 validated: {n_validated}/10 ({validation_rate:.0f}%)'
    add_stats_box(ax, stats_text, loc='upper right')
    
    ax.set_title('In Silico Target Screening', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=10)
    add_panel_label(ax, 'a')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'fig5_panel_a_waterfall', output_dir)


# ============================================================
# Panel b: Counterfactual Survival Curves
# ============================================================

def plot_counterfactual_km(ax, data_path, intervention='PD1_PDL1_inhibition',
                           save_panel_pdf=False, output_dir=None):
    """
    Panel b: Counterfactual survival curves
    使用蓝金学术配色系统
    
    同一患者群体:
    - Original (实线) — 深蓝色
    - Perturbed (虚线) — 深金色
    - ΔMedian Survival 大号数字突出
    """
    df = pd.read_csv(data_path / 'counterfactual_survival.csv')
    df_int = df[df['intervention'] == intervention].copy()
    
    if len(df_int) == 0:
        # 使用第一个可用的干预
        intervention = df['intervention'].unique()[0]
        df_int = df[df['intervention'] == intervention].copy()
    
    kmf = KaplanMeierFitter()
    
    # 原始生存 (高风险组) — 深蓝色代表基线状态
    kmf.fit(df_int['time_original'], event_observed=df_int['event_original'],
            label='Original (High Risk)')
    kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.12,
                               color=PALETTE['model_ours'], linewidth=2.5)
    
    # 干预后生存 — 深金色代表治疗效果
    intervention_label = intervention.replace('_', ' ').replace(' inhibition', '').title()
    kmf.fit(df_int['time_perturbed'], event_observed=df_int['event_perturbed'],
            label=f'After {intervention_label}')
    kmf.plot_survival_function(ax=ax, ci_show=True, ci_alpha=0.12,
                               color=PALETTE['clinical_dark'], linewidth=2.5, 
                               linestyle='--')
    
    # Log-rank test
    result = logrank_test(
        df_int['time_original'], df_int['time_perturbed'],
        df_int['event_original'], df_int['event_perturbed']
    )
    
    # 计算中位生存时间
    median_orig = df_int['time_original'].median()
    median_pert = df_int['time_perturbed'].median()
    delta_median = median_pert - median_orig
    
    # 添加「Predicted Benefit」箭头 — 使用深金色
    arrow_x = max(median_orig, median_pert) * 0.6
    ax.annotate('', xy=(arrow_x, 0.55), xytext=(arrow_x, 0.35),
               arrowprops=dict(arrowstyle='->', color=PALETTE['clinical_dark'], 
                              lw=2.5, mutation_scale=15))
    ax.text(arrow_x + 3, 0.45, 'Predicted\nBenefit', fontsize=FONT_CONFIG['size_annotation'],
           color=PALETTE['clinical_dark'], fontweight='bold', ha='left')
    
    # ΔMedian 大号数字 — 使用深金色
    delta_text = f'+{delta_median:.1f}' if delta_median > 0 else f'{delta_median:.1f}'
    ax.text(0.5, 0.12, f'ΔMedian = {delta_text} mo', 
           transform=ax.transAxes, ha='center',
           fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold',
           color=PALETTE['clinical_dark'],
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                    edgecolor=PALETTE['clinical_dark'], linewidth=1.5, alpha=0.95))
    
    # 统计信息框
    stats_text = f'N = {len(df_int)}\n{format_pvalue(result.p_value)}'
    add_stats_box(ax, stats_text, loc='upper right')
    
    # 设置
    ax.set_xlabel('Time (months)', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_ylabel('Survival Probability', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left', frameon=False, fontsize=FONT_CONFIG['size_legend'])
    
    # 标题
    title_intervention = intervention.replace('_', '-').replace('-inhibition', '').upper()
    ax.set_title(f'Counterfactual Survival ({title_intervention})', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=10)
    
    add_panel_label(ax, 'b')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'fig5_panel_b_counterfactual_km', output_dir)


# ============================================================
# Panel c: UMAP Trajectory
# ============================================================

def plot_umap_trajectory(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel c: UMAP trajectory (Cold → Hot tumor transition)
    
    展示:
    - TME 状态分布
    - 治疗响应轨迹
    - 响应者标记
    """
    df = pd.read_csv(data_path / 'umap_trajectory.csv')
    
    # TME 状态颜色 — 使用蓝金学术配色系统
    # Cold (免疫冷): 深蓝色，代表"静止"状态
    # Intermediate: 金色系，代表"过渡"状态  
    # Hot (免疫热): 琥珀/橙色，代表"激活"状态
    state_colors = {
        'Cold': PALETTE['model_ours'],       # 深蓝 #1D3557 (Prussian Blue)
        'Intermediate': PALETTE['clinical_light'],  # 金色 #FFD166 (Maize)
        'Hot': PALETTE['clinical_dark']      # 深金 #CA6702 (Burnt Sienna)
    }
    
    # 按状态绘制
    for state in ['Cold', 'Intermediate', 'Hot']:
        mask = df['tme_state'] == state
        ax.scatter(df.loc[mask, 'umap1'], df.loc[mask, 'umap2'],
                  c=state_colors[state], s=22, alpha=0.65, label=state,
                  edgecolors='white', linewidths=0.3, zorder=2)
    
    # 计算质心
    cold_center = df[df['tme_state'] == 'Cold'][['umap1', 'umap2']].mean()
    hot_center = df[df['tme_state'] == 'Hot'][['umap1', 'umap2']].mean()
    
    # 绘制转换箭头 (Cold → Hot) — 使用深金色表示治疗方向
    ax.annotate('', 
               xy=(hot_center['umap1'] - 0.6, hot_center['umap2'] - 0.3),
               xytext=(cold_center['umap1'] + 0.6, cold_center['umap2'] + 0.3),
               arrowprops=dict(arrowstyle='-|>', color=PALETTE['clinical_dark'],
                              lw=3, connectionstyle='arc3,rad=0.15',
                              mutation_scale=18),
               zorder=3)
    
    # 箭头标签 — 使用深金色
    mid_x = (cold_center['umap1'] + hot_center['umap1']) / 2
    mid_y = (cold_center['umap2'] + hot_center['umap2']) / 2
    ax.text(mid_x - 0.5, mid_y + 1, 'Treatment\nResponse', 
           fontsize=FONT_CONFIG['size_axis_title'], ha='center',
           color=PALETTE['clinical_dark'], fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    alpha=0.85, edgecolor='none'))
    
    # 响应者标记 — 使用深金色
    responders = df[df['response'] == 'Responder']
    if len(responders) > 0:
        ax.scatter(responders['umap1'], responders['umap2'],
                  facecolors='none', edgecolors=PALETTE['clinical_dark'],
                  s=60, linewidths=2, alpha=0.9, label='Responder', zorder=4)
    
    # 设置
    ax.set_xlabel('UMAP 1', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_ylabel('UMAP 2', fontsize=FONT_CONFIG['size_axis_title'])
    ax.legend(loc='upper left', frameon=False, fontsize=FONT_CONFIG['size_annotation'],
             markerscale=0.8)
    ax.set_title('TME State Transition', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=10)
    
    # 响应统计
    n_responders = (df['response'] == 'Responder').sum()
    n_treated = (df['treatment_status'] == 'Post-treatment').sum()
    if n_treated > 0:
        response_rate = n_responders / n_treated * 100
        ax.text(0.98, 0.02, f'Response: {n_responders}/{n_treated} ({response_rate:.0f}%)',
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=FONT_CONFIG['size_annotation'], color=PALETTE['text_secondary'])
    
    add_panel_label(ax, 'c')
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'fig5_panel_c_umap', output_dir)


# ============================================================
# Panel d: External Validation Evidence Matrix (热图矩阵设计)
# ============================================================

def plot_external_validation(ax, data_path, save_panel_pdf=False, output_dir=None):
    """
    Panel d: External validation evidence matrix
    使用蓝金学术配色的热图矩阵设计
    
    展示 Top 靶点的外部验证:
    - CRISPR/DepMap 依赖性
    - Drug sensitivity
    - Clinical trial status (Phase I/II/III/Approved)
    """
    df = pd.read_csv(data_path / 'external_validation.csv')
    
    # 选择 Top 8 靶点展示
    df_top = df.head(8).copy()
    n_targets = len(df_top)
    
    # 验证类型
    validation_types = ['CRISPR\nDepMap', 'Drug\nSensitivity', 'Clinical\nTrial']
    n_types = len(validation_types)
    
    # 创建验证矩阵 (行=靶点, 列=验证类型)
    # 值: 0=无, 1=有验证, 2=临床批准
    matrix = np.zeros((n_targets, n_types))
    
    for i, (_, row) in enumerate(df_top.iterrows()):
        # CRISPR
        if row['crispr_depmap']:
            # 根据 CRISPR score 强度着色 (越负越强)
            score = abs(row['crispr_score']) if pd.notna(row.get('crispr_score')) else 0.5
            matrix[i, 0] = min(score / 1.0, 1.0)  # 归一化到 0-1
        
        # Drug sensitivity
        if row['drug_sensitivity']:
            matrix[i, 1] = 1.0
        
        # Clinical trial
        if row['clinical_trial']:
            phase = str(row.get('trial_phase', ''))
            if 'Approved' in phase:
                matrix[i, 2] = 1.0
            elif 'III' in phase:
                matrix[i, 2] = 0.85
            elif 'II' in phase:
                matrix[i, 2] = 0.65
            elif 'I' in phase:
                matrix[i, 2] = 0.45
            else:
                matrix[i, 2] = 0.3
    
    # 使用蓝金学术配色创建自定义colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap_validation = LinearSegmentedColormap.from_list(
        'validation',
        ['#F8F9FA',      # 无验证 - 浅灰
         PALETTE['baseline_1'],  # 弱验证 - 浅蓝
         PALETTE['baseline_3'],  # 中等 - 中蓝
         PALETTE['clinical_main'],  # 强验证 - 琥珀
         PALETTE['clinical_dark']], # 已批准 - 深金
        N=256
    )
    
    # 绘制热图
    im = ax.imshow(matrix, cmap=cmap_validation, aspect='auto', vmin=0, vmax=1)
    
    # 靶点标签 (Y轴)
    targets = df_top['target'].tolist()
    # 简化靶点名称
    target_labels = [t.split('-')[0] if '-' in t else t for t in targets]
    ax.set_yticks(range(n_targets))
    ax.set_yticklabels(target_labels, fontsize=FONT_CONFIG['size_axis_tick'])
    
    # 验证类型标签 (X轴)
    ax.set_xticks(range(n_types))
    ax.set_xticklabels(validation_types, fontsize=FONT_CONFIG['size_annotation'], 
                       ha='center')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    
    # 添加单元格标注
    for i in range(n_targets):
        for j in range(n_types):
            val = matrix[i, j]
            if val > 0:
                # 根据值决定标注内容
                if j == 2:  # Clinical trial
                    row = df_top.iloc[i]
                    phase = str(row.get('trial_phase', ''))
                    if 'Approved' in phase:
                        text = 'Appr.'
                    elif phase and phase != 'nan':
                        text = phase.replace('Phase ', 'Ph')
                    else:
                        text = '+'
                else:
                    text = '+'
                
                # 文字颜色：深色背景用白色，浅色背景用深色
                text_color = 'white' if val > 0.5 else PALETTE['text']
                ax.text(j, i, text, ha='center', va='center',
                       fontsize=FONT_CONFIG['size_annotation'], 
                       fontweight='bold', color=text_color)
    
    # 添加网格线
    ax.set_xticks(np.arange(-0.5, n_types, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_targets, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # 去掉边框
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.08, shrink=0.7)
    cbar.set_label('Evidence\nStrength', fontsize=FONT_CONFIG['size_annotation'], 
                   rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['None', 'Partial', 'Strong'])
    
    # 标题
    ax.set_title('External Validation Matrix', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', 
                pad=25, y=1.02)
    
    # 添加汇总统计
    crispr_rate = df['crispr_depmap'].mean() * 100
    drug_rate = df['drug_sensitivity'].mean() * 100
    clinical_rate = df['clinical_trial'].mean() * 100
    approved_count = df[df['trial_phase'].str.contains('Approved', na=False)].shape[0]
    
    stats_text = (f'CRISPR: {crispr_rate:.0f}%  |  Drug: {drug_rate:.0f}%  |  '
                  f'Clinical: {clinical_rate:.0f}%  |  Approved: {approved_count}')
    ax.text(0.5, -0.12, stats_text,
           transform=ax.transAxes, ha='center', 
           fontsize=FONT_CONFIG['size_annotation'],
           color=PALETTE['text_secondary'],
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA', 
                    edgecolor=PALETTE['grid'], linewidth=0.5, alpha=0.9))
    
    add_panel_label(ax, 'd', x=-0.08)
    
    if save_panel_pdf and output_dir:
        save_single_panel(ax, 'fig5_panel_d_validation', output_dir)


# ============================================================
# Supplementary Figures
# ============================================================

def plot_pathway_enrichment(ax, data_path):
    """
    Supplementary: Pathway enrichment bar plot
    使用蓝金学术配色系统
    """
    df = pd.read_csv(data_path / 'pathway_enrichment.csv')
    df = df.sort_values('fold_enrichment', ascending=True)
    
    # 显著性使用深金色，非显著使用浅蓝色 (蓝金配色系统)
    colors = [PALETTE['clinical_dark'] if sig else PALETTE['baseline_1'] 
              for sig in df['significant']]
    
    y_pos = range(len(df))
    bars = ax.barh(y_pos, df['fold_enrichment'], color=colors, 
                   edgecolor='white', height=0.7, alpha=0.88)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['pathway'], fontsize=FONT_CONFIG['size_annotation'])
    ax.set_xlabel('Fold Enrichment', fontsize=FONT_CONFIG['size_axis_title'])
    ax.axvline(x=1, color=PALETTE['text'], linewidth=0.8, linestyle='--', alpha=0.5)
    
    # 添加 FDR 标签 — 使用深金色
    for i, (_, row) in enumerate(df.iterrows()):
        if row['fdr'] < 0.1:
            ax.text(row['fold_enrichment'] + 0.08, i, '*', 
                   fontsize=FONT_CONFIG['size_axis_title'], va='center',
                   color=PALETTE['clinical_dark'])
    
    ax.set_title('Target Pathway Enrichment', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=8)
    ax.text(0.98, 0.02, '* FDR < 0.1', transform=ax.transAxes, ha='right',
           fontsize=FONT_CONFIG['size_annotation'], color=PALETTE['text_secondary'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_null_distribution(ax, data_path):
    """
    Supplementary: Null distribution with real values
    使用蓝金学术配色系统
    """
    df = pd.read_csv(data_path / 'null_distribution.csv')
    
    null_data = df[df['is_null']]['delta_risk']
    real_data = df[~df['is_null']]['delta_risk']
    
    # 直方图 (Null) — 使用浅蓝色
    ax.hist(null_data, bins=50, color=PALETTE['baseline_1'], alpha=0.78,
            edgecolor='white', label='Null distribution', density=True)
    
    # 真实值垂直线 — 使用深金色突出显示
    for i, val in enumerate(real_data):
        ax.axvline(x=val, color=PALETTE['clinical_dark'], linewidth=2.5,
                  label='Top targets' if i == 0 else None, alpha=0.92)
    
    # 95th percentile 阈值
    threshold_95 = np.percentile(null_data, 95)
    ax.axvline(x=threshold_95, color=PALETTE['model_ours'], linewidth=1.2, 
              linestyle='--', label='95th percentile')
    
    ax.set_xlabel('ΔRisk', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_ylabel('Density', fontsize=FONT_CONFIG['size_axis_title'])
    ax.legend(loc='upper right', frameon=False, fontsize=FONT_CONFIG['size_annotation'])
    ax.set_title('Perturbation Effect vs Null', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=8)
    
    # 统计
    add_stats_box(ax, f'Null μ: {null_data.mean():.4f}\nNull σ: {null_data.std():.4f}',
                 loc='upper left')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_dose_response(ax, data_path):
    """
    Supplementary: Dose-response curves
    使用蓝金学术配色系统 — 渐变色彩区分不同靶点
    """
    df = pd.read_csv(data_path / 'dose_response.csv')
    
    targets = df['target'].unique()
    # 蓝金渐变配色：从深蓝到金色的学术级配色
    colors = [
        PALETTE['model_ours'],     # 深蓝 #1D3557
        PALETTE['baseline_3'],     # 中蓝 #457B9D
        PALETTE['clinical_main'],  # 琥珀 #EE9B00
        PALETTE['clinical_dark']   # 深金 #CA6702
    ]
    
    for target, color in zip(targets, colors[:len(targets)]):
        target_data = df[df['target'] == target]
        
        ax.errorbar(target_data['dose_uM'], target_data['response'],
                   yerr=target_data['response_sd'],
                   fmt='o-', color=color, label=target, 
                   markersize=5, capsize=2, linewidth=1.8,
                   markeredgecolor='white', markeredgewidth=0.5)
    
    ax.set_xscale('log')
    ax.set_xlabel('Dose (μM)', fontsize=FONT_CONFIG['size_axis_title'])
    ax.set_ylabel('Response (ΔRisk reduction)', fontsize=FONT_CONFIG['size_axis_title'])
    ax.legend(loc='lower right', frameon=False, fontsize=FONT_CONFIG['size_annotation'])
    ax.set_title('Dose-Response Curves', 
                fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold', pad=8)
    ax.set_ylim(0, 1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ============================================================
# 主图生成
# ============================================================

def create_figure5(data_path, output_dir, save_panels=True):
    """生成完整的 Figure 5"""
    
    setup_nature_style()
    
    fig = plt.figure(figsize=(7.5, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], 
                          hspace=0.45, wspace=0.45,
                          left=0.08, right=0.95, top=0.94, bottom=0.08)
    
    panels_dir = output_dir / "panels" if save_panels else None
    
    # Panel a: Screening waterfall
    ax_a = fig.add_subplot(gs[0, 0])
    plot_screening_waterfall(ax_a, data_path, save_panels, panels_dir)
    
    # Panel b: Counterfactual KM
    ax_b = fig.add_subplot(gs[0, 1])
    plot_counterfactual_km(ax_b, data_path, save_panel_pdf=save_panels, output_dir=panels_dir)
    
    # Panel c: UMAP trajectory
    ax_c = fig.add_subplot(gs[1, 0])
    plot_umap_trajectory(ax_c, data_path, save_panels, panels_dir)
    
    # Panel d: External validation
    ax_d = fig.add_subplot(gs[1, 1])
    plot_external_validation(ax_d, data_path, save_panels, panels_dir)
    
    # 保存
    print("\n📊 Saving main figure...")
    save_figure(fig, 'fig5_virtual_trial', output_dir)
    plt.close(fig)


def create_supplementary(data_path, output_dir):
    """生成补充图"""
    
    setup_nature_style()
    
    # 补充图1: Null distribution + Dose response
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    
    plot_null_distribution(axes[0], data_path)
    add_panel_label(axes[0], 'a')
    
    plot_dose_response(axes[1], data_path)
    add_panel_label(axes[1], 'b')
    
    plt.tight_layout()
    save_figure(fig, 'fig5_supp_null_dose', output_dir)
    plt.close(fig)
    
    # 补充图2: Pathway enrichment
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    plot_pathway_enrichment(ax, data_path)
    plt.tight_layout()
    save_figure(fig, 'fig5_supp_pathway', output_dir)
    plt.close(fig)
    
    # 补充图3: 多干预对比 — 使用蓝金学术配色
    df_cf = pd.read_csv(data_path / 'counterfactual_survival.csv')
    interventions = df_cf['intervention'].unique()[:3]
    
    if len(interventions) >= 3:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
        
        for ax, intervention in zip(axes, interventions):
            df_int = df_cf[df_cf['intervention'] == intervention]
            
            kmf = KaplanMeierFitter()
            
            # 原始生存：深蓝色 (baseline/original 状态)
            kmf.fit(df_int['time_original'], event_observed=df_int['event_original'],
                    label='Original')
            kmf.plot_survival_function(ax=ax, ci_show=False, 
                                       color=PALETTE['model_ours'], linewidth=1.8)
            
            # 干预后生存：深金色 (treatment/intervention 效果)
            kmf.fit(df_int['time_perturbed'], event_observed=df_int['event_perturbed'],
                    label='Perturbed')
            kmf.plot_survival_function(ax=ax, ci_show=False, 
                                       color=PALETTE['clinical_dark'], 
                                       linewidth=1.8, linestyle='--')
            
            title = intervention.replace('_', '-').replace('-inhibition', '').upper()
            ax.set_title(title, fontsize=FONT_CONFIG['size_figure_title'], fontweight='bold')
            ax.set_xlabel('Time (months)', fontsize=FONT_CONFIG['size_axis_title'])
            ax.set_ylabel('Survival', fontsize=FONT_CONFIG['size_axis_title'])
            ax.legend(loc='lower left', frameon=False, fontsize=FONT_CONFIG['size_annotation'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        save_figure(fig, 'fig5_supp_multi_intervention', output_dir)
        plt.close(fig)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate Figure 5 (Nature Biotechnology Standard)')
    parser.add_argument('--data_dir', type=str, default='../../data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='../../figures/fig5_virtual_trial',
                        help='Output directory for figures')
    parser.add_argument('--no-panels', action='store_true',
                        help='Skip saving individual panel PDFs')
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # 检查数据文件
    required_files = [
        'perturbation_effects.csv',
        'counterfactual_survival.csv',
        'umap_trajectory.csv',
        'external_validation.csv',
        'null_distribution.csv',
        'dose_response.csv',
        'pathway_enrichment.csv'
    ]
    
    missing_files = [f for f in required_files if not (data_path / f).exists()]
    if missing_files:
        print("❌ Missing data files:")
        for f in missing_files:
            print(f"   - {data_path / f}")
        print("\n   Please run generate_fig5_data.py first.")
        return
    
    print("=" * 65)
    print("  Figure 5: Virtual Clinical Trials")
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
    create_figure5(data_path, output_dir, save_panels=not args.no_panels)
    print("  ✓ Main figure complete")
    
    print("\n🎨 Generating supplementary figures...")
    create_supplementary(data_path, output_dir)
    print("  ✓ Supplementary figures complete")
    
    print("\n" + "=" * 65)
    print("  ✅ Figure 5 generation complete!")
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
