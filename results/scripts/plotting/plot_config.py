"""
Causal-STFM 论文统一绑图配置 — Nature Biotechnology 标准
===========================================================
严格遵循《论文可视化设计.md》的设计规范：
- 三阶语义色彩系统
- Nature-grade 排版规范
- 完整的统计呈现标准
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
import numpy as np

# ============================================================
# 1. Nature-Level 语义配色 (Semantic Color Palette)
# ============================================================

PALETTE = {
    # ─────────────────────────────────────────────────────────
    # PRIMARY SEMANTICS — 四象限语义色
    # ─────────────────────────────────────────────────────────
    
    # 🔷 SPATIAL (Source/Truth) - 蓝绿系
    "spatial_main": "#0A9396",      # Teal
    "spatial_light": "#94D2BD",     # Seafoam
    "spatial_dark": "#005F73",      # Ocean
    
    # 🔶 CLINICAL (Target/Outcome) - 琥珀系
    "clinical_main": "#EE9B00",     # Amber
    "clinical_light": "#FFD166",    # Maize
    "clinical_dark": "#CA6702",     # Burnt Sienna
    
    # 🟣 CAUSAL (Prior/Constraint) - 紫罗兰系
    "causal_main": "#9B5DE5",       # Violet
    "causal_light": "#C9B1FF",      # Lavender
    "causal_dark": "#6A0DAD",       # Purple
    
    # 🔴 INTERVENTION (Perturbation) - 深红系
    "intervention_main": "#E63946", # Crimson
    "intervention_light": "#F4A3A8",# Salmon
    "intervention_dark": "#9D0208", # Wine
    
    # ─────────────────────────────────────────────────────────
    # MODEL & BASELINE — 方法对比
    # ─────────────────────────────────────────────────────────
    "model_ours": "#1D3557",        # Prussian Blue — 所有图中高亮我们的方法
    "baseline_1": "#A8DADC",        # 灰蓝渐变 (按性能排序)
    "baseline_2": "#6D9DC5",
    "baseline_3": "#457B9D",
    "baseline_4": "#2B5A7C",
    
    # ─────────────────────────────────────────────────────────
    # SURVIVAL ANALYSIS — 生存分析专用
    # ─────────────────────────────────────────────────────────
    "high_risk": "#9D0208",         # 高风险组 (Wine)
    "low_risk": "#2A9D8F",          # 低风险组 (Teal variant)
    "medium_risk": "#F4A261",       # 中风险组 (如需要)
    
    # ─────────────────────────────────────────────────────────
    # NEUTRALS & ACCENTS — 中性色与强调
    # ─────────────────────────────────────────────────────────
    "background": "#F8F9FA",        # Cultured White
    "grid": "#ADB5BD",              # Gray 40%
    "text": "#212529",              # Near black
    "text_secondary": "#6C757D",    # Gray 60%
    
    # ─────────────────────────────────────────────────────────
    # SIGNIFICANCE LEVELS — 显著性标记
    # ─────────────────────────────────────────────────────────
    "sig_3star": "#1D3557",         # *** P < 0.001
    "sig_2star": "#457B9D",         # ** P < 0.01
    "sig_1star": "#A8DADC",         # * P < 0.05
    "sig_ns": "#E9ECEF",            # ns
    
    # ─────────────────────────────────────────────────────────
    # CELL TYPES — 细胞类型 (空间组学)
    # ─────────────────────────────────────────────────────────
    "tumor": "#E63946",             # Tumor cells
    "stroma": "#8B4513",            # CAF/Stromal (Brown)
    "immune_cd8": "#2A9D8F",        # CD8+ T cells
    "immune_cd4": "#90EE90",        # CD4+ T cells
    "macrophage": "#FFD700",        # Macrophages (Gold)
    "b_cell": "#4169E1",            # B cells
    "endothelial": "#9B59B6",       # Endothelial
    "nk_cell": "#FF69B4",           # NK cells
    
    # ─────────────────────────────────────────────────────────
    # GRAPH LEVELS — 图层级
    # ─────────────────────────────────────────────────────────
    "level_gene": "#94D2BD",        # Gene layer (Seafoam)
    "level_cell": "#2A9D8F",        # Cell layer (Teal)
    "level_patch": "#FFD166",       # Patch layer (Maize)
}

# 兼容旧接口
COLORS = PALETTE

# ============================================================
# 1.1 蓝金双色热图配色 (Blue-Gold Heatmap Colormaps)
# ============================================================

from matplotlib.colors import LinearSegmentedColormap

# 蓝金单调渐变 (用于 C-index, performance metrics 等正值数据)
# 低值→高值: 浅蓝 → 深蓝 → 金色 → 深金
CMAP_BLUE_GOLD = LinearSegmentedColormap.from_list(
    'blue_gold',
    ['#A8DADC',  # 浅蓝 (baseline_1)
     '#457B9D',  # 中蓝 (baseline_3)
     '#1D3557',  # 深蓝 (model_ours)
     '#CA6702',  # 深金 (clinical_dark)
     '#EE9B00'], # 琥珀 (clinical_main)
    N=256
)

# 蓝金发散渐变 (用于相关性等有正负值的数据)
# 负值(蓝) → 零(白) → 正值(金)
CMAP_BLUE_WHITE_GOLD = LinearSegmentedColormap.from_list(
    'blue_white_gold',
    ['#1D3557',  # 深蓝 (强负相关)
     '#457B9D',  # 中蓝
     '#A8DADC',  # 浅蓝
     '#FFFFFF',  # 白 (零相关)
     '#FFD166',  # 浅金
     '#EE9B00',  # 琥珀
     '#CA6702'], # 深金 (强正相关)
    N=256
)

# 蓝金 attention 热图 (用于 WSI attention overlay)
# 透明/白 → 蓝 → 金 (高attention区域)
CMAP_ATTENTION = LinearSegmentedColormap.from_list(
    'attention_blue_gold',
    ['#FFFFFF',  # 白 (无attention)
     '#E8F4F8',  # 极浅蓝
     '#A8DADC',  # 浅蓝
     '#457B9D',  # 中蓝
     '#1D3557',  # 深蓝
     '#CA6702',  # 深金
     '#EE9B00'], # 琥珀 (高attention)
    N=256
)

# 方法对比专用调色板 (用于 benchmark 图)
METHOD_PALETTE = {
    "Causal-STFM": PALETTE["model_ours"],
    "MCAT": PALETTE["baseline_1"],
    "Porpoise": PALETTE["baseline_2"],
    "Nicheformer": PALETTE["baseline_3"],
    "ABMIL": PALETTE["baseline_4"],
    "TransMIL": "#566573",
    "Random": "#E9ECEF",
}

# 基线方法渐变列表 (按性能排序使用)
BASELINE_GRADIENT = [
    PALETTE["baseline_1"],
    PALETTE["baseline_2"],
    PALETTE["baseline_3"],
    PALETTE["baseline_4"],
]

# ============================================================
# 2. 图表尺寸 (Figure Dimensions) — Nature Biotechnology Specs
# ============================================================

# 期刊尺寸规范 (单位: inches，已从 mm 转换)
# 1 inch = 25.4 mm
FIGSIZE = {
    # Nature Biotechnology standard sizes
    "single_col": (3.35, 2.5),       # 85mm width
    "single_col_tall": (3.35, 4.0),
    "single_col_square": (3.35, 3.35),
    "1.5_col": (4.49, 3.5),          # 114mm width
    "double_col": (7.09, 4.5),       # 180mm width
    "double_col_tall": (7.09, 6.5),
    "full_page": (7.09, 9.06),       # 180mm × 230mm
    
    # 专用尺寸
    "km_curve": (3.5, 3.0),          # KM曲线专用
    "heatmap": (4.5, 3.5),           # 热图专用
    "panel_small": (2.5, 2.5),       # 单独 panel
    "panel_medium": (3.5, 3.0),
    "panel_large": (4.5, 3.5),
}

# ============================================================
# 3. 字体设置 (Typography) — Nature Standards
# ============================================================

FONT_CONFIG = {
    "family": "sans-serif",
    "sans_serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "math": ["STIX Two Math", "Times New Roman"],
    
    # Nature Biotechnology 字号规范
    "size_figure_title": 10,        # 每个Figure的主标题
    "size_panel_label": 9,          # a, b, c... 面板标签
    "size_axis_title": 8,           # x轴/y轴标题
    "size_axis_tick": 7,            # 刻度值
    "size_legend": 7,               # 图例文字
    "size_annotation": 6,           # 图内注释
    "size_stats_box": 6.5,          # 统计信息框
    
    # 字重
    "weight_emphasis": 600,         # 强调
    "weight_normal": 400,           # 正常
}

# ============================================================
# 4. 线条与标记 (Lines & Markers)
# ============================================================

LINE_CONFIG = {
    "axis_linewidth": 0.5,          # 轴线 (不要太粗，避免抢夺数据视觉)
    "plot_linewidth": 1.5,          # 主数据线
    "thin_linewidth": 0.75,         # 次要连线
    "grid_linewidth": 0.25,         # 网格线 (几乎不可见)
    "errorbar_linewidth": 1.0,      # 误差棒线
    "errorbar_capwidth": 0.75,      # 误差棒帽线
    
    "marker_size": 4,               # 默认标记大小
    "errorbar_capsize": 2,          # 误差棒帽长度
}

# ============================================================
# 5. 应用 Nature Biotechnology 风格
# ============================================================

def setup_nature_style():
    """Configure Nature Biotechnology publication style"""
    
    mpl.rcParams.update({
        # ─── Typography ───
        'font.family': FONT_CONFIG['family'],
        'font.sans-serif': FONT_CONFIG['sans_serif'],
        'font.size': FONT_CONFIG['size_axis_tick'],
        'axes.titlesize': FONT_CONFIG['size_figure_title'],
        'axes.labelsize': FONT_CONFIG['size_axis_title'],
        'xtick.labelsize': FONT_CONFIG['size_axis_tick'],
        'ytick.labelsize': FONT_CONFIG['size_axis_tick'],
        'legend.fontsize': FONT_CONFIG['size_legend'],
        
        # ─── Lines ───
        'axes.linewidth': LINE_CONFIG['axis_linewidth'],
        'lines.linewidth': LINE_CONFIG['plot_linewidth'],
        'lines.markersize': LINE_CONFIG['marker_size'],
        
        # ─── Grid (nearly invisible) ───
        'axes.grid': True,
        'grid.alpha': 0.15,
        'grid.linewidth': LINE_CONFIG['grid_linewidth'],
        'grid.color': PALETTE['grid'],
        
        # ─── Spines (remove top/right) ───
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # ─── Legend ───
        'legend.frameon': False,
        'legend.borderpad': 0.3,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.5,
        
        # ─── Figure ───
        'figure.facecolor': 'white',
        'figure.dpi': 150,
        'figure.autolayout': False,
        
        # ─── Saving ───
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'savefig.transparent': False,
        'savefig.facecolor': 'white',
        
        # ─── PDF Font Embedding (CRITICAL for publication) ───
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        
        # ─── Axes ───
        'axes.facecolor': 'white',
        'axes.edgecolor': PALETTE['text'],
        'axes.labelcolor': PALETTE['text'],
        'xtick.color': PALETTE['text'],
        'ytick.color': PALETTE['text'],
        
        # ─── Error bars ───
        'errorbar.capsize': LINE_CONFIG['errorbar_capsize'],
    })
    
    print("✓ Nature Biotechnology style configured")


# 兼容旧接口
def setup_style():
    """旧接口兼容 - 调用 Nature 风格"""
    setup_nature_style()


# ============================================================
# 6. 图表创建与保存
# ============================================================

def get_figure(size_key="single_col", **kwargs):
    """获取指定尺寸的 figure"""
    figsize = FIGSIZE.get(size_key, FIGSIZE["single_col"])
    fig = plt.figure(figsize=figsize, **kwargs)
    fig.set_facecolor('white')
    return fig


def create_figure(nrows=1, ncols=1, size_key="single_col", **kwargs):
    """创建带 subplots 的 figure"""
    figsize = FIGSIZE.get(size_key, FIGSIZE["single_col"])
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    fig.set_facecolor('white')
    return fig, axes


def save_figure(fig, name, output_dir, formats=["pdf", "png"], 
                save_panels=False, panel_axes=None):
    """保存图表到指定目录
    
    Args:
        fig: matplotlib figure
        name: 文件名 (不含扩展名)
        output_dir: 输出目录
        formats: 输出格式列表
        save_panels: 是否保存单独的 panel PDF
        panel_axes: dict of {panel_label: ax} for individual saving
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存完整图
    for fmt in formats:
        filepath = output_dir / f"{name}.{fmt}"
        dpi = 300 if fmt == "png" else None
        fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"  ✓ Saved: {filepath}")
    
    # 保存单独的 panel PDF
    if save_panels and panel_axes:
        panels_dir = output_dir / "panels"
        panels_dir.mkdir(exist_ok=True)
        
        for label, ax in panel_axes.items():
            save_single_panel(ax, f"{name}_panel_{label}", panels_dir)


def save_single_panel(ax, name, output_dir, pad_inches=0.1):
    """保存单个 panel 为独立 PDF
    
    Args:
        ax: matplotlib axes
        name: 文件名
        output_dir: 输出目录
        pad_inches: 边距
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取 axes 的边界
    extent = ax.get_tightbbox(ax.figure.canvas.get_renderer())
    
    # 转换为 figure 坐标
    extent_inches = extent.transformed(ax.figure.dpi_scale_trans.inverted())
    
    # 保存
    filepath = output_dir / f"{name}.pdf"
    ax.figure.savefig(
        filepath,
        format='pdf',
        bbox_inches=extent_inches.expanded(1.0 + pad_inches, 1.0 + pad_inches),
        facecolor='white',
        edgecolor='none'
    )
    print(f"    ✓ Panel saved: {filepath}")


# ============================================================
# 7. 统计格式化函数 — Nature Standards
# ============================================================

def format_pvalue(p, include_prefix=True):
    """Nature-style P值格式化
    
    规则:
    - P < 0.0001: "P < 0.0001"
    - P < 0.001:  "P = 0.000X"
    - P < 0.01:   "P = 0.00X"
    - P >= 0.01:  "P = 0.0X"
    """
    prefix = "P " if include_prefix else ""
    
    if p < 0.0001:
        return f"{prefix}< 0.0001"
    elif p < 0.001:
        return f"{prefix}= {p:.4f}"
    elif p < 0.01:
        return f"{prefix}= {p:.3f}"
    else:
        return f"{prefix}= {p:.2f}"


def format_hr_ci(hr, ci_lower, ci_upper):
    """格式化 Hazard Ratio 与置信区间
    
    Output: "HR = 2.34 (95% CI: 1.87–2.93)"
    """
    return f"HR = {hr:.2f} (95% CI: {ci_lower:.2f}–{ci_upper:.2f})"


def format_stats_complete(hr, ci_lower, ci_upper, p):
    """完整统计信息格式化
    
    Output: "HR = 2.34 (95% CI: 1.87–2.93), P < 0.0001"
    """
    hr_str = format_hr_ci(hr, ci_lower, ci_upper)
    p_str = format_pvalue(p)
    return f"{hr_str}, {p_str}"


def get_significance_level(p):
    """获取显著性星号标记
    
    Returns: (stars, color)
    - *** : P < 0.001
    - **  : P < 0.01
    - *   : P < 0.05
    - ns  : P >= 0.05
    """
    if p < 0.001:
        return "***", PALETTE["sig_3star"]
    elif p < 0.01:
        return "**", PALETTE["sig_2star"]
    elif p < 0.05:
        return "*", PALETTE["sig_1star"]
    else:
        return "ns", PALETTE["sig_ns"]


# ============================================================
# 8. 图形辅助函数
# ============================================================

def add_panel_label(ax, label, x=-0.12, y=1.08, fontsize=None, fontweight='bold'):
    """添加面板标签 (a, b, c, ...)
    
    符合 Nature 规范：9pt，加粗，左上角
    """
    if fontsize is None:
        fontsize = FONT_CONFIG['size_panel_label']
    
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight,
            va='bottom', ha='right',
            color=PALETTE['text'])


def add_significance_bar(ax, x1, x2, y, p_value, h=0.02, fontsize=None):
    """添加显著性标记线 (bracket with stars)"""
    if fontsize is None:
        fontsize = FONT_CONFIG['size_annotation']
    
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], 
            lw=0.8, c=PALETTE['text'])
    
    stars, _ = get_significance_level(p_value)
    ax.text((x1+x2)/2, y+h, stars, ha='center', va='bottom', 
            fontsize=fontsize, color=PALETTE['text'])


def add_stats_box(ax, text, loc='upper right', fontsize=None, alpha=0.95):
    """添加统计信息框"""
    if fontsize is None:
        fontsize = FONT_CONFIG['size_stats_box']
    
    # 位置映射
    loc_map = {
        'upper right': (0.97, 0.97, 'right', 'top'),
        'upper left': (0.03, 0.97, 'left', 'top'),
        'lower right': (0.97, 0.03, 'right', 'bottom'),
        'lower left': (0.03, 0.03, 'left', 'bottom'),
    }
    
    x, y, ha, va = loc_map.get(loc, loc_map['upper right'])
    
    ax.text(x, y, text, transform=ax.transAxes, 
            ha=ha, va=va, fontsize=fontsize,
            bbox=dict(boxstyle='round,pad=0.4', 
                     facecolor='white',
                     edgecolor=PALETTE['grid'], 
                     alpha=alpha,
                     linewidth=0.5))


def add_risk_table(ax, times, at_risk_high, at_risk_low, y_offset=-0.25):
    """为 KM 曲线添加 Risk Table
    
    Args:
        ax: matplotlib axes
        times: 时间点列表 (e.g., [0, 12, 24, 36, 48, 60])
        at_risk_high: 高风险组人数列表
        at_risk_low: 低风险组人数列表
    """
    # 创建表格文本
    ax.text(-0.02, y_offset, 'At risk:', transform=ax.transAxes,
           fontsize=FONT_CONFIG['size_annotation'], fontweight='bold',
           ha='right', va='top')
    
    ax.text(-0.02, y_offset - 0.06, 'High:', transform=ax.transAxes,
           fontsize=FONT_CONFIG['size_annotation'],
           ha='right', va='top', color=PALETTE['high_risk'])
    
    ax.text(-0.02, y_offset - 0.12, 'Low:', transform=ax.transAxes,
           fontsize=FONT_CONFIG['size_annotation'],
           ha='right', va='top', color=PALETTE['low_risk'])
    
    # 添加数字
    xlim = ax.get_xlim()
    for i, t in enumerate(times):
        x_pos = (t - xlim[0]) / (xlim[1] - xlim[0])
        
        # 时间标签
        ax.text(x_pos, y_offset, f'{t}', transform=ax.transAxes,
               fontsize=FONT_CONFIG['size_annotation'], ha='center', va='top')
        
        # 高风险人数
        ax.text(x_pos, y_offset - 0.06, f'{at_risk_high[i]}', 
               transform=ax.transAxes, fontsize=FONT_CONFIG['size_annotation'],
               ha='center', va='top', color=PALETTE['high_risk'])
        
        # 低风险人数
        ax.text(x_pos, y_offset - 0.12, f'{at_risk_low[i]}', 
               transform=ax.transAxes, fontsize=FONT_CONFIG['size_annotation'],
               ha='center', va='top', color=PALETTE['low_risk'])


def set_axis_style(ax, xlabel=None, ylabel=None, title=None, 
                   xlim=None, ylim=None, despine=True):
    """统一设置轴样式"""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_CONFIG['size_axis_title'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_CONFIG['size_axis_title'])
    if title:
        ax.set_title(title, fontsize=FONT_CONFIG['size_figure_title'], 
                    fontweight='bold', pad=8)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    if despine:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


# ============================================================
# 9. 高级可视化组件
# ============================================================

def create_beeswarm_forest(ax, data, y_col, x_col, group_col, 
                          ci_lower_col=None, ci_upper_col=None,
                          highlight_value=None):
    """创建蜂群森林图 (Beeswarm Forest Plot)
    
    替代传统的点+CI图，同时展示分布
    """
    groups = data[group_col].unique()
    y_positions = range(len(groups))
    
    for i, group in enumerate(groups):
        group_data = data[data[group_col] == group]
        
        # 确定颜色
        if highlight_value and group == highlight_value:
            color = PALETTE['model_ours']
            alpha = 0.9
            s = 25
        else:
            color = PALETTE['baseline_2']
            alpha = 0.6
            s = 15
        
        # 绘制散点 (jittered)
        jitter = np.random.normal(0, 0.1, len(group_data))
        ax.scatter(group_data[x_col], [i] * len(group_data) + jitter,
                  c=color, s=s, alpha=alpha, edgecolors='white', linewidths=0.3)
        
        # 绘制 CI (如果提供)
        if ci_lower_col and ci_upper_col:
            ci_l = group_data[ci_lower_col].mean()
            ci_u = group_data[ci_upper_col].mean()
            mean_val = group_data[x_col].mean()
            
            ax.plot([ci_l, ci_u], [i, i], color=color, linewidth=2, alpha=0.8)
            ax.scatter([mean_val], [i], c=color, s=50, marker='|', 
                      linewidths=2, zorder=5)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(groups)
    
    return ax


def create_confidence_waterfall(ax, data, name_col, value_col, 
                                ci_lower_col=None, ci_upper_col=None,
                                highlight_name=None):
    """创建置信带瀑布图 (Confidence Waterfall)
    
    同时显示 mean + CI + 排名的条形图
    """
    data_sorted = data.sort_values(value_col, ascending=True)
    
    y_positions = range(len(data_sorted))
    
    for i, (_, row) in enumerate(data_sorted.iterrows()):
        name = row[name_col]
        value = row[value_col]
        
        # 确定颜色
        if highlight_name and name == highlight_name:
            color = PALETTE['model_ours']
            alpha = 1.0
        else:
            # 渐变色
            color_idx = min(i // (len(data_sorted) // 4 + 1), 3)
            color = BASELINE_GRADIENT[color_idx]
            alpha = 0.8
        
        # 绘制条形
        ax.barh(i, value, color=color, alpha=alpha, 
               edgecolor='white', linewidth=0.5, height=0.7)
        
        # 绘制 CI
        if ci_lower_col and ci_upper_col:
            ci_l = row[ci_lower_col]
            ci_u = row[ci_upper_col]
            ax.plot([ci_l, ci_u], [i, i], color='black', 
                   linewidth=1.5, alpha=0.5)
        
        # 数值标签
        ax.text(value + 0.01, i, f'{value:.3f}', 
               va='center', fontsize=FONT_CONFIG['size_annotation'],
               color=PALETTE['text_secondary'])
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(data_sorted[name_col])
    
    return ax


# ============================================================
# 10. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = RESULTS_DIR / "data"


# ============================================================
# 11. 初始化
# ============================================================

if __name__ == "__main__":
    setup_nature_style()
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Results dir: {RESULTS_DIR}")
    
    # 显示配色
    print("\n=== Nature Biotechnology Palette ===")
    for name, color in PALETTE.items():
        print(f"  {name}: {color}")
