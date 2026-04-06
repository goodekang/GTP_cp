# Causal-STFM 论文图表拼图策略指南

> **项目**: Gene-to-Phenotype Causal Transformer  
> **版本**: Publication-Ready  
> **更新日期**: 2025-12-29

---

## 📊 最终文件结构

```
publication/
├── figure/                      # Main Figures (5)
│   ├── fig1_concept.pdf
│   ├── fig2_architecture.pdf
│   ├── fig3_benchmark.pdf
│   ├── fig4_mechanism.pdf
│   └── fig5_virtual_trial.pdf
│
└── supplementary/
    ├── figures/                 # Supplementary Figures (6)
    │   ├── Supp Fig S1 - Data QC & Graph Construction
    │   ├── Supp Fig S2 - Hyperparameter Optimization
    │   ├── Supp Fig S3 - Pan-cancer Detailed Analysis
    │   ├── Supp Fig S4 - Spatial Mechanism Supplements
    │   ├── Supp Fig S5 - Virtual Trial Supplements
    │   └── Supp Fig S6 - Training Curves
    └── tables/                  # Supplementary Tables (4)
        ├── Table S1 - Demographics
        ├── Table S2 - CV Results
        ├── Table S3 - Statistics
        └── Table S4 - External Validation
```

---

## 🎯 核心策略：无 Extended Data

### 决策依据
- **兼容性**：适用于 Cell/Science/PNAS 等不支持 Extended Data 的期刊
- **简洁性**：关键验证内容合并到主图，方法细节放补充图
- **可读性**：避免主图过于拥挤

### 原 Extended Data 内容归属

| 原 ED 内容 | 新归属 | 说明 |
|------------|--------|------|
| ED1 QC | **Supp Fig S1** | 数据质量控制 |
| ED2 超参数 | **Supp Fig S2** | 模型优化细节 |
| ED3 稳定性 | **Fig 4 扩展** | 解释性核心证据 |
| ED4 泛癌 | **Fig 3 扩展** + **Table S1** | 亚组分析进主图，表格进补充 |
| ED5 验证 | **Fig 5 扩展** + **Supp Fig S5** | 统计验证进主图 |

---

## 📐 面板规格

### 主图标准
| 属性 | 值 |
|------|-----|
| 宽度 | 180mm（双栏） |
| 字体 | Arial/Helvetica |
| 轴标签 | 8pt |
| 刻度 | 7pt |
| Panel 标签 | 9pt bold |

### 紧凑面板（用于拼图）
| 属性 | 值 |
|------|-----|
| 尺寸 | 65mm × 55mm |
| 字体放大 | +1-2pt |
| 热图尺寸 | 70mm × 58mm |

---

## 🎨 配色规范

### 统一语义色

```
├── 空间/源数据  → 青绿系 (#005F73, #0A9396, #94D2BD)
├── 临床/目标    → 琥珀系 (#CA6702, #EE9B00, #FFD166)
├── 因果约束     → 紫色系 (#6A0DAD, #9B5DE5)
├── 干预/风险    → 红色系 (#9D0208, #E63946)
└── 我们的方法   → 深蓝 (#1D3557)
```

### 各图配色一致性
- Fig 3-5 统一使用上述配色
- 森林图：酒红点/线，深蓝菱形（Overall）
- 热图：白→琥珀→深橙 暖色渐变
- KM曲线：青绿(Low Risk) vs 酒红(High Risk)

---

## 📋 关键改动记录

### Fig 3 Benchmark
- **新增 Panel d**: 亚组森林图（原 ED4-a）
- **新增 Panel g**: CV 稳定性（原 ED4-c）
- 表格内容 (Demographics) → **Table S1**

### Fig 4 Mechanism  
- **新增 Panel f-h**: 注意力熵/扰动稳定性/层级分析（原 ED3）
- 保持与主图配色协调

### Fig 5 Virtual Trial
- **新增 Panel e**: 零分布排列检验（原 ED5-d）
- **新增 Panel f**: 靶点重叠分析（原 ED5-a）

---

## 📝 补充材料说明

| 补充图 | 内容 | 原来源 |
|--------|------|--------|
| S1 | QC + 图构建 | ED Fig 1 |
| S2 | 超参数优化 | ED Fig 2 |
| S3 | 泛癌详细分析 | ED Fig 4 部分 + Fig3 supp |
| S4 | 空间机制补充 | Fig4 supp |
| S5 | 虚拟试验补充 | Fig5 supp + ED Fig 5 部分 |
| S6 | 训练曲线 | 原 Supp |

| 补充表 | 内容 |
|--------|------|
| S1 | 队列人口统计（原 ED4-b） |
| S2 | 交叉验证详细结果 |
| S3 | 统计检验汇总（原 ED5-c） |
| S4 | 外部验证靶点 |

---

## ✅ 投稿检查清单

- [x] 主图 5 个，符合期刊限制
- [x] 无 Extended Data，全部整合
- [x] 配色统一，与项目 plot_config 一致
- [x] 字体可读，符合 Nature 标准
- [x] 统计信息完整（CI, P-value, n）
- [x] 表格从图中移出，放入补充材料

---

*Generated: 2025-12-29*

