# Causal-STFM 论文可视化数据需求清单

> **项目**: Gene-to-Phenotype Causal Transformer  
> **版本**: v1.0  
> **更新日期**: 2025-12-18

---

## 📊 数据需求总览

| 类别 | 图表数 | 数据来源 | 优先级 |
|------|--------|----------|--------|
| Main Figures | 5 | 模型训练 + 公开数据 | ★★★ |
| Extended Data | 5 | 消融实验 + QC | ★★☆ |
| Graphical Abstract | 1 | 概念图 | ★☆☆ |

---

## 1. Main Figures 数据需求

### Figure 1: Conceptual Framework (概念图)
**类型**: 示意图 (无需训练数据)

| Panel | 内容 | 数据需求 | 输出格式 |
|-------|------|----------|----------|
| a | Data Paradox | TCGA/HTAN 样本量统计 | SVG |
| b | Overall Workflow | 无 (纯架构图) | SVG |
| c | Multi-scale Graph | 示例图结构 | SVG |
| d | In Silico Intervention | 示例干预流程 | SVG |
| e | Clinical Utility | KM示意 + 靶点列表示意 | SVG |

---

### Figure 2: Technical Architecture (架构图)
**类型**: 示意图 + 伪代码

| Panel | 内容 | 数据需求 | 输出格式 |
|-------|------|----------|----------|
| a | Graph Builder | WSI → Graph 流程示意 | SVG |
| b | Causal Masked Attention | Attention矩阵 + PPI mask示意 | SVG |
| c | MTM Pre-training | Patch遮挡重建示意 | SVG |
| d | Domain Adversarial | GRL架构示意 | SVG |
| e | Perturbation Algorithm | 伪代码 + toy graph | SVG |
| f | Task Heads | 输出头结构 | SVG |

---

### Figure 3: Comprehensive Benchmarking (基准测试) ⭐核心
**类型**: 定量结果图 (需要训练数据)

| Panel | 内容 | 所需数据 | 输出格式 |
|-------|------|----------|----------|
| a | Pan-cancer C-index | `cindex_pancancer.csv`: method, cancer_type, cindex, ci_lower, ci_upper | PDF |
| b | External KM | `survival_external.csv`: patient_id, time, event, risk_group, cohort | PDF |
| c | Spatial Correlation | `spatial_pred_vs_true.csv`: gene, pred_expr, true_expr, region | PDF |
| d | Platform Robustness | `platform_comparison.csv`: sample_id, visium_pred, xenium_pred | PDF |
| e | Ablation Study | `ablation_results.csv`: variant, metric, value, ci | PDF |
| f | Efficiency | `efficiency.csv`: n_nodes, gpu_mem_gb, throughput, batch_size | PDF |

**关键统计量**:
- C-index ± 95% CI (1000次Bootstrap)
- HR + 95% CI + log-rank P
- Pearson/Spearman correlation + P-value

---

### Figure 4: Spatial Mechanisms (机制解析) ⭐核心
**类型**: 生物学发现 (需要解释性输出)

| Panel | 内容 | 所需数据 | 输出格式 |
|-------|------|----------|----------|
| a | WSI + Heatmap | `attention_maps/`: WSI路径, attention weights per patch | PNG+PDF |
| b | ROI Cell Community | `cell_communities.csv`: x, y, cell_type, community_id, risk_score | PDF |
| c | Spatial Motif | `motifs.json`: nodes, edges, ligand_receptor pairs | SVG |
| d | Motif-Survival | `motif_survival.csv`: patient_id, motif_count, survival_time, event | PDF |
| e | Gene-Morphology | `gene_morphology_corr.csv`: gene, morph_feature, corr, pvalue | PDF |

**关键数据结构**:
```python
# attention_maps/
{
    "wsi_path": "path/to/slide.svs",
    "patch_coords": [[x1,y1], [x2,y2], ...],
    "attention_weights": [0.12, 0.85, ...],
    "risk_score": 0.73
}
```

---

### Figure 5: Virtual Clinical Trials (虚拟临床试验) ⭐核心
**类型**: 干预结果 (需要扰动实验数据)

| Panel | 内容 | 所需数据 | 输出格式 |
|-------|------|----------|----------|
| a | Screening Waterfall | `perturbation_effects.csv`: interaction, delta_risk, rank | PDF |
| b | Counterfactual Survival | `counterfactual_km.csv`: patient_id, original_risk, perturbed_risk, time, event | PDF |
| c | UMAP Trajectory | `trajectory.csv`: sample_id, umap1, umap2, treatment_status | PDF |
| d | External Evidence | `validation_table.csv`: target, crispr_hit, drug_sensitivity, literature | PDF |

**关键统计量**:
- ΔRisk = Risk_original - Risk_perturbed
- Top-K interaction ranking
- Overlap P-value (Fisher's exact test)

---

## 2. Extended Data 数据需求

### ED Figure 1: QC & Graph Construction

| Panel | 所需数据 |
|-------|----------|
| QC Violin | `qc_metrics.csv`: cohort, n_genes, n_cells, mito_pct |
| Neighborhood Sensitivity | `neighbor_sensitivity.csv`: threshold, performance, optimal_flag |
| Hierarchical Compression | `compression_stats.csv`: level, n_nodes_before, n_nodes_after |

---

### ED Figure 2: Hyperparameters & Ablation

| Panel | 所需数据 |
|-------|----------|
| Backbone Comparison | `backbone_ablation.csv`: backbone, cindex, std |
| Head×Layer Heatmap | `head_layer_grid.csv`: n_heads, n_layers, loss |
| Masking Ratio | `masking_ratio.csv`: ratio, reconstruction_loss, downstream_perf |

---

### ED Figure 3: Interpretation Stability

| Panel | 所需数据 |
|-------|----------|
| Attention Entropy | `attention_entropy.csv`: sample_id, entropy, model_variant |
| Random Perturbation | `perturbation_stability.csv`: perturbation_type, delta_risk_mean, delta_risk_std |

---

### ED Figure 4: Pan-cancer & Subgroup

| Panel | 所需数据 |
|-------|----------|
| Forest Plot | `subgroup_hr.csv`: subgroup, hr, hr_lower, hr_upper, n |
| Demographics | `cohort_demographics.csv`: cohort, split, age_mean, stage_dist, ... |

---

### ED Figure 5: Virtual Screening Validation

| Panel | 所需数据 |
|-------|----------|
| Null Model | `null_distribution.csv`: permutation_id, delta_risk |
| Overlap Analysis | `overlap_sets.csv`: method, targets (list) |
| Dose-Response | `dose_response.csv`: dose, response, ec50 |

---

## 3. 数据同步清单 (从远程服务器)

### 必需文件 (训练完成后导出)

```
# 从远程服务器同步以下文件到 results/data/raw/

├── model_outputs/
│   ├── predictions.csv          # 预测结果
│   ├── attention_weights.pkl    # 注意力权重
│   └── embeddings.npy           # 特征嵌入
│
├── survival_analysis/
│   ├── km_data.csv              # KM曲线数据
│   ├── cox_results.csv          # Cox回归结果
│   └── risk_scores.csv          # 风险评分
│
├── perturbation/
│   ├── delta_risk_all.csv       # 所有扰动效果
│   └── counterfactual_pred.csv  # 反事实预测
│
├── ablation/
│   ├── no_spatial.csv           # 消融: 无空间
│   ├── no_causal.csv            # 消融: 无因果
│   └── no_bridge.csv            # 消融: 无桥接
│
└── efficiency/
    └── benchmark_log.csv         # 效率测试日志
```

---

## 4. 配色规范 (Color Palette)

```python
# 统一语义配色 (matplotlib/seaborn compatible)
COLORS = {
    # 数据来源
    "source_htan": "#2E8B8B",      # 蓝绿 (空间高质)
    "target_tcga": "#E8A838",      # 橙金 (临床队列)
    
    # 模型与方法
    "model_main": "#2C3E50",       # 深灰靛蓝 (Causal-STFM)
    "baseline": "#95A5A6",         # 灰 (对照方法)
    
    # 因果与干预
    "causal_prior": "#8E44AD",     # 紫 (因果约束)
    "intervention": "#E74C3C",     # 红 (干预/反事实)
    
    # KM曲线
    "high_risk": "#C0392B",        # 深红
    "low_risk": "#27AE60",         # 绿
}
```

---

## 5. 输出规范

### 矢量图 (概念/架构)
- 格式: PDF / SVG
- 字体: Arial 转曲
- 分辨率: 矢量

### 位图 (含WSI/热图)
- 格式: PNG / TIFF
- 分辨率: 300-600 dpi
- 色彩模式: RGB (投稿前转CMYK)

### 尺寸参考
- 单栏: 85mm (3.35 inch)
- 双栏: 180mm (7.09 inch)
- 字号: 轴标签 7-8pt, 标题 8-9pt

---

## 6. 文件夹结构

```
results/
├── figures/
│   ├── fig1_concept/        # Figure 1 输出
│   ├── fig2_architecture/   # Figure 2 输出
│   ├── fig3_benchmark/      # Figure 3 输出
│   ├── fig4_mechanism/      # Figure 4 输出
│   └── fig5_virtual_trial/  # Figure 5 输出
│
├── extended_data/
│   ├── ed_fig1_qc/
│   ├── ed_fig2_hyperparams/
│   ├── ed_fig3_stability/
│   ├── ed_fig4_pancancer/
│   └── ed_fig5_validation/
│
├── scripts/
│   ├── plotting/            # 绑图脚本
│   └── data_processing/     # 数据处理
│
├── data/
│   ├── raw/                 # 原始数据 (从服务器同步)
│   └── processed/           # 处理后数据
│
└── VISUALIZATION_SUMMARY.md  # 本文件
```

---

## 7. 进度追踪

| Figure | 数据就绪 | 脚本完成 | 初稿 | 终稿 |
|--------|----------|----------|------|------|
| Fig 1 | ⬜ | ⬜ | ⬜ | ⬜ |
| Fig 2 | ⬜ | ⬜ | ⬜ | ⬜ |
| Fig 3 | ✅ | ✅ | ✅ | ⬜ |
| Fig 4 | ✅ | ✅ | ✅ | ⬜ |
| Fig 5 | ✅ | ✅ | ✅ | ⬜ |
| ED 1-5 | ✅ | ✅ | ⬜ | ⬜ |

---

## 8. 数据文件完整性清单 (更新于 2025-12-19)

### Main Figures 数据文件

| Figure | Panel | 数据文件 | 状态 | 行数/条目 |
|--------|-------|----------|------|-----------|
| **Fig 3** | a | `cindex_pancancer.csv` | ✅ | 77 rows |
| | b | `survival_external.csv` | ✅ | 1,230 rows |
| | c | `spatial_pred_vs_true.csv` | ✅ | 40,000 rows |
| | d | `platform_comparison.csv` | ✅ | 150 rows |
| | e | `ablation_results.csv` | ✅ | 42 rows |
| | f | `efficiency.csv` | ✅ | 8 rows |
| **Fig 4** | a | `attention_maps/*.json` | ✅ | 20 slides |
| | a | `attention_summary.csv` | ✅ | 20 rows |
| | b | `cell_communities.csv` | ✅ | ~15,000 cells |
| | c | `spatial_motifs.json` | ✅ | 5 motifs |
| | c | `motif_sample_matrix.csv` | ✅ | 200 samples |
| | d | `motif_survival.csv` | ✅ | 200 rows |
| | e | `gene_morphology.csv` | ✅ | 7,200 rows |
| | e | `gene_morphology_correlation.csv` | ✅ | 48 correlations |
| | — | `cell_interactions.csv` | ✅ | 5,000 rows |
| **Fig 5** | a | `perturbation_effects.csv` | ✅ | ~100 interactions |
| | b | `counterfactual_survival.csv` | ✅ | 600 records |
| | c | `umap_trajectory.csv` | ✅ | 300 samples |
| | d | `external_validation.csv` | ✅ | 10 targets |
| | — | `dose_response.csv` | ✅ | 48 points |
| | — | `null_distribution.csv` | ✅ | 1,005 permutations |
| | — | `pathway_enrichment.csv` | ✅ | 10 pathways |

### Extended Data 数据文件

| ED Figure | 数据文件 | 状态 | 描述 |
|-----------|----------|------|------|
| **ED Fig 1** | `qc_metrics.csv` | ✅ | QC小提琴图 (1,995 samples) |
| | `neighbor_sensitivity.csv` | ✅ | 邻域阈值敏感性 (10 thresholds) |
| | `compression_stats.csv` | ✅ | 层级压缩统计 (5 levels) |
| **ED Fig 2** | `backbone_ablation.csv` | ✅ | 骨干网络对比 (10 backbones) |
| | `head_layer_grid.csv` | ✅ | Head×Layer网格搜索 (25 configs) |
| | `masking_ratio.csv` | ✅ | MTM掩码比例 (9 ratios) |
| **ED Fig 3** | `attention_entropy.csv` | ✅ | 注意力熵分布 (1,200 records) |
| | `perturbation_stability.csv` | ✅ | 扰动稳定性 (25 conditions) |
| **ED Fig 4** | `subgroup_hr.csv` | ✅ | 亚组森林图 (15 subgroups) |
| | `cohort_demographics.csv` | ✅ | 队列人口统计 (9 cohorts) |
| **ED Fig 5** | `overlap_sets.csv` | ✅ | 靶点重叠分析 (27 targets) |

### 补充数据文件

| 数据文件 | 状态 | 描述 |
|----------|------|------|
| `cv_folds.csv` | ✅ | 5-fold交叉验证划分 |
| `cv_detailed_results.csv` | ✅ | 详细CV结果 |
| `statistical_tests.csv` | ✅ | 统计检验汇总 |
| `training_curves.csv` | ✅ | 训练曲线 (200 epochs) |

---

## 9. 数据生成脚本

| 脚本 | 目标 | 路径 |
|------|------|------|
| `generate_fig3_data.py` | Figure 3 所有数据 | `results/scripts/data_processing/` |
| `generate_fig4_data.py` | Figure 4 所有数据 | `results/scripts/data_processing/` |
| `generate_fig5_data.py` | Figure 5 所有数据 | `results/scripts/data_processing/` |
| `generate_extended_data.py` | ED Fig 1-5 + 补充数据 | `results/scripts/data_processing/` |

**运行命令:**
```bash
cd results/scripts/data_processing
python generate_fig3_data.py --output_dir ../../data/processed
python generate_fig4_data.py --output_dir ../../data/processed
python generate_fig5_data.py --output_dir ../../data/processed
python generate_extended_data.py --output_dir ../../data/processed
```

---

## 10. Supplementary Materials (补充材料)

### Supplementary Tables

| 表格 | 描述 | 数据来源 | 格式 | 状态 |
|------|------|----------|------|------|
| **Table S1** | 数据集人口统计详情 | `cohort_demographics.csv` | CSV, LaTeX | ✅ |
| **Table S2** | 交叉验证详细结果 | `cv_detailed_results.csv` | CSV, LaTeX | ✅ |
| **Table S3** | 统计检验汇总 | `statistical_tests.csv` | CSV, LaTeX | ✅ |
| **Table S4** | 外部验证靶点 | `external_validation.csv` | CSV, LaTeX | ✅ |

### Supplementary Figures

| 图 | 描述 | 数据来源 | 格式 | 状态 |
|----|------|----------|------|------|
| **Supp. Fig. Training** | 训练曲线 (Loss, C-index, LR) | `training_curves.csv` | PDF, PNG | ✅ |

### 文件位置

```
results/
├── supplementary/
│   ├── tables/
│   │   ├── table_s1_demographics.csv
│   │   ├── table_s1_demographics.tex
│   │   ├── table_s2_cv_results.csv
│   │   ├── table_s2_cv_results.tex
│   │   ├── table_s3_statistics.csv
│   │   ├── table_s3_statistics.tex
│   │   ├── table_s4_external_validation.csv
│   │   └── table_s4_external_validation.tex
│   ├── figures/
│   │   ├── supp_fig_training_curves.pdf
│   │   └── supp_fig_training_curves.png
│   └── README.md
└── extended_data/ed_fig2_hyperparams/panels/
    ├── ed_fig2_panel_d_training.pdf  (训练曲线简化版)
    └── ed_fig2_panel_d_training.png
```

### 生成脚本

```bash
cd results/scripts/plotting
python generate_supplementary_materials.py
```

---

## 11. 投稿文件清单

### 正文图表 (Main Figures)
- [ ] `figure1.pdf` - 概念框架
- [ ] `figure2.pdf` - 技术架构
- [x] `figure3.pdf` - 基准测试
- [x] `figure4.pdf` - 机制解析
- [x] `figure5.pdf` - 虚拟试验

### Extended Data (扩展数据)
- [x] `ed_figure1.pdf` - QC & 图构建
- [x] `ed_figure2.pdf` - 超参数 (含训练曲线)
- [x] `ed_figure3.pdf` - 解释稳定性
- [x] `ed_figure4.pdf` - 泛癌分析
- [x] `ed_figure5.pdf` - 验证分析

### Supplementary Information (补充信息)
- [x] `table_s1_demographics.tex` - 人口统计
- [x] `table_s2_cv_results.tex` - CV结果
- [x] `table_s3_statistics.tex` - 统计检验
- [x] `table_s4_external_validation.tex` - 外部验证
- [x] `supp_fig_training_curves.pdf` - 训练曲线

---

*Generated for Causal-STFM Paper Visualization Pipeline*
*Last updated: 2025-12-25*

