# Supplementary Materials Summary

## Supplementary Tables

| Table | Description | Format |
|-------|-------------|--------|
| Table S1 | Dataset Demographics | CSV, LaTeX |
| Table S2 | Cross-Validation Detailed Results | CSV, LaTeX |
| Table S3 | Statistical Tests Summary | CSV, LaTeX |
| Table S4 | External Validation of Predicted Targets | CSV, LaTeX |

## Supplementary Figures

| Figure | Description | Source | Format |
|--------|-------------|--------|--------|
| Supp. Fig. S1 | Training curves (Loss, C-index, LR schedule) | — | PDF, PNG |
| Supp. Fig. S2 | Pan-cancer C-index Heatmap | Fig 3 | PDF, PNG |
| Supp. Fig. S3 | Attention Map Analysis | Fig 4 | PDF, PNG |
| Supp. Fig. S4 | Spatial Motif Patterns | Fig 4 | PDF, PNG |
| Supp. Fig. S5 | Null Distribution & Dose-Response | Fig 5 | PDF, PNG |
| Supp. Fig. S6 | Pathway Enrichment Analysis | Fig 5 | PDF, PNG |
| Supp. Fig. S7 | Multi-Intervention Comparison | Fig 5 | PDF, PNG |

## Files Generated

### Tables Directory
- `table_s1_demographics.csv`
- `table_s1_demographics.tex`
- `table_s2_cv_results.csv`
- `table_s2_cv_results.tex`
- `table_s3_statistics.csv`
- `table_s3_statistics.tex`
- `table_s4_external_validation.csv`
- `table_s4_external_validation.tex`

### Figures Directory
- `supp_fig_training_curves.pdf` / `.png` — Training curves (S1)
- `supp_fig_s2_cindex_heatmap.pdf` / `.png` — Pan-cancer C-index heatmap
- `supp_fig_s3_attention.pdf` / `.png` — Attention map analysis
- `supp_fig_s4_motifs.pdf` / `.png` — Spatial motif patterns
- `supp_fig_s5_null_dose.pdf` / `.png` — Null distribution & dose-response
- `supp_fig_s6_pathway.pdf` / `.png` — Pathway enrichment
- `supp_fig_s7_multi_intervention.pdf` / `.png` — Multi-intervention comparison

## Figure Details

### Supp. Fig. S1: Training Curves
- Training/validation loss over epochs
- C-index progression during training
- Learning rate schedule

### Supp. Fig. S2: Pan-cancer C-index Heatmap
- Comprehensive C-index comparison across cancer types
- Method-wise performance heatmap
- Statistical significance annotations

### Supp. Fig. S3: Attention Map Analysis
- Layer-wise attention entropy distribution
- Attention weight visualization
- Head-specific patterns

### Supp. Fig. S4: Spatial Motif Patterns
- Identified spatial motifs in tumor microenvironment
- Motif frequency distribution
- Cell type composition within motifs

### Supp. Fig. S5: Null Distribution & Dose-Response
- Permutation null distribution for significance testing
- Dose-response curves for in silico perturbations
- Statistical validation

### Supp. Fig. S6: Pathway Enrichment Analysis
- Top enriched pathways from perturbation analysis
- Enrichment scores and p-values
- Pathway network visualization

### Supp. Fig. S7: Multi-Intervention Comparison
- Comparison of single vs. combinatorial interventions
- Synergy analysis
- Risk reduction quantification

## Usage Notes

1. **LaTeX Tables**: Import using `\input{table_s1_demographics.tex}`
2. **CSV Tables**: For data analysis or custom formatting
3. **PDF Figures**: Vector format for publication
4. **PNG Figures**: For preview and presentations

## File Organization Summary

```
supplementary/
├── figures/
│   ├── supp_fig_training_curves.pdf/png     # S1: Training
│   ├── supp_fig_s2_cindex_heatmap.pdf/png   # S2: From Fig 3
│   ├── supp_fig_s3_attention.pdf/png        # S3: From Fig 4
│   ├── supp_fig_s4_motifs.pdf/png           # S4: From Fig 4
│   ├── supp_fig_s5_null_dose.pdf/png        # S5: From Fig 5
│   ├── supp_fig_s6_pathway.pdf/png          # S6: From Fig 5
│   └── supp_fig_s7_multi_intervention.pdf/png # S7: From Fig 5
├── tables/
│   ├── table_s1_demographics.csv/tex
│   ├── table_s2_cv_results.csv/tex
│   ├── table_s3_statistics.csv/tex
│   └── table_s4_external_validation.csv/tex
└── README.md
```

---

Generated: 2025-12-25
Updated: Consolidated supplementary figures from fig3-fig5 folders
