# GTP-CP (Causal-STFM) — Research Code Scaffold

This repository contains a **minimal, elegant, reproducible** scaffold for the project described in `实验设计.md`.

## Quickstart (recommended: conda, synthetic demo)

1) Create env (Windows PowerShell):

```bash
conda env create -f environment.yml
conda activate gtp-cp
```

2) Install the project (editable):

```bash
pip install -e ".[dev]" --no-deps
```

3) Train on synthetic survival graph data:

```bash
gtp-train
```

4) Evaluate the last checkpoint (prints metrics):

```bash
gtp-eval
```

## What this is (and isn’t)

- **Is**: a clean project layout + config system + minimal runnable pipeline.
- **Isn’t**: full production implementation of WSI tiling / spatial transcriptomics ingestion.

See `docs/DESIGN.md` for the design document and extension points.
See `docs/ENV_SETUP.md` for environment setup details (CUDA/CPU).
See `docs/BRCA_PIPELINE.md` for the TCGA-BRCA publication-oriented pipeline.

## TCGA-BRCA (publication pipeline)

Generate public manifests/labels/splits (no private data required):

```bash
gtp-brca-manifest --root data/brca
gtp-brca-clinical --root data/brca
gtp-brca-split --root data/brca --seed 7
```

After you prepare patch features (`patch_features.csv`), build graphs and train:

```bash
gtp-brca-graphs --root data/brca --features-csv data/brca/features/patch_features.csv --k 8
gtp-brca-train --graphs-index data/brca/graphs/graphs_index.csv --out artifacts/brca --device cuda
```
