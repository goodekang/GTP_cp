## TCGA-BRCA 一次到位 Pipeline（公开数据 → 图 → 训练 → 可投稿输出）

> 目标：在不依赖私有数据的前提下，生成 Nature 子刊可写作的"证据产物"：
> - 可复现的 manifest / clinical / split
> - 图构建与训练日志、checkpoint
> - Figure-ready 的 metrics 与 per-patient predictions（CSV）

---

## 0) 前置：环境

按 `docs/ENV_SETUP.md` 创建 conda 环境，并确保 `torch_geometric` 可用。

---

## 快速开始（Demo 模式，无需下载 WSI）

如果只想快速测试 pipeline，可以使用 demo 数据：

```bash
# 1. 获取元数据
gtp-brca-manifest --root data/brca
gtp-brca-clinical --root data/brca
gtp-brca-split --root data/brca --seed 7

# 2. 生成合成 demo 数据（50 patients × 100 patches）
gtp-brca-demo --root data/brca --n-patients 50 --n-patches 100

# 3. 构建图
gtp-brca-graphs --root data/brca --features-csv data/brca/features/patch_features.csv --k 8

# 4. 训练
gtp-brca-train --graphs-index data/brca/graphs/graphs_index.csv --out artifacts/brca --epochs 20 --device cpu
```

---

## 完整流程（使用真实 WSI 数据）

### 1) 生成 TCGA-BRCA WSI manifest（公开 GDC API）

```bash
gtp-brca-manifest --root data/brca
```

输出：
- `data/brca/manifests/tcga_brca_wsi_manifest.csv`

说明：
- 该 manifest 只负责列出 `file_id/file_name/patient_id` 等信息，**不自动下载 SVS**。
- 下载推荐使用 `gdc-client`（GDC 官方工具），并把下载的 slide 存放到你自己的 raw 目录。

---

### 1.1) 生成 gdc-client manifest.tsv

```bash
gtp-brca-gdc-manifest --root data/brca --svs-only
```

输出：
- `data/brca/manifests/gdc_manifest.tsv`

---

### 1.2) 下载 gdc-client（Windows）

（可选）如果你还没有 `gdc-client.exe`，可运行：

```powershell
.\scripts\brca\download_gdc_client.ps1 -Version 1.6.1 -OutPath gdc-client.exe
```

> 如果该 URL 版本更新导致下载失败，请到 GDC 官网下载 Windows x64 版本并将 `gdc-client.exe` 放在仓库根目录或自行指定路径。

---

### 1.3) 下载 TCGA-BRCA WSI（SVS）

```powershell
.\scripts\brca\download_gdc_wsi.ps1 -Root data\brca -OutDir data\brca\raw\gdc_wsi -GdcClient .\gdc-client.exe
```

如果需要 token（受控数据）：

```powershell
.\scripts\brca\download_gdc_wsi.ps1 -Root data\brca -OutDir data\brca\raw\gdc_wsi -GdcClient .\gdc-client.exe -TokenFile .\gdc_token.txt
```

下载后的目录约定：
- `data/brca/raw/gdc_wsi/` 下会包含多个以 `file_id` 命名的子目录或文件（由 gdc-client 决定），保持原样即可。

> ⚠️ **注意**：TCGA-BRCA 全量 WSI 约 500GB+，请确保有足够磁盘空间。

---

### 2) 获取临床 OS 标签（公开 GDC API）

```bash
gtp-brca-clinical --root data/brca
```

输出：
- `data/brca/clinical/tcga_brca_clinical.csv`（`patient_id, os_days, os_event`）

---

### 3) patient-level 切分（固定随机种子，可复现）

```bash
gtp-brca-split --root data/brca --seed 7
```

输出：
- `data/brca/splits/patient_split.csv`

---

### 4) WSI 切片（Tiling）

将 SVS 文件切成 patch 图像：

```bash
gtp-brca-tile --root data/brca --wsi-dir data/brca/raw/gdc_wsi --patch-size 256 --mag 20.0
```

输出：
- `data/brca/patches/<slide_id>/<patch_idx>.png`
- `data/brca/patches/patches_index.csv`

参数说明：
- `--patch-size`: patch 大小（默认 256）
- `--mag`: 目标放大倍数（默认 20x）
- `--tissue-thresh`: 组织比例阈值（默认 0.7，低于此值的 patch 会被丢弃）

---

### 5) 特征提取

使用预训练模型提取 patch 特征：

```bash
gtp-brca-features --root data/brca --model resnet50 --device cuda
```

支持的模型：
- `resnet50`: ImageNet 预训练 ResNet50（2048 维）
- `uni`: UNI 病理基础模型（需额外下载权重）
- `dinov2`: DINOv2 ViT-B/14（768 维）

输出：
- `data/brca/features/<patient_id>/<patch_idx>.npy`
- `data/brca/features/patch_features.csv`

---

### 6) 构图（patch kNN）

```bash
gtp-brca-graphs --root data/brca --features-csv data/brca/features/patch_features.csv --k 8
```

输出：
- `data/brca/graphs/<patient_id>.pt`
- `data/brca/graphs/graphs_index.csv`

---

### 7) 训练与导出可投稿 CSV

```bash
gtp-brca-train --graphs-index data/brca/graphs/graphs_index.csv --out artifacts/brca --epochs 30 --device cuda
```

输出（写作直接用）：
- `artifacts/brca/metrics_summary.csv`（主指标）
- `artifacts/brca/predictions.csv`（KM/HR/分层需要的 per-patient 风险）
- `artifacts/brca/best.pt`（checkpoint）

---

## 一键运行（自动串联所有步骤）

```bash
# 完整 pipeline（需要已下载 WSI）
gtp-brca-preprocess --root data/brca --wsi-dir data/brca/raw/gdc_wsi --model resnet50

# 只运行元数据步骤（不需要 WSI）
gtp-brca-preprocess --root data/brca --start-from manifest

# 从特征提取开始（已有 patches）
gtp-brca-preprocess --root data/brca --start-from features --model resnet50
```

---

## CLI 命令汇总

| 命令 | 功能 |
|------|------|
| `gtp-brca-manifest` | 从 GDC API 获取 WSI 文件清单 |
| `gtp-brca-gdc-manifest` | 生成 gdc-client 下载清单 |
| `gtp-brca-clinical` | 获取临床生存数据 |
| `gtp-brca-split` | 创建 train/val/test 切分 |
| `gtp-brca-tile` | WSI 切片（需要 OpenSlide） |
| `gtp-brca-features` | 提取 patch 特征 |
| `gtp-brca-graphs` | 构建 kNN 图 |
| `gtp-brca-train` | 训练模型 |
| `gtp-brca-demo` | 生成 demo 数据（测试用） |
| `gtp-brca-preprocess` | 一键运行完整 pipeline |


