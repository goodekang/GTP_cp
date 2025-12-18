## GTP-CP 简要设计文档（优雅 / 高级 / 高效 / 简洁）

### 1. 设计目标
- **优雅**：模块边界清晰；同一概念只在一个地方实现；接口尽量小。
- **高级**：配置驱动（Hydra）；可复现（seed）；指标与训练逻辑独立。
- **高效**：最小可运行闭环（合成数据也能训练/评估）；可扩展到真实数据。
- **简洁**：不引入重量级框架（Lightning 等）也能保持可读；只保留必要依赖。

### 2. 目录结构（建议长期保持不变）

```
configs/
  train.yaml              # 训练配置（默认）
  eval.yaml               # 评估配置（默认）
docs/
  DESIGN.md               # 本文档
src/gtp_cp/
  cli/                    # 命令行入口（gtp-train / gtp-eval）
  data/                   # dataset & loader（当前：合成数据）
  graphs/                 # 图构建接口（当前：占位接口）
  models/                 # 模型（当前：最小 CausalSTFM）
  train/                  # loss、train/eval engine
  utils/                  # seed、metrics、io
```

### 3. 模块边界与核心数据结构

- **图数据**：统一使用 PyG `Data`/`Batch`
  - `x`: 节点特征
  - `edge_index`: 图边
  - `node_type`: {0 patch, 1 cell, 2 gene}（演示用）
- **标签**：生存 `time`（float），`event`（0/1）
- **模型输出**：`log_risk`（shape [B]），越大表示风险越高（hazard 越高）

### 4. 训练与评估

- **loss**：Cox partial likelihood（`train/losses.py`）
- **metric**：Harrell C-index（`utils/metrics.py`）
- **engine**：
  - `fit(...)`：训练 + val 选择 best checkpoint
  - `evaluate(...)`：在 val/test 上输出 loss 与 c-index

### 5. “因果掩码”的工程落点（接口先行）

当前 `models/CausalSTFM` 里保留 `use_causal_mask` 与 `_edge_weight(...)`，作为：
- **演示**：对某些边对进行权重调整
- **未来真实实现**：把 PPI/Reactome/ligand–receptor priors 编译成 mask，并在注意力里做约束/门控

> 关键：先把 mask 的入口固定住，后续替换内部实现不会影响训练/评估脚本。

### 6. 如何接入真实数据（HTAN/TCGA）

建议按“数据→图→训练”三段替换：

1) `graphs/builder.py`: 实现 `GraphBuilder.build(raw) -> Data`
   - WSI → patch/superpixel
   - cell mapping / spatial neighborhood
   - gene selection / GRN / pathway edges

2) `data/`: 增加真实 `Dataset`
   - 统一输出 `(Data, time, event)`，不要把复杂 preprocessing 塞进训练脚本

3) `configs/`: 新增 `dataset: htan/tcga` 的配置分支

### 7. 运行方式

训练（默认合成数据）：

```bash
gtp-train
```

评估（自动找最新 checkpoint）：

```bash
gtp-eval split=test
```

覆盖任意配置项（Hydra）：

```bash
gtp-train train.epochs=50 model.num_layers=6 data.batch_size=8
```


