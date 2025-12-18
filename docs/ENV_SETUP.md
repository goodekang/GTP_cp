## Conda 环境创建与记录（Windows 推荐）

> 目标：在 Windows 上稳定安装 `torch` + `torch_geometric`（PyG），并记录一条可复现的环境搭建路径。

---

## 1) 环境创建

仓库已提供 `environment.yml`。

```bash
conda env create -f environment.yml
conda activate gtp-cp
```

说明：
- `python=3.11`：在 Windows 上对 PyG 生态更稳（相较 3.12）。

---

## 2) CUDA / CPU 选择

### 2.1 RTX 5060 / RTX 50xx (Blackwell 架构) - CUDA 12.8 ⭐推荐

RTX 50 系列 (Blackwell 架构) 需要 **CUDA 12.8** 才能获得最佳支持。

参考官方安装指南: https://pytorch.org/get-started/locally/

**安装步骤：**

1. 创建基础 conda 环境：
```bash
conda env create -f environment.yml
conda activate gtp-cp
```

2. 安装 PyTorch 2.7.0 + CUDA 12.8：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

3. 验证安装：
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**重要提示：**
- 确保 NVIDIA 驱动版本 >= 560.94（支持 CUDA 12.8）
- RTX 5060 需要最新驱动以获得完整 Blackwell 架构支持

### 2.2 旧版 GPU (RTX 30xx/40xx) - CUDA 12.1/12.4

如果你使用 RTX 30/40 系列 GPU：

```bash
# CUDA 12.4 (推荐用于 RTX 40xx)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 或 CUDA 12.1 (兼容性更好)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 CPU-only

如果你只用 CPU：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 3) 安装本项目（可编辑）

即使 `environment.yml` 已包含 `-e .`，你也可以手动执行：

```bash
pip install -e ".[dev]" --no-deps
```

> 为什么 `--no-deps`：避免 pip 尝试重新解析并覆盖 conda 已安装的 torch/pyg。

---

## 4) 冒烟验证（建议每次换环境后跑一次）

```bash
pytest -q
gtp-train
gtp-eval
```

---

## 5) 环境记录（提交到论文/补充材料时可用）

```bash
conda env export --no-builds > env.lock.yml
python -V
pip -V
python -c "import torch; print(torch.__version__)"
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

生成的 `env.lock.yml` 建议纳入版本控制（或作为补充材料归档）。


