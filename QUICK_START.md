# FIF-A Quick Start Guide

**文档信息**
- 创建日期：2024-12-08
- 适用版本：v1.2.0+
- 相关文档：[EXPERIMENT_LAUNCHER_GUIDE.md](EXPERIMENT_LAUNCHER_GUIDE.md), [README.md](README.md)

---

## 最简运行方式

### 1. 运行所有实验（4个）

```bash
./run.sh
```

这会运行：
- SNLI Baseline
- SNLI Hybrid (with FIF)
- SST-2 Baseline
- SST-2 Hybrid (with FIF)

### 2. 快速选择器（推荐）

使用 `quick.sh` 快速运行特定实验：

```bash
# 运行所有 SNLI 实验（baseline + hybrid）
./quick.sh snli

# 运行所有 SST-2 实验（baseline + hybrid）
./quick.sh sst2

# 运行所有 baseline 实验（SNLI + SST-2）
./quick.sh baseline

# 运行所有 hybrid 实验（SNLI + SST-2）
./quick.sh hybrid

# 运行单个实验
./quick.sh snli_baseline
./quick.sh snli_hybrid
./quick.sh sst2_baseline
./quick.sh sst2_hybrid

# 预览命令（不执行）
./quick.sh snli --dry-run
./quick.sh sst2 --dry-run
```

### 3. 查看帮助

```bash
./quick.sh --help
```

---

## 实验配置

所有实验都配置在 `scripts/experiments.yaml`：

| 实验名 | 任务 | 模型 | Epochs | 批大小 | 说明 |
|---|---|---|---:|---:|---|
| `snli_baseline` | SNLI | Baseline | 5 | 256 | Transformer baseline |
| `snli_hybrid` | SNLI | Hybrid | 5 | 256 | FIF hybrid (K=3, rank) |
| `sst2_baseline` | SST-2 | Baseline | 3 | 512 | Transformer baseline |
| `sst2_hybrid` | SST-2 | Hybrid | 3 | 512 | FIF hybrid (K=3, rank) |

**预计运行时间**（单 A100 GPU）：
- SNLI: ~15分钟/实验
- SST-2: ~8分钟/实验
- 全部4个实验：~45分钟

---

## 结果查看

实验结果保存在 `./result/` 目录：

```
result/
├── snli_baseline_20241208_143022_seed42/
│   ├── test_summary.json         # 最终测试结果
│   ├── metrics_epoch.csv          # 训练曲线
│   ├── energy_error_correlation.json  # 能量相关性
│   └── ...
├── snli_hybrid_20241208_150145_seed42/
├── sst2_baseline_20241208_153020_seed42/
└── sst2_hybrid_20241208_160315_seed42/
```

**查看关键结果**：

```bash
# 查看测试集准确率
cat result/snli_baseline_*/test_summary.json | jq '.acc'

# 查看能量-错误相关性
cat result/snli_hybrid_*/energy_error_correlation.json | jq '.pearson_r'
```

---

## 常见场景

### 场景 1：快速验证代码修改

```bash
# 只运行最快的实验验证代码
./quick.sh sst2_baseline
```

### 场景 2：对比 Baseline vs Hybrid

```bash
# SNLI 对比
./quick.sh snli

# SST-2 对比
./quick.sh sst2
```

### 场景 3：评估 FIF 效果

```bash
# 运行所有 hybrid 实验
./quick.sh hybrid
```

### 场景 4：预览配置不执行

```bash
# 查看命令但不运行
./quick.sh all --dry-run
```

---

## 高级用法

### 修改实验参数

编辑 `scripts/experiments.yaml`：

```yaml
# 示例：增加 SNLI epochs
- name: snli_baseline
  params:
    epochs: 10  # 从 5 改为 10
```

### 直接运行 CLI（绕过 YAML）

```bash
python -m fif_mvp.cli.run_experiment \
  --task snli \
  --model baseline \
  --epochs 5 \
  --batch_size 256 \
  --seed 42
```

完整参数说明：`python -m fif_mvp.cli.run_experiment --help`

### 运行归档的实验（noisy/v1.1.0）

旧的实验配置保存在 `scripts/experiments_archive.yaml`：

```bash
# 运行 noisy SST-2
python scripts/run_experiments.py \
  --config scripts/experiments_archive.yaml \
  --select sst2_noisy_baseline

# 运行 v1.1.0 失败方案（用于对比）
python scripts/run_experiments.py \
  --config scripts/experiments_archive.yaml \
  --select snli_hybrid_k1_absolute
```

---

## 故障排查

### 问题：找不到 quick.sh

```bash
# 确保在项目根目录
cd /path/to/FIF-A

# 确保有执行权限
chmod +x quick.sh
```

### 问题：CUDA out of memory

编辑 `scripts/experiments.yaml` 减小批大小：

```yaml
params:
  batch_size: 128  # 从 256/512 减小
```

### 问题：实验中断如何恢复

实验会自动保存，但不支持断点续训。需要重新运行：

```bash
./quick.sh <experiment_name>
```

---

## 与文档的关联

**实验前**：
- 查看 [WORK_BOARD.md](WORK_BOARD.md) 了解当前任务

**实验后**：
- 更新 [WORK_BOARD.md](WORK_BOARD.md) 任务状态
- 重要结果补充到 [PHASE_RESULTS.md](PHASE_RESULTS.md)
- 版本 milestone 达成时更新 [PROJECT_TRACKER.md](PROJECT_TRACKER.md)

---

## 快速参考卡

```bash
# 运行全部
./run.sh

# 快速选择
./quick.sh snli          # SNLI 两个实验
./quick.sh sst2          # SST-2 两个实验
./quick.sh baseline      # 两个 baseline
./quick.sh hybrid        # 两个 hybrid

# 单个实验
./quick.sh snli_baseline
./quick.sh snli_hybrid
./quick.sh sst2_baseline
./quick.sh sst2_hybrid

# 预览
./quick.sh <target> --dry-run

# 帮助
./quick.sh --help
```

---

**详细文档**：见 [EXPERIMENT_LAUNCHER_GUIDE.md](EXPERIMENT_LAUNCHER_GUIDE.md)
