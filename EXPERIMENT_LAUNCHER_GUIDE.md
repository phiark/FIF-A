# FIF-A 实验启动器使用指南

**文档元数据**
- 文档类型：实验启动与配置说明
- 适用版本：v1.2.0+
- 硬件基线：Tesla V100 16G ×1
- 最后更新：2025-12-19
- 相关文档：`run.sh`, `quick.sh`, `scripts/experiments.yaml`, `README.md`, `docs/experiment_design.md`

**快速导航**
- [概述](#1-概述)
- [硬件基线](#2-硬件基线)
- [快速开始](#3-快速开始)
- [配置详解](#4-experimentsyaml-配置详解)
- [输出与产物](#5-输出与产物)
- [常见使用场景](#6-常见使用场景)
- [故障排查](#7-故障排查)
- [最佳实践](#8-最佳实践)
- [快速参考](#9-快速参考)

---

## 1. 概述

FIF-A 提供统一的实验启动器，通过 YAML 配置集中管理实验，减少重复脚本并保证复现一致性。

**核心组件**：
1. `run.sh` - 一键运行标准实验
2. `quick.sh` - 快速选择器（按任务/模型筛选）
3. `scripts/experiments.yaml` - 实验配置清单
4. `scripts/run_experiments.py` - 启动器实现

---

## 2. 硬件基线

- **固定硬件**：Tesla V100 16G ×1（v1.2.0+）
- **支持后端**：CUDA 单卡
- **不再维护**：MPS/多卡 DDP 兼容路径

---

## 3. 快速开始

### 3.1 运行所有标准实验

```bash
./run.sh
```

包含：
- SNLI baseline
- SNLI hybrid (FIF, rank)
- SST-2 baseline
- SST-2 hybrid (FIF, rank)

### 3.2 快速选择器

```bash
# 运行 SNLI 两个实验
./quick.sh snli

# 运行 SST-2 两个实验
./quick.sh sst2

# 运行单个实验
./quick.sh snli_baseline
```

### 3.3 运行指定实验（高级）

```bash
python scripts/run_experiments.py --config scripts/experiments.yaml --select snli_baseline
python scripts/run_experiments.py --config scripts/experiments.yaml --select snli_baseline snli_hybrid

# 查看命令但不执行（dry-run）
python scripts/run_experiments.py --dry-run
```

---

## 4. experiments.yaml 配置详解

### 4.1 基本结构

```yaml
experiments:
  - name: 实验名称
    params:
      task: snli
      model: baseline
      epochs: 5
      batch_size: 128
      save_dir: ./result/1_2_0
    flags: [sortish_batches]
    sweep:
      energy_reg_weight: [0.0, 0.1, 0.3]
```

### 4.2 已配置的标准实验

#### 1) snli_baseline
```yaml
name: snli_baseline
params:
  task: snli
  model: baseline
  epochs: 5
  batch_size: 128
  lr: 3e-4
  seed: 42
  save_dir: ./result/1_2_0
  workers: -1
flags: [sortish_batches]
```

#### 2) snli_hybrid
```yaml
name: snli_hybrid
params:
  task: snli
  model: hybrid
  epochs: 5
  batch_size: 128
  lr: 3e-4
  seed: 42
  friction.K: 3
  friction.radius: 2
  friction.neighbor: window
  energy_reg_weight: 0.1
  energy_reg_scope: last
  energy_reg_target: rank
  energy_rank_margin: 0.5
  energy_rank_topk: 1
  save_dir: ./result/1_2_0
  workers: -1
flags: [sortish_batches]
```

#### 3) sst2_baseline
```yaml
name: sst2_baseline
params:
  task: sst2
  model: baseline
  epochs: 3
  batch_size: 256
  lr: 3e-4
  seed: 42
  save_dir: ./result/1_2_0
  workers: -1
flags: [sortish_batches]
```

#### 4) sst2_hybrid
```yaml
name: sst2_hybrid
params:
  task: sst2
  model: hybrid
  epochs: 3
  batch_size: 256
  lr: 3e-4
  seed: 42
  friction.K: 3
  friction.radius: 2
  friction.neighbor: window
  energy_reg_weight: 0.1
  energy_reg_scope: last
  energy_reg_target: rank
  energy_rank_margin: 0.5
  energy_rank_topk: 1
  save_dir: ./result/1_2_0
  workers: -1
flags: [sortish_batches]
```

---

## 5. 输出与产物

### 5.1 标准目录结构

```
result/
└── 1_2_0/
    ├── snli_baseline_<timestamp>_seed42/
    ├── snli_hybrid_<timestamp>_seed42/
    ├── sst2_baseline_<timestamp>_seed42/
    └── sst2_hybrid_<timestamp>_seed42/
```

### 5.2 关键产物文件

| 文件 | 内容 | 用途 |
|---|---|---|
| `config.json` | 完整配置参数 | 复现实验 |
| `env.txt` | 环境信息（PyTorch版本、设备） | 调试 |
| `metrics_epoch.csv` | 训练/验证指标（每 epoch） | 训练曲线 |
| `energy_epoch.csv` | 能量统计（每 epoch） | 能量监控 |
| `test_summary.json` | 测试集最终结果 | 论文表格 |
| `energy_error_correlation.json` | 能量-错误相关性、AUROC、Coverage | 论文图表 |
| `confusion_matrix.csv` | 混淆矩阵 | 错误分析 |
| `timing.json` | 训练时间统计 | 性能分析 |
| `alerts.json` | 能量告警记录（如有） | 调试 |

---

## 6. 常见使用场景

### 场景 1：快速验证代码修改

```bash
./quick.sh sst2_baseline
```

### 场景 2：复现最佳结果（v1.0.4）

```yaml
# best_v104.yaml
experiments:
  - name: v104_best
    params:
      task: sst2_noisy
      model: hybrid
      noise_intensity: low
      epochs: 3
      batch_size: 512
      friction.K: 1
      energy_reg_weight: 1e-4
      energy_reg_scope: last
      energy_reg_target: normalized
      save_dir: ./result/reproduce_v104
    flags: [sortish_batches, friction.no_recompute_mu]
```

```bash
python scripts/run_experiments.py --config best_v104.yaml
```

### 场景 3：参数扫描

```yaml
experiments:
  - name: snli_lambda_sweep
    sweep:
      energy_reg_weight: [0.0, 0.05, 0.1, 0.3]
    params:
      task: snli
      model: hybrid
      epochs: 5
      friction.K: 2
      energy_reg_target: rank
      save_dir: ./result/lambda_sweep
```

### 场景 4：运行归档实验（参考用）

```bash
python scripts/run_experiments.py \
  --config scripts/experiments_archive.yaml \
  --select sst2_noisy_baseline
```

---

## 7. 故障排查

### 问题 1：找不到 YAML 文件

```bash
FileNotFoundError: [Errno 2] No such file or directory: 'scripts/experiments.yaml'
```

**解决**：确保在项目根目录运行

```bash
cd /path/to/FIF-A
./run.sh
```

### 问题 2：内存不足

```bash
CUDA out of memory. Tried to allocate 2.00 GiB
```

**解决**：减小 `batch_size`

```yaml
params:
  batch_size: 128
```

### 问题 3：实验跳过

```bash
python scripts/run_experiments.py --select my_exp
# 没有任何输出
```

**原因**：实验名不匹配

**解决**：检查 `experiments.yaml` 中的 `name` 字段

---

## 8. 最佳实践

1. 使用版本化 `save_dir`（例如 `./result/1_2_0`）
2. 大规模扫描先 `--dry-run` 校验
3. 为论文指标保留 `test_summary.json` 与 `energy_error_correlation.json`
4. 重要实验完成后同步 `PROJECT_TRACKER.md` 与 `PHASE_RESULTS.md`

---

## 9. 快速参考

```bash
# 运行全部
./run.sh

# 快速选择
./quick.sh snli
./quick.sh sst2
./quick.sh baseline
./quick.sh hybrid

# 单个实验
./quick.sh snli_baseline
./quick.sh snli_hybrid
./quick.sh sst2_baseline
./quick.sh sst2_hybrid

# 预览命令
./quick.sh <target> --dry-run
```
