# FIF 实验设计文档

**文档信息**
- 版本：v1.1.0
- 最后更新：2024-12-03
- 状态：Active
- 相关文档：`PROJECT_TRACKER.md`, `v1_1_0_energy_rework_plan.md`

---

## 概述

本文档记录 FIF 项目的实验设计规范，包括各版本的目标、数据管线、模型配置、实验矩阵和验证步骤。

---

## 1. 实验目标
1. 验证在含噪训练 + 噪声条件化下，FIF 混合架构能否超越 Transformer 基准线的鲁棒性。
2. 评估改进版摩擦层（动态 μ、度归一化、η 衰减）对能量稳定性的影响。
3. 通过能量正则与对数能量指标，建立“能量≈错误概率”的可量化证据。

## 2. 数据与管线
- **数据源**：GLUE SST-2。
- **训练噪声策略**：`clean + {low, med, high}` 全量复制，`noise_level` 作为条件嵌入输入模型。
- **验证/测试**：单一噪声强度（low/med/high）分别评估，保持与 v0.0.0 可比。
- **实现要点**：
  - `train_noise_levels` CLI 控制复制列表，默认 `clean,low,med,high`。
  - `SequenceCollator` 生成 `noise_level_ids`，模型在 token 嵌入后加对应噪声向量。

## 3. 模型与公式
- **Friction Layer**：
  - 每次迭代重算 μ，并做 `μ = clamp(softplus(MLP))`。
  - 采用归一化拉普拉斯 `D^{-1/2} L D^{-1/2}`，并在每步后执行 1D 平滑。
  - 步长 `η_t = η ⋅ decay^t`，默认 `decay=0.5`。
- **能量正则**：训练损失 `L = CE + λ ⋅ \phi(E_batch)`，其中 `φ` 可选：
  - `--energy_reg_scope {all,last}` 决定取全部摩擦层能量（默认）或仅取最后一层能量；
  - `--energy_reg_target {absolute,normalized,margin,rank}` 控制正则目标：`absolute` 直接惩罚 `log1p(E)`（默认），`normalized` 惩罚 batch 内方差，`margin`/`rank` 让能量与分类难度对齐；`--energy_reg_mode` 为兼容旧版的别名；
  - 默认 `--energy_reg_scope last`、`--energy_reg_target absolute`，可选 normalized/margin/rank 作为对照；λ 依旧通过 CLI 控制。
- **能量指标**：记录 `energy_mean/log_mean/std/p90`，写入 `metrics_epoch.csv`、`energy_epoch.csv` 以及 `test_summary.json`（均值/对数均值）。

## 4. 实验矩阵

| 版本 | 模型 | 训练噪声 | 评估噪声 | 能量权重 λ | 备注 |
|---|---|---|---|---|---|
| v1.0.0-A | Baseline | clean+low+med+high | low/med/high | 0.0 | 控制组 |
| v1.0.0-B | Hybrid | clean+low+med+high | low/med/high | 1e-4 | 主实验 |
| v1.0.0-C | Hybrid | clean+low+med+high | low/med/high | 5e-4 | 能量正则敏感性 |

## 5. 指标与产物
- 指标：`acc`, `macro_f1`, `loss`, `ece`, `energy_mean_test`, `energy_log_mean_test`，能量‑错误相关指标（`pearson_r`、`energy_auroc`、`energy_auprc`、`coverage_aurc`、`coverage_risk_at_{80,90,95}`），以及正确/错误样本的 `energy_p50/p90/p99`。
- 产物：`metrics_epoch.csv`, `energy_epoch.csv`（含 log）、`confusion_matrix.csv`, `energy_error_correlation.json`（含 coverage‑risk 曲线与分位统计）、per-sample 能量（可选导出）。

## 6. 验证步骤

0. **DDP 配置**
   - 多 GPU 节点统一通过 CLI `--ddp --nproc_per_node=<gpu_count>` 启用单机 DDP
   - 或脚本自动侦测 GPU 数量
   - 禁止 DataParallel 造成的标量 gather 同步瓶颈
1. 使用 `scripts/sst2_noisy_baseline.sh` & `sst2_noisy_hybrid.sh` 重新跑 v1.0.0。
2. 核对 `PROJECT_TRACKER.md` 记录的版本字段已更新。
3. 在论文草稿中记录新的能量/准确率对照图（计划使用 `docs/figures/` 输出）。 

## 7. v1.0.3 扩展实验

### 7.1 配置新增
  - 默认 `--energy_reg_scope last`、`--energy_reg_target normalized`（旧脚本可用 `--energy_reg_mode` 等价），λ 扫描 `{1e-5, 5e-5}` 并在训练中基于 `energy_std` 阈值（0.1）动态降权。
  - `--friction.recompute_mu` 默认开启，并提供 `--friction.k_warmup_epochs`（SNLI=2、SST-2=1）先用 K=1 预热。
  - `--friction.knn_mode` 提供 `per_sample` 向量化路径且允许 `--friction.graph_cache_size` 以摊薄构图成本。
  - `--energy_watch std=0.1,p90=0.5` 触发实时告警，结果写入 `alerts.json`

### 7.2 实验矩阵

| 版本 | 模型 | 数据 | 正则设置 | 预热 | 目标 |
| --- | --- | --- | --- | --- | --- |
| v1.0.3-A | Hybrid | SNLI | scope=last, mode=normalized, λ=1e-5 | 2 epoch K=1 → K=3 | `acc ≥ 0.74`, `pearson_r ≥ 0.1`, walltime <3k s |
| v1.0.3-B | Hybrid | SNLI | scope=last, mode=normalized, λ=5e-5 | 同上 | 观察 λ 敏感性，至少一项指标优于 v1.0.2 |
| v1.0.3-C | Hybrid | SST-2 noisy low/med/high | 同上，但低噪声 λ=1e-5、med/high λ=5e-5 | 1 epoch 预热 | `Δacc ≥ 0`、`ece` 不劣于 baseline |

### 7.3 输出交付

完成后撰写 `docs/reports/v1_0_3_results.md` 并在 `PROJECT_TRACKER.md` / `WORK_BOARD.md` 标记任务状态。

---

## 8. v1.1.0 能量重构要点

### 8.1 核心目标

- 以 `absolute` 为默认能量正则，并引入 margin/rank 作为可选 A/B
- 能量监控改为上下界带告警
### 8.2 新增指标

测试阶段输出：
- `energy_auroc` / `energy_auprc`
- `coverage_aurc`
- `coverage_risk_at_{80,90,95}`
- 在 `energy_error_correlation.json` 记录 coverage-risk 子采样曲线
- 记录正确/错误样本的能量分位数

### 8.3 监控机制

- **energy_guard**
  - 支持 `std_low/std_high/p90_low/p90_high`
  - λ 的上下调（`factor` 与 `up`）

- **energy_watch**
  - 支持 mean/std/p90 上下界
  - 告警写入 `alerts.json`

### 8.4 脚本更新

- 新增 `scripts/snli_hybrid_k1_absolute.sh`
- 新增 `scripts/sst2_noisy_hybrid_k1_absolute.sh`
- 默认保存至 `result/1_1_0`
- 用于 K=1、`target=absolute` 的快速验证

### 8.5 结果汇总

- `scripts/summarize_results.py` 默认扫描 `result/1_1_0`
- 可传根目录参数以聚合旧版本结果
