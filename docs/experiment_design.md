# FIF v1.0.0 实验设计

## 1. 目标
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
  - `--energy_reg_mode {absolute,normalized}` 选择直接惩罚 `log1p(E)` 还是惩罚 Batch 内 `log1p(E)` 相对均值的平方，避免能量塌缩；
  - λ 依旧通过 CLI 控制。
- **能量指标**：记录 `energy_mean/log_mean/std/p90`，写入 `metrics_epoch.csv`、`energy_epoch.csv` 以及 `test_summary.json`（均值/对数均值）。

## 4. 实验矩阵
| 版本 | 模型 | 训练噪声 | 评估噪声 | 能量权重 λ | 备注 |
| --- | --- | --- | --- | --- | --- |
| v1.0.0-A | Baseline | clean+low+med+high | low/med/high | 0.0 | 控制组 |
| v1.0.0-B | Hybrid | clean+low+med+high | low/med/high | 1e-4 | 主实验 |
| v1.0.0-C | Hybrid | clean+low+med+high | low/med/high | 5e-4 | 能量正则敏感性 |

## 5. 指标与产物
- 指标：`acc`, `macro_f1`, `loss`, `ece`, `energy_mean_test`, `energy_log_mean_test`, `energy-error 皮尔逊相关`.
- 产物：`metrics_epoch.csv`, `energy_epoch.csv`（含 log）、`confusion_matrix.csv`, `energy_error_correlation.json`, per-sample 能量可选导出。

## 6. 验证步骤
0. 多 GPU 节点统一通过 CLI `--ddp --nproc_per_node=<gpu_count>`（或脚本自动侦测）启用单机 DDP，禁止 DataParallel 造成的标量 gather 同步瓶颈。
1. 使用 `scripts/sst2_noisy_baseline.sh` & `sst2_noisy_hybrid.sh` 重新跑 v1.0.0。
2. 核对 `PROJECT_TRACKER.md` 记录的版本字段已更新。
3. 在论文草稿中记录新的能量/准确率对照图（计划使用 `docs/figures/` 输出）。 
