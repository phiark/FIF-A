# FIF 实验设计文档

**文档元数据**
- 版本：v1.2.0
- 最后更新：2025-12-19
- 状态：Active
- 相关文档：`PROJECT_TRACKER.md`, `PHASE_RESULTS.md`, `EXPERIMENT_LAUNCHER_GUIDE.md`

---

## 概述

本文档定义 FIF 项目的实验设计规范，确保实验与论文方法/结果章节对齐，覆盖数据管线、模型与能量机制、实验矩阵、指标口径与复现约束。

---

## 1. 研究目标（论文导向）

1. 验证引入 FIF 是否提升轻量 Transformer 在噪声数据上的鲁棒性。
2. 评估能量信号作为置信度代理的可证伪性（能量-错误单调性）。
3. 比较不同能量正则目标（rank/margin/absolute）的效果与稳定性。
4. 保证正则使用的能量与评估/告警刻度一致（末层 vs 跨层可选）。

---

## 2. 硬件基线与复现约束

- **硬件基线**：Tesla V100 16G ×1（v1.2.0+ 固定）
- **支持后端**：CUDA 单卡
- **不再维护**：MPS/多卡 DDP 路径
- **确定性**：默认开启 deterministic；使用 `--no_deterministic` 可提高吞吐

---

## 3. 数据与管线

- **数据源**：GLUE SST-2、SNLI。
- **训练噪声策略**：`clean + {low, med, high}` 全量复制，`noise_level` 作为条件嵌入输入模型。
- **验证/测试**：单一噪声强度（low/med/high）分别评估，保持与 v0.0.0 可比。
- **实现要点**：
  - `train_noise_levels` CLI 控制复制列表，默认 `clean,low,med,high`。
  - `SequenceCollator` 生成 `noise_level_ids`，模型在 token 嵌入后加对应噪声向量。

---

## 4. 模型与能量设计

### 4.1 Friction Layer

- 动态 μ：每次迭代重算 `μ = clamp(softplus(MLP))`。
- 归一化拉普拉斯：`D^{-1/2} L D^{-1/2}`，每步后执行 1D 平滑。
- 步长衰减：`η_t = η ⋅ decay^t`（默认 `decay=0.5`）。

### 4.2 能量正则与评估对齐

- 训练损失：`L = CE + λ ⋅ φ(E_batch)`。
- 默认目标：`energy_reg_target=rank`，对 batch z-score 能量执行排序/间隔约束。
- 对齐策略：
  - `energy_reg_scope {all,last}` 选择能量范围；
  - `energy_eval_scope {auto,per_sample}` 使评估刻度与正则一致。
- 备用目标：`absolute`（惩罚 `log1p(E)`）、`normalized`（惩罚 batch 内方差）、`margin`（间隔约束）。

---

## 5. v1.2.0 实验矩阵（当前）

| 版本 | 模型 | 数据 | 能量目标 | λ | 备注 |
|---|---|---|---|---|---|
| v1.2.0-A | Baseline | SNLI | - | 0.0 | 基线修复 |
| v1.2.0-B | Hybrid | SNLI | rank | {0.0, 0.1, 0.3} | margin=0.5 |
| v1.2.0-C | Baseline | SST-2 | - | 0.0 | 干净集验证 |
| v1.2.0-D | Hybrid | SST-2 | rank | {0.0, 0.1, 0.3} | margin=0.5 |

---

## 6. 指标与产物

- **指标**：`acc`, `macro_f1`, `loss`, `ece`, `energy_mean_test`, `energy_log_mean_test`，以及能量‑错误指标（`pearson_r`, `energy_auroc`, `energy_auprc`, `coverage_aurc`, `coverage_risk_at_{80,90,95}`）。
- **产物**：`metrics_epoch.csv`, `energy_epoch.csv`, `test_summary.json`, `energy_error_correlation.json`, `confusion_matrix.csv`。

---

## 7. 验证步骤

1. 使用 `run.sh` 或 `quick.sh` 运行标准实验。
2. 检查 `result/1_2_0/` 输出完整性与指标口径。
3. 在 `PROJECT_TRACKER.md` 与 `PHASE_RESULTS.md` 同步版本记录与论文素材。

---

## 8. 历史版本参考（摘要）

- **v1.0.3**：引入 energy guard/watch、normalized 能量目标；详见 `docs/reports/v1_0_3_results.md`。
- **v1.1.0**：absolute 方案失败冻结；详见 `PROJECT_TRACKER.md` 与 `docs/v1_1_0_energy_rework_plan.md`。
