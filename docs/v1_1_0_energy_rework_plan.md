# v1.1.0 能量正则重构与实验规划

**文档信息**
- 版本：v1.1.0
- 创建日期：2024-12-03
- 状态：Implemented
- 相关文档：`experiment_design.md`, `PROJECT_TRACKER.md`

---

## 概述

本文档针对 normalized 方差问题，详细说明 v1.1.0 的能量正则重构方案与实验规划。

---

## 1. 背景：当前机制与默认值
- 能量定义：`E = 0.5 Σ_{(i,j)} μ_ij ||h_i - h_j||^2`；`μ` 由 edge MLP 动态生成，`L_μ` 采用度归一化拉普拉斯（`fif_mvp/models/friction_layer.py`）。
- 迭代：`state = state - η (L_μ state - q)`，K 步 + η 衰减 + 1D 平滑；`q_proj` 为可训练牵引项。
- 正则：`_compute_energy_regularizer`（`fif_mvp/train/loop.py`）现支持 `target ∈ {absolute, normalized, margin, rank}`，默认 `--energy_reg_target absolute --energy_reg_scope last`；`--energy_reg_mode` 仅作为兼容别名。
- 监控：`energy_guard` 仅在 `energy_std < threshold` 时下调 λ，本质上强化“能量集中”。`energy_watch` 以 std/p90 做告警（写入 `alerts.json`）。

## 2. 问题诊断：为何 normalized 与"能量=置信度"冲突

### 2.1 核心问题

1. **目标缺失**
   - 损失从未显式要求"错样本能量更高"
   - CE 只关注 logits，能量正则只关注分布形状

2. **方差惩罚的反向作用**
   - `(log1p(E) - mean)^2` 驱动 batch 内能量收敛到同一值
   - 直接抹平易错样本与易分样本的能量差异

3. **实验证据**
   - v1.0.4 报告中 SNLI Hybrid `pearson_r ≈ 0.04~0.09`
   - SST-2 的相关性也弱
   - 能量绝对量级在 guard/watch 的双阈值下被压小或对齐

4. **结构放大效应**
   - 动态 μ + K>1 + 强 q_proj 让表征有足够自由度
   - 在不影响 CE 的前提下压缩能量分布
   - normalized 正则进一步鼓励这种"平滑但无判别力"的解

## 3. 改进方案

### 3.1 P0 优先级：重定能量正则目标
  - 将默认改为 `absolute`（仅控制尺度，不消除差异），并补充一项显式将能量与分类难度绑定的损失（候选：与 logit margin 的单调约束或错误样本排名损失）。
  - 备选：在 normalized 基础上拆分“尺度 + 离散度”，即保留 `mean(log1p(E))` 约束，方差项仅占极小权重（避免完全抹平）。
  - 提供新开关（如 `--energy_reg_target {none,absolute,margin,rank}`）以便逐步 A/B

### 3.2 P0 优先级：指标与监控升级
  - 增加能量对错误的 AUROC/AUPRC、Coverage-Risk 曲线，替代单一 Pearson；在 `energy_error_correlation.json` 里写入新指标。
  - 训练期记录分组统计（正确 vs 错误的能量分布、p50/p90），避免仅凭 batch 方差判断。
  - guard/watch 逻辑改为"区间"监控（下界防塌、上界防爆），避免单向压缩

### 3.3 P1 优先级：结构与调度
  - SNLI 默认降 K→1/2，或仅前层插入摩擦，减弱过度低通；评估 `eta_decay=0/0.7` 与 `mu_max` 上限对能量差异的影响。
  - 试验 cross-sentence 边或注意力引导的 kNN 作为 SNLI 对齐感知的轻量改动（小规模验证）

### 3.4 P1 优先级：数据/脚本侧实验对照
  - 现有 v1.0.4 分支先做最小改动对照：`mode ∈ {normalized, absolute}` × `λ ∈ {0,1e-5,1e-4}` × `K ∈ {1,3}`，确认方差惩罚的影响幅度。
  - 若 margin/排名正则实现完毕，再在 SNLI + Noisy SST-2 各跑至少一组 sanity（小步数）看相关性方向。

## 4. 建议的实验矩阵

### 4.1 实验轴

- **轴 1**：正则目标 `{none, absolute(mean), normalized, margin, rank}`
  - 其中 margin/rank 需实现新项
- **轴 2**：λ `{0, 1e-5, 1e-4}`
- **轴 3**：K `{1,2}`
- **轴 4**：η 衰减 `{0,0.5}`

### 4.2 任务覆盖

- **SNLI**：核心关注相关性与吞吐
- **SST-2 noisy**：验证不会损伤已存在的收益

### 4.3 评估指标

- acc/macro_f1/ECE
- `energy_log_mean_test`
- AUROC/AUPRC/coverage-risk
- 能量分布分位（p50/p90/p99）分开统计正确/错误

## 5. 落地步骤与交付

### 5.1 代码修改

- 在 `loop.py::_compute_energy_regularizer` 增加 margin/排名模式
- `run_experiment.py` 暴露新目标开关并调整默认为 `absolute`
- `metrics_epoch.csv`/`energy_error_correlation.json` 扩充新指标字段

### 5.2 脚本更新

- 为 SNLI/SST-2 各加一份"absolute + λ=1e-4 + K=1"快速对照脚本
- 便于回归测试

### 5.3 文档更新

- 更新 `PROJECT_TRACKER.md`（1.1.0 目标）
- 更新 `WORK_BOARD.md` 新任务条目
- 报告模板追加能量 AUROC/coverage-risk

## 6. 风险与资源评估

### 6.1 训练时间

- 新指标（AUROC/coverage）需要额外一次前向/统计
- 可在测试期计算，开销可控

### 6.2 模式交互风险

- 若 margin/排名损失过强，可能与 CE 竞争
- 导致过拟合或能量爆炸
- 需要 λ 网格与 guard 上下界双向控制

### 6.3 结构改动影响

- 跨句边/kNN 可能影响吞吐
- 需在小规模子集验证后再大规模运行

## 7. 实施状态

### 7.1 已完成功能

- **能量正则目标**
  - 默认 `energy_reg_target=absolute`
  - 保留 margin/rank 作为可选 A/B
  - `energy_reg_mode` 兼容旧版

- **新增指标**
  - 测试期输出：`energy_auroc/energy_auprc`
  - coverage-risk（`coverage_aurc`、`coverage_risk_at_{80,90,95}`）
  - 正确/错误分位写入 `test_summary.json` 与 `energy_error_correlation.json`
  - 包含 coverage 曲线子采样

- **监控机制**
  - guard/watch 支持能量上下界
  - `std_low/std_high/p90_low/p90_high/mean_low/mean_high`
  - guard 允许 λ 上下调并在 `alerts.json` 记录原因

- **脚本工具**
  - 新增 `scripts/snli_hybrid_k1_absolute.sh`
  - 新增 `scripts/sst2_noisy_hybrid_k1_absolute.sh`
  - K=1、absolute λ=1e-4，保存至 `result/1_1_0`
  - `scripts/summarize_results.py` 默认聚合该目录并带上新指标
