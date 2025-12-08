# FIF 工作板

**文档元数据**
- 文档类型：任务追踪与项目管理
- 当前活跃版本：v1.2.0
- 最后更新：2024-12-08
- 总任务数：39
- 已完成：30 (77%)
- 相关文档：`PROJECT_TRACKER.md`, `PHASE_RESULTS.md`

**快速导航**
- [当前冲刺状态](#当前冲刺-v120)
- [优先级说明](#任务板标准)
- [活跃任务](#v120-活跃任务)
- [历史任务](#历史版本任务)

---

## 当前冲刺 (v1.2.0)

**冲刺信息**
- 版本：v1.2.0
- 开始日期：2024-12-04
- 目标完成：2024-12-15
- 进度：40% (4/10)
- 状态：🟡 有风险（T-031 阻塞）

**里程碑进度**
- [ ] M1：SNLI 基线恢复到 >0.70 acc (截止 12-10) - 🟡 40%
  - 🔄 T-031: SNLI 基线排查 (In Progress)
  - 📋 T-032: 实现排名损失 (Backlog)

- [ ] M2：干净数据集验证 (截止 12-12) - 🔴 0%
  - 📋 T-033: 小网格实验 (Backlog)

- [ ] M3：Noisy 数据集扩展 (截止 12-15) - 🔴 0%
  - 📋 T-034: Noisy SST-2 (Backlog)

**优先级任务**
- 🔴 P0：T-031 (阻塞中)
- 🟡 P1：T-032, T-035
- 🟢 P2：T-036, T-037
- ⚪ P3：T-038, T-039

---

## 概述

本工作板旨在跟踪面向论文级实验的版本演进，可随项目增长扩展任务列表。

## 下一个版本（1.2.0）概述
- **版本号**：1.2.0（基于 1.1.0 诊断失败后的重构）
- **为什么需要修改 1.1.0**：
  - absolute 能量正则跨任务行为不一致：SST-2 过度压缩能量且掉点，SNLI 能量爆炸且准确率崩溃。
  - 目标应聚焦排序关系而非绝对刻度；跨任务刻度统一是伪需求。
  - SNLI 基线本身过低（~54% acc），需要先修复再谈能量判别力。
- **版本目标**：
  1. 采用 batch 归一化能量，并用 margin 排名损失直接优化“正确 < 错误”排序；
  2. 修复 SNLI 基线（标签/长度/训练配置），建立可靠对照；
  3. 在干净 SST-2/SNLI 上做小网格 λ∈{0,0.1,0.3}、m=0.5，确认 AUROC/校准收益后再扩展 noisy SST-2。

## 任务板标准
- **列定义**：`Backlog`（未启动）、`In Progress`、`Blocked`、`Review`、`Done`。
- **任务字段**：`ID`（唯一编号）、`描述`、`负责人`、`状态`、`目标版本`、`关联输出`（代码路径或结果目录）。
- **版本绑定方式**：每个任务需显式写明 `目标版本`；若跨版本，拆分任务或在备注中标记后续滚动。
- **更新节奏**：每次实验或合入 PR 后更新状态；若状态 >7 天未更新需标记为 `Blocked` 并说明阻塞原因。

## v1.0.0 活动任务
| ID | 描述 | 负责人 | 状态 | 目标版本 | 关联输出/备注 |
| --- | --- | --- | --- | --- | --- |
| T-001 | 在训练集注入 low/med/high 噪声并将 `noise_level` 作为条件输入（Embedding 或 Adapter），验证对鲁棒性的影响 | Codex | Done | 1.0.0 | `fif_mvp/data/sst2.py`, `fif_mvp/data/__init__.py`, 模型噪声嵌入 |
| T-002 | 重新推导/实现 FrictionLayer：逐迭代更新 µ，加入度归一化与 η 调度，避免能量爆炸 | Codex | Done | 1.0.0 | `fif_mvp/models/friction_layer.py` |
| T-003 | 在训练损失中加入能量正则（例如高能量样本惩罚或校准损失），并记录能量-错误曲线 | Codex | Done | 1.0.0 | `fif_mvp/train/loop.py`, `scripts/sst2_noisy_*.sh` |
| T-004 | 统一能量量纲并引入对照指标（z-score 或对数能量），便于与基准线比较 | Codex | Done | 1.0.0 | `metrics_epoch.csv`/`energy_epoch.csv` 字段、`test_summary.json` |
| T-005 | 编写实验设计文档 & 论文草稿提纲，明确需要的图表/表格 | Codex | Done | 1.0.0 | `docs/experiment_design.md` |
| T-006 | 修复多 GPU DataParallel 警告并默认使用单机 DDP（CLI `--ddp` + 自动脚本） | Codex | Done | 1.0.1 | `fif_mvp/cli/run_experiment.py`, `scripts/*.sh`, `README.md` |
| T-007 | SNLI Hybrid v1.0.0-B：启用 `λ=1e-4` 能量正则并将摩擦迭代深度对齐 (K=3)，记录能量轨迹 | Codex | In Progress | 1.0.0 | `scripts/snli_hybrid.sh`, `docs/reports/v1_0_0_snli.md`, `fif_mvp/models/friction_layer.py`（已完成批量化与能量回落优化，等待新一轮跑数） |
| T-008 | v1.0.1 SNLI 实验分析：验证 DDP/能量聚合改动并出具报告 | Codex | Done | 1.0.1 | `docs/reports/v1_0_1_snli.md`, `PROJECT_TRACKER.md` |
| T-009 | 能量正则可调：支持 scope/mode 选择并输出 `energy_std/p90` | Codex | Done | 1.0.1 | `fif_mvp/cli/run_experiment.py`, `fif_mvp/train/loop.py`, `fif_mvp/models/hybrid_model.py`, `README.md`, `docs/experiment_design.md` |

## v1.0.2 活动任务
| ID | 描述 | 负责人 | 状态 | 目标版本 | 关联输出/备注 |
| --- | --- | --- | --- | --- | --- |
| T-010 | 统一模型输出为 `ModelOutput` 并精简训练端判型 | Codex | Done | 1.0.2 | `fif_mvp/models/*`, `fif_mvp/train/loop.py` |
| T-011 | 修复 DDP 下 GPU→CPU 回退策略（禁用自动回退） | Codex | Done | 1.0.2 | `fif_mvp/cli/run_experiment.py` |
| T-012 | Baseline 窗口能量向量化，消除逐样本循环 | Codex | Done | 1.0.2 | `fif_mvp/train/energy.py` |
| T-013 | CLI 暴露完整 Friction 超参开关 | Codex | Done | 1.0.2 | `fif_mvp/cli/run_experiment.py`, `README.md` |
| T-014 | 确定性默认开启，提供 `--no_deterministic` 关闭 | Codex | Done | 1.0.2 | `fif_mvp/cli/run_experiment.py`, `README.md` |
| T-015 | 小清理：重复 import/噪声字符集/去除未用依赖 | Codex | Done | 1.0.2 | `fif_mvp/data/__init__.py`, `fif_mvp/data/noise.py`, `requirements.txt` |
| T-016 | 训练集长度 Sortish 采样（非 DDP 可选） | Codex | Done | 1.0.2 | `fif_mvp/data/*`, `README.md` |

> 注：负责人暂未指派；当任务进入 `In Progress` 时需补充姓名与预计完成时间。

## v1.0.3 活动任务
| ID | 描述 | 负责人 | 状态 | 目标版本 | 关联输出/备注 |
| --- | --- | --- | --- | --- | --- |
| T-017 | 能量正则重构：默认 `scope=last, mode=normalized`、λ 动态降权 + λ 扫描脚本 | Codex | In Progress | 1.0.3 | `fif_mvp/cli/run_experiment.py`, `fif_mvp/train/loop.py`, `README.md`, `scripts/snli_hybrid.sh` |
| T-018 | Friction 迭代升级：`recompute_mu=True`、K 预热与阶段化 η/μ，更新 Hybrid 训练脚本 | Codex | Backlog | 1.0.3 | `fif_mvp/models/friction_layer.py`, `fif_mvp/models/hybrid_model.py`, `scripts/snli_hybrid.sh` |
| T-019 | kNN/监控系统：per-sample 向量化、graph cache、`energy_alert`/`alerts.json` 告警 | Codex | Backlog | 1.0.3 | `fif_mvp/models/friction_layer.py`, `fif_mvp/train/loop.py`, `README.md` |
| T-024 | SNLI/SST-2 训练脚本吞吐优化：开启 `--sortish_batches` 并新增 fixed-μ 变体 | Codex | Done | 1.0.3 | `scripts/snli_*`, `scripts/sst2_noisy_*` |

## v1.0.4 活动任务
| ID | 描述 | 负责人 | 状态 | 目标版本 | 关联输出/备注 |
| --- | --- | --- | --- | --- | --- |
| T-020 | 代码结构规划：梳理目录职责、列出清理路线并文档化 | Codex | Done | 1.0.4 | `docs/code_structure.md`（新增） |
| T-021 | 批量 kNN 构图向量化，清理 FrictionLayer 遗留辅助函数，记录在版本追踪 | Codex | Done | 1.0.4 | `fif_mvp/utils/sparse.py`, `fif_mvp/models/friction_layer.py`, `PROJECT_TRACKER.md` |
| T-022 | 运行级回归：基于新构图跑 SNLI/SST-2，并将实验指标写入 `docs/reports/` | Codex | In Progress | 1.0.4 | `docs/reports/v1_0_4_experiment_report.md` 已建立模板，脚本输出指向 `result/1_0_4/`，待回填结果 |
| T-025 | 结果汇总与验真工具：扫描 `result/1_0_4` 聚合 `test_summary/timing` | Codex | Done | 1.0.4 | `scripts/summarize_results.py` |
| T-023 | Legacy / CI 清理：迁移 `fif_simple/` 至 archive、补齐静态检查（ruff/pytest smoke） | Codex | Backlog | 1.0.4 | 需新增 `tox.ini` + GitHub Actions（规划中） |

## v1.1.0 活动任务
| ID | 描述 | 负责人 | 状态 | 目标版本 | 关联输出/备注 |
| --- | --- | --- | --- | --- | --- |
| T-026 | 能量正则目标重构：默认改为 absolute，新增 margin/排名目标与 CLI 开关，重写 `_compute_energy_regularizer` | Codex | Done | 1.1.0 | `fif_mvp/train/loop.py`, `fif_mvp/cli/run_experiment.py`, `fif_mvp/config.py` |
| T-027 | 能量-错误指标扩展：AUROC/AUPRC、coverage-risk、正确/错误分组分位写入 `energy_error_correlation.json` | Codex | Done | 1.1.0 | `fif_mvp/train/loop.py`, `fif_mvp/train/metrics.py`, `scripts/summarize_results.py` |
| T-028 | 监控/guard 升级：支持能量上下界阈值与告警，避免单向压缩 | Codex | Done | 1.1.0 | `fif_mvp/train/loop.py`, `fif_mvp/config.py`, `fif_mvp/cli/run_experiment.py`, `README.md` |
| T-029 | 结构/超参对照：K∈{1,2} 与 η 衰减扫描，新增 SNLI/SST-2 “absolute+λ=1e-4+K=1” 快速脚本 | Codex | Done | 1.1.0 | `scripts/snli_hybrid_k1_absolute.sh`, `scripts/sst2_noisy_hybrid_k1_absolute.sh`（保存至 `result/1_1_0`） |
| T-030 | 文档/报告同步：更新版本追踪与报告模板，记录新指标与实验矩阵 | Codex | Done | 1.1.0 | `PROJECT_TRACKER.md`, `docs/experiment_design.md`, `docs/v1_1_0_energy_rework_plan.md`, `docs/reports/v1_0_4_experiment_report.md` |
> 结论：absolute 能量正则被判定为诊断失败，1.1.0 冻结，不再调参。

## v1.2.0 活动任务
| ID | 描述 | 负责人 | 状态 | 目标版本 | 关联输出/备注 |
| --- | --- | --- | --- | --- | --- |
| T-031 | SNLI 基线排查：标签映射、tokenization/max_len、训练 epoch/LR，确认无噪声注入 | Codex | In Progress | 1.2.0 | 提升脚本为 5 epoch、bs=256，save_dir `result/1_2_0`，待复现 acc |
| T-032 | 实现 batch 归一化能量与 margin 排名损失（架构对齐 1.0.x），CLI 暴露 margin/topk/λ | Codex | In Progress | 1.2.0 | `fif_mvp/train/loop.py`, `fif_mvp/cli/run_experiment.py`, `fif_mvp/config.py`（默认 target=rank + 归一化能量，待跑数） |
| T-033 | 干净 SST-2/SNLI 小网格：λ∈{0,0.1,0.3}、margin=0.5、seed=42；评估 acc/F1/ECE/AUROC/AURC | Codex | Backlog | 1.2.0 | 仅在基线稳定后执行，结果写入 `result/1_2_0` |
| T-034 | Noisy SST-2 follow-up：在低/中/高噪声下复测排序能量对 selective risk 的作用 | Codex | Backlog | 1.2.0 | 依赖 T-033 成果，脚本沿用 1.2.0 配置 |
| T-035 | 评估刻度对齐：推理导出 z-score 能量/分位，并让 AUROC/coverage 使用与正则一致的能量源（末层 vs 跨层可选） | Codex | In Progress | 1.2.0 | 规划改动 `fif_mvp/train/loop.py`, `fif_mvp/models/hybrid_model.py`, `fif_mvp/config.py` |
| T-036 | 能量归一化与噪声条件化：能量除以有效边/长度；将 `noise_level_ids` 或 logits margin 注入 μ/η 以减轻域间漂移 | Codex | Backlog | 1.2.0 | 需更新 `fif_mvp/models/friction_layer.py` 及噪声嵌入路径 |
| T-037 | Guard/fallback 稳定化：在 rank/margin 下禁用 std_low 下压，增加全对/全错 batch 的 fallback 正则（entropy/absolute） | Codex | Backlog | 1.2.0 | `fif_mvp/train/loop.py`, `fif_mvp/config.py` |
| T-038 | 受控合成基准：移植 `fif_simple` 冲突数据集为 CI/单元实验，验证能量-错误单调性 | Codex | Backlog | 1.2.0 | 新增 `tests/` 或 `scripts/` 中的小型 runner，结果进入 `docs/diagnostics/` |
| T-039 | 文档与报告：同步 1.2.0 设计/矩阵/报告模板，更新 README/PROJECT_TRACKER | Codex | Backlog | 1.2.0 | `docs/experiment_design.md`, `PROJECT_TRACKER.md`, `README.md`, `docs/reports/` |
