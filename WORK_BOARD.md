# FIF Work Board

本工作板旨在跟踪面向论文级实验的版本演进，可随项目增长扩展任务列表。

## 下一个版本（1.1.0）概述
- **版本号**：1.1.0（基于 v1.0.4 的能量正则重构）
- **为什么需要修改 1.0.4**：
  - 当前 `mode=normalized` 将 batch 内 `log1p(E)` 拉平，弱化“能量=置信度”，SNLI 相关性接近 0（见 `docs/v1_1_0_energy_rework_plan.md`）。
  - guard/watch 仅有下界，实质鼓励能量集中；监控指标缺少 AUROC/coverage-risk，无法评估拒判能力。
  - 结构/超参（K=3、强 η 衰减）可能过度低通，未有轻量脚本验证。
- **版本目标**：
  1. 重设能量正则默认为 `absolute`，新增 margin/排名目标与 CLI 开关，使能量与分类难度绑定。
  2. 扩充能量-错误指标（AUROC/AUPRC、coverage-risk、分组分位），guard/watch 改为上下界阈值。
  3. 提供低 K/低衰减的对照脚本（SNLI/SST-2）验证能量判别力与吞吐折中。

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
