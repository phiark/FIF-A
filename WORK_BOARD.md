# FIF Work Board

本工作板旨在跟踪面向论文级实验的版本演进，可随项目增长扩展任务列表。

## 下一个版本（1.0.0）概述
- **版本号**：1.0.0（首次对训练/评估管线及摩擦层做系统级调整）
- **为什么需要修改 0.0.0**：
  - 混合架构在全部噪声级别上均落后基准线，`energy_mean_test` 呈现 1–2 个数量级的漂移，能量与错误相关性不足 0.16。
  - 训练集缺少噪声样本，摩擦层 µ/η 未调优，导致能量在训练早期爆炸，违背“能量指示置信度”的设想。
  - 当前能量计算、日志及损失函数之间没有耦合，无法支撑论文主张。
- **我们为何要做 1.0.0**：验证“在含噪训练与受控能量正则下，混合架构是否具备鲁棒性收益”，为论文提供正向证据或明确失败原因。
- **版本目标**：
  1. 在训练阶段引入与评测一致的噪声，并可按 `noise_level` 条件化模型。
  2. 重构 Friction Layer 的 µ/迭代策略，抑制能量爆炸（归一化、可学习温度、梯度稳定）。
  3. 在损失中加入能量/不确定性正则化，使能量信号与错误概率产生可测联系。

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

> 注：负责人暂未指派；当任务进入 `In Progress` 时需补充姓名与预计完成时间。
