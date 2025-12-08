# FIF-A: Frictional Interaction Field for Robust NLP

**项目状态**
- 当前版本：v1.2.0 (🔄 进行中)
- 最新稳定版：v1.0.4 (2024-12-02)
- 论文状态：实验阶段
- 文档：[PROJECT_TRACKER](PROJECT_TRACKER.md) | [WORK_BOARD](WORK_BOARD.md) | [PHASE_RESULTS](PHASE_RESULTS.md) | [docs/](docs/)

---

## 概述

本项目研究在轻量级 Transformer 中引入**摩擦交互场 (Frictional Interaction Field, FIF)** 对噪声鲁棒性的影响，并探索能量信号作为置信度代理的可行性。

**核心创新**：
- **动态摩擦层**：通过迭代优化隐状态，引入可学习的摩擦系数 μ
- **能量正则化**：使用能量信号 (`E = 0.5 Σ μ_ij ||h_i - h_j||^2`) 监督模型置信度
- **噪声条件化**：训练时混合多强度噪声数据

**研究问题**：
1. FIF 层能否提升模型在噪声数据上的鲁棒性？
2. 能量信号是否能有效预测预测错误？
3. 最佳能量正则化策略是什么？

**当前最佳结果** (v1.0.4):
- SST-2 Low Noisy: Hybrid **0.808** acc (vs Baseline 0.782)
- ECE降低: 0.124 → **0.064**
- ⚠️ SNLI任务待改进 (Hybrid 0.69 vs Baseline 0.76)

---

## 实验历史

This experimental repo benchmarks whether introducing a Frictional Interaction Field (FIF, 信息摩擦层) into a lightweight Transformer improves robustness and whether the induced energy signal correlates with prediction errors. Since v1.0.0 we:

- inject `clean + low/med/high` noise directly into the SST-2 training split,
- condition encoders on `noise_level` embeddings,
- upgrade the friction layer with dynamic μ、度归一化及 η 衰减 + 平滑，
- add log-energy regularization/metrics to tighten “能量≈错误概率”的假设。

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Running experiments

### Quick Start (推荐)

```bash
# 运行所有实验（SNLI + SST-2, baseline + hybrid）
./run.sh

# 快速选择器 - 运行特定实验
./quick.sh snli          # SNLI baseline + hybrid
./quick.sh sst2          # SST-2 baseline + hybrid
./quick.sh baseline      # 两个 baseline 实验
./quick.sh hybrid        # 两个 hybrid 实验

# 运行单个实验
./quick.sh snli_baseline
./quick.sh snli_hybrid
./quick.sh sst2_baseline
./quick.sh sst2_hybrid

# 预览命令
./quick.sh snli --dry-run
```

查看完整指南：[QUICK_START.md](QUICK_START.md)

### 实验配置

所有实验配置在 `scripts/experiments.yaml`（4个标准实验）：
- `snli_baseline` - SNLI Transformer baseline
- `snli_hybrid` - SNLI with FIF (K=3, rank regularization)
- `sst2_baseline` - SST-2 Transformer baseline
- `sst2_hybrid` - SST-2 with FIF (K=3, rank regularization)

旧实验（noisy variants, v1.1.0 failures）存档在 `scripts/experiments_archive.yaml`。

结果保存在 `./result/` 目录（带时间戳）。The runner auto-detects the best accelerator in priority order (CUDA/NVIDIA or AMD ROCm, Apple MPS, then CPU) and only enables DDP when multiple CUDA devices are available.

### Multi-backend / Multi-GPU

- CUDA (NVIDIA/AMD ROCm): DDP is enabled automatically by the runner when `torch.cuda.device_count()>1`. TF32 tuning is skipped on AMD to avoid unsupported settings.
- Apple MPS: runs single-process; DDP is skipped.
- CPU: runs single-process.
- You can still call `python -m fif_mvp.cli.run_experiment ...` manually; pass `--ddp` for single-node multi-GPU CUDA launches or `torchrun ...` if you prefer manual control. DataParallel remains available but DDP is recommended to avoid gather warnings.

Key CLI knobs:

- `--train_noise_levels clean,low,med,high` controls which noise intensities are duplicated in the training split (default mixes全部四档)；
- `--energy_reg_weight 1e-4` 在训练损失中加入能量正则；
- `--energy_reg_scope {all,last}` 控制能量正则施加在全部摩擦层能量之和还是仅最后一层（默认 `last`）；
- `--energy_reg_target {absolute,normalized,margin,rank}`：默认 `rank`，对 batch 归一化能量执行排序/间隔约束；`absolute` 直接惩罚 `log1p(E)`，`normalized` 惩罚 batch 内 `log1p(E)` 方差；`--energy_reg_mode` 已废弃，仍保留向后兼容。
- `--energy_rank_margin`、`--energy_rank_topk`：用于 `margin/rank` 目标时的间隔与对比的 hardest 错误样本数量。
- `--energy_rank_fallback {absolute,none}`：当一个 batch 全对或全错时的退化正则（默认 `absolute` 确保 λ 仍有梯度）。
- `--energy_eval_scope {auto,per_sample}`：metrics/告警使用哪种能量，`auto` 与正则 scope 对齐（例如仅末层），`per_sample` 使用跨层求和。
- `--energy_metrics_source {normalized,raw}`：能量相关性指标默认使用 z-score 能量，可选改回 raw。
- `--energy_guard std_low=0.1,std_high=6,p90_high=8,factor=0.5,up=1.2,min_weight=1e-5,max=1e-3` 在训练过程中监控能量上下界并自动下/上调 λ（可用 `--energy_guard off` 禁用）；
- `--energy_watch std=0.1,std_high=10,p90=0.5,p90_high=8,mean_low=0.1` 在训练/验证/测试阶段记录能量告警并写入 `alerts.json`（可用 `--energy_watch off` 关闭）。
- existing `--noise_intensity {low,med,high}` 选择验证/测试噪声强度。
- Friction knobs: `--friction.eta_decay`, `--friction.mu_max`, `--friction.smooth_lambda`,
  `--friction.{normalize_laplacian,no_normalize_laplacian}`, `--friction.{recompute_mu,no_recompute_mu}`。
 - Data sampling: `--sortish_batches`（非 DDP）与 `--sortish_chunk_mult` 控制长度近似分桶，减少 padding 开销。

## Result artifacts

Every run subdirectory contains:

* `config.json`, `env.txt`, `timing.json`
* `train_log.txt`, `metrics_epoch.csv`, `energy_epoch.csv`（记录 `energy_log_mean/energy_std/p90`、`energy_norm_{mean,std,p90}`、`energy_alert` 和活跃的 λ）
* `test_summary.json` with `acc`, `macro_f1`, `loss`, `ece`, `energy_mean_test`, `energy_log_mean_test`, optional `energy_norm_*`, and energy‑error metrics (`energy_auroc/auprc`, `coverage_aurc`, `coverage_risk_at_{80,90,95}`, energy 分位)
* `confusion_matrix.csv`
* `energy_error_correlation.json`（包含 `pearson_r/auroc/auprc/aurc`、coverage‑risk 曲线子采样、正确/错误能量分位，并记录 energy_metrics_source）
* `alerts.json`（当 `--energy_watch` 或 `--energy_guard` 触发事件时生成，列出原因与 λ 回退）
* (Noisy SST-2 only) `noise_config.json`

Optional per-sample energy dumps live in `energy_per_sample.csv` when explicitly enabled.

## Reproducibility

Determinism defaults to ON. We seed Python/NumPy/PyTorch via `utils.seed.set_seed` and enable deterministic algorithms/cudnn. You can opt out with `--no_deterministic` to favor throughput (non-reproducible). Package/device metadata is recorded in `env.txt`. If dataset downloads fail, the CLI raises a descriptive error so you can pre-download via `datasets` cache.

Note on DDP: automatic GPU→CPU fallback is disabled for DDP jobs to avoid multi-process divergence. In DDP, failures are surfaced for an explicit rerun on CPU if needed.

## 项目文档导航

**快速开始**
- [`QUICK_START.md`](QUICK_START.md) - 实验快速启动指南（推荐新手阅读）
- [`EXPERIMENT_LAUNCHER_GUIDE.md`](EXPERIMENT_LAUNCHER_GUIDE.md) - 实验启动器详细文档

**项目管理**
- [`PROJECT_TRACKER.md`](PROJECT_TRACKER.md) - 版本追踪与实验记录（含跨版本对比表）
- [`WORK_BOARD.md`](WORK_BOARD.md) - 任务看板与里程碑进度
- [`PHASE_RESULTS.md`](PHASE_RESULTS.md) - 阶段性结果汇总（论文素材）
- [`DOCUMENT_IMPROVEMENT_ANALYSIS.md`](DOCUMENT_IMPROVEMENT_ANALYSIS.md) - 文档改进建议

**技术文档**
- [`docs/experiment_design.md`](docs/experiment_design.md) - 实验设计规范
- [`docs/code_structure.md`](docs/code_structure.md) - 代码库结构
- [`docs/FORMAT_STANDARD.md`](docs/FORMAT_STANDARD.md) - 文档格式标准
- [`docs/v1_1_0_energy_rework_plan.md`](docs/v1_1_0_energy_rework_plan.md) - v1.1.0 能量重构方案
- [`docs/diagnostics/`](docs/diagnostics/) - 问题诊断文档

**实验报告**
- [`docs/reports/`](docs/reports/) - 各版本实验报告（v1.0.0 ~ v1.0.4）

**AI 编程指南**
- [`GEMINI.md`](GEMINI.md) / [`AGENTS.md`](AGENTS.md) - AI coding prompts
