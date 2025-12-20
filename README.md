# FIF-A: Frictional Interaction Field for Robust NLP

**文档元数据**
- 文档类型：项目总览与论文/实验入口
- 当前版本：v1.2.0 (🔄 进行中)
- 论文状态：实验阶段
- 硬件基线：Tesla V100 16G ×1（v1.2.0+ 固定）
- 最后更新：2025-12-19
- 相关文档：`PROJECT_TRACKER.md`, `WORK_BOARD.md`, `PHASE_RESULTS.md`, `docs/`

**快速导航**
- [研究定位与贡献](#研究定位与贡献)
- [架构设计与独特性](#架构设计与独特性)
- [论文产出绑定](#论文产出绑定)
- [运行实验](#运行实验)
- [实验配置](#实验配置)
- [结果产物](#结果产物)
- [复现与约束](#复现与约束)
- [项目文档导航](#项目文档导航)

---

## 研究定位与贡献

本项目研究在轻量级 Transformer 中引入**摩擦交互场 (Frictional Interaction Field, FIF)** 对噪声鲁棒性的影响，并探索能量信号作为置信度代理的可行性，围绕论文实验与结果沉淀展开。

**核心创新**：
- **动态摩擦层**：迭代优化隐状态，引入可学习摩擦系数 μ 与归一化拉普拉斯。
- **能量正则化闭环**：训练正则与评估/告警使用同源能量刻度，支持 rank/margin 目标。
- **噪声条件化**：训练时混合多强度噪声数据，并显式注入 `noise_level_ids`。

**当前最佳结果（v1.0.4, SST-2 Low Noisy）**
- Hybrid acc **0.808**（Baseline 0.782）
- ECE 下降 **0.124 → 0.064**
- ⚠️ SNLI 基线仍需修复（详见 `PROJECT_TRACKER.md`）

---

## 架构设计与独特性

**架构主旨**：在 Transformer 内部引入可控的“摩擦交互场”，将 token 之间的相互作用能量作为可学习信号，并与噪声/难度条件联合建模，从而提升鲁棒性与可校准性。

**审稿人视角（创新点与可证伪性）**：
- 将“摩擦场”作为可微结构组件嵌入轻量 Transformer，区别于纯后处理的校准方法。
- 以能量排序/间隔目标直接优化“正确 < 错误”的单调性，可通过 AUROC/AURC 等量化指标验证。
- 噪声强度显式条件化到 μ/η 或损失结构中，强调跨域漂移可控，而非单纯调参。

**科研视角（机制假设）**：
- 能量刻度与错误概率应具有单调关系，rank/margin 正则是该假设的直接实现形式。
- 动态 μ + 归一化拉普拉斯可缓解能量爆炸并提高跨任务尺度一致性。
- 噪声条件化为“可控扰动”提供因果通路，便于分析域间鲁棒性变化。

---

## 论文产出绑定

论文写作所需的“方法—实验—结论”素材在以下文档闭环中维护：

- `PHASE_RESULTS.md`：阶段性结果总表 + 论文图表清单
- `PROJECT_TRACKER.md`：版本级实验记录与跨版本对比表
- `docs/experiment_design.md`：实验设计与指标口径（Methods/Setup）
- `docs/reports/`：单次运行报告（Appendix/补充材料）
- `result/1_2_0/*/test_summary.json`：可直接入表的最终指标

---

## 运行实验

```bash
# 运行 v1.2.0 标准实验（SNLI + SST-2，baseline + hybrid）
./run.sh

# 快速选择器
./quick.sh snli          # SNLI baseline + hybrid
./quick.sh sst2          # SST-2 baseline + hybrid
./quick.sh baseline      # 两个 baseline
./quick.sh hybrid        # 两个 hybrid

# 运行单个实验
./quick.sh snli_baseline
./quick.sh snli_hybrid
```

详细用法见 `EXPERIMENT_LAUNCHER_GUIDE.md`，速查见 `QUICK_START.md`。

---

## 实验配置

- 标准实验配置：`scripts/experiments.yaml`（默认保存至 `result/1_2_0/`）
- 历史实验配置：`scripts/experiments_archive.yaml`（存档用途）

常用 CLI 参数：
- `--energy_reg_weight` / `--energy_reg_target {absolute,normalized,margin,rank}`
- `--energy_rank_margin` / `--energy_rank_topk`
- `--energy_eval_scope {auto,per_sample}` / `--energy_metrics_source {normalized,raw}`
- `--friction.K` / `--friction.radius` / `--friction.neighbor`
- `--train_noise_levels clean,low,med,high`

---

## 结果产物

每次运行目录包含：

- `config.json`, `env.txt`, `timing.json`
- `metrics_epoch.csv`, `energy_epoch.csv`
- `test_summary.json`（acc/F1/ECE/能量指标/AUROC/AURC）
- `energy_error_correlation.json`（能量-错误相关性与 coverage-risk 曲线）
- `confusion_matrix.csv`, `alerts.json`（如启用监控）

---

## 复现与约束

- 默认开启确定性；可用 `--no_deterministic` 提升吞吐。
- 仅支持单卡 CUDA（Tesla V100 16G ×1）基线复现，不再维护 MPS/DDP 兼容路径。
- 所有实验结果需写入 `result/<version>/` 并在 `PROJECT_TRACKER.md`/`PHASE_RESULTS.md` 中登记。

---

## 项目文档导航

**实验入口**
- `QUICK_START.md` - 快速启动
- `EXPERIMENT_LAUNCHER_GUIDE.md` - 详细实验指南

**论文与项目管理**
- `PROJECT_TRACKER.md` - 版本追踪与跨版本对比
- `WORK_BOARD.md` - 任务看板与里程碑
- `PHASE_RESULTS.md` - 阶段性结果与论文素材

**技术文档**
- `docs/experiment_design.md` - 实验设计规范
- `docs/code_structure.md` - 代码结构
- `docs/FORMAT_STANDARD.md` - 文档格式标准
- `DOCUMENT_IMPROVEMENT_ANALYSIS.md` - 文档改进分析
- `docs/reports/` - 历史实验报告
