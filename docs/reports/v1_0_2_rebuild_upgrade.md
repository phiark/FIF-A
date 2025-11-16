# FIF v1.0.2 重构与升级指南

本指南基于 `docs/reports/v1_0_2_results.md` 的 SNLI 与 Noisy SST-2 结果，旨在把 v1.0.2 中暴露的能量塌缩、吞吐量和鲁棒性瓶颈转化为系统化的重构与升级计划。所有条目均按「动机 → 方案 → 交付物 → 验收指标」格式组织，便于分工与跟踪。

## 1. 关键诊断回顾
- **能量塌缩**：SNLI Hybrid 的 `energy_mean/std` 仅 `0.071/0.0185`，能量-错误相关性甚至转为正值（见 `docs/reports/v1_0_2_results.md` §2），说明 `scope=all + mode=absolute + λ=1e-4` 会把多层摩擦能量一并压制。
- **SST-2 Hybrid 无显著鲁棒性收益**：low/high 噪声下依旧落后基线 0.6–0.9pt，而 `ecc` 优势仅来自输出熵上升（§3）。
- **吞吐量恶化**：SNLI Hybrid 训练耗时 ~6.3k 秒（vs. Baseline 656 秒），主要来自 `_run_knn_batch` 的桶级构图 + `recompute_mu=False` 引发的串行迭代（§4）。
- **监控不足**：虽然添加了 `energy_std/p90`，但缺少阈值与告警逻辑，导致能量塌缩只能事后诊断。

## 2. 重构目标与方案

### 2.1 调整能量正则作用域与量纲
- **动机**：`scope=all` 把所有摩擦层能量以绝对值累加，λ=1e-4 足以把所有层压到零。
- **方案**：
  1. 将默认正则改为 `--energy_reg_scope last`，只作用在最后一层能量；同步将默认模式设为 `normalized` 并允许 `absolute` 作为备选。
  2. 扩展 CLI 以暴露 λ 扫描组合，建议默认 sweep `{1e-5, 5e-5}`。
  3. 在 `train/loop.py` 新增 `energy_std_threshold`（默认 0.1），当 `running_energy_std` 跌破阈值时自动降低 λ（例如 ×0.5）并在日志标记。
- **交付物**：`fif_mvp/cli/run_experiment.py` 的默认参数与帮助文本、`fif_mvp/train/loop.py` 的动态 λ 逻辑、`README.md` + `docs/experiment_design.md` 文档更新。
- **验收**：SNLI & SST-2 Hybrid 的 `energy_std_test ≥ 0.15` 且 `pearson_r(E,err)` 恢复到 `[-0.1, 0.1]` 之外（理想为正相关）。

### 2.2 恢复 `recompute_mu` 并引入 K warmup
- **动机**：`recompute_mu=False` 让批内所有摩擦层共享一套 μ，无法应对后期表示分布变化；K=3 从第一步起直接迭代也导致前期收敛慢。
- **方案**：
  1. 恢复 `recompute_mu=True` 为默认；在 DDP 下重算时缓存局部邻接矩阵以避免重复构图。
  2. 引入 `--friction.k_schedule t0,k0,k1`（或更简单 `--friction.k_warmup_epochs`）：前 `w` 个 epoch 使用 K=1，随后切换到配置的 `K`。
  3. 在 `HybridClassifier.forward` 中区分「预热」与「完全迭代」阶段，避免在 K=1 期内执行多余的能量 write-back。
- **交付物**：`fif_mvp/models/friction_layer.py`, `fif_mvp/models/hybrid_model.py`, CLI 与脚本（`scripts/snli_hybrid.sh`、`scripts/sst2_noisy_hybrid.sh`）。
- **验收**：SNLI Hybrid 训练 walltime 降至 <3k 秒（单机 A100），验证 acc ≥ 0.72，`train_log` 中能量曲线首两 epoch 不再贴地。

### 2.3 重构 kNN 构图与能量计算
- **动机**：v1.0.2 的桶级 kNN 仍串行 `_run_single`，导致 GPU 利用率低；且 KNN 图顶点在 batch 间共享会放大误差。
- **方案**：
  1. 提供切换：`--friction.knn_mode {per_sample,batch}`，默认 per-sample（回退到 v1.0.1 的模式，但向量化）。
  2. 把 `_run_knn_batch` 改为纯张量化，避免 Python for-loop；参考 `train/energy.py` 的窗口实现，预先构建 `edge_index` 并复用。
  3. 为 SNLI 引入 `--friction.graph_cache_size`，允许跨 batch 重用 kNN 邻接以摊薄构图成本。
- **交付物**：`fif_mvp/models/friction_layer.py`, `fif_mvp/utils/graph.py`（新文件，可选）以及 `README.md` 的性能提示。
- **验收**：在相同硬件上，Hybrid step time 较 v1.0.2 降低 ≥40%；`timing.json` 中 `construct_graph_seconds` 占比 <30%。

### 2.4 建立能量监控与告警
- **动机**：目前仅离线 CSV，无法在训练时阻断能量塌缩。
- **方案**：
  1. 在 `metrics_epoch.csv` 增加 `energy_alert` 字段（0/1），记录是否触发阈值。
  2. 启用 `--energy_watch metrics,std_threshold=0.1,p90_threshold=0.5` 之类的 CLI，允许配置多条告警规则。
  3. 当告警触发时，在 `train_log` 打印以及 `result/.../alerts.json` 中落盘，供 sweep 抓取。
- **交付物**：`fif_mvp/train/loop.py`, `fif_mvp/train/metrics.py`（若需拆出）、`README.md` 说明。
- **验收**：在 v1.0.2 设定下重新 dry-run，可观察在 Epoch 0-1 触发告警，并能记录在 `alerts.json`；在调参后二次运行告警消失。

## 3. 升级路线图

| 阶段 | 目标 | 任务 | 预期产出 |
| --- | --- | --- | --- |
| Phase A（本周） | 恢复能量可辨识度 | 完成 §2.1 & §2.4，跑 SNLI Hybrid λ 扫描（`scope=last`） | `result/1.0.3/snli_hybrid_last_norm_*`, 更新报告 |
| Phase B（下周） | 降低 Hybrid 吞吐成本 | 完成 §2.2 & §2.3，更新所有脚本/README | `timing.json` 改进、`docs/reports/v1_0_3_results.md` |
| Phase C（待定） | 扩展能量信号利用 | 若能量-错误相关恢复为正，设计能量阈值拒识或蒸馏实验 | 新增 CLI `--energy_threshold_eval`、`docs/experiment_design.md` 扩展 |

## 4. 实施清单
1. **配置层**：更新 `fif_mvp/cli/run_experiment.py` 默认参数 + 帮助文本；脚本补充 λ/K 预热。
2. **模型层**：在 `HybridClassifier` 内注入 `k_schedule` 与动态 μ；需要对 `FrictionLayer.forward` 的签名新增 `k_override`。
3. **训练循环**：实现能量告警、动态 λ、自适应日志；校验 `DistributedDataParallel` 下的 reduce 行为。
4. **文档与追踪**：同步 `README.md`、`docs/experiment_design.md`、`PROJECT_TRACKER.md`、`WORK_BOARD.md`，并在报告中记录新的指标。

## 5. 实验矩阵（v1.0.3 规划）

| 数据集 | 模型 | 正则设置 | K 预热 | kNN 模式 | 目标 |
| --- | --- | --- | --- | --- | --- |
| SNLI | Baseline | λ=0 | 无 | N/A | 作为吞吐/评价基线 |
| SNLI | Hybrid | scope=last, mode=normalized, λ∈{1e-5,5e-5} | 2 epoch @K=1 → K=3 | per-sample + graph cache | `acc ≥ baseline-2pt`, `pearson_r ≥ 0.1` |
| SST-2 noisy (low/med/high) | Hybrid | 同上 | 1 epoch 预热 | per-sample | 相比 baseline `Δacc ≥ 0`，`ece` 不回退 |

## 6. 风险与备用方案
- **λ 仍导致塌缩**：准备 `mode=normalized`, `scope=last_two` 备用，实现上只聚合倒数两层能量。
- **K 预热不足以稳住能量**：考虑在预热阶段调低 `η_decay` 或 `mu_max`，CLI 需支持按阶段设置。
- **kNN Vectorization 过慢**：如 GPU 内存受限，可保留批量模式但加速 `_run_knn_batch`，并记录开关。

## 7. 交付节奏
1. 完成配置/模型改造 → 提交 PR 并附 SNLI 片段日志。
2. 跑完整 SNLI + SST-2 sweep → 更新 `docs/reports/v1_0_3_results.md`。
3. 把新的能量监控写进 README/Tracker，确保 `run.sh` 默认参数同步。

> 所有实现都需保持 determinism，更新后的 λ/K 默认必须记录在 `config.json` 的顶层键值，并在 `env.txt` 标识新的 CLI 版本。
