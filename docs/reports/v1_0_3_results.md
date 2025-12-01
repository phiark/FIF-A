# FIF v1.0.3 – SNLI 与 Noisy SST-2 实验报告

## 1. 版本概览
- **目标**：检验 `scope=last + mode=normalized`、`energy_guard/std=0.1` 与 `energy_watch(std=0.1,p90=0.5)` 的默认配置能否阻止能量塌缩，并验证能量是否重新与错误相关；同时评估该配置在 SNLI 与 SST-2 不同噪声强度下的鲁棒性。
- **公共配置**：hidden=256、ff=1024、heads=4、epochs=5、batch=256、seed=42；Hybrid 使用 `K=3`、`neighbor=window`，λ 固定 `1e-4`（guard 仅在 `std<0.1` 时触发退火）。
- **运行目录**：`result/103/<task>_<noise?>_{baseline|hybrid}_20251116_*_seed42/`，保存完整 `metrics_epoch.csv`、`energy_epoch.csv`、`alerts.json` 等产物。

## 2. SNLI 指标

| 模型 | acc | macro_f1 | loss | ece | energy_mean | energy_std | pearson_r | 训练时长 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.7767 | 0.7749 | 0.5521 | 0.0288 | 431.57 | 348.24 | -0.0968 | 653 s |
| Hybrid | 0.6971 | 0.6957 | 0.6951 | 0.0192 | 6318.40 | 1501.19 | +0.0236 | 8699 s |

- 数据来源：`test_summary.json`、`energy_epoch.csv`、`timing.json`、`energy_error_correlation.json`（例如 Baseline：`result/103/snli_baseline_20251116_133127_seed42/test_summary.json:2-7`；Hybrid：`result/103/snli_hybrid_20251116_134230_seed42/energy_epoch.csv:1-11`）。
- Hybrid 的能量均值/标准差仍比 Baseline 高约 15×/4×，guard 未触发（`energy_alert=0`），训练时长 2.4 小时，远高于 Baseline 的 11 分钟（`timing.json:1-4`）。

## 3. Noisy SST-2 指标

| 噪声 | 模型 | acc | macro_f1 | loss | ece | energy_mean | energy_std | pearson_r |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| low | Baseline | 0.8016 | 0.8010 | 0.5015 | 0.0906 | 1044.70 | 566.49 | -0.0228 |
| low | Hybrid | 0.7867 | 0.7864 | 0.4836 | 0.0746 | 1.86 | 1.15 | -0.0659 |
| med | Baseline | 0.7615 | 0.7585 | 0.5493 | 0.1062 | 743.29 | 401.78 | -0.0511 |
| med | Hybrid | 0.7890 | 0.7890 | 0.4716 | 0.0464 | 1.50 | 0.84 | -0.0865 |
| high | Baseline | 0.6995 | 0.6880 | 0.6588 | 0.1164 | 421.91 | 232.03 | -0.0173 |
| high | Hybrid | 0.7213 | 0.7169 | 0.5658 | 0.0522 | 2.44 | 1.25 | -0.0439 |

- 指标来自各目录的 `test_summary.json` 与 `energy_epoch.csv`（例如 low hybrid：`result/103/sst2_noisy_low_hybrid_20251116_162623_seed42/test_summary.json:2-7`）。
- Guard 未触发（λ 始终 1e-4），但 `energy_watch` 在 low/med Hybrid 的第 1 个验证 epoch 因 `p90 < 0.5` 抛出了告警（`alerts.json:1-19`），说明能量曲线极度贴地。

## 4. 关键诊断
1. **能量幅度仍失衡**：SNLI Hybrid 的 `energy_mean_test=6.3e3`、`energy_std=1.5e3`（`energy_epoch.csv:1-11`），不仅远超 Baseline，也远高于 v1.0.2 期望的 “接近零” 区间。normalized 模式没有阻止整体量纲漂移，只是让 per-layer 差异保留，使 guard 永远不触发。
2. **吞吐量问题未解**：Hybrid 训练 5 epoch 需要 8700 s（`timing.json:1-4`），比 Baseline 慢 13×；v1.0.3 仍沿用 `K=3 + recompute_mu=False` 的旧图构建流程，`_run_knn_batch` 串行瓶颈依旧存在。
3. **SST-2 仅在 med/high 有收益**：Hybrid 在 med/high 噪声分别领先 Baseline 2.7/2.2pt acc（上表），但 low 噪声仍落后 1.5pt，并伴随能量塌缩（`energy_std_test≈1`）。能量-错误皮尔逊相关在所有噪声下依旧接近 0 或弱负值（`energy_error_correlation.json`）。
4. **能量监控仅提示“p90 过低”**：`alerts.json` 显示 watch 在 low/med Hybrid 的 epoch 1 报警，但 guard 未退火 λ，这意味着阈值逻辑尚不足以调整被压平的能量；SNLI Hybrid 没有任何告警，虽然能量均值暴走。

## 5. 建议
1. **重新设计 guard 条件**：在 `energy_guard` 中加入均值上限或 `energy_std` 上/下界双阈值，让 λ 在能量低于 0.1 或高于设定上限时都能调整；同时记录 `energy_mean` 的指数滑动平均以便决策。
2. **实现 K 预热 + `recompute_mu=True`**：按照 T-018 设计，前 1-2 epoch 使用 K=1 且强制重算 μ，可在能量刚被正则压制时提升 std，避免 watch 的长尾告警，并缩短 early-epoch walltime。
3. **矢量化 kNN 构图**：落实 T-019（graph cache/per-sample 模式），把 `_run_knn_batch` 里的 for-loop 替换为 batched matmul + mask，以期把 SNLI Hybrid 的 `avg_step_sec` 从 0.80 s 降至 <0.3 s。
4. **能量信号后处理**：在 `metrics_epoch.csv` 中追加 `energy_mean_z` 或 `log_energy_z`，辅助 guard 判断“偏高”场景；同时将 `alerts.json` 接入 sweep 脚本，自动降低 λ 或启用 normalized-last-two scope。

> 本报告将同步记录到 `PROJECT_TRACKER.md` 的 v1.0.3 小结中，以支撑后续 T-017~T-019 的评估。
