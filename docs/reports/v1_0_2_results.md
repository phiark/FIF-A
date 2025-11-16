# FIF v1.0.2 – SNLI & Noisy SST-2 实验报告

## 1. 版本概览
- **目标**：在 v1.0.1 的 DDP/能量聚合基础上，加入 `energy_reg_scope/mode`、`recompute_mu=False` 以及 KNN 桶级构图优化后，重新评估 SNLI 与 SST-2 noisy（low/med/high）下的 Baseline 与 Hybrid。
- **公共配置**：hidden=256、ff=1024、heads=4、epochs=5、batch_size=256、seed=42；Hybrid 在 SNLI 上使用 `K=3`、`recompute_mu=False`、`energy_reg_weight=1e-4`、`scope=all`、`mode=absolute`（`result/1.0.2/snli_hybrid_20251115_020337_seed42/config.json:1-38`）；SST-2 Hybrid 使用 `K=1` 但同样的能量正则设置（例如 `result/1.0.2/sst2_noisy_med_hybrid_20251115_043747_seed42/config.json:1-38`）。
- **结果目录**：`result/1.0.2/<experiment_name>_20251115_*_seed42/`。

## 2. SNLI 指标

| 模型 | λ / scope / mode | acc | macro_f1 | loss | ece | energy_mean | energy_std | pearson_r | 训练时长 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | `0` / — / — | 0.7767 | 0.7749 | 0.5521 | 0.0288 | 431.57 | 348.24 | -0.0968 | 656 s |
| Hybrid | `1e-4` / all / absolute | 0.6944 | 0.6928 | 0.6994 | 0.0253 | 0.0713 | 0.0185 | +0.0470 | 6319 s |

- 数值来源：`result/1.0.2/snli_baseline_20251115_015230_seed42/test_summary.json:2-7`、`metrics_epoch.csv:1-2`、`energy_error_correlation.json:2`、`timing.json:1-4`; Hybrid 对应文件为 `snli_hybrid_20251115_020337_seed42`.
- Hybrid 相比 Baseline 在 acc/macro_f1 上落后 ~8.2pt，loss 高 0.15，同时能量均值与标准差被压至 ~0.07 / 0.018，导致能量-错误皮尔逊相关转为正值。
- 训练耗时暴增（6.3k vs 0.66k 秒），来自新的批量 kNN 构图 + `recompute_mu=False` 设置仍需全批共享迭代。

## 3. Noisy SST-2 指标

| 噪声 | 模型 | acc | macro_f1 | loss | ece | energy_mean | pearson_r |
| --- | --- | --- | --- | --- | --- | --- | --- |
| low | Baseline | 0.7959 | 0.7957 | 0.5539 | 0.1130 | 1049.41 | -0.0238 |
| low | Hybrid | 0.7867 | 0.7860 | 0.4883 | 0.0786 | 0.79 | -0.0833 |
| med | Baseline | 0.7741 | 0.7734 | 0.5490 | 0.1117 | 757.91 | -0.0068 |
| med | Hybrid | 0.7775 | 0.7775 | 0.4739 | 0.0544 | 0.88 | -0.0902 |
| high | Baseline | 0.7362 | 0.7351 | 0.5942 | 0.0728 | 390.27 | +0.0275 |
| high | Hybrid | 0.7305 | 0.7267 | 0.5673 | 0.0436 | 1.23 | -0.0247 |

- 数据引用：`test_summary.json`、`energy_error_correlation.json`（例如 `result/1.0.2/sst2_noisy_low_baseline_20251115_034912_seed42/test_summary.json:2-7` 等）。
- Hybrid 仅在 med 噪声下略超 Baseline（+0.35pt acc），low/high 仍低 0.9~0.6pt；但 ECE 全面下降，主要源自输出熵升高。
- 所有 Hybrid 运行的 `energy_mean_test` 均接近 1，`pearson_r` 介于 -0.09 与 -0.02，说明 λ=1e-4 + all-scope 正则将能量压至近零，且能量与错误几乎无关。

## 4. 关键诊断
1. **能量塌缩依旧**：尽管引入了 `energy_reg_scope/mode`，本次运行仍使用 `scope=all`、`mode=absolute`，导致所有摩擦层贡献的能量被合并后整体压制（SNLI Hybrid 的 energy_std 仅 0.0185；`result/1.0.2/snli_hybrid_20251115_020337_seed42/metrics_epoch.csv:1-6`）。
2. **能量-错误相关性失效**：SNLI Hybrid 相关系数从 v1.0.1 的 +0.057 下滑至 +0.047，而 SST-2 hybrid 也都接近 0 或轻微负相关，无法支撑“能量≈不确定性”的假设。
3. **吞吐量恶化**：SNLI Hybrid 每步耗时 0.58s（`timing.json:1-4`），是 Baseline 的 10×；主要原因是新的 kNN 桶级构图 + 仍串行的 `_run_single` 迭代，既耗费 matmul/top-k，又没真正矢量化。
4. **SST-2 表现僵持**：med 噪声下 Hybrid 略胜（0.7775 vs 0.7741），但 low/high 依旧落后；能量优势依旧来自正则钳制，而非真实判别力。

## 5. 后续建议
1. **启用 normalized/last-scope 正则**：将 SNLI & SST-2 Hybrid 改为 `--energy_reg_scope last --energy_reg_mode normalized`，并扫 λ∈{1e-5,5e-5}，确认能量不再塌缩且相关性提升。
2. **恢复 `recompute_mu=True` 并引入 K warmup**：在 SNLI Hybrid 先用 K=1 预热 1-2 epoch，再切换到 K=3，以减少早期能量爆炸和训练耗时。
3. **重构 kNN 批处理**：要么回退到 v1.0.1 的 per-sample kNN（保证吞吐），要么把 `_run_knn_batch` 改为真正的矢量化拉普拉斯（参考 window 路径），避免“昂贵构图 + 串行迭代”的双重成本。
4. **监控能量分布**：基于新增的 `energy_std/p90` 指标，设定阈值（如 energy_std < 0.1 判定塌缩），训练时触发报警或自动降低 λ。
