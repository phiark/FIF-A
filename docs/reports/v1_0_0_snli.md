# FIF v1.0.0 – SNLI 混合架构实验报告

## 1. 实验概览
- **数据集**：SNLI（3 类自然语言推理）。
- **对照组**：Transformer 基线 vs. Friction Hybrid（attention ↔ friction 交替）。
- **训练设定**：`train_noise_levels = clean,low,med,high`；batch size 256；5 epoch；`energy_reg_weight = 0`（尚未启用能量正则）。
- **结果目录**：`result/1_0_0_snli/snli_{baseline,hybrid}_20251112_*_seed42`.

> v1.0.0 目标原本聚焦噪声鲁棒性与能量正则。但本轮 SNLI 运行采用 `energy_reg_weight = 0`，未能纳入能量约束，诊断应重点关注“无正则 + 动态 μ”下的能量行为。

## 2. 核心指标（test set）
| 模型 | acc | macro_f1 | loss | ece | energy_mean | energy_log_mean |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.7834 | 0.7824 | 0.5347 | 0.0186 | 456.56 | 5.90 |
| Hybrid | 0.6877 | 0.6868 | 0.7188 | 0.0128 | 1441.28 | 7.20 |
| Δ (Hybrid−Base) | −0.0957 | −0.0956 | +0.1841 | −0.0058 | ×3.16 | +1.30 |

> Baseline 稳定逼近 78% acc，而 Hybrid 在所有核心指标（除 ECE 外）全面落后，且能量量纲仍大幅漂移。

## 3. 训练/验证动态
1. **Baseline**：验证 acc 从 0.62 → 0.78 稳步提升，但能量均值在前后期从 185 → 463 上升，显示噪声混合训练仍会抬升能量基线。
2. **Hybrid**：验证 acc 最高 0.70，显著低于 Baseline；然而能量均值自 11k → 1.5k 逐 epoch 下降一个数量级，说明动态 μ + η 衰减确实对能量爆炸有抑制，但仍停留在大幅高于基线的区间。
3. **训练收敛**：Hybrid 训练 loss 在 0.74 左右停滞，且 acc 未跟随验证指标同步增长，暗示摩擦层梯度/学习率配置可能导致欠拟合。

## 4. 能量-错误关联
| 模型 | Pearson r (能量, 预测错误) |
| --- | --- |
| Baseline | −0.11 |
| Hybrid | −0.01 |

- Baseline 仍呈弱负相关（能量大的样本略易正确，违反能量≙风险假设）。
- Hybrid 几乎无相关（−0.009），摩擦层输出能量无法提供置信度线索，印证能量正则缺失+尺度不稳的风险。

## 5. 错误分析
1. **类别分布**：Hybrid 在类 0/2（常对应 entailment / contradiction）上误差显著上升；正确样本从 ~2.9k/2.5k 降至 2.5k/2.0k，而误判为“中性”比例增加。
2. **Confusion pattern**：
   - Baseline：偏差集中在中性类，错分约 368/492。
   - Hybrid：对类 0/2 的错分分别扩散到 548/634，说明噪声嵌入或摩擦层滤波未能分离极性语义。
3. **Calibration**：虽然 Hybrid ECE 更低（0.0128），但该收益来自更大的 softmax 熵（整体置信度下降），并未反映真实校准。

## 6. 关键诊断
1. **能量正则缺失**：`energy_reg_weight = 0` 使能量信号仅由摩擦层自组织，阻碍了“能量≈错误”目标；下轮需启用 λ=1e-4/5e-4 方案。
2. **摩擦更新深度**：Hybrid 仅配置 `K=1`，比 Baseline `K=3` 更浅，可能过度依赖高能量态但又缺少多步迭代的稳定化。
3. **噪声条件化迁移**：SNLI 训练噪声与任务不匹配（原计划面向 SST-2），导致噪声嵌入50%概率学习到“任务无关偏移”，进一步污染注意力特征。

## 7. 后续建议
1. **重新运行 v1.0.0-B/C**：在 SNLI 上至少启用 `λ = 1e-4` 并记录 `energy_log_mean` 曲线，确认能量是否随错误上升。
2. **调参 Hybrid**：恢复 `K=3` 或增大 `smooth_lambda`，并对 `eta` 做更激进衰减，缩小能量均值与 Baseline 的 3× 差距。
3. **校验噪声管线**：SNLI 目前无噪声注入脚本；若仅复制 SST-2 噪声策略，需明确噪声如何映射到 premise/hypothesis，避免训练/测试分布错配。
4. **能量后处理**：考虑将 `energy_log_mean` z-score 化并写入 `metrics_epoch.csv`，以免被绝对量纲主导。

---

**产出**：本报告为 v1.0.0 SNLI 运行的正式诊断，可直接引用于 `docs/experiment_design.md` 的结果段或论文补充材料。
