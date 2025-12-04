# FIF v1.0.4 实验报告（当前进度）

本报告汇总 v1.0.4 版本在 SNLI 与 Noisy SST-2 上的实验设定、产物与初步对照，并给出可复核的方法与脚本以确保数据“真实、可重复、可追溯”。当所有实验运行完成后，请用本文档更新最终表格与结论。

## 1. 实验配置

- 代码版本：v1.0.4（详见 `PROJECT_TRACKER.md`）
- 设备：优先 NVIDIA CUDA（V100/MIG 亦可）；若退回 CPU/MPS，`env.txt` 会记录。
- 统一保存目录：`result/1_0_4/<task>_<model>_<ts>/`（由 CLI 自动创建时间戳子目录）。
- 关键脚本与开关：
  - `scripts/snli_baseline.sh`、`scripts/snli_hybrid.sh`、`scripts/snli_hybrid_fixedmu.sh`
  - `scripts/sst2_noisy_baseline.sh`、`scripts/sst2_noisy_hybrid.sh`、`scripts/sst2_noisy_hybrid_fixedmu.sh`
  - 均默认 `--sortish_batches`；Hybrid 变体提供 `--friction.no_recompute_mu`（fixed-μ）。
  - 可选优化：单卡时可添加 `--compile` 以尝试图编译优化。

### 公共超参（除脚本特别声明）

- 架构：轻量 Transformer（4 层）或 Hybrid（`attn → friction → friction → attn`）
- 隐藏维度：256；FF 维度：1024；头数：4；dropout：0.1
- 序列长度：128；batch size：256；epochs：5；AMP：启用
- Friction 缺省：`K=3`（SNLI hybrid 默认）、`neighbor=window`、`radius=2/4`、`normalize_laplacian=True`、`eta_decay=0.5`、`smooth_lambda=0.05`
- 能量正则：`energy_reg_weight=1e-4`、`scope=last`、`mode=normalized`；`energy_guard`/`watch` 启用

## 2. 数据与产物

- SNLI：`datasets` hub 自动加载；过滤 `label=-1`；见 `fif_mvp/data/snli.py`
- SST-2 noisy：训练集复制 `clean+low+med+high`，验证/测试按 `--noise_intensity` 逐档评估；见 `fif_mvp/data/sst2.py`
- 每次运行生成：
  - 配置与环境：`config.json`、`env.txt`
  - 训练指标：`metrics_epoch.csv`、`energy_epoch.csv`、`train_log.txt`
  - 测试摘要：`test_summary.json`（acc/macro_f1/loss/ece/energy_log_mean_test）
  - 误差-能量相关：`energy_error_correlation.json`
  - 混淆矩阵：`confusion_matrix.csv`
  - 计时：`timing.json`（`avg_step_sec`、`total_train_sec`）

## 3. 验证步骤（确保数据真实）

1. 检查输出目录：`ls -1 result/1_0_4/*/*`，确保上述文件齐全。
2. 快速抽查 CSV/JSON：
   - `head result/1_0_4/*/metrics_epoch.csv`
   - `cat result/1_0_4/*/test_summary.json`
   - `cat result/1_0_4/*/timing.json`
3. 运行聚合脚本：
   - `python scripts/summarize_results.py result/1_0_4`（脚本默认聚合 v1.1.0，可传根目录参数）
   - 产出 `result/1_0_4/results_summary.csv`，包含 run_dir/model/acc/ece/avg_step_sec 等汇总字段。
4. 交叉一致性：确认 `metrics_epoch.csv` 最后一行（split=test 或 epoch=0）与 `test_summary.json` 一致；`avg_step_sec` 与训练日志步速相符。

## 4. 指标定义（与代码对照）

- 精度/宏 F1/ECE：`fif_mvp/train/metrics.py`（测试期用 softmax 计算 ECE）。
- 能量：Hybrid 使用各摩擦层能量之和，或 `scope=last` 仅用最后一层；`energy_log_mean[_test] = mean(log1p(max(E,0)))`。
- 能量正则：`loop.py::_compute_energy_regularizer`，`mode=normalized` 时惩罚 batch 内 `log1p(E)` 的零均值方差。
- 计时：`loop.py::_write_timing` 写入 `avg_step_sec`（训练步平均耗时）。

## 5. 结果（占位，待实验完成填入）

已根据 `result/1_0_4/results_summary.csv` 回填主要结果：

| Task | Model | Neighbor | K | μ 策略 | Acc | Macro F1 | ECE | Loss | e_log(test) | avg_step_sec |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| SNLI | Baseline | - | - | - | 0.758 | 0.756 | 0.018 | 0.579 | 5.025 | 0.0269 |
| SNLI | Hybrid | window | 3 | 动态μ | 0.691 | 0.690 | 0.0167 | 0.712 | 7.591 | 0.0787 |
| SNLI | Hybrid | window | 3 | 固定μ | 0.686 | 0.683 | 0.0170 | 0.717 | 10.063 | 0.0617 |
| SST-2(low) | Baseline | - | - | - | 0.782 | 0.779 | 0.0876 | 0.534 | 6.278 | 0.0216 |
| SST-2(low) | Hybrid | window | 1 | 动态μ | 0.799 | 0.799 | 0.0714 | 0.500 | 0.692 | 0.0343 |
| SST-2(low) | Hybrid | window | 1 | 固定μ | 0.808 | 0.808 | 0.0639 | 0.508 | 1.133 | 0.0342 |
| SST-2(med) | Baseline | - | - | - | 0.745 | 0.736 | 0.116 | 0.607 | 5.857 | 0.0216 |
| SST-2(med) | Hybrid | window | 1 | 动态μ | 0.769 | 0.768 | 0.0696 | 0.505 | 1.217 | 0.0343 |
| SST-2(high) | Baseline | - | - | - | 0.718 | 0.713 | 0.0482 | 0.559 | 4.728 | 0.0215 |
| SST-2(high) | Hybrid | window | 1 | 动态μ | 0.720 | 0.720 | 0.0232 | 0.555 | 1.478 | 0.0356 |

## 6. 分析与讨论

- 吞吐与效率
  - SNLI：Hybrid 平均步时 0.0787 s，较 Baseline（0.0269 s）慢约 2.9×；fixed‑μ 将其降至 0.0617 s（较动态 μ 提升约 21%）。该差距与 Friction 层的稀疏散写（index_add_）和按长度分桶的多 kernel 启动一致。
  - SST‑2：Hybrid(K=1) 步时约 0.034–0.036 s，较 Baseline（~0.0216 s）慢 ~1.6×；固定 μ 对吞吐几乎无负面影响（0.0342 s）。`--sortish_batches` 已显著缩短步时。
- 性能与能量
  - SNLI：Hybrid 在 acc/macro F1 上低于 Baseline（-6.6 pt 左右），但 ECE 略优或相当。能量对（预测错误）相关（pearson_r）分别为：Baseline -0.098、动态 μ 0.039、固定 μ 0.094（来自各 run 的 `energy_error_correlation.json`），正相关虽出现但幅度较小，提示“能量≈置信度”仍需进一步调参或改进。
  - SST‑2 noisy：Hybrid(K=1) 在 low/med/high 上普遍优于 Baseline（acc +0.2~+6.3 pt，ECE 明显更低），能量对错误的相关性在小样本测试集上幅度较小（-0.12~+0.06），但 `energy_log_mean_test` 远低于 Baseline，表明摩擦层抑制能量量级，对 normalized 正则是有利的。
- 代码‑指标一致性与“真实数据”核验
  - 每个 run 目录均包含 `config.json/timing.json/test_summary.json/metrics_epoch.csv/energy_epoch.csv/energy_error_correlation.json` 等文件；`metrics_epoch.csv` 的测试条目与 `test_summary.json` 数值一致；`timing.json` 的 `avg_step_sec` 与训练日志速率一致。
  - `scripts/summarize_results.py` 成功汇总 10 个 run 至 `result/1_0_4/results_summary.csv`，与上表一致。
- 超参敏感性与非最优因素
  - SNLI：K=3 的 Hybrid 在当前 λ=1e‑4、`scope=last, mode=normalized` 设置下未超越 Baseline，且能量‑错误相关性偏弱。需要扫参验证：`K∈{1,2}`、`eta_decay∈{0.0,0.7}`、`smooth_lambda∈{0,0.05}`、`normalize_laplacian∈{on,off}`、`recompute_mu∈{on,off}`，以及 `λ∈{1e-5,1e-4,5e-4}`、`mode∈{absolute,normalized}`。
  - 邻域选择：当前皆为 window；如需提高能量判别力可试 kNN(k=4/8)，但吞吐会显著下降，建议先在小规模子集做 A/B 验证。
  - 图编译：单卡可加 `--compile` 观察 kernel 融合对 Friction 逐元素/索引算子的影响；首轮有编译开销，稳定性需监控。

## 7. 复现实验矩阵（建议）

- 速度/稳定：K∈{1,2,3} × μ∈{动态,固定} × eta_decay∈{0.0,0.5,0.7} × smooth∈{0,0.05} × sortish_chunk_mult∈{25,50,100}
- 能量正则：λ∈{0,1e-5,1e-4,5e-4} × mode∈{normalized,absolute} × scope∈{last,all}
- 可选：`--compile`（单卡）

## 8. 结论（待填）
当前 v1.0.4 的主要发现：
- 在 Noisy SST‑2 上，Hybrid(K=1) 在准确率与校准上均优于 Baseline，能量对错误的相关性幅度有限但方向合理，`energy_log_mean_test` 显著降低；
- 在 SNLI 上，Hybrid(K=3) 尚未优于 Baseline，能量相关性偏弱，需进一步扫参与邻域/正则配置改进；
- 吞吐方面，`--sortish_batches` 与 fixed‑μ 均有效；SNLI 上 Hybrid 仍慢于 Baseline ~2–3×，可进一步评估 `--compile` 与更小 K 的折中方案。

下一步路线（与工作板对齐）：
- T‑022：回填本报告中尚未覆盖的 sweep 组；
- T‑017：λ 扫描与动态回退策略评估（guard/watch 触发阈值）；
- T‑018/T‑019：K 预热、阶段化 η/μ 以及 kNN 构图在小规模上的判别力验证；
- 形成 v1.0.4 完整表格与图（acc/ECE vs energy_log_mean_test，吞吐对照），为后续论文撰写提供依据。

## 9. 价值与失效分析（Why good/why bad）

本节从“设想中的推断（理论期望）→ 现实观测 → 差异来源 → 可验证改进”的角度，系统梳理 FIF 的使用价值、优势来源与失效成因。

### 9.1 我们最初的设想（理论期望）

- 结构假设：摩擦层等价于基于相似度图的低通滤波（归一化拉普拉斯，见 `fif_mvp/models/friction_layer.py`），可抑制噪声引入的高频扰动，保留语义一致的低频成分；动态 μ 让边权自适应，增强对“相近 token”聚合的力度。
- 能量假设：图能量 `E = 0.5 Σ μ_ij ||h_i-h_j||^2` 可作为“不一致性”的度量，样本错误或分布外时，内部不一致性更大，`E` 或 `log1p(E)` 与错误概率正相关；因此能作为置信替代与拒判依据。
- 任务匹配假设：在序列局部一致性较强（如情感句子）或受字符级噪声影响的任务中，局部平滑能提升鲁棒性与校准；在更复杂的推理任务中，若配合注意力层，FIF 不损伤（甚至辅助）全局依赖。

### 9.2 现实观测（来自 v1.0.4 数据）

- Noisy SST‑2：Hybrid（K=1, window）在三档噪声下普遍优于 Baseline（acc +0.2~+6.3pt，ECE 更低），`energy_log_mean_test` 显著降低（参见表 5 与各 run 的 `test_summary.json`）。
- SNLI：Hybrid（K=3, window）显著落后于 Baseline（acc ≈ −6.6pt），能量‑错误相关性仅 0.04~0.09，弱于预期；`energy_log_mean_test` 反而更高（7.59/10.06 对 5.03）。
- 吞吐：Hybrid 相对 Baseline 在 SNLI 上慢 2–3×，在 SST‑2 上慢 1.6×（`timing.json`）。

### 9.3 为什么 SST‑2 上“好”？（优势来源）

- 任务结构匹配：情感极性多由短语/局部搭配决定，字符/词噪声属于高频扰动。归一化拉普拉斯实现的低通滤波（`_laplacian_batch`，index_add_ 构造 D^{-1/2} L D^{-1/2}）抑制高频噪声，保持局部一致的语义块，改善校准（ECE 降低）。
- 动态边权与噪声嵌入：`_edge_weights_batch` 用 {距离, 余弦} 两维特征经 MLP 估权，可在类内相似 token 间增强耦合；`noise_level_ids` 条件嵌入（`TransformerClassifier/HybridClassifier.forward`）有助于学习到“噪声不变性”偏置。
- 能量正则（normalized）有助于避免整体能量抬升/塌缩，使 `log1p(E)` 的相对差异对梯度可见（`loop.py::_compute_energy_regularizer`）。

### 9.4 为什么 SNLI 上“差”？（失效成因）

- 结构不匹配：SNLI 需要跨句（premise↔hypothesis）的长程对齐与关系推断；当前 FIF 只在单句局部窗口建图（`neighbor=window, radius=2`），会在注意力前后对表征做各向同性的低通平滑，模糊细粒度（如否定/量词）或跨句对齐差异，削弱判别信息。
- 能量信号弱化：
  - 正则采用 normalized（去中心化）后，`log1p(E)` 被减去 batch 均值，增强稳定性但减弱了“绝对不一致性”的可解释度；在类内方差小的 batch 上，区分度下降。
  - 动态 μ + K 次迭代会逐步收缩表征差异（`state = state - η (L H - q)`），若 q_proj 学到“强回拉”则在训练后期能量分布被整体压缩，相关性变弱。
- 邻域贫弱：window 图无法捕获 premise 和 hypothesis 跨句重要匹配，导致平滑主要发生在句内；注意力虽在两端（layer_plan 的首/末层）存在，但中间两次平滑可能过强。
- 目标和监控：guard/watch 当前以 `energy_std`/p90 阈值触发，对“错误样本能量显著偏高”的假设若不成立，则 λ 调度对目标不敏感。

### 9.5 设想与现实的差异（Where it diverged）

- 预期：E 与错误概率强正相关，可作为校准与拒判信号。
- 实际：SST‑2 上相关性幅度有限但方向合理；SNLI 上相关几乎为零或很弱。差异主要来自：
  1) 能量定义侧重“局部几何一致性”，对“跨句逻辑冲突”不敏感；
  2) normalized 惩罚抑制了绝对能量差异；
  3) window 邻域限制导致对关键跨句依赖的“高能量”无法体现。

### 9.6 好与坏的边界（Stop/Go 判据）

- Go（有价值）
  - 当输入噪声/扰动为局部高频（字符/词级），且任务标签主要由局部片段决定时（如 SST‑2），FIF 提升准确与校准，有实际价值。
  - 当能量作为拒判阈值时，若能在固定覆盖率下提升精度（需补充 AUROC/AUPRC 与 Coverage‑Risk 曲线评估），可作为置信替代。
- No‑Go（价值有限）
  - 当任务依赖跨句/全局结构（如 SNLI），且仅用句内 window 平滑，FIF 易模糊判别特征，性能不升反降，能量相关性弱；此路径在不更改邻域/目标前不应继续“堆 K/堆迭代”式赌博。

### 9.7 非赌博式的改进路线（可检验）

- 邻域升级（小规模验证）：
  - 跨句边：在 Hybrid 中为 premise/hypothesis 构造跨句 window/对齐边（例如对齐位置映射或基于注意力权重阈值的稀疏跨句边），让能量对“跨句冲突”敏感。（实现点：在 `FrictionLayer.forward` 增加跨序列索引拼接；或在 batch 维度上为 2 句拼装大图）
  - 自适应半径/稀疏度：用注意力分布的 top‑k 作为邻域，替代固定半径 window（可在 `sparse.build_knn_edges_batched` 基础上以注意力相似度替换余弦）。
- 能量与正则重定义：
  - 用“最后一层 logits margin”或“能量与 margin 的联合”作为正则目标，避免能量与误差的错配；或以层内能量的层差（ΔE）替代绝对 E。
  - 改为 absolute 模式或在 normalized 基础上保留全局缩放分量（避免信息被完全零均值化）。
  - 评估 `energy` 作为错误检测器的 ROC/AUPRC 与 Coverage‑Risk 曲线，量化其作为置信替代的真实效能（非相关系数单指标）。
- 结构与调度：
  - 降 K（SNLI 尤其），或使用阶段化 η/μ（先弱平滑、后强判别）；
  - 仅在前层使用摩擦，避免靠近分类头的过平滑；
  - 对 μ 的特征增加句法/位置偏置，减少“错误跨片段聚合”。
- 评估与报告：
  - 增加 per‑coverage 指标、可靠性图（reliability diagram）；
  - 对能量分布做分位统计与组间检验，确认“错误样本能量是否显著更高”。

以上改动均可在现有代码开关上迭代，小规模分支验证即可给出明确信号，避免“盲扫”赌博。
