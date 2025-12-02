# FIF 项目追踪文档

本文件记录 FIF 混合架构实验的版本沿革，所有版本条目按以下统一技术格式撰写，任何新增版本都应在末尾追加同结构内容。

## 文档格式（新增版本需遵循的顺序与字段）
1. **版本号**：遵循 `MAJOR.MINOR.PATCH`（管线/公式/框架改动、微调、缺陷修复）。
2. **更新时间**：`YYYY-MM-DD`。
3. **迭代来源**：说明该版本基于哪个版本或分支演进。
4. **版本目标**：量化目标或验证假设。
5. **公式与管线**：若有改动需写“变更：...”；若沿用则注明“沿用上一版”，并给出关键公式/流程。
6. **实验记录**：列出实验设置、核心指标（可用表格），并指向数据目录。
7. **修改与改进点**：归纳主要实现/配置调整及其动机。

---

## 版本 0.0.0
- **版本号**：0.0.0
- **更新时间**：2025-11-09
- **迭代来源**：根据早期 `../FIF/fif_simple` 原型与草稿实验整理为 MVP 代码库的首个实验版。
- **版本目标**：
  - 在 SST-2 噪声测试集上对比 Transformer 基准线与 Friction Hybrid，验证能量信号是否可指示错误。
  - 建立统一 CLI、数据管线与结果产物目录，为后续论文撰写积累基线数据。
- **公式与管线**：
  - **能量公式**：`E = 0.5 Σ_ij μ_ij ||h_i - h_j||^2`；`μ_ij = softplus(MLP([||h_i-h_j||, cos(h_i, h_j)])) + 1e-5`。
  - **迭代方程**：`H^{t+1} = H^{t} - η (L(H^t) - q)`，其中 `q = W_q H^0` 且 `μ` 在迭代前一次性估计。
  - **管线**：在干净 SST-2 训练上训练，验证/测试阶段按照 `noise_config.json` 注入 low/med/high 级噪声；CLI 统一写入 `result/<task>_<noise>_<model>_<ts>`.
- **实验记录**（SST-2 noisy，seed=42；详细日志见 `result/` 子目录）：

| Noise | Model | acc | macro_f1 | loss | ece | energy_mean |
| --- | --- | --- | --- | --- | --- | --- |
| low | baseline | 0.7706 | 0.7706 | 0.5394 | 0.0969 | 77.75 |
| low | hybrid | 0.7523 | 0.7516 | 0.6184 | 0.1235 | 5495.70 |
| med | baseline | 0.7030 | 0.7019 | 0.8036 | 0.1798 | 147.52 |
| med | hybrid | 0.6892 | 0.6825 | 0.8199 | 0.1915 | 1495.78 |
| high | baseline | 0.6663 | 0.6586 | 0.8557 | 0.1886 | 54.95 |
| high | hybrid | 0.6273 | 0.6093 | 0.9565 | 0.2338 | 1843.41 |

- **修改与改进点**：
  1. 将 `fif_simple` 的玩具 FIF 层重写为 `FrictionLayer`，支持窗口/knn 建图与能量导出（`fif_mvp/models/friction_layer.py`）。
  2. 构建 `HybridClassifier`，以 `attention → friction → friction → attention` 层计划混合 MHSA 与摩擦层。
  3. 引入统一 CLI（`fif_mvp/cli/run_experiment.py`），规范化配置、噪声注入与结果落盘，作为论文实验的初始基线。

## 版本 1.0.0
- **版本号**：1.0.0
- **更新时间**：2025-11-09
- **迭代来源**：基于 0.0.0 同步完成工作板 T-001~T-005。
- **版本目标**：
  - 训练阶段覆盖 `clean+low+med+high` 噪声并提供噪声条件输入，缩小域间 gap。
  - 稳定摩擦层（动态 μ、度归一化、η 衰减、序列平滑）以抑制能量爆炸。
  - 通过能量正则和对数能量指标，将能量信号纳入损失与评估闭环。
- **公式与管线（变更）**：
  - **摩擦层**：循环内重算 `μ = clamp(softplus(MLP(...)))`，构建归一化拉普拉斯 `D^{-1/2} L D^{-1/2}`，每步步长 `η_t = η·decay^t` 并附加 1D 平滑。
  - **训练损失**：`L = CE + λ_energy ⋅ log(1 + E_batch)`，`λ_energy` 由 CLI `--energy_reg_weight` 控制。
  - **数据/模型**：`train_noise_levels` 控制训练复制的噪声强度；`SequenceCollator` 输出 `noise_level_ids`，编码器通过噪声嵌入调节特征。
- **实验记录**（SNLI，seed=42；详见 `docs/reports/v1_0_0_snli.md`，原始日志位于 `result/1_0_0_snli/`）：

| Model | acc | macro_f1 | loss | ece | energy_mean | energy_log_mean | pearson_r(energy,error) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.7834 | 0.7824 | 0.5347 | 0.0186 | 456.56 | 5.90 | -0.11 |
| Hybrid | 0.6877 | 0.6868 | 0.7188 | 0.0128 | 1441.28 | 7.20 | -0.01 |

  - Hybrid 相比基线在 acc/macro_f1 上约 -9.6pt，loss +0.18，能量均值高出 3.2×，能量-错误相关性趋近 0，暂未实现“能量≈风险”目标。
  - 本次运行采用 `energy_reg_weight = 0`（尚未启用 λ=1e-4/5e-4），后续需在相同设置下补跑 v1.0.0-B/C 以验证能量正则的收益。
- **修改与改进点**：
  1. 数据与模型条件化：`fif_mvp/data/sst2.py`, `fif_mvp/data/__init__.py`, `fif_mvp/models/transformer_baseline.py`, `fif_mvp/models/hybrid_model.py`.
  2. 摩擦层稳定性：`fif_mvp/models/friction_layer.py` 动态 μ + 归一化拉普拉斯 + η 衰减 + 平滑。
  3. 能量正则与指标：`fif_mvp/train/loop.py`, `scripts/sst2_noisy_*.sh`, `README.md`.
  4. 文档：`docs/experiment_design.md`, 更新 `WORK_BOARD.md`.
  5. 训练性能优化：摩擦层窗口邻接共享批量化（`fif_mvp/models/friction_layer.py`, `fif_mvp/utils/sparse.py`, `fif_mvp/train/energy.py`）并移除 Hybrid 中冗余的 fallback 能量计算（`fif_mvp/models/hybrid_model.py`），使 v1.0.0-B 运行不再受 6s/step 的 Python 循环瓶颈限制。

## 版本 1.0.1
- **版本号**：1.0.1
- **更新时间**：2025-11-12
- **迭代来源**：基于 1.0.0 继续完善训练基础设施。
- **版本目标**：
  - 消除多 GPU DataParallel 的标量 gather 警告与 CPU 同步开销，让能量正则仍保持梯度。
  - 内置单机 DDP launcher，并让脚本自动切换到 DDP，避免手动 `torchrun` 的易错流程。
- **公式与管线**：沿用 1.0.0 的建模与损失，仅在训练管线中改写能量正则（改为使用张量平均）并新增 DDP 启动逻辑。
- **实验记录**（SNLI，seed=42；详见 `docs/reports/v1_0_1_snli.md`）：

| Model | energy_reg_weight | acc | macro_f1 | loss | ece | energy_mean | energy_log_mean | pearson_r(E, err) | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.0 | 0.7767 | 0.7749 | 0.5521 | 0.0288 | 431.57 | 5.81 | -0.097 | DDP 训练 44 分钟，性能接近 v1.0.0 |
| Hybrid | 1e-4 | 0.6967 | 0.6951 | 0.6978 | 0.0166 | 0.070 | 0.068 | +0.057 | 能量被正则压至近 0，仍落后基线 8pt |

  - 新的 per-sample 能量聚合消除了 DataParallel 警告，但 Hybrid 的能量在 λ=1e-4 下崩塌，需要重新调节 λ/K/η 以恢复能量-错误关联。
- **修改与改进点**：
  1. `fif_mvp/models/*`, `fif_mvp/train/loop.py`：移除 `batch_energy` 标量输出，改为从 `per_sample_energy` 聚合，彻底消除 DataParallel 警告。
  2. `fif_mvp/cli/run_experiment.py`：新增 `--ddp/--nproc_per_node` 单机 mp.spawn launcher，并修正 DDP 场景下的目录创建。
  3. `scripts/*.sh`：检测 `torch.cuda.device_count()` 自动附加 `--ddp`。
  4. 文档同步：`README.md`, `docs/experiment_design.md`, 更新 `WORK_BOARD.md` 以记录任务闭环。
  5. 能量正则可调化：新增 `--energy_reg_scope/--energy_reg_mode`、记录 `energy_std`/`energy_p90`（`fif_mvp/cli/run_experiment.py`, `fif_mvp/train/loop.py`, `fif_mvp/models/hybrid_model.py`, `docs/reports/*`），为 v1.0.0-B 之后的 λ 扫描提供工具。

## 版本 1.0.2
- **版本号**：1.0.2
- **更新时间**：2025-11-14
- **迭代来源**：针对稳健性与可复现性的小版本修复与清理。
- **版本目标**：
  - 统一模型输出形态，简化训练循环逻辑；
  - 修复 DDP 环境下的 GPU→CPU 回退策略，避免多进程分歧；
  - 将 Baseline 的窗口能量计算向量化，降低 Python 循环开销；
  - 完整暴露 Friction 超参到 CLI，便于实验扫描；
  - 明确并默认启用确定性训练；清理小冗余。
- **公式与管线**：沿用 1.0.1，未更改建模与损失公式；仅优化训练/CLI 基础设施与能量实现。
- **修改与改进点**：
  1. 输出统一：`TransformerClassifier`/`HybridClassifier` 均返回 `ModelOutput`（`fif_mvp/models/*`），`Trainer` 仅在 DataParallel tuple 情况做解包。
  2. DDP 回退：在 DDP 下禁用自动 CPU 回退（`fif_mvp/cli/run_experiment.py: run_with_device`）。
  3. 能量向量化：`train/energy.sequence_energy` 按长度分桶，使用一次 window 图 + 批量能量（`edge_energy_batch`）。
  4. CLI 扩展：新增 `--friction.{eta_decay,mu_max,smooth_lambda,normalize_laplacian/no_normalize_laplacian,recompute_mu/no_recompute_mu}`（`fif_mvp/cli/run_experiment.py`）。
  5. 确定性：默认 `--deterministic` 开启，允许 `--no_deterministic` 显式关闭；移除相互矛盾的二次设置（`README.md`/`run_experiment.py`）。
  6. 清理：移除重复 import（`data/__init__.py`）、简化噪声字符集（`data/noise.py`）、去除未使用依赖 `torchvision`（`requirements.txt`）。
  7. 采样优化（可选）：新增 `--sortish_batches`（非 DDP）在训练集上进行长度近似分桶采样，减少 padding（`fif_mvp/data/*`, `README.md`）。

## 版本 1.0.3
- **版本号**：1.0.3
- **更新时间**：2025-11-16
- **迭代来源**：基于 1.0.2 结果报告与《`docs/reports/v1_0_2_rebuild_upgrade.md`》执行的新一轮实验。
- **版本目标**：
  1. 通过 `scope=last + mode=normalized`、`energy_guard/std=0.1` 与 `energy_watch(std=0.1,p90=0.5)` 恢复 Hybrid 能量可辨识度。
  2. 让监控流程在能量贴地时提供可追溯告警，并写入 `alerts.json`。
  3. 收集 SNLI + Noisy SST-2（low/med/high）在该配置下的对照指标，为后续 K 预热与 kNN 重构提供基线。
- **公式与管线（当前状态）**：
  - **能量正则**：CL I 默认 `scope=last`、`mode=normalized`、λ=1e-4，并允许 `--energy_guard/--energy_watch` 配置；训练循环聚合 `energy_alert` 并持久化 `alerts.json`（`fif_mvp/cli/run_experiment.py`, `fif_mvp/train/loop.py`）。
  - **摩擦层**：仍沿用 v1.0.2（`recompute_mu=False`、K=3），尚未落实预热或 kNN 重构——本轮结果证实吞吐/能量问题依旧（详见报告 §4）。
  - **监控产物**：`metrics_epoch.csv`/`energy_epoch.csv` 追加 `energy_alert` 与 `energy_reg_weight` 列；当 watch/guard 触发时写入 `alerts.json`。
- **实验记录**：
  - SNLI：Hybrid 仍落后 Baseline 8pt（0.697 vs 0.777），能量均值/Std=6.3k/1.5k，pearson_r≈+0.02，训练 8700 s；详见 `docs/reports/v1_0_3_results.md` §2（引用 `result/103/snli_*`）。
  - SST-2 noisy：Hybrid 在 low 噪声落后 1.5pt，但在 med/high 领先 2.7/2.2pt；能量均值约 1–2，watch 在 low/med 的 epoch 1 因 `p90<0.5` 报警（`alerts.json`），pearson_r 仍接近 0（`docs/reports/v1_0_3_results.md` §3-4）。
- **修改与改进点（后续）**：
  1. CLI/脚本已切到新默认；下一步需要在 `fif_mvp/models/*` 中加入 K 预热、`recompute_mu=True` 以及 per-sample kNN。
  2. 在 guard 中加入能量上限或标准差上/下界逻辑，让 SNLI Hybrid 的高能量也能触发 λ 调整。
  3. 基于本轮 `timing.json`（Hybrid 平均 0.80 s/step）聚焦 `_run_knn_batch` 重构，以实现路线图中“<3k 秒” 的目标。

## 版本 1.0.4（进行中）
- **版本号**：1.0.4
- **更新时间**：2025-12-02
- **迭代来源**：基于 v1.0.3 的性能瓶颈与文档可维护性需求。
- **版本目标**：
  1. 清理死代码 / 过时辅助函数，明确核心模块边界，降低新成员理解门槛；
  2. 重写批量 kNN 构图为张量化实现，缓解 SNLI Hybrid 的 `_run_knn_batch` Python 热点；
  3. 为开发 / 故障排查提供结构化说明，记录重构计划（`docs/code_structure.md`）。
- **公式与管线（变更）**：
  - **kNN 构图**：`build_knn_edges_batched` 由 `O(B⋅L⋅k)` Python 集合遍历改为一次 `B×L×L` 相似度矩阵 + `topk` 筛选，仍输出无向边 `(batch, i, j)`，对 `FrictionLayer` 的能量/梯度保持兼容。
  - **FrictionLayer**：移除未被调用的 `_build_edges` 接口，避免二义性；其它数学形式沿用 v1.0.3。
- **实验记录**：本次提交仅完成代码整理 + 算子重写，尚未重新跑 SNLI/SST-2；待 T-021 执行完整复现实验后补充 acc/ECE/能量对照。
- **修改与改进点**：
  1. `fif_mvp/utils/sparse.py`：kNN 批量构图向量化（mask + `torch.topk` + `torch.unique`），删除 Python 集合瓶颈；
  2. `fif_mvp/models/friction_layer.py`：去除遗留 `_build_edges`，收敛公开 API；
  3. `docs/code_structure.md`：新增代码组织与清理路线图，为 v1.0.4 提供规范说明；
  4. 工作板补充 v1.0.4 任务，方便跟踪重构 / 性能验证。
  5. 脚本优化：`scripts/snli_baseline.sh` 与 `scripts/snli_hybrid.sh` 默认启用 `--sortish_batches` 减少 padding；新增 `scripts/snli_hybrid_fixedmu.sh` 以 `--friction.no_recompute_mu` 作为对照，便于吞吐与精度权衡实验；SST-2 相同优化同步至 `scripts/sst2_noisy_baseline.sh`、`scripts/sst2_noisy_hybrid.sh` 并新增 `scripts/sst2_noisy_hybrid_fixedmu.sh`；全部脚本统一 `--save_dir ./result/1_0_4`。
  6. 文档与验真：新增 `docs/reports/v1_0_4_experiment_report.md`（报告模板与验证流程）与 `scripts/summarize_results.py`（结果聚合）。
