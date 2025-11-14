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
