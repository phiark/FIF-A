# FIF MVP

This experimental repo benchmarks whether introducing a Frictional Interaction Field (FIF, 信息摩擦层) into a lightweight Transformer improves robustness and whether the induced energy signal (`E = 0.5 Σ μ_ij ||h_i - h_j||^2`) correlates with prediction errors. Since v1.0.0 we:

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

Each script stores outputs under `./result/` in timestamped folders.

```bash
# SNLI baseline vs friction-hybrid
./scripts/snli_baseline.sh
./scripts/snli_hybrid.sh
./scripts/snli_hybrid_k1_absolute.sh   # K=1 + absolute λ=1e-4 sanity (saves to ./result/1_1_0)

# Noisy SST-2 (train on clean+noisy mix, eval per intensity)
./scripts/sst2_noisy_baseline.sh    # runs low/med/high
./scripts/sst2_noisy_hybrid.sh      # runs low/med/high

# Full sweep
./run.sh
```

`run.sh` executes all four experiments sequentially and prints a one-line success message containing the absolute `result/` path when everything completes without error. The CLI auto-detects the best accelerator in priority order (CUDA/NVIDIA or AMD ROCm, Apple MPS, then CPU) and emits a large warning if it must fall back to CPU because a GPU backend is unavailable or fails to initialize.

### Multi-GPU (DDP) runs

- Pass `--ddp` to `python -m fif_mvp.cli.run_experiment ...` to launch single-node DistributedDataParallel jobs without manually calling `torchrun`. Use `--nproc_per_node <num_gpus>` to override the default of “all visible CUDA devices”.
- Existing helper scripts now auto-detect `torch.cuda.device_count()` and append `--ddp --nproc_per_node=<count>` whenever more than one GPU is visible, so `./run.sh` scales out-of-the-box on multi-GPU servers.
- If you prefer to manage processes yourself, `torchrun --nproc_per_node=2 python -m fif_mvp.cli.run_experiment ...` still works; the CLI recognizes the `LOCAL_RANK/RANK/WORLD_SIZE` environment variables and skips the built-in launcher.
- DataParallel remains available for legacy workflows, but DDP eliminates the scalar gather warning (`Was asked to gather along dimension 0...`) and keeps computation on the GPUs.

Key CLI knobs:

- `--train_noise_levels clean,low,med,high` controls which noise intensities are duplicated in the training split (default mixes全部四档)；
- `--energy_reg_weight 1e-4` 在训练损失中加入 `log(1+E)` 正则；
- `--energy_reg_scope {all,last}` 控制能量正则施加在全部摩擦层能量之和还是仅最后一层（默认 `last`）；
- `--energy_reg_target {absolute,normalized,margin,rank}`：`absolute` 直接惩罚 `log1p(E)`（默认），`normalized` 惩罚 batch 内 `log1p(E)` 方差，`margin`/`rank` 使能量与分类难度对齐；`--energy_reg_mode` 已废弃，仍保留向后兼容。
- `--energy_guard std_low=0.1,std_high=6,p90_high=8,factor=0.5,up=1.2,min_weight=1e-5,max=1e-3` 在训练过程中监控能量上下界并自动下/上调 λ（可用 `--energy_guard off` 禁用）；
- `--energy_watch std=0.1,std_high=10,p90=0.5,p90_high=8,mean_low=0.1` 在训练/验证/测试阶段记录能量告警并写入 `alerts.json`（可用 `--energy_watch off` 关闭）。
- existing `--noise_intensity {low,med,high}` 选择验证/测试噪声强度。
- Friction knobs: `--friction.eta_decay`, `--friction.mu_max`, `--friction.smooth_lambda`,
  `--friction.{normalize_laplacian,no_normalize_laplacian}`, `--friction.{recompute_mu,no_recompute_mu}`。
 - Data sampling: `--sortish_batches`（非 DDP）与 `--sortish_chunk_mult` 控制长度近似分桶，减少 padding 开销。

## Result artifacts

Every run subdirectory contains:

* `config.json`, `env.txt`, `timing.json`
* `train_log.txt`, `metrics_epoch.csv`, `energy_epoch.csv`（记录 `energy_log_mean/energy_std/p90`、`energy_alert` 和活跃的 λ）
* `test_summary.json` with `acc`, `macro_f1`, `loss`, `ece`, `energy_mean_test`, `energy_log_mean_test`, and energy‑error metrics (`energy_auroc/auprc`, `coverage_aurc`, `coverage_risk_at_{80,90,95}`, energy 分位)
* `confusion_matrix.csv`
* `energy_error_correlation.json`（包含 `pearson_r/auroc/auprc/aurc`、coverage‑risk 曲线子采样、正确/错误能量分位）
* `alerts.json`（当 `--energy_watch` 或 `--energy_guard` 触发事件时生成，列出原因与 λ 回退）
* (Noisy SST-2 only) `noise_config.json`

Optional per-sample energy dumps live in `energy_per_sample.csv` when explicitly enabled.

## Reproducibility

Determinism defaults to ON. We seed Python/NumPy/PyTorch via `utils.seed.set_seed` and enable deterministic algorithms/cudnn. You can opt out with `--no_deterministic` to favor throughput (non-reproducible). Package/device metadata is recorded in `env.txt`. If dataset downloads fail, the CLI raises a descriptive error so you can pre-download via `datasets` cache.

Note on DDP: automatic GPU→CPU fallback is disabled for DDP jobs to avoid multi-process divergence. In DDP, failures are surfaced for an explicit rerun on CPU if needed.

## Project tracking

- `PROJECT_TRACKER.md`：版本沿革
- `WORK_BOARD.md`：当前任务及状态
- `docs/experiment_design.md`：实验方案与指标（含 v1.1.0 能量重构要点）
