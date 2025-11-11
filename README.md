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

# Noisy SST-2 (train on clean+noisy mix, eval per intensity)
./scripts/sst2_noisy_baseline.sh    # runs low/med/high
./scripts/sst2_noisy_hybrid.sh      # runs low/med/high

# Full sweep
./run.sh
```

`run.sh` executes all four experiments sequentially and prints a one-line success message containing the absolute `result/` path when everything completes without error. The CLI auto-detects the best accelerator in priority order (CUDA/NVIDIA or AMD ROCm, Apple MPS, then CPU) and emits a large warning if it must fall back to CPU because a GPU backend is unavailable or fails to initialize.

Key CLI knobs:

- `--train_noise_levels clean,low,med,high` controls which noise intensities are duplicated in the training split (default mixes全部四档)；
- `--energy_reg_weight 1e-4` 在训练损失中加入 `log(1+E)` 正则；
- existing `--noise_intensity {low,med,high}` 选择验证/测试噪声强度。

## Result artifacts

Every run subdirectory contains:

* `config.json`, `env.txt`, `timing.json`
* `train_log.txt`, `metrics_epoch.csv`, `energy_epoch.csv`（新增 `energy_log_mean` 列）
* `test_summary.json` with `acc`, `macro_f1`, `loss`, `ece`, `energy_mean_test`, `energy_log_mean_test`
* `confusion_matrix.csv`
* `energy_error_correlation.json`
* (Noisy SST-2 only) `noise_config.json`

Optional per-sample energy dumps live in `energy_per_sample.csv` when explicitly enabled.

## Reproducibility

The code fixes seeds (Python, NumPy, PyTorch) via `utils.seed.set_seed`, enforces deterministic cuDNN behavior, and records package/device metadata in `env.txt`. Small models (hidden size 256, four layers) keep runtime reasonable on CPUs or commodity GPUs. If dataset downloads fail, the CLI raises a descriptive error so you can pre-download via `datasets` cache.

## Project tracking

- `PROJECT_TRACKER.md`：版本沿革
- `WORK_BOARD.md`：当前任务及状态
- `docs/experiment_design.md`：v1.0.0 实验方案、指标与需要产出的图表
