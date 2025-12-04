#!/usr/bin/env bash
set -euo pipefail

SEED=42

GPU_COUNT=0
if command -v python >/dev/null 2>&1; then
  if output=$(python - <<'PY' 2>/dev/null
import torch
print(torch.cuda.device_count())
PY
  ); then
    GPU_COUNT=$(echo "$output" | tr -d '[:space:]')
  fi
fi
DDP_ARGS=()
if [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] && (( GPU_COUNT > 1 )); then
  DDP_ARGS=(--ddp --nproc_per_node="${GPU_COUNT}")
fi

python -m fif_mvp.cli.run_experiment \
  "${DDP_ARGS[@]}" \
  --task snli \
  --model hybrid \
  --epochs 3 \
  --batch_size 256 \
  --lr 3e-4 \
  --seed "${SEED}" \
  --friction.K 1 \
  --friction.radius 2 \
  --friction.neighbor window \
  --friction.eta_decay 0.0 \
  --sortish_batches \
  --no_deterministic \
  --energy_reg_weight 1e-4 \
  --energy_reg_scope last \
  --energy_reg_target absolute \
  --energy_guard std_low=0.05,std_high=6,p90_high=8,factor=0.5,up=1.2,min_weight=1e-5,max=1e-3 \
  --energy_watch std=0.05,std_high=8,p90=0.5,p90_high=10,mean_low=0.05 \
  --workers -1 \
  --save_dir ./result/1_1_0
