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
  --epochs 5 \
  --batch_size 256 \
  --lr 3e-4 \
  --seed "${SEED}" \
  --friction.K 3 \
  --friction.radius 2 \
  --friction.neighbor window \
  --friction.no_recompute_mu \
  --sortish_batches \
  --no_deterministic \
  --energy_reg_weight 1e-4 \
  --energy_reg_scope last \
  --energy_reg_mode normalized \
  --energy_guard std=0.1,factor=0.5,min_weight=1e-5 \
  --energy_watch std=0.1,p90=0.5 \
  --workers -1 \
  --save_dir ./result
