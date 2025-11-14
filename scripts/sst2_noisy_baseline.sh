#!/usr/bin/env bash
set -euo pipefail

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

SEED=42
INTENSITIES=(low med high)

for intensity in "${INTENSITIES[@]}"; do
  python -m fif_mvp.cli.run_experiment \
    "${DDP_ARGS[@]}" \
    --task sst2_noisy \
    --model baseline \
    --epochs 5 \
    --batch_size 256 \
    --lr 3e-4 \
    --seed "${SEED}" \
    --energy_reg_weight 1e-4 \
    --noise_intensity "${intensity}" \
    --no_deterministic \
    --workers -1 \
    --save_dir ./result
done
