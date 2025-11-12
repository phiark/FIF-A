#!/usr/bin/env bash
set -euo pipefail

SEED=42
INTENSITIES=(low med high)

for intensity in "${INTENSITIES[@]}"; do
  python -m fif_mvp.cli.run_experiment \
    --task sst2_noisy \
    --model hybrid \
    --epochs 5 \
    --batch_size 256 \
    --lr 3e-4 \
    --seed "${SEED}" \
    --energy_reg_weight 1e-4 \
    --noise_intensity "${intensity}" \
    --friction.K 1 \
    --friction.radius 2 \
    --friction.neighbor window \
    --workers 4 \
    --save_dir ./result
done
