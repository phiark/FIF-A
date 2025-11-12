#!/usr/bin/env bash
set -euo pipefail

SEED=42

python -m fif_mvp.cli.run_experiment \
  --task snli \
  --model hybrid \
  --epochs 5 \
  --batch_size 256 \
  --lr 3e-4 \
  --seed "${SEED}" \
  --friction.K 1 \
  --friction.radius 2 \
  --friction.neighbor window \
  --workers 4 \
  --save_dir ./result
