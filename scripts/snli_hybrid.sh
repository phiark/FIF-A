#!/usr/bin/env bash
set -euo pipefail

SEED=42

python -m fif_mvp.cli.run_experiment \
  --task snli \
  --model hybrid \
  --epochs 5 \
  --batch_size 64 \
  --lr 3e-4 \
  --seed "${SEED}" \
  --save_dir ./result
