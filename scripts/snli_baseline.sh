#!/usr/bin/env bash
set -euo pipefail

SEED=42

python -m fif_mvp.cli.run_experiment \
  --task snli \
  --model baseline \
  --epochs 5 \
  --batch_size 256 \
  --lr 3e-4 \
  --seed "${SEED}" \
  --workers 8 \
  --save_dir ./result
