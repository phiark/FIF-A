#!/usr/bin/env bash
set -euo pipefail

SEED=42
GPUS=${GPUS:-1}

torchrun --standalone --nproc_per_node=${GPUS} \
  -m fif_mvp.cli.run_experiment \
  --task snli 
  --model hybrid 
  --epochs 1 
  --batch_size 1024 
  --lr 3e-4 
  --seed "${SEED}" 
  --friction.K 1 
  --friction.radius 2 
  --friction.neighbor window 
  --workers -1 
  --save_dir ./result
