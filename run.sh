#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_step() {
  local script_path="$1"
  echo "[RUN] $script_path"
  bash "$SCRIPT_DIR/$script_path"
}

run_step "scripts/snli_baseline.sh"
run_step "scripts/snli_hybrid.sh"
run_step "scripts/sst2_noisy_baseline.sh"
run_step "scripts/sst2_noisy_hybrid.sh"
run_step "scripts/sst2_noisy_hybrid_k1_absolute.sh"
run_step "scripts/snli_hybrid_k1_absolute.sh"


RESULT_DIR="$SCRIPT_DIR/result"
echo "FIF MVP finished successfully. Results at $RESULT_DIR"
