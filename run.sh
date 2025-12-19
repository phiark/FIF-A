#!/usr/bin/env bash
# Run all regular SNLI and SST-2 experiments
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo "FIF-A: Running v1.2.0 experiments (single GPU)"
echo "  - SNLI Baseline"
echo "  - SNLI Hybrid"
echo "  - SST-2 Baseline"
echo "  - SST-2 Hybrid"
echo "========================================="
echo ""

python "$SCRIPT_DIR/scripts/run_experiments.py" --config "$SCRIPT_DIR/scripts/experiments.yaml"

RESULT_DIR="$SCRIPT_DIR/result/1_2_0"
echo ""
echo "========================================="
echo "All experiments finished successfully!"
echo "Results at: $RESULT_DIR"
echo "========================================="
