#!/usr/bin/env bash
# Quick experiment launcher - convenient shortcuts for running experiments
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/scripts/experiments.yaml"

usage() {
    cat <<EOF
Usage: ./quick.sh <target> [--dry-run]

Quick shortcuts for running experiments:
  snli        Run both SNLI experiments (baseline + hybrid)
  sst2        Run both SST-2 experiments (baseline + hybrid)
  baseline    Run both baseline experiments (SNLI + SST-2)
  hybrid      Run both hybrid experiments (SNLI + SST-2)
  all         Run all 4 experiments

Or run specific experiments by name:
  snli_baseline
  snli_hybrid
  sst2_baseline
  sst2_hybrid

Options:
  --dry-run   Show commands without executing

Examples:
  ./quick.sh snli              # Run SNLI baseline and hybrid
  ./quick.sh sst2 --dry-run    # Preview SST-2 commands
  ./quick.sh snli_baseline     # Run only SNLI baseline
  ./quick.sh all               # Run everything

EOF
    exit 1
}

# Check arguments
if [ $# -eq 0 ]; then
    usage
fi

TARGET="$1"
EXTRA_ARGS="${2:-}"

# Map shortcuts to experiment names
case "$TARGET" in
    snli)
        EXPERIMENTS="snli_baseline snli_hybrid"
        ;;
    sst2)
        EXPERIMENTS="sst2_baseline sst2_hybrid"
        ;;
    baseline)
        EXPERIMENTS="snli_baseline sst2_baseline"
        ;;
    hybrid)
        EXPERIMENTS="snli_hybrid sst2_hybrid"
        ;;
    all)
        EXPERIMENTS="snli_baseline snli_hybrid sst2_baseline sst2_hybrid"
        ;;
    snli_baseline|snli_hybrid|sst2_baseline|sst2_hybrid)
        EXPERIMENTS="$TARGET"
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "Error: Unknown target '$TARGET'"
        echo ""
        usage
        ;;
esac

# Print what we're running
echo "========================================="
echo "FIF-A Quick Launcher"
echo "========================================="
echo "Target: $TARGET"
echo "Experiments: $EXPERIMENTS"
if [ "$EXTRA_ARGS" = "--dry-run" ]; then
    echo "Mode: DRY-RUN (preview only)"
fi
echo "========================================="
echo ""

# Execute
if [ "$EXTRA_ARGS" = "--dry-run" ]; then
    python "$SCRIPT_DIR/scripts/run_experiments.py" --config "$CONFIG" --select $EXPERIMENTS --dry-run
else
    python "$SCRIPT_DIR/scripts/run_experiments.py" --config "$CONFIG" --select $EXPERIMENTS
fi

if [ "$EXTRA_ARGS" != "--dry-run" ]; then
    echo ""
    echo "========================================="
    echo "Finished! Results in: $SCRIPT_DIR/result"
    echo "========================================="
fi
