#!/usr/bin/env python
"""Wrapper script to run experiments with correct PYTHONPATH."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run the main function
from fif_mvp.cli.run_experiment import main

if __name__ == "__main__":
    main()
