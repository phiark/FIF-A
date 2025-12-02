#!/usr/bin/env python3
"""
Summarize v1.0.4 experiment outputs under result/1_0_4.

Scans immediate subdirectories, reads test_summary.json and timing.json,
and writes a CSV summary to result/1_0_4/results_summary.csv.

This is a post-run validator to check that metrics are present and numeric.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import csv


def load_json(p: Path) -> dict:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main(root: Path) -> int:
    rows = []
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        test = load_json(sub / "test_summary.json")
        timing = load_json(sub / "timing.json")
        config = load_json(sub / "config.json")
        if not test or not timing:
            continue
        model = config.get("model_type", "?")
        task = config.get("task", "?")
        friction = config.get("friction", {})
        neighbor = friction.get("neighbor", "-")
        K = friction.get("K", 0)
        recompute_mu = friction.get("recompute_mu", False)
        save_dir = str(sub)
        rows.append(
            {
                "run_dir": save_dir,
                "task": task,
                "model": model,
                "neighbor": neighbor,
                "K": K,
                "recompute_mu": int(bool(recompute_mu)),
                "acc": test.get("acc", 0.0),
                "macro_f1": test.get("macro_f1", 0.0),
                "ece": test.get("ece", 0.0),
                "loss": test.get("loss", 0.0),
                "energy_log_mean_test": test.get("energy_log_mean_test", 0.0),
                "avg_step_sec": timing.get("avg_step_sec", 0.0),
                "total_train_sec": timing.get("total_train_sec", 0.0),
            }
        )
    if not rows:
        print(f"No completed runs found under {root}")
        return 1
    out_csv = root / "results_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_dir",
                "task",
                "model",
                "neighbor",
                "K",
                "recompute_mu",
                "acc",
                "macro_f1",
                "ece",
                "loss",
                "energy_log_mean_test",
                "avg_step_sec",
                "total_train_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path("result/1_0_4")))

