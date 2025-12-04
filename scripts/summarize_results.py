#!/usr/bin/env python3
"""
Summarize experiment outputs under the given result root (default: result/1_1_0).

Scans immediate subdirectories, reads test_summary.json / timing.json /
energy_error_correlation.json (if present), and writes a CSV summary to
<root>/results_summary.csv. This is a post-run validator to check that metrics are
present and numeric.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


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
        corr = load_json(sub / "energy_error_correlation.json")
        if not test or not timing:
            continue
        model = config.get("model_type", "?")
        task = config.get("task", "?")
        friction = config.get("friction", {})
        neighbor = friction.get("neighbor", "-")
        K = friction.get("K", 0)
        recompute_mu = friction.get("recompute_mu", False)
        eta_decay = friction.get("eta_decay", None)
        energy_reg_target = config.get(
            "energy_reg_target", config.get("energy_reg_mode", "?")
        )
        energy_reg_weight = config.get("energy_reg_weight", 0.0)
        save_dir = str(sub)
        rows.append(
            {
                "run_dir": save_dir,
                "task": task,
                "model": model,
                "neighbor": neighbor,
                "K": K,
                "recompute_mu": int(bool(recompute_mu)),
                "eta_decay": eta_decay,
                "energy_reg_target": energy_reg_target,
                "energy_reg_weight": energy_reg_weight,
                "acc": test.get("acc", 0.0),
                "macro_f1": test.get("macro_f1", 0.0),
                "ece": test.get("ece", 0.0),
                "loss": test.get("loss", 0.0),
                "energy_log_mean_test": test.get("energy_log_mean_test", 0.0),
                "energy_auroc": test.get("energy_auroc", corr.get("auroc", 0.0)),
                "energy_auprc": test.get("energy_auprc", corr.get("auprc", 0.0)),
                "coverage_aurc": test.get("coverage_aurc", corr.get("aurc", 0.0)),
                "coverage_risk_at_80": test.get(
                    "coverage_risk_at_80",
                    corr.get("coverage_risk_at", {}).get("0.8", 0.0),
                ),
                "coverage_risk_at_90": test.get(
                    "coverage_risk_at_90",
                    corr.get("coverage_risk_at", {}).get("0.9", 0.0),
                ),
                "coverage_risk_at_95": test.get(
                    "coverage_risk_at_95",
                    corr.get("coverage_risk_at", {}).get("0.95", 0.0),
                ),
                "pearson_r": corr.get("pearson_r", 0.0),
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
                "eta_decay",
                "energy_reg_target",
                "energy_reg_weight",
                "acc",
                "macro_f1",
                "ece",
                "loss",
                "energy_log_mean_test",
                "energy_auroc",
                "energy_auprc",
                "coverage_aurc",
                "coverage_risk_at_80",
                "coverage_risk_at_90",
                "coverage_risk_at_95",
                "pearson_r",
                "avg_step_sec",
                "total_train_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    root_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("result/1_1_0")
    sys.exit(main(root_arg))
