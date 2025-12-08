#!/usr/bin/env python
"""Unified experiment launcher using a YAML config.

Features:
- Auto-detects backend (CUDA/NVIDIA or AMD ROCm, Apple MPS, CPU) and only enables DDP on multi-GPU CUDA.
- Supports sweep expansion (e.g., noise_intensity over [low, med, high]).
- Keeps CLI parity with existing shell scripts while reducing duplication.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import yaml


@dataclass
class DeviceInfo:
    backend: str  # cuda | mps | cpu
    brand: str
    count: int
    description: str


def detect_device() -> DeviceInfo:
    """Return best-available device info with MPS/AMD awareness."""

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA device"
        brand = "AMD" if "amd" in name.lower() else "NVIDIA"
        count = torch.cuda.device_count()
        return DeviceInfo(
            backend="cuda",
            brand=brand,
            count=count,
            description=f"{brand} GPU ({name}) x{count}",
        )

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return DeviceInfo(
            backend="mps", brand="Apple", count=1, description="Apple GPU via MPS"
        )

    return DeviceInfo(backend="cpu", brand="CPU", count=1, description="CPU")


def expand_params(params: Dict, sweep: Dict | None) -> List[Dict]:
    """Expand sweep dictionary into a list of parameter dicts."""

    if not sweep:
        return [params]
    keys = list(sweep.keys())
    values = [sweep[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        new_params = dict(params)
        for k, v in zip(keys, combo):
            new_params[k] = v
        combos.append(new_params)
    return combos


def build_command(
    params: Dict[str, str | int | float],
    flags: Sequence[str],
    negative_flags: Sequence[str],
    ddp_args: Sequence[str],
) -> List[str]:
    cmd: List[str] = ["python", "-m", "fif_mvp.cli.run_experiment"]
    cmd.extend(ddp_args)
    for flag in flags:
        cmd.append(f"--{flag}")
    for nflag in negative_flags:
        cmd.append(f"--{nflag}")
    for key, val in params.items():
        cmd.append(f"--{key}")
        cmd.append(str(val))
    return cmd


def run_experiments(config_path: Path, select: Sequence[str], dry_run: bool) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    experiments = data.get("experiments", [])
    if not experiments:
        raise ValueError(f"No experiments found in {config_path}")

    device = detect_device()
    print(f"[Device] {device.description}")

    for exp in experiments:
        name = exp.get("name", "unnamed")
        if select and name not in select:
            continue
        params = exp.get("params", {}) or {}
        sweep = exp.get("sweep", {}) or {}
        flags = exp.get("flags", []) or []
        negative_flags = exp.get("negative_flags", []) or []
        ddp_pref = exp.get("ddp", "auto")
        enable_ddp = (
            ddp_pref not in {"off", False}
            and device.backend == "cuda"
            and device.count > 1
        )
        ddp_args: List[str] = []
        if enable_ddp:
            ddp_args = ["--ddp", "--nproc_per_node", str(device.count)]
        elif ddp_pref not in {"auto", "off", False} and device.backend != "cuda":
            print(
                f"[warn] DDP requested for {name} but backend={device.backend}; skipping DDP."
            )

        param_sets = expand_params(params, sweep)
        for param_set in param_sets:
            # Append sweep tag to name for clarity
            suffix_parts = [
                f"{k}={param_set[k]}" for k in sweep.keys() if k in param_set
            ]
            tagged_name = (
                name if not suffix_parts else f"{name}__{'__'.join(suffix_parts)}"
            )
            cmd = build_command(param_set, flags, negative_flags, ddp_args)
            print(f"\n[RUN] {tagged_name}")
            print("      ", " ".join(cmd))
            if dry_run:
                continue
            subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiments from a YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("experiments.yaml"),
        help="Path to experiments YAML file.",
    )
    parser.add_argument(
        "--select",
        type=str,
        nargs="*",
        default=[],
        help="Optional experiment names to run (subset).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    args = parser.parse_args()
    run_experiments(args.config, args.select, args.dry_run)


if __name__ == "__main__":
    main()
