"""Main experiment entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import datasets
import torch
import transformers
from transformers import AutoTokenizer

from fif_mvp.config import ExperimentConfig, FrictionConfig, OptimizationConfig
from fif_mvp.data import build_dataloaders
from fif_mvp.models import build_model
from fif_mvp.train import run_training
from fif_mvp.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FIF MVP experiment runner")
    parser.add_argument("--task", choices=["snli", "sst2", "sst2_noisy"], required=True)
    parser.add_argument(
        "--model", choices=["baseline", "hybrid", "full_friction"], required=True
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--ff", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    parser.add_argument("--friction.K", dest="friction_K", type=int, default=3)
    parser.add_argument("--friction.eta", dest="friction_eta", type=float, default=0.3)
    parser.add_argument(
        "--friction.neighbor",
        dest="friction_neighbor",
        choices=["window", "knn"],
        default="window",
    )
    parser.add_argument(
        "--friction.radius", dest="friction_radius", type=int, default=4
    )
    parser.add_argument("--friction.k", dest="friction_k", type=int, default=8)
    parser.add_argument(
        "--noise_intensity", choices=["low", "med", "high"], default=None
    )
    parser.add_argument(
        "--train_noise_levels",
        type=str,
        default="clean,low,med,high",
        help="Comma separated noise levels to apply to the training split when task=sst2_noisy.",
    )
    parser.add_argument("--dump_energy_per_sample", action="store_true")
    parser.add_argument(
        "--energy_reg_weight",
        type=float,
        default=0.0,
        help="Weight for log-energy regularization added to the training loss.",
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Must reside under ./result"
    )
    return parser.parse_args()


@dataclass
class DeviceChoice:
    device: str
    description: str
    brand: str
    backend: str


def choose_device() -> DeviceChoice:
    """Pick the best available accelerator."""

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vendor = "AMD" if "amd" in name.lower() else "NVIDIA"
        backend = "ROCm" if vendor == "AMD" else "CUDA"
        desc = f"{vendor} GPU ({name}) via {backend}"
        return DeviceChoice(
            device="cuda", description=desc, brand=vendor, backend=backend
        )

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return DeviceChoice(
            device="mps", description="Apple GPU via MPS", brand="Apple", backend="MPS"
        )

    return DeviceChoice(
        device="cpu", description="CPU execution", brand="CPU", backend="CPU"
    )


def verify_device(choice: DeviceChoice) -> Tuple[bool, str]:
    """Attempt to allocate a tensor on the selected device."""

    if choice.device == "cpu":
        return True, ""
    try:
        _ = torch.zeros(1, device=choice.device)
        if choice.device == "cuda":
            torch.cuda.synchronize()
        return True, ""
    except Exception as exc:  # pragma: no cover - hardware dependent
        return False, str(exc)


def emit_warning(message: str) -> None:
    banner = "=" * 80
    print(f"\n{banner}\nWARNING: {message}\n{banner}\n")


def main() -> None:
    args = parse_args()
    train_noise_levels = [
        level.strip() for level in args.train_noise_levels.split(",") if level.strip()
    ] or ["clean"]
    device_choice = choose_device()
    print(f"[Device] Preferred backend: {device_choice.description}")
    ready, error_msg = verify_device(device_choice)
    if not ready:
        emit_warning(
            f"Failed to initialize {device_choice.description}. Error: {error_msg}. "
            "Falling back to CPU execution."
        )
        device_choice = DeviceChoice(
            device="cpu", description="CPU execution", brand="CPU", backend="CPU"
        )
    base_result = Path(args.save_dir).expanduser().resolve()
    expected_root = (Path.cwd() / "result").resolve()
    if expected_root not in base_result.parents and base_result != expected_root:
        raise ValueError("save_dir must be within ./result")
    base_result.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name_parts = [args.task]
    if args.task == "sst2_noisy" and args.noise_intensity:
        run_name_parts.append(args.noise_intensity)
    run_name_parts.append(args.model)
    run_name = "_".join(run_name_parts) + f"_{timestamp}_seed{args.seed}"
    run_dir = base_result / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    friction = FrictionConfig(
        K=args.friction_K,
        eta=args.friction_eta,
        neighbor=args.friction_neighbor,
        radius=args.friction_radius,
        k=args.friction_k,
    )
    optim = OptimizationConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
    )
    config = ExperimentConfig(
        task=args.task,
        model_type=args.model,
        hidden_size=args.hidden,
        ff_size=args.ff,
        num_heads=args.heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        device=device_choice.device,
        tokenizer_name=args.tokenizer,
        friction=friction,
        optimization=optim,
        dump_energy_per_sample=args.dump_energy_per_sample,
        noise_intensity=args.noise_intensity,
        train_noise_levels=train_noise_levels,
        noise_vocab=train_noise_levels,
        energy_reg_weight=args.energy_reg_weight,
    )

    set_seed(config.seed)
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    except Exception as exc:  # pragma: no cover - network failure
        raise RuntimeError(
            f"Unable to load tokenizer '{config.tokenizer_name}'. Ensure it is cached or internet is available."
        ) from exc
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    config.vocab_size = len(tokenizer)
    config.tokenizer_name = args.tokenizer

    data_bundle = build_dataloaders(
        task=config.task,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_seq_len,
        seed=config.seed,
        noise_intensity=config.noise_intensity,
        train_noise_levels=train_noise_levels,
    )
    config.num_labels = data_bundle.num_labels
    config.noise_vocab = data_bundle.noise_vocab

    save_config(run_dir / "config.json", config.to_dict())
    write_env(run_dir / "env.txt", device_choice)
    if data_bundle.noise_config:
        save_json(run_dir / "noise_config.json", data_bundle.noise_config)

    run_with_device(
        config=config,
        tokenizer=tokenizer,
        data_bundle=data_bundle,
        run_dir=run_dir,
        device_choice=device_choice,
    )


def run_with_device(
    config: ExperimentConfig,
    tokenizer,
    data_bundle,
    run_dir: Path,
    device_choice: DeviceChoice,
) -> None:
    """Run training with automatic CPU fallback if GPU execution fails."""

    def build() -> torch.nn.Module:
        return build_model(
            config=config,
            vocab_size=config.vocab_size,
            num_labels=config.num_labels,
            pad_token_id=tokenizer.pad_token_id,
        )

    model = build()
    try:
        run_training(
            config=config, model=model, loaders=data_bundle.loaders, save_dir=run_dir
        )
    except RuntimeError as exc:
        if device_choice.device == "cpu":
            raise
        emit_warning(
            f"RuntimeError encountered on {device_choice.description}: {exc}. Switching to CPU for a retry."
        )
        config.device = "cpu"
        cpu_choice = DeviceChoice(
            device="cpu", description="CPU execution", brand="CPU", backend="CPU"
        )
        write_env(run_dir / "env.txt", cpu_choice)
        model = build()
        run_training(
            config=config, model=model, loaders=data_bundle.loaders, save_dir=run_dir
        )


def save_config(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_env(path: Path, device_choice: DeviceChoice) -> None:
    lines = [
        f"python: {sys.version}",
        f"torch: {torch.__version__}",
        f"cuda_available: {torch.cuda.is_available()}",
        f"device: {device_choice.device}",
        f"device_detail: {device_choice.description}",
        f"datasets: {datasets.__version__}",
        f"transformers: {transformers.__version__}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
