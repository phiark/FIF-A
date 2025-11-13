# pyright: reportGeneralTypeIssues=false
"""Main experiment entrypoint."""

from __future__ import annotations

import argparse
import os as _os

_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
import sys
import time
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
    # Acceleration toggles
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision (enabled by default on NVIDIA).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile to optimize the model (PyTorch 2.x).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1 = auto for CUDA, 0 = main thread).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic algorithms (slower). By default disabled for speed.",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Enable DistributedDataParallel via torchrun (single-node).",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=-1,
        help="Number of CUDA processes to launch when using --ddp. Defaults to all visible GPUs.",
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


def _run_cli(args: argparse.Namespace) -> None:
    # Avoid tokenizer multiprocessing + DataLoader workers deadlock warnings
    import os as _os

    _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    train_noise_levels = [
        level.strip() for level in args.train_noise_levels.split(",") if level.strip()
    ] or ["clean"]
    import os

    is_ddp = bool(os.environ.get("LOCAL_RANK") is not None)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
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
    if is_ddp:
        if local_rank == 0:
            run_dir.mkdir(parents=True, exist_ok=False)
        else:
            wait_sec = 0.0
            while not run_dir.exists():
                time.sleep(0.05)
                wait_sec += 0.05
                if wait_sec > 30.0:
                    raise RuntimeError(
                        "Timed out waiting for run directory creation from rank 0."
                    )
    else:
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
        use_amp=not args.no_amp,
        compile_model=args.compile,
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

    # Tune CUDA backends when using NVIDIA GPUs
    if device_choice.device == "cuda":
        try:
            # New-style TF32/matmul controls (avoid deprecated flags)
            if torch.cuda.is_available():
                major, _ = torch.cuda.get_device_capability(0)
                if hasattr(torch.backends.cuda, "matmul") and hasattr(
                    torch.backends.cuda.matmul, "fp32_precision"
                ):
                    torch.backends.cuda.matmul.fp32_precision = (
                        "tf32" if major >= 8 else "ieee"
                    )
                if (
                    hasattr(torch.backends, "cudnn")
                    and hasattr(torch.backends.cudnn, "conv")
                    and hasattr(torch.backends.cudnn.conv, "fp32_precision")
                ):
                    torch.backends.cudnn.conv.fp32_precision = (
                        "tf32" if major >= 8 else "ieee"
                    )
            torch.backends.cudnn.benchmark = True
            # Only set float32 matmul precision on Ampere+ to avoid warnings on V100
            if hasattr(torch, "set_float32_matmul_precision") and major >= 8:
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Determinism: default off for performance unless explicitly requested
    try:
        torch.use_deterministic_algorithms(
            args.deterministic, warn_only=not args.deterministic
        )
    except Exception:
        pass

    # DDP distributed flags
    world_size = (
        int(os.environ.get("WORLD_SIZE", "1")) if "WORLD_SIZE" in os.environ else 1
    )
    rank_env = int(os.environ.get("RANK", "0")) if "RANK" in os.environ else 0

    data_bundle = build_dataloaders(
        task=config.task,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_seq_len,
        seed=config.seed,
        noise_intensity=config.noise_intensity,
        train_noise_levels=train_noise_levels,
        workers=(None if args.workers < 0 else args.workers),
        distributed=(world_size > 1),
        world_size=world_size,
        rank=rank_env,
    )
    config.num_labels = data_bundle.num_labels
    config.noise_vocab = data_bundle.noise_vocab

    if (not is_ddp) or int(os.environ.get("LOCAL_RANK", 0)) == 0:
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


def _distributed_worker(
    local_rank: int, world_size: int, args: argparse.Namespace
) -> None:
    import os

    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    _run_cli(args)


def _launch_ddp(args: argparse.Namespace) -> None:
    import os

    if not torch.cuda.is_available():
        emit_warning(
            "--ddp requested but CUDA is unavailable. Running single-process instead."
        )
        _run_cli(args)
        return
    world_size = (
        args.nproc_per_node if args.nproc_per_node > 0 else torch.cuda.device_count()
    )
    if world_size <= 1:
        emit_warning(
            "--ddp requested but only one CUDA device detected. Running without DDP."
        )
        _run_cli(args)
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    import torch.multiprocessing as mp

    mp.spawn(_distributed_worker, nprocs=world_size, args=(world_size, args))


def main() -> None:
    args = parse_args()
    if args.ddp and "LOCAL_RANK" not in os.environ:
        _launch_ddp(args)
        return
    _run_cli(args)


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

    import os

    is_ddp = bool(os.environ.get("LOCAL_RANK") is not None)
    if is_ddp:
        # Initialize process group
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        config.distributed = True
        config.rank = global_rank
        config.world_size = world_size
        config.device = f"cuda:{local_rank}"

    model = build()
    if config.compile_model and hasattr(torch, "compile") and not is_ddp:
        try:
            model = torch.compile(model, mode="default", fullgraph=False)
            print("[Compile] Model compiled with torch.compile")
        except Exception as exc:  # pragma: no cover - backend dependent
            emit_warning(
                f"torch.compile failed: {exc}. Proceeding without compilation."
            )

    # Simple multi-GPU via DataParallel for CUDA (non-DDP)
    if (
        device_choice.device == "cuda" and torch.cuda.device_count() > 1
    ) and not is_ddp:
        print(f"[Multi-GPU] Using DataParallel on {torch.cuda.device_count()} GPUs")
        from fif_mvp.models import DataParallelFriendly

        model = torch.nn.DataParallel(DataParallelFriendly(model))
    # DDP wrapping
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = DDP(
            model.to(config.device),
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
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
        if (not is_ddp) or int(os.environ.get("LOCAL_RANK", 0)) == 0:
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
