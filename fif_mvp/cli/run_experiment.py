# pyright: reportGeneralTypeIssues=false
"""Main experiment entrypoint."""

from __future__ import annotations

import argparse
import os as _os
import warnings

_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import datasets
import torch
import transformers
from transformers import AutoTokenizer

from fif_mvp.config import (
    EnergyGuardConfig,
    EnergyWatchConfig,
    ExperimentConfig,
    FrictionConfig,
    OptimizationConfig,
)
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
        "--friction.eta_decay", dest="friction_eta_decay", type=float, default=0.5
    )
    parser.add_argument(
        "--friction.mu_max", dest="friction_mu_max", type=float, default=5.0
    )
    parser.add_argument(
        "--friction.smooth_lambda",
        dest="friction_smooth_lambda",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--friction.normalize_laplacian",
        dest="friction_normalize_laplacian",
        action="store_true",
        default=True,
        help="Use normalized Laplacian D^{-1/2} L D^{-1/2} (default: on).",
    )
    parser.add_argument(
        "--friction.no_normalize_laplacian",
        dest="friction_normalize_laplacian",
        action="store_false",
        help="Disable normalized Laplacian (use unnormalized).",
    )
    parser.add_argument(
        "--friction.recompute_mu",
        dest="friction_recompute_mu",
        action="store_true",
        default=True,
        help="Recompute edge weights mu at each inner step (default: on).",
    )
    parser.add_argument(
        "--friction.no_recompute_mu",
        dest="friction_recompute_mu",
        action="store_false",
        help="Use fixed mu estimated once per forward.",
    )
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
        "--energy_reg_scope",
        choices=["all", "last"],
        default="last",
        help="Use energies from all friction layers or only the last one when applying regularization (default: last).",
    )
    parser.add_argument(
        "--energy_reg_target",
        choices=["absolute", "normalized", "margin", "rank"],
        default="rank",
        help="Energy regularization target: absolute (mean log1p), normalized (variance), margin alignment, or rank alignment (default: rank).",
    )
    parser.add_argument(
        "--energy_reg_mode",
        choices=["absolute", "normalized"],
        default=None,
        help="Deprecated alias for --energy_reg_target (kept for backward compatibility).",
    )
    parser.add_argument(
        "--energy_rank_margin",
        type=float,
        default=0.5,
        help="Margin used when energy_reg_target is margin/rank (batch-normalized energies).",
    )
    parser.add_argument(
        "--energy_rank_topk",
        type=int,
        default=1,
        help="Number of hardest incorrect samples to contrast against each correct sample when computing the ranking loss.",
    )
    parser.add_argument(
        "--energy_rank_fallback",
        choices=["absolute", "none"],
        default="absolute",
        help="Fallback regularizer when a batch has only correct or only incorrect predictions (default: absolute).",
    )
    parser.add_argument(
        "--energy_eval_scope",
        choices=["auto", "per_sample"],
        default="auto",
        help="Energy tensor used for metrics/alerts: auto aligns with regularization scope (e.g., last layer), per_sample uses the summed energy.",
    )
    parser.add_argument(
        "--energy_metrics_source",
        choices=["normalized", "raw"],
        default="normalized",
        help="Use z-score normalized energy for AUROC/coverage metrics (default) or raw energy.",
    )
    parser.add_argument(
        "--energy_guard",
        type=str,
        default=None,
        help=(
            "Dynamic lambda guard with lower/upper thresholds, e.g. std_low=0.1,std_high=5,"
            "p90_low=0.5,p90_high=10,factor=0.5,up=1.2,min_weight=1e-5,max=1e-3. "
            "Use 'off' to disable."
        ),
    )
    parser.add_argument(
        "--energy_watch",
        type=str,
        default=None,
        help=(
            "Energy monitoring thresholds (low/high), e.g. std=0.1,std_high=10,p90=0.5,p90_high=5,"
            "mean_low=0.1 (use 'off' to disable)."
        ),
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
        "--timing_steps",
        type=int,
        default=0,
        help="Record per-step timing breakdown for the first N steps (0 = disable).",
    )
    parser.add_argument(
        "--timing_warmup",
        type=int,
        default=10,
        help="Warmup steps to skip before timing breakdown.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1 = auto for CUDA, 0 = main thread).",
    )
    # Determinism defaults to ON for reproducibility; allow explicit opt-out
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=True,
        help="Enable deterministic algorithms (default: on).",
    )
    parser.add_argument(
        "--no_deterministic",
        dest="deterministic",
        action="store_false",
        help="Disable deterministic algorithms (faster, less reproducible).",
    )
    parser.add_argument(
        "--sortish_batches",
        action="store_true",
        help="Enable sortish batching on training split to reduce padding.",
    )
    parser.add_argument(
        "--sortish_chunk_mult",
        type=int,
        default=50,
        help="Chunk multiple for sortish batching (chunk_size = batch_size * mult).",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save model checkpoint after training (saved to save_dir/model.pt).",
    )
    return parser.parse_args()


def _parse_key_value_arg(raw: str) -> Dict[str, str]:
    """Parse comma separated key=value pairs."""

    entries: Dict[str, str] = {}
    for token in raw.split(","):
        piece = token.strip()
        if not piece or "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        entries[key.strip().lower()] = value.strip()
    return entries


def _build_energy_guard(arg: str | None) -> EnergyGuardConfig:
    """Convert CLI guard string to config."""

    guard = EnergyGuardConfig()
    if arg is None:
        return guard
    lowered = arg.strip().lower()
    if lowered in {"off", "none", "0"}:
        guard.std_threshold = 0.0
        return guard
    kv = _parse_key_value_arg(arg)
    try:
        if "std" in kv or "std_threshold" in kv:
            guard.std_threshold = float(
                kv.get("std", kv.get("std_threshold", guard.std_threshold))
            )
        if "std_low" in kv:
            guard.std_threshold = float(kv["std_low"])
        if "std_high" in kv or "std_upper" in kv:
            guard.std_high_threshold = float(
                kv.get("std_high", kv.get("std_upper", guard.std_high_threshold))
            )
        if "factor" in kv or "scale" in kv:
            guard.factor = float(kv.get("factor", kv.get("scale", guard.factor)))
        if "min" in kv or "min_weight" in kv:
            guard.min_weight = float(
                kv.get("min", kv.get("min_weight", guard.min_weight))
            )
        if "p90_low" in kv:
            guard.p90_low_threshold = float(kv["p90_low"])
        if "p90_high" in kv or "p90_upper" in kv:
            guard.p90_high_threshold = float(
                kv.get("p90_high", kv.get("p90_upper", guard.p90_high_threshold))
            )
        if "max" in kv or "max_weight" in kv:
            guard.max_weight = float(kv.get("max", kv.get("max_weight")))
        if "up" in kv or "increase" in kv:
            guard.increase_factor = float(kv.get("up", kv.get("increase")))
    except ValueError as exc:  # pragma: no cover - CLI validation
        raise ValueError(
            f"Invalid --energy_guard value: '{arg}'. Expected floats."
        ) from exc
    return guard


def _build_energy_watch(arg: str | None) -> EnergyWatchConfig:
    """Convert CLI watch string to config."""

    watch = EnergyWatchConfig()
    if arg is None:
        return watch
    lowered = arg.strip().lower()
    if lowered in {"off", "none", "0"}:
        return watch
    kv = _parse_key_value_arg(arg)
    try:
        if "std" in kv or "std_threshold" in kv:
            watch.std_threshold = float(kv.get("std", kv.get("std_threshold")))  # type: ignore[arg-type]
        if "std_high" in kv or "std_upper" in kv:
            watch.std_high_threshold = float(kv.get("std_high", kv.get("std_upper")))  # type: ignore[arg-type]
        if "p90" in kv or "p90_threshold" in kv:
            watch.p90_threshold = float(kv.get("p90", kv.get("p90_threshold")))  # type: ignore[arg-type]
        if "p90_high" in kv or "p90_upper" in kv:
            watch.p90_high_threshold = float(kv.get("p90_high", kv.get("p90_upper")))  # type: ignore[arg-type]
        if "mean_low" in kv:
            watch.mean_low_threshold = float(kv["mean_low"])  # type: ignore[arg-type]
        if "mean_high" in kv or "mean_upper" in kv:
            watch.mean_high_threshold = float(kv.get("mean_high", kv.get("mean_upper")))  # type: ignore[arg-type]
    except ValueError as exc:  # pragma: no cover - CLI validation
        raise ValueError(
            f"Invalid --energy_watch value: '{arg}'. Expected floats."
        ) from exc
    return watch


@dataclass
class DeviceChoice:
    device: str
    description: str
    brand: str
    backend: str


def choose_device() -> DeviceChoice:
    """Pick the best available accelerator (CUDA preferred)."""

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vendor = "AMD" if "amd" in name.lower() else "NVIDIA"
        backend = "ROCm" if vendor == "AMD" else "CUDA"
        desc = f"{vendor} GPU ({name}) via {backend}"
        return DeviceChoice(
            device="cuda", description=desc, brand=vendor, backend=backend
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


def _parse_train_noise_levels(raw: str) -> List[str]:
    levels = [level.strip() for level in raw.split(",") if level.strip()]
    return levels or ["clean"]


def _initialize_device_choice() -> DeviceChoice:
    device_choice = choose_device()
    print(f"[Device] Preferred backend: {device_choice.description}")
    ready, error_msg = verify_device(device_choice)
    if ready:
        return device_choice
    emit_warning(
        f"Failed to initialize {device_choice.description}. Error: {error_msg}. "
        "Falling back to CPU execution."
    )
    return DeviceChoice(
        device="cpu", description="CPU execution", brand="CPU", backend="CPU"
    )


def _ensure_save_root(save_dir: str) -> Path:
    base_result = Path(save_dir).expanduser().resolve()
    expected_root = (Path.cwd() / "result").resolve()
    if expected_root not in base_result.parents and base_result != expected_root:
        raise ValueError("save_dir must be within ./result")
    base_result.mkdir(parents=True, exist_ok=True)
    return base_result


def _create_run_directory(args: argparse.Namespace, base_result: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name_parts = [args.task]
    if args.task == "sst2_noisy" and args.noise_intensity:
        run_name_parts.append(args.noise_intensity)
    run_name_parts.append(args.model)
    run_name = "_".join(run_name_parts) + f"_{timestamp}_seed{args.seed}"
    run_dir = base_result / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _build_friction_config(args: argparse.Namespace) -> FrictionConfig:
    return FrictionConfig(
        K=args.friction_K,
        eta=args.friction_eta,
        neighbor=args.friction_neighbor,
        radius=args.friction_radius,
        k=args.friction_k,
        eta_decay=getattr(args, "friction_eta_decay", 0.5),
        mu_max=getattr(args, "friction_mu_max", 5.0),
        smooth_lambda=getattr(args, "friction_smooth_lambda", 0.05),
        normalize_laplacian=getattr(args, "friction_normalize_laplacian", True),
        recompute_mu=getattr(args, "friction_recompute_mu", True),
    )


def _build_optim_config(args: argparse.Namespace) -> OptimizationConfig:
    return OptimizationConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
    )


def _build_experiment_config(
    args: argparse.Namespace,
    device_choice: DeviceChoice,
    guard_config: EnergyGuardConfig,
    watch_config: EnergyWatchConfig,
    train_noise_levels: List[str],
) -> ExperimentConfig:
    friction = _build_friction_config(args)
    optim = _build_optim_config(args)
    energy_reg_target = args.energy_reg_target
    if args.energy_reg_mode is not None:
        print(
            f"[warn] --energy_reg_mode is deprecated; using its value '{args.energy_reg_mode}' as energy_reg_target."
        )
        energy_reg_target = args.energy_reg_mode
    return ExperimentConfig(
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
        energy_reg_scope=args.energy_reg_scope,
        energy_reg_target=energy_reg_target,
        energy_reg_mode=energy_reg_target,
        energy_rank_margin=args.energy_rank_margin,
        energy_rank_topk=args.energy_rank_topk,
        energy_rank_fallback=args.energy_rank_fallback,
        energy_eval_scope=args.energy_eval_scope,
        energy_metrics_source=args.energy_metrics_source,
        energy_guard=guard_config,
        energy_watch=watch_config,
        use_amp=not args.no_amp,
        compile_model=args.compile,
        timing_steps=args.timing_steps,
        timing_warmup=args.timing_warmup,
    )


def _load_tokenizer_for_config(config: ExperimentConfig, tokenizer_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    except Exception as exc:  # pragma: no cover - network failure
        raise RuntimeError(
            f"Unable to load tokenizer '{config.tokenizer_name}'. Ensure it is cached or internet is available."
        ) from exc
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    config.vocab_size = len(tokenizer)
    config.tokenizer_name = tokenizer_name
    return tokenizer


def _configure_cuda_backends(device_choice: DeviceChoice) -> None:
    if device_choice.device != "cuda":
        return
    is_amd = device_choice.brand.lower() == "amd"
    major = 0
    try:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if (
                not is_amd
                and hasattr(torch.backends.cuda, "matmul")
                and hasattr(torch.backends.cuda.matmul, "fp32_precision")
            ):
                torch.backends.cuda.matmul.fp32_precision = (
                    "tf32" if major >= 8 else "ieee"
                )
            if (
                not is_amd
                and hasattr(torch.backends, "cudnn")
                and hasattr(torch.backends.cudnn, "conv")
                and hasattr(torch.backends.cudnn.conv, "fp32_precision")
            ):
                torch.backends.cudnn.conv.fp32_precision = (
                    "tf32" if major >= 8 else "ieee"
                )
        torch.backends.cudnn.benchmark = True
        if not is_amd and hasattr(torch, "set_float32_matmul_precision") and major >= 8:
            torch.set_float32_matmul_precision("high")
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"Failed to configure CUDA backend optimizations: {exc}")


def _maybe_disable_determinism(flag: bool) -> None:
    if not flag:
        return
    try:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"Failed to disable deterministic algorithms: {exc}")


def _build_data_bundle(
    args: argparse.Namespace,
    config: ExperimentConfig,
    tokenizer,
    train_noise_levels: List[str],
):
    bundle = build_dataloaders(
        task=config.task,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_seq_len,
        seed=config.seed,
        noise_intensity=config.noise_intensity,
        train_noise_levels=train_noise_levels,
        workers=(None if args.workers < 0 else args.workers),
        distributed=False,
        world_size=1,
        rank=0,
        sortish_batches=args.sortish_batches,
        sortish_chunk_mult=args.sortish_chunk_mult,
    )
    config.num_labels = bundle.num_labels
    config.noise_vocab = bundle.noise_vocab
    return bundle


def _maybe_save_metadata(
    run_dir: Path, config: ExperimentConfig, device_choice: DeviceChoice, data_bundle
) -> None:
    save_config(run_dir / "config.json", config.to_dict())
    write_env(run_dir / "env.txt", device_choice)
    if data_bundle.noise_config:
        save_json(run_dir / "noise_config.json", data_bundle.noise_config)


def _run_cli(args: argparse.Namespace) -> None:
    # Avoid tokenizer multiprocessing + DataLoader workers deadlock warnings
    import os as _os

    _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    train_noise_levels = _parse_train_noise_levels(args.train_noise_levels)
    guard_config = _build_energy_guard(args.energy_guard)
    watch_config = _build_energy_watch(args.energy_watch)
    device_choice = _initialize_device_choice()
    base_result = _ensure_save_root(args.save_dir)
    run_dir = _create_run_directory(args, base_result)
    config = _build_experiment_config(
        args, device_choice, guard_config, watch_config, train_noise_levels
    )
    set_seed(config.seed)
    tokenizer = _load_tokenizer_for_config(config, args.tokenizer)
    _configure_cuda_backends(device_choice)
    if args.deterministic is False:
        _maybe_disable_determinism(True)
    data_bundle = _build_data_bundle(args, config, tokenizer, train_noise_levels)
    _maybe_save_metadata(run_dir, config, device_choice, data_bundle)
    run_with_device(
        config=config,
        tokenizer=tokenizer,
        data_bundle=data_bundle,
        run_dir=run_dir,
        device_choice=device_choice,
    )


def main() -> None:
    args = parse_args()
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

    model = build()
    if config.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="default", fullgraph=False)
            print("[Compile] Model compiled with torch.compile")
        except Exception as exc:  # pragma: no cover - backend dependent
            emit_warning(
                f"torch.compile failed: {exc}. Proceeding without compilation."
            )

    try:
        run_training(
            config=config, model=model, loaders=data_bundle.loaders, save_dir=run_dir
        )
    except RuntimeError as exc:
        if device_choice.device == "cpu":
            raise
        # If failure is related to torch.compile/triton toolchain, retry on the SAME device without compile first.
        msg = str(exc)
        if config.compile_model and (
            "triton" in msg.lower()
            or "dynamo" in msg.lower()
            or "Python.h" in msg
            or "CalledProcessError" in msg
        ):
            emit_warning(
                "torch.compile failed at runtime; retrying on GPU without compilation."
            )
            config.compile_model = False
            model = build()
            run_training(
                config=config,
                model=model,
                loaders=data_bundle.loaders,
                save_dir=run_dir,
            )
            return
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
