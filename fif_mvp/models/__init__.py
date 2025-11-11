"""Model factory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from fif_mvp.config import ExperimentConfig


@dataclass
class ModelOutput:
    """Standardized model output."""

    logits: torch.Tensor
    per_sample_energy: torch.Tensor
    batch_energy: torch.Tensor
    hidden_states: torch.Tensor


from .hybrid_model import HybridClassifier  # noqa: E402
from .transformer_baseline import TransformerClassifier  # noqa: E402


def build_model(
    config: ExperimentConfig,
    vocab_size: int,
    num_labels: int,
    pad_token_id: int,
) -> torch.nn.Module:
    """Instantiate the requested model."""

    common_kwargs = {
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "hidden_size": config.hidden_size,
        "num_heads": config.num_heads,
        "ff_size": config.ff_size,
        "dropout": config.dropout,
        "max_seq_len": config.max_seq_len,
        "num_labels": num_labels,
        "noise_vocab_size": len(config.noise_vocab or ["clean"]),
    }

    if config.model_type == "baseline":
        return TransformerClassifier(**common_kwargs)

    layer_plan: list[Literal["attention", "friction"]]
    if config.model_type == "hybrid":
        layer_plan = ["attention", "friction", "friction", "attention"]
    elif config.model_type == "full_friction":
        layer_plan = ["friction"] * 4
    else:  # pragma: no cover - guarded by CLI
        raise ValueError(f"Unknown model_type '{config.model_type}'.")

    return HybridClassifier(
        **common_kwargs,
        layer_plan=layer_plan,
        friction_config=config.friction,
    )


__all__ = ["ModelOutput", "build_model", "TransformerClassifier", "HybridClassifier"]
