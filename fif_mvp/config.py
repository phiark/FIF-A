"""Configuration dataclasses for FIF experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal


@dataclass
class FrictionConfig:
    """Hyperparameters governing the frictional interaction field."""

    K: int = 3
    eta: float = 0.3
    neighbor: Literal["window", "knn"] = "window"
    radius: int = 4
    k: int = 8
    eta_decay: float = 0.5
    mu_max: float = 5.0
    smooth_lambda: float = 0.05
    normalize_laplacian: bool = True
    recompute_mu: bool = True


@dataclass
class OptimizationConfig:
    """Optimizer/scheduler configuration."""

    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    max_steps: int = 0
    epochs: int = 5
    grad_clip: float = 1.0


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    task: Literal["snli", "sst2", "sst2_noisy"] = "snli"
    model_type: Literal["baseline", "hybrid", "full_friction"] = "baseline"
    hidden_size: int = 256
    ff_size: int = 1024
    num_heads: int = 4
    dropout: float = 0.1
    batch_size: int = 64
    max_seq_len: int = 128
    vocab_size: int = 30522
    num_labels: int = 3
    seed: int = 42
    device: str = "cpu"
    tokenizer_name: str = "bert-base-uncased"
    friction: FrictionConfig = field(default_factory=FrictionConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    dump_energy_per_sample: bool = False
    noise_intensity: str | None = None
    noise_vocab: List[str] = field(default_factory=lambda: ["clean"])
    train_noise_levels: List[str] = field(default_factory=list)
    energy_reg_weight: float = 0.0
    # Acceleration options
    use_amp: bool = True
    compile_model: bool = False
    # Distributed options
    distributed: bool = False
    rank: int = 0
    world_size: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the config to a plain dict."""
        data = asdict(self)
        data["friction"] = asdict(self.friction)
        data["optimization"] = asdict(self.optimization)
        return data
