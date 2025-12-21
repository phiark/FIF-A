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
class EnergyGuardConfig:
    """Dynamic guard for energy regularization weight."""

    std_threshold: float = 0.1
    factor: float = 0.5
    min_weight: float = 0.0
    std_high_threshold: float | None = None
    p90_low_threshold: float | None = None
    p90_high_threshold: float | None = None
    max_weight: float | None = None
    increase_factor: float = 1.0


@dataclass
class EnergyWatchConfig:
    """Run-time monitoring thresholds for energy statistics."""

    std_threshold: float | None = None
    std_high_threshold: float | None = None
    p90_threshold: float | None = None
    p90_high_threshold: float | None = None
    mean_low_threshold: float | None = None
    mean_high_threshold: float | None = None


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
    save_model: bool = False
    noise_intensity: str | None = None
    noise_vocab: List[str] = field(default_factory=lambda: ["clean"])
    train_noise_levels: List[str] = field(default_factory=list)
    energy_reg_weight: float = 0.0
    energy_reg_scope: Literal["all", "last"] = "all"
    energy_reg_target: Literal["absolute", "normalized", "margin", "rank"] = "rank"
    # Deprecated alias retained for backward compatibility; prefer energy_reg_target.
    energy_reg_mode: Literal["absolute", "normalized", "margin", "rank"] = "rank"
    energy_rank_margin: float = 0.5
    energy_rank_topk: int = 1
    # If rank/margin has no correct/incorrect samples, fall back to absolute/none.
    energy_rank_fallback: Literal["absolute", "none"] = "absolute"
    # Which energy tensor to use for metrics/alerts: align with reg scope or always per-sample sum.
    energy_eval_scope: Literal["auto", "per_sample"] = "auto"
    # Whether correlation metrics use normalized (z-score) energy or raw.
    energy_metrics_source: Literal["normalized", "raw"] = "normalized"
    energy_guard: EnergyGuardConfig = field(default_factory=EnergyGuardConfig)
    energy_watch: EnergyWatchConfig = field(default_factory=EnergyWatchConfig)
    # Acceleration options
    use_amp: bool = True
    compile_model: bool = False
    # Timing diagnostics (0 = disabled)
    timing_steps: int = 0
    timing_warmup: int = 10
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
