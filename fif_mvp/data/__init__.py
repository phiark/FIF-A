"""Dataset builders and shared utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from .noise import DEFAULT_NOISE_LEVELS, NOISE_PRESETS
from .snli import load_snli
from .sst2 import load_sst2


@dataclass
class DataBundle:
    """Container for dataloaders and metadata."""

    loaders: Dict[str, DataLoader]
    num_labels: int
    label_names: List[str]
    noise_config: Optional[Dict[str, float]] = None
    noise_vocab: List[str] = field(default_factory=lambda: ["clean"])


class SequenceCollator:
    """Pad sequences in a batch to the maximum length."""

    def __init__(
        self, pad_token_id: int, noise_vocab: Optional[Sequence[str]] = None
    ) -> None:
        self.pad_token_id = pad_token_id
        vocab = list(noise_vocab) if noise_vocab else ["clean"]
        self.noise_to_id = {name: idx for idx, name in enumerate(vocab)}
        self.noise_vocab = vocab

    def __call__(self, batch: Iterable[Dict]) -> Dict[str, torch.Tensor]:
        examples = list(batch)
        max_len = max(len(item["input_ids"]) for item in examples)
        batch_size = len(examples)
        input_ids = torch.full(
            (batch_size, max_len), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.zeros(batch_size, dtype=torch.long)
        noise_levels: List[str] = []
        noise_ids = torch.zeros(batch_size, dtype=torch.long)
        for idx, item in enumerate(examples):
            length = len(item["input_ids"])
            input_ids[idx, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[idx, :length] = 1
            labels[idx] = int(item["labels"])
            level = item.get("noise_level", "clean")
            noise_levels.append(level)
            noise_ids[idx] = self.noise_to_id.get(level, 0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "noise_level": noise_levels,
            "noise_level_ids": noise_ids,
        }


def build_dataloaders(
    task: str,
    tokenizer,
    batch_size: int,
    max_length: int,
    seed: int,
    noise_intensity: Optional[str] = None,
    train_noise_levels: Optional[List[str]] = None,
) -> DataBundle:
    """Return dataloaders and metadata for the requested task."""

    noise_vocab = train_noise_levels or DEFAULT_NOISE_LEVELS
    collator = SequenceCollator(
        pad_token_id=tokenizer.pad_token_id, noise_vocab=noise_vocab
    )
    if task == "snli":
        loaders, info = load_snli(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            seed=seed,
            collate_fn=collator,
        )
        return DataBundle(
            loaders=loaders,
            num_labels=3,
            label_names=info["label_names"],
            noise_vocab=list(noise_vocab),
        )

    if task == "sst2":
        loaders, info = load_sst2(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            seed=seed,
            collate_fn=collator,
            noise_settings=None,
        )
        return DataBundle(
            loaders=loaders,
            num_labels=2,
            label_names=info["label_names"],
            noise_vocab=list(noise_vocab),
        )

    if task == "sst2_noisy":
        intensity = noise_intensity or "low"
        if intensity not in NOISE_PRESETS:
            raise ValueError(
                f"Unsupported noise intensity '{intensity}'. Expected one of {list(NOISE_PRESETS)}."
            )
        noise_config = NOISE_PRESETS[intensity]
        loaders, info = load_sst2(
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            seed=seed,
            collate_fn=collator,
            noise_settings={"level": intensity, **noise_config},
            train_noise_levels=train_noise_levels or list(noise_vocab),
        )
        return DataBundle(
            loaders=loaders,
            num_labels=2,
            label_names=info["label_names"],
            noise_config={"level": intensity, **noise_config},
            noise_vocab=list(noise_vocab),
        )

    raise ValueError(f"Unknown task '{task}'.")
