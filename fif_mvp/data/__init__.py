"""Dataset builders and shared utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence
import os
import torch
from torch.utils.data import DataLoader, Sampler

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


class SortishSampler(Sampler[int]):
    """Sortish sampler: shuffles indices, then sorts within chunks by length.

    This reduces padding while preserving randomness. Non-distributed only.
    """

    def __init__(self, lengths: List[int], batch_size: int, chunk_mult: int = 50) -> None:
        self.lengths = lengths
        self.batch_size = max(1, batch_size)
        self.chunk_size = max(self.batch_size * max(1, chunk_mult), self.batch_size)

        import numpy as _np

        n = len(lengths)
        idx = _np.arange(n)
        _np.random.shuffle(idx)
        chunks = [idx[i : i + self.chunk_size] for i in range(0, n, self.chunk_size)]
        # Sort each chunk by length descending
        self.order = []
        for ch in chunks:
            ch_list = list(ch)
            ch_list.sort(key=lambda i: lengths[i], reverse=True)
            self.order.extend(ch_list)

    def __iter__(self):  # type: ignore[override]
        return iter(self.order)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.order)


def build_dataloaders(
    task: str,
    tokenizer,
    batch_size: int,
    max_length: int,
    seed: int,
    noise_intensity: Optional[str] = None,
    train_noise_levels: Optional[List[str]] = None,
    workers: Optional[int] = None,
    distributed: bool = False,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    sortish_batches: bool = False,
    sortish_chunk_mult: int = 50,
) -> DataBundle:
    """Return dataloaders and metadata for the requested task."""

    noise_vocab = train_noise_levels or DEFAULT_NOISE_LEVELS
    # Loader performance knobs (auto-tuned for GPU backends)
    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    prefer_accel = has_cuda or has_mps
    auto_workers = min(8, max(0, (os.cpu_count() or 1) - 1))
    if workers is not None and workers >= 0:
        num_workers = workers
    else:
        num_workers = auto_workers if prefer_accel else 0
    # pin_memory only benefits CUDA
    pin_memory = has_cuda
    persistent_workers = num_workers > 0
    common_loader_args = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )
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
            loader_kwargs=common_loader_args,
            distributed=distributed,
            world_size=world_size,
            rank=rank,
            sortish_batches=sortish_batches,
            sortish_chunk_mult=sortish_chunk_mult,
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
            loader_kwargs=common_loader_args,
            noise_settings=None,
            distributed=distributed,
            world_size=world_size,
            rank=rank,
            sortish_batches=sortish_batches,
            sortish_chunk_mult=sortish_chunk_mult,
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
            loader_kwargs=common_loader_args,
            noise_settings={"level": intensity, **noise_config},
            train_noise_levels=train_noise_levels or list(noise_vocab),
            distributed=distributed,
            world_size=world_size,
            rank=rank,
            sortish_batches=sortish_batches,
            sortish_chunk_mult=sortish_chunk_mult,
        )
        return DataBundle(
            loaders=loaders,
            num_labels=2,
            label_names=info["label_names"],
            noise_config={"level": intensity, **noise_config},
            noise_vocab=list(noise_vocab),
        )

    raise ValueError(f"Unknown task '{task}'.")
