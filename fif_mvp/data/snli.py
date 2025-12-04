"""SNLI dataloader utilities."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader

from .common import build_loaders_for_splits


def load_snli(
    tokenizer,
    batch_size: int,
    max_length: int,
    seed: int,
    collate_fn: Callable,
    loader_kwargs: Optional[Dict] = None,
    distributed: bool = False,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    sortish_batches: bool = False,
    sortish_chunk_mult: int = 50,
) -> Tuple[Dict[str, DataLoader], Dict]:
    """Load SNLI via the datasets hub."""

    try:
        dataset: DatasetDict = load_dataset("snli")
    except Exception as exc:  # pragma: no cover - download failure
        raise RuntimeError(
            "Unable to load SNLI. Please ensure datasets cache is available or internet access is enabled."
        ) from exc

    dataset = dataset.filter(lambda example: example["label"] != -1)

    def tokenize(batch):
        enc = tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": batch["label"],
            "noise_level": ["clean"] * len(batch["label"]),
        }

    dataset = dataset.map(
        tokenize, batched=True, remove_columns=None, load_from_cache_file=True
    )
    dataset.set_format(type="python")

    loaders = build_loaders_for_splits(
        dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        loader_kwargs=loader_kwargs,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
        sortish_batches=sortish_batches,
        sortish_chunk_mult=sortish_chunk_mult,
    )

    label_names = ["entailment", "neutral", "contradiction"]
    return loaders, {"label_names": label_names, "seed": seed}
