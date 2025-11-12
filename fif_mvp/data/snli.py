"""SNLI dataloader utilities."""

from __future__ import annotations

from typing import Callable, Dict, Tuple, Optional

from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


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

    loader_kwargs = loader_kwargs or {}
    loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "validation", "test"):
        shuffle = split == "train"
        sampler = None
        if distributed and split == "train":
            sampler = DistributedSampler(
                dataset[split],
                num_replicas=world_size or 1,
                rank=rank or 0,
                shuffle=True,
                drop_last=False,
            )
            shuffle = False
        loaders[split] = DataLoader(
            dataset[split],
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            **loader_kwargs,
        )

    label_names = ["entailment", "neutral", "contradiction"]
    return loaders, {"label_names": label_names, "seed": seed}
