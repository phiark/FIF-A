"""Shared dataloader utilities."""

from __future__ import annotations

from typing import Callable, Dict, Optional

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def build_loaders_for_splits(
    dataset,
    collate_fn: Callable,
    batch_size: int,
    loader_kwargs: Optional[Dict] = None,
    distributed: bool = False,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    sortish_batches: bool = False,
    sortish_chunk_mult: int = 50,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for train/validation/test with shared logic."""

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
        elif (not distributed) and sortish_batches and split == "train":
            lengths = [len(x) for x in dataset[split]["input_ids"]]
            from . import SortishSampler  # local import to avoid cycles

            sampler = SortishSampler(
                lengths, batch_size=batch_size, chunk_mult=sortish_chunk_mult
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
    return loaders
