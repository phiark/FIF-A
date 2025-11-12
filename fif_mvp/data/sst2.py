"""SST-2 dataloaders with configurable noise."""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

from datasets import DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader

from .noise import (
    DEFAULT_NOISE_LEVELS,
    LEVEL_HASH,
    apply_noise,
    resolve_noise_config,
)


def load_sst2(
    tokenizer,
    batch_size: int,
    max_length: int,
    seed: int,
    collate_fn: Callable,
    noise_settings: Optional[Dict[str, float]] = None,
    train_noise_levels: Optional[List[str]] = None,
    loader_kwargs: Optional[Dict] = None,
) -> Tuple[Dict[str, DataLoader], Dict]:
    """Load SST-2, optionally applying evaluation noise."""

    try:
        raw: DatasetDict = load_dataset("glue", "sst2")
    except Exception as exc:  # pragma: no cover - download failure
        raise RuntimeError(
            "Unable to load SST-2. Please ensure datasets cache is available or internet access is enabled."
        ) from exc

    eval_level = (noise_settings or {}).get("level", "clean")
    train_levels = (
        [lvl for lvl in (train_noise_levels or DEFAULT_NOISE_LEVELS)]
        if noise_settings
        else ["clean"]
    )
    split_dataset = raw["train"].train_test_split(test_size=0.1, seed=seed)
    dataset = DatasetDict(
        {
            "train": split_dataset["train"],
            "validation": split_dataset["test"],
            "test": raw["validation"],
        }
    )

    def preprocess(batch, indices=None):
        sentences = []
        noise_levels = []
        for offset, sentence in enumerate(batch["sentence"]):
            idx = indices[offset] if indices is not None else offset
            level = payload["level"]
            config = payload["config"]
            if config is not None:
                rng_seed = seed * 7919 + idx * 17 + payload["seed_offset"]
                rng = random.Random(rng_seed)
                sentence = apply_noise(sentence, config, rng)
            noise_levels.append(level)
            sentences.append(sentence)
        enc = tokenizer(
            sentences,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": batch["label"],
            "noise_level": noise_levels,
        }

    def map_with_level(split: str, level: str, config: Optional[Dict[str, float]]):
        payload["level"] = level
        payload["config"] = config
        payload["seed_offset"] = LEVEL_HASH.get(level, 0)
        return dataset[split].map(
            preprocess,
            batched=True,
            with_indices=True,
        )

    payload = {"level": "clean", "config": None, "seed_offset": 0}
    processed_splits = {}

    train_variants = []
    for lvl in train_levels:
        cfg = resolve_noise_config(lvl)
        train_variants.append(map_with_level("train", lvl, cfg))
    processed_splits["train"] = (
        concatenate_datasets(train_variants)
        if len(train_variants) > 1
        else train_variants[0]
    )

    eval_cfg = resolve_noise_config(eval_level) if noise_settings else None
    for split in ("validation", "test"):
        processed_splits[split] = map_with_level(split, eval_level, eval_cfg)

    dataset = DatasetDict(processed_splits)
    dataset.set_format(type="python")

    loader_kwargs = loader_kwargs or {}
    # Remove None-valued kwargs to avoid TypeErrors on older PyTorch
    loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}

    loaders = {
        split: DataLoader(
            dataset[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_fn,
            **loader_kwargs,
        )
        for split in ("train", "validation", "test")
    }

    return loaders, {
        "label_names": ["negative", "positive"],
        "noise_level": eval_level,
    }
