"""Text noise injectors for SST-2."""

from __future__ import annotations

import random
import string
from typing import Dict, List

NOISE_PRESETS: Dict[str, Dict[str, float]] = {
    "low": {"char_noise_p": 0.02, "word_drop_p": 0.01, "span_drop_p": 0.0},
    "med": {"char_noise_p": 0.05, "word_drop_p": 0.03, "span_drop_p": 0.02},
    "high": {"char_noise_p": 0.10, "word_drop_p": 0.05, "span_drop_p": 0.05},
}

DEFAULT_NOISE_LEVELS: List[str] = ["clean", "low", "med", "high"]
LEVEL_HASH = {"clean": 3, "low": 11, "med": 23, "high": 37}

VOCAB = string.ascii_letters + string.digits


def apply_noise(text: str, config: Dict[str, float], rng: random.Random) -> str:
    """Apply character/word/span drops."""

    words = text.split()
    words = _word_drop(words, config.get("word_drop_p", 0.0), rng)
    words = _span_drop(words, config.get("span_drop_p", 0.0), rng)
    noised = " ".join(words)
    return _char_noise(noised, config.get("char_noise_p", 0.0), rng)


def _char_noise(text: str, prob: float, rng: random.Random) -> str:
    if prob <= 0.0:
        return text
    chars: List[str] = []
    for ch in text:
        if rng.random() < prob:
            if rng.random() < 0.5 and len(chars) > 0:
                continue  # delete char
            chars.append(rng.choice(VOCAB))
        else:
            chars.append(ch)
    return "".join(chars)


def _word_drop(words: List[str], prob: float, rng: random.Random) -> List[str]:
    if prob <= 0.0:
        return words
    kept = [w for w in words if rng.random() >= prob]
    return kept if kept else words


def _span_drop(words: List[str], prob: float, rng: random.Random) -> List[str]:
    if prob <= 0.0 or not words:
        return words
    tokens = words[:]
    length = len(tokens)
    mask = [True] * length
    idx = 0
    while idx < length:
        if rng.random() < prob:
            span_len = _geometric(rng, p=0.4)
            for j in range(span_len):
                if idx + j < length:
                    mask[idx + j] = False
            idx += span_len
        else:
            idx += 1
    kept = [tok for tok, keep in zip(tokens, mask) if keep]
    return kept if kept else tokens


def _geometric(rng: random.Random, p: float) -> int:
    """Sample from Geometric(p) with support {1,2,...}."""

    count = 1
    while rng.random() > p:
        count += 1
    return count


def resolve_noise_config(level: str) -> Dict[str, float] | None:
    """Return concrete noise config for a symbolic level."""

    if level == "clean":
        return None
    preset = NOISE_PRESETS.get(level)
    if not preset:
        raise ValueError(f"Unknown noise level '{level}'.")
    return {"level": level, **preset}
