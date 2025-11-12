"""Minimal sparse adjacency helpers.

Includes a small LRU-like cache for window edges to avoid rebuilding the
same index tensors on every forward pass. This especially helps GPU/MPS
where Python-side list building dominates time.
"""

from __future__ import annotations

from typing import Iterable, List, Set, Tuple, Dict, Optional

import torch
from torch.nn import functional as F


_WINDOW_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}


def build_window_edges(
    length: int, radius: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Return undirected edges for a sliding window, cached by (length, radius).

    Edges are cached as a CPU LongTensor and moved to the requested device
    on demand to amortize Python-side construction cost.
    """

    if length <= 1 or radius <= 0:
        return torch.zeros((0, 2), dtype=torch.long, device=device or torch.device("cpu"))
    key = (length, radius)
    cached = _WINDOW_CACHE.get(key)
    if cached is None:
        edges: List[Tuple[int, int]] = []
        for i in range(length):
            lo = max(0, i - radius)
            hi = min(length, i + radius + 1)
            for j in range(lo, hi):
                if i < j:
                    edges.append((i, j))
        if not edges:
            cached = torch.zeros((0, 2), dtype=torch.long)
        else:
            cached = torch.tensor(edges, dtype=torch.long)
        _WINDOW_CACHE[key] = cached
    if device is None or str(device) == "cpu":
        return cached
    return cached.to(device, non_blocking=(device.type == "cuda"))


def build_knn_edges(hidden: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Construct edges via cosine-similarity kNN."""

    length = int(mask.sum().item())
    if length <= 1:
        return torch.zeros((0, 2), dtype=torch.long, device=hidden.device)
    vecs = hidden[:length]
    normed = F.normalize(vecs, dim=-1)
    sim = normed @ normed.transpose(0, 1)
    k_eff = min(k + 1, length)
    topk = torch.topk(sim, k=k_eff, dim=-1).indices
    edges: Set[Tuple[int, int]] = set()
    for i in range(length):
        for j in topk[i].tolist():
            if i == j:
                continue
            pair = (i, j) if i < j else (j, i)
            edges.add(pair)
    if not edges:
        return torch.zeros((0, 2), dtype=torch.long, device=hidden.device)
    return torch.tensor(sorted(edges), dtype=torch.long, device=hidden.device)
