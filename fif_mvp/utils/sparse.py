"""Minimal sparse adjacency helpers.

Includes a small LRU-like cache for window edges to avoid rebuilding the
same index tensors on every forward pass. This especially helps GPU/MPS
where Python-side list building dominates time.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.nn import functional as F

MAX_WINDOW_CACHE = 128
_WINDOW_CACHE: "OrderedDict[Tuple[int, int], torch.Tensor]" = OrderedDict()
_WINDOW_CACHE_DEVICE: "OrderedDict[Tuple[int, int, str], torch.Tensor]" = OrderedDict()


def clear_window_cache() -> None:
    """Clear all cached window edges (used for tests or memory pressure)."""

    _WINDOW_CACHE.clear()
    _WINDOW_CACHE_DEVICE.clear()


def _prune_cache(cache: OrderedDict, max_size: int) -> None:
    while len(cache) > max_size:
        cache.popitem(last=False)


def build_window_edges(
    length: int, radius: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Return undirected edges for a sliding window, cached by (length, radius).

    Edges are cached as a CPU LongTensor and moved to the requested device
    on demand to amortize Python-side construction cost.
    """

    if length <= 1 or radius <= 0:
        return torch.zeros(
            (0, 2), dtype=torch.long, device=device or torch.device("cpu")
        )
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
        _WINDOW_CACHE.move_to_end(key)
        _prune_cache(_WINDOW_CACHE, MAX_WINDOW_CACHE)
    if device is None or str(device) == "cpu":
        return cached
    key_dev = (length, radius, str(device))
    dev_cached = _WINDOW_CACHE_DEVICE.get(key_dev)
    if dev_cached is None or dev_cached.device != device:
        dev_cached = cached.to(device, non_blocking=(device.type == "cuda"))
        _WINDOW_CACHE_DEVICE[key_dev] = dev_cached
        _WINDOW_CACHE_DEVICE.move_to_end(key_dev)
        _prune_cache(_WINDOW_CACHE_DEVICE, MAX_WINDOW_CACHE)
    return dev_cached


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


def build_knn_edges_batched(
    hidden: torch.Tensor, mask: torch.Tensor, k: int
) -> torch.Tensor:
    """Construct kNN edges for an entire bucket with shared length (vectorized).

    The previous Python-loop implementation repeatedly normalized each sequence
    and inserted edges into Python sets, which dominated forward latency in
    Hybrid training. This version leverages batched matmul and masking so the
    complexity is bound by a single `B×L×L` similarity matrix per bucket.
    """

    batch_size, length, _ = hidden.shape
    if batch_size == 0 or length <= 1:
        return hidden.new_zeros((0, 3), dtype=torch.long)

    mask_bool = mask.to(dtype=torch.bool, copy=False)
    if not mask_bool.any():
        return hidden.new_zeros((0, 3), dtype=torch.long)

    normed = F.normalize(hidden, dim=-1)
    sim = torch.matmul(normed, normed.transpose(1, 2))
    pair_mask = mask_bool.unsqueeze(2) & mask_bool.unsqueeze(1)
    diag = torch.eye(length, device=hidden.device, dtype=torch.bool).unsqueeze(0)
    pair_mask = pair_mask & ~diag
    sim = sim.masked_fill(~pair_mask, float("-inf"))

    k_eff = min(k + 1, length)
    topk = sim.topk(k=k_eff, dim=-1).indices  # (B, L, k_eff)

    batch_idx = (
        torch.arange(batch_size, device=hidden.device)
        .view(batch_size, 1, 1)
        .expand(-1, length, k_eff)
    )
    src_idx = (
        torch.arange(length, device=hidden.device)
        .view(1, length, 1)
        .expand(batch_size, -1, k_eff)
    )

    src_valid = mask_bool.view(batch_size, length, 1).expand(-1, -1, k_eff)
    mask_expanded = mask_bool.unsqueeze(1).expand(-1, length, -1)
    dst_valid = torch.gather(mask_expanded, 2, topk.clamp(min=0))
    valid_mask = src_valid & dst_valid

    edges = torch.stack((batch_idx, src_idx, topk), dim=-1).view(-1, 3)
    if not valid_mask.any():
        return hidden.new_zeros((0, 3), dtype=torch.long)
    edges = edges[valid_mask.view(-1)]

    if edges.numel() == 0:
        return hidden.new_zeros((0, 3), dtype=torch.long)

    src = edges[:, 1]
    dst = edges[:, 2]
    low = torch.minimum(src, dst)
    high = torch.maximum(src, dst)
    valid = low != high
    if not valid.any():
        return hidden.new_zeros((0, 3), dtype=torch.long)

    canon = torch.stack((edges[:, 0], low, high), dim=1)[valid]
    canon = torch.unique(canon, dim=0)
    return canon
