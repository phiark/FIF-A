"""Minimal sparse adjacency helpers."""

from __future__ import annotations

from typing import Iterable, List, Set, Tuple

import torch
from torch.nn import functional as F


def build_window_edges(length: int, radius: int, device: torch.device) -> torch.Tensor:
    """Return undirected edges for a sliding window."""

    if length <= 1 or radius <= 0:
        return torch.zeros((0, 2), dtype=torch.long, device=device)
    edges: List[Tuple[int, int]] = []
    for i in range(length):
        lo = max(0, i - radius)
        hi = min(length, i + radius + 1)
        for j in range(lo, hi):
            if i < j:
                edges.append((i, j))
    if not edges:
        return torch.zeros((0, 2), dtype=torch.long, device=device)
    return torch.tensor(edges, dtype=torch.long, device=device)


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
