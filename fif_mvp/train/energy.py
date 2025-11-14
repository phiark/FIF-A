"""Energy computations."""

from __future__ import annotations

import torch

from fif_mvp.utils import sparse as sparse_utils


def edge_energy(
    mu: torch.Tensor, hidden: torch.Tensor, edges: torch.Tensor
) -> torch.Tensor:
    """Exact energy for a single sequence."""

    if edges.numel() == 0:
        return hidden.new_tensor(0.0)
    h_i = hidden[edges[:, 0]]
    h_j = hidden[edges[:, 1]]
    diff = h_i - h_j
    squared = diff.pow(2).sum(dim=-1)
    energy = 0.5 * (mu.squeeze(-1) * squared).sum()
    return energy


def edge_energy_batch(
    mu: torch.Tensor, hidden: torch.Tensor, edges: torch.Tensor
) -> torch.Tensor:
    """Vectorized energy for a batch sharing the same edge index."""

    if edges.numel() == 0:
        return hidden.new_zeros(hidden.size(0))
    idx_i = edges[:, 0]
    idx_j = edges[:, 1]
    h_i = hidden[:, idx_i, :]
    h_j = hidden[:, idx_j, :]
    diff = h_i - h_j
    squared = diff.pow(2).sum(dim=-1)
    energy = 0.5 * (mu.squeeze(-1) * squared).sum(dim=-1)
    return energy


def sequence_energy(
    hidden: torch.Tensor, mask: torch.Tensor, radius: int = 1
) -> torch.Tensor:
    """Window-based energy usable for any encoder (vectorized by length buckets).

    Groups sequences by effective length and computes batched edge energies
    using a shared window graph per unique length.
    """

    device = hidden.device
    batch_size = hidden.size(0)
    lengths = mask.sum(dim=1).to(torch.int64)
    out = hidden.new_zeros(batch_size)
    # Map: length -> indices in batch
    unique_lengths = torch.unique(lengths).tolist()
    for L in unique_lengths:
        L_int = int(L)
        idx = (lengths == L_int).nonzero(as_tuple=True)[0]
        if L_int <= 1 or idx.numel() == 0:
            continue
        edges = sparse_utils.build_window_edges(L_int, radius, device=device)
        mu = hidden.new_ones((idx.numel(), edges.size(0), 1))
        seq_hidden = hidden.index_select(0, idx)[:, :L_int, :]
        out_vals = edge_energy_batch(mu, seq_hidden, edges)
        out.index_copy_(0, idx, out_vals)
    return out


def per_token_energy(
    mu: torch.Tensor, hidden: torch.Tensor, edges: torch.Tensor, length: int
) -> torch.Tensor:
    """Split edge energies evenly across incident tokens."""

    if edges.numel() == 0:
        return hidden.new_zeros(length)
    h_i = hidden[edges[:, 0]]
    h_j = hidden[edges[:, 1]]
    diff = h_i - h_j
    squared = diff.pow(2).sum(dim=-1)
    edge_vals = 0.5 * mu.squeeze(-1) * squared
    contrib = hidden.new_zeros(length)
    contrib.index_add_(0, edges[:, 0], 0.5 * edge_vals)
    contrib.index_add_(0, edges[:, 1], 0.5 * edge_vals)
    return contrib
