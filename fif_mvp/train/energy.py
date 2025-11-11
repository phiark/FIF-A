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


def sequence_energy(
    hidden: torch.Tensor, mask: torch.Tensor, radius: int = 1
) -> torch.Tensor:
    """Window-based energy usable for any encoder."""

    batch_size = hidden.size(0)
    energies = []
    for b in range(batch_size):
        length = int(mask[b].sum().item())
        if length <= 1:
            energies.append(hidden.new_tensor(0.0))
            continue
        edges = sparse_utils.build_window_edges(length, radius, device=hidden.device)
        mu = hidden.new_ones((edges.size(0), 1))
        energies.append(edge_energy(mu, hidden[b, :length], edges))
    return torch.stack(energies, dim=0)


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
