"""Frictional interaction layer.

Energy: :math:`E = 0.5 \\sum_{i,j} \\mu_{ij} \\lVert h_i - h_j \\rVert^2`
Update: :math:`H^{t+1} = H^{t} - \\eta (L H^{t} - q)` where :math:`L` is the sparse Laplacian.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from fif_mvp.config import FrictionConfig
from fif_mvp.train import energy as energy_utils
from fif_mvp.utils import sparse as sparse_utils


class FrictionLayer(nn.Module):
    """Implements the frictional equilibrium step."""

    def __init__(self, hidden_size: int, config: FrictionConfig) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        inner = max(hidden_size // 2, 32)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2, inner),
            nn.GELU(),
            nn.Linear(inner, 1),
        )
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden.shape
        outputs = torch.zeros_like(hidden)
        per_sample_energy = []
        # Reuse window edges per unique length within the batch to avoid redundant transfers
        local_edge_cache: dict[int, torch.Tensor] = {}

        for b in range(batch_size):
            length = int(attention_mask[b].sum().item())
            if length <= 1:
                outputs[b] = hidden[b]
                per_sample_energy.append(hidden.new_tensor(0.0))
                continue
            seq_hidden = hidden[b, :length]
            seq_mask = attention_mask[b, :length]
            edges = self._build_edges(seq_hidden, seq_mask, local_edge_cache)
            seq_out, seq_energy = self._run_single(seq_hidden, edges)
            padded = hidden[b].clone()
            padded[:length] = seq_out
            outputs[b] = padded
            per_sample_energy.append(seq_energy)

        per_sample = torch.stack(per_sample_energy, dim=0)
        return outputs, per_sample

    def _build_edges(
        self, seq_hidden: torch.Tensor, seq_mask: torch.Tensor, cache: dict[int, torch.Tensor] | None = None
    ) -> torch.Tensor:
        length = seq_hidden.size(0)
        device = seq_hidden.device
        if self.config.neighbor == "window":
            if cache is not None and length in cache:
                return cache[length]
            edges = sparse_utils.build_window_edges(
                length, radius=self.config.radius, device=device
            )
            if cache is not None:
                cache[length] = edges
            return edges
        return sparse_utils.build_knn_edges(seq_hidden, seq_mask, k=self.config.k)

    def _run_single(
        self, seq_hidden: torch.Tensor, edges: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if edges.numel() == 0:
            return seq_hidden, seq_hidden.new_tensor(0.0)

        h_in = seq_hidden
        q = self.q_proj(h_in)
        state = h_in
        last_mu = None
        for step in range(self.config.K):
            if self.config.recompute_mu or last_mu is None:
                current = state if self.config.recompute_mu else h_in
                mu = self._edge_weights(current, edges)
            else:
                mu = last_mu
            if self.config.mu_max > 0:
                mu = mu.clamp_max(self.config.mu_max)
            lap = self._laplacian(state, edges, mu)
            eta = self._step_eta(step)
            state = state - eta * (lap - q)
            state = self._smooth(state)
            last_mu = mu
        out = self.norm(state + h_in)
        final_mu = last_mu if last_mu is not None else self._edge_weights(state, edges)
        energy = energy_utils.edge_energy(final_mu, state, edges)
        return out, energy

    def _edge_weights(self, hidden: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        h_i = hidden[edges[:, 0]]
        h_j = hidden[edges[:, 1]]
        diff = h_i - h_j
        dist = torch.norm(diff, dim=-1, keepdim=True)
        cos = F.cosine_similarity(h_i, h_j, dim=-1, eps=1e-6).unsqueeze(-1)
        feats = torch.cat([dist, cos], dim=-1)
        mu = F.softplus(self.edge_mlp(feats)) + 1e-5
        return mu

    def _laplacian(
        self, hidden: torch.Tensor, edges: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        lap = torch.zeros_like(hidden)
        diffs = hidden[edges[:, 0]] - hidden[edges[:, 1]]
        weighted = mu * diffs
        if self.config.normalize_laplacian and edges.numel() > 0:
            length = hidden.size(0)
            deg = hidden.new_zeros(length)
            weights = mu.squeeze(-1)
            deg.index_add_(0, edges[:, 0], weights)
            deg.index_add_(0, edges[:, 1], weights)
            deg = deg.clamp_min(1e-6)
            inv_sqrt = deg.pow(-0.5)
            scale = (inv_sqrt[edges[:, 0]] * inv_sqrt[edges[:, 1]]).unsqueeze(-1)
            weighted = weighted * scale
        lap.index_add_(0, edges[:, 0], weighted)
        lap.index_add_(0, edges[:, 1], -weighted)
        return lap

    def _step_eta(self, step: int) -> float:
        if self.config.eta_decay <= 0:
            return self.config.eta
        decay = self.config.eta_decay
        return self.config.eta * (decay**step)

    def _smooth(self, state: torch.Tensor) -> torch.Tensor:
        if self.config.smooth_lambda <= 0:
            return state
        lam = self.config.smooth_lambda
        left = torch.roll(state, 1, dims=0)
        right = torch.roll(state, -1, dims=0)
        left[0] = state[0]
        right[-1] = state[-1]
        return state - lam * (2 * state - left - right)
