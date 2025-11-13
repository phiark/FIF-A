"""Frictional interaction layer.

Energy: :math:`E = 0.5 \\sum_{i,j} \\mu_{ij} \\lVert h_i - h_j \\rVert^2`
Update: :math:`H^{t+1} = H^{t} - \\eta (L H^{t} - q)` where :math:`L` is the sparse Laplacian.
"""

from __future__ import annotations

from collections import defaultdict
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
        batch_size, _, _ = hidden.shape
        outputs = hidden.clone()
        energies = hidden.new_zeros(batch_size)
        lengths = attention_mask.sum(dim=1).to(torch.int64)
        buckets: dict[int, List[int]] = defaultdict(list)
        for idx, length in enumerate(lengths.tolist()):
            buckets[int(length)].append(idx)

        for length, indices in buckets.items():
            if length <= 1:
                continue
            seq_hidden = hidden[indices, :length].contiguous()
            if self.config.neighbor == "window":
                edges = sparse_utils.build_window_edges(
                    length, radius=self.config.radius, device=hidden.device
                )
                seq_out, seq_energy = self._run_window_batch(seq_hidden, edges)
            else:
                seq_out_chunks = []
                seq_energy_vals = []
                for local_idx, sample_idx in enumerate(indices):
                    seq_mask = attention_mask[sample_idx, :length]
                    edges = self._build_edges(
                        hidden[sample_idx, :length], seq_mask, cache=None
                    )
                    out_single, energy_single = self._run_single(
                        hidden[sample_idx, :length], edges
                    )
                    seq_out_chunks.append(out_single.unsqueeze(0))
                    seq_energy_vals.append(energy_single.unsqueeze(0))
                seq_out = torch.cat(seq_out_chunks, dim=0)
                seq_energy = torch.cat(seq_energy_vals, dim=0)

            outputs[indices, :length] = seq_out
            energies[indices] = seq_energy

        return outputs, energies

    def _build_edges(
        self,
        seq_hidden: torch.Tensor,
        seq_mask: torch.Tensor,
        cache: dict[int, torch.Tensor] | None = None,
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

    def _run_window_batch(
        self, seq_hidden: torch.Tensor, edges: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = seq_hidden.size(0)
        if edges.numel() == 0 or batch_size == 0:
            return seq_hidden, seq_hidden.new_zeros(batch_size)
        q = self.q_proj(seq_hidden)
        state = seq_hidden
        last_mu = None
        for step in range(self.config.K):
            if self.config.recompute_mu or last_mu is None:
                base = state if self.config.recompute_mu else seq_hidden
                mu = self._edge_weights_batch(base, edges)
            else:
                mu = last_mu
            if self.config.mu_max > 0:
                mu = mu.clamp_max(self.config.mu_max)
            lap = self._laplacian_batch(state, edges, mu)
            eta = self._step_eta(step)
            state = state - eta * (lap - q)
            state = self._smooth_batch(state)
            last_mu = mu
        out = self.norm(state + seq_hidden)
        final_mu = (
            last_mu if last_mu is not None else self._edge_weights_batch(state, edges)
        )
        energy = energy_utils.edge_energy_batch(final_mu, state, edges)
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

    def _edge_weights_batch(
        self, hidden: torch.Tensor, edges: torch.Tensor
    ) -> torch.Tensor:
        idx_i = edges[:, 0]
        idx_j = edges[:, 1]
        h_i = hidden[:, idx_i, :]
        h_j = hidden[:, idx_j, :]
        diff = h_i - h_j
        dist = torch.norm(diff, dim=-1, keepdim=True)
        cos = F.cosine_similarity(h_i, h_j, dim=-1, eps=1e-6).unsqueeze(-1)
        feats = torch.cat([dist, cos], dim=-1)
        flat = feats.reshape(-1, feats.size(-1))
        mu = F.softplus(self.edge_mlp(flat)) + 1e-5
        return mu.reshape(hidden.size(0), feats.size(1), 1)

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

    def _laplacian_batch(
        self, hidden: torch.Tensor, edges: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        batch_size, length, hidden_size = hidden.shape
        if edges.numel() == 0:
            return torch.zeros_like(hidden)
        idx_i = edges[:, 0]
        idx_j = edges[:, 1]
        h_i = hidden[:, idx_i, :]
        h_j = hidden[:, idx_j, :]
        weighted = mu * (h_i - h_j)
        base = (
            torch.arange(batch_size, device=hidden.device, dtype=torch.long).view(-1, 1)
            * length
        )
        global_i = (base + idx_i.view(1, -1)).reshape(-1)
        global_j = (base + idx_j.view(1, -1)).reshape(-1)
        if self.config.normalize_laplacian:
            weights = mu.squeeze(-1)
            deg = hidden.new_zeros(batch_size * length)
            weight_flat = weights.reshape(-1)
            deg.index_add_(0, global_i, weight_flat)
            deg.index_add_(0, global_j, weight_flat)
            deg = deg.clamp_min(1e-6)
            inv = deg.pow(-0.5)
            inv_i = inv[global_i].view(batch_size, -1)
            inv_j = inv[global_j].view(batch_size, -1)
            scale = (inv_i * inv_j).unsqueeze(-1)
            weighted = weighted * scale
        lap = hidden.new_zeros(batch_size * length, hidden_size)
        lap.index_add_(0, global_i, weighted.reshape(-1, hidden_size))
        lap.index_add_(0, global_j, -weighted.reshape(-1, hidden_size))
        return lap.view(batch_size, length, hidden_size)

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

    def _smooth_batch(self, state: torch.Tensor) -> torch.Tensor:
        if self.config.smooth_lambda <= 0:
            return state
        lam = self.config.smooth_lambda
        left = torch.roll(state, 1, dims=1)
        right = torch.roll(state, -1, dims=1)
        left[:, 0] = state[:, 0]
        right[:, -1] = state[:, -1]
        return state - lam * (2 * state - left - right)
