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
        batch_size, _, _ = hidden.shape
        outputs = hidden.clone()
        energies = hidden.new_zeros(batch_size)
        lengths = attention_mask.sum(dim=1).to(torch.int64)
        unique_lengths = torch.unique(lengths)

        for length_val in unique_lengths:
            length = int(length_val.item())
            if length <= 1:
                continue
            mask = lengths == length_val
            indices = mask.nonzero(as_tuple=True)[0]
            seq_hidden = hidden.index_select(0, indices)[:, :length].contiguous()
            if self.config.neighbor == "window":
                edges = sparse_utils.build_window_edges(
                    length, radius=self.config.radius, device=hidden.device
                )
                seq_out, seq_energy = self._run_window_batch(seq_hidden, edges)
            else:
                bucket_mask = attention_mask.index_select(0, indices)[:, :length]
                edges = sparse_utils.build_knn_edges_batched(
                    seq_hidden, bucket_mask, k=self.config.k
                )
                seq_out, seq_energy = self._run_knn_batch(seq_hidden, edges)

            outputs[indices, :length, :] = seq_out
            energies[indices] = seq_energy

        return outputs, energies

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

    def _run_knn_batch(
        self, seq_hidden: torch.Tensor, batched_edges: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, length, _ = seq_hidden.shape
        if batched_edges.numel() == 0 or batch_size == 0:
            return seq_hidden, seq_hidden.new_zeros(batch_size)
        batch_ids = batched_edges[:, 0]
        idx_i = batched_edges[:, 1]
        idx_j = batched_edges[:, 2]
        offsets = batch_ids * length
        global_edges = torch.stack((offsets + idx_i, offsets + idx_j), dim=1).long()

        q = self.q_proj(seq_hidden)
        state = seq_hidden
        last_mu = None
        for step in range(self.config.K):
            base = state if self.config.recompute_mu else seq_hidden
            mu = self._edge_weights_variable(base, global_edges)
            if self.config.mu_max > 0:
                mu = mu.clamp_max(self.config.mu_max)
            lap = self._laplacian_variable(state, global_edges, mu)
            eta = self._step_eta(step)
            state = state - eta * (lap - q)
            state = self._smooth_batch(state)
            last_mu = mu
        out = self.norm(state + seq_hidden)
        final_mu = (
            last_mu
            if last_mu is not None
            else self._edge_weights_variable(state, global_edges)
        )
        energy = self._edge_energy_variable(
            final_mu, state, global_edges, batch_size, length
        )
        return out, energy

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

    def _edge_weights_variable(
        self, hidden: torch.Tensor, global_edges: torch.Tensor
    ) -> torch.Tensor:
        flat = hidden.reshape(-1, hidden.size(-1))
        h_i = flat[global_edges[:, 0]]
        h_j = flat[global_edges[:, 1]]
        diff = h_i - h_j
        dist = torch.norm(diff, dim=-1, keepdim=True)
        cos = F.cosine_similarity(h_i, h_j, dim=-1, eps=1e-6).unsqueeze(-1)
        feats = torch.cat([dist, cos], dim=-1)
        mu = F.softplus(self.edge_mlp(feats)) + 1e-5
        return mu

    def _laplacian_variable(
        self, hidden: torch.Tensor, global_edges: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        batch_size, length, hidden_dim = hidden.shape
        flat = hidden.reshape(batch_size * length, hidden_dim)
        idx_i = global_edges[:, 0]
        idx_j = global_edges[:, 1]
        diffs = flat[idx_i] - flat[idx_j]
        weighted = mu * diffs
        if self.config.normalize_laplacian and global_edges.numel() > 0:
            deg = flat.new_zeros(batch_size * length)
            weights = mu.squeeze(-1)
            deg.index_add_(0, idx_i, weights)
            deg.index_add_(0, idx_j, weights)
            deg = deg.clamp_min(1e-6)
            inv = deg.pow(-0.5)
            scale = (inv[idx_i] * inv[idx_j]).unsqueeze(-1)
            weighted = weighted * scale
        lap = flat.new_zeros_like(flat)
        lap.index_add_(0, idx_i, weighted)
        lap.index_add_(0, idx_j, -weighted)
        return lap.view(batch_size, length, hidden_dim)

    def _edge_energy_variable(
        self,
        mu: torch.Tensor,
        hidden: torch.Tensor,
        global_edges: torch.Tensor,
        batch_size: int,
        length: int,
    ) -> torch.Tensor:
        flat = hidden.reshape(batch_size * length, hidden.size(-1))
        h_i = flat[global_edges[:, 0]]
        h_j = flat[global_edges[:, 1]]
        diff = h_i - h_j
        squared = diff.pow(2).sum(dim=-1)
        edge_vals = 0.5 * mu.squeeze(-1) * squared
        per_sample = hidden.new_zeros(batch_size)
        batch_ids = torch.div(global_edges[:, 0], length, rounding_mode="floor").to(
            torch.long
        )
        per_sample.index_add_(0, batch_ids, edge_vals)
        return per_sample

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

    def _smooth_batch(self, state: torch.Tensor) -> torch.Tensor:
        if self.config.smooth_lambda <= 0:
            return state
        lam = self.config.smooth_lambda
        left = torch.roll(state, 1, dims=1)
        right = torch.roll(state, -1, dims=1)
        left[:, 0] = state[:, 0]
        right[:, -1] = state[:, -1]
        return state - lam * (2 * state - left - right)
