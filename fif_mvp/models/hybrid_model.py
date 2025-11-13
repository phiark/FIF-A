"""Hybrid transformer + friction classifiers."""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from torch import nn

from fif_mvp.config import FrictionConfig
from fif_mvp.train import energy as energy_utils

from .friction_layer import FrictionLayer
from .transformer_baseline import TransformerBlock


class HybridClassifier(nn.Module):
    """Encoder that mixes MHSA and friction layers."""

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        hidden_size: int,
        num_heads: int,
        ff_size: int,
        dropout: float,
        max_seq_len: int,
        num_labels: int,
        layer_plan: Iterable[str],
        friction_config: FrictionConfig,
        noise_vocab_size: int = 1,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.noise_embeddings = nn.Embedding(noise_vocab_size, hidden_size)
        self.layers = nn.ModuleList()
        for layer_type in layer_plan:
            if layer_type == "attention":
                self.layers.append(
                    TransformerBlock(hidden_size, num_heads, ff_size, dropout)
                )
            else:
                self.layers.append(FrictionLayer(hidden_size, friction_config))
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        noise_level_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        positions = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        positions = positions.clamp_max(self.max_seq_len - 1)
        hidden = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        if noise_level_ids is not None:
            noise_bias = self.noise_embeddings(noise_level_ids).unsqueeze(1)
            hidden = hidden + noise_bias
        hidden = self.dropout(hidden)

        energy_terms: List[torch.Tensor] = []
        for layer in self.layers:
            if isinstance(layer, FrictionLayer):
                hidden, energy = layer(hidden, attention_mask)
                energy_terms.append(energy)
            else:
                hidden = layer(hidden, attention_mask)

        pooled = self._pool(hidden, attention_mask)
        logits = self.classifier(self.norm(pooled))
        if energy_terms:
            per_sample_energy = torch.stack(energy_terms, dim=0).sum(dim=0)
        else:
            per_sample_energy = energy_utils.sequence_energy(hidden, attention_mask)
        return (logits, per_sample_energy, hidden)

    @staticmethod
    def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.size(1) > 0 and attention_mask[
            :, 0
        ].sum() == attention_mask.size(0):
            return hidden[:, 0]
        lengths = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        return (hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / lengths
