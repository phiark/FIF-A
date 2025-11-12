"""Baseline Transformer classifier."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from fif_mvp.train import energy as energy_utils


class TransformerBlock(nn.Module):
    """Standard MHSA + FFN block."""

    def __init__(
        self, hidden_size: int, num_heads: int, ff_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        key_padding_mask = attention_mask == 0
        attn_output, _ = self.attn(
            hidden,
            hidden,
            hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = self.norm1(hidden + attn_output)
        hidden = self.norm2(hidden + self.ff(hidden))
        return hidden


class TransformerClassifier(nn.Module):
    """Lightweight Transformer encoder with classification head."""

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
        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, ff_size, dropout)
                for _ in range(4)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        noise_level_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        pooled = self._pool(hidden, attention_mask)
        logits = self.classifier(self.norm(pooled))
        per_sample_energy = energy_utils.sequence_energy(hidden, attention_mask)
        batch_energy = per_sample_energy.mean()
        return (logits, per_sample_energy, batch_energy, hidden)

    @staticmethod
    def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Use first token if available, else mean pool valid tokens."""

        if attention_mask.size(1) > 0 and attention_mask[
            :, 0
        ].sum() == attention_mask.size(0):
            return hidden[:, 0]
        lengths = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        return (hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / lengths
