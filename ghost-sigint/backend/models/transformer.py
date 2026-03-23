"""Temporal Transformer with CLS token for STFT input (batch, T, F)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn


def sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
    return pe.unsqueeze(0)  # (1, max_len, d_model)


class TemporalTransformer(nn.Module):
    """Transformer encoder with CLS token. Input: (batch, T=32, F=129)."""

    def __init__(self, n_classes: int = 10, d_model: int = 256, nhead: int = 8,
                 n_layers: int = 4, d_ff: int = 1024, dropout: float = 0.1,
                 max_len: int = 64, d_input: int = 129):
        super().__init__()
        self.proj = nn.Linear(d_input, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("pe", sinusoidal_pe(max_len + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", norm_first=True, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout * 2),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.proj(x)  # (B, T, d_model)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)
        x = x + self.pe[:, : T + 1]
        x = self.encoder(x)
        return self.head(x[:, 0])  # CLS token
