"""3-layer BiLSTM with scaled attention for mel-frame input (batch, T, 128)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMAttention(nn.Module):
    """BiLSTM + attention. Input: (batch, T=50, d=128)."""

    def __init__(self, n_classes: int = 10, input_size: int = 128,
                 hidden_size: int = 256, n_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=n_layers, batch_first=True,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.attn_w = nn.Linear(hidden_size * 2, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)  # (B, T, 512)
        h = self.norm(h)
        scores = self.attn_w(h).squeeze(-1) / math.sqrt(self.hidden_size * 2)
        alpha = F.softmax(scores, dim=1).unsqueeze(-1)
        ctx = (alpha * h).sum(dim=1)
        return self.fc(self.drop(ctx))
