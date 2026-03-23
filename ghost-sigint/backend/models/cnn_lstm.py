"""CNN-LSTM hybrid for IQ input (batch, 2, 1024)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMHybrid(nn.Module):
    """CNN feature extractor + BiLSTM + attention. Input: (batch, 2, 1024)."""

    def __init__(self, n_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # → (B, 64, 512)
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.2,
        )
        self.attn = nn.Linear(256, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)             # (B, 64, 512)
        x = x.permute(0, 2, 1)     # (B, 512, 64)
        h, _ = self.lstm(x)         # (B, 512, 256)
        scores = F.softmax(self.attn(h).squeeze(-1), dim=1).unsqueeze(-1)
        ctx = (scores * h).sum(1)   # (B, 256)
        return self.fc(self.drop(ctx))
