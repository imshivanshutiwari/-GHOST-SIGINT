"""1-D CNN with Squeeze-Excitation and SiLU for IQ input (2, 1024)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SE1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        r = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(r, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=-1)
        s = self.fc(s).unsqueeze(-1)
        return x * s


class ResBlock1d(nn.Module):
    def __init__(self, in_c: int, out_c: int, reduction: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.SiLU(inplace=True),
        )
        self.se = SE1d(out_c, reduction)
        self.skip = nn.Conv1d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)
        # SE scales conv_out channel-wise; add identity skip for residual
        return F.silu(self.se(conv_out) + self.skip(x))


class CNN1DSE(nn.Module):
    """1-D CNN + SE for IQ input (batch, 2, 1024)."""

    def __init__(self, n_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace=True),
            nn.Conv1d(32, 64, 5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
        )
        self.se64 = SE1d(64, 4)
        self.rb1 = ResBlock1d(64, 128, 8)
        self.rb2 = ResBlock1d(128, 256, 16)
        self.head = nn.Sequential(
            nn.Conv1d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.se64(x)
        x = self.rb1(x)
        x = self.rb2(x)
        return self.head(x)
