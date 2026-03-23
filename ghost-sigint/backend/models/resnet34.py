"""ResNet-34 with Squeeze-Excitation blocks for mel-spectrogram classification."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        r = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(r, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))
        s = self.fc(s).view(s.size(0), -1, 1, 1)
        return x * s


class BasicBlockSE(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock2d(out_c, reduction)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


class ResNet34SE(nn.Module):
    """ResNet-34 + SE for mel-spectrogram input (3, 128, 128)."""

    def __init__(self, n_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, 3, stride=1, reduction=4)
        self.layer2 = self._make_layer(64, 128, 4, stride=2, reduction=8)
        self.layer3 = self._make_layer(128, 256, 6, stride=2, reduction=16)
        self.layer4 = self._make_layer(256, 512, 3, stride=2, reduction=32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(512, n_classes)
        self._init_weights()

    @staticmethod
    def _make_layer(in_c: int, out_c: int, n: int, stride: int, reduction: int) -> nn.Sequential:
        layers = [BasicBlockSE(in_c, out_c, stride=stride, reduction=reduction)]
        for _ in range(1, n):
            layers.append(BasicBlockSE(out_c, out_c, stride=1, reduction=reduction))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)
