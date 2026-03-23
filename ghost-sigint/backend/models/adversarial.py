"""FGSM, PGD, CutMix, SpecAugment adversarial utilities."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                epsilon: float = 0.01) -> torch.Tensor:
    x_adv = x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x_adv), y).backward()
    with torch.no_grad():
        x_adv = x + epsilon * x_adv.grad.sign()
        x_adv = torch.clamp(x_adv, x.min(), x.max())
    return x_adv.detach()


def pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
               epsilon: float = 0.01, alpha: float = 0.002,
               n_steps: int = 10) -> torch.Tensor:
    x_adv = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, x.min(), x.max())
    for _ in range(n_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        F.cross_entropy(model(x_adv), y).backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
            x_adv = torch.clamp(x_adv, x.min(), x.max())
    return x_adv.detach()


def cutmix(x1: torch.Tensor, x2: torch.Tensor, lam: float = 0.5) -> torch.Tensor:
    T = x1.shape[-1]
    cut = int(T * (1 - lam))
    start = np.random.randint(0, max(1, T - cut + 1))
    out = x1.clone()
    out[..., start:start + cut] = x2[..., start:start + cut]
    return out


def spec_augment(x: torch.Tensor, time_mask: int = 50, freq_mask: int = 30) -> torch.Tensor:
    if x.ndim < 4:
        return x
    x = x.clone()
    B, C, F, T = x.shape
    t0 = np.random.randint(0, max(1, T - time_mask))
    x[..., t0:t0 + np.random.randint(1, time_mask + 1)] = 0.0
    f0 = np.random.randint(0, max(1, F - freq_mask))
    x[:, :, f0:f0 + np.random.randint(1, freq_mask + 1), :] = 0.0
    return x
