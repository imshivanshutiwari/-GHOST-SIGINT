"""Ensemble classifier with learned weights and MC-Dropout uncertainty."""
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleClassifier(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.w_logits = nn.Parameter(torch.zeros(len(models)))

    def forward(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(self.w_logits, dim=0)
        probs = []
        for model, x in zip(self.models, inputs):
            probs.append(F.softmax(model(x), dim=-1))
        stacked = torch.stack(probs, dim=0)          # (N, B, C)
        final = (weights.view(-1, 1, 1) * stacked).sum(0)
        return final, final.max(dim=-1).values

    def mc_uncertainty(self, inputs: List[torch.Tensor],
                       n_mc: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train()
        all_p = []
        with torch.no_grad():
            for _ in range(n_mc):
                p, _ = self.forward(inputs)
                all_p.append(p)
        self.eval()
        stacked = torch.stack(all_p, dim=0)
        return stacked.mean(0), stacked.std(0).max(dim=-1).values
