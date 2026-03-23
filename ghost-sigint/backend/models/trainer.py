"""Training pipeline: label smoothing, AdamW + OneCycleLR, early stopping, adversarial training."""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset, random_split
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from constants import N_CLASSES, LABEL_SMOOTHING, LEARNING_RATE, WEIGHT_DECAY
from .adversarial import fgsm_attack, pgd_attack


class LabelSmoothingCE(nn.Module):
    def __init__(self, eps: float = 0.1, n: int = 10):
        super().__init__()
        self.eps, self.n = eps, n

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        lp = F.log_softmax(logits, dim=-1)
        smooth = self.eps / self.n
        oh = torch.zeros_like(lp).scatter_(1, targets.unsqueeze(1), 1.0)
        return -((oh * (1 - self.eps) + smooth) * lp).sum(-1).mean()


class EarlyStopping:
    def __init__(self, patience: int = 15, delta: float = 1e-3):
        self.patience, self.delta = patience, delta
        self.best = -float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val: float) -> bool:
        if val > self.best + self.delta:
            self.best = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def _train_epoch(model, loader, opt, criterion, scaler, scheduler, device, adv):
    model.train()
    tot_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        r = np.random.random()
        if adv and r > 0.4:
            x = fgsm_attack(model, x, y) if r < 0.7 else pgd_attack(model, x, y)
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        tot_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return tot_loss / total, correct / total


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    tot_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        tot_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return tot_loss / total, correct / total


def train_model(model: nn.Module, data: Tuple[torch.Tensor, torch.Tensor],
                epochs: int = 100, batch_size: int = 128,
                checkpoint: Optional[str] = None, adv_train: bool = True) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ds = TensorDataset(*data)
    n = len(ds)
    n_tr, n_vl = int(0.7 * n), int(0.15 * n)
    tr, vl, te = random_split(ds, [n_tr, n_vl, n - n_tr - n_vl])
    tr_l = DataLoader(tr, batch_size, shuffle=True, pin_memory=True)
    vl_l = DataLoader(vl, batch_size * 2, pin_memory=True)
    te_l = DataLoader(te, batch_size * 2, pin_memory=True)
    criterion = LabelSmoothingCE(LABEL_SMOOTHING, N_CLASSES)
    opt = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = OneCycleLR(opt, max_lr=LEARNING_RATE, total_steps=epochs * len(tr_l), pct_start=0.3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    es = EarlyStopping()
    best_acc, best_state = 0.0, None
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for _ in range(epochs):
        tl, ta = _train_epoch(model, tr_l, opt, criterion, scaler, sched, device, adv_train)
        vl_, va = _eval_epoch(model, vl_l, criterion, device)
        hist["train_loss"].append(tl)
        hist["train_acc"].append(ta)
        hist["val_loss"].append(vl_)
        hist["val_acc"].append(va)
        if va > best_acc:
            best_acc = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint:
                torch.save(best_state, checkpoint)
        if es.step(va):
            break
    if best_state:
        model.load_state_dict(best_state)
    _, te_acc = _eval_epoch(model, te_l, criterion, device)
    hist["test_acc"] = te_acc
    return hist
