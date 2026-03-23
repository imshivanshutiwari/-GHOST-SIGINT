"""Deflationary FastICA for Blind Source Separation."""
from __future__ import annotations
import numpy as np
from typing import List, Tuple


def fast_ica(
    X: np.ndarray,
    n_components: int = 2,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    FastICA (deflationary).

    Args:
        X: (n_features, n_samples) mixed signal matrix
        n_components: number of independent components

    Returns:
        S (n_components, n_samples), W (unmixing), convergence curve
    """
    n_features, n_samples = X.shape
    X = X - X.mean(axis=1, keepdims=True)

    # Whiten
    cov = X @ X.T / n_samples
    d, E = np.linalg.eigh(cov)
    idx = np.argsort(d)[::-1]
    d, E = np.maximum(d[idx], 1e-10), E[:, idx]
    W_white = np.diag(1.0 / np.sqrt(d)) @ E.T
    Xw = W_white @ X

    W = np.zeros((n_components, n_features))
    convergence: List[float] = []

    for k in range(n_components):
        w = np.random.randn(n_features)
        w /= np.linalg.norm(w) + 1e-12
        for _ in range(max_iter):
            wx = w @ Xw
            g = np.tanh(wx)
            gp = 1.0 - g ** 2
            w_new = (Xw @ g) / n_samples - gp.mean() * w
            for j in range(k):
                w_new -= (w_new @ W[j]) * W[j]
            norm = np.linalg.norm(w_new) + 1e-12
            w_new /= norm
            conv = float(abs(abs(w_new @ w) - 1.0))
            convergence.append(conv)
            w = w_new
            if conv < tol:
                break
        W[k] = w

    S = W @ Xw
    return S, W, convergence
