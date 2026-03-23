"""TLS-ESPRIT Direction-of-Arrival estimation."""
from __future__ import annotations
import numpy as np


def tls_esprit(X: np.ndarray, D: int) -> np.ndarray:
    """
    TLS-ESPRIT DOA estimation.

    Args:
        X: (M, N) complex snapshot matrix
        D: number of sources

    Returns:
        DOA array in degrees, length D
    """
    M = X.shape[0]
    R = (X @ X.conj().T) / X.shape[1]
    _, U = np.linalg.eigh(R)
    Es = U[:, -D:]  # signal subspace (eigh gives ascending order)

    Es1 = Es[: M - 1, :]
    Es2 = Es[1:, :]

    C = np.hstack([Es1, Es2])
    _, _, Vh = np.linalg.svd(C, full_matrices=False)
    V = Vh.conj().T
    V12 = V[:D, D:]
    V22 = V[D:, D:]
    Phi = -V12 @ np.linalg.pinv(V22)
    mu = np.linalg.eigvals(Phi)
    sin_th = np.clip(np.angle(mu) / np.pi, -1.0, 1.0)
    return np.rad2deg(np.arcsin(sin_th))
