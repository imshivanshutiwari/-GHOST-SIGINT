"""MUSIC Direction-of-Arrival estimation for ULA."""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from typing import Tuple


def steering_vector(theta_rad: float, M: int) -> np.ndarray:
    """ULA steering vector with λ/2 element spacing."""
    return np.exp(1j * np.pi * np.sin(theta_rad) * np.arange(M))


def _mdl(eigenvalues: np.ndarray, N: int) -> int:
    M = len(eigenvalues)
    best, best_k = float("inf"), 1
    for k in range(1, M):
        noise = eigenvalues[k:]
        if len(noise) == 0:
            break
        gm = np.exp(np.mean(np.log(noise + 1e-12)))
        am = np.mean(noise) + 1e-12
        mdl = -(M - k) * N * np.log(gm / am) + 0.5 * k * (2 * M - k) * np.log(N)
        if mdl < best:
            best, best_k = mdl, k
    return best_k


def music_pseudospectrum(R: np.ndarray, D: int, thetas_deg: np.ndarray) -> np.ndarray:
    M = R.shape[0]
    vals, vecs = np.linalg.eigh(R)
    idx = np.argsort(vals)[::-1]
    En = vecs[:, idx[D:]]  # noise subspace
    EnEnH = En @ En.conj().T
    P = np.zeros(len(thetas_deg))
    for i, th in enumerate(thetas_deg):
        a = steering_vector(np.deg2rad(th), M)
        P[i] = 1.0 / (np.real(a.conj() @ EnEnH @ a) + 1e-12)
    return P


def estimate_doa_music(
    X: np.ndarray,
    M: int = 8,
    theta_range: Tuple[float, float] = (-90.0, 90.0),
    step: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Full MUSIC pipeline. X: (M, N_snapshots)."""
    R = (X @ X.conj().T) / X.shape[1]
    vals = np.sort(np.linalg.eigvalsh(R))[::-1]
    D = max(1, _mdl(vals, X.shape[1]))
    thetas = np.arange(theta_range[0], theta_range[1] + step, step)
    P = music_pseudospectrum(R, D, thetas)
    peaks, _ = find_peaks(P, height=np.max(P) * 0.1)
    doa = thetas[peaks[:D]] if len(peaks) > 0 else np.array([0.0])
    return doa, P, D
