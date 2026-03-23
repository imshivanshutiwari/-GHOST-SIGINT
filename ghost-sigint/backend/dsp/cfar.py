"""
CA-CFAR and OS-CFAR detectors.
N_guard=2, N_ref=8, Pfa=1e-6.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

N_GUARD = 2
N_REF = 8
PFA = 1e-6


def _ca_cfar_threshold(noise_cells: NDArray[np.float64], pfa: float, n_ref: int) -> float:
    """CA-CFAR threshold factor α: α = N_ref · (Pfa^{-1/N_ref} - 1)"""
    alpha = n_ref * (pfa ** (-1 / n_ref) - 1)
    return float(alpha * np.mean(noise_cells))


def _os_cfar_threshold(
    noise_cells: NDArray[np.float64],
    pfa: float,
    n_ref: int,
    k_rank: int | None = None,
) -> float:
    """
    OS-CFAR: use k-th order statistic (k ≈ 3/4 · N_ref).
    T = α_os · x_(k) where α_os derived from Pfa.
    """
    if k_rank is None:
        k_rank = int(0.75 * n_ref)
    k_rank = min(k_rank, len(noise_cells) - 1)
    sorted_cells = np.sort(noise_cells)
    x_k = sorted_cells[k_rank]
    # Exact OS-CFAR alpha: complex but approximated
    alpha_os = 1.0 / (pfa ** (1 / k_rank) - 1) if pfa < 1 else 1.0
    return float(alpha_os * x_k)


def ca_cfar(
    power: NDArray[np.float64],
    n_guard: int = N_GUARD,
    n_ref: int = N_REF,
    pfa: float = PFA,
) -> NDArray[np.bool_]:
    """
    CA-CFAR: Cell-Averaging CFAR detection.
    Two-pass (leading + lagging reference windows).
    Returns boolean detection mask.
    """
    n = len(power)
    detections = np.zeros(n, dtype=bool)
    half_guard = n_guard
    half_ref = n_ref

    for i in range(half_guard + half_ref, n - half_guard - half_ref):
        # Lagging reference
        lag_start = i - half_guard - half_ref
        lag_end = i - half_guard
        # Leading reference
        lead_start = i + half_guard + 1
        lead_end = i + half_guard + half_ref + 1
        noise_cells = np.concatenate([
            power[lag_start:lag_end],
            power[lead_start:lead_end],
        ])
        threshold = _ca_cfar_threshold(noise_cells, pfa, n_ref * 2)
        detections[i] = power[i] > threshold

    return detections


def os_cfar(
    power: NDArray[np.float64],
    n_guard: int = N_GUARD,
    n_ref: int = N_REF,
    pfa: float = PFA,
    k_rank: int | None = None,
) -> NDArray[np.bool_]:
    """
    OS-CFAR: Ordered-Statistics CFAR. More robust to clutter edges.
    """
    n = len(power)
    detections = np.zeros(n, dtype=bool)
    half_guard = n_guard
    half_ref = n_ref

    for i in range(half_guard + half_ref, n - half_guard - half_ref):
        lag = power[i - half_guard - half_ref: i - half_guard]
        lead = power[i + half_guard + 1: i + half_guard + half_ref + 1]
        noise_cells = np.concatenate([lag, lead])
        threshold = _os_cfar_threshold(noise_cells, pfa, len(noise_cells), k_rank)
        detections[i] = power[i] > threshold

    return detections


def detect(
    power_spectrum: NDArray[np.float64],
    method: str = "ca",
    **kwargs,
) -> dict:
    """
    Run CFAR detection on a power spectrum.
    Returns dict with 'detections' mask and 'n_detected'.
    """
    fn = ca_cfar if method == "ca" else os_cfar
    mask = fn(power_spectrum, **kwargs)
    indices = np.where(mask)[0]
    return {
        "detections": mask,
        "indices": indices,
        "n_detected": int(len(indices)),
        "peak_power": float(power_spectrum[indices].max()) if len(indices) > 0 else -np.inf,
    }
