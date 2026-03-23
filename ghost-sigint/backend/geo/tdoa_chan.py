"""Chan's Weighted Least-Squares TDOA geolocation algorithm."""
from __future__ import annotations
import numpy as np
from typing import Tuple

C = 3e8  # speed of light m/s


def chan_wls(
    sensor_positions: np.ndarray,
    tdoa_sec: np.ndarray,
    sigma_tdoa: float = 1e-9,
) -> Tuple[np.ndarray, float, float]:
    """
    Chan's WLS TDOA geolocation.

    Args:
        sensor_positions: (M, 2) sensor coordinates in metres
        tdoa_sec: (M-1,) TDOA measurements w.r.t. sensor 0 (seconds)
        sigma_tdoa: TDOA measurement noise std (seconds)

    Returns:
        position (2,) metres, GDOP, position sigma metres
    """
    M = len(sensor_positions)
    x, y = sensor_positions[:, 0], sensor_positions[:, 1]
    K = x ** 2 + y ** 2
    d = tdoa_sec * C  # range differences

    G = np.zeros((M - 1, 3))
    h = np.zeros(M - 1)
    for i in range(1, M):
        G[i - 1, 0] = 2.0 * (x[0] - x[i])
        G[i - 1, 1] = 2.0 * (y[0] - y[i])
        G[i - 1, 2] = 2.0 * d[i - 1]
        h[i - 1] = d[i - 1] ** 2 - K[i] + K[0]

    W = np.eye(M - 1) / (sigma_tdoa ** 2)
    GtW = G.T @ W
    try:
        a = np.linalg.solve(GtW @ G, GtW @ h)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(G, h, rcond=None)[0]

    position = a[:2]

    try:
        GTWG_inv = np.linalg.inv(GtW @ G)
        gdop = float(np.sqrt(np.trace(GTWG_inv[:2, :2])))
    except Exception:
        gdop = float("nan")

    sigma_pos = sigma_tdoa * C * gdop if not np.isnan(gdop) else float("nan")
    return position, gdop, sigma_pos
