"""Friis path-loss based RF ranging."""
from __future__ import annotations
import numpy as np

C = 3e8

_FSPL_K = 20 * np.log10(4 * np.pi / C)  # ≈ -147.55 dB


def friis_ranging(
    p_rx_dbm: float,
    p_tx_dbm: float,
    freq_hz: float,
    g_tx_dbi: float = 0.0,
    g_rx_dbi: float = 0.0,
    sigma_p_db: float = 1.0,
) -> dict:
    """
    Estimate range from received power.

    FSPL(dB) = 20log10(d) + 20log10(f) + 20log10(4π/c)
    """
    fspl_db = p_tx_dbm + g_tx_dbi + g_rx_dbi - p_rx_dbm
    log10_d = (fspl_db - 20 * np.log10(freq_hz) - _FSPL_K) / 20.0
    d_m = 10.0 ** log10_d
    sigma_d = (d_m * np.log(10) / 20.0) * sigma_p_db
    return {
        "range_m": float(d_m),
        "sigma_m": float(sigma_d),
        "fspl_db": float(fspl_db),
    }
