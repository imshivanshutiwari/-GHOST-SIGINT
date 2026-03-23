"""
Automatic Gain Control — single-pole IIR AGC.
α_attack  = 1 - exp(-1 / (FS · t_attack))
α_release = 1 - exp(-1 / (FS · t_release))
P_target  = 10^(TARGET_POWER_DBM / 10)
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

FS = 1e6
TARGET_POWER_DB: float = -20.0   # dBFS
_P_TARGET: float = 10 ** (TARGET_POWER_DB / 10)
_ALPHA_ATTACK: float = 1 - np.exp(-1 / (FS * 1e-3))    # 1 ms
_ALPHA_RELEASE: float = 1 - np.exp(-1 / (FS * 0.1))    # 100 ms


def agc(
    signal: NDArray[np.complex128],
    target_power_db: float = TARGET_POWER_DB,
    alpha_attack: float | None = None,
    alpha_release: float | None = None,
) -> NDArray[np.complex128]:
    """
    Apply single-pole IIR AGC. Returns gain-corrected signal.
    Power estimate P̂[n] tracks instantaneous |s[n]|² with asymmetric attack/release.
    """
    p_target = 10 ** (target_power_db / 10)
    aa = alpha_attack if alpha_attack is not None else _ALPHA_ATTACK
    ar = alpha_release if alpha_release is not None else _ALPHA_RELEASE

    n = len(signal)
    out = np.empty(n, dtype=np.complex128)
    p_hat = p_target  # initialise at target

    for i in range(n):
        inst_power = (signal[i].real ** 2 + signal[i].imag ** 2)
        if inst_power > p_hat:
            p_hat = aa * inst_power + (1 - aa) * p_hat
        else:
            p_hat = ar * inst_power + (1 - ar) * p_hat
        p_hat = max(p_hat, 1e-12)
        gain = np.sqrt(p_target / p_hat)
        out[i] = signal[i] * gain

    return out


def measure_power_db(signal: NDArray[np.complex128]) -> float:
    """Return average power in dBFS."""
    p = float(np.mean(np.abs(signal) ** 2))
    return 10 * np.log10(p + 1e-30)


def apply_agc(
    signal: NDArray,
    fs: float = FS,
    target_power_db: float = TARGET_POWER_DB,
) -> NDArray[np.complex128]:
    """
    Convenience wrapper around :func:`agc` that accepts a ``fs`` parameter
    and automatically derives attack/release time constants from the sample rate.
    """
    aa = 1.0 - np.exp(-1.0 / (fs * 1e-3))   # 1 ms attack
    ar = 1.0 - np.exp(-1.0 / (fs * 0.1))    # 100 ms release
    sig = signal.astype(np.complex128) if not np.iscomplexobj(signal) else signal
    return agc(sig, target_power_db=target_power_db, alpha_attack=aa, alpha_release=ar)
