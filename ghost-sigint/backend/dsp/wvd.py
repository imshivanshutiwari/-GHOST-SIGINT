"""
Pseudo Wigner-Ville Distribution (PWVD) with Hann window.
window=256 samples.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FS = 1e6
WINDOW_SIZE = 256


def pwvd(
    signal: NDArray[np.complex128],
    window_size: int = WINDOW_SIZE,
    fs: float = FS,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Pseudo Wigner-Ville Distribution.
    W(t, f) = ∫ w(τ) x(t+τ/2) x*(t-τ/2) exp(-j2πfτ) dτ
    Windowed (Pseudo) version uses finite Hann window.

    Returns (time_s, freq_hz, W_db) where W is the PWVD in dB.
    """
    n = len(signal)
    half_win = window_size // 2
    hann = np.hanning(window_size)

    # Output: (n_freqs, n_time) = (window_size, n)
    W = np.zeros((window_size, n), dtype=float)

    for t_idx in range(n):
        # Build lag product with Hann windowing
        lag_product = np.zeros(window_size, dtype=complex)
        for tau_idx in range(half_win):
            tau = tau_idx
            t_plus = t_idx + tau
            t_minus = t_idx - tau
            if 0 <= t_plus < n and 0 <= t_minus < n:
                lag_product[tau] = hann[half_win + tau] * signal[t_plus] * np.conj(signal[t_minus])
            if tau > 0:
                t_plus2 = t_idx - tau
                t_minus2 = t_idx + tau
                if 0 <= t_plus2 < n and 0 <= t_minus2 < n:
                    lag_product[window_size - tau] = hann[half_win - tau] * signal[t_plus2] * np.conj(signal[t_minus2])
        W[:, t_idx] = np.fft.fft(lag_product).real

    W = np.abs(W)
    W_db = 10 * np.log10(W / (W.max() + 1e-12) + 1e-12)

    time_s = np.arange(n) / fs
    freq_hz = np.fft.fftfreq(window_size, d=1.0 / fs)

    return time_s, freq_hz, W_db
