"""
Ambiguity function: χ(τ, f_d) via 2D FFT.
Axes returned in physical units (seconds, Hz).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FS = 1e6


def ambiguity_function(
    signal: NDArray[np.complex128],
    n_delay: int = 256,
    n_doppler: int = 256,
    fs: float = FS,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Compute the narrowband ambiguity function via 2D FFT.
    |χ(τ, f_d)|² = |∫ u(t) u*(t-τ) exp(-j2πf_d t) dt|²

    Returns (delay_s, doppler_hz, chi2) where chi2 is |χ|² in dB.
    """
    n = min(len(signal), n_delay)
    signal = signal[:n]

    # Build delay-Doppler matrix via sliding correlation
    matrix = np.zeros((n_doppler, n_delay), dtype=complex)
    t = np.arange(n) / fs

    for tau_idx in range(n_delay):
        tau = tau_idx
        if tau == 0:
            product = signal * np.conj(signal)
        else:
            product = signal[tau:] * np.conj(signal[:-tau]) if tau < n else np.zeros(1, dtype=complex)
        # Pad to n for consistent FFT
        pad_len = n - len(product)
        if pad_len > 0:
            product = np.concatenate([product, np.zeros(pad_len, dtype=complex)])
        matrix[:, tau_idx] = np.fft.fft(product * np.exp(-1j * 2 * np.pi * 0 * t), n=n_doppler)

    chi2 = np.abs(matrix) ** 2
    chi2_db = 10 * np.log10(chi2 / (chi2.max() + 1e-12) + 1e-12)

    delay_s = np.arange(n_delay) / fs
    doppler_hz = np.fft.fftfreq(n_doppler, d=1.0 / fs)

    return delay_s, doppler_hz, chi2_db
