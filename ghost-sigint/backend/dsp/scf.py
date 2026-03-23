"""
Spectral Correlation Function (SCF) via Strip Spectral Correlation Analyzer (SSCA).
Detects cyclostationary features — cyclic frequency α vs spectral frequency f.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FS = 1e6


def ssca(
    signal: NDArray[np.complex128],
    n_fft: int = 256,
    decimation: int = 8,
    fs: float = FS,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Strip Spectral Correlation Analyzer (SSCA).
    Computes |S_x^α(f)|² for cyclic frequency α and spectral freq f.

    Returns (alpha_hz, freq_hz, SCF_magnitude_db).
    """
    n = len(signal)
    # Downsampled length
    n_strips = n // decimation
    # Channel filter (rectangular for simplicity)
    strips = signal[:n_strips * decimation].reshape(n_strips, decimation)
    # DFT of each strip
    strips_fft = np.fft.fft(strips, n=n_fft, axis=1)  # (n_strips, n_fft)

    # SCF via outer product of FFT strips
    # S^α(f) ≈ (1/N) Σ_n X(n, f+α/2) · X*(n, f-α/2)
    scf = np.zeros((n_fft, n_fft), dtype=complex)
    for k in range(n_fft):
        for l in range(n_fft):
            # Cyclic freq: α = (k-l) * Δα, spectral: f = (k+l)/2 * Δf
            scf[k, l] = np.mean(strips_fft[:, k] * np.conj(strips_fft[:, l]))

    scf_mag = np.abs(scf)
    scf_db = 10 * np.log10(scf_mag / (scf_mag.max() + 1e-12) + 1e-12)

    df = fs / (decimation * n_fft)
    freq_hz = np.fft.fftfreq(n_fft, d=1.0 / (fs / decimation))
    alpha_hz = np.fft.fftfreq(n_fft, d=1.0 / (fs / decimation))

    return alpha_hz, freq_hz, scf_db
