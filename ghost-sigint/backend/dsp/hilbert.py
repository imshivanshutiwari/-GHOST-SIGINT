"""
FFT-based analytic signal (Hilbert transform) and instantaneous
amplitude / phase / frequency extraction.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FS = 1e6


def analytic_signal(x: NDArray) -> NDArray[np.complex128]:
    """
    Compute analytic signal via FFT-based Hilbert transform.
    H(f) = 0 for f < 0, 2·X(f) for f > 0, X(f) for f = 0.
    """
    n = len(x)
    X = np.fft.fft(x.real)
    H = np.zeros(n, dtype=complex)
    if n % 2 == 0:
        H[0] = X[0]
        H[1:n // 2] = 2 * X[1:n // 2]
        H[n // 2] = X[n // 2]
    else:
        H[0] = X[0]
        H[1:(n + 1) // 2] = 2 * X[1:(n + 1) // 2]
    return np.fft.ifft(H).astype(np.complex128)


def instantaneous_amplitude(signal: NDArray[np.complex128]) -> NDArray[np.float64]:
    """|s(t)| — envelope."""
    return np.abs(signal)


def instantaneous_phase(signal: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Unwrapped instantaneous phase (radians)."""
    return np.unwrap(np.angle(signal))


def instantaneous_frequency(
    signal: NDArray[np.complex128],
    fs: float = FS,
) -> NDArray[np.float64]:
    """
    f_i(t) = (1/2π) · dφ/dt  via numpy.gradient.
    Returns frequency array in Hz.
    """
    phi = instantaneous_phase(signal)
    dphi = np.gradient(phi)
    return (dphi * fs / (2 * np.pi)).astype(np.float64)
