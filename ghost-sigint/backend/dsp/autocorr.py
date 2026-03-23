"""
FFT-based autocorrelation and PRI (Pulse Repetition Interval) detection.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

FS = 1e6


def autocorrelation(
    signal: NDArray[np.complex128],
    max_lag: int | None = None,
) -> NDArray[np.complex128]:
    """
    Unbiased FFT-based autocorrelation.
    R[k] = (1/(N-|k|)) Σ x[n] x*[n-k]
    """
    n = len(signal)
    if max_lag is None:
        max_lag = n // 2
    # FFT-based: R = IFFT(|X(f)|²)
    n_fft = 1 << (2 * n - 1).bit_length()
    X = np.fft.fft(signal, n=n_fft)
    R_full = np.fft.ifft(X * np.conj(X)).real
    # Extract lags 0..max_lag
    R = R_full[:max_lag + 1]
    # Unbiased normalization
    norm = np.arange(n, n - max_lag - 1, -1, dtype=float)
    R = R / norm
    return R.astype(np.complex128)


def detect_pri(
    signal: NDArray[np.complex128],
    fs: float = FS,
    min_pri_s: float = 1e-5,
    max_pri_s: float = 1e-2,
) -> dict:
    """
    Detect Pulse Repetition Interval via peaks in autocorrelation magnitude.
    Returns dict with 'pri_samples', 'pri_s', 'prf_hz', 'confidence'.
    """
    max_lag = int(max_pri_s * fs)
    R = np.abs(autocorrelation(signal, max_lag=max_lag))
    # Normalize
    R = R / (R[0] + 1e-12)
    min_lag = int(min_pri_s * fs)
    # Find peaks beyond minimum PRI lag
    peaks, props = find_peaks(R[min_lag:], height=0.1, distance=min_lag)
    if len(peaks) == 0:
        return {"pri_samples": None, "pri_s": None, "prf_hz": None, "confidence": 0.0}
    # Best peak (highest amplitude)
    best = peaks[np.argmax(props["peak_heights"])] + min_lag
    return {
        "pri_samples": int(best),
        "pri_s": float(best / fs),
        "prf_hz": float(fs / best),
        "confidence": float(R[best]),
    }
