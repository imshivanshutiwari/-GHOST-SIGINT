"""
Matched filter via FFT cross-correlation.
Computes PSL (Peak Sidelobe Level) and ISLR (Integrated Sidelobe Ratio).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def matched_filter(
    received: NDArray[np.complex128],
    reference: NDArray[np.complex128],
) -> dict:
    """
    MF output: y[n] = x[n] ⋆ h[n], h[n] = conj(s[N-1-n])
    Implemented via FFT for efficiency: Y(f) = X(f) · conj(S(f))

    Returns dict with keys:
      ``output``  — complex MF output array (length len(received)+len(reference)-1)
      ``psl_db``  — Peak Sidelobe Level in dB
      ``islr_db`` — Integrated Sidelobe Ratio in dB
    """
    n = len(received) + len(reference) - 1
    # Zero-pad to next power of 2 for efficiency
    n_fft = 1 << (n - 1).bit_length()
    X = np.fft.fft(received, n=n_fft)
    S = np.fft.fft(reference, n=n_fft)
    Y = X * np.conj(S)
    output = np.fft.ifft(Y)[:n].astype(np.complex128)
    return {
        "output": output,
        "psl_db": psl(output),
        "islr_db": islr(output),
    }


def psl(mf_output: NDArray[np.complex128]) -> float:
    """
    Peak Sidelobe Level (dB) relative to main lobe peak.
    PSL = 20·log10(max_sidelobe / peak)
    """
    mag = np.abs(mf_output)
    peak_idx = np.argmax(mag)
    peak = mag[peak_idx]
    # Exclude main lobe (3 bins each side)
    mask = np.ones(len(mag), dtype=bool)
    lo = max(0, peak_idx - 3)
    hi = min(len(mag), peak_idx + 4)
    mask[lo:hi] = False
    if mask.any():
        max_sidelobe = mag[mask].max()
        return float(20 * np.log10(max_sidelobe / (peak + 1e-12)))
    return -np.inf


def islr(mf_output: NDArray[np.complex128], main_lobe_width: int = 7) -> float:
    """
    Integrated Sidelobe Ratio (dB).
    ISLR = 10·log10(P_sidelobe / P_mainlobe)
    """
    power = np.abs(mf_output) ** 2
    peak_idx = int(np.argmax(power))
    half = main_lobe_width // 2
    lo = max(0, peak_idx - half)
    hi = min(len(power), peak_idx + half + 1)
    p_main = np.sum(power[lo:hi])
    p_side = np.sum(power) - p_main
    return float(10 * np.log10(p_side / (p_main + 1e-12)))
