"""
Butterworth bandpass filter — order N=6, SOS form via scipy.
Provides frequency response and Bode data helpers.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_sig

FS = 1e6


def butterworth_bandpass(
    low_hz: float,
    high_hz: float,
    order: int = 6,
    fs: float = FS,
) -> NDArray[np.float64]:
    """Return SOS coefficients for an order-N Butterworth bandpass filter."""
    nyq = fs / 2
    low_n = low_hz / nyq
    high_n = high_hz / nyq
    sos = sp_sig.butter(order, [low_n, high_n], btype="bandpass", output="sos")
    return sos


def apply_filter(
    sos: NDArray[np.float64],
    signal: NDArray,
) -> NDArray[np.complex128]:
    """Apply SOS filter (zero-phase via sosfiltfilt)."""
    if np.iscomplexobj(signal):
        r = sp_sig.sosfiltfilt(sos, signal.real)
        i = sp_sig.sosfiltfilt(sos, signal.imag)
        return (r + 1j * i).astype(np.complex128)
    return sp_sig.sosfiltfilt(sos, signal).astype(np.complex128)


def frequency_response(
    sos: NDArray[np.float64],
    n_points: int = 2048,
    fs: float = FS,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """
    Return (frequencies_hz, H) — complex frequency response.
    Magnitude in dB: 20·log10(|H|).
    """
    worN = np.linspace(0, np.pi, n_points, endpoint=False)
    w, h = sp_sig.sosfreqz(sos, worN=worN, fs=fs)
    return w, h


def bode(
    sos: NDArray[np.float64],
    n_points: int = 2048,
    fs: float = FS,
) -> dict[str, NDArray]:
    """Return dict with 'freq_hz', 'magnitude_db', 'phase_deg'."""
    w, h = frequency_response(sos, n_points=n_points, fs=fs)
    return {
        "freq_hz": w,
        "magnitude_db": 20 * np.log10(np.abs(h) + 1e-30),
        "phase_deg": np.degrees(np.unwrap(np.angle(h))),
    }


def design_bandpass(
    fl: float,
    fh: float,
    fs: float = FS,
    order: int = 6,
) -> NDArray[np.float64]:
    """
    Convenience alias for :func:`butterworth_bandpass` using keyword-argument
    names ``fl``/``fh`` instead of ``low_hz``/``high_hz``.
    Returns SOS coefficients (shape ``(n_sections, 6)``).
    """
    return butterworth_bandpass(low_hz=fl, high_hz=fh, order=order, fs=fs)
