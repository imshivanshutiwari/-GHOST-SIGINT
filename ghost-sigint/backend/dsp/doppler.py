"""
Range-Doppler processing: 2D FFT over N_chirps × N_range.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FS = 1e6
N_CHIRPS = 128
N_RANGE = 256
C = 3e8


def range_doppler(
    iq_data: NDArray[np.complex128],
    n_chirps: int = N_CHIRPS,
    n_range: int = N_RANGE,
    chirp_duration: float = 1e-3,
    bandwidth: float = 100e6,
    fc: float = 77e9,
    fs: float = FS,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    2D Range-Doppler processing:
    1. Reshape into (N_chirps, N_range)
    2. Range FFT along fast-time axis
    3. Doppler FFT along slow-time axis (with Hann window)

    Returns (range_m, velocity_mps, RD_map_db).
    """
    n_total = n_chirps * n_range
    if len(iq_data) < n_total:
        iq_data = np.pad(iq_data, (0, n_total - len(iq_data)))
    cube = iq_data[:n_total].reshape(n_chirps, n_range)

    # Range FFT
    range_fft = np.fft.fft(cube, n=n_range, axis=1)

    # Doppler FFT with Hann window
    hann = np.hanning(n_chirps)[:, np.newaxis]
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft * hann, n=n_chirps, axis=0), axes=0)

    rd_map = np.abs(doppler_fft) ** 2
    rd_map_db = 10 * np.log10(rd_map / (rd_map.max() + 1e-12) + 1e-12)

    # Physical axes
    k_r = bandwidth / (C * chirp_duration * fs)      # range resolution factor
    range_m = np.arange(n_range) / (2 * k_r + 1e-12)

    lambda_c = C / fc
    doppler_bins = np.fft.fftshift(np.fft.fftfreq(n_chirps, d=chirp_duration))
    velocity_mps = doppler_bins * lambda_c / 2

    return range_m, velocity_mps, rd_map_db
