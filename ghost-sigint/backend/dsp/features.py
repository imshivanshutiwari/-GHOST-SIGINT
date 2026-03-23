"""
RF fingerprint feature extraction: 7 features.
(carrier_offset, IQ_amp_imbalance, IQ_phase_imbalance,
 transient_slope, AM_AM, spectral_flatness, kurtosis)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis as scipy_kurtosis

FS = 1e6
N_RF_FEATURES = 7


def extract_features(
    signal: NDArray[np.complex128],
    fs: float = FS,
) -> NDArray[np.float64]:
    """
    Extract 7 RF fingerprint features from a complex baseband signal.

    Features:
      0. Carrier frequency offset (Hz) — via instantaneous frequency mean
      1. IQ amplitude imbalance — std(|I|)/std(|Q|) ratio deviation from 1
      2. IQ phase imbalance — mean(I·Q) normalized
      3. Transient slope — max gradient of envelope in first 10% of samples
      4. AM-AM nonlinearity — correlation(|input|, |output|) deviation
      5. Spectral flatness — geometric/arithmetic mean of power spectrum
      6. Kurtosis of envelope
    """
    n = len(signal)
    I = signal.real
    Q = signal.imag
    envelope = np.abs(signal)

    # 0. Carrier frequency offset via instantaneous frequency
    phase = np.unwrap(np.angle(signal))
    inst_freq = np.gradient(phase) * fs / (2 * np.pi)
    carrier_offset = float(np.mean(inst_freq))

    # 1. IQ amplitude imbalance
    std_I = float(np.std(I)) + 1e-12
    std_Q = float(np.std(Q)) + 1e-12
    iq_amp_imbalance = float(std_I / std_Q - 1.0)

    # 2. IQ phase imbalance
    iq_phase_imbalance = float(np.mean(I * Q) / (std_I * std_Q + 1e-12))

    # 3. Transient slope (first 10% of samples)
    n_transient = max(2, n // 10)
    transient = envelope[:n_transient]
    transient_slope = float(np.max(np.abs(np.gradient(transient))))

    # 4. AM-AM (correlation between input envelope and output envelope proxy)
    # Use correlation of envelope with its delayed version as AM-AM proxy
    if n > 1:
        am_am = float(np.corrcoef(envelope[:-1], envelope[1:])[0, 1])
    else:
        am_am = 0.0

    # 5. Spectral flatness
    power_spectrum = np.abs(np.fft.fft(signal)) ** 2
    power_spectrum = power_spectrum[: n // 2]
    geo_mean = np.exp(np.mean(np.log(power_spectrum + 1e-12)))
    arith_mean = float(np.mean(power_spectrum)) + 1e-12
    spectral_flatness = float(geo_mean / arith_mean)

    # 6. Kurtosis of envelope
    kurt = float(scipy_kurtosis(envelope, fisher=True))

    features = np.array([
        carrier_offset,
        iq_amp_imbalance,
        iq_phase_imbalance,
        transient_slope,
        am_am,
        spectral_flatness,
        kurt,
    ], dtype=np.float64)

    return features


# Alias used by dsp tests
extract_rf_features = extract_features
