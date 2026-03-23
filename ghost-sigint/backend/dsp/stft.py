"""
Short-Time Fourier Transform, Mel filterbank, log-mel spectrogram,
delta and delta-delta features.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import stft as scipy_stft
from scipy.fft import fft

FS = 1e6
N_FFT = 256
HOP = 128
N_MEL = 128
F_MIN = 0.0
F_MAX = FS / 2


def compute_stft(
    signal: NDArray,
    n_fft: int = N_FFT,
    hop: int = HOP,
    fs: float = FS,
) -> dict:
    """
    STFT with Hann window.
    Returns dict with keys: ``freq_hz``, ``time_s``, ``S_complex``, ``S_db``.
    """
    f, t, Zxx = scipy_stft(
        signal if not np.iscomplexobj(signal) else signal.real,
        fs=fs,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        return_onesided=not np.iscomplexobj(signal),
    )
    S_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
    return {
        "freq_hz": f.astype(np.float64),
        "time_s": t.astype(np.float64),
        "S_complex": Zxx,
        "S_db": S_db.astype(np.float64),
    }


def mel_filterbank(
    n_fft: int = N_FFT,
    n_mel: int = N_MEL,
    fs: float = FS,
    f_min: float = F_MIN,
    f_max: float | None = None,
) -> NDArray[np.float64]:
    """
    Mel filterbank matrix: shape (n_mel, n_fft // 2 + 1).
    Mel scale: m = 2595·log10(1 + f/700).
    """
    if f_max is None:
        f_max = fs / 2
    n_freqs = n_fft // 2 + 1
    # Mel-scale frequency mapping
    mel_min = 2595 * np.log10(1 + f_min / 700)
    mel_max = 2595 * np.log10(1 + f_max / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mel + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / fs).astype(int)

    fbank = np.zeros((n_mel, n_freqs))
    for m in range(1, n_mel + 1):
        lo, center, hi = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(lo, center):
            if center != lo:
                fbank[m - 1, k] = (k - lo) / (center - lo)
        for k in range(center, hi):
            if hi != center:
                fbank[m - 1, k] = (hi - k) / (hi - center)
    return fbank


def log_mel_spectrogram(
    signal: NDArray,
    n_fft: int = N_FFT,
    hop: int = HOP,
    n_mel: int = N_MEL,
    fs: float = FS,
) -> NDArray[np.float64]:
    """
    Log-mel spectrogram: shape (n_mel, T).
    """
    out = compute_stft(signal, n_fft=n_fft, hop=hop, fs=fs)
    power = np.abs(out["S_complex"]) ** 2
    fb = mel_filterbank(n_fft=n_fft, n_mel=n_mel, fs=fs)
    mel = fb @ power
    return (10 * np.log10(mel + 1e-10)).astype(np.float64)


def compute_mel_spectrogram(
    signal: NDArray,
    n_fft: int = N_FFT,
    hop: int = HOP,
    n_mel: int = N_MEL,
    fs: float = FS,
) -> dict:
    """
    Compute log-mel spectrogram and return as a dict.
    Returns ``{"log_mel": ndarray(n_mel, T)}``.
    """
    log_mel = log_mel_spectrogram(signal, n_fft=n_fft, hop=hop, n_mel=n_mel, fs=fs)
    return {"log_mel": log_mel}


def delta_features(
    spec: NDArray[np.float64],
    width: int = 9,
) -> NDArray[np.float64]:
    """
    Compute delta (first-order derivative) of spectrogram.
    Uses least-squares linear regression over ±(width//2) frames.
    """
    pad = width // 2
    padded = np.pad(spec, ((0, 0), (pad, pad)), mode="edge")
    delta = np.zeros_like(spec)
    denom = 2 * sum(i ** 2 for i in range(1, pad + 1))
    for t in range(spec.shape[1]):
        for n in range(1, pad + 1):
            delta[:, t] += n * (padded[:, t + pad + n] - padded[:, t + pad - n])
    delta /= max(denom, 1e-10)
    return delta
