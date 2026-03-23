"""
Physically-accurate RF channel impairments.
All models use proper statistical/physics equations.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FS = 1e6  # Sample rate Hz


def awgn(
    signal: NDArray[np.complex128],
    snr_db: float,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    AWGN via Box-Muller transform.
    SNR computed from signal power vs noise power.
    """
    rng = np.random.default_rng(seed)
    sig_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    sigma = np.sqrt(noise_power / 2)
    # Box-Muller transform for Gaussian noise
    u1 = rng.uniform(1e-12, 1.0, size=len(signal))
    u2 = rng.uniform(0.0, 1.0, size=len(signal))
    n_real = sigma * np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    n_imag = sigma * np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    return (signal + n_real + 1j * n_imag).astype(np.complex128)


def rayleigh_fading(
    signal: NDArray[np.complex128],
    fd_max: float = 50.0,
    n_paths: int = 20,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    Rayleigh fading via Clarke's model (sum-of-sinusoids).
    N=20 paths, uniform AoA distribution.
    """
    rng = np.random.default_rng(seed)
    N = n_paths
    t = np.arange(len(signal)) / FS
    # Clarke's model: N equal-power paths with uniform AoA
    aoa = np.linspace(0, 2 * np.pi, N, endpoint=False)
    phases = rng.uniform(0, 2 * np.pi, size=N)
    # In-phase and quadrature fading processes
    g_i = np.zeros(len(signal))
    g_q = np.zeros(len(signal))
    for k in range(N):
        fd_k = fd_max * np.cos(aoa[k])
        g_i += np.cos(2 * np.pi * fd_k * t + phases[k])
        g_q += np.cos(2 * np.pi * fd_k * t + phases[k] + np.pi / 2)
    # Normalize to unit power
    g = (g_i + 1j * g_q) / np.sqrt(N)
    return (signal * g).astype(np.complex128)


def phase_noise(
    signal: NDArray[np.complex128],
    linewidth_hz: float = 1e3,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    Phase noise via Wiener (random walk) process.
    σ² per sample = 2π·Δν/FS (Lorentzian linewidth model).
    """
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(2 * np.pi * linewidth_hz / FS)
    increments = rng.normal(0, sigma, size=len(signal))
    phase_walk = np.cumsum(increments)
    return (signal * np.exp(1j * phase_walk)).astype(np.complex128)


def cfo(
    signal: NDArray[np.complex128],
    delta_f: float = 1e3,
) -> NDArray[np.complex128]:
    """
    Carrier Frequency Offset: x_cfo(t) = x(t) · exp(j2π·Δf·t)
    """
    t = np.arange(len(signal)) / FS
    return (signal * np.exp(1j * 2 * np.pi * delta_f * t)).astype(np.complex128)


def iq_imbalance(
    signal: NDArray[np.complex128],
    amplitude_imbalance_db: float = 0.5,
    phase_imbalance_deg: float = 2.0,
) -> NDArray[np.complex128]:
    """
    Full IQ imbalance model:
    I_out = (1 + ε/2)·cos(φ/2)·I - (1 + ε/2)·sin(φ/2)·Q
    Q_out = (1 - ε/2)·sin(φ/2)·I + (1 - ε/2)·cos(φ/2)·Q
    where ε = amplitude imbalance (linear), φ = phase imbalance (rad).
    """
    eps = 10 ** (amplitude_imbalance_db / 20) - 1.0
    phi = np.radians(phase_imbalance_deg)
    I = signal.real
    Q = signal.imag
    I_out = (1 + eps / 2) * np.cos(phi / 2) * I - (1 + eps / 2) * np.sin(phi / 2) * Q
    Q_out = (1 - eps / 2) * np.sin(phi / 2) * I + (1 - eps / 2) * np.cos(phi / 2) * Q
    return (I_out + 1j * Q_out).astype(np.complex128)


def multipath(
    signal: NDArray[np.complex128],
    n_paths: int = 5,
    max_delay_samples: int = 20,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    Multipath channel: L=5 paths, Rayleigh amplitudes, exponential delays.
    h(τ) = Σ a_l · δ(τ - τ_l)
    """
    rng = np.random.default_rng(seed)
    # Exponential delay profile
    delays = np.sort(rng.integers(0, max_delay_samples, size=n_paths))
    delays[0] = 0  # LOS path
    # Rayleigh-distributed amplitudes (exponential decay)
    decay = np.exp(-delays / (max_delay_samples / 3))
    rayleigh_amp = np.abs(rng.standard_normal(n_paths) + 1j * rng.standard_normal(n_paths)) / np.sqrt(2)
    coeffs = rayleigh_amp * decay
    coeffs /= np.sqrt(np.sum(np.abs(coeffs) ** 2))  # normalize

    out = np.zeros(len(signal), dtype=complex)
    for l, (d, c) in enumerate(zip(delays, coeffs)):
        if d == 0:
            out += c * signal
        else:
            out[d:] += c * signal[:-d]
    return out.astype(np.complex128)


def apply_channel(
    signal: NDArray[np.complex128],
    snr_db: float = 20.0,
    apply_fading: bool = True,
    apply_phase_noise: bool = True,
    apply_cfo: bool = True,
    apply_iq_imbalance: bool = True,
    apply_multipath: bool = True,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """Apply all impairments in a standard order."""
    rng = np.random.default_rng(seed)
    s = signal.copy()
    if apply_fading:
        s = rayleigh_fading(s, seed=int(rng.integers(0, 2**31)))
    if apply_phase_noise:
        s = phase_noise(s, seed=int(rng.integers(0, 2**31)))
    if apply_cfo:
        s = cfo(s, delta_f=rng.uniform(-5e3, 5e3))
    if apply_iq_imbalance:
        s = iq_imbalance(s)
    if apply_multipath:
        s = multipath(s, seed=int(rng.integers(0, 2**31)))
    s = awgn(s, snr_db=snr_db, seed=int(rng.integers(0, 2**31)))
    return s


# Convenience alias — same as awgn() but with a more descriptive name.
apply_awgn = awgn
