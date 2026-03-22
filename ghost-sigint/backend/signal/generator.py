"""
RF signal generators — all signals use exact physics equations.
Returns complex baseband arrays at FS = 1 MHz.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc

# Physical constants and parameters
FS = 1e6          # Sample rate Hz
C  = 3e8          # Speed of light m/s

# ─── Helper: RRC filter ────────────────────────────────────────────────────────

def _rrc_taps(sps: int, rolloff: float = 0.35, n_taps: int = 64) -> NDArray[np.float64]:
    """
    Root-Raised-Cosine filter impulse response.
    h(t) = [sin(π·t/T·(1-α)) + 4α·t/T·cos(π·t/T·(1+α))]
           / [π·t/T·(1 - (4α·t/T)²)]
    with l'Hôpital limit applied at t=0 and t=±T/(2α).
    """
    T = float(sps)
    alpha = rolloff
    t = np.arange(-(n_taps // 2), n_taps // 2 + 1, dtype=float)
    h = np.zeros(len(t))
    for i, ti in enumerate(t):
        x = ti / T
        if abs(ti) < 1e-10:
            h[i] = (1 - alpha + 4 * alpha / np.pi)
        elif abs(abs(4 * alpha * x) - 1.0) < 1e-10:
            h[i] = (alpha / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            numer = (np.sin(np.pi * x * (1 - alpha))
                     + 4 * alpha * x * np.cos(np.pi * x * (1 + alpha)))
            denom = np.pi * x * (1 - (4 * alpha * x) ** 2)
            h[i] = numer / denom
    h /= np.sqrt(np.sum(h ** 2))
    return h


def _bits(n: int, rng: np.random.Generator) -> NDArray[np.int_]:
    return rng.integers(0, 2, size=n)


# ─── 1. BPSK ──────────────────────────────────────────────────────────────────

def generate_bpsk(
    n_samples: int = 4096,
    fc: float = 0.0,
    sps: int = 8,
    amplitude: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    BPSK: s_bb = A · exp(jπ·m(t)), m∈{0,1} → phases {0,π}
    RRC pulse shaping with α=0.35.
    BER_theoretical = erfc(sqrt(Eb/N0)) (returned as metadata via .ber attr not used here).
    """
    rng = np.random.default_rng(seed)
    n_symbols = n_samples // sps + 1
    bits = _bits(n_symbols, rng)
    symbols = amplitude * np.exp(1j * np.pi * bits.astype(float))  # {+1,-1}
    # Upsample
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    # RRC filter
    h = _rrc_taps(sps, rolloff=0.35, n_taps=64)
    shaped = np.convolve(up, h, mode="full")[:n_samples]
    if fc != 0.0:
        t = np.arange(n_samples) / FS
        shaped *= np.exp(1j * 2 * np.pi * fc * t)
    return shaped.astype(np.complex128)


# ─── 2. QPSK ──────────────────────────────────────────────────────────────────

def generate_qpsk(
    n_samples: int = 4096,
    fc: float = 0.0,
    sps: int = 8,
    amplitude: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    QPSK Gray mapping: 00→π/4, 01→3π/4, 11→5π/4, 10→7π/4
    IQ: I=cos(φ_k), Q=sin(φ_k), RRC pulse shaping.
    """
    rng = np.random.default_rng(seed)
    n_symbols = n_samples // sps + 1
    dibits = rng.integers(0, 4, size=n_symbols)
    # Gray-coded phase map
    gray_phase = np.array([np.pi/4, 3*np.pi/4, 7*np.pi/4, 5*np.pi/4])  # 00,01,10,11
    phases = gray_phase[dibits]
    symbols = amplitude * np.exp(1j * phases)
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    h = _rrc_taps(sps, rolloff=0.35, n_taps=64)
    shaped = np.convolve(up, h, mode="full")[:n_samples]
    if fc != 0.0:
        t = np.arange(n_samples) / FS
        shaped *= np.exp(1j * 2 * np.pi * fc * t)
    return shaped.astype(np.complex128)


# ─── 3. 16-QAM ────────────────────────────────────────────────────────────────

def generate_16qam(
    n_samples: int = 4096,
    fc: float = 0.0,
    sps: int = 8,
    amplitude: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    16QAM: I,Q ∈ {-3,-1,+1,+3}/√10, 4-bit Gray-coded symbols.
    s(t) = I(t)·cos(2πfct) - Q(t)·sin(2πfct)
    """
    rng = np.random.default_rng(seed)
    constellation = np.array([-3, -1, 1, 3]) / np.sqrt(10.0)
    n_symbols = n_samples // sps + 1
    I = rng.choice(constellation, size=n_symbols)
    Q = rng.choice(constellation, size=n_symbols)
    symbols = amplitude * (I + 1j * Q)
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    h = _rrc_taps(sps, rolloff=0.35, n_taps=64)
    shaped = np.convolve(up, h, mode="full")[:n_samples]
    if fc != 0.0:
        t = np.arange(n_samples) / FS
        shaped *= np.exp(1j * 2 * np.pi * fc * t)
    return shaped.astype(np.complex128)


# ─── 4. OFDM ──────────────────────────────────────────────────────────────────

def generate_ofdm(
    n_samples: int = 4096,
    fc: float = 0.0,
    n_fft: int = 64,
    cp_len: int = 16,
    subcarrier_spacing: float = 15e3,
    pilot_indices: tuple = (0, 11, 25, 32, 43, 57),
    guard_bands: int = 6,
    amplitude: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    OFDM: N=64 subcarriers, Δf=15kHz, CP=Ts/4=16, QPSK per subcarrier.
    Pilots at {0,11,25,32,43,57}, guard bands 6 each side.
    """
    rng = np.random.default_rng(seed)
    symbol_len = n_fft + cp_len
    n_symbols = n_samples // symbol_len + 1

    gray_phase = np.array([np.pi/4, 3*np.pi/4, 7*np.pi/4, 5*np.pi/4])
    active_carriers = list(range(guard_bands, n_fft - guard_bands))
    data_carriers = [c for c in active_carriers if c not in pilot_indices]
    n_data = len(data_carriers)

    output = []
    for _ in range(n_symbols):
        fd = np.zeros(n_fft, dtype=complex)
        # Data subcarriers: QPSK
        dibits = rng.integers(0, 4, size=n_data)
        fd[data_carriers] = np.exp(1j * gray_phase[dibits])
        # Pilot subcarriers: known BPSK
        fd[list(pilot_indices)] = 1.0 + 0j
        # IFFT + cyclic prefix
        td = np.fft.ifft(fd) * np.sqrt(n_fft)
        with_cp = np.concatenate([td[-cp_len:], td])
        output.append(with_cp)

    sig = np.concatenate(output)[:n_samples] * amplitude
    if fc != 0.0:
        t = np.arange(n_samples) / FS
        sig *= np.exp(1j * 2 * np.pi * fc * t)
    return sig.astype(np.complex128)


# ─── 5. FHSS ──────────────────────────────────────────────────────────────────

def _lfsr_sequence(n_hops: int) -> list[int]:
    """LFSR x^15 + x^14 + 1 generating channel indices mod 80."""
    state = 0x5A5A  # non-zero seed
    seq = []
    for _ in range(n_hops):
        bit = ((state >> 14) ^ (state >> 13)) & 1
        state = ((state << 1) | bit) & 0x7FFF
        seq.append(state % 80)
    return seq


def generate_fhss(
    n_samples: int = 4096,
    fc_base: float = 100e3,
    t_hop: float = 1e-3,
    channel_spacing: float = 25e3,
    data_rate: float = 9600.0,
    amplitude: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    FHSS: LFSR x^15+x^14+1, Δf_ch=25kHz, 80 channels.
    Phase-continuous at hop boundaries. BPSK at 9.6kbps during dwell.
    """
    rng = np.random.default_rng(seed)
    sps = max(1, int(FS / data_rate))
    samples_per_hop = int(FS * t_hop)
    n_hops = n_samples // samples_per_hop + 2

    channels = _lfsr_sequence(n_hops)
    output = np.zeros(n_samples, dtype=complex)
    phase = 0.0

    for hop_idx, ch in enumerate(channels):
        start = hop_idx * samples_per_hop
        end = min(start + samples_per_hop, n_samples)
        if start >= n_samples:
            break
        n = end - start
        t_local = np.arange(n) / FS
        f_hop = fc_base + ch * channel_spacing
        # BPSK data during dwell
        n_bits = n // sps + 1
        bits = rng.integers(0, 2, size=n_bits)
        bpsk_symbols = (2 * bits - 1).astype(float)
        # Upsample
        data_up = np.repeat(bpsk_symbols, sps)[:n]
        # Phase-continuous carrier
        carrier = np.exp(1j * (2 * np.pi * f_hop * t_local + phase))
        output[start:end] = amplitude * data_up * carrier
        # Maintain phase continuity
        phase += 2 * np.pi * f_hop * (n / FS)
        phase = phase % (2 * np.pi)

    return output.astype(np.complex128)


# ─── 6. FMCW ──────────────────────────────────────────────────────────────────

def generate_fmcw(
    n_samples: int = 16384,
    fc: float = 77e9,
    bandwidth: float = 100e6,
    chirp_duration: float = 1e-3,
    n_chirps: int = 128,
    targets: list[tuple[float, float]] | None = None,
    amplitude: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    FMCW: s_tx = A·cos(2π·(fc + B/(2T)·t)·t)
    Beat signal with 3 simulated targets: (range_m, velocity_mps).
    f_range = 2·R·B/(c·T), f_doppler = 2·v·fc/c
    """
    if targets is None:
        targets = [(50.0, 10.0), (120.0, -5.0), (200.0, 25.0)]

    T = chirp_duration
    B = bandwidth
    k = B / T  # chirp rate

    samples_per_chirp = int(FS * T)
    output_list = []

    for chirp_i in range(n_chirps):
        t = np.arange(samples_per_chirp) / FS
        # TX chirp (baseband)
        tx = amplitude * np.exp(1j * np.pi * k * t ** 2)
        # Beat signal: sum of target returns
        beat = np.zeros(samples_per_chirp, dtype=complex)
        for R, v in targets:
            tau = 2 * R / C        # round-trip delay
            fd = 2 * v * fc / C   # Doppler
            f_beat = k * tau - fd  # beat frequency
            phase0 = -2 * np.pi * fc * tau + 2 * np.pi * fd * chirp_i * T
            beat += np.exp(1j * (2 * np.pi * f_beat * t + phase0))
        output_list.append(beat * tx.real)

    output = np.concatenate(output_list)[:n_samples]
    return output.astype(np.complex128)


# ─── 7. LFM Chirp ─────────────────────────────────────────────────────────────

def generate_lfm_chirp(
    n_samples: int = 4096,
    fc: float = 0.0,
    bandwidth: float = 10e6,
    pulse_duration: float = 10e-6,
    amplitude: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    LFM: s(t) = A·rect(t/τ)·exp(jπ·k·t²), k = B/τ
    """
    tau = pulse_duration
    k = bandwidth / tau
    n_pulse = int(FS * tau)
    t = np.arange(n_pulse) / FS - tau / 2  # centered at zero
    chirp = amplitude * np.exp(1j * np.pi * k * t ** 2)
    # Zero-pad or truncate to n_samples
    out = np.zeros(n_samples, dtype=complex)
    out[:min(n_pulse, n_samples)] = chirp[:min(n_pulse, n_samples)]
    if fc != 0.0:
        t_full = np.arange(n_samples) / FS
        out *= np.exp(1j * 2 * np.pi * fc * t_full)
    return out.astype(np.complex128)


# ─── 8. MANPADS ───────────────────────────────────────────────────────────────

def generate_manpads(
    n_samples: int = 4096,
    fc: float = 0.0,
    f_spin: float = 150.0,
    am_index: float = 0.5,
    pulse_prf: float = 1000.0,
    amplitude: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    MANPADS: pulse train + spin-scan AM.
    A(t) = A0·(1 + m·cos(2π·f_spin·t)), f_spin ∈ [100,200] Hz.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / FS
    # Spin-scan amplitude modulation
    A_t = amplitude * (1 + am_index * np.cos(2 * np.pi * f_spin * t))
    # Pulse train
    samples_per_pulse = int(FS / pulse_prf)
    pulse_width = max(1, samples_per_pulse // 10)
    envelope = np.zeros(n_samples, dtype=float)
    for start in range(0, n_samples, samples_per_pulse):
        end = min(start + pulse_width, n_samples)
        envelope[start:end] = 1.0
    # Carrier
    sig = A_t * envelope * np.exp(1j * 2 * np.pi * fc * t)
    return sig.astype(np.complex128)


# ─── 9. Bluetooth FHSS ────────────────────────────────────────────────────────

def generate_bluetooth(
    n_samples: int = 4096,
    fc_base: float = 2.402e9,
    hop_rate: float = 1600.0,
    gfsk_bt: float = 0.5,
    modulation_index: float = 0.35,
    amplitude: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    Bluetooth FHSS: 79 channels, GFSK BT=0.5, h=0.35, 1600 hops/sec.
    Simulated at baseband (fc_base offset not applied to avoid aliasing).
    """
    rng = np.random.default_rng(seed)
    samples_per_hop = int(FS / hop_rate)
    n_hops = n_samples // samples_per_hop + 2
    n_channels = 79

    # GFSK Gaussian filter
    sps = 8  # samples per symbol at 1 Mbps → FS/1e6 = 1
    # Gaussian filter for BT=0.5
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * gfsk_bt)
    gauss_len = 4 * sps
    t_g = np.arange(-gauss_len // 2, gauss_len // 2 + 1) / sps
    gauss_filter = np.exp(-t_g ** 2 / (2 * sigma ** 2))
    gauss_filter /= gauss_filter.sum()

    output = np.zeros(n_samples, dtype=complex)
    phase = 0.0
    hop_channels = rng.integers(0, n_channels, size=n_hops)

    for hop_i, ch in enumerate(hop_channels):
        start = hop_i * samples_per_hop
        end = min(start + samples_per_hop, n_samples)
        if start >= n_samples:
            break
        n = end - start
        # Random bits for this hop
        bits = rng.integers(0, 2, size=n // sps + 4)
        nrz = (2 * bits - 1).astype(float)
        nrz_up = np.repeat(nrz, sps)
        # Gaussian smoothing
        smoothed = np.convolve(nrz_up, gauss_filter, mode="same")[: n]
        # FM modulation
        phase_inc = np.pi * modulation_index * smoothed / sps
        phases = np.cumsum(phase_inc) + phase
        output[start:end] = amplitude * np.exp(1j * phases)
        phase = phases[-1] if len(phases) > 0 else phase

    return output.astype(np.complex128)


# ─── 10. WiFi OFDM ────────────────────────────────────────────────────────────

def generate_wifi(
    n_samples: int = 4096,
    bandwidth: float = 20e6,
    n_fft: int = 64,
    cp_len: int = 16,
    n_data_carriers: int = 52,
    n_pilot_carriers: int = 4,
    amplitude: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """
    WiFi OFDM: 20MHz, 52 data + 4 pilot subcarriers, 64-FFT.
    Short Training Sequence (STS) prepended (12 active subcarriers, repeated 10×).
    """
    rng = np.random.default_rng(seed)
    symbol_len = n_fft + cp_len

    # Short Training Sequence subcarriers (802.11a standard subset)
    sts_carriers = [-24, -20, -16, -12, -8, -4, 4, 8, 12, 16, 20, 24]
    sts_values = np.array([1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1], dtype=complex)
    sts_fd = np.zeros(n_fft, dtype=complex)
    for idx, val in zip(sts_carriers, sts_values):
        sts_fd[idx % n_fft] = val * np.sqrt(13.0 / 6)
    sts_td = np.fft.ifft(sts_fd) * np.sqrt(n_fft)
    # STS is 16 samples repeated 10 times
    sts = np.tile(sts_td[:16], 10)

    # Data/pilot carrier indices (802.11a style)
    pilot_indices = [7, 21, 43, 57]  # mapped into 0..63
    data_indices = [
        i for i in range(6, 59)
        if i not in pilot_indices and i not in (32,)  # DC null
    ][:n_data_carriers]

    output = [sts]
    n_data_syms = (n_samples - len(sts)) // symbol_len + 1

    for _ in range(n_data_syms):
        fd = np.zeros(n_fft, dtype=complex)
        # 16QAM on data carriers
        constellation = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        I = rng.choice(constellation, size=len(data_indices))
        Q = rng.choice(constellation, size=len(data_indices))
        fd[data_indices] = I + 1j * Q
        # Known pilots
        fd[pilot_indices] = [1, 1, 1, -1]
        td = np.fft.ifft(fd) * np.sqrt(n_fft)
        with_cp = np.concatenate([td[-cp_len:], td])
        output.append(with_cp)

    sig = np.concatenate(output)[:n_samples] * amplitude
    return sig.astype(np.complex128)


# ─── Dispatcher ───────────────────────────────────────────────────────────────

_GENERATORS = {
    "BPSK":       generate_bpsk,
    "QPSK":       generate_qpsk,
    "16QAM":      generate_16qam,
    "OFDM":       generate_ofdm,
    "FHSS":       generate_fhss,
    "FMCW":       generate_fmcw,
    "LFM_CHIRP":  generate_lfm_chirp,
    "MANPADS":    generate_manpads,
    "Bluetooth":  generate_bluetooth,
    "WiFi":       generate_wifi,
}


def generate_signal(
    signal_class: str,
    n_samples: int = 4096,
    **kwargs,
) -> NDArray[np.complex128]:
    """Generate a named signal class. Raises KeyError for unknown classes."""
    fn = _GENERATORS[signal_class]
    return fn(n_samples=n_samples, **kwargs)
