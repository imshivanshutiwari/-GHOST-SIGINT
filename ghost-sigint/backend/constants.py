"""
Global constants for GHOST-SIGINT backend.
All magic numbers live here.
"""
from typing import List

# Sampling / RF
FS: float = 1e6            # Sample rate (Hz)
C: float = 3e8             # Speed of light (m/s)
TARGET_POWER: float = -20.0  # Target AGC power (dBFS)

# Signal classes
SIGNAL_CLASSES: List[str] = [
    "BPSK", "QPSK", "16QAM", "OFDM",
    "FHSS", "FMCW", "LFM_CHIRP", "MANPADS",
    "Bluetooth", "WiFi",
]
N_CLASSES: int = len(SIGNAL_CLASSES)  # 10

# OFDM
OFDM_N_FFT: int = 64
OFDM_CP: int = OFDM_N_FFT // 4      # 16
OFDM_SUBCARRIER_SPACING: float = 15e3  # Hz
OFDM_PILOT_INDICES: List[int] = [0, 11, 25, 32, 43, 57]
OFDM_GUARD_BANDS: int = 6

# FHSS / Bluetooth
FHSS_N_CHANNELS: int = 80
FHSS_CHANNEL_SPACING: float = 25e3  # Hz
FHSS_DATA_RATE: float = 9600.0      # bps
BT_N_CHANNELS: int = 79
BT_HOP_RATE: float = 1600.0         # hops/sec
BT_GFSK_BT: float = 0.5
BT_GFSK_H: float = 0.35

# FMCW / LFM
FMCW_N_CHIRPS: int = 128
FMCW_BANDWIDTH: float = 100e6      # Hz
FMCW_CHIRP_DURATION: float = 1e-3  # s
LFM_BANDWIDTH: float = 10e6        # Hz
LFM_PULSE_DURATION: float = 10e-6  # s

# MANPADS
MANPADS_SPIN_FREQ_MIN: float = 100.0  # Hz
MANPADS_SPIN_FREQ_MAX: float = 200.0  # Hz
MANPADS_AM_INDEX: float = 0.5

# RRC filter
RRC_ROLLOFF: float = 0.35
RRC_NTAPS: int = 64

# DSP
STFT_N_FFT: int = 256
STFT_HOP: int = 128
MEL_FILTERS: int = 128
AGC_ATTACK_TIME: float = 1e-3   # s
AGC_RELEASE_TIME: float = 0.1   # s
CFAR_N_GUARD: int = 2
CFAR_N_REF: int = 8
CFAR_PFA: float = 1e-6
DOPPLER_N_CHIRPS: int = 128
DOPPLER_N_RANGE: int = 256

# ML / Models
MODEL_INPUT_MEL: tuple = (3, 128, 128)
MODEL_INPUT_IQ: tuple = (2, 1024)
MODEL_INPUT_STFT: tuple = (32, 256)
MODEL_INPUT_MEL_FRAMES: tuple = (50, 128)
MC_DROPOUT_PASSES: int = 20
UNCERTAINTY_THRESHOLD: float = 0.20
ADVERSARIAL_EPSILON: float = 0.01
ADVERSARIAL_ALPHA: float = 0.002
ADVERSARIAL_PGD_STEPS: int = 10
LABEL_SMOOTHING: float = 0.1
LEARNING_RATE: float = 3e-4
WEIGHT_DECAY: float = 0.01
TRAIN_SAMPLES_PER_CLASS: int = 10_000
SNR_MIN: float = -20.0  # dB
SNR_MAX: float = 30.0   # dB

# Geo / Array
MUSIC_N_ELEMENTS: int = 8   # ULA elements
MUSIC_FINE_SEARCH_DEG: float = 5.0
KALMAN_CHI2_GATE: float = 5.991  # χ²(0.95, 2)
TDOA_M_SENSORS: int = 4
ICA_MAX_ITER: int = 500
ICA_TOL: float = 1e-6

# RF Fingerprint thresholds
FINGERPRINT_KNOWN_THRESHOLD: float = 0.85
FINGERPRINT_SUSPECTED_THRESHOLD: float = 0.60
N_RF_FEATURES: int = 7

# API / Auth
DEFAULT_ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
REFRESH_TOKEN_EXPIRE_DAYS: int = 7
BCRYPT_ROUNDS: int = 12
MAX_CONNECTIONS_PER_USER: int = 5
RATE_LIMIT_PER_MINUTE: int = 100
LOGIN_LOCKOUT_ATTEMPTS: int = 5
LOGIN_LOCKOUT_MINUTES: int = 15

# WebSocket channels and rates (Hz)
WS_SPECTRUM_RATE: float = 10.0
WS_STFT_RATE: float = 5.0
WS_GEOLOCATION_RATE: float = 2.0
WS_METRICS_RATE: float = 1.0
