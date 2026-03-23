"""Digital Signal Processing modules."""
from .stft import compute_stft, compute_mel_spectrogram, log_mel_spectrogram, mel_filterbank, delta_features
from .agc import agc, apply_agc, measure_power_db
from .cfar import ca_cfar, os_cfar, detect
from .filters import butterworth_bandpass, design_bandpass, apply_filter, frequency_response, bode
from .matched_filter import matched_filter, psl, islr
from .features import extract_features, extract_rf_features
from .hilbert import analytic_signal
from .autocorr import autocorrelation
from .doppler import range_doppler
from .wvd import pwvd
from .scf import ssca
from .ambiguity import ambiguity_function

__all__ = [
    "compute_stft", "compute_mel_spectrogram", "log_mel_spectrogram",
    "mel_filterbank", "delta_features",
    "agc", "apply_agc", "measure_power_db",
    "ca_cfar", "os_cfar", "detect",
    "butterworth_bandpass", "design_bandpass", "apply_filter",
    "frequency_response", "bode",
    "matched_filter", "psl", "islr",
    "extract_features", "extract_rf_features",
    "analytic_signal",
    "autocorrelation",
    "range_doppler",
    "pwvd",
    "ssca",
    "ambiguity_function",
]
