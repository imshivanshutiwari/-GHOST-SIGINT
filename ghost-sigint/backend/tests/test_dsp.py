from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pytest
from constants import FS, MEL_FILTERS
from dsp.agc import apply_agc
from dsp.filters import design_bandpass, apply_filter
from dsp.stft import compute_stft, compute_mel_spectrogram
from dsp.cfar import ca_cfar, os_cfar
from dsp.matched_filter import matched_filter
from dsp.features import extract_rf_features

def test_agc_target_power():
    t = np.linspace(0, 0.01, int(0.01*FS))
    x = 10.0 * np.sin(2*np.pi*1000*t)
    y = apply_agc(x, fs=FS)
    assert abs(10*np.log10(np.mean(y**2)+1e-12) - (-20.0)) < 5.0

def test_bandpass_filter_shape():
    sos = design_bandpass(fl=1e3, fh=10e3, fs=FS, order=6)
    assert sos.shape[1] == 6

def test_stft_keys():
    t = np.linspace(0, 0.01, int(0.01*FS))
    out = compute_stft(np.sin(2*np.pi*1000*t), fs=FS)
    assert "S_db" in out and "time_s" in out and "freq_hz" in out

def test_mel_spec_n_mels():
    t = np.linspace(0, 0.01, int(0.01*FS))
    out = compute_mel_spectrogram(np.sin(2*np.pi*1000*t), fs=FS)
    assert out["log_mel"].shape[0] == MEL_FILTERS

def test_cfar_detects_target():
    noise = np.random.exponential(1.0, 256)
    noise[128] = 100.0
    assert 128 in ca_cfar(noise)

def test_matched_filter_peak():
    template = np.ones(10)
    signal = np.zeros(100)
    signal[50:60] = template
    out = matched_filter(signal, template)
    peak = np.argmax(np.abs(out["output"]))
    assert abs(peak - 59) <= 2

def test_rf_features_length():
    iq = np.exp(1j * 2*np.pi*0.1*np.arange(1024))
    feats = extract_rf_features(iq, fs=FS)
    assert len(feats) == 7
