from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pytest
from constants import FS, SIGNAL_CLASSES
from signal.generator import SignalGenerator
from signal.impairments import apply_awgn

GEN = SignalGenerator(fs=FS)

@pytest.mark.parametrize("sc", SIGNAL_CLASSES)
def test_signal_nonzero(sc):
    sig = GEN.generate(sc, duration=0.001, fc=50e3, fs=FS)
    assert len(sig) > 0
    assert not np.allclose(sig, 0)

@pytest.mark.parametrize("sc", SIGNAL_CLASSES)
def test_signal_complex_or_real(sc):
    sig = GEN.generate(sc, duration=0.001, fc=50e3, fs=FS)
    assert sig.dtype in [np.complex64, np.complex128, np.float32, np.float64]

def test_awgn_preserves_length():
    sig = GEN.generate("BPSK", 0.01, fc=50e3, fs=FS)
    noisy = apply_awgn(sig, snr_db=15.0)
    assert len(noisy) == len(sig)

def test_16qam_power_reasonable():
    sig = GEN.generate("16QAM", 0.005, fc=50e3, fs=FS)
    power = np.mean(np.abs(sig)**2)
    assert 0.01 < power < 10.0

def test_manpads_am_envelope():
    sig = GEN.generate("MANPADS", 0.005, fc=50e3, fs=FS)
    assert np.std(np.abs(sig)) > 0
