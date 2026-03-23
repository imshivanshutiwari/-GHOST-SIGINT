from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pytest
import torch
from constants import FS, SIGNAL_CLASSES, N_CLASSES
from signal.generator import SignalGenerator
from signal.impairments import apply_awgn
from dsp.agc import apply_agc
from dsp.stft import compute_stft
from geo.rf_fingerprint import RFFingerprinter, extract_features

GEN = SignalGenerator(fs=FS)

def test_full_dsp_bpsk():
    sig = GEN.generate("BPSK", 0.01, fc=50e3, fs=FS)
    x = apply_agc(np.real(apply_awgn(sig, 15.0)), fs=FS)
    out = compute_stft(x, fs=FS)
    assert out["S_db"] is not None

def test_rf_fingerprint_pipeline():
    iq = GEN.generate("BPSK", 0.01, fc=50e3, fs=FS)
    fp = RFFingerprinter()
    fp.add_device("dev1", extract_features(iq))
    r = fp.identify(extract_features(iq))
    assert r["verdict"] in ["KNOWN","SUSPECTED","UNKNOWN"]

@pytest.mark.parametrize("sc", SIGNAL_CLASSES)
def test_all_signals_agc(sc):
    sig = GEN.generate(sc, 0.002, fc=50e3, fs=FS)
    out = apply_agc(np.real(apply_awgn(sig, 10.0)), fs=FS)
    assert len(out) > 0

def test_all_models_forward():
    from models.resnet34 import ResNet34SE
    from models.cnn1d import CNN1DSE
    from models.transformer import TemporalTransformer
    from models.bilstm import BiLSTMAttention
    from models.cnn_lstm import CNNLSTMHybrid
    for Model, x in [
        (ResNet34SE,        torch.randn(2, 3, 128, 128)),
        (CNN1DSE,           torch.randn(2, 2, 1024)),
        (TemporalTransformer, torch.randn(2, 32, 129)),
        (BiLSTMAttention,   torch.randn(2, 50, 128)),
        (CNNLSTMHybrid,     torch.randn(2, 2, 1024)),
    ]:
        m = Model(N_CLASSES); m.eval()
        with torch.no_grad():
            assert m(x).shape == (2, N_CLASSES)
