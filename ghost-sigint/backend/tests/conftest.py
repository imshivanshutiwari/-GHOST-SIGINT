from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
import torch

from constants import FS, SIGNAL_CLASSES
from signal.generator import SignalGenerator
from signal.impairments import apply_awgn

GEN = SignalGenerator(fs=FS)


@pytest.fixture(params=SIGNAL_CLASSES)
def signal_class(request):
    return request.param

@pytest.fixture
def bpsk_iq():
    return apply_awgn(GEN.generate("BPSK", 0.002, fc=50e3, fs=FS), snr_db=20.0)

@pytest.fixture
def iq_batch():
    return torch.randn(4, 2, 1024)

@pytest.fixture
def mel_batch():
    return torch.randn(4, 3, 128, 128)

@pytest.fixture
def stft_batch():
    return torch.randn(4, 32, 129)

@pytest.fixture
def mel_frames():
    return torch.randn(4, 50, 128)
