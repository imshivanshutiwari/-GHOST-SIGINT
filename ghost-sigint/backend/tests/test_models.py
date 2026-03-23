from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn.functional as F
import pytest
from constants import N_CLASSES


def test_resnet34_shape(mel_batch):
    from models.resnet34 import ResNet34SE
    m = ResNet34SE(N_CLASSES); m.eval()
    with torch.no_grad():
        assert m(mel_batch).shape == (4, N_CLASSES)


def test_cnn1d_shape(iq_batch):
    from models.cnn1d import CNN1DSE
    m = CNN1DSE(N_CLASSES); m.eval()
    with torch.no_grad():
        assert m(iq_batch).shape == (4, N_CLASSES)


def test_transformer_shape(stft_batch):
    from models.transformer import TemporalTransformer
    m = TemporalTransformer(N_CLASSES); m.eval()
    with torch.no_grad():
        assert m(stft_batch).shape == (4, N_CLASSES)


def test_bilstm_shape(mel_frames):
    from models.bilstm import BiLSTMAttention
    m = BiLSTMAttention(N_CLASSES); m.eval()
    with torch.no_grad():
        assert m(mel_frames).shape == (4, N_CLASSES)


def test_cnn_lstm_shape(iq_batch):
    from models.cnn_lstm import CNNLSTMHybrid
    m = CNNLSTMHybrid(N_CLASSES); m.eval()
    with torch.no_grad():
        assert m(iq_batch).shape == (4, N_CLASSES)


def test_fgsm_bounded(iq_batch):
    from models.adversarial import fgsm_attack
    from models.cnn1d import CNN1DSE
    m = CNN1DSE(N_CLASSES)
    y = torch.zeros(4, dtype=torch.long)
    x_adv = fgsm_attack(m, iq_batch, y, epsilon=0.01)
    assert (x_adv - iq_batch).abs().max().item() <= 0.01 + 1e-5


def test_ensemble_weights(iq_batch):
    from models.ensemble import EnsembleClassifier
    from models.cnn1d import CNN1DSE
    ens = EnsembleClassifier([CNN1DSE(N_CLASSES) for _ in range(2)])
    w = F.softmax(ens.w_logits, dim=0)
    assert abs(float(w.sum()) - 1.0) < 1e-5
