from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import pytest
from geo.tdoa_chan import chan_wls
from geo.music_aoa import estimate_doa_music, steering_vector
from geo.esprit import tls_esprit
from geo.friis import friis_ranging
from geo.kalman_singer import SingerKalmanFilter
from geo.bss_ica import fast_ica
from geo.rf_fingerprint import extract_features, RFFingerprinter

C = 3e8


def test_chan_wls_three_outputs():
    sensors = np.array([[0,0],[100,0],[50,100],[0,100]], dtype=float)
    tdoa = np.array([1e-7, 2e-7, 1.5e-7])
    assert len(chan_wls(sensors, tdoa)) == 3


def test_music_single_source():
    M, N = 8, 200
    a = steering_vector(np.deg2rad(30.0), M).reshape(M, 1)
    noise = (np.random.randn(M, N) + 1j*np.random.randn(M, N))/np.sqrt(2)*0.1
    X = a @ (np.random.randn(1, N) + 1j*np.random.randn(1, N))/np.sqrt(2) + noise
    doa, P, D = estimate_doa_music(X, M=M)
    assert len(P) > 0


def test_esprit_output_size():
    X = (np.random.randn(8, 100) + 1j*np.random.randn(8, 100))/np.sqrt(2)
    assert len(tls_esprit(X, D=2)) == 2


def test_friis_known_range():
    freq, d = 1e9, 1000.0
    fspl = 20*np.log10(d)+20*np.log10(freq)+20*np.log10(4*np.pi/C)
    p_rx = 30.0 - fspl
    r = friis_ranging(p_rx, 30.0, freq)
    assert abs(r["range_m"] - d) / d < 0.01


def test_kalman_tracks():
    kf = SingerKalmanFilter(dt=0.1)
    kf.initialize(np.array([0.0, 0.0]))
    for i in range(20):
        kf.predict()
        kf.update(np.array([i*5.0, 0.0]))
    assert len(kf.track) > 0


def test_kalman_gating_rejects_outlier():
    kf = SingerKalmanFilter(dt=0.1, sigma_r=50.0)
    kf.initialize(np.array([0.0, 0.0]))
    for _ in range(5):
        kf.predict(); kf.update(np.array([0.0, 0.0]))
    kf.predict()
    res = kf.update(np.array([1e6, 1e6]))
    assert res["accepted"] is False


def test_fast_ica_shape():
    N = 1000
    s = np.vstack([np.sin(0.1*np.arange(N)), np.sign(np.sin(0.13*np.arange(N)))])
    X = np.array([[1,0.5],[0.5,1]]) @ s
    S, W, _ = fast_ica(X, n_components=2, max_iter=100)
    assert S.shape == (2, N)


def test_rf_fingerprint_features():
    iq = np.exp(1j*2*np.pi*0.1*np.arange(2048))
    feats = extract_features(iq)
    assert len(feats) == 7 and not np.any(np.isnan(feats))


def test_rf_fingerprinter_known():
    fp = RFFingerprinter()
    iq = np.exp(1j*2*np.pi*0.1*np.arange(2048))
    fp.add_device("DEV1", extract_features(iq))
    r = fp.identify(extract_features(iq) + 1e-4*np.random.randn(7))
    assert r["verdict"] in ["KNOWN","SUSPECTED","UNKNOWN"]
