"""RF hardware fingerprinting: 7 features + cosine-similarity device library."""
from __future__ import annotations
import itertools
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
from scipy.stats import linregress

N_FEATURES = 7


def extract_features(
    iq: np.ndarray,
    fs: float = 1e6,
    fc_nominal: float = 100e3,
) -> np.ndarray:
    """Extract 7-dimensional RF fingerprint from complex IQ."""
    I, Q = np.real(iq), np.imag(iq)

    # f1: carrier frequency offset (use full FFT for complex IQ)
    spectrum = np.fft.fft(iq, n=len(iq))
    freqs = np.fft.fftfreq(len(iq), d=1.0 / fs)
    f1 = float(freqs[np.argmax(np.abs(spectrum))] - fc_nominal)

    # f2: IQ amplitude imbalance
    f2 = float(np.std(I) / (np.std(Q) + 1e-12) - 1.0)

    # f3: IQ phase imbalance
    cc = float(np.clip(np.corrcoef(I, Q)[0, 1], -1.0, 1.0))
    f3 = float(np.arccos(cc))

    # f4: turn-on transient slope
    env = np.abs(iq)
    n = min(50, len(env))
    slope, *_ = linregress(np.arange(n), env[:n])
    f4 = float(slope)

    # f5: 3rd-order AM-AM coefficient
    ei, eo = env[:-1], env[1:]
    f5 = float(np.polyfit(ei, eo, 3)[0]) if len(ei) >= 4 else 0.0

    # f6: spectral flatness
    psd = np.abs(spectrum) ** 2 + 1e-12
    f6 = float(np.exp(np.mean(np.log(psd))) / np.mean(psd))

    # f7: kurtosis of envelope
    f7 = float(stats.kurtosis(env, fisher=True))

    return np.array([f1, f2, f3, f4, f5, f6, f7], dtype=float)


class RFFingerprinter:
    KNOWN = 0.85
    SUSPECTED = 0.60

    def __init__(self):
        self.library: Dict[str, np.ndarray] = {}
        self._unknown_counter = itertools.count(1)

    def _norm(self, v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)

    def add_device(self, device_id: str, features: np.ndarray):
        self.library[device_id] = self._norm(features)

    def match(self, features: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        v = self._norm(features)
        scores = [(did, float(v @ lv)) for did, lv in self.library.items()]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    def identify(self, features: np.ndarray) -> Dict:
        matches = self.match(features)
        if not matches:
            uid = f"UNKNOWN_{next(self._unknown_counter):03d}"
            self.add_device(uid, features)
            return {"verdict": "UNKNOWN", "device_id": None, "similarity": 0.0, "top_matches": []}
        best_id, best_score = matches[0]
        verdict = "KNOWN" if best_score >= self.KNOWN else (
            "SUSPECTED" if best_score >= self.SUSPECTED else "UNKNOWN"
        )
        if verdict == "UNKNOWN":
            uid = f"UNKNOWN_{next(self._unknown_counter):03d}"
            self.add_device(uid, features)
        return {"verdict": verdict, "device_id": best_id, "similarity": best_score,
                "top_matches": matches}

    def fingerprint_from_iq(self, iq: np.ndarray, fs: float = 1e6,
                             fc_nominal: float = 100e3) -> Dict:
        feats = extract_features(iq, fs, fc_nominal)
        result = self.identify(feats)
        result["features"] = feats.tolist()
        return result
