"""Singer maneuver-model Kalman filter (6-state: [px,py,vx,vy,ax,ay])."""
from __future__ import annotations
import numpy as np
from typing import Tuple

CHI2_95_2DOF = 5.991


class SingerKalmanFilter:
    """6-state Singer KF with χ²(0.95, 2) gating and covariance ellipse output."""

    def __init__(self, dt: float = 0.1, sigma_m: float = 10.0,
                 tau_m: float = 20.0, sigma_r: float = 50.0):
        T = dt
        self.F = np.array([
            [1, 0, T, 0, T**2/2, 0],
            [0, 1, 0, T, 0, T**2/2],
            [0, 0, 1, 0, T,      0],
            [0, 0, 0, 1, 0,      T],
            [0, 0, 0, 0, 1,      0],
            [0, 0, 0, 0, 0,      1],
        ], dtype=float)
        q = sigma_m ** 2 / 3.0
        t2, t3, t4, t5 = T**2, T**3, T**4, T**5
        Q3 = q * np.array([
            [t5/20, t4/8,  t3/6],
            [t4/8,  t3/3,  t2/2],
            [t3/6,  t2/2,  T   ],
        ])
        self.Q = np.zeros((6, 6))
        ix = [0, 2, 4]
        iy = [1, 3, 5]
        self.Q[np.ix_(ix, ix)] = Q3
        self.Q[np.ix_(iy, iy)] = Q3
        self.H = np.zeros((2, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.R = np.diag([sigma_r**2, sigma_r**2])
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1e6
        self.track: list = []
        self.rejected: list = []

    def initialize(self, z: np.ndarray):
        self.x[:2] = z[:2]
        self.P = np.eye(6) * 1e4

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z: np.ndarray) -> dict:
        innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            S_inv = np.linalg.inv(S)
            nis = float(innov @ S_inv @ innov)
        except Exception:
            nis = 0.0
        if nis > CHI2_95_2DOF:
            self.rejected.append({"z": z.copy(), "nis": nis})
            return {"accepted": False, "nis": nis, "x": self.x.copy()}
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ innov
        self.P = (np.eye(6) - K @ self.H) @ self.P
        self.track.append({"x": self.x.copy(), "z": z.copy(), "nis": nis})
        return {"accepted": True, "nis": nis, "x": self.x.copy()}

    def uncertainty_ellipse(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (eigenvalues, eigenvectors) of the 2D position covariance for ellipse plotting."""
        return np.linalg.eigh(self.P[:2, :2])
