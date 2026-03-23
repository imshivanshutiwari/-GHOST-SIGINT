"""Signal generation REST routes."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.rbac import require_role, Role
from signal.generator import SignalGenerator
from signal.impairments import apply_awgn
from constants import SIGNAL_CLASSES, FS

router = APIRouter(prefix="/signals", tags=["signals"])
_gen = SignalGenerator(fs=FS)


class SignalRequest(BaseModel):
    signal_type: str
    duration: float = Field(0.001, ge=1e-4, le=0.1)
    snr_db: float = Field(20.0, ge=-30.0, le=50.0)
    fc: float = Field(100e3, ge=1e3, le=500e3)
    fs: float = Field(FS)


class SignalResponse(BaseModel):
    signal_type: str
    n_samples: int
    fs: float
    snr_db: float
    i_data: list
    q_data: list
    power_dbfs: float


@router.post("/generate", response_model=SignalResponse)
async def generate_signal(req: SignalRequest, _=Depends(require_role(Role.ANALYST))):
    if req.signal_type not in SIGNAL_CLASSES:
        raise HTTPException(400, f"Unknown signal type. Valid: {SIGNAL_CLASSES}")
    sig = _gen.generate(req.signal_type, duration=req.duration, fc=req.fc, fs=req.fs)
    sig = apply_awgn(sig, snr_db=req.snr_db)
    power_dbfs = float(10 * np.log10(np.mean(np.abs(sig)**2) + 1e-12))
    return SignalResponse(
        signal_type=req.signal_type, n_samples=len(sig), fs=req.fs, snr_db=req.snr_db,
        i_data=np.real(sig).tolist(), q_data=np.imag(sig).tolist(), power_dbfs=power_dbfs,
    )


@router.get("/params")
async def signal_params(_=Depends(require_role(Role.OPERATOR))):
    return {"signal_classes": SIGNAL_CLASSES, "snr_range": [-30, 50],
            "duration_range": [1e-4, 0.1], "fc_range": [1000, 500000], "fs": FS}
