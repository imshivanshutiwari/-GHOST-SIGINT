"""Detection history, recording and CSV export."""
from __future__ import annotations
import csv, io
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from auth.rbac import require_role, Role

router = APIRouter(prefix="/detections", tags=["detections"])
_store: List[dict] = []


class Detection(BaseModel):
    timestamp: float
    signal_class: str
    confidence: float
    snr_db: float
    threat_score: float
    frequency_hz: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@router.get("/history")
async def history(limit: int = Query(100, le=1000), since: Optional[float] = None,
                  _=Depends(require_role(Role.OPERATOR))):
    r = _store if not since else [d for d in _store if d["timestamp"] >= since]
    return {"detections": r[-limit:], "total": len(r)}


@router.post("/record")
async def record(det: Detection, _=Depends(require_role(Role.OPERATOR))):
    _store.append(det.model_dump())
    return {"status": "recorded"}


@router.get("/export")
async def export_csv(_=Depends(require_role(Role.OPERATOR))):
    buf = io.StringIO()
    if _store:
        w = csv.DictWriter(buf, fieldnames=list(_store[0].keys()))
        w.writeheader(); w.writerows(_store)
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=detections.csv"})
