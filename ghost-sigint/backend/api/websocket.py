"""WebSocket hub — 6 real-time channels (spectrum, stft, detections, geolocation, metrics, alerts)."""
from __future__ import annotations
import asyncio, time, logging
from typing import Dict, Set
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from auth.jwt_handler import verify_token
from constants import (FS, WS_SPECTRUM_RATE, WS_STFT_RATE,
                        WS_GEOLOCATION_RATE, WS_METRICS_RATE, SIGNAL_CLASSES)
from signal.generator import SignalGenerator
from dsp.stft import compute_stft

logger = logging.getLogger(__name__)
router = APIRouter()
_gen = SignalGenerator(fs=FS)


class ConnectionManager:
    def __init__(self):
        CHANNELS = ("spectrum", "stft", "detections", "geolocation", "metrics", "alerts")
        self.active: Dict[str, Set[WebSocket]] = {c: set() for c in CHANNELS}

    async def connect(self, ch: str, ws: WebSocket):
        await ws.accept()
        self.active[ch].add(ws)

    def disconnect(self, ch: str, ws: WebSocket):
        self.active[ch].discard(ws)

    async def broadcast(self, ch: str, data: dict):
        dead = set()
        for ws in list(self.active[ch]):
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active[ch].discard(ws)


mgr = ConnectionManager()
_seq = 0


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ")

def _nxt() -> int:
    global _seq; _seq += 1; return _seq


async def _auth(token: str) -> bool:
    try:
        return bool(verify_token(token))
    except Exception:
        return False


@router.websocket("/ws/spectrum")
async def ws_spectrum(ws: WebSocket, token: str = Query("")):
    if not await _auth(token):
        await ws.close(code=4401); return
    await mgr.connect("spectrum", ws)
    try:
        while True:
            sc = np.random.choice(SIGNAL_CLASSES)
            sig = _gen.generate(sc, duration=0.001, fc=100e3, fs=FS)
            freq = np.fft.rfftfreq(len(sig), d=1/FS)
            psd = 20*np.log10(np.abs(np.fft.rfft(sig))**2/len(sig)+1e-12)
            await ws.send_json({"channel":"spectrum","timestamp":_ts(),"seq":_nxt(),
                                "signal_class":sc,"data":{"freq_hz":freq.tolist(),"power_dbm":psd.tolist()}})
            await asyncio.sleep(1.0/WS_SPECTRUM_RATE)
    except WebSocketDisconnect:
        mgr.disconnect("spectrum", ws)


@router.websocket("/ws/stft")
async def ws_stft(ws: WebSocket, token: str = Query("")):
    if not await _auth(token):
        await ws.close(code=4401); return
    await mgr.connect("stft", ws)
    try:
        while True:
            sc = np.random.choice(SIGNAL_CLASSES)
            sig = _gen.generate(sc, duration=0.005, fc=100e3, fs=FS)
            out = compute_stft(np.real(sig), fs=FS)
            await ws.send_json({"channel":"stft","timestamp":_ts(),"seq":_nxt(),
                                "signal_class":sc,
                                "data":{"S_db":out["S_db"].tolist(),
                                        "time_s":out["time_s"].tolist(),
                                        "freq_hz":out["freq_hz"].tolist()}})
            await asyncio.sleep(1.0/WS_STFT_RATE)
    except WebSocketDisconnect:
        mgr.disconnect("stft", ws)


@router.websocket("/ws/detections")
async def ws_detections(ws: WebSocket, token: str = Query("")):
    if not await _auth(token):
        await ws.close(code=4401); return
    await mgr.connect("detections", ws)
    try:
        while True:
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        mgr.disconnect("detections", ws)


@router.websocket("/ws/geolocation")
async def ws_geolocation(ws: WebSocket, token: str = Query("")):
    if not await _auth(token):
        await ws.close(code=4401); return
    await mgr.connect("geolocation", ws)
    lat0, lon0, t = 28.6139, 77.2090, 0.0
    try:
        while True:
            t += 1.0/WS_GEOLOCATION_RATE
            await ws.send_json({"channel":"geolocation","timestamp":_ts(),"seq":_nxt(),
                                "data":{"latitude": lat0+0.001*np.sin(0.1*t),
                                        "longitude": lon0+0.001*np.cos(0.1*t),
                                        "uncertainty_m": 50.0, "gdop": 1.8}})
            await asyncio.sleep(1.0/WS_GEOLOCATION_RATE)
    except WebSocketDisconnect:
        mgr.disconnect("geolocation", ws)


@router.websocket("/ws/metrics")
async def ws_metrics(ws: WebSocket, token: str = Query("")):
    if not await _auth(token):
        await ws.close(code=4401); return
    await mgr.connect("metrics", ws)
    try:
        import psutil
        while True:
            mem = psutil.virtual_memory()
            await ws.send_json({"channel":"metrics","timestamp":_ts(),"seq":_nxt(),
                                "data":{"cpu_pct":psutil.cpu_percent(),"ram_pct":mem.percent}})
            await asyncio.sleep(1.0/WS_METRICS_RATE)
    except WebSocketDisconnect:
        mgr.disconnect("metrics", ws)


@router.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket, token: str = Query("")):
    if not await _auth(token):
        await ws.close(code=4401); return
    await mgr.connect("alerts", ws)
    try:
        while True:
            await asyncio.sleep(5.0)
    except WebSocketDisconnect:
        mgr.disconnect("alerts", ws)
