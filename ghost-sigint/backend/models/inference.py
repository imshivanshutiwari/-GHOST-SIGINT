"""Auto-select fastest backend: CUDA FP16 → ONNX CPU → PyTorch CPU."""
from __future__ import annotations
import logging
import os
import time
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self, model: nn.Module, onnx_path: Optional[str] = None):
        self.model = model
        self.onnx_path = onnx_path
        self.backend = "cpu"
        self._ort = None
        self._setup()

    def _setup(self):
        if torch.cuda.is_available():
            self.model = self.model.half().cuda()
            torch.backends.cudnn.benchmark = True
            self.backend = "cuda_fp16"
            logger.info("InferenceEngine: CUDA FP16 (target <15 ms)")
            return
        if self.onnx_path and os.path.exists(self.onnx_path):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._ort = ort.InferenceSession(
                    self.onnx_path, sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self.backend = "onnx_cpu"
                logger.info("InferenceEngine: ONNX CPU (target <50 ms)")
                return
            except Exception as e:
                logger.warning(f"ONNX unavailable: {e}")
        self.model = self.model.float().cpu()
        logger.info("InferenceEngine: PyTorch CPU")

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> Tuple[int, float, float]:
        t0 = time.perf_counter()
        if self.backend == "onnx_cpu" and self._ort:
            out = self._ort.run(None, {"input": x.astype(np.float32)})[0]
            probs = F.softmax(torch.from_numpy(out), dim=-1).numpy()
        else:
            t = torch.from_numpy(x)
            if self.backend == "cuda_fp16":
                t = t.half().cuda()
            logits = self.model(t)
            probs = F.softmax(logits.float().cpu(), dim=-1).numpy()
        ms = (time.perf_counter() - t0) * 1000
        idx = int(np.argmax(probs[0]))
        return idx, float(probs[0][idx]), ms
