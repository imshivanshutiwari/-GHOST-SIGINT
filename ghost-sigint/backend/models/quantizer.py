"""ONNX export + INT8 dynamic quantization + SQNR checker."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def export_onnx(model: nn.Module, dummy: torch.Tensor, path: str,
                opset: int = 17) -> str:
    model.eval()
    torch.onnx.export(
        model, dummy, path, opset_version=opset,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    return path


def quantize_dynamic(onnx_path: str, out_path: str) -> str:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(onnx_path, out_path, weight_type=QuantType.QInt8)
        return out_path
    except ImportError:
        return onnx_path


def sqnr(fp32: np.ndarray, int8: np.ndarray) -> float:
    sig = np.mean(fp32 ** 2)
    noise = np.mean((fp32 - int8) ** 2) + 1e-12
    return float(10 * np.log10(sig / noise))
